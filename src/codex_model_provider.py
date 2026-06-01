"""Experimental Codex CLI model-provider capability boundary.

This module does not implement chat dispatch. It reports whether a future
Codex-backed provider can be exposed safely, without treating completed CLI
output as token streaming or reading Codex credential files.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.codex_auth import get_codex_auth_service


CODEX_MODEL_PROVIDER_FLAG = "ODYSSEUS_CODEX_MODEL_PROVIDER_ENABLED"
CODEX_EXPERIMENTAL_MODEL_ID = "codex-cli/chatgpt-experimental"
CODEX_EXPERIMENTAL_MODEL_DISPLAY = "Codex CLI / ChatGPT (experimental, non-streaming)"
CODEX_VIRTUAL_ENDPOINT_URL = "odysseus://codex-cli"
CODEX_CHAT_TIMEOUT_SECONDS = 120

_TOKEN_PATTERNS = (
    re.compile(r"(?i)(access_token|refresh_token|id_token)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
)

_LIMITATIONS = [
    "Non-streaming: returns one completed assistant message.",
    "Session resume depends on the installed Codex CLI.",
    "Codex tool execution is not mapped into Odysseus agent rounds.",
    "The adapter requires Codex CLI read-only sandbox support before running.",
]

_UNSAFE_FLAGS = (
    "--dangerously-bypass-approvals-and-sandbox",
    "--yolo",
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_model_provider_enabled() -> bool:
    return _truthy(os.getenv(CODEX_MODEL_PROVIDER_FLAG, "false"))


def _sanitize_text(text: str | None, limit: int = 2000) -> str:
    safe = text or ""
    for pattern in _TOKEN_PATTERNS:
        safe = pattern.sub("<redacted-token>", safe)
    return safe.strip()[:limit]


def is_codex_virtual_endpoint(endpoint_url: str | None, model: str | None = None) -> bool:
    return (endpoint_url or "").strip() == CODEX_VIRTUAL_ENDPOINT_URL or (
        (model or "").strip() == CODEX_EXPERIMENTAL_MODEL_ID
    )


@dataclass(frozen=True)
class CodexCliCapabilities:
    sandbox_flag: str = ""
    sandbox_read_only_supported: bool = False
    sandbox_modes_visible: bool = False
    approval_flag: str = ""
    approval_never_value: str = "never"
    supports_json: bool = False
    supports_model: bool = False
    session_resume_supported: bool = False
    resume_last_supported: bool = False
    resume_session_id_supported: bool = False

    @property
    def sandbox_supported(self) -> bool:
        return bool(self.sandbox_flag)

    @property
    def approval_control_supported(self) -> bool:
        return bool(self.approval_flag)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sandbox_supported": self.sandbox_supported,
            "sandbox_flag": self.sandbox_flag,
            "sandbox_read_only_supported": self.sandbox_read_only_supported,
            "sandbox_modes_visible": self.sandbox_modes_visible,
            "approval_control_supported": self.approval_control_supported,
            "approval_flag": self.approval_flag,
            "supports_json": self.supports_json,
            "supports_model": self.supports_model,
            "session_resume_supported": self.session_resume_supported,
            "resume_last_supported": self.resume_last_supported,
            "resume_session_id_supported": self.resume_session_id_supported,
        }


def _limitations_for_capabilities(capabilities: CodexCliCapabilities | dict[str, Any] | None = None) -> list[str]:
    limitations = list(_LIMITATIONS)
    approval_supported = False
    if isinstance(capabilities, CodexCliCapabilities):
        approval_supported = capabilities.approval_control_supported
    elif isinstance(capabilities, dict):
        approval_supported = bool(capabilities.get("approval_control_supported"))
    if capabilities is not None and not approval_supported:
        limitations.append(
            "This Codex CLI does not expose an approval-control flag; Odysseus relies on read-only sandbox mode."
        )
    return limitations


class CodexCliChatAdapter:
    """Admin-only, non-streaming adapter boundary for `codex exec`.

    This intentionally does not provide a normal chat-provider hook yet. It
    refuses to run unless the installed CLI advertises read-only sandbox support
    for a constrained one-shot completion probe.
    """

    def __init__(
        self,
        auth_service_getter: Callable[[], Any] | None = None,
        runner: Callable[..., Any] | None = None,
        session_root: str | Path | None = None,
    ) -> None:
        self._auth_service_getter = auth_service_getter or get_codex_auth_service
        self._runner = runner
        self._session_root = Path(session_root) if session_root else Path(tempfile.gettempdir()) / "odysseus-codex-sessions"
        self._session_map: dict[str, dict[str, Any]] = {}
        self._reset_versions: dict[str, int] = {}

    async def available(self) -> dict[str, Any]:
        preflight = await self._preflight()
        if not preflight.get("ok"):
            return preflight
        capabilities_result = await self._detect_cli_capabilities(preflight["bin_path"], preflight["env"])
        if not capabilities_result.get("ok"):
            return capabilities_result
        capabilities = capabilities_result["capabilities"]
        return {
            "ok": True,
            "status": "available",
            "chat_supported": True,
            "streaming_supported": False,
            "session_resume_supported": capabilities.session_resume_supported,
            "tool_execution_allowed": False,
            "supports_json": capabilities.supports_json,
            "supports_model": capabilities.supports_model,
            "approval_control_supported": capabilities.approval_control_supported,
            "capabilities": capabilities.as_dict(),
            "limitations": _limitations_for_capabilities(capabilities),
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
        odysseus_session_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        availability = await self.available()
        if not availability.get("ok"):
            return {
                **availability,
                "ok": False,
                "duration_ms": round((time.time() - started) * 1000),
                "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            }

        preflight = await self._preflight()
        prompt = self._build_prompt(messages)
        timeout = max(1, min(int(timeout_seconds or CODEX_CHAT_TIMEOUT_SECONDS), 300))
        resume_supported = bool(availability.get("session_resume_supported", False))
        session_key = self._session_key(odysseus_session_id)
        args, resume_mode = self._build_exec_args(
            preflight["bin_path"],
            prompt,
            odysseus_session_key=session_key,
            capabilities=availability.get("capabilities") or {},
        )
        workdir = self._ensure_workdir_for_session(session_key)
        rc, out, err = await self._run(
            args,
            timeout=timeout,
            cwd=str(workdir),
            env=preflight["env"],
        )

        duration_ms = round((time.time() - started) * 1000)
        if rc == 124:
            return self._error("timeout", "Codex CLI timed out", duration_ms, model)
        if rc != 0:
            detail = _sanitize_text(err or out, limit=500)
            return self._error("cli_failed", detail or "Codex CLI failed", duration_ms, model)

        codex_session_id = self._extract_session_id("\n".join([out or "", err or ""]))
        if session_key:
            self._session_map[session_key] = {
                "codex_session_id": codex_session_id or (self._session_map.get(session_key) or {}).get("codex_session_id", ""),
                "workdir": str(workdir),
                "updated_at": time.time(),
            }

        message = self._extract_message(out)
        if not message:
            return self._error("empty_response", "Codex CLI returned no assistant message", duration_ms, model)

        return {
            "ok": True,
            "status": "ok",
            "message": message,
            "duration_ms": duration_ms,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "limitations": _limitations_for_capabilities(availability.get("capabilities")),
            "streaming_supported": False,
            "session_resume_supported": resume_supported,
            "session_resumed": resume_mode != "new",
            "codex_resume_mode": resume_mode,
            "tool_execution_allowed": False,
        }

    def reset_session(self, odysseus_session_id: str | None) -> dict[str, Any]:
        session_key = self._session_key(odysseus_session_id)
        if not session_key:
            return {"ok": False, "status": "invalid_request", "error": "session_id is required"}
        existed = session_key in self._session_map
        self._session_map.pop(session_key, None)
        self._reset_versions[session_key] = self._reset_versions.get(session_key, 0) + 1
        return {"ok": True, "status": "reset", "session_mapping_cleared": existed}

    async def _preflight(self) -> dict[str, Any]:
        if not codex_model_provider_enabled():
            return {"ok": False, "status": "disabled", "error": "Codex model provider is disabled"}

        service = self._auth_service_getter()
        try:
            auth = await service.status()
        except Exception as exc:
            return {"ok": False, "status": "auth_status_failed", "error": exc.__class__.__name__}

        cli_available = bool(
            auth.get("codex_cli_available")
            or (auth.get("cli_found") and auth.get("cli_executable"))
        )
        if not cli_available:
            return {"ok": False, "status": "cli_unavailable", "error": "Codex CLI is unavailable"}

        authenticated = bool(auth.get("codex_authenticated") or auth.get("authenticated"))
        if not authenticated:
            return {"ok": False, "status": "sign_in_required", "error": "Sign in with Codex / ChatGPT first"}

        bin_path = ""
        try:
            bin_path = service._bin_path()  # Existing auth service owns CLI resolution.
        except Exception:
            bin_path = auth.get("resolved_binary_path") or ""
        if not bin_path:
            bin_path = auth.get("resolved_binary_path") or os.getenv("CODEX_BIN", "codex")

        try:
            env = service._env()
        except Exception:
            env = os.environ.copy()

        return {"ok": True, "status": "preflight_ok", "bin_path": bin_path, "env": env}

    async def _detect_cli_capabilities(self, bin_path: str, env: dict[str, str]) -> dict[str, Any]:
        rc, out, err = await self._run([bin_path, "exec", "--help"], timeout=20, env=env)
        top_rc, top_out, top_err = await self._run([bin_path, "--help"], timeout=20, env=env)
        if rc != 0 and top_rc != 0:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Unable to inspect Codex CLI safety capabilities",
                "detail": _sanitize_text(err or out or top_err or top_out, limit=500),
            }
        help_text = "\n".join([out or "", err or "", top_out or "", top_err or ""])
        resume = await self._detect_resume_support(bin_path, env, help_text)
        capabilities = self._parse_cli_capabilities(help_text, resume)
        if not capabilities.sandbox_supported or not capabilities.sandbox_read_only_supported:
            missing = []
            if not capabilities.sandbox_supported:
                missing.append("--sandbox/-s")
            if capabilities.sandbox_supported and not capabilities.sandbox_read_only_supported:
                missing.append("read-only sandbox mode")
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Codex CLI does not advertise required read-only sandbox support",
                "missing_flags": missing,
                "capabilities": capabilities.as_dict(),
            }
        return {
            "ok": True,
            "status": "capabilities_ok",
            "capabilities": capabilities,
        }

    async def _detect_resume_support(self, bin_path: str, env: dict[str, str], help_text: str) -> dict[str, bool]:
        rc, out, err = await self._run([bin_path, "exec", "resume", "--help"], timeout=20, env=env)
        text = "\n".join([help_text or "", out or "", err or ""]).lower()
        resume_supported = bool((rc == 0 and "resume" in text) or (re.search(r"\bresume\b", text) and "exec" in text))
        resume_last_supported = bool(re.search(r"(?<!\w)--last\b", text))
        resume_session_id_supported = bool(
            re.search(r"<\s*(session|conversation)[_-]?id\s*>", text, re.IGNORECASE)
            or re.search(r"\[\s*(session|conversation)[_-]?id\s*\]", text, re.IGNORECASE)
            or re.search(r"\b(session|conversation)[_-]?id\b", text, re.IGNORECASE)
        )
        return {
            "session_resume_supported": resume_supported and (resume_last_supported or resume_session_id_supported),
            "resume_last_supported": resume_last_supported,
            "resume_session_id_supported": resume_session_id_supported,
        }

    @staticmethod
    def _parse_cli_capabilities(help_text: str, resume: dict[str, bool]) -> CodexCliCapabilities:
        text = help_text or ""
        lower = text.lower()
        sandbox_flag = ""
        if "--sandbox" in text:
            sandbox_flag = "--sandbox"
        elif re.search(r"(^|[\s,])-s([,\s]|$)", text):
            sandbox_flag = "-s"

        sandbox_modes_visible = bool(
            "read-only" in lower
            or "read only" in lower
            or "workspace-write" in lower
            or "danger-full-access" in lower
            or "sandbox_mode" in lower
        )
        sandbox_read_only_supported = bool(sandbox_flag and ("read-only" in lower or "read only" in lower))
        if sandbox_flag and not sandbox_modes_visible:
            # Older/current help can show only "--sandbox <SANDBOX_MODE>". The
            # CLI will reject an unknown mode, so keeping read-only here is a
            # fail-closed probe while still supporting terse help output.
            sandbox_read_only_supported = True

        approval_flag = ""
        approval_candidates = (
            "--ask-for-approval",
            "--approval-policy",
            "--approval",
            "--approval-mode",
            "--approval_policy",
        )
        for candidate in approval_candidates:
            if candidate in text:
                approval_flag = candidate
                break
        if not approval_flag:
            match = re.search(r"--[A-Za-z0-9][A-Za-z0-9_-]*approval[A-Za-z0-9_-]*", text)
            if match and match.group(0) not in _UNSAFE_FLAGS:
                approval_flag = match.group(0)

        return CodexCliCapabilities(
            sandbox_flag=sandbox_flag,
            sandbox_read_only_supported=sandbox_read_only_supported,
            sandbox_modes_visible=sandbox_modes_visible,
            approval_flag=approval_flag,
            supports_json=bool("--json" in text or "--output-format" in text),
            supports_model=bool("--model" in text or re.search(r"(^|[\s,])-m([,\s]|$)", text)),
            session_resume_supported=bool(resume.get("session_resume_supported")),
            resume_last_supported=bool(resume.get("resume_last_supported")),
            resume_session_id_supported=bool(resume.get("resume_session_id_supported")),
        )

    def _build_exec_args(
        self,
        bin_path: str,
        prompt: str,
        *,
        odysseus_session_key: str,
        capabilities: CodexCliCapabilities | dict[str, Any],
    ) -> tuple[list[str], str]:
        caps = self._coerce_capabilities(capabilities)
        safety = self._safety_args(caps)
        args = [bin_path, "exec"]
        resume_mode = "new"
        if caps.session_resume_supported and odysseus_session_key:
            mapped = self._session_map.get(odysseus_session_key) or {}
            codex_session_id = str(mapped.get("codex_session_id") or "").strip()
            if codex_session_id and caps.resume_session_id_supported:
                args.extend(["resume", codex_session_id])
                resume_mode = "session_id"
            else:
                mapped_workdir_raw = str(mapped.get("workdir") or "")
                can_resume_last = bool(mapped_workdir_raw and Path(mapped_workdir_raw).exists())
                if can_resume_last and caps.resume_last_supported:
                    args.extend(["resume", "--last"])
                    resume_mode = "last_in_workdir"
        args.extend([*safety, prompt])
        self._assert_safe_args(args)
        return args, resume_mode

    @staticmethod
    def _coerce_capabilities(capabilities: CodexCliCapabilities | dict[str, Any]) -> CodexCliCapabilities:
        if isinstance(capabilities, CodexCliCapabilities):
            return capabilities
        return CodexCliCapabilities(
            sandbox_flag=str(capabilities.get("sandbox_flag") or ""),
            sandbox_read_only_supported=bool(capabilities.get("sandbox_read_only_supported")),
            sandbox_modes_visible=bool(capabilities.get("sandbox_modes_visible")),
            approval_flag=str(capabilities.get("approval_flag") or ""),
            supports_json=bool(capabilities.get("supports_json")),
            supports_model=bool(capabilities.get("supports_model")),
            session_resume_supported=bool(capabilities.get("session_resume_supported")),
            resume_last_supported=bool(capabilities.get("resume_last_supported")),
            resume_session_id_supported=bool(capabilities.get("resume_session_id_supported")),
        )

    @staticmethod
    def _safety_args(capabilities: CodexCliCapabilities) -> list[str]:
        if not capabilities.sandbox_supported or not capabilities.sandbox_read_only_supported:
            raise ValueError("Codex provider requires read-only sandbox support")
        args = [capabilities.sandbox_flag, "read-only"]
        if capabilities.approval_control_supported:
            args.extend([capabilities.approval_flag, capabilities.approval_never_value])
        return args

    def _assert_safe_args(self, args: list[str]) -> None:
        if "exec" not in args:
            raise ValueError("Codex provider must use codex exec")
        for unsafe in _UNSAFE_FLAGS:
            if unsafe in args:
                raise ValueError("Unsafe Codex CLI flag blocked")
        if ("--sandbox" not in args and "-s" not in args) or "read-only" not in args:
            raise ValueError("Codex provider requires read-only sandbox")
        if CODEX_EXPERIMENTAL_MODEL_ID in args:
            raise ValueError("Internal Codex model id must not be passed to Codex CLI")

    def _session_key(self, odysseus_session_id: str | None) -> str:
        value = (odysseus_session_id or "").strip()
        if not value:
            return ""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:120]

    def _workdir_path_for_session(self, session_key: str) -> Path:
        version = self._reset_versions.get(session_key, 0)
        safe = session_key or "one-shot"
        return self._session_root / f"{safe}-{version}"

    def _ensure_workdir_for_session(self, session_key: str) -> Path:
        workdir = self._workdir_path_for_session(session_key)
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    async def _run(
        self,
        args: list[str],
        *,
        timeout: int,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        if self._runner:
            return await self._runner(args=args, timeout=timeout, cwd=cwd, env=env)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 124, "", "Command timed out"
            return (
                proc.returncode or 0,
                (out or b"").decode("utf-8", errors="replace"),
                (err or b"").decode("utf-8", errors="replace"),
            )
        except Exception as exc:
            return 1, "", f"Failed to start Codex CLI: {exc.__class__.__name__}"

    @staticmethod
    def _build_prompt(messages: list[dict[str, Any]]) -> str:
        parts = [
            "You are replying through Odysseus' experimental Codex CLI provider.",
            "Return only the final assistant response.",
            "Do not run tools, shell commands, file edits, or web requests.",
            "If a request requires tools, say that this experimental provider does not support tools yet.",
            "",
            "Conversation:",
        ]
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user").strip() or "user"
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
            parts.append(f"{role}: {content}")
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_message(output: str) -> str:
        text = _sanitize_text(output)
        if not text:
            return ""
        # If a future CLI emits JSONL, prefer common completed-message fields.
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            for key in ("message", "content", "text", "output"):
                value = data.get(key) if isinstance(data, dict) else None
                if isinstance(value, str) and value.strip():
                    return _sanitize_text(value)
        return text

    @staticmethod
    def _extract_session_id(output: str) -> str:
        text = _sanitize_text(output, limit=8000)
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                except Exception:
                    data = None
                if isinstance(data, dict):
                    for key in ("session_id", "conversation_id", "id"):
                        value = data.get(key)
                        if isinstance(value, str) and re.match(r"^[A-Za-z0-9_.:-]{8,}$", value):
                            return value[:200]
            match = re.search(r"(?i)\b(?:session|conversation)[ _-]?id\b[:= ]+([A-Za-z0-9_.:-]{8,})", line)
            if match:
                return match.group(1)[:200]
        return ""

    @staticmethod
    def _error(status: str, error: str, duration_ms: int, model: str | None) -> dict[str, Any]:
        return {
            "ok": False,
            "status": status,
            "error": _sanitize_text(error, limit=500),
            "duration_ms": duration_ms,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "limitations": list(_LIMITATIONS),
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
        }


class CodexModelProvider:
    """Status/capability adapter for a future Codex CLI provider."""

    def __init__(
        self,
        auth_service_getter: Callable[[], Any] | None = None,
        chat_adapter: CodexCliChatAdapter | None = None,
    ) -> None:
        self._auth_service_getter = auth_service_getter or get_codex_auth_service
        self._chat_adapter = chat_adapter or CodexCliChatAdapter(self._auth_service_getter)

    async def status(self) -> dict[str, Any]:
        enabled = codex_model_provider_enabled()
        base = {
            "feature_enabled": enabled,
            "feature_flag": CODEX_MODEL_PROVIDER_FLAG,
            "provider": "codex_cli",
            "experimental": True,
            "models": [],
            "chat_supported": False,
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "limitations": list(_LIMITATIONS),
        }
        if not enabled:
            return {
                **base,
                "status": "disabled",
                "cli_available": False,
                "authenticated": False,
                "requires_sign_in": False,
            }

        try:
            auth = await self._auth_service_getter().status()
        except Exception as exc:
            return {
                **base,
                "status": "auth_status_failed",
                "cli_available": False,
                "authenticated": False,
                "requires_sign_in": False,
                "error": exc.__class__.__name__,
            }

        cli_available = bool(
            auth.get("codex_cli_available")
            or (auth.get("cli_found") and auth.get("cli_executable"))
        )
        authenticated = bool(auth.get("codex_authenticated") or auth.get("authenticated"))
        auth_status = str(auth.get("status") or "")
        auth_mode = str(auth.get("auth_mode") or "")

        if not cli_available:
            status = "cli_unavailable"
            requires_sign_in = False
        elif not authenticated:
            status = "sign_in_required"
            requires_sign_in = True
        else:
            status = "available"
            requires_sign_in = False

        models = []
        chat_available = {"ok": False}
        availability_details: dict[str, Any] = {}
        if status == "available":
            chat_available = await self._chat_adapter.available()
            if not chat_available.get("ok"):
                status = chat_available.get("status") or "unsupported_unsafe_cli_mode"
                for key in ("error", "missing_flags", "capabilities"):
                    if key in chat_available:
                        availability_details[key] = chat_available[key]
            else:
                base["chat_supported"] = True
                base["session_resume_supported"] = bool(chat_available.get("session_resume_supported", False))
                base["limitations"] = chat_available.get("limitations") or base["limitations"]
                availability_details["capabilities"] = chat_available.get("capabilities", {})
                availability_details["approval_control_supported"] = bool(
                    chat_available.get("approval_control_supported", False)
                )
                models.append({
                    "id": CODEX_EXPERIMENTAL_MODEL_ID,
                    "display": CODEX_EXPERIMENTAL_MODEL_DISPLAY,
                    "experimental": True,
                    "streaming_supported": False,
                    "session_resume_supported": bool(chat_available.get("session_resume_supported", False)),
                })

        return {
            **base,
            "status": status,
            "cli_available": cli_available,
            "authenticated": authenticated,
            "requires_sign_in": requires_sign_in,
            "sign_in_route": "/api/codex-auth/start",
            "auth": {
                "status": auth_status,
                "auth_mode": auth_mode,
                "codex_home": auth.get("codex_home", ""),
            },
            "models": models,
            **availability_details,
        }

    async def test_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
        odysseus_session_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._chat_adapter.complete(
            messages,
            model=model,
            timeout_seconds=timeout_seconds,
            odysseus_session_id=odysseus_session_id,
        )

    def reset_session(self, odysseus_session_id: str | None) -> dict[str, Any]:
        return self._chat_adapter.reset_session(odysseus_session_id)
