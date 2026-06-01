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
    "The adapter requires Codex CLI sandbox/approval flags before running.",
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


class CodexCliChatAdapter:
    """Admin-only, non-streaming adapter boundary for `codex exec`.

    This intentionally does not provide a normal chat-provider hook yet. It
    refuses to run unless the installed CLI advertises the sandbox and approval
    flags needed for a constrained one-shot completion probe.
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
        help_result = await self._exec_help(preflight["bin_path"], preflight["env"])
        if not help_result.get("ok"):
            return help_result
        return {
            "ok": True,
            "status": "available",
            "chat_supported": True,
            "streaming_supported": False,
            "session_resume_supported": bool(help_result.get("session_resume_supported", False)),
            "tool_execution_allowed": False,
            "supports_json": help_result.get("supports_json", False),
            "limitations": list(_LIMITATIONS),
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
            resume_supported=resume_supported,
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
            "limitations": list(_LIMITATIONS),
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

    async def _exec_help(self, bin_path: str, env: dict[str, str]) -> dict[str, Any]:
        rc, out, err = await self._run([bin_path, "exec", "--help"], timeout=20, env=env)
        top_rc, top_out, top_err = 1, "", ""
        if rc != 0 or "--sandbox" not in (out or "") or "--ask-for-approval" not in (out or ""):
            top_rc, top_out, top_err = await self._run([bin_path, "--help"], timeout=20, env=env)
        help_text = "\n".join([out or "", top_out or ""])
        if rc != 0 and top_rc != 0:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Unable to inspect codex exec safety flags",
                "detail": _sanitize_text(err or out or top_err or top_out, limit=500),
            }
        missing = [flag for flag in ("--sandbox", "--ask-for-approval") if flag not in help_text]
        if missing:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Codex CLI does not advertise required safety flags",
                "missing_flags": missing,
            }
        resume_supported = await self._detect_resume_support(bin_path, env, help_text)
        return {
            "ok": True,
            "status": "exec_help_ok",
            "supports_json": "--json" in help_text,
            "supports_model": "--model" in help_text or " -m" in help_text,
            "session_resume_supported": resume_supported,
        }

    async def _detect_resume_support(self, bin_path: str, env: dict[str, str], help_text: str) -> bool:
        rc, out, err = await self._run([bin_path, "exec", "resume", "--help"], timeout=20, env=env)
        text = "\n".join([help_text or "", out or "", err or ""]).lower()
        if rc == 0 and "resume" in text:
            return True
        return bool(re.search(r"\bresume\b", text) and "exec" in text)

    def _build_exec_args(
        self,
        bin_path: str,
        prompt: str,
        *,
        odysseus_session_key: str,
        resume_supported: bool,
    ) -> tuple[list[str], str]:
        safety = ["--sandbox", "read-only", "--ask-for-approval", "never"]
        args = [bin_path, "exec"]
        resume_mode = "new"
        if resume_supported and odysseus_session_key:
            mapped = self._session_map.get(odysseus_session_key) or {}
            codex_session_id = str(mapped.get("codex_session_id") or "").strip()
            if codex_session_id:
                args.extend(["resume", codex_session_id])
                resume_mode = "session_id"
            else:
                mapped_workdir_raw = str(mapped.get("workdir") or "")
                can_resume_last = bool(mapped_workdir_raw and Path(mapped_workdir_raw).exists())
                if can_resume_last:
                    args.extend(["resume", "--last"])
                    resume_mode = "last_in_workdir"
        args.extend([*safety, prompt])
        self._assert_safe_args(args)
        return args, resume_mode

    def _assert_safe_args(self, args: list[str]) -> None:
        if "exec" not in args:
            raise ValueError("Codex provider must use codex exec")
        for unsafe in _UNSAFE_FLAGS:
            if unsafe in args:
                raise ValueError("Unsafe Codex CLI flag blocked")
        if "--sandbox" not in args or "read-only" not in args:
            raise ValueError("Codex provider requires read-only sandbox")
        if "--ask-for-approval" not in args or "never" not in args:
            raise ValueError("Codex provider requires approvals disabled")
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
        if status == "available":
            chat_available = await self._chat_adapter.available()
            if not chat_available.get("ok"):
                status = chat_available.get("status") or "unsupported_unsafe_cli_mode"
            else:
                base["chat_supported"] = True
                base["session_resume_supported"] = bool(chat_available.get("session_resume_supported", False))
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
