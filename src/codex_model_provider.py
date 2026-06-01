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
from typing import Any, Callable

from src.codex_auth import get_codex_auth_service


CODEX_MODEL_PROVIDER_FLAG = "ODYSSEUS_CODEX_MODEL_PROVIDER_ENABLED"
CODEX_EXPERIMENTAL_MODEL_ID = "codex-cli/chatgpt-experimental"
CODEX_EXPERIMENTAL_MODEL_DISPLAY = "Codex CLI / ChatGPT (experimental, non-streaming)"
CODEX_CHAT_TIMEOUT_SECONDS = 120

_TOKEN_PATTERNS = (
    re.compile(r"(?i)(access_token|refresh_token|id_token)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
)

_BASE_LIMITATIONS = [
    "Non-streaming: returns one completed assistant message.",
    "Stateless: session/resume is not implemented.",
    "Codex tool execution is not mapped into Odysseus agent rounds.",
    "The adapter requires Codex CLI sandbox support and runs with read-only sandbox mode.",
]

_APPROVAL_FLAGS = ("--ask-for-approval", "--approval-policy", "--approval")
_DANGEROUS_FLAGS = ("--dangerously-bypass-approvals-and-sandbox", "--yolo")


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_model_provider_enabled() -> bool:
    return _truthy(os.getenv(CODEX_MODEL_PROVIDER_FLAG, "false"))


def _sanitize_text(text: str | None, limit: int = 2000) -> str:
    safe = text or ""
    for pattern in _TOKEN_PATTERNS:
        safe = pattern.sub("<redacted-token>", safe)
    return safe.strip()[:limit]


@dataclass(frozen=True)
class CodexCliCapabilities:
    """Detected Codex CLI command surface for the installed binary."""

    sandbox_flag: str | None
    sandbox_modes: tuple[str, ...] = ()
    approval_flag: str | None = None
    supports_json: bool = False
    supports_model: bool = False
    resume_supported: bool = False
    resume_last_supported: bool = False
    resume_session_id_supported: bool = False
    dangerous_flags: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.sandbox_flag)

    def limitations(self) -> list[str]:
        out = list(_BASE_LIMITATIONS)
        if self.approval_flag:
            out.append(f"Approval prompts are suppressed with Codex CLI {self.approval_flag}.")
        else:
            out.append(
                "Approval-control flag is not available on this Codex CLI; "
                "Odysseus relies on read-only sandbox mode."
            )
        return out

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "sandbox_supported": bool(self.sandbox_flag),
            "sandbox_flag": self.sandbox_flag,
            "sandbox_modes": list(self.sandbox_modes),
            "sandbox_mode": "read-only",
            "approval_control_supported": bool(self.approval_flag),
            "approval_flag": self.approval_flag,
            "resume_supported": self.resume_supported,
            "resume_last_supported": self.resume_last_supported,
            "resume_session_id_supported": self.resume_session_id_supported,
            "json_output_supported": self.supports_json,
            "model_flag_supported": self.supports_model,
            "dangerous_flags_advertised": list(self.dangerous_flags),
        }

    def build_exec_args(self, bin_path: str, prompt: str) -> list[str]:
        if not self.sandbox_flag:
            raise ValueError("Codex CLI sandbox support is required")
        args = [bin_path, "exec", self.sandbox_flag, "read-only"]
        if self.approval_flag:
            args.extend([self.approval_flag, "never"])
        args.append(prompt)
        if any(flag in args for flag in _DANGEROUS_FLAGS):
            raise ValueError("Unsafe Codex CLI flag refused")
        return args


def _detect_sandbox_flag(help_text: str) -> str | None:
    if "--sandbox" in help_text:
        return "--sandbox"
    if re.search(r"(^|\s)-s([,\s]|$)", help_text):
        return "-s"
    return None


def _detect_approval_flag(help_text: str) -> str | None:
    for flag in _APPROVAL_FLAGS:
        if flag in help_text:
            return flag
    return None


def _detect_cli_capabilities_from_help(
    exec_help: str,
    root_help: str = "",
    resume_help: str = "",
) -> CodexCliCapabilities:
    combined = "\n".join([exec_help or "", root_help or "", resume_help or ""])
    resume_text = "\n".join([exec_help or "", root_help or ""])
    return CodexCliCapabilities(
        sandbox_flag=_detect_sandbox_flag(exec_help or ""),
        sandbox_modes=tuple(
            mode for mode in ("read-only", "workspace-write", "danger-full-access")
            if mode in (exec_help or "")
        ),
        approval_flag=_detect_approval_flag(exec_help or ""),
        supports_json="--json" in exec_help,
        supports_model="--model" in exec_help or " -m" in exec_help,
        resume_supported=bool(re.search(r"\bresume\b", resume_text)),
        resume_last_supported="--last" in resume_help,
        resume_session_id_supported=bool(re.search(r"\b(session|SESSION)(_ID| id| id)?\b", resume_help, re.I)),
        dangerous_flags=tuple(flag for flag in _DANGEROUS_FLAGS if flag in combined),
    )


class CodexCliChatAdapter:
    """Admin-only, non-streaming adapter boundary for `codex exec`.

    This intentionally does not provide a normal chat-provider hook yet. It
    refuses to run unless the installed CLI advertises sandbox support needed
    for a constrained one-shot completion probe.
    """

    def __init__(
        self,
        auth_service_getter: Callable[[], Any] | None = None,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self._auth_service_getter = auth_service_getter or get_codex_auth_service
        self._runner = runner

    async def available(self) -> dict[str, Any]:
        return self._public_result(await self._available_internal())

    async def _available_internal(self) -> dict[str, Any]:
        preflight = await self._preflight()
        if not preflight.get("ok"):
            return preflight
        help_result = await self._detect_cli_capabilities(preflight["bin_path"], preflight["env"])
        if not help_result.get("ok"):
            return help_result
        capabilities = help_result["_capabilities"]
        return {
            "ok": True,
            "status": "available",
            "chat_supported": True,
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "supports_json": capabilities.supports_json,
            "supports_model": capabilities.supports_model,
            "cli_capabilities": capabilities.to_public_dict(),
            "limitations": capabilities.limitations(),
            "_preflight": preflight,
            "_capabilities": capabilities,
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
        odysseus_session_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        availability = await self._available_internal()
        if not availability.get("ok"):
            return {
                **self._public_result(availability),
                "ok": False,
                "duration_ms": round((time.time() - started) * 1000),
                "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            }

        preflight = availability["_preflight"]
        capabilities = availability["_capabilities"]
        prompt = self._build_prompt(messages)
        timeout = max(1, min(int(timeout_seconds or CODEX_CHAT_TIMEOUT_SECONDS), 300))

        with tempfile.TemporaryDirectory(prefix="odysseus-codex-chat-") as workdir:
            args = capabilities.build_exec_args(preflight["bin_path"], prompt)
            rc, out, err = await self._run(
                args,
                timeout=timeout,
                cwd=workdir,
                env=preflight["env"],
            )

        duration_ms = round((time.time() - started) * 1000)
        if rc == 124:
            return self._error("timeout", "Codex CLI timed out", duration_ms, model, capabilities)
        if rc != 0:
            detail = _sanitize_text(err or out, limit=500)
            return self._error("cli_failed", detail or "Codex CLI failed", duration_ms, model, capabilities)

        message = self._extract_message(out)
        if not message:
            return self._error("empty_response", "Codex CLI returned no assistant message", duration_ms, model, capabilities)

        return {
            "ok": True,
            "status": "ok",
            "message": message,
            "duration_ms": duration_ms,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "limitations": capabilities.limitations(),
            "cli_capabilities": capabilities.to_public_dict(),
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
        }

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
        if rc != 0:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Unable to inspect codex exec safety capabilities",
                "detail": _sanitize_text(err or out, limit=500),
            }
        exec_help = out or ""
        root_rc, root_out, _ = await self._run([bin_path, "--help"], timeout=20, env=env)
        root_help = root_out if root_rc == 0 else ""
        resume_help = ""
        if re.search(r"\bresume\b", "\n".join([exec_help, root_help])):
            resume_rc, resume_out, _ = await self._run([bin_path, "exec", "resume", "--help"], timeout=20, env=env)
            if resume_rc == 0:
                resume_help = resume_out or ""
        capabilities = _detect_cli_capabilities_from_help(exec_help, root_help, resume_help)
        if not capabilities.ok:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Codex CLI does not advertise required sandbox support",
                "missing_flags": ["--sandbox"],
                "cli_capabilities": capabilities.to_public_dict(),
            }
        return {
            "ok": True,
            "status": "exec_help_ok",
            "cli_capabilities": capabilities.to_public_dict(),
            "_capabilities": capabilities,
        }

    @staticmethod
    def _public_result(result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if not k.startswith("_")}

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
    def _error(
        status: str,
        error: str,
        duration_ms: int,
        model: str | None,
        capabilities: CodexCliCapabilities | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "status": status,
            "error": _sanitize_text(error, limit=500),
            "duration_ms": duration_ms,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "limitations": capabilities.limitations() if capabilities else list(_BASE_LIMITATIONS),
            "cli_capabilities": capabilities.to_public_dict() if capabilities else {},
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
            "limitations": list(_BASE_LIMITATIONS),
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
                base["limitations"] = chat_available.get("limitations") or base["limitations"]
                base["cli_capabilities"] = chat_available.get("cli_capabilities") or {}
                models.append({
                    "id": CODEX_EXPERIMENTAL_MODEL_ID,
                    "display": CODEX_EXPERIMENTAL_MODEL_DISPLAY,
                    "experimental": True,
                    "streaming_supported": False,
                    "session_resume_supported": False,
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
