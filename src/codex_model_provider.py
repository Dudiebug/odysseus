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

_LIMITATIONS = [
    "Non-streaming: returns one completed assistant message.",
    "Stateless: session/resume is not implemented.",
    "Codex tool execution is not mapped into Odysseus agent rounds.",
    "The adapter requires Codex CLI sandbox/approval flags before running.",
]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_model_provider_enabled() -> bool:
    return _truthy(os.getenv(CODEX_MODEL_PROVIDER_FLAG, "false"))


def _sanitize_text(text: str | None, limit: int = 2000) -> str:
    safe = text or ""
    for pattern in _TOKEN_PATTERNS:
        safe = pattern.sub("<redacted-token>", safe)
    return safe.strip()[:limit]


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
    ) -> None:
        self._auth_service_getter = auth_service_getter or get_codex_auth_service
        self._runner = runner

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
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "supports_json": help_result.get("supports_json", False),
            "limitations": list(_LIMITATIONS),
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
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

        with tempfile.TemporaryDirectory(prefix="odysseus-codex-chat-") as workdir:
            args = [
                preflight["bin_path"],
                "exec",
                "--sandbox",
                "read-only",
                "--ask-for-approval",
                "never",
                prompt,
            ]
            rc, out, err = await self._run(
                args,
                timeout=timeout,
                cwd=workdir,
                env=preflight["env"],
            )

        duration_ms = round((time.time() - started) * 1000)
        if rc == 124:
            return self._error("timeout", "Codex CLI timed out", duration_ms, model)
        if rc != 0:
            detail = _sanitize_text(err or out, limit=500)
            return self._error("cli_failed", detail or "Codex CLI failed", duration_ms, model)

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

    async def _exec_help(self, bin_path: str, env: dict[str, str]) -> dict[str, Any]:
        rc, out, err = await self._run([bin_path, "exec", "--help"], timeout=20, env=env)
        if rc != 0:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Unable to inspect codex exec safety flags",
                "detail": _sanitize_text(err or out, limit=500),
            }
        help_text = out or ""
        missing = [flag for flag in ("--sandbox", "--ask-for-approval") if flag not in help_text]
        if missing:
            return {
                "ok": False,
                "status": "unsupported_unsafe_cli_mode",
                "error": "Codex CLI does not advertise required safety flags",
                "missing_flags": missing,
            }
        return {
            "ok": True,
            "status": "exec_help_ok",
            "supports_json": "--json" in help_text,
            "supports_model": "--model" in help_text or " -m" in help_text,
        }

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
    ) -> dict[str, Any]:
        return await self._chat_adapter.complete(messages, model=model, timeout_seconds=timeout_seconds)
