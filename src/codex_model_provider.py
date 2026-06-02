"""Codex provider status shell backend."""

from __future__ import annotations

import os
from typing import Any


CODEX_MODEL_PROVIDER_FLAG = "ODYSSEUS_CODEX_MODEL_PROVIDER_ENABLED"
CODEX_EXPERIMENTAL_MODEL_ID = "codex-cli/chatgpt-experimental"
_CODEX_EXPERIMENTAL_MODEL_LABEL = "Codex CLI / ChatGPT (experimental)"
_BASE_LIMITATIONS = [
    "Admin-only provider status shell.",
    "Feature flag defaults to disabled.",
    "CLI capability probing, model configuration, and chat routing are not wired in this branch.",
]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_model_provider_enabled() -> bool:
    return _truthy(os.getenv(CODEX_MODEL_PROVIDER_FLAG, "false"))


def _synthetic_model() -> dict[str, Any]:
    return {
        "id": CODEX_EXPERIMENTAL_MODEL_ID,
        "display": _CODEX_EXPERIMENTAL_MODEL_LABEL,
        "label": _CODEX_EXPERIMENTAL_MODEL_LABEL,
        "description": "Synthetic provider shell only. Capability probing is not wired yet.",
        "source": "synthetic",
        "experimental": True,
        "hidden": False,
        "enabled": True,
        "thinking_effort": None,
        "streaming_supported": False,
        "thinking_supported": False,
        "thinking_effort_levels": [],
        "thinking_activity_supported": False,
        "session_resume_supported": False,
        "tool_calling_supported": False,
        "agent_tools_supported": False,
    }


class CodexModelProvider:
    """Admin-only provider status shell."""

    async def status(self) -> dict[str, Any]:
        enabled = codex_model_provider_enabled()
        base = {
            "feature_enabled": enabled,
            "feature_flag": CODEX_MODEL_PROVIDER_FLAG,
            "provider": "codex_cli",
            "experimental": True,
            "connector_enabled": False,
            "selected_models": [],
            "manual_models": [],
            "hidden_models": [],
            "disabled_models": [],
            "recommended_models": [],
            "models": [],
            "chat_supported": False,
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "thinking_supported": False,
            "thinking_effort_levels": [],
            "thinking_activity_supported": False,
            "model_discovery": {"source": "disabled" if not enabled else "static"},
            "limitations": list(_BASE_LIMITATIONS),
            "auth": {"status": "", "auth_mode": ""},
        }
        if not enabled:
            return {
                **base,
                "status": "disabled",
                "cli_available": False,
                "authenticated": False,
                "requires_sign_in": False,
            }

        return {
            **base,
            "status": "enabled",
            "cli_available": None,
            "authenticated": None,
            "requires_sign_in": False,
            "models": [_synthetic_model()],
        }
