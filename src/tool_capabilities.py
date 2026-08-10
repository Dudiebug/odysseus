"""Deterministic capability metadata for agent tools.

Model output requests an action; it never supplies the authority for that
action.  This module classifies the effects of each built-in tool and applies
run-local integrity gates before dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from src.tool_security import BUILTIN_EMAIL_TOOLS


class ToolEffect(str, Enum):
    READ_PUBLIC = "read_public"
    READ_WORKSPACE = "read_workspace"
    READ_PRIVATE = "read_private"
    WRITE_WORKSPACE = "write_workspace"
    WRITE_PRIVATE = "write_private"
    EXECUTE_CODE = "execute_code"
    BROKERED_NETWORK_READ = "brokered_network_read"
    NETWORK_EGRESS = "network_egress"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    UI_SIDE_EFFECT = "ui_side_effect"
    ADMIN_CHANGE = "admin_change"
    DESTRUCTIVE = "destructive"
    USER_INTERACTION = "user_interaction"


class ResultIntegrity(str, Enum):
    SYSTEM = "system"
    WORKSPACE_UNTRUSTED = "workspace_untrusted"
    EXTERNAL_UNTRUSTED = "external_untrusted"


@dataclass(frozen=True)
class ToolCapabilities:
    effects: frozenset[ToolEffect]
    result_integrity: ResultIntegrity = ResultIntegrity.SYSTEM
    known: bool = True


def _capabilities(
    *effects: ToolEffect,
    result_integrity: ResultIntegrity = ResultIntegrity.SYSTEM,
) -> ToolCapabilities:
    return ToolCapabilities(frozenset(effects), result_integrity)


_REGISTRY: dict[str, ToolCapabilities] = {}


def _register(
    names: Iterable[str],
    *effects: ToolEffect,
    result_integrity: ResultIntegrity = ResultIntegrity.SYSTEM,
) -> None:
    capabilities = _capabilities(*effects, result_integrity=result_integrity)
    for name in names:
        if name in _REGISTRY:
            raise RuntimeError(f"Duplicate tool capability classification: {name}")
        _REGISTRY[name] = capabilities


_register(
    {"ask_user", "update_plan"},
    ToolEffect.USER_INTERACTION,
)
_register(
    {
        "list_cached_models",
        "list_cookbook_servers",
        "list_downloads",
        "list_models",
        "list_serve_presets",
        "list_served_models",
        "search_hf_models",
    },
    ToolEffect.READ_PUBLIC,
)
_register(
    {"get_workspace", "glob", "grep", "ls", "read_file"},
    ToolEffect.READ_WORKSPACE,
    result_integrity=ResultIntegrity.WORKSPACE_UNTRUSTED,
)
_register(
    {"web_fetch", "web_search"},
    ToolEffect.BROKERED_NETWORK_READ,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_register(
    {
        "list_email_accounts",
        "list_emails",
        "read_email",
        "resolve_contact",
        "scan_email_unsubscribes",
        "search_chats",
        "search_emails",
        "list_sessions",
        "tail_serve_output",
        "vault_get",
        "vault_search",
    },
    ToolEffect.READ_PRIVATE,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_register(
    {"bash", "manage_bg_jobs", "python"},
    ToolEffect.EXECUTE_CODE,
)
_register(
    {"apply_patch", "edit_file", "write_file"},
    ToolEffect.WRITE_WORKSPACE,
)
_register(
    {
        "ai_draft_email_reply",
        "create_document",
        "create_session",
        "draft_email",
        "draft_email_reply",
        "edit_document",
        "manage_calendar",
        "manage_contact",
        "manage_documents",
        "manage_memory",
        "manage_notes",
        "manage_research",
        "manage_session",
        "manage_skills",
        "manage_tasks",
        "pipeline",
        "send_to_session",
        "suggest_document",
        "todowrite",
        "update_document",
    },
    ToolEffect.WRITE_PRIVATE,
)
_register(
    {"chat_with_model", "ask_teacher"},
    ToolEffect.NETWORK_EGRESS,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_register(
    {"download_attachment"},
    ToolEffect.READ_PRIVATE,
    ToolEffect.WRITE_WORKSPACE,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_register(
    {"edit_image", "generate_image", "trigger_research"},
    ToolEffect.NETWORK_EGRESS,
    ToolEffect.WRITE_PRIVATE,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_register(
    {
        "archive_email",
        "bulk_email",
        "mark_email_read",
        "reply_to_email",
        "send_email",
        "unsubscribe_email",
    },
    ToolEffect.EXTERNAL_SIDE_EFFECT,
)
_register(
    {"delete_email"},
    ToolEffect.EXTERNAL_SIDE_EFFECT,
    ToolEffect.DESTRUCTIVE,
)
_register(
    {"ui_control"},
    ToolEffect.UI_SIDE_EFFECT,
)
_register(
    {
        "adopt_served_model",
        "api_call",
        "app_api",
        "cancel_download",
        "download_model",
        "manage_endpoints",
        "manage_mcp",
        "manage_settings",
        "manage_tokens",
        "manage_webhooks",
        "serve_model",
        "serve_preset",
        "stop_served_model",
        "vault_unlock",
    },
    ToolEffect.ADMIN_CHANGE,
)


TOOL_CAPABILITIES: Mapping[str, ToolCapabilities] = MappingProxyType(dict(_REGISTRY))
KNOWN_CAPABILITY_TOOLS = frozenset(TOOL_CAPABILITIES)

_UNKNOWN_CAPABILITIES = _capabilities(
    ToolEffect.READ_PRIVATE,
    ToolEffect.WRITE_WORKSPACE,
    ToolEffect.WRITE_PRIVATE,
    ToolEffect.EXECUTE_CODE,
    ToolEffect.NETWORK_EGRESS,
    ToolEffect.EXTERNAL_SIDE_EFFECT,
    ToolEffect.ADMIN_CHANGE,
    ToolEffect.DESTRUCTIVE,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_UNKNOWN_CAPABILITIES = ToolCapabilities(
    _UNKNOWN_CAPABILITIES.effects,
    _UNKNOWN_CAPABILITIES.result_integrity,
    known=False,
)
_BROWSER_MCP_READ_CAPABILITIES = _capabilities(
    ToolEffect.BROKERED_NETWORK_READ,
    result_integrity=ResultIntegrity.EXTERNAL_UNTRUSTED,
)
_BROWSER_MCP_READ_TOOLS = frozenset(
    {
        "mcp__builtin_browser__browser_console_messages",
        "mcp__builtin_browser__browser_network_requests",
        "mcp__builtin_browser__browser_snapshot",
        "mcp__builtin_browser__browser_take_screenshot",
    }
)


def capabilities_for_tool(tool_name: Any) -> ToolCapabilities:
    """Return deterministic capabilities; malformed and unknown tools fail high."""
    if not isinstance(tool_name, str) or not tool_name:
        return _UNKNOWN_CAPABILITIES
    capabilities = TOOL_CAPABILITIES.get(tool_name)
    if capabilities is not None:
        return capabilities
    if tool_name.startswith("mcp__email__"):
        bare_name = tool_name[len("mcp__email__"):]
        capabilities = TOOL_CAPABILITIES.get(bare_name)
        if bare_name in BUILTIN_EMAIL_TOOLS and capabilities is not None:
            return capabilities
    if tool_name in _BROWSER_MCP_READ_TOOLS:
        return _BROWSER_MCP_READ_CAPABILITIES
    return _UNKNOWN_CAPABILITIES


POST_EXTERNAL_BLOCKED_EFFECTS = frozenset(
    {
        ToolEffect.READ_PRIVATE,
        ToolEffect.WRITE_WORKSPACE,
        ToolEffect.WRITE_PRIVATE,
        ToolEffect.EXECUTE_CODE,
        ToolEffect.NETWORK_EGRESS,
        ToolEffect.EXTERNAL_SIDE_EFFECT,
        ToolEffect.UI_SIDE_EFFECT,
        ToolEffect.ADMIN_CHANGE,
        ToolEffect.DESTRUCTIVE,
    }
)


@dataclass(frozen=True)
class ToolGateDecision:
    allowed: bool
    reason: str | None = None


_EXTERNAL_MESSAGE_SOURCES = frozenset(
    {
        "injected research context",
        "prefetched search context",
        "research context",
        "web search results",
        "youtube transcript",
    }
)
_EXTERNAL_MESSAGE_SOURCE_PREFIXES = ("web page:",)


def messages_contain_external_untrusted_context(messages: Iterable[dict]) -> bool:
    """Detect explicitly labelled external context already present in a run."""
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        metadata = message.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("trusted") is not False:
            continue
        if metadata.get("provenance_origin") == "external":
            return True
        source = metadata.get("source")
        if not isinstance(source, str):
            continue
        normalized_source = source.strip().casefold()
        if normalized_source in _EXTERNAL_MESSAGE_SOURCES:
            return True
        if normalized_source.startswith(_EXTERNAL_MESSAGE_SOURCE_PREFIXES):
            return True
    return False


@dataclass
class ToolRunSecurityContext:
    """Server-owned integrity state for one agent run."""

    external_untrusted_context_seen: bool = False
    external_sources: list[str] = field(default_factory=list)

    def decision_for(self, tool_name: Any) -> ToolGateDecision:
        if not self.external_untrusted_context_seen:
            return ToolGateDecision(True)
        capabilities = capabilities_for_tool(tool_name)
        blocked_effects = capabilities.effects & POST_EXTERNAL_BLOCKED_EFFECTS
        if capabilities.known and not blocked_effects:
            return ToolGateDecision(True)
        effects = ", ".join(sorted(effect.value for effect in blocked_effects))
        if not capabilities.known:
            effects = "unknown/high-impact"
        return ToolGateDecision(
            False,
            (
                "External untrusted context has already influenced this run. "
                f"Tool '{tool_name}' requires a separate user-authorized action "
                f"because it can cause {effects}."
            ),
        )

    def observe_tool_result(self, tool_name: Any, result: Any) -> None:
        if not isinstance(result, dict):
            return
        if result.get("blocked") or result.get("error") or result.get("exit_code") not in (None, 0):
            return
        capabilities = capabilities_for_tool(tool_name)
        if capabilities.result_integrity is ResultIntegrity.EXTERNAL_UNTRUSTED:
            self.external_untrusted_context_seen = True
            if isinstance(tool_name, str) and tool_name not in self.external_sources:
                self.external_sources.append(tool_name)


def blocked_tool_result(tool_name: Any, reason: str) -> tuple[str, dict]:
    return (
        f"{tool_name}: BLOCKED",
        {
            "error": reason,
            "exit_code": 1,
            "blocked": True,
            "policy": "external_untrusted_context",
        },
    )
