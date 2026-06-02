"""Codex CLI model-provider capability boundary.

This module implements the Codex CLI provider boundary without reading Codex
credential files or treating completed CLI output as token streaming. Normal
API-provider behavior remains deliberately out of scope.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from src.codex_auth import get_codex_auth_service
from src.settings import load_settings, save_settings


CODEX_MODEL_PROVIDER_FLAG = "ODYSSEUS_CODEX_MODEL_PROVIDER_ENABLED"
CODEX_EXPERIMENTAL_MODEL_ID = "codex-cli/chatgpt-experimental"
CODEX_EXPERIMENTAL_ENDPOINT_URL = "odysseus://codex-cli"
CODEX_CHAT_TIMEOUT_SECONDS = 120
CODEX_SETTINGS_KEY = "codex_model_provider"
CODEX_RECOMMENDED_MODELS = (
    {
        "id": "gpt-5.5",
        "label": "GPT-5.5",
        "description": "newest frontier model for complex coding, computer use, knowledge work, and research workflows",
        "default": True,
    },
    {
        "id": "gpt-5.4",
        "label": "GPT-5.4",
        "description": "flagship frontier model for professional work, stronger reasoning, tool use, and agentic workflows",
    },
    {
        "id": "gpt-5.4-mini",
        "label": "GPT-5.4 Mini",
        "description": "fast and efficient model for responsive coding tasks and subagents",
    },
    {
        "id": "gpt-5.3-codex",
        "label": "GPT-5.3 Codex",
        "description": "coding model for complex software engineering",
    },
    {
        "id": "gpt-5.3-codex-spark",
        "label": "GPT-5.3 Codex Spark",
        "description": "research preview for near-instant real-time coding iteration; availability may depend on plan",
    },
    {
        "id": "gpt-5.2",
        "label": "GPT-5.2",
        "description": "previous general-purpose coding and agentic model",
    },
)
_CODEX_RECOMMENDED_BY_ID = {item["id"]: item for item in CODEX_RECOMMENDED_MODELS}

_TOKEN_PATTERNS = (
    re.compile(r"(?i)(access_token|refresh_token|id_token)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
)

_BASE_LIMITATIONS = [
    "Chat picker integration only; not a normal API endpoint.",
    "Stateless: session/resume is not implemented.",
    "Codex tool execution is not mapped into Odysseus agent rounds.",
    "The adapter requires Codex CLI sandbox support and runs with read-only sandbox mode.",
]

_APPROVAL_FLAGS = ("--ask-for-approval", "--approval-policy", "--approval")
_DANGEROUS_FLAGS = ("--dangerously-bypass-approvals-and-sandbox", "--yolo")
_SKIP_GIT_REPO_CHECK_FLAG = "--skip-git-repo-check"
_JSON_OUTPUT_FLAG = "--json"
_MODEL_FLAGS = ("--model", "-m")
_REASONING_FLAGS = ("--reasoning-effort", "--effort", "--thinking", "--thinking-level")
_REASONING_LEVELS = ("low", "medium", "high", "maximum")
_STREAM_CONTAINER_KEYS = ("message", "item", "data", "response")
_STREAM_NESTED_KEYS = (*_STREAM_CONTAINER_KEYS, "output")
_STREAM_METRIC_KEYS = ("metrics", "usage")
_LIFECYCLE_EVENT_NAMES = {
    "started",
    "completed",
    "failed",
    "thread.started",
    "thread.completed",
    "turn.started",
    "turn.completed",
    "turn.failed",
}
_LIFECYCLE_EVENT_SUFFIXES = (".started", ".completed", ".failed")
_TRUST_DIRECTORY_PATTERNS = (
    re.compile(r"not\s+(inside\s+)?(a\s+)?trusted\s+(directory|git\s+repository)", re.I),
    re.compile(r"trust(ed)?\s+directory", re.I),
    re.compile(r"--skip-git-repo-check", re.I),
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_model_provider_enabled() -> bool:
    return _truthy(os.getenv(CODEX_MODEL_PROVIDER_FLAG, "false"))


def is_codex_model_selection(endpoint_url: str | None, model: str | None = None) -> bool:
    return (endpoint_url or "").strip() == CODEX_EXPERIMENTAL_ENDPOINT_URL


def _sanitize_model_id(model_id: Any) -> str:
    text = str(model_id or "").strip()
    if not text or len(text) > 160:
        return ""
    if any(ch.isspace() for ch in text):
        return ""
    if any(ord(ch) < 32 for ch in text):
        return ""
    return text


def _unique_model_ids(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for value in values:
        model_id = _sanitize_model_id(value)
        if model_id and model_id not in seen:
            seen.add(model_id)
            out.append(model_id)
    return out


def _sanitize_optional_text(value: Any, *, limit: int = 300) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if any(ord(ch) < 32 for ch in text):
        return ""
    return text[:limit]


def _sanitize_thinking_effort(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in _REASONING_LEVELS else None


def codex_recommended_models() -> list[dict[str, Any]]:
    return [dict(item) for item in CODEX_RECOMMENDED_MODELS]


def _recommended_model_meta(model_id: str) -> dict[str, Any]:
    item = _CODEX_RECOMMENDED_BY_ID.get(model_id) or {}
    return {
        "id": model_id,
        "label": item.get("label") or _model_display(model_id),
        "description": item.get("description") or "",
        "source": "recommended" if item else "custom",
        "default": bool(item.get("default")),
    }


def _sanitize_selected_model_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        value = {"id": value}
    model_id = _sanitize_model_id(value.get("id") or value.get("model_id"))
    if not model_id:
        return None
    recommended = _recommended_model_meta(model_id)
    source = str(value.get("source") or recommended["source"]).strip().lower()
    if source not in {"recommended", "custom", "manual", "codex_cli"}:
        source = recommended["source"]
    source = "recommended" if source in {"recommended", "codex_cli"} and model_id in _CODEX_RECOMMENDED_BY_ID else "custom"
    label = _sanitize_optional_text(value.get("label")) or recommended["label"]
    description = _sanitize_optional_text(value.get("description")) or recommended["description"]
    return {
        "id": model_id,
        "label": label,
        "description": description,
        "source": source,
        "enabled": bool(value.get("enabled", True)),
        "hidden": bool(value.get("hidden", False)),
        "thinking_effort": _sanitize_thinking_effort(value.get("thinking_effort")),
    }


def _normalize_selected_models(values: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for value in values:
        item = _sanitize_selected_model_entry(value)
        if not item:
            continue
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        out.append(item)
    return out


def _legacy_selected_models(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    manual = _unique_model_ids(cfg.get("manual_models"))
    hidden = set(_unique_model_ids(cfg.get("hidden_models")))
    disabled = set(_unique_model_ids(cfg.get("disabled_models")))
    out: list[dict[str, Any]] = []
    for model_id in manual:
        meta = _recommended_model_meta(model_id)
        out.append({
            "id": model_id,
            "label": meta["label"],
            "description": meta["description"],
            "source": meta["source"],
            "enabled": model_id not in disabled,
            "hidden": model_id in hidden,
            "thinking_effort": None,
        })
    return out


def _config_projections(selected_models: list[dict[str, Any]]) -> dict[str, Any]:
    selected_ids = [item["id"] for item in selected_models]
    hidden_ids = [item["id"] for item in selected_models if item.get("hidden")]
    disabled_ids = [item["id"] for item in selected_models if not item.get("enabled", True)]
    custom_models = [item["id"] for item in selected_models if item.get("source") == "custom"]
    return {
        "manual_models": selected_ids,
        "hidden_models": hidden_ids,
        "disabled_models": disabled_ids,
        "custom_models": custom_models,
        "hidden_model_ids": hidden_ids,
    }


def load_codex_model_config() -> dict[str, Any]:
    cfg = load_settings().get(CODEX_SETTINGS_KEY) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    selected_models = _normalize_selected_models(cfg.get("selected_models"))
    if not selected_models:
        selected_models = _legacy_selected_models(cfg)
    connector_enabled = cfg.get("connector_enabled")
    if connector_enabled is None:
        connector_enabled = bool(selected_models)
    else:
        connector_enabled = bool(connector_enabled)
    task_defaults = cfg.get("task_defaults") or {}
    if not isinstance(task_defaults, dict):
        task_defaults = {}
    projections = _config_projections(selected_models)
    return {
        "connector_enabled": connector_enabled,
        "selected_models": selected_models,
        **projections,
        "task_defaults": task_defaults,
    }


def save_codex_model_config(config: dict[str, Any]) -> dict[str, Any]:
    selected_models = _normalize_selected_models(config.get("selected_models"))
    projections = _config_projections(selected_models)
    clean = {
        "connector_enabled": bool(config.get("connector_enabled")) or bool(selected_models),
        "selected_models": selected_models,
        **projections,
        "task_defaults": config.get("task_defaults") if isinstance(config.get("task_defaults"), dict) else {},
    }
    settings = load_settings()
    settings[CODEX_SETTINGS_KEY] = clean
    save_settings(settings)
    return load_codex_model_config()


def update_codex_model_config(
    *,
    add_model: str | None = None,
    hide_model: str | None = None,
    restore_model: str | None = None,
    enable_model: str | None = None,
    disable_model: str | None = None,
    remove_model: str | None = None,
    thinking_model: str | None = None,
    thinking_effort: str | None = None,
    clear_all_models: bool = False,
    connector_enabled: bool | None = None,
) -> dict[str, Any]:
    cfg = load_codex_model_config()
    selected = [dict(item) for item in cfg["selected_models"]]

    def _find(model_id: str) -> dict[str, Any] | None:
        for item in selected:
            if item["id"] == model_id:
                return item
        return None

    if clear_all_models:
        selected = []
        cfg["connector_enabled"] = False

    if connector_enabled is not None:
        cfg["connector_enabled"] = bool(connector_enabled)

    model_id = _sanitize_model_id(add_model)
    if model_id:
        item = _find(model_id)
        if item:
            item["hidden"] = False
            item["enabled"] = True
        else:
            meta = _recommended_model_meta(model_id)
            selected.append({
                "id": model_id,
                "label": meta["label"],
                "description": meta["description"],
                "source": meta["source"],
                "enabled": True,
                "hidden": False,
                "thinking_effort": None,
            })
        cfg["connector_enabled"] = True

    model_id = _sanitize_model_id(remove_model)
    if model_id:
        selected = [item for item in selected if item["id"] != model_id]

    model_id = _sanitize_model_id(hide_model)
    if model_id:
        item = _find(model_id)
        if item:
            item["hidden"] = True
            item["enabled"] = True

    model_id = _sanitize_model_id(restore_model)
    if model_id:
        item = _find(model_id)
        if item:
            item["hidden"] = False

    model_id = _sanitize_model_id(enable_model)
    if model_id:
        item = _find(model_id)
        if item:
            item["enabled"] = True
            item["hidden"] = False

    model_id = _sanitize_model_id(disable_model)
    if model_id:
        item = _find(model_id)
        if item:
            item["enabled"] = False

    model_id = _sanitize_model_id(thinking_model)
    if model_id:
        item = _find(model_id)
        if item:
            item["thinking_effort"] = _sanitize_thinking_effort(thinking_effort)

    if not selected and connector_enabled is None and not clear_all_models:
        cfg["connector_enabled"] = False

    cfg["selected_models"] = selected
    return save_codex_model_config(cfg)


def _model_display(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1] or model_id


def first_enabled_codex_model(config: dict[str, Any] | None = None) -> str:
    cfg = config or load_codex_model_config()
    if not cfg.get("connector_enabled"):
        return ""
    for item in cfg.get("selected_models") or []:
        if item.get("enabled", True) and not item.get("hidden"):
            return item.get("id") or ""
    return ""


def codex_model_default_reasoning_effort(model_id: str, config: dict[str, Any] | None = None) -> str | None:
    target = _sanitize_model_id(model_id)
    if not target:
        return None
    cfg = config or load_codex_model_config()
    for item in cfg.get("selected_models") or []:
        if item.get("id") == target:
            return _sanitize_thinking_effort(item.get("thinking_effort"))
    return None


def is_codex_model_available(model_id: str | None, *, public_only: bool = True, config: dict[str, Any] | None = None) -> bool:
    target = _sanitize_model_id(model_id)
    if not target:
        return False
    cfg = config or load_codex_model_config()
    if not cfg.get("connector_enabled"):
        return False
    for item in cfg.get("selected_models") or []:
        if item.get("id") != target:
            continue
        if not public_only:
            return True
        return bool(item.get("enabled", True)) and not bool(item.get("hidden"))
    return False


def codex_model_list_item(models: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    public_models = [m for m in (models or []) if m.get("enabled", True) and not m.get("hidden")]
    model_ids = [m["id"] for m in public_models if m.get("id")]
    return {
        "host": "codex",
        "port": 0,
        "url": CODEX_EXPERIMENTAL_ENDPOINT_URL,
        "models": model_ids,
        "models_display": [m.get("display") or _model_display(m["id"]) for m in public_models if m.get("id")],
        "models_extra": [],
        "models_extra_display": [],
        "endpoint_id": None,
        "endpoint_name": "Codex / ChatGPT",
        "category": "api",
        "model_type": "llm",
        "experimental": True,
        "provider": "codex_cli",
        "capabilities": {
            "chat_supported": True,
            "streaming_supported": any(m.get("streaming_supported") for m in public_models),
            "tool_calling_supported": False,
            "agent_tools_supported": False,
        },
    }


def _looks_like_lifecycle_event_name(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and (text in _LIFECYCLE_EVENT_NAMES or text.endswith(_LIFECYCLE_EVENT_SUFFIXES))


def _sanitize_text(text: str | None, limit: int = 2000, *, strip: bool = True) -> str:
    safe = text or ""
    for pattern in _TOKEN_PATTERNS:
        safe = pattern.sub("<redacted-token>", safe)
    if strip:
        safe = safe.strip()
    return safe[:limit]


@dataclass(frozen=True)
class CodexCliCapabilities:
    """Detected Codex CLI command surface for the installed binary."""

    sandbox_flag: str | None
    sandbox_modes: tuple[str, ...] = ()
    approval_flag: str | None = None
    skip_git_repo_check_flag: str | None = None
    supports_json: bool = False
    supports_model: bool = False
    model_flag: str | None = None
    reasoning_flag: str | None = None
    reasoning_levels: tuple[str, ...] = ()
    resume_supported: bool = False
    resume_last_supported: bool = False
    resume_session_id_supported: bool = False
    dangerous_flags: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.sandbox_flag)

    @property
    def skip_git_repo_check_supported(self) -> bool:
        return bool(self.skip_git_repo_check_flag)

    @property
    def streaming_supported(self) -> bool:
        return self.ok and self.supports_json

    def limitations(self) -> list[str]:
        out = list(_BASE_LIMITATIONS)
        if self.streaming_supported:
            out.append("Codex CLI JSON event streaming is available for chat streaming.")
        else:
            out.append("Streaming is not available because Codex CLI JSON event output is not advertised.")
        if self.approval_flag:
            out.append(f"Codex CLI advertises {self.approval_flag}, but this provider does not pass approval flags.")
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
            "skip_git_repo_check_supported": self.skip_git_repo_check_supported,
            "skip_git_repo_check_flag": self.skip_git_repo_check_flag,
            "resume_supported": self.resume_supported,
            "resume_last_supported": self.resume_last_supported,
            "resume_session_id_supported": self.resume_session_id_supported,
            "json_output_supported": self.supports_json,
            "streaming_supported": self.streaming_supported,
            "model_flag_supported": self.supports_model,
            "model_flag": self.model_flag,
            "reasoning_effort_supported": bool(self.reasoning_flag),
            "reasoning_effort_flag": self.reasoning_flag,
            "reasoning_effort_levels": list(self.reasoning_levels),
            "dangerous_flags_advertised": list(self.dangerous_flags),
        }

    def build_exec_args(
        self,
        bin_path: str,
        prompt: str,
        *,
        json_output: bool = False,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> list[str]:
        if not self.sandbox_flag:
            raise ValueError("Codex CLI sandbox support is required")
        if json_output and not self.supports_json:
            raise ValueError("Codex CLI JSON output support is required for streaming")
        args = [bin_path, "exec", self.sandbox_flag, "read-only"]
        selected_model = _sanitize_model_id(model)
        if selected_model and selected_model != CODEX_EXPERIMENTAL_MODEL_ID:
            if not self.supports_model or not self.model_flag:
                raise ValueError("Codex CLI model selection is not supported")
            args.extend([self.model_flag, selected_model])
        selected_effort = str(reasoning_effort or "").strip().lower()
        if selected_effort:
            if not self.reasoning_flag or selected_effort not in self.reasoning_levels:
                raise ValueError("Codex CLI reasoning effort selection is not supported")
            args.extend([self.reasoning_flag, selected_effort])
        if self.skip_git_repo_check_flag:
            args.append(self.skip_git_repo_check_flag)
        if json_output:
            args.append(_JSON_OUTPUT_FLAG)
        args.append(prompt)
        if any(flag in args for flag in _DANGEROUS_FLAGS):
            raise ValueError("Unsafe Codex CLI flag refused")
        return args


def _detect_sandbox_flag(help_text: str) -> str | None:
    if "--sandbox" in help_text:
        return "--sandbox"
    return None


def _detect_skip_git_repo_check_flag(help_text: str) -> str | None:
    if _SKIP_GIT_REPO_CHECK_FLAG in help_text:
        return _SKIP_GIT_REPO_CHECK_FLAG
    return None


def _detect_approval_flag(help_text: str) -> str | None:
    for flag in _APPROVAL_FLAGS:
        if flag in help_text:
            return flag
    return None


def _detect_model_flag(help_text: str) -> str | None:
    if "--model" in help_text:
        return "--model"
    if re.search(r"(^|\s)-m([,\s]|$)", help_text):
        return "-m"
    return None


def _detect_reasoning_flag(help_text: str) -> str | None:
    for flag in _REASONING_FLAGS:
        if flag in help_text:
            return flag
    return None


def _detect_reasoning_levels(help_text: str, flag: str | None) -> tuple[str, ...]:
    if not flag:
        return ()
    text = help_text.lower()
    found = [level for level in _REASONING_LEVELS if level in text]
    if found:
        return tuple(found)
    return ("low", "medium", "high")


def _detect_cli_capabilities_from_help(
    exec_help: str,
    root_help: str = "",
    resume_help: str = "",
) -> CodexCliCapabilities:
    combined = "\n".join([exec_help or "", root_help or "", resume_help or ""])
    resume_text = "\n".join([exec_help or "", root_help or ""])
    model_flag = _detect_model_flag(exec_help or "")
    reasoning_flag = _detect_reasoning_flag(exec_help or "")
    return CodexCliCapabilities(
        sandbox_flag=_detect_sandbox_flag(exec_help or ""),
        sandbox_modes=tuple(
            mode for mode in ("read-only", "workspace-write", "danger-full-access")
            if mode in (exec_help or "")
        ),
        approval_flag=_detect_approval_flag(exec_help or ""),
        skip_git_repo_check_flag=_detect_skip_git_repo_check_flag(combined),
        supports_json=_JSON_OUTPUT_FLAG in exec_help,
        supports_model=bool(model_flag),
        model_flag=model_flag,
        reasoning_flag=reasoning_flag,
        reasoning_levels=_detect_reasoning_levels(exec_help or "", reasoning_flag),
        resume_supported=bool(re.search(r"\bresume\b", resume_text)),
        resume_last_supported="--last" in resume_help,
        resume_session_id_supported=bool(re.search(r"\b(session|SESSION)(_ID| id| id)?\b", resume_help, re.I)),
        dangerous_flags=tuple(flag for flag in _DANGEROUS_FLAGS if flag in combined),
    )


def _advertised_model_list_commands(root_help: str) -> list[list[str]]:
    commands: list[list[str]] = []
    if re.search(r"(?im)^\s+models?\s+", root_help or ""):
        commands.extend([["models", "--json"], ["models"], ["model", "list", "--json"], ["model", "list"]])
    elif re.search(r"(?im)^\s+model\s+", root_help or ""):
        commands.extend([["model", "list", "--json"], ["model", "list"]])
    return commands


def _extract_model_ids_from_payload(value: Any) -> list[str]:
    if isinstance(value, str):
        return [_sanitize_model_id(value)]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_extract_model_ids_from_payload(item))
        return [model_id for model_id in out if model_id]
    if isinstance(value, dict):
        for key in ("id", "name", "model"):
            model_id = _sanitize_model_id(value.get(key))
            if model_id:
                return [model_id]
        out: list[str] = []
        for key in ("data", "models", "items"):
            out.extend(_extract_model_ids_from_payload(value.get(key)))
        return [model_id for model_id in out if model_id]
    return []


def _parse_model_list_output(output: str) -> list[str]:
    text = output or ""
    parsed: list[str] = []
    try:
        parsed = _extract_model_ids_from_payload(json.loads(text))
    except Exception:
        parsed = []
    if not parsed:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.lower().startswith(("usage:", "error", "warning")):
                continue
            parsed.append(stripped.split()[0])
    return _unique_model_ids(parsed)


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
            "streaming_supported": capabilities.streaming_supported,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "supports_json": capabilities.supports_json,
            "supports_model": capabilities.supports_model,
            "discovered_models": help_result.get("discovered_models") or [],
            "model_discovery": help_result.get("model_discovery") or {},
            "cli_capabilities": capabilities.to_public_dict(),
            "limitations": capabilities.limitations(),
            "_preflight": preflight,
            "_capabilities": capabilities,
        }

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        reasoning_effort: str | None = None,
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
            try:
                args = capabilities.build_exec_args(
                    preflight["bin_path"],
                    prompt,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            except ValueError as exc:
                return self._error("unsupported_option", str(exc), round((time.time() - started) * 1000), model, capabilities)
            rc, out, err = await self._run(
                args,
                timeout=timeout,
                cwd=workdir,
                env=preflight["env"],
            )
            if rc != 0 and _is_trust_directory_error(err or out) and not capabilities.skip_git_repo_check_supported:
                capabilities = replace(capabilities, skip_git_repo_check_flag=_SKIP_GIT_REPO_CHECK_FLAG)
                args = capabilities.build_exec_args(
                    preflight["bin_path"],
                    prompt,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
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
            if _is_trust_directory_error(detail) and not capabilities.skip_git_repo_check_supported:
                return self._error(
                    "trusted_directory_required",
                    "Codex CLI requires --skip-git-repo-check, but this CLI did not advertise that flag.",
                    duration_ms,
                    model,
                    capabilities,
                )
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
            "reasoning_effort": reasoning_effort if capabilities.reasoning_flag else None,
            "limitations": capabilities.limitations(),
            "cli_capabilities": capabilities.to_public_dict(),
            "streaming_supported": capabilities.streaming_supported,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
        }

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
        odysseus_session_id: str | None = None,
        allow_one_shot_fallback: bool = False,
    ):
        started = time.time()
        availability = await self._available_internal()
        if not availability.get("ok"):
            yield self._stream_error_event(
                self._public_result(availability),
                started=started,
                model=model,
            )
            return

        preflight = availability["_preflight"]
        capabilities = availability["_capabilities"]
        if not capabilities.streaming_supported:
            if allow_one_shot_fallback:
                result = await self.complete(
                    messages,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    timeout_seconds=timeout_seconds,
                    odysseus_session_id=odysseus_session_id,
                )
                if result.get("ok"):
                    message = result.get("message") or ""
                    if message:
                        yield {"type": "delta", "delta": message}
                    yield {
                        "type": "done",
                        "message": message,
                        "model": result.get("model") or model or CODEX_EXPERIMENTAL_MODEL_ID,
                        "duration_ms": result.get("duration_ms"),
                    }
                else:
                    yield self._stream_error_event(result, started=started, model=model)
                return
            yield self._stream_error_event(
                {
                    "ok": False,
                    "status": "streaming_not_supported",
                    "error": "Codex CLI does not advertise JSON event output; use test-chat one-shot fallback.",
                    "cli_capabilities": capabilities.to_public_dict(),
                    "limitations": capabilities.limitations(),
                },
                started=started,
                model=model,
            )
            return

        prompt = self._build_prompt(messages)
        timeout = max(1, min(int(timeout_seconds or CODEX_CHAT_TIMEOUT_SECONDS), 300))
        with tempfile.TemporaryDirectory(prefix="odysseus-codex-chat-stream-") as workdir:
            try:
                args = capabilities.build_exec_args(
                    preflight["bin_path"],
                    prompt,
                    json_output=True,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            except ValueError as exc:
                yield self._stream_error_event(
                    self._error("unsupported_option", str(exc), round((time.time() - started) * 1000), model, capabilities),
                    started=started,
                    model=model,
                )
                return
            async for event in self._stream_exec(
                args,
                timeout=timeout,
                cwd=workdir,
                env=preflight["env"],
                model=model,
                reasoning_effort=reasoning_effort,
                capabilities=capabilities,
            ):
                yield event

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
        discovered_models: list[str] = []
        model_discovery = {"source": "none", "commands_tried": []}
        for command in _advertised_model_list_commands(root_help):
            model_discovery["commands_tried"].append(" ".join(command))
            model_rc, model_out, _ = await self._run([bin_path, *command], timeout=20, env=env)
            if model_rc != 0:
                continue
            discovered_models = _parse_model_list_output(model_out or "")
            if discovered_models:
                model_discovery["source"] = "codex_cli"
                model_discovery["command"] = " ".join(command)
                break
        if not discovered_models and capabilities.supports_model:
            model_discovery["source"] = "manual"
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
            "discovered_models": discovered_models,
            "model_discovery": model_discovery,
            "_capabilities": capabilities,
        }

    @staticmethod
    def _public_result(result: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in result.items() if not k.startswith("_")}

    async def _stream_exec(
        self,
        args: list[str],
        *,
        timeout: int,
        cwd: str,
        env: dict[str, str],
        model: str | None,
        capabilities: CodexCliCapabilities,
        reasoning_effort: str | None = None,
        allow_trust_retry: bool = True,
    ):
        started = time.time()
        stdout_lines: list[str] = []
        stderr_text = ""
        response_parts: list[str] = []
        emitted_delta = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        except Exception as exc:
            yield self._stream_error_event(
                {
                    "ok": False,
                    "status": "cli_start_failed",
                    "error": f"Failed to start Codex CLI: {exc.__class__.__name__}",
                    "cli_capabilities": capabilities.to_public_dict(),
                    "limitations": capabilities.limitations(),
                },
                started=started,
                model=model,
            )
            return

        async def read_stderr() -> str:
            if not proc.stderr:
                return ""
            raw = await proc.stderr.read()
            return (raw or b"").decode("utf-8", errors="replace")

        stderr_task = asyncio.create_task(read_stderr())
        try:
            while True:
                if not proc.stdout:
                    break
                line_bytes = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace")
                stdout_lines.append(line)
                for event in self._events_from_json_line(line):
                    if event["type"] == "delta":
                        emitted_delta = True
                        response_parts.append(event["delta"])
                    yield event
            return_code = await asyncio.wait_for(proc.wait(), timeout=5)
            stderr_text = await asyncio.wait_for(stderr_task, timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            if not stderr_task.done():
                stderr_task.cancel()
            yield self._stream_error_event(
                {
                    "ok": False,
                    "status": "timeout",
                    "error": "Codex CLI timed out",
                    "cli_capabilities": capabilities.to_public_dict(),
                    "limitations": capabilities.limitations(),
                },
                started=started,
                model=model,
            )
            return

        if return_code != 0:
            detail = _sanitize_text(stderr_text or "".join(stdout_lines), limit=500)
            if _is_trust_directory_error(detail) and not capabilities.skip_git_repo_check_supported and allow_trust_retry:
                retry_capabilities = replace(capabilities, skip_git_repo_check_flag=_SKIP_GIT_REPO_CHECK_FLAG)
                retry_args = retry_capabilities.build_exec_args(
                    args[0],
                    args[-1],
                    json_output=True,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
                async for event in self._stream_exec(
                    retry_args,
                    timeout=timeout,
                    cwd=cwd,
                    env=env,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    capabilities=retry_capabilities,
                    allow_trust_retry=False,
                ):
                    yield event
                return
            status = "trusted_directory_required" if _is_trust_directory_error(detail) else "cli_failed"
            yield self._stream_error_event(
                {
                    "ok": False,
                    "status": status,
                    "error": detail or "Codex CLI failed",
                    "cli_capabilities": capabilities.to_public_dict(),
                    "limitations": capabilities.limitations(),
                },
                started=started,
                model=model,
            )
            return

        full_response = "".join(response_parts)
        if not emitted_delta:
            full_response = self._extract_message("".join(stdout_lines))
            if full_response:
                yield {"type": "delta", "delta": full_response}
        if not full_response:
            yield self._stream_error_event(
                {
                    "ok": False,
                    "status": "empty_response",
                    "error": "Codex CLI returned no assistant message",
                    "cli_capabilities": capabilities.to_public_dict(),
                    "limitations": capabilities.limitations(),
                },
                started=started,
                model=model,
            )
            return
        yield {
            "type": "done",
            "message": full_response,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "duration_ms": round((time.time() - started) * 1000),
        }

    @classmethod
    def _events_from_json_line(cls, line: str) -> list[dict[str, Any]]:
        stripped = line.strip()
        if stripped.startswith("data:"):
            stripped = stripped[5:].strip()
        if not stripped or stripped == "[DONE]" or not stripped.startswith("{"):
            return []
        try:
            data = json.loads(stripped)
        except Exception:
            return []

        events: list[dict[str, Any]] = []
        metrics = cls._extract_metrics(data)
        if metrics:
            events.append({"type": "metrics", "data": metrics})
        delta = cls._extract_delta(data)
        if delta:
            events.append({"type": "delta", "delta": _sanitize_text(delta, limit=2000, strip=False)})
        return events

    @classmethod
    def _is_lifecycle_event(cls, value: Any) -> bool:
        if isinstance(value, list):
            return bool(value) and all(cls._is_lifecycle_event(item) for item in value)
        if not isinstance(value, dict):
            return _looks_like_lifecycle_event_name(value)
        for key in ("type", "event"):
            event_name = value.get(key)
            if isinstance(event_name, str) and _looks_like_lifecycle_event_name(event_name):
                content = cls._extract_delta(value)
                metrics = cls._extract_metrics(value)
                return not content and not metrics
        return False

    @classmethod
    def _is_reasoning_event(cls, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        if value.get("thinking") is True:
            return True
        for key in ("type", "event", "kind", "channel", "name"):
            text = str(value.get(key) or "").strip().lower()
            if "reasoning" in text or "thinking" in text:
                return True
        return False

    @classmethod
    def _extract_delta(cls, value: Any, depth: int = 0, *, in_content: bool = False) -> str:
        if depth > 6:
            return ""
        if isinstance(value, str):
            if in_content and cls._is_assistant_text(value):
                return value
            return ""
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                found = cls._extract_delta(item, depth + 1, in_content=in_content)
                if found:
                    parts.append(found)
            return "".join(parts)
        if not isinstance(value, dict):
            return ""
        if cls._is_reasoning_event(value):
            return ""

        event_type = str(value.get("type") or "").strip().lower()
        if event_type in {"text", "output_text"}:
            for key in ("text", "output_text", "content"):
                text = value.get(key)
                if isinstance(text, str) and cls._is_assistant_text(text):
                    return text

        for key in ("delta", "output_text", "text"):
            text = value.get(key)
            if isinstance(text, str) and cls._is_assistant_text(text):
                return text

        content = value.get("content")
        if isinstance(content, str) and cls._is_assistant_text(content):
            return content
        if isinstance(content, (dict, list)):
            found = cls._extract_delta(content, depth + 1, in_content=True)
            if found:
                return found

        for key in _STREAM_CONTAINER_KEYS:
            nested = value.get(key)
            if isinstance(nested, (dict, list)):
                found = cls._extract_delta(nested, depth + 1)
                if found:
                    return found
        for key, nested in value.items():
            if key in {"type", "event", "status"} or not isinstance(nested, (dict, list)):
                continue
            found = cls._extract_delta(nested, depth + 1)
            if found:
                return found
        return ""

    @staticmethod
    def _is_assistant_text(value: str) -> bool:
        return bool(value.strip()) and not _looks_like_lifecycle_event_name(value)

    @classmethod
    def _extract_metrics(cls, value: Any, depth: int = 0) -> dict[str, Any]:
        if depth > 4:
            return {}
        if isinstance(value, list):
            for item in value:
                found = cls._extract_metrics(item, depth + 1)
                if found:
                    return found
            return {}
        if not isinstance(value, dict):
            return {}
        event_type = str(value.get("type") or value.get("event") or "").lower()
        for key in _STREAM_METRIC_KEYS:
            metrics = value.get(key)
            if isinstance(metrics, dict):
                return cls._sanitize_metrics(metrics)
        for key in _STREAM_NESTED_KEYS:
            found = cls._extract_metrics(value.get(key), depth + 1)
            if found:
                return found
        if event_type in {"metrics", "metric", "usage"}:
            return cls._sanitize_metrics({
                key: metric_value
                for key, metric_value in value.items()
                if key not in {"type", "event"}
            })
        return {}

    @staticmethod
    def _sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe[str(key)] = _sanitize_text(str(value), limit=200) if isinstance(value, str) else value
        return safe

    @staticmethod
    def _stream_error_event(result: dict[str, Any], *, started: float, model: str | None) -> dict[str, Any]:
        return {
            "type": "error",
            "status": result.get("status") or "error",
            "error": _sanitize_text(result.get("error") or "Codex CLI streaming failed", limit=500),
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "duration_ms": round((time.time() - started) * 1000),
            "cli_capabilities": result.get("cli_capabilities") or {},
            "limitations": result.get("limitations") or list(_BASE_LIMITATIONS),
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
            "You are replying through Odysseus' Codex CLI provider.",
            "Return only the final assistant response.",
            "Do not run tools, shell commands, file edits, or web requests.",
            "If a request requires tools, say that this provider does not support tools yet.",
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
        parsed_json = False
        non_json_lines: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                if line:
                    non_json_lines.append(line)
                continue
            try:
                data = json.loads(line)
            except Exception:
                non_json_lines.append(line)
                continue
            parsed_json = True
            if CodexCliChatAdapter._is_lifecycle_event(data):
                continue
            delta = CodexCliChatAdapter._extract_delta(data)
            if delta:
                return _sanitize_text(delta)
            if CodexCliChatAdapter._extract_metrics(data):
                continue
            for key in ("message", "content", "text", "output"):
                value = data.get(key) if isinstance(data, dict) else None
                if (
                    isinstance(value, str)
                    and value.strip()
                    and not _looks_like_lifecycle_event_name(value)
                ):
                    return _sanitize_text(value)
        if parsed_json:
            return _sanitize_text("\n".join(non_json_lines))
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
            "model_discovery": {"source": "disabled"},
            "recommended_models": codex_recommended_models(),
            "connector_enabled": False,
            "selected_models": [],
            "manual_models": [],
            "hidden_models": [],
            "disabled_models": [],
            "thinking_supported": False,
            "thinking_effort_levels": [],
            "thinking_activity_supported": False,
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

        cfg = load_codex_model_config()
        base["connector_enabled"] = cfg["connector_enabled"]
        base["selected_models"] = [dict(item) for item in cfg["selected_models"]]
        base["manual_models"] = cfg["manual_models"]
        base["hidden_models"] = cfg["hidden_models"]
        base["disabled_models"] = cfg["disabled_models"]
        models: list[dict[str, Any]] = []
        chat_available = {"ok": False}
        if status == "available":
            chat_available = await self._chat_adapter.available()
            if not chat_available.get("ok"):
                status = chat_available.get("status") or "unsupported_unsafe_cli_mode"
            else:
                base["chat_supported"] = True
                base["streaming_supported"] = bool(chat_available.get("streaming_supported"))
                base["limitations"] = chat_available.get("limitations") or base["limitations"]
                base["cli_capabilities"] = chat_available.get("cli_capabilities") or {}
                base["model_discovery"] = chat_available.get("model_discovery") or {"source": "manual"}
                cap = base["cli_capabilities"]
                base["thinking_supported"] = bool(cap.get("reasoning_effort_supported"))
                base["thinking_effort_levels"] = cap.get("reasoning_effort_levels") or []
                base["thinking_activity_supported"] = bool(chat_available.get("streaming_supported"))

        for item in cfg["selected_models"]:
            models.append({
                "id": item["id"],
                "display": item.get("label") or _model_display(item["id"]),
                "label": item.get("label") or _model_display(item["id"]),
                "description": item.get("description") or "",
                "source": item.get("source") or "custom",
                "experimental": True,
                "hidden": bool(item.get("hidden")),
                "enabled": bool(item.get("enabled", True)) and not bool(item.get("hidden")),
                "thinking_effort": item.get("thinking_effort"),
                "streaming_supported": bool(base["streaming_supported"]),
                "thinking_supported": bool(base["thinking_supported"]),
                "thinking_effort_levels": list(base["thinking_effort_levels"]),
                "thinking_activity_supported": bool(base["thinking_activity_supported"]),
                "session_resume_supported": False,
                "tool_calling_supported": False,
                "agent_tools_supported": False,
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
        reasoning_effort: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
        odysseus_session_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._chat_adapter.complete(
            messages,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            odysseus_session_id=odysseus_session_id,
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: int = CODEX_CHAT_TIMEOUT_SECONDS,
        odysseus_session_id: str | None = None,
        allow_one_shot_fallback: bool = False,
    ):
        async for event in self._chat_adapter.stream_chat(
            messages,
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            odysseus_session_id=odysseus_session_id,
            allow_one_shot_fallback=allow_one_shot_fallback,
        ):
            yield event


def codex_model_list_item_if_available(provider: CodexModelProvider | None = None) -> dict[str, Any] | None:
    """Return the picker item only when Codex has at least one enabled model."""
    if not codex_model_provider_enabled():
        return None
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        running_loop = False
    else:
        running_loop = True
    if running_loop:
        return None
    try:
        status = asyncio.run((provider or CodexModelProvider()).status())
    except Exception:
        return None
    if status.get("status") == "available" and status.get("chat_supported") is True:
        item = codex_model_list_item(status.get("models") or [])
        if item.get("models"):
            return item
    return None


def _is_trust_directory_error(text: str | None) -> bool:
    detail = text or ""
    return any(pattern.search(detail) for pattern in _TRUST_DIRECTORY_PATTERNS)
