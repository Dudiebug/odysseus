"""Codex provider capability and configuration backend.
Admin-only status/config only; chat dispatch, picker exposure, streaming,
and tool support are deliberately out of scope for this split.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.codex_auth import get_codex_auth_service
from src.settings import load_settings, save_settings


CODEX_MODEL_PROVIDER_FLAG = "ODYSSEUS_CODEX_MODEL_PROVIDER_ENABLED"
CODEX_EXPERIMENTAL_MODEL_ID = "codex-cli/chatgpt-experimental"
CODEX_EXPERIMENTAL_ENDPOINT_URL = "odysseus://codex-cli"
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
_APPROVAL_FLAGS = ("--ask-for-approval", "--approval-policy", "--approval")
_DANGEROUS_FLAGS = ("--dangerously-bypass-approvals-and-sandbox", "--yolo")
_SKIP_GIT_REPO_CHECK_FLAG = "--skip-git-repo-check"
_JSON_OUTPUT_FLAG = "--json"
_REASONING_FLAGS = ("--reasoning-effort", "--effort", "--thinking-level")
_REASONING_LEVELS = ("low", "medium", "high", "maximum")
_TRUST_DIRECTORY_PATTERNS = (
    re.compile(r"not\s+(inside\s+)?(a\s+)?trusted\s+(directory|git\s+repository)", re.I),
    re.compile(r"trust(ed)?\s+directory", re.I),
    re.compile(r"--skip-git-repo-check", re.I),
)
_BASE_LIMITATIONS = [
    "Admin-only provider status/config backend.",
    "Feature flag defaults to disabled.",
    "Chat routing, model picker exposure, session resume, and tool execution are not wired in this branch.",
    "Codex CLI must advertise the long --sandbox flag so Odysseus can require read-only sandbox mode.",
]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def codex_model_provider_enabled() -> bool:
    return _truthy(os.getenv(CODEX_MODEL_PROVIDER_FLAG, "false"))


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


def _model_display(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1] or model_id


def _recommended_model_meta(model_id: str) -> dict[str, Any]:
    item = _CODEX_RECOMMENDED_BY_ID.get(model_id) or {}
    return {
        "id": model_id,
        "label": item.get("label") or _model_display(model_id),
        "description": item.get("description") or "",
        "source": "recommended" if item else "custom",
        "default": bool(item.get("default")),
    }


def _selected_model_entry(
    model_id: str,
    *,
    enabled: bool = True,
    hidden: bool = False,
    thinking_effort: str | None = None,
    source: str | None = None,
    label: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    meta = _recommended_model_meta(model_id)
    return {
        "id": model_id,
        "label": label or meta["label"],
        "description": description or meta["description"],
        "source": source or meta["source"],
        "enabled": enabled,
        "hidden": hidden,
        "thinking_effort": thinking_effort,
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
    if source in {"recommended", "codex_cli"} and model_id in _CODEX_RECOMMENDED_BY_ID:
        source = "recommended"
    elif source != "recommended":
        source = "custom"
    return _selected_model_entry(
        model_id,
        enabled=bool(value.get("enabled", True)),
        hidden=bool(value.get("hidden", False)),
        thinking_effort=_sanitize_thinking_effort(value.get("thinking_effort")),
        source=source,
        label=_sanitize_optional_text(value.get("label")) or recommended["label"],
        description=_sanitize_optional_text(value.get("description")) or recommended["description"],
    )


def _normalize_selected_models(values: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for value in values:
        item = _sanitize_selected_model_entry(value)
        if not item or item["id"] in seen:
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
        out.append(
            _selected_model_entry(
                model_id,
                enabled=model_id not in disabled,
                hidden=model_id in hidden,
            )
        )
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
    projections = _config_projections(selected_models)
    return {
        "connector_enabled": connector_enabled,
        "selected_models": selected_models,
        **projections,
    }


def save_codex_model_config(config: dict[str, Any]) -> dict[str, Any]:
    selected_models = _normalize_selected_models(config.get("selected_models"))
    projections = _config_projections(selected_models)
    clean = {
        "connector_enabled": bool(config.get("connector_enabled")) or bool(selected_models),
        "selected_models": selected_models,
        **projections,
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
            selected.append(_selected_model_entry(model_id))
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


def _sanitize_text(text: str | None, limit: int = 500) -> str:
    safe = text or ""
    for pattern in _TOKEN_PATTERNS:
        safe = pattern.sub("<redacted-token>", safe)
    return safe.strip()[:limit]


def _is_trust_directory_error(text: str | None) -> bool:
    detail = text or ""
    return any(pattern.search(detail) for pattern in _TRUST_DIRECTORY_PATTERNS)


@dataclass(frozen=True)
class CodexCliCapabilities:
    sandbox_flag: str | None
    sandbox_modes: tuple[str, ...] = ()
    approval_flag: str | None = None
    skip_git_repo_check_flag: str | None = None
    supports_json: bool = False
    supports_model: bool = False
    model_flag: str | None = None
    reasoning_flag: str | None = None
    reasoning_levels: tuple[str, ...] = ()
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
            out.append("Codex CLI advertises JSON output for future streaming integration.")
        else:
            out.append("This Codex CLI does not advertise JSON output for future streaming integration.")
        if self.reasoning_flag and self.reasoning_levels:
            out.append("Thinking controls are probeable because the CLI advertises explicit reasoning levels.")
        else:
            out.append("Thinking controls remain hidden until the CLI advertises explicit reasoning levels.")
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
            "json_output_supported": self.supports_json,
            "streaming_supported": self.streaming_supported,
            "model_flag_supported": self.supports_model,
            "model_flag": self.model_flag,
            "reasoning_effort_supported": bool(self.reasoning_flag and self.reasoning_levels),
            "reasoning_effort_flag": self.reasoning_flag,
            "reasoning_effort_levels": list(self.reasoning_levels),
            "dangerous_flags_advertised": list(self.dangerous_flags),
        }


def _detect_sandbox_flag(help_text: str) -> str | None:
    return "--sandbox" if "--sandbox" in help_text else None


def _detect_skip_git_repo_check_flag(help_text: str) -> str | None:
    return _SKIP_GIT_REPO_CHECK_FLAG if _SKIP_GIT_REPO_CHECK_FLAG in help_text else None


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
    found: list[str] = []
    for line in (help_text or "").splitlines():
        if flag not in line:
            continue
        lowered = line.lower()
        for level in _REASONING_LEVELS:
            if re.search(rf"\b{re.escape(level)}\b", lowered) and level not in found:
                found.append(level)
    return tuple(found)


def _detect_cli_capabilities_from_help(exec_help: str, root_help: str = "") -> CodexCliCapabilities:
    combined = "\n".join([exec_help or "", root_help or ""])
    model_flag = _detect_model_flag(exec_help or "")
    reasoning_flag = _detect_reasoning_flag(exec_help or "")
    return CodexCliCapabilities(
        sandbox_flag=_detect_sandbox_flag(exec_help or ""),
        sandbox_modes=tuple(
            mode
            for mode in ("read-only", "workspace-write", "danger-full-access")
            if mode in (exec_help or "")
        ),
        approval_flag=_detect_approval_flag(exec_help or ""),
        skip_git_repo_check_flag=_detect_skip_git_repo_check_flag(combined),
        supports_json=_JSON_OUTPUT_FLAG in (exec_help or ""),
        supports_model=bool(model_flag),
        model_flag=model_flag,
        reasoning_flag=reasoning_flag,
        reasoning_levels=_detect_reasoning_levels(exec_help or "", reasoning_flag),
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
        model_id = _sanitize_model_id(value)
        return [model_id] if model_id else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_extract_model_ids_from_payload(item))
        return _unique_model_ids(out)
    if isinstance(value, dict):
        for key in ("id", "name", "model"):
            model_id = _sanitize_model_id(value.get(key))
            if model_id:
                return [model_id]
        out: list[str] = []
        for key in ("data", "models", "items"):
            out.extend(_extract_model_ids_from_payload(value.get(key)))
        return _unique_model_ids(out)
    return []


def _parse_model_list_output(output: str) -> list[str]:
    text = output or ""
    parsed: list[str] = []
    try:
        parsed = _extract_model_ids_from_payload(json.loads(text))
    except Exception:
        parsed = []
    if parsed:
        return _unique_model_ids(parsed)
    fallback: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith(("usage:", "error", "warning")):
            continue
        fallback.append(stripped.split()[0])
    return _unique_model_ids(fallback)


class CodexCliChatAdapter:
    """Capability probe for the installed Codex CLI."""

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
            "model_discovery": help_result.get("model_discovery") or {"source": "none"},
            "cli_capabilities": capabilities.to_public_dict(),
            "limitations": capabilities.limitations(),
            "_preflight": preflight,
            "_capabilities": capabilities,
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

        try:
            bin_path = service._bin_path()
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
                "detail": _sanitize_text(err or out),
            }
        exec_help = out or ""
        root_rc, root_out, _ = await self._run([bin_path, "--help"], timeout=20, env=env)
        root_help = root_out if root_rc == 0 else ""
        capabilities = _detect_cli_capabilities_from_help(exec_help, root_help)
        discovered_models: list[str] = []
        model_discovery: dict[str, Any] = {"source": "none", "commands_tried": []}
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


class CodexModelProvider:
    """Admin-only provider status/config wrapper around the Codex CLI."""

    def __init__(
        self,
        auth_service_getter: Callable[[], Any] | None = None,
        chat_adapter: CodexCliChatAdapter | None = None,
    ) -> None:
        self._auth_service_getter = auth_service_getter or get_codex_auth_service
        self._chat_adapter = chat_adapter or CodexCliChatAdapter(self._auth_service_getter)

    async def status(self) -> dict[str, Any]:
        enabled = codex_model_provider_enabled()
        cfg = load_codex_model_config()
        base = {
            "feature_enabled": enabled,
            "feature_flag": CODEX_MODEL_PROVIDER_FLAG,
            "provider": "codex_cli",
            "experimental": True,
            "connector_enabled": cfg["connector_enabled"],
            "selected_models": [dict(item) for item in cfg["selected_models"]],
            "manual_models": cfg["manual_models"],
            "hidden_models": cfg["hidden_models"],
            "disabled_models": cfg["disabled_models"],
            "recommended_models": codex_recommended_models(),
            "models": [],
            "chat_supported": False,
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "thinking_supported": False,
            "thinking_effort_levels": [],
            "thinking_activity_supported": False,
            "model_discovery": {"source": "disabled" if not enabled else "none"},
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
        base["auth"] = {"status": auth_status, "auth_mode": auth_mode}

        if not cli_available:
            return {
                **base,
                "status": "cli_unavailable",
                "cli_available": False,
                "authenticated": authenticated,
                "requires_sign_in": False,
            }
        if not authenticated:
            return {
                **base,
                "status": "sign_in_required",
                "cli_available": True,
                "authenticated": False,
                "requires_sign_in": True,
                "sign_in_route": "/api/codex-auth/start",
            }

        chat_available = await self._chat_adapter.available()
        if not chat_available.get("ok"):
            return {
                **base,
                "status": chat_available.get("status") or "unsupported_unsafe_cli_mode",
                "cli_available": True,
                "authenticated": True,
                "requires_sign_in": False,
                "cli_capabilities": chat_available.get("cli_capabilities") or {},
                "limitations": chat_available.get("limitations") or list(_BASE_LIMITATIONS),
            }

        models = []
        for item in cfg["selected_models"]:
            models.append(
                {
                    "id": item["id"],
                    "display": item.get("label") or _model_display(item["id"]),
                    "label": item.get("label") or _model_display(item["id"]),
                    "description": item.get("description") or "",
                    "source": item.get("source") or "custom",
                    "experimental": True,
                    "hidden": bool(item.get("hidden")),
                    "enabled": bool(item.get("enabled", True)) and not bool(item.get("hidden")),
                    "thinking_effort": item.get("thinking_effort"),
                    "streaming_supported": bool(chat_available.get("streaming_supported")),
                    "thinking_supported": bool(
                        chat_available.get("cli_capabilities", {}).get("reasoning_effort_supported")
                    ),
                    "thinking_effort_levels": list(
                        chat_available.get("cli_capabilities", {}).get("reasoning_effort_levels") or []
                    ),
                    "thinking_activity_supported": False,
                    "session_resume_supported": False,
                    "tool_calling_supported": False,
                    "agent_tools_supported": False,
                }
            )

        return {
            **base,
            "status": "available",
            "cli_available": True,
            "authenticated": True,
            "requires_sign_in": False,
            "sign_in_route": "/api/codex-auth/start",
            "models": models,
            "chat_supported": True,
            "streaming_supported": bool(chat_available.get("streaming_supported")),
            "model_discovery": chat_available.get("model_discovery") or {"source": "none"},
            "limitations": chat_available.get("limitations") or list(_BASE_LIMITATIONS),
            "cli_capabilities": chat_available.get("cli_capabilities") or {},
            "thinking_supported": bool(
                chat_available.get("cli_capabilities", {}).get("reasoning_effort_supported")
            ),
            "thinking_effort_levels": list(
                chat_available.get("cli_capabilities", {}).get("reasoning_effort_levels") or []
            ),
            "thinking_activity_supported": False,
        }
