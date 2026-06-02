import asyncio
import os
import sys
import types
from types import SimpleNamespace

import pytest

from routes.codex_model_provider_routes import setup_codex_model_provider_routes
from src.codex_model_provider import (
    CODEX_MODEL_PROVIDER_FLAG,
    CodexCliChatAdapter,
    CodexModelProvider,
    codex_recommended_models,
    load_codex_model_config,
    update_codex_model_config,
)


if "core" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")]
    sys.modules["core"] = core_pkg


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _isolated_codex_settings(monkeypatch, tmp_path):
    import src.settings as settings_module

    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", str(settings_file))
    settings_module._invalidate_caches()
    yield
    settings_module._invalidate_caches()


class _FakeService:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def status(self):
        self.calls += 1
        return dict(self.payload)

    def _bin_path(self):
        return "/usr/bin/codex"

    def _env(self):
        return {"PATH": "/usr/bin"}


class _FakeAvailableAdapter:
    async def available(self):
        return {
            "ok": True,
            "status": "available",
            "chat_supported": True,
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "supports_model": True,
            "discovered_models": ["gpt-5.2-codex"],
            "model_discovery": {"source": "codex_cli", "command": "models"},
            "cli_capabilities": {
                "model_flag_supported": True,
                "reasoning_effort_supported": False,
                "reasoning_effort_levels": [],
            },
            "limitations": [],
        }


def _provider(payload):
    svc = _FakeService(payload)
    return CodexModelProvider(lambda: svc, chat_adapter=_FakeAvailableAdapter()), svc


CODEX_EXEC_HELP = """Usage: codex exec [OPTIONS] [PROMPT]

Options:
  -s, --sandbox <SANDBOX_MODE>
  --json

Commands:
  resume
"""

CODEX_ROOT_HELP = """Usage: codex [OPTIONS] <COMMAND>

Commands:
  exec
"""

CODEX_THINKING_TOGGLE_HELP = """Usage: codex exec [OPTIONS] [PROMPT]

Options:
  -s, --sandbox <SANDBOX_MODE>
  --json
  --thinking
"""


async def _codex_help_runner(args, timeout, cwd=None, env=None, exec_help=CODEX_EXEC_HELP):
    if args[1:] == ["exec", "--help"]:
        return 0, exec_help, ""
    if args[1:] == ["--help"]:
        return 0, CODEX_ROOT_HELP, ""
    return 0, "", ""


class _AuthManager:
    is_configured = True

    def is_admin(self, user):
        return user == "admin"


def _request(user="admin"):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        headers={},
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=_AuthManager())),
    )


def _endpoint(router, path, method):
    for route in router.routes:
        if getattr(route, "path", "") == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {path}")


def test_codex_model_provider_hidden_when_flag_disabled(monkeypatch):
    monkeypatch.delenv(CODEX_MODEL_PROVIDER_FLAG, raising=False)
    provider, svc = _provider({"codex_cli_available": True, "authenticated": True})

    out = run(provider.status())

    assert out["feature_enabled"] is False
    assert out["status"] == "disabled"
    assert out["models"] == []
    assert svc.calls == 0


def test_codex_manual_model_config_persists_hide_disable_restore(monkeypatch, tmp_path):
    import src.settings as settings_module

    settings_file = tmp_path / "settings.json"
    settings_file.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings_module, "SETTINGS_FILE", str(settings_file))
    settings_module._invalidate_caches()

    cfg = update_codex_model_config(add_model="gpt-5.2-codex")
    assert cfg["manual_models"] == ["gpt-5.2-codex"]
    assert load_codex_model_config()["manual_models"] == ["gpt-5.2-codex"]

    cfg = update_codex_model_config(disable_model="gpt-5.2-codex")
    assert cfg["disabled_models"] == ["gpt-5.2-codex"]

    cfg = update_codex_model_config(hide_model="gpt-5.2-codex")
    assert cfg["hidden_models"] == ["gpt-5.2-codex"]
    assert cfg["disabled_models"] == []

    cfg = update_codex_model_config(restore_model="gpt-5.2-codex")
    assert cfg["hidden_models"] == []
    settings_module._invalidate_caches()


def test_codex_model_config_remove_and_clear_connector():
    update_codex_model_config(add_model="gpt-5.5")
    update_codex_model_config(add_model="gpt-5.4")

    cfg = update_codex_model_config(remove_model="gpt-5.4")
    assert cfg["manual_models"] == ["gpt-5.5"]

    cfg = update_codex_model_config(clear_all_models=True, connector_enabled=False)
    assert cfg["connector_enabled"] is False
    assert cfg["manual_models"] == []
    assert cfg["selected_models"] == []


def test_codex_connector_can_exist_without_selected_models():
    cfg = update_codex_model_config(connector_enabled=True)

    assert cfg["connector_enabled"] is True
    assert cfg["selected_models"] == []
    assert cfg["manual_models"] == []


def test_codex_recommended_presets_default_to_gpt_5_5_and_dedupe_selection():
    presets = codex_recommended_models()

    assert presets[0]["id"] == "gpt-5.5"
    assert presets[0]["label"] == "GPT-5.5"

    update_codex_model_config(add_model="gpt-5.5")
    cfg = update_codex_model_config(add_model="gpt-5.5")

    assert [item["id"] for item in cfg["selected_models"]] == ["gpt-5.5"]


def test_codex_model_provider_requires_sign_in_when_unauthenticated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({
        "codex_cli_available": True,
        "authenticated": False,
        "status": "not_authenticated",
    })

    out = run(provider.status())

    assert out["status"] == "sign_in_required"
    assert out["requires_sign_in"] is True
    assert out["models"] == []
    assert out["chat_supported"] is False
    assert out["streaming_supported"] is False


def test_codex_model_provider_reports_experimental_model_when_authenticated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    update_codex_model_config(add_model="gpt-5.2-codex")
    provider, _ = _provider({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
        "auth_mode": "ChatGPT",
        "access_token": "secret",
        "refresh_token": "secret",
        "resolved_binary_path": "/private/bin/codex",
        "codex_home": "/private/.codex",
    })

    out = run(provider.status())

    assert out["status"] == "available"
    assert out["authenticated"] is True
    assert out["models"][0]["id"] == "gpt-5.2-codex"
    assert out["models"][0]["display"] == "gpt-5.2-codex"
    assert out["models"][0]["experimental"] is True
    assert out["chat_supported"] is True
    assert out["streaming_supported"] is False
    assert out["session_resume_supported"] is False
    assert out["tool_execution_allowed"] is False
    assert out["auth"] == {"status": "authenticated", "auth_mode": "ChatGPT"}
    assert "secret" not in str(out)
    assert "access_token" not in str(out)
    assert "refresh_token" not in str(out)
    assert "/private/.codex" not in str(out)
    assert "/private/bin/codex" not in str(out)


def test_codex_model_provider_reports_cli_unavailable(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({
        "codex_cli_available": False,
        "cli_found": False,
        "cli_executable": False,
        "status": "cli_missing",
    })

    out = run(provider.status())

    assert out["status"] == "cli_unavailable"
    assert out["cli_available"] is False
    assert out["models"] == []


def test_codex_model_provider_route_is_admin_gated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({"codex_cli_available": True, "authenticated": True})
    router = setup_codex_model_provider_routes(provider)
    status = _endpoint(router, "/api/codex-model-provider/status", "GET")

    out = run(status(_request(user="admin")))
    assert out["status"] == "available"

    try:
        run(status(_request(user="bob")))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("non-admin request should fail")


def test_codex_model_provider_connector_route_persists_empty_connector(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({"codex_cli_available": True, "authenticated": True})
    router = setup_codex_model_provider_routes(provider)
    add_connector = _endpoint(router, "/api/codex-model-provider/connector", "POST")

    out = run(add_connector(_request(user="admin")))

    assert out["ok"] is True
    assert out["config"]["connector_enabled"] is True
    assert out["config"]["selected_models"] == []


def test_status_available_with_current_cli_help(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    adapter = CodexCliChatAdapter(lambda: svc, runner=_codex_help_runner)
    provider = CodexModelProvider(lambda: svc, chat_adapter=adapter)

    out = run(provider.status())

    assert out["status"] == "available"
    assert out["chat_supported"] is True
    assert out["streaming_supported"] is True
    assert out["models"] == []
    assert out["model_discovery"]["source"] == "none"
    assert out["cli_capabilities"]["sandbox_supported"] is True
    assert out["cli_capabilities"]["approval_control_supported"] is False
    assert out["cli_capabilities"]["json_output_supported"] is True
    assert out["cli_capabilities"]["streaming_supported"] is True
    assert out["cli_capabilities"]["skip_git_repo_check_supported"] is False
    assert out["cli_capabilities"]["sandbox_mode"] == "read-only"
    assert out["thinking_supported"] is False
    assert out["thinking_effort_levels"] == []
    assert out["thinking_activity_supported"] is False


def test_status_does_not_advertise_thinking_from_toggle_only_help(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_THINKING_TOGGLE_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_ROOT_HELP, ""
        return 0, "", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    provider = CodexModelProvider(lambda: svc, chat_adapter=adapter)

    out = run(provider.status())

    assert out["status"] == "available"
    assert out["cli_capabilities"]["reasoning_effort_supported"] is False
    assert out["cli_capabilities"]["reasoning_effort_levels"] == []
    assert out["thinking_supported"] is False
    assert out["thinking_effort_levels"] == []


def test_adapter_requires_long_sandbox_flag(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, "Usage: codex exec -s <MODE> --json", ""
        if args[1:] == ["--help"]:
            return 0, CODEX_ROOT_HELP, ""
        return 0, "", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.available())

    assert out["ok"] is False
    assert out["status"] == "unsupported_unsafe_cli_mode"
    assert out["cli_capabilities"]["sandbox_supported"] is False


def test_status_does_not_expose_model_when_adapter_is_unsafe(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    update_codex_model_config(add_model="gpt-5.2-codex")

    class UnsafeAdapter:
        async def available(self):
            return {"ok": False, "status": "unsupported_unsafe_cli_mode"}

    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    provider = CodexModelProvider(lambda: svc, chat_adapter=UnsafeAdapter())

    out = run(provider.status())

    assert out["status"] == "unsupported_unsafe_cli_mode"
    assert out["models"] == []
