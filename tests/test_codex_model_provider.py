import asyncio
import json
import os
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.codex_model_provider import (
    CODEX_EXPERIMENTAL_ENDPOINT_URL,
    CODEX_EXPERIMENTAL_MODEL_ID,
    CODEX_MODEL_PROVIDER_FLAG,
    CODEX_EXPERIMENTAL_MODEL_DISPLAY,
    CodexCliChatAdapter,
    CodexModelProvider,
    codex_recommended_models,
    load_codex_model_config,
    update_codex_model_config,
    codex_model_list_item_if_available,
    codex_model_list_item,
    first_enabled_codex_model,
    is_codex_model_selection,
)
from src.llm_core import llm_call_async
from routes.codex_model_provider_routes import setup_codex_model_provider_routes


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _fake_request(user="alice"):
    return SimpleNamespace(
        state=SimpleNamespace(current_user=user),
        headers={},
        app=SimpleNamespace(state=SimpleNamespace(auth_manager=None)),
        client=SimpleNamespace(host="127.0.0.1"),
    )


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


class _FakeAdapter:
    async def available(self):
        return {
            "ok": True,
            "status": "available",
            "chat_supported": True,
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
            "limitations": [],
            "supports_model": True,
            "discovered_models": ["gpt-5.2-codex"],
            "model_discovery": {"source": "codex_cli", "command": "models"},
            "cli_capabilities": {
                "model_flag_supported": True,
                "reasoning_effort_supported": False,
                "reasoning_effort_levels": [],
            },
        }

    async def complete(self, messages, model=None, reasoning_effort=None, timeout_seconds=120, odysseus_session_id=None):
        return {
            "ok": True,
            "status": "ok",
            "message": "mock response",
            "duration_ms": 1,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "limitations": [],
            "streaming_supported": False,
            "session_resume_supported": False,
            "tool_execution_allowed": False,
        }

    async def stream_chat(
        self,
        messages,
        model=None,
        reasoning_effort=None,
        timeout_seconds=120,
        odysseus_session_id=None,
        allow_one_shot_fallback=False,
    ):
        yield {"type": "delta", "delta": "mock stream"}
        yield {"type": "done", "message": "mock stream", "model": model or CODEX_EXPERIMENTAL_MODEL_ID}


def _provider(payload):
    svc = _FakeService(payload)
    return CodexModelProvider(lambda: svc, chat_adapter=_FakeAdapter()), svc


CODEX_0135_EXEC_HELP = """Usage: codex exec [OPTIONS] [PROMPT]

Options:
  -s, --sandbox <SANDBOX_MODE>
  --json

Commands:
  resume
"""

CODEX_0135_EXEC_HELP_WITH_SKIP = CODEX_0135_EXEC_HELP.replace(
    "  --json\n",
    "  --json\n  --skip-git-repo-check\n",
)

CODEX_0135_ROOT_HELP = """Usage: codex [OPTIONS] <COMMAND>

Commands:
  exec
"""

CODEX_0135_RESUME_HELP = """Usage: codex exec resume [OPTIONS] [SESSION_ID]

Options:
  --last
"""

CODEX_OLD_EXEC_HELP = "Usage: codex exec --sandbox <MODE> --ask-for-approval <POLICY> --json --model <MODEL>"


async def _codex_help_runner(args, timeout, cwd=None, env=None, exec_help=CODEX_0135_EXEC_HELP):
    if args[1:] == ["exec", "--help"]:
        return 0, exec_help, ""
    if args[1:] == ["--help"]:
        return 0, CODEX_0135_ROOT_HELP, ""
    if args[1:] == ["exec", "resume", "--help"]:
        return 0, CODEX_0135_RESUME_HELP, ""
    return 0, "codex provider test ok", ""


async def _collect(agen):
    out = []
    async for event in agen:
        out.append(event)
    return out


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    async def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeStderr:
    def __init__(self, payload=b""):
        self._payload = payload

    async def read(self):
        return self._payload


class _FakeProcess:
    def __init__(self, stdout_lines, stderr=b"", returncode=0):
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = _FakeStderr(stderr)
        self.returncode = returncode
        self.killed = False

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


def _stream_events_from_lines(monkeypatch, lines):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })

    async def runner(args, timeout, cwd=None, env=None):
        return await _codex_help_runner(
            args,
            timeout,
            cwd=cwd,
            env=env,
            exec_help=CODEX_0135_EXEC_HELP_WITH_SKIP,
        )

    async def fake_create_subprocess_exec(*args, stdout=None, stderr=None, cwd=None, env=None):
        return _FakeProcess([line if isinstance(line, bytes) else line.encode("utf-8") for line in lines])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    return run(_collect(adapter.stream_chat([{"role": "user", "content": "Say ok"}], timeout_seconds=5)))


def test_codex_model_provider_hidden_when_flag_disabled(monkeypatch):
    monkeypatch.delenv(CODEX_MODEL_PROVIDER_FLAG, raising=False)
    provider, svc = _provider({"codex_cli_available": True, "authenticated": True})

    out = run(provider.status())

    assert out["feature_enabled"] is False
    assert out["status"] == "disabled"
    assert out["models"] == []
    assert svc.calls == 0


def test_codex_model_list_item_is_synthetic_not_db_endpoint():
    item = codex_model_list_item([{
        "id": "gpt-5.2-codex",
        "display": "gpt-5.2-codex",
        "enabled": True,
        "streaming_supported": True,
    }])

    assert item["url"] == CODEX_EXPERIMENTAL_ENDPOINT_URL
    assert item["models"] == ["gpt-5.2-codex"]
    assert item["endpoint_id"] is None
    assert item["experimental"] is True
    assert is_codex_model_selection(item["url"], item["models"][0]) is True
    assert is_codex_model_selection(None, item["models"][0]) is False
    assert is_codex_model_selection("https://api.openai.com/v1/chat/completions", item["models"][0]) is False


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
    assert first_enabled_codex_model(cfg) == "gpt-5.5"

    cfg = update_codex_model_config(clear_all_models=True, connector_enabled=False)
    assert cfg["connector_enabled"] is False
    assert cfg["manual_models"] == []
    assert cfg["selected_models"] == []


def test_codex_recommended_presets_default_to_gpt_5_5_and_dedupe_selection():
    presets = codex_recommended_models()

    assert presets[0]["id"] == "gpt-5.5"
    assert presets[0]["label"] == "GPT-5.5"

    update_codex_model_config(add_model="gpt-5.5")
    cfg = update_codex_model_config(add_model="gpt-5.5")

    assert [item["id"] for item in cfg["selected_models"]] == ["gpt-5.5"]


def test_codex_model_list_item_if_available_requires_available_status(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    update_codex_model_config(add_model="gpt-5.2-codex")

    available_provider, _ = _provider({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    signed_out_provider, _ = _provider({
        "codex_cli_available": True,
        "authenticated": False,
        "status": "not_authenticated",
    })

    assert codex_model_list_item_if_available(available_provider)["url"] == CODEX_EXPERIMENTAL_ENDPOINT_URL
    assert codex_model_list_item_if_available(signed_out_provider) is None

    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "false")
    assert codex_model_list_item_if_available(available_provider) is None


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
    assert "secret" not in str(out)
    assert "access_token" not in str(out)
    assert "refresh_token" not in str(out)


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


def test_codex_model_provider_test_chat_route_is_admin_gated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({"codex_cli_available": True, "authenticated": True})
    router = setup_codex_model_provider_routes(provider)
    test_chat = _endpoint(router, "/api/codex-model-provider/test-chat", "POST")

    body = SimpleNamespace(prompt="hello", messages=None, model=None, timeout_seconds=None)
    out = run(test_chat(_request(user="admin"), body))
    assert out["ok"] is True
    assert out["message"] == "mock response"

    try:
        run(test_chat(_request(user="bob"), body))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("non-admin request should fail")


def test_codex_model_provider_test_chat_stream_route_is_admin_gated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({"codex_cli_available": True, "authenticated": True})
    router = setup_codex_model_provider_routes(provider)
    test_chat_stream = _endpoint(router, "/api/codex-model-provider/test-chat-stream", "POST")

    body = SimpleNamespace(prompt="hello", messages=None, model=None, timeout_seconds=None)
    response = run(test_chat_stream(_request(user="admin"), body))
    chunks = run(_collect(response.body_iterator))
    payload = "".join(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk for chunk in chunks)
    assert '"type":"delta"' in payload or '"type": "delta"' in payload
    assert "mock stream" in payload

    try:
        run(test_chat_stream(_request(user="bob"), body))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("non-admin request should fail")


def test_codex_model_provider_test_chat_requires_body(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({"codex_cli_available": True, "authenticated": True})
    router = setup_codex_model_provider_routes(provider)
    test_chat = _endpoint(router, "/api/codex-model-provider/test-chat", "POST")

    body = SimpleNamespace(prompt="", messages=None, model=None, timeout_seconds=None)
    out = run(test_chat(_request(user="admin"), body))
    assert out["ok"] is False
    assert out["status"] == "invalid_request"


def test_adapter_success_from_mocked_subprocess(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append((args, cwd, env))
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_OLD_EXEC_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "codex provider test ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "Say ok"}]))

    assert out["ok"] is True
    assert out["message"] == "codex provider test ok"
    assert out["streaming_supported"] is True
    assert out["session_resume_supported"] is False
    assert out["tool_execution_allowed"] is False
    exec_args = calls[-1][0]
    assert calls[-1][1]
    assert "--sandbox" in exec_args
    assert "read-only" in exec_args
    assert "--ask-for-approval" not in exec_args
    assert "--approval-policy" not in exec_args
    assert "--approval" not in exec_args
    assert "never" not in exec_args


def test_adapter_accepts_current_cli_without_approval_flag(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        return await _codex_help_runner(args, timeout, cwd=cwd, env=env)

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete(
        [{"role": "user", "content": "Say ok"}],
        model=CODEX_EXPERIMENTAL_MODEL_ID,
        odysseus_session_id="test-session",
    ))

    assert out["ok"] is True
    assert out["message"] == "codex provider test ok"
    assert out["cli_capabilities"]["sandbox_supported"] is True
    assert out["cli_capabilities"]["approval_control_supported"] is False
    assert out["cli_capabilities"]["resume_supported"] is True
    assert out["cli_capabilities"]["resume_last_supported"] is True
    exec_args = calls[-1]
    assert exec_args[:4] == ["/usr/bin/codex", "exec", "--sandbox", "read-only"]
    assert "--ask-for-approval" not in exec_args
    assert "--approval-policy" not in exec_args
    assert "--approval" not in exec_args
    assert "--dangerously-bypass-approvals-and-sandbox" not in exec_args
    assert "--yolo" not in exec_args
    assert CODEX_EXPERIMENTAL_MODEL_ID not in exec_args
    assert any("Approval-control flag is not available" in item for item in out["limitations"])


def test_adapter_uses_skip_git_repo_check_when_advertised(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        return await _codex_help_runner(
            args,
            timeout,
            cwd=cwd,
            env=env,
            exec_help=CODEX_0135_EXEC_HELP_WITH_SKIP,
        )

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "Say ok"}]))

    assert out["ok"] is True
    assert out["cli_capabilities"]["skip_git_repo_check_supported"] is True
    assert out["cli_capabilities"]["skip_git_repo_check_flag"] == "--skip-git-repo-check"
    exec_args = calls[-1]
    assert exec_args[:4] == ["/usr/bin/codex", "exec", "--sandbox", "read-only"]
    assert "--skip-git-repo-check" in exec_args
    assert "--json" not in exec_args
    assert "--dangerously-bypass-approvals-and-sandbox" not in exec_args
    assert "--yolo" not in exec_args


def test_adapter_passes_custom_model_only_when_model_flag_supported(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_OLD_EXEC_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "codex provider test ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "Say ok"}], model="gpt-5.2-codex"))

    assert out["ok"] is True
    exec_args = calls[-1]
    assert "--model" in exec_args
    assert exec_args[exec_args.index("--model") + 1] == "gpt-5.2-codex"


def test_adapter_rejects_custom_model_without_model_flag(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_0135_EXEC_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "should not run", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "Say ok"}], model="gpt-5.2-codex"))

    assert out["ok"] is False
    assert out["status"] == "unsupported_option"
    assert not any(args[1] == "exec" and "--help" not in args for args in calls)


def test_adapter_retries_with_skip_git_repo_check_when_trust_error_requires_it(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_0135_EXEC_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        if "--skip-git-repo-check" not in args:
            return 2, "", "not inside a trusted directory; retry with --skip-git-repo-check"
        return 0, "codex provider test ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "Say ok"}]))

    assert out["ok"] is True
    exec_calls = [args for args in calls if args[1] == "exec" and "--help" not in args and args[2] != "resume"]
    assert "--skip-git-repo-check" not in exec_calls[0]
    assert "--skip-git-repo-check" in exec_calls[1]
    assert "--dangerously-bypass-approvals-and-sandbox" not in exec_calls[1]
    assert "--yolo" not in exec_calls[1]


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


def test_adapter_streams_json_events_with_safe_args(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })
    created = []

    async def runner(args, timeout, cwd=None, env=None):
        return await _codex_help_runner(
            args,
            timeout,
            cwd=cwd,
            env=env,
            exec_help=CODEX_0135_EXEC_HELP_WITH_SKIP,
        )

    async def fake_create_subprocess_exec(*args, stdout=None, stderr=None, cwd=None, env=None):
        created.append({"args": list(args), "cwd": cwd, "env": env})
        return _FakeProcess([
            b'{"type":"response.output_text.delta","delta":"hello "}\n',
            b'{"type":"delta","data":{"text":"world"}}\n',
            b'{"usage":{"input_tokens":1,"output_tokens":2}}\n',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    events = run(_collect(adapter.stream_chat([{"role": "user", "content": "Say ok"}], timeout_seconds=5)))

    assert [event["type"] for event in events] == ["delta", "delta", "metrics", "done"]
    assert events[0]["delta"] == "hello "
    assert events[1]["delta"] == "world"
    assert events[2]["data"]["input_tokens"] == 1
    assert events[-1]["message"] == "hello world"
    exec_args = created[0]["args"]
    assert exec_args[:4] == ["/usr/bin/codex", "exec", "--sandbox", "read-only"]
    assert "--skip-git-repo-check" in exec_args
    assert "--json" in exec_args
    assert "--dangerously-bypass-approvals-and-sandbox" not in exec_args
    assert "--yolo" not in exec_args


def test_adapter_stream_ignores_lifecycle_events(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })

    async def runner(args, timeout, cwd=None, env=None):
        return await _codex_help_runner(
            args,
            timeout,
            cwd=cwd,
            env=env,
            exec_help=CODEX_0135_EXEC_HELP_WITH_SKIP,
        )

    async def fake_create_subprocess_exec(*args, stdout=None, stderr=None, cwd=None, env=None):
        return _FakeProcess([
            b'{"type":"thread.started"}\n',
            b'{"event":"turn.started"}\n',
            b'{"type":"response.output_text.delta","delta":"clean "}\n',
            b'{"type":"turn.completed"}\n',
            b'{"event":"request.completed"}\n',
            b'{"type":"delta","data":{"text":"answer"}}\n',
            b'{"type":"metrics","usage":{"input_tokens":1,"output_tokens":2}}\n',
        ])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    events = run(_collect(adapter.stream_chat([{"role": "user", "content": "Say ok"}], timeout_seconds=5)))

    assert [event["type"] for event in events] == ["delta", "delta", "metrics", "done"]
    assert [event["delta"] for event in events if event["type"] == "delta"] == ["clean ", "answer"]
    assert events[-1]["message"] == "clean answer"
    assert "thread.started" not in str(events)
    assert "turn.started" not in str(events)
    assert "turn.completed" not in str(events)


def test_adapter_streams_prompt_sample_delta_and_metrics(monkeypatch):
    events = _stream_events_from_lines(monkeypatch, [
        '{"type":"thread.started"}\n',
        '{"type":"turn.started"}\n',
        '{"type":"message.delta","delta":"odysseus streaming ok"}\n',
        '{"type":"metrics","usage":{"input_tokens":10,"output_tokens":3}}\n',
        '{"type":"turn.completed"}\n',
    ])

    assert [event["type"] for event in events] == ["delta", "metrics", "done"]
    assert events[0]["delta"] == "odysseus streaming ok"
    assert events[1]["data"] == {"input_tokens": 10, "output_tokens": 3}
    assert events[-1]["message"] == "odysseus streaming ok"
    assert "thread.started" not in events[-1]["message"]
    assert "turn.started" not in events[-1]["message"]
    assert "turn.completed" not in events[-1]["message"]


def test_adapter_streams_event_data_delta_shape(monkeypatch):
    events = _stream_events_from_lines(monkeypatch, [
        '{"event":"response.output_text.delta","data":{"delta":"odysseus streaming ok"}}\n',
    ])

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == "odysseus streaming ok"
    assert events[-1]["message"] == "odysseus streaming ok"


def test_adapter_streams_final_only_message_completed_content(monkeypatch):
    events = _stream_events_from_lines(monkeypatch, [
        '{"type":"message.completed","message":{"content":[{"type":"text","text":"odysseus streaming ok"}]}}\n',
    ])

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == "odysseus streaming ok"
    assert events[-1]["message"] == "odysseus streaming ok"


def test_adapter_streams_nested_content_on_lifecycle_event(monkeypatch):
    events = _stream_events_from_lines(monkeypatch, [
        '{"type":"turn.completed","item":{"content":[{"type":"output_text","text":"odysseus streaming ok"}]}}\n',
    ])

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == "odysseus streaming ok"
    assert events[-1]["message"] == "odysseus streaming ok"
    assert "turn.completed" not in events[-1]["message"]


def test_adapter_stream_malformed_json_lines_do_not_crash(monkeypatch):
    events = _stream_events_from_lines(monkeypatch, [
        '{"type":"thread.started"}\n',
        '{"type":"message.delta","delta":\n',
        '{"type":"message.delta","delta":"odysseus streaming ok"}\n',
    ])

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == "odysseus streaming ok"
    assert events[-1]["message"] == "odysseus streaming ok"


def test_adapter_stream_lifecycle_only_returns_empty_response(monkeypatch):
    events = _stream_events_from_lines(monkeypatch, [
        '{"type":"thread.started"}\n',
        '{"event":"turn.started"}\n',
        '{"type":"turn.completed"}\n',
    ])

    assert [event["type"] for event in events] == ["error"]
    assert events[0]["status"] == "empty_response"
    assert "thread.started" not in str(events[0])
    assert "turn.started" not in str(events[0])
    assert "turn.completed" not in str(events[0])


def test_stream_parser_ignores_lifecycle_values_in_text_like_fields():
    assert CodexCliChatAdapter._events_from_json_line(
        '{"type":"message.delta","message":"thread.started","text":"turn.started","delta":"turn.completed"}'
    ) == []


def test_stream_parser_drops_reasoning_deltas_but_keeps_metrics():
    events = CodexCliChatAdapter._events_from_json_line(
        '{"type":"reasoning.delta","delta":"secret","usage":{"reasoning_output_tokens":41}}'
    )

    assert events == [{"type": "metrics", "data": {"reasoning_output_tokens": 41}}]


def test_extract_message_ignores_lifecycle_json_lines():
    output = "\n".join([
        '{"type":"thread.started"}',
        '{"event":"turn.started"}',
        '{"type":"request.completed"}',
        '{"message":"final answer"}',
        '{"type":"turn.completed"}',
    ])

    assert CodexCliChatAdapter._extract_message(output) == "final answer"


def test_adapter_stream_signed_out_returns_clean_error(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": False,
        "status": "not_authenticated",
    })

    adapter = CodexCliChatAdapter(lambda: svc, runner=_codex_help_runner)
    events = run(_collect(adapter.stream_chat(
        [{"role": "user", "content": "Say ok"}],
        timeout_seconds=5,
        allow_one_shot_fallback=True,
    )))

    assert events[0]["type"] == "error"
    assert events[0]["status"] == "sign_in_required"
    assert "sign in" in events[0]["error"].lower()
    assert "503" not in str(events[0])


def test_normal_chat_routes_use_explicit_codex_branches_before_generic_llm():
    source = (REPO_ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    non_stream_branch = source.index("if is_codex_model_selection(sess.endpoint_url, sess.model):")
    generic_non_stream = source.index("reply = await llm_call_async(", non_stream_branch)
    assert non_stream_branch < generic_non_stream
    assert "CodexModelProvider().test_chat(" in source[non_stream_branch:generic_non_stream]

    stream_branch = source.index("elif is_codex_model_selection(sess.endpoint_url, sess.model):")
    generic_stream = source.index("elif chat_mode == \"chat\":", stream_branch)
    assert stream_branch < generic_stream
    branch_source = source[stream_branch:generic_stream]
    assert "if chat_mode != \"chat\":" in branch_source
    assert "Agent tools are not available" in branch_source
    assert "CodexModelProvider().stream_chat(" in branch_source
    assert "allow_one_shot_fallback=True" in branch_source


def test_codex_auto_name_does_not_call_generic_llm(monkeypatch):
    from routes import chat_helpers
    import src.task_endpoint as task_endpoint
    import src.llm_core as llm_core

    monkeypatch.setattr(
        task_endpoint,
        "resolve_task_endpoint",
        lambda fallback_url=None, fallback_model=None, fallback_headers=None: (
            CODEX_EXPERIMENTAL_ENDPOINT_URL,
            CODEX_EXPERIMENTAL_MODEL_ID,
            {},
        ),
    )

    async def fail_llm_call(*args, **kwargs):
        raise AssertionError("auto-name must not POST to odysseus://codex-cli")

    monkeypatch.setattr(llm_core, "llm_call_async", fail_llm_call)
    sess = SimpleNamespace(
        id="codex-session",
        endpoint_url=CODEX_EXPERIMENTAL_ENDPOINT_URL,
        model=CODEX_EXPERIMENTAL_MODEL_ID,
        headers={},
        history=[SimpleNamespace(role="user", content="hello codex")],
    )

    run(chat_helpers.auto_name_session(SimpleNamespace(update_session_name=lambda *a: None), sess))


def test_codex_post_response_skips_generic_background_llm_tasks(monkeypatch):
    from routes import chat_helpers

    created = []
    monkeypatch.setattr(chat_helpers, "_resolve_http_task_endpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_helpers.asyncio, "create_task", lambda coro: created.append(coro))
    sess = SimpleNamespace(
        id="codex-session",
        name="Already Named",
        endpoint_url=CODEX_EXPERIMENTAL_ENDPOINT_URL,
        model=CODEX_EXPERIMENTAL_MODEL_ID,
        headers={},
        history=[
            SimpleNamespace(role="user", content="one"),
            SimpleNamespace(role="assistant", content="two"),
            SimpleNamespace(role="user", content="three"),
            SimpleNamespace(role="assistant", content="four"),
        ],
    )

    chat_helpers.run_post_response_tasks(
        sess,
        session_manager=SimpleNamespace(),
        session_id=sess.id,
        message="three",
        full_response="four",
        last_metrics=None,
        uprefs={"auto_memory": True, "auto_skills": True},
        memory_manager=object(),
        memory_vector=object(),
        webhook_manager=None,
        agent_rounds=2,
        agent_tool_calls=2,
        skills_manager=object(),
    )

    assert created == []


def test_adapter_stream_falls_back_to_completed_stdout_when_no_deltas(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })

    async def runner(args, timeout, cwd=None, env=None):
        return await _codex_help_runner(args, timeout, cwd=cwd, env=env)

    async def fake_create_subprocess_exec(*args, stdout=None, stderr=None, cwd=None, env=None):
        return _FakeProcess([b"plain completed response\n"])

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    events = run(_collect(adapter.stream_chat([{"role": "user", "content": "Say ok"}], timeout_seconds=5)))

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == "plain completed response"
    assert events[1]["message"] == "plain completed response"


def test_adapter_stream_reports_not_supported_without_json(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, "Usage: codex exec --sandbox <MODE>", ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    available = run(adapter.available())
    events = run(_collect(adapter.stream_chat([{"role": "user", "content": "Say ok"}], timeout_seconds=5)))

    assert available["streaming_supported"] is False
    assert available["cli_capabilities"]["json_output_supported"] is False
    assert events[0]["type"] == "error"
    assert events[0]["status"] == "streaming_not_supported"


def test_adapter_stream_can_use_one_shot_fallback_without_json(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({
        "codex_cli_available": True,
        "authenticated": True,
        "codex_authenticated": True,
        "status": "authenticated",
    })

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, "Usage: codex exec --sandbox <MODE>", ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "one shot response", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    events = run(_collect(adapter.stream_chat(
        [{"role": "user", "content": "Say ok"}],
        timeout_seconds=5,
        allow_one_shot_fallback=True,
    )))

    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["delta"] == "one shot response"
    assert events[1]["message"] == "one shot response"


def test_adapter_handles_timeout(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_OLD_EXEC_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 124, "", "access_token=secret"

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "hello"}], timeout_seconds=1))

    assert out["ok"] is False
    assert out["status"] == "timeout"
    assert "secret" not in str(out)


def test_adapter_handles_cli_nonzero_and_redacts(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, CODEX_OLD_EXEC_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 2, "", "refresh_token=secret failed"

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "hello"}]))

    assert out["ok"] is False
    assert out["status"] == "cli_failed"
    assert "secret" not in str(out)
    assert "refresh_token" not in str(out)


def test_adapter_refuses_unsafe_cli_help(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        return 0, "Usage: codex exec --json", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.available())

    assert out["ok"] is False
    assert out["status"] == "unsupported_unsafe_cli_mode"
    assert "--sandbox" in out["missing_flags"]


def test_adapter_requires_long_sandbox_flag(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        if args[1:] == ["exec", "--help"]:
            return 0, "Usage: codex exec -s <MODE> --json", ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.available())

    assert out["ok"] is False
    assert out["status"] == "unsupported_unsafe_cli_mode"
    assert out["cli_capabilities"]["sandbox_supported"] is False


def test_adapter_never_uses_advertised_dangerous_flags(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    calls = []
    dangerous_help = CODEX_0135_EXEC_HELP + "\n  --dangerously-bypass-approvals-and-sandbox\n  --yolo\n"

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        return await _codex_help_runner(args, timeout, cwd=cwd, env=env, exec_help=dangerous_help)

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "hello"}]))

    assert out["ok"] is True
    assert out["cli_capabilities"]["dangerous_flags_advertised"] == [
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
    ]
    exec_args = calls[-1]
    assert "--dangerously-bypass-approvals-and-sandbox" not in exec_args
    assert "--yolo" not in exec_args


def test_status_does_not_expose_model_when_adapter_is_unsafe(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")

    class UnsafeAdapter:
        async def available(self):
            return {"ok": False, "status": "unsupported_unsafe_cli_mode"}

    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    provider = CodexModelProvider(lambda: svc, chat_adapter=UnsafeAdapter())

    out = run(provider.status())

    assert out["status"] == "unsupported_unsafe_cli_mode"
    assert out["models"] == []


def test_group_sessions_are_protected_from_auto_sort():
    from src.session_actions import session_protected_from_auto_sort

    now = datetime.now()

    assert session_protected_from_auto_sort(SimpleNamespace(
        name="[GRP] Roundtable",
        created_at=now - timedelta(minutes=10),
        updated_at=None,
    ), now=now) is True

    assert session_protected_from_auto_sort(SimpleNamespace(
        name="Fresh chat",
        created_at=now - timedelta(seconds=30),
        updated_at=None,
    ), now=now) is True

    assert session_protected_from_auto_sort(SimpleNamespace(
        name="Old chat",
        created_at=now - timedelta(minutes=5),
        updated_at=None,
    ), now=now) is False


def test_llm_call_async_uses_codex_provider_for_sentinel(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")

    calls = []

    async def fake_test_chat(self, messages, model=None, reasoning_effort=None, timeout_seconds=120, odysseus_session_id=None):
        calls.append({
            "messages": messages,
            "model": model,
            "timeout_seconds": timeout_seconds,
        })
        return {"ok": True, "message": "codex async ok"}

    monkeypatch.setattr(CodexModelProvider, "test_chat", fake_test_chat)

    out = run(llm_call_async(
        CODEX_EXPERIMENTAL_ENDPOINT_URL,
        "gpt-5.5",
        [{"role": "user", "content": "Say ok"}],
        timeout=42,
    ))

    assert out == "codex async ok"
    assert calls == [{
        "messages": [{"role": "user", "content": "Say ok"}],
        "model": "gpt-5.5",
        "timeout_seconds": 42,
    }]


def test_research_start_same_as_chat_uses_codex_sentinel(monkeypatch):
    from routes import research_routes
    import src.auth_helpers as auth_helpers

    update_codex_model_config(add_model="gpt-5.5")

    research_handler = SimpleNamespace(
        _active_tasks={},
        start_research=MagicMock(),
    )
    session_manager = SimpleNamespace(
        get_session=lambda sid: SimpleNamespace(
            endpoint_url=CODEX_EXPERIMENTAL_ENDPOINT_URL,
            model="gpt-5.5",
            headers={},
            owner="alice",
        )
    )

    monkeypatch.setattr(auth_helpers, "require_privilege", lambda request, privilege: "alice")
    monkeypatch.setattr(
        research_routes,
        "resolve_endpoint",
        lambda purpose, fallback_url="", fallback_model="", fallback_headers=None: (
            fallback_url,
            fallback_model,
            fallback_headers or {},
        ),
    )

    router = research_routes.setup_research_routes(research_handler, session_manager=session_manager)
    target = next(r.endpoint for r in router.routes if getattr(r, "path", "") == "/api/research/start")
    body = SimpleNamespace(
        query="Investigate Codex routing",
        max_rounds=0,
        search_provider=None,
        endpoint_id=None,
        model=None,
        source_session_id="chat-1",
        max_time=300,
        extraction_timeout=None,
        extraction_concurrency=None,
        category=None,
    )

    out = run(target(body=body, request=_fake_request("alice")))

    assert out["status"] == "running"
    kwargs = research_handler.start_research.call_args.kwargs
    assert kwargs["llm_endpoint"] == CODEX_EXPERIMENTAL_ENDPOINT_URL
    assert kwargs["llm_model"] == "gpt-5.5"


def test_research_spinoff_reuses_saved_codex_sentinel(monkeypatch, tmp_path):
    from routes import research_routes

    update_codex_model_config(add_model="gpt-5.5")
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data" / "deep_research"
    data_dir.mkdir(parents=True)
    (data_dir / "rp-123.json").write_text(json.dumps({
        "owner": "alice",
        "query": "Codex research",
        "result": "Saved report",
        "sources": ["https://example.test"],
        "llm_endpoint": CODEX_EXPERIMENTAL_ENDPOINT_URL,
        "llm_model": "gpt-5.5",
    }), encoding="utf-8")

    core_models = types.ModuleType("core.models")

    class _ChatMessage:
        def __init__(self, role, content, metadata=None):
            self.role = role
            self.content = content
            self.metadata = metadata or {}

    core_models.ChatMessage = _ChatMessage
    monkeypatch.setitem(sys.modules, "core.models", core_models)
    if "core" in sys.modules:
        setattr(sys.modules["core"], "models", core_models)

    event_bus = types.ModuleType("src.event_bus")
    event_bus.fire_event = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "src.event_bus", event_bus)

    created = {}

    class _FakeSession:
        def __init__(self):
            self.headers = {}
            self.messages = []

        def add_message(self, msg):
            self.messages.append(msg)

    class _SessionManager:
        def get_session(self, sid):
            raise KeyError(sid)

        def create_session(self, session_id, name, endpoint_url, model, rag, owner):
            created.update({
                "session_id": session_id,
                "name": name,
                "endpoint_url": endpoint_url,
                "model": model,
                "owner": owner,
            })
            sess = _FakeSession()
            created["session"] = sess
            return sess

        def save_sessions(self):
            created["saved"] = created.get("saved", 0) + 1

    research_handler = SimpleNamespace(
        get_result=lambda session_id: None,
        get_sources=lambda session_id: [],
    )
    router = research_routes.setup_research_routes(research_handler, session_manager=_SessionManager())
    target = next(r.endpoint for r in router.routes if getattr(r, "path", "") == "/api/research/spinoff/{session_id}")

    out = run(target(session_id="rp-123", request=_fake_request("alice")))

    assert out["source_count"] == 1
    assert created["endpoint_url"] == CODEX_EXPERIMENTAL_ENDPOINT_URL
    assert created["model"] == "gpt-5.5"
    assert created["owner"] == "alice"
    assert created["session"].messages[0].metadata["research_spinoff_from"] == "rp-123"


def test_chat_stream_codex_agent_mode_uses_structured_capability_response():
    source = (REPO_ROOT / "routes" / "chat_routes.py").read_text(encoding="utf-8")

    assert "agent_mode_unsupported" in source
    assert "status_code=409" in source


def test_group_source_uses_chat_stream_and_saves_only_accumulated_text():
    source = (REPO_ROOT / "static" / "js" / "group.js").read_text(encoding="utf-8")

    assert "/api/chat_stream" in source
    assert "if (!res.ok || !res.body)" in source
    assert "content: accumulated" in source
