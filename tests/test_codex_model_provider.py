import asyncio
import os
import sys
import types
from types import SimpleNamespace

import pytest

from src.codex_model_provider import (
    CODEX_EXPERIMENTAL_MODEL_ID,
    CODEX_MODEL_PROVIDER_FLAG,
    CodexCliChatAdapter,
    CodexModelProvider,
    update_codex_model_config,
)
from routes.codex_model_provider_routes import setup_codex_model_provider_routes


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
CODEX_THINKING_TOGGLE_HELP = """Usage: codex exec [OPTIONS] [PROMPT]

Options:
  -s, --sandbox <SANDBOX_MODE>
  --json
  --thinking
"""


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


def test_codex_model_provider_hidden_when_flag_disabled(monkeypatch):
    monkeypatch.delenv(CODEX_MODEL_PROVIDER_FLAG, raising=False)
    provider, svc = _provider({"codex_cli_available": True, "authenticated": True})

    out = run(provider.status())

    assert out["feature_enabled"] is False
    assert out["status"] == "disabled"
    assert out["models"] == []
    assert svc.calls == 0


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
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "codex provider test ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    provider = CodexModelProvider(lambda: svc, chat_adapter=adapter)

    out = run(provider.status())

    assert out["status"] == "available"
    assert out["cli_capabilities"]["reasoning_effort_supported"] is False
    assert out["cli_capabilities"]["reasoning_effort_levels"] == []
    assert out["thinking_supported"] is False
    assert out["thinking_effort_levels"] == []


def test_adapter_ignores_unsupported_reasoning_effort(monkeypatch):
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
            return 0, CODEX_THINKING_TOGGLE_HELP, ""
        if args[1:] == ["--help"]:
            return 0, CODEX_0135_ROOT_HELP, ""
        return 0, "codex provider test ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete(
        [{"role": "user", "content": "Say ok"}],
        reasoning_effort="low",
    ))

    assert out["ok"] is True
    assert out["reasoning_effort"] is None
    assert "--thinking" not in calls[-1]


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
    assert out["chat_supported"] is False
