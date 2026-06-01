import asyncio
import os
import sys
import types
from types import SimpleNamespace

if "core" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")]
    sys.modules["core"] = core_pkg

from src.codex_model_provider import (
    CODEX_EXPERIMENTAL_MODEL_ID,
    CODEX_MODEL_PROVIDER_FLAG,
    CODEX_EXPERIMENTAL_MODEL_DISPLAY,
    CODEX_VIRTUAL_ENDPOINT_URL,
    CodexCliChatAdapter,
    CodexModelProvider,
    is_codex_virtual_endpoint,
)
from routes.codex_model_provider_routes import setup_codex_model_provider_routes


def run(coro):
    return asyncio.run(coro)


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

    async def complete(self, messages, model=None, timeout_seconds=120, odysseus_session_id=None):
        return {
            "ok": True,
            "status": "ok",
            "message": "mock response",
            "duration_ms": 1,
            "model": model or CODEX_EXPERIMENTAL_MODEL_ID,
            "limitations": [],
            "streaming_supported": False,
            "session_resume_supported": False,
            "session_resumed": False,
            "tool_execution_allowed": False,
        }

    def reset_session(self, session_id):
        return {"ok": True, "status": "reset", "session_mapping_cleared": False}


def _provider(payload):
    svc = _FakeService(payload)
    return CodexModelProvider(lambda: svc, chat_adapter=_FakeAdapter()), svc


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
    assert out["models"][0]["id"] == CODEX_EXPERIMENTAL_MODEL_ID
    assert out["models"][0]["display"] == CODEX_EXPERIMENTAL_MODEL_DISPLAY
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
        if args[-1] == "--help":
            return 0, "Usage: codex exec --sandbox <MODE> --ask-for-approval <POLICY> --json", ""
        return 0, "codex provider test ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.complete([{"role": "user", "content": "Say ok"}]))

    assert out["ok"] is True
    assert out["message"] == "codex provider test ok"
    assert out["streaming_supported"] is False
    assert out["session_resume_supported"] is False
    assert out["tool_execution_allowed"] is False
    assert calls[2][1]
    assert "--sandbox" in calls[2][0]
    assert "read-only" in calls[2][0]
    assert "--ask-for-approval" in calls[2][0]
    assert "never" in calls[2][0]
    assert "--yolo" not in calls[2][0]
    assert "--dangerously-bypass-approvals-and-sandbox" not in calls[2][0]
    assert CODEX_EXPERIMENTAL_MODEL_ID not in calls[2][0]


def test_codex_virtual_endpoint_detection():
    assert is_codex_virtual_endpoint(CODEX_VIRTUAL_ENDPOINT_URL, CODEX_EXPERIMENTAL_MODEL_ID) is True
    assert is_codex_virtual_endpoint(CODEX_VIRTUAL_ENDPOINT_URL, "") is True
    assert is_codex_virtual_endpoint("https://api.openai.com/v1/chat/completions", "gpt-4o") is False


def test_adapter_handles_timeout(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        if args[-1] == "--help":
            return 0, "Usage: codex exec --sandbox --ask-for-approval", ""
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
        if args[-1] == "--help":
            return 0, "Usage: codex exec --sandbox --ask-for-approval", ""
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
    assert "--ask-for-approval" in out["missing_flags"]


def test_adapter_detects_safety_flags_from_top_level_help(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    calls = []

    async def runner(args, timeout, cwd=None, env=None):
        calls.append(args)
        if args == ["/usr/bin/codex", "exec", "--help"]:
            return 0, "Usage: codex exec", ""
        if args == ["/usr/bin/codex", "--help"]:
            return 0, "Options: --sandbox --ask-for-approval", ""
        if args == ["/usr/bin/codex", "exec", "resume", "--help"]:
            return 1, "", "unknown"
        return 0, "ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.available())

    assert out["ok"] is True
    assert out["session_resume_supported"] is False
    assert ["/usr/bin/codex", "--help"] in calls


def test_adapter_detects_resume_support(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})

    async def runner(args, timeout, cwd=None, env=None):
        if args == ["/usr/bin/codex", "exec", "--help"]:
            return 0, "Usage: codex exec --sandbox --ask-for-approval", ""
        if args == ["/usr/bin/codex", "exec", "resume", "--help"]:
            return 0, "Usage: codex exec resume <SESSION_ID>", ""
        return 0, "ok", ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner)
    out = run(adapter.available())

    assert out["ok"] is True
    assert out["session_resume_supported"] is True


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


def test_same_odysseus_session_resumes_codex_session(monkeypatch, tmp_path):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    exec_calls = []

    async def runner(args, timeout, cwd=None, env=None):
        if args == ["/usr/bin/codex", "exec", "--help"]:
            return 0, "Usage: codex exec --sandbox --ask-for-approval", ""
        if args == ["/usr/bin/codex", "exec", "resume", "--help"]:
            return 0, "Usage: codex exec resume <SESSION_ID>", ""
        exec_calls.append((args, cwd))
        if "resume" in args:
            return 0, "resumed response", ""
        return 0, '{"session_id":"codex-session-1","message":"first response"}', ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner, session_root=tmp_path)
    first = run(adapter.complete([{"role": "user", "content": "one"}], odysseus_session_id="ody-1"))
    second = run(adapter.complete([{"role": "user", "content": "two"}], odysseus_session_id="ody-1"))

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["session_resumed"] is True
    assert exec_calls[1][0][0:4] == ["/usr/bin/codex", "exec", "resume", "codex-session-1"]
    assert "--sandbox" in exec_calls[1][0]
    assert "read-only" in exec_calls[1][0]
    assert "--ask-for-approval" in exec_calls[1][0]
    assert "never" in exec_calls[1][0]
    assert "--all" not in exec_calls[1][0]


def test_different_odysseus_sessions_do_not_share_codex_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    exec_calls = []
    next_id = {"n": 0}

    async def runner(args, timeout, cwd=None, env=None):
        if args == ["/usr/bin/codex", "exec", "--help"]:
            return 0, "Usage: codex exec --sandbox --ask-for-approval", ""
        if args == ["/usr/bin/codex", "exec", "resume", "--help"]:
            return 0, "Usage: codex exec resume <SESSION_ID>", ""
        exec_calls.append(args)
        next_id["n"] += 1
        return 0, f'{{"session_id":"codex-session-{next_id["n"]}","message":"ok"}}', ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner, session_root=tmp_path)
    run(adapter.complete([{"role": "user", "content": "one"}], odysseus_session_id="ody-1"))
    run(adapter.complete([{"role": "user", "content": "two"}], odysseus_session_id="ody-2"))

    assert "resume" not in exec_calls[0]
    assert "resume" not in exec_calls[1]


def test_reset_clears_only_selected_codex_session_mapping(monkeypatch, tmp_path):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    exec_calls = []
    ids = iter(["codex-a-1", "codex-b-1", "codex-a-2"])

    async def runner(args, timeout, cwd=None, env=None):
        if args == ["/usr/bin/codex", "exec", "--help"]:
            return 0, "Usage: codex exec --sandbox --ask-for-approval", ""
        if args == ["/usr/bin/codex", "exec", "resume", "--help"]:
            return 0, "Usage: codex exec resume <SESSION_ID>", ""
        exec_calls.append(args)
        if "resume" in args:
            return 0, "resumed", ""
        return 0, f'{{"session_id":"{next(ids)}","message":"ok"}}', ""

    adapter = CodexCliChatAdapter(lambda: svc, runner=runner, session_root=tmp_path)
    run(adapter.complete([{"role": "user", "content": "one"}], odysseus_session_id="ody-a"))
    run(adapter.complete([{"role": "user", "content": "one"}], odysseus_session_id="ody-b"))

    reset = adapter.reset_session("ody-a")
    assert reset["ok"] is True

    run(adapter.complete([{"role": "user", "content": "again"}], odysseus_session_id="ody-a"))
    run(adapter.complete([{"role": "user", "content": "again"}], odysseus_session_id="ody-b"))

    assert "resume" not in exec_calls[2]
    assert exec_calls[3][0:4] == ["/usr/bin/codex", "exec", "resume", "codex-b-1"]


def test_test_chat_route_accepts_session_id_for_reuse(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    seen = {}

    class CapturingAdapter(_FakeAdapter):
        async def complete(self, messages, model=None, timeout_seconds=120, odysseus_session_id=None):
            seen["session_id"] = odysseus_session_id
            return await super().complete(messages, model=model, timeout_seconds=timeout_seconds, odysseus_session_id=odysseus_session_id)

    svc = _FakeService({"codex_cli_available": True, "authenticated": True})
    provider = CodexModelProvider(lambda: svc, chat_adapter=CapturingAdapter())
    router = setup_codex_model_provider_routes(provider)
    test_chat = _endpoint(router, "/api/codex-model-provider/test-chat", "POST")
    body = SimpleNamespace(prompt="hello", messages=None, model=None, timeout_seconds=None, session_id="ody-test")

    out = run(test_chat(_request(user="admin"), body))

    assert out["ok"] is True
    assert seen["session_id"] == "ody-test"


def test_reset_session_route_is_admin_gated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    provider, _ = _provider({"codex_cli_available": True, "authenticated": True})
    router = setup_codex_model_provider_routes(provider)
    reset = _endpoint(router, "/api/codex-model-provider/reset-session", "POST")
    body = SimpleNamespace(session_id="ody-test")

    out = run(reset(_request(user="admin"), body))
    assert out["ok"] is True

    try:
        run(reset(_request(user="bob"), body))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("non-admin request should fail")
