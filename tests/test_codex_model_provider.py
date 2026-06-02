import asyncio
import os
import sys
import types
from types import SimpleNamespace


class _HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _APIRouter:
    def __init__(self, prefix="", tags=None):
        self.prefix = prefix
        self.tags = tags or []
        self.routes = []

    def _register(self, path, method):
        def decorator(endpoint):
            self.routes.append(
                SimpleNamespace(
                    path=f"{self.prefix}{path}",
                    methods={method},
                    endpoint=endpoint,
                )
            )
            return endpoint

        return decorator

    def get(self, path):
        return self._register(path, "GET")


fastapi_stub = types.ModuleType("fastapi")
fastapi_stub.APIRouter = _APIRouter
fastapi_stub.HTTPException = _HTTPException
fastapi_stub.Request = object
sys.modules["fastapi"] = fastapi_stub


if "core" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")]
    sys.modules["core"] = core_pkg


from routes.codex_model_provider_routes import setup_codex_model_provider_routes
from src.codex_model_provider import (
    CODEX_EXPERIMENTAL_MODEL_ID,
    CODEX_MODEL_PROVIDER_FLAG,
    CodexModelProvider,
)


def run(coro):
    return asyncio.run(coro)


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

    out = run(CodexModelProvider().status())

    assert out["feature_enabled"] is False
    assert out["status"] == "disabled"
    assert out["models"] == []
    assert out["model_discovery"] == {"source": "disabled"}


def test_codex_model_provider_reports_static_payload_when_enabled(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")

    out = run(CodexModelProvider().status())

    assert out["feature_enabled"] is True
    assert out["status"] == "enabled"
    assert out["cli_available"] is None
    assert out["authenticated"] is None
    assert out["requires_sign_in"] is False
    assert out["connector_enabled"] is False
    assert out["selected_models"] == []
    assert out["manual_models"] == []
    assert out["hidden_models"] == []
    assert out["disabled_models"] == []
    assert out["recommended_models"] == []
    assert out["model_discovery"] == {"source": "static"}
    assert out["auth"] == {"status": "", "auth_mode": ""}
    assert "sign_in_route" not in out

    model = out["models"][0]
    assert model["id"] == CODEX_EXPERIMENTAL_MODEL_ID
    assert model["experimental"] is True
    assert model["streaming_supported"] is False
    assert model["thinking_supported"] is False
    assert model["thinking_effort_levels"] == []
    assert model["thinking_activity_supported"] is False
    assert model["session_resume_supported"] is False
    assert model["tool_calling_supported"] is False
    assert model["agent_tools_supported"] is False
    assert out["chat_supported"] is False
    assert out["streaming_supported"] is False
    assert out["session_resume_supported"] is False
    assert out["tool_execution_allowed"] is False
    assert out["thinking_supported"] is False
    assert out["thinking_effort_levels"] == []
    assert out["thinking_activity_supported"] is False


def test_codex_model_provider_route_is_admin_gated(monkeypatch):
    monkeypatch.setenv(CODEX_MODEL_PROVIDER_FLAG, "true")
    router = setup_codex_model_provider_routes(CodexModelProvider())
    status = _endpoint(router, "/api/codex-model-provider/status", "GET")

    out = run(status(_request(user="admin")))
    assert out["status"] == "enabled"

    try:
        run(status(_request(user="bob")))
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 403
    else:
        raise AssertionError("non-admin request should fail")
