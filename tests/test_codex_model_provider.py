import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()

from src.codex_model_provider import (
    CODEX_EXPERIMENTAL_MODEL_ID,
    CODEX_MODEL_PROVIDER_FLAG,
    CodexModelProvider,
)


def run(coro):
    return asyncio.run(coro)


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


def _restore_modules(saved):
    for name, module in saved.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_routes_module():
    saved = {name: sys.modules.get(name, _MISSING) for name in ("fastapi", "core", "core.middleware")}
    try:
        fastapi_stub = types.ModuleType("fastapi")
        fastapi_stub.APIRouter = _APIRouter
        fastapi_stub.HTTPException = _HTTPException
        fastapi_stub.Request = object
        sys.modules["fastapi"] = fastapi_stub

        core_pkg = types.ModuleType("core")
        core_pkg.__path__ = [str(ROOT / "core")]
        sys.modules["core"] = core_pkg
        middleware_spec = importlib.util.spec_from_file_location("core.middleware", ROOT / "core" / "middleware.py")
        middleware = importlib.util.module_from_spec(middleware_spec)
        sys.modules["core.middleware"] = middleware
        assert middleware_spec and middleware_spec.loader
        middleware_spec.loader.exec_module(middleware)

        routes_spec = importlib.util.spec_from_file_location(
            "_codex_model_provider_routes_under_test",
            ROOT / "routes" / "codex_model_provider_routes.py",
        )
        routes_module = importlib.util.module_from_spec(routes_spec)
        assert routes_spec and routes_spec.loader
        routes_spec.loader.exec_module(routes_module)
        return routes_module
    finally:
        _restore_modules(saved)


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
    routes_module = _load_routes_module()
    router = routes_module.setup_codex_model_provider_routes(CodexModelProvider())
    status = _endpoint(router, "/api/codex-model-provider/status", "GET")

    out = run(status(_request(user="admin")))
    assert out["status"] == "enabled"

    with pytest.raises(Exception) as exc_info:
        run(status(_request(user="bob")))
    assert getattr(exc_info.value, "status_code", None) == 403
