import asyncio
import os
import sys
import types
from types import SimpleNamespace

from src.codex_model_provider import (
    CODEX_EXPERIMENTAL_MODEL_ID,
    CODEX_MODEL_PROVIDER_FLAG,
    CodexModelProvider,
)
from routes.codex_model_provider_routes import setup_codex_model_provider_routes


if "core" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")]
    sys.modules["core"] = core_pkg


def run(coro):
    return asyncio.run(coro)


class _FakeService:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def status(self):
        self.calls += 1
        return dict(self.payload)


def _provider(payload):
    svc = _FakeService(payload)
    return CodexModelProvider(lambda: svc), svc


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
    assert out["models"][0]["experimental"] is True
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
