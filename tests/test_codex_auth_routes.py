import asyncio
from types import SimpleNamespace

from routes.codex_auth_routes import setup_codex_auth_routes
from src.codex_auth import set_codex_auth_service


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


class _FakeService:
    def __init__(self):
        self.calls = []

    async def status(self):
        self.calls.append("status")
        return {"status": "not_authenticated"}

    async def start(self):
        self.calls.append("start")
        return {
            "status": "pending",
            "verification_url": "https://auth.openai.com/codex/device",
            "user_code": "CODE-12345",
        }

    async def cancel(self):
        self.calls.append("cancel")
        return {"status": "canceled"}

    async def logout(self):
        self.calls.append("logout")
        return {"status": "logged_out"}


def _endpoint(router, route_path, method):
    for route in router.routes:
        if getattr(route, "path", "") == route_path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"route not found: {method} {route_path}")


def test_codex_auth_routes_use_service():
    service = _FakeService()
    set_codex_auth_service(service)
    try:
        router = setup_codex_auth_routes()
        cases = [
            ("/api/codex-auth/status", "GET", "status"),
            ("/api/codex-auth/start", "POST", "start"),
            ("/api/codex-auth/cancel", "POST", "cancel"),
            ("/api/codex-auth/logout", "POST", "logout"),
        ]
        for route_path, method, expected_call in cases:
            endpoint = _endpoint(router, route_path, method)
            asyncio.run(endpoint(_request()))
            assert service.calls[-1] == expected_call
        assert "GET" in next(route.methods for route in router.routes if route.path == "/api/codex-auth/status")
    finally:
        set_codex_auth_service(None)


def test_codex_auth_routes_admin_gated():
    set_codex_auth_service(_FakeService())
    try:
        router = setup_codex_auth_routes()
        for route_path, method in [
            ("/api/codex-auth/status", "GET"),
            ("/api/codex-auth/start", "POST"),
            ("/api/codex-auth/cancel", "POST"),
            ("/api/codex-auth/logout", "POST"),
        ]:
            endpoint = _endpoint(router, route_path, method)
            try:
                asyncio.run(endpoint(_request(user="bob")))
            except Exception as exc:
                assert getattr(exc, "status_code", None) == 403
            else:
                raise AssertionError(f"non-admin request should fail: {method} {route_path}")
    finally:
        set_codex_auth_service(None)


def test_codex_auth_routes_do_not_expose_test_endpoint():
    router = setup_codex_auth_routes()
    route_names = {route.name for route in router.routes}
    assert route_names == {"status", "start", "cancel", "logout"}
