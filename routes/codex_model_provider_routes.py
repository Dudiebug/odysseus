"""Admin-gated Codex model-provider routes."""

from typing import Any

from fastapi import APIRouter, Request

from core.middleware import require_admin
from src.codex_model_provider import CodexModelProvider


def setup_codex_model_provider_routes(provider: CodexModelProvider | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/codex-model-provider", tags=["codex-model-provider"])
    provider = provider or CodexModelProvider()

    @router.get("/status")
    async def status(request: Request) -> dict[str, Any]:
        require_admin(request)
        return await provider.status()

    return router
