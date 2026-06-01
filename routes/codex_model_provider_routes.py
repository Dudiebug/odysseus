"""Admin-gated experimental Codex model-provider routes."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Any

from core.middleware import require_admin
from src.codex_model_provider import CodexModelProvider


class CodexTestChatRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    model: str | None = None
    timeout_seconds: int | None = None


def setup_codex_model_provider_routes(provider: CodexModelProvider | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/codex-model-provider", tags=["codex-model-provider"])
    provider = provider or CodexModelProvider()

    @router.get("/status")
    async def status(request: Request):
        require_admin(request)
        return await provider.status()

    @router.post("/test-chat")
    async def test_chat(request: Request, body: CodexTestChatRequest):
        require_admin(request)
        messages = body.messages or []
        if not messages and body.prompt:
            messages = [{"role": "user", "content": body.prompt}]
        if not messages:
            return {
                "ok": False,
                "status": "invalid_request",
                "error": "Provide either prompt or messages",
            }
        return await provider.test_chat(
            messages,
            model=body.model,
            timeout_seconds=body.timeout_seconds or 120,
        )

    return router
