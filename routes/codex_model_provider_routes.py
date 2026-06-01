"""Admin-gated experimental Codex model-provider routes."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

from core.middleware import require_admin
from src.codex_model_provider import CodexModelProvider


class CodexTestChatRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    model: str | None = None
    timeout_seconds: int | None = None
    odysseus_session_id: str | None = None


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
            odysseus_session_id=getattr(body, "odysseus_session_id", None),
        )

    @router.post("/test-chat-stream")
    async def test_chat_stream(request: Request, body: CodexTestChatRequest):
        require_admin(request)
        messages = body.messages or []
        if not messages and body.prompt:
            messages = [{"role": "user", "content": body.prompt}]
        if not messages:
            async def invalid_request():
                yield 'data: {"type":"error","status":"invalid_request","error":"Provide either prompt or messages"}\n\n'

            return StreamingResponse(invalid_request(), media_type="text/event-stream")

        async def event_stream():
            async for event in provider.stream_chat(
                messages,
                model=body.model,
                timeout_seconds=body.timeout_seconds or 120,
                odysseus_session_id=getattr(body, "odysseus_session_id", None),
            ):
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
