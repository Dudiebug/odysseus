"""Admin-gated experimental Codex model-provider routes."""

import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

from core.middleware import require_admin
from src.codex_model_provider import CodexModelProvider, update_codex_model_config


class CodexTestChatRequest(BaseModel):
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    model: str | None = None
    timeout_seconds: int | None = None
    odysseus_session_id: str | None = None
    reasoning_effort: str | None = None


class CodexModelUpdateRequest(BaseModel):
    model_id: str | None = None
    action: str | None = None
    thinking_effort: str | None = None


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
            reasoning_effort=getattr(body, "reasoning_effort", None),
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
                reasoning_effort=getattr(body, "reasoning_effort", None),
                timeout_seconds=body.timeout_seconds or 120,
                odysseus_session_id=getattr(body, "odysseus_session_id", None),
            ):
                yield f"data: {json.dumps(event, separators=(',', ':'))}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.get("/models")
    async def list_models(request: Request):
        require_admin(request)
        return await provider.status()

    @router.post("/models")
    async def add_model(request: Request, body: CodexModelUpdateRequest):
        require_admin(request)
        if not (body.model_id or "").strip():
            return {"ok": False, "status": "invalid_request", "error": "Model ID is required"}
        cfg = update_codex_model_config(
            add_model=body.model_id,
            thinking_model=body.model_id,
            thinking_effort=body.thinking_effort,
            connector_enabled=True,
        )
        return {"ok": True, "config": cfg}

    @router.post("/connector")
    async def add_connector(request: Request):
        require_admin(request)
        cfg = update_codex_model_config(connector_enabled=True)
        return {"ok": True, "config": cfg}

    @router.delete("/connector")
    async def remove_connector(request: Request):
        require_admin(request)
        cfg = update_codex_model_config(clear_all_models=True, connector_enabled=False)
        return {"ok": True, "config": cfg}

    @router.patch("/models")
    async def update_model(request: Request, body: CodexModelUpdateRequest):
        require_admin(request)
        action = (body.action or "").strip().lower()
        kwargs = {}
        if action == "hide":
            kwargs["hide_model"] = body.model_id
        elif action == "remove":
            kwargs["remove_model"] = body.model_id
        elif action == "restore":
            kwargs["restore_model"] = body.model_id
        elif action == "disable":
            kwargs["disable_model"] = body.model_id
        elif action == "enable":
            kwargs["enable_model"] = body.model_id
        elif action == "set_thinking_effort":
            kwargs["thinking_model"] = body.model_id
            kwargs["thinking_effort"] = body.thinking_effort
        else:
            return {"ok": False, "status": "invalid_request", "error": "Unsupported model action"}
        if not (body.model_id or "").strip():
            return {"ok": False, "status": "invalid_request", "error": "Model ID is required"}
        cfg = update_codex_model_config(**kwargs)
        return {"ok": True, "config": cfg}

    return router
