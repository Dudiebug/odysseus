"""Admin-gated Codex model-provider routes."""

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from core.middleware import require_admin
from src.codex_model_provider import CodexModelProvider, update_codex_model_config

class CodexModelUpdateRequest(BaseModel):
    model_id: str | None = None
    action: str | None = None
    thinking_effort: str | None = None

def setup_codex_model_provider_routes(provider: CodexModelProvider | None = None) -> APIRouter:
    router = APIRouter(prefix="/api/codex-model-provider", tags=["codex-model-provider"])
    provider = provider or CodexModelProvider()

    @router.get("/status")
    async def status(request: Request) -> dict[str, Any]:
        require_admin(request)
        return await provider.status()

    @router.get("/models")
    async def list_models(request: Request) -> dict[str, Any]:
        require_admin(request)
        return await provider.status()

    @router.post("/models")
    async def add_model(request: Request, body: CodexModelUpdateRequest) -> dict[str, Any]:
        require_admin(request)
        model_id = (body.model_id or "").strip()
        if not model_id:
            return {"ok": False, "status": "invalid_request", "error": "Model ID is required"}
        cfg = update_codex_model_config(
            add_model=model_id,
            thinking_model=model_id,
            thinking_effort=body.thinking_effort,
            connector_enabled=True,
        )
        return {"ok": True, "config": cfg}

    @router.post("/connector")
    async def add_connector(request: Request) -> dict[str, Any]:
        require_admin(request)
        cfg = update_codex_model_config(connector_enabled=True)
        return {"ok": True, "config": cfg}
    @router.delete("/connector")
    async def remove_connector(request: Request) -> dict[str, Any]:
        require_admin(request)
        cfg = update_codex_model_config(clear_all_models=True, connector_enabled=False)
        return {"ok": True, "config": cfg}
    @router.patch("/models")
    async def update_model(request: Request, body: CodexModelUpdateRequest) -> dict[str, Any]:
        require_admin(request)
        model_id = (body.model_id or "").strip()
        action = (body.action or "").strip().lower()
        if not model_id:
            return {"ok": False, "status": "invalid_request", "error": "Model ID is required"}

        kwargs: dict[str, Any] = {}
        if action == "hide":
            kwargs["hide_model"] = model_id
        elif action == "remove":
            kwargs["remove_model"] = model_id
        elif action == "restore":
            kwargs["restore_model"] = model_id
        elif action == "disable":
            kwargs["disable_model"] = model_id
        elif action == "enable":
            kwargs["enable_model"] = model_id
        elif action == "set_thinking_effort":
            kwargs["thinking_model"] = model_id
            kwargs["thinking_effort"] = body.thinking_effort
        else:
            return {"ok": False, "status": "invalid_request", "error": "Unsupported model action"}

        cfg = update_codex_model_config(**kwargs)
        return {"ok": True, "config": cfg}

    return router
