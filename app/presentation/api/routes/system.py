from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi import Cookie

from app.presentation.api.context import ApiContext


def build(ctx: ApiContext) -> APIRouter:
    user_from_cookie = ctx.user_from_cookie
    router = APIRouter()

    @router.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/api/system/preflight")
    def system_preflight(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        user_from_cookie(ia_session)
        from app.application.pipeline.preflight import run_preflight

        items = run_preflight()
        return {
            "ok": all(item.ok for item in items),
            "items": [
                {"component": item.component, "ok": item.ok, "detail": item.detail}
                for item in items
            ],
        }

    return router
