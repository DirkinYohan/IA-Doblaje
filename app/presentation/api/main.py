from __future__ import annotations

import os
from typing import Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.presentation.api import db
from app.presentation.api import mailer
from app.presentation.api import media_store
from app.presentation.api import worker
from app.presentation.api.common import _paths
from app.presentation.api.context import build_context
from app.presentation.api.routes import auth
from app.presentation.api.routes import jobs
from app.presentation.api.routes import live
from app.presentation.api.routes import projects
from app.presentation.api.routes import render
from app.presentation.api.routes import subtitles
from app.presentation.api.routes import system
from app.presentation.api.cluster import DbRateLimiter


class _StampArrival:
    """Anota cuándo llegó la petición, antes de leer el cuerpo (subidas largas)."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "http":
            import time

            scope.setdefault("state", {})["started_at"] = time.time()
        await self.app(scope, receive, send)


def create_app() -> FastAPI:
    settings = get_settings()
    paths = _paths()
    conn = db.connect(paths["db"])
    db.migrate(conn)
    paths["mail"].mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        yield
        conn.close()

    app = FastAPI(title="IA Doblaje", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(_StampArrival)
    origins = [
        o.strip()
        for o in os.environ.get(
            "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["*"],
    )
    app.state.conn = conn

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Datos inválidos"})

    app.state.paths = paths
    app.state.login_limit = DbRateLimiter(conn, "login", 10, 60)
    app.state.register_limit = DbRateLimiter(conn, "register", 5, 60)
    app.state.forgot_limit = DbRateLimiter(conn, "forgot", 5, 300)
    app.state.verify_limit = DbRateLimiter(conn, "verify", 5, 600)
    app.state.mailer = mailer.from_env(paths["mail"])
    try:
        media_store.purge_temp(Path(str(settings.paths.data_temp_dir)), attempts=5, interval=0.2)
        media_store.purge_pending(Path(str(settings.paths.data_temp_dir)))
        uploads = Path(str(settings.paths.data_temp_dir)) / "uploads"
        if uploads.is_dir():
            import shutil as _shutil

            _shutil.rmtree(uploads, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass
    if os.environ.get("IA_PRELOAD_MODELS", "0") == "1":
        worker.preload_models()
    app.state.worker_enabled = os.environ.get("IA_DISABLE_WORKER") != "1"
    if app.state.worker_enabled:
        worker.start(conn)
        worker.reconcile_orphans(conn)

    ctx = build_context(app, conn, paths, settings)
    for module in (auth, system, projects, subtitles, jobs, render, live):
        app.include_router(module.build(ctx))
    if app.state.worker_enabled:
        # Al despertar el servidor: los modelos cargan mientras el usuario entra,
        # no cuando pulsa play.
        ctx._warm_live_models()
    return app
