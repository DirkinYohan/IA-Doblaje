from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi import Cookie
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from app.presentation.api import db
from app.presentation.api import worker
from app.presentation.api.common import _create_job
from app.presentation.api.common import _job
from app.presentation.api.common import _own_job
from app.presentation.api.common import _own_project
from app.presentation.api.context import ApiContext
from app.presentation.api.schemas import CreateJobBody
from app.presentation.api.schemas import ProduceBody


def build(ctx: ApiContext) -> APIRouter:
    conn = ctx.conn
    user_from_cookie = ctx.user_from_cookie
    user_from_socket = ctx.user_from_socket
    router = APIRouter()

    @router.post("/api/jobs")
    def create_job(body: CreateJobBody, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, body.project_id, row["id"])
        mode = body.mode or "progressive"
        return {"id": _create_job(conn, project["id"], row["id"], mode)}

    @router.post("/api/videos/{project_id}/produce")
    def produce_video(
        project_id: str,
        body: ProduceBody | None = None,
        ia_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])

        mode = (body.mode if body else "") or "progressive"
        if mode not in {"progressive", "full"}:
            mode = "progressive"

        running = db.one(
            conn,
            "SELECT id FROM jobs WHERE project_id=? AND status IN "
            "('QUEUED','PROCESSING','TRANSCRIBING','TRANSLATING',"
            "'GENERATING_SUBTITLES') ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        if running is not None:
            return {"id": running["id"], "reused": True, "mode": mode}

        pending = db.one(
            conn,
            "SELECT id FROM jobs WHERE project_id=? AND status='PENDING' "
            "ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        if pending is not None:
            job_id = str(pending["id"])
            db.run(conn, "UPDATE jobs SET mode=? WHERE id=?", (mode, job_id))
            worker.enqueue(conn, job_id)
            return {"id": job_id, "reused": True, "mode": mode, "started": True}

        job_id = _create_job(conn, project["id"], row["id"], mode)
        return {"id": job_id, "reused": False, "mode": mode, "started": True}

    @router.websocket("/api/jobs/{job_id}/events")
    async def job_events(socket: WebSocket, job_id: str) -> None:
        import asyncio

        await socket.accept()
        try:
            user = await asyncio.to_thread(user_from_socket, socket)
        except HTTPException:
            await socket.send_json({"type": "error", "code": "unauthorized", "message": "No autenticado"})
            await socket.close()
            return

        def _load(job: str, owner: str) -> Any:
            return db.one(
                conn, "SELECT * FROM jobs WHERE id=? AND user_id=?", (job, owner)
            )

        if await asyncio.to_thread(_load, job_id, user["id"]) is None:
            await socket.send_json({"type": "error", "message": "job inexistente"})
            await socket.close()
            return

        channel = worker.subscribe(job_id)
        try:
            current = await asyncio.to_thread(db.one, conn, "SELECT * FROM jobs WHERE id=?", (job_id,))
            if current is not None:
                await socket.send_json({"type": "snapshot", **_job(current)})
                if current["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    await socket.send_json({"type": "processing_finished", **_job(current)})

            while True:
                try:
                    event = await asyncio.to_thread(channel.get, True, 1.0)
                except Exception:  # noqa: BLE001 - vacío: se comprueba el cierre
                    event = None
                if event is not None:
                    await socket.send_json(event)
                    if event.get("type") in {"processing_completed", "processing_failed"}:
                        break
                    continue
                row_now = await asyncio.to_thread(
                    db.one, conn, "SELECT * FROM jobs WHERE id=?", (job_id,)
                )
                if row_now is None:
                    break
                if row_now["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    await socket.send_json({"type": "processing_finished", **_job(row_now)})
                    break
        except WebSocketDisconnect:
            return
        finally:
            worker.unsubscribe(job_id, channel)

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        job = _own_job(conn, job_id, row["id"])
        return _job(job)

    @router.get("/api/jobs/{job_id}/progress")
    def job_progress(job_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        job = _own_job(conn, job_id, row["id"])
        payload = _job(job)
        return {
            "id": payload["id"],
            "project_id": payload["project_id"],
            "status": payload["status"],
            "progress": payload["progress"],
            "stage": payload["stage"],
            "error": payload["error"],
            "updated_at": payload["updated_at"],
            "stage_history": payload["stage_history"],
        }

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        _own_job(conn, job_id, row["id"])
        worker.cancel(conn, job_id)
        return {"id": job_id, "status": "CANCELLED"}

    return router
