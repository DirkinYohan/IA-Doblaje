from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from app.presentation.api import db
from app.presentation.api.common import _job_is_running
from app.presentation.api.common import _live_translator
from app.presentation.api.common import _own_project
from app.presentation.api.context import ApiContext
from app.presentation.api.realtime import RealtimeSession
from app.presentation.api.realtime import whisper_transcriber


def build(ctx: ApiContext) -> APIRouter:
    conn = ctx.conn
    user_from_socket = ctx.user_from_socket
    router = APIRouter()

    @router.websocket("/api/videos/{project_id}/live")
    async def live_subtitles(socket: WebSocket, project_id: str) -> None:
        import asyncio

        from app.presentation.api import live

        await socket.accept()
        try:
            user = await asyncio.to_thread(user_from_socket, socket)
            project = await asyncio.to_thread(_own_project, conn, project_id, user["id"])
        except HTTPException as exc:
            code = "unauthorized" if exc.status_code == 401 else "not_found"
            await socket.send_json({"type": "error", "code": code, "message": str(exc.detail)})
            await socket.close()
            return

        job = db.one(
            conn,
            "SELECT id, status, mode FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        if job is not None and _job_is_running(str(job["status"])) and str(job["mode"]) == "full":
            await socket.send_json({"type": "busy"})
            await socket.close()
            return

        runner = live.get_runner(project_id)
        if runner is None:
            existing = await asyncio.to_thread(live.load_existing, conn, project_id)
            duration = int(project["duration_ms"] or 0)
            targets = [t for t in str(project["target_languages"] or "").split(",") if t]
            source = str(project["source_language"] or "")
            languages = [source] + [t for t in targets if t != source]
            done = duration > 0 and source not in {"", "und"} and all(
                code in existing and existing[code][1] >= duration for code in languages
            )
            if done:
                tracks = {code: [dict(c, id=live.segment_id(code, c["start_ms"], c["end_ms"]))
                                 for c in cues] for code, (cues, _cov) in existing.items()}
                covered = {code: [[0, cov]] for code, (_cues, cov) in existing.items()}
                await socket.send_json({
                    "type": "snapshot", "seq": 0, "source_language": source, "duration_ms": duration,
                    "tracks": tracks, "covered": covered, "failed": [], "complete": True,
                })
                await socket.send_json({"type": "completed", "seq": 0, "failed": []})
                try:
                    while True:
                        await socket.receive_json()
                except (WebSocketDisconnect, RuntimeError, ValueError):
                    return
            runner = await asyncio.to_thread(
                live.start_for_project, conn, project,
                job_id=str(job["id"]) if job is not None and _job_is_running(str(job["status"])) else None,
            )

        loop = asyncio.get_running_loop()
        outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        snapshot, detach = runner.attach(lambda event: loop.call_soon_threadsafe(outbox.put_nowait, event))
        await socket.send_json(snapshot)
        if runner.done.is_set():
            await socket.send_json({"type": "completed", "seq": snapshot["seq"], "failed": snapshot["failed"]})

        async def _sender() -> None:
            while True:
                event = await outbox.get()
                if int(event.get("seq", 0)) <= int(snapshot["seq"]):
                    continue
                await socket.send_json(event)

        sender = asyncio.create_task(_sender())
        try:
            while True:
                message = await socket.receive_json()
                if message.get("type") == "position":
                    runner.set_position(
                        int(message.get("time_ms") or 0),
                        str(message.get("language") or "") or None,
                    )
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        finally:
            detach()
            sender.cancel()

    @router.websocket("/api/realtime")
    async def realtime(socket: WebSocket) -> None:
        import asyncio

        await socket.accept()
        try:
            await asyncio.to_thread(user_from_socket, socket)
        except HTTPException:
            await socket.send_json({"type": "error", "code": "unauthorized", "message": "No autenticado"})
            await socket.close()
            return
        try:
            from app.infrastructure.audio.faster_whisper_asr_adapter import (
                FasterWhisperSmallASRAdapter,
            )

            session = RealtimeSession(
                whisper_transcriber(FasterWhisperSmallASRAdapter(compute_type="int8"))
            )
            session.translator = _live_translator(session)
        except Exception as exc:  # noqa: BLE001
            await socket.send_json({"type": "error", "message": f"ASR en vivo no disponible: {exc}"})
            await socket.close()
            return
        inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def _reader() -> None:
            try:
                while True:
                    await inbox.put(await socket.receive_json())
            except (WebSocketDisconnect, RuntimeError, ValueError):
                await inbox.put(None)

        reader = asyncio.create_task(_reader())
        warming = False
        try:
            while True:
                first = await inbox.get()
                if first is None:
                    return
                batch = [first]
                closed = False
                while not inbox.empty():
                    item = inbox.get_nowait()
                    if item is None:
                        closed = True
                        break
                    batch.append(item)
                events = await asyncio.to_thread(session.handle_batch, batch)
                if not warming and session.targets and hasattr(session.translator, "warm"):
                    warming = True
                    asyncio.get_running_loop().run_in_executor(None, session.translator.warm)
                for event in events:
                    await socket.send_json(event)
                if closed:
                    return
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return
        finally:
            reader.cancel()

    return router
