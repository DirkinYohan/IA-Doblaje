from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from fastapi import Cookie
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse

from app.presentation.api import db
from app.presentation.api import media_store
from app.presentation.api import media_upload
from app.presentation.api import storage_fit
from app.presentation.api.common import OFFICIAL
from app.presentation.api.common import UPLOAD_CHUNK
from app.presentation.api.common import _create_job
from app.presentation.api.common import _ensure_project_metadata
from app.presentation.api.common import _guess_mime
from app.presentation.api.common import _job
from app.presentation.api.common import _now
from app.presentation.api.common import _own_project
from app.presentation.api.common import _probe_duration_ms
from app.presentation.api.common import _probe_streams
from app.presentation.api.common import _project
from app.presentation.api.common import _raise_engine_error
from app.presentation.api.common import _safe_filename
from app.presentation.api.common import _write_storyboard
from app.presentation.api.common import _write_thumbnail
from app.presentation.api.context import ApiContext
from app.presentation.api.schemas import RenameBody


def _remove_when_free(path: Path, attempts: int = 30, interval: float = 1.0) -> None:
    import threading
    import time

    def remove() -> bool:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    if remove():
        return

    def retry() -> None:
        for _ in range(attempts):
            time.sleep(interval)
            if remove():
                return

    threading.Thread(target=retry, name="borrar-medio", daemon=True).start()


def build(ctx: ApiContext) -> APIRouter:
    app = ctx.app
    conn = ctx.conn
    paths = ctx.paths
    settings = ctx.settings
    _warm_live_models = ctx._warm_live_models
    user_from_cookie = ctx.user_from_cookie
    router = APIRouter()

    @router.get("/api/videos")
    def list_videos(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        _warm_live_models()
        projects = db.many(conn, "SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC", (row["id"],))
        projects = [_ensure_project_metadata(conn, paths["input"], p) for p in projects]
        jobs = db.many(conn, "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (row["id"],))
        return {"videos": [_project(p) for p in projects], "jobs": [_job(j) for j in jobs]}

    @router.post("/api/videos")
    def upload_video(
        request: Request,
        file: UploadFile = File(...),
        source_language: str = Form("und"),
        target_languages: str = Form("es"),
        ticket: str | None = None,
        ia_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        row = ctx.user_from_ticket_or_cookie(ticket, ia_session)
        source = source_language.strip().lower()
        targets = [p.strip().lower() for p in target_languages.split(",") if p.strip()]
        if source not in OFFICIAL | {"und"} or not targets or any(t not in OFFICIAL for t in targets):
            raise HTTPException(status_code=422, detail="Idioma no soportado")
        filename = _safe_filename(file.filename or "video.bin")
        if not filename:
            raise HTTPException(status_code=422, detail="Nombre de archivo inválido")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        allowed = settings.safety.allowed_extensions_set
        if ext not in allowed:
            raise HTTPException(status_code=422, detail="Formato no permitido")

        max_bytes = int(os.environ.get("IA_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
        project_id = str(uuid.uuid4())

        use_storage = media_store.configured()
        if use_storage:
            target = paths["temp"] / "uploads" / f"{project_id}-{filename}"
        else:
            folder = (paths["input"] / row["id"] / project_id).resolve()
            root = paths["input"].resolve()
            if not str(folder).startswith(str(root)):
                raise HTTPException(status_code=400, detail="Ruta rechazada")
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / filename

        written = 0
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as sink:
                while True:
                    chunk = file.file.read(UPLOAD_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"El archivo supera el límite de {max_bytes // (1024 * 1024)} MB",
                        )
                    sink.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            _raise_engine_error("No se pudo guardar el archivo", exc)
        finally:
            file.file.close()

        if written == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="El archivo está vacío")

        duration_ms = _probe_duration_ms(target)

        background: dict[str, Any] | None = None
        if use_storage:
            try:
                compress = storage_fit.needs_fit(written, duration_ms)
            except storage_fit.TooLongForStorage as exc:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            stored_name = f"{Path(filename).stem}.mp4" if compress else filename
            media_key = f"{row['id']}/{project_id}/{stored_name}"
            thumb_key = f"{row['id']}/{project_id}/thumbnail.jpg"
            thumb_temp = paths["temp"] / "uploads" / f"{project_id}-thumbnail.jpg"
            has_thumb = _write_thumbnail(target, thumb_temp)
            try:
                pending = media_store.pending_path(paths["temp"], project_id, target.suffix)
                pending.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(target, pending)
                except OSError:
                    shutil.copyfile(target, pending)
            except Exception:  # noqa: BLE001 - el worker usará la copia en preparación
                pass

            bucket = os.environ.get("SUPABASE_BUCKET", "media")
            media_value = media_store.make_uri(bucket, media_key)
            thumb_value = media_store.make_uri(bucket, thumb_key) if has_thumb else ""
            background = {
                "project_id": project_id, "source": target, "duration_ms": duration_ms,
                "media_key": media_key, "content_type": _guess_mime(stored_name),
                "thumbnail": thumb_temp if has_thumb else None, "thumbnail_key": thumb_key,
                "still_exists": lambda: db.one(conn, "SELECT id FROM projects WHERE id=?", (project_id,)) is not None,
            }
        else:
            thumbnail = folder / "thumbnail.jpg"
            has_thumb = _write_thumbnail(target, thumbnail)
            media_value = str(target)
            thumb_value = str(thumbnail) if has_thumb else ""

        db.run(
            conn,
            """
            INSERT INTO projects (id, user_id, title, source_language, target_languages, status,
                                  media_path, created_at, duration_ms, thumbnail_path, updated_at)
            VALUES (?, ?, ?, ?, ?, 'READY', ?, ?, ?, ?, ?)
            """,
            (
                project_id, row["id"], filename, source, ",".join(targets), media_value,
                _now(), duration_ms, thumb_value, _now(),
            ),
        )
        if background is not None:
            media_upload.store_in_background(**background)
        job_id = _create_job(conn, project_id, row["id"], "progressive", start=app.state.worker_enabled)
        uploaded = db.one(conn, "SELECT * FROM projects WHERE id=?", (project_id,))
        if uploaded is not None and app.state.worker_enabled:
            from app.presentation.api import live

            live.start_for_project(conn, uploaded, job_id=job_id)
        return {"id": project_id, "job_id": job_id, "title": filename, "duration_ms": duration_ms}

    @router.get("/api/videos/{project_id}")
    def get_video(project_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        project = _ensure_project_metadata(conn, paths["input"], project)
        jobs = db.many(conn, "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC", (project_id,))
        return {"video": _project(project), "jobs": [_job(j) for j in jobs]}

    @router.delete("/api/videos/{project_id}")
    def delete_video(project_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        from app.presentation.api import live

        running = live.get_runner(project_id)
        if running is not None:
            running.stop()
            running.done.wait(5)
        from app.presentation.api import render

        render.forget(conn, paths["output"], project_id)

        media_path = project["media_path"]
        if media_store.is_remote(media_path):
            key = media_store.object_key(media_path)
            if key:
                try:
                    media_store.delete(key)
                except Exception:  # noqa: BLE001 - borrar el proyecto manda
                    pass
        else:
            _remove_when_free(Path(media_path))

        thumb_path = str(project["thumbnail_path"] or "")
        if media_store.is_remote(thumb_path):
            thumb_key = media_store.object_key(thumb_path)
            if thumb_key:
                try:
                    media_store.delete(thumb_key)
                except Exception:  # noqa: BLE001
                    pass

        with db.transaction(conn) as tx:
            tx.run("DELETE FROM live_leases WHERE project_id=?", (project_id,))
            tx.run("DELETE FROM subtitles WHERE project_id=?", (project_id,))
            tx.run("DELETE FROM jobs WHERE project_id=?", (project_id,))
            tx.run("DELETE FROM projects WHERE id=?", (project_id,))
        return {"ok": True}

    @router.put("/api/videos/{project_id}")
    def rename_video(project_id: str, body: RenameBody, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        title = body.title.strip()
        if not title or len(title) > 180:
            raise HTTPException(status_code=422, detail="Título inválido")
        db.run(conn, "UPDATE projects SET title=? WHERE id=?", (title, project_id))
        return {"title": title}

    @router.get("/api/videos/{project_id}/media")
    def media(project_id: str, ia_session: str | None = Cookie(default=None)) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        media_path = project["media_path"]

        if media_store.is_remote(media_path):
            local = media_store.find_pending(paths["temp"], project_id)
            if local is not None:
                return FileResponse(local, media_type=_guess_mime(local.name))
            url = media_store.signed_url(media_path, expires_in=3600)
            if url:
                return RedirectResponse(url, status_code=307)
            raise HTTPException(status_code=503, detail="Almacenamiento no disponible")

        path = Path(media_path).resolve()
        if not path.is_file() or not str(path).startswith(str(paths["input"].resolve())):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        return FileResponse(path)

    @router.get("/api/videos/{project_id}/audio-check")
    def audio_check(project_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        media_path = project["media_path"]

        remote = media_store.is_remote(media_path)
        media: Path | None = None
        try:
            if remote:
                media = media_store.materialize(
                    media_path, paths["temp"], label=f"check-{project_id}"
                )
            else:
                candidate = Path(str(media_path)).resolve()
                root = paths["input"].resolve()
                if not str(candidate).startswith(str(root)) or not candidate.is_file():
                    raise HTTPException(status_code=404, detail="Archivo original no encontrado")
                media = candidate

            report = _probe_streams(media)
        finally:
            if remote:
                media_store.release(media, paths["temp"])

        audio_ok = report["audio_streams"] >= 1 and report["video_streams"] >= 1
        return {
            "project_id": project_id,
            "original": report,
            "preserved": audio_ok,
            "duration_ms": int(project["duration_ms"] or 0),
            "storage": "supabase" if remote else "local",
        }

    @router.get("/api/videos/{project_id}/thumbnail")
    def thumbnail(project_id: str, ia_session: str | None = Cookie(default=None)) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        raw = str(project["thumbnail_path"] or "")
        if not raw:
            raise HTTPException(status_code=404, detail="Sin miniatura")

        if media_store.is_remote(raw):
            local = media_store.staged(project_id, "thumbnail")
            if local is not None:
                return FileResponse(local, media_type="image/jpeg")
            url = media_store.signed_url(raw, expires_in=3600)
            if not url:
                raise HTTPException(status_code=503, detail="Almacenamiento no disponible")
            return RedirectResponse(url, status_code=307)

        path = Path(raw).resolve()
        root = paths["input"].resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            raise HTTPException(status_code=404, detail="Sin miniatura")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/api/videos/{project_id}/storyboard")
    def storyboard(
        project_id: str,
        frames: int = 14,
        ia_session: str | None = Cookie(default=None),
    ) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        media_path = project["media_path"]

        count = max(4, min(24, int(frames)))
        cached = paths["temp"] / "storyboards" / f"{project_id}-{count}.jpg"
        if cached.is_file() and cached.stat().st_size > 0:
            return FileResponse(cached, media_type="image/jpeg")

        remote = media_store.is_remote(media_path)
        local: Path | None = None
        try:
            if remote:
                local = media_store.materialize(
                    media_path, paths["temp"], label=f"story-{project_id}"
                )
            else:
                candidate = Path(str(media_path)).resolve()
                if not candidate.is_file():
                    raise HTTPException(status_code=404, detail="Archivo no encontrado")
                local = candidate

            if not _write_storyboard(local, cached, frames=count):
                raise HTTPException(status_code=503, detail="No se pudo generar la tira")
        finally:
            if remote:
                media_store.release(local, paths["temp"])

        return FileResponse(
            cached,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    return router
