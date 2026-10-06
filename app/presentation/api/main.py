"""API FastAPI: auth, videos, jobs, subtítulos, exportación y tiempo real."""
from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.application.use_cases.build_subtitles import cue_from_dict, cue_to_dict, zip_exports
from app.core.config import PROJECT_ROOT, get_settings
from app.infrastructure.subtitles.exporters import to_ass, to_srt, to_vtt
from app.presentation.api import db, worker
from app.presentation.api.realtime import RealtimeSession, m2m_translator, whisper_transcriber
from app.presentation.api.security import RateLimiter, hash_password, new_token, token_hash, verify_password

OFFICIAL = {"es", "en", "fr", "de", "it", "pt", "ja", "zh"}
COOKIE = "ia_session"
UPLOAD_CHUNK = 1024 * 1024  # 1 MiB por lectura: nunca el archivo entero en RAM
MAX_NAME = 80
MAX_TITLE = 180
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _paths() -> dict[str, Path]:
    settings = get_settings()
    root = PROJECT_ROOT
    return {
        "db": Path(os.environ.get("IA_DB_PATH", root / "data" / "app.sqlite")),
        "mail": Path(os.environ.get("IA_MAIL_DIR", root / "data" / "mail")),
        "input": Path(settings.paths.data_input_dir),
        "output": Path(settings.paths.data_output_dir),
    }


def _safe_filename(raw: str) -> str:
    """Nombre de archivo seguro: sin directorios, sin caracteres raros."""
    name = Path(str(raw or "")).name.strip()
    if not name:
        return ""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return _SAFE_STEM.sub("_", name)[:120]
    clean_stem = _SAFE_STEM.sub("_", stem)[:100] or "video"
    clean_ext = re.sub(r"[^A-Za-z0-9]", "", ext)[:8].lower()
    return f"{clean_stem}.{clean_ext}" if clean_ext else clean_stem


def _probe_streams(media_path: Path) -> dict[str, Any]:
    """Inspecciona los streams reales del archivo con ffprobe."""
    from app.infrastructure.audio.ffmpeg_adapters import (
        SubprocessFFmpegBinaryResolver,
        _run_ffprobe_json,
    )

    settings = get_settings()
    probe_bin = str(getattr(settings.ffmpeg, "ffprobe_bin", "ffprobe") or "ffprobe")
    if probe_bin in {"", "ffprobe"}:
        ok, located, _ = SubprocessFFmpegBinaryResolver().resolve("ffprobe")
        if ok:
            probe_bin = located
    data = _run_ffprobe_json(probe_bin, media_path, 60)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    video = [s for s in streams if str(s.get("codec_type")) == "video"]
    audio = [s for s in streams if str(s.get("codec_type")) == "audio"]
    first_audio = audio[0] if audio else {}
    return {
        "container": str(fmt.get("format_name") or ""),
        "duration_ms": int(round(float(fmt.get("duration") or 0) * 1000)),
        "video_streams": len(video),
        "audio_streams": len(audio),
        "video_codec": str(video[0].get("codec_name") or "") if video else "",
        "audio_codec": str(first_audio.get("codec_name") or ""),
        "audio_sample_rate": int(first_audio.get("sample_rate") or 0),
        "audio_channels": int(first_audio.get("channels") or 0),
    }


def _probe_duration_ms(media_path: Path) -> int:
    """Duración real vía ffprobe. 0 si no se puede determinar."""
    try:
        from app.infrastructure.audio.ffmpeg_adapters import (
            SubprocessFFmpegBinaryResolver,
            _run_ffprobe_json,
        )

        settings = get_settings()
        probe_bin = str(getattr(settings.ffmpeg, "ffprobe_bin", "ffprobe") or "ffprobe")
        if probe_bin in {"", "ffprobe"}:
            ok, located, _ = SubprocessFFmpegBinaryResolver().resolve("ffprobe")
            if ok:
                probe_bin = located
        data = _run_ffprobe_json(probe_bin, media_path, 60)
        fmt = data.get("format") or {}
        raw = fmt.get("duration")
        if raw is not None:
            return max(0, int(round(float(raw) * 1000)))
        best = 0.0
        for stream in data.get("streams") or []:
            if str(stream.get("codec_type")) == "video" and stream.get("duration") is not None:
                best = max(best, float(stream["duration"]))
        return max(0, int(round(best * 1000)))
    except Exception:  # noqa: BLE001
        return 0


def _write_thumbnail(media_path: Path, target: Path) -> bool:
    """Extrae un fotograma como JPEG. Devuelve False si no es posible."""
    try:
        from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver

        ok, ffmpeg_bin, _ = SubprocessFFmpegBinaryResolver().resolve("ffmpeg")
        if not ok:
            return False
        import subprocess

        target.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", "1", "-i", str(media_path),
                "-frames:v", "1", "-vf", "scale=480:-2",
                "-q:v", "4", str(target),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def _raise_engine_error(prefix: str, exc: Exception) -> None:
    """Convierte un fallo del motor en un HTTPException con detalle legible."""
    from app.core.exceptions import EngineBaseError

    detail = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, EngineBaseError):
        raise HTTPException(status_code=422, detail=f"{prefix}: {detail}") from exc
    raise HTTPException(status_code=500, detail=f"{prefix}: {detail}") from exc


def _ensure_project_metadata(conn: Any, input_root: Path, project: Any) -> Any:
    """Completa duración y miniatura de un proyecto que aún no las tenga.

    Los proyectos subidos antes de existir esta función tienen `duration_ms=0`
    y `thumbnail_path=""`. Se rellenan una sola vez, de forma perezosa, y
    cualquier fallo se ignora: leer la lista de proyectos nunca debe romperse.
    """
    keys = project.keys()
    duration_ms = int(project["duration_ms"] or 0) if "duration_ms" in keys else 0
    thumbnail_path = str(project["thumbnail_path"] or "") if "thumbnail_path" in keys else ""
    thumb_ok = bool(thumbnail_path) and Path(thumbnail_path).is_file()
    if duration_ms > 0 and thumb_ok:
        return project

    try:
        media = Path(project["media_path"]).resolve()
        if not str(media).startswith(str(input_root.resolve())) or not media.is_file():
            return project
    except (OSError, ValueError, TypeError):
        return project

    new_duration = duration_ms
    new_thumbnail = thumbnail_path if thumb_ok else ""
    try:
        if new_duration <= 0:
            new_duration = _probe_duration_ms(media)
        if not new_thumbnail:
            candidate = media.parent / "thumbnail.jpg"
            if _write_thumbnail(media, candidate):
                new_thumbnail = str(candidate)
    except Exception:  # noqa: BLE001
        return project

    if new_duration == duration_ms and new_thumbnail == thumbnail_path:
        return project
    try:
        db.run(
            conn,
            "UPDATE projects SET duration_ms=?, thumbnail_path=?, updated_at=? WHERE id=?",
            (new_duration, new_thumbnail, _now(), project["id"]),
        )
        return db.one(conn, "SELECT * FROM projects WHERE id=?", (project["id"],)) or project
    except Exception:  # noqa: BLE001
        return project


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
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    app.state.conn = conn
    app.state.paths = paths
    app.state.login_limit = RateLimiter(10, 60)
    app.state.register_limit = RateLimiter(5, 60)
    app.state.forgot_limit = RateLimiter(5, 300)
    app.state.worker_enabled = os.environ.get("IA_DISABLE_WORKER") != "1"
    if app.state.worker_enabled:
        worker.start(conn)
        # Un reinicio no debe dejar jobs eternamente en PROCESSING.
        worker.reconcile_orphans(conn)

    def client_key(request: Request) -> str:
        return request.client.host if request.client else "local"

    def user_from_cookie(token: str | None) -> sqlite3_row:
        if not token:
            raise HTTPException(status_code=401, detail="No autenticado")
        row = db.one(
            conn,
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash=? AND sessions.expires_at > ?
            """,
            (token_hash(token), _now()),
        )
        if row is None:
            raise HTTPException(status_code=401, detail="Sesión inválida")
        return row

    def set_session(response: Response, user_id: str, remember: bool) -> None:
        token = new_token()
        hours = 24 * 14 if remember else 12
        expires = datetime.now(timezone.utc) + timedelta(hours=hours)
        db.run(
            conn,
            "INSERT INTO sessions (token_hash, user_id, expires_at, remember) VALUES (?, ?, ?, ?)",
            (token_hash(token), user_id, expires.isoformat(), int(remember)),
        )
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            samesite="lax",
            max_age=hours * 3600,
            path="/",
        )

    @app.post("/api/auth/register")
    async def register(request: Request, response: Response) -> dict[str, Any]:
        if not app.state.register_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiados registros")
        body = await request.json()
        name = str(body.get("name") or "").strip()
        email = str(body.get("email") or "").strip().lower()
        password = str(body.get("password") or "")
        confirm = str(body.get("confirm") or "")
        if len(name) < 2 or "@" not in email or len(password) < 8 or password != confirm:
            raise HTTPException(status_code=422, detail="Datos de registro inválidos")
        if db.one(conn, "SELECT id FROM users WHERE email=?", (email,)):
            raise HTTPException(status_code=409, detail="El email ya está registrado")
        user_id = str(uuid.uuid4())
        db.run(
            conn,
            "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, email, hash_password(password), _now()),
        )
        set_session(response, user_id, False)
        return {"id": user_id, "name": name, "email": email}

    @app.post("/api/auth/login")
    async def login(request: Request, response: Response) -> dict[str, Any]:
        if not app.state.login_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiados intentos")
        body = await request.json()
        email = str(body.get("email") or "").strip().lower()
        password = str(body.get("password") or "")
        remember = bool(body.get("remember"))
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if row is None or not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
        set_session(response, row["id"], remember)
        return {"id": row["id"], "name": row["name"], "email": row["email"]}

    @app.post("/api/auth/logout")
    def logout(response: Response, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        if ia_session:
            db.run(conn, "DELETE FROM sessions WHERE token_hash=?", (token_hash(ia_session),))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.post("/api/auth/forgot-password")
    async def forgot(request: Request) -> dict[str, str]:
        if not app.state.forgot_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiadas solicitudes")
        body = await request.json()
        email = str(body.get("email") or "").strip().lower()
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if row is not None:
            token = new_token()
            expires = datetime.now(timezone.utc) + timedelta(hours=1)
            db.run(
                conn,
                "INSERT INTO reset_tokens (token_hash, user_id, expires_at, used) VALUES (?, ?, ?, 0)",
                (token_hash(token), row["id"], expires.isoformat()),
            )
            paths["mail"].mkdir(parents=True, exist_ok=True)
            (paths["mail"] / f"{token_hash(token)}.txt").write_text(
                f"to={email}\ntoken={token}\nexpires={expires.isoformat()}\n",
                encoding="utf-8",
            )
        return {"status": "sent"}

    @app.post("/api/auth/reset-password")
    async def reset_password(request: Request) -> dict[str, bool]:
        body = await request.json()
        token = str(body.get("token") or "")
        password = str(body.get("password") or "")
        confirm = str(body.get("confirm") or "")
        if len(password) < 8 or password != confirm or not token:
            raise HTTPException(status_code=422, detail="Contraseña inválida")
        row = db.one(
            conn,
            "SELECT * FROM reset_tokens WHERE token_hash=? AND used=0 AND expires_at > ?",
            (token_hash(token), _now()),
        )
        if row is None:
            raise HTTPException(status_code=400, detail="Token inválido o vencido")
        db.run(conn, "UPDATE users SET password_hash=? WHERE id=?", (hash_password(password), row["user_id"]))
        db.run(conn, "UPDATE reset_tokens SET used=1 WHERE token_hash=?", (token_hash(token),))
        db.run(conn, "DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        return {"id": row["id"], "name": row["name"], "email": row["email"], "created_at": row["created_at"]}

    @app.post("/api/auth/change-email")
    async def change_email(request: Request, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        body = await request.json()
        email = str(body.get("email") or "").strip().lower()
        password = str(body.get("password") or "")
        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(status_code=422, detail="Correo inválido")
        if not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        taken = db.one(conn, "SELECT id FROM users WHERE email=? AND id<>?", (email, row["id"]))
        if taken is not None:
            raise HTTPException(status_code=409, detail="El email ya está registrado")
        db.run(conn, "UPDATE users SET email=? WHERE id=?", (email, row["id"]))
        return {"email": email}

    @app.post("/api/auth/change-password")
    async def change_password(request: Request, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        row = user_from_cookie(ia_session)
        body = await request.json()
        current = str(body.get("current") or "")
        password = str(body.get("password") or "")
        confirm = str(body.get("confirm") or "")
        if not verify_password(current, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        if len(password) < 8 or password != confirm:
            raise HTTPException(status_code=422, detail="Contraseña inválida")
        db.run(conn, "UPDATE users SET password_hash=? WHERE id=?", (hash_password(password), row["id"]))
        return {"ok": True}

    @app.patch("/api/users/me")
    async def update_me(request: Request, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        body = await request.json()
        if "name" not in body:
            raise HTTPException(status_code=422, detail="Nada que actualizar")
        name = str(body.get("name") or "").strip()
        if len(name) < 2 or len(name) > MAX_NAME:
            raise HTTPException(status_code=422, detail=f"El nombre debe tener entre 2 y {MAX_NAME} caracteres")
        db.run(conn, "UPDATE users SET name=? WHERE id=?", (name, row["id"]))
        refreshed = db.one(conn, "SELECT * FROM users WHERE id=?", (row["id"],))
        return {
            "id": refreshed["id"],
            "name": refreshed["name"],
            "email": refreshed["email"],
            "created_at": refreshed["created_at"],
        }

    @app.get("/api/system/preflight")
    def system_preflight(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        """Diagnóstico de preparación del motor (solo lectura)."""
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

    @app.get("/api/videos")
    def list_videos(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        projects = db.many(conn, "SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC", (row["id"],))
        # Rellena duración/miniatura de proyectos antiguos (una sola vez).
        projects = [_ensure_project_metadata(conn, paths["input"], p) for p in projects]
        jobs = db.many(conn, "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (row["id"],))
        return {"videos": [_project(p) for p in projects], "jobs": [_job(j) for j in jobs]}

    @app.post("/api/videos")
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),
        source_language: str = Form("und"),
        target_languages: str = Form("es"),
        ia_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
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
        folder = (paths["input"] / row["id"] / project_id).resolve()
        root = paths["input"].resolve()
        if not str(folder).startswith(str(root)):
            raise HTTPException(status_code=400, detail="Ruta rechazada")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / filename

        # Streaming a disco: el archivo NUNCA se carga entero en RAM.
        written = 0
        try:
            with target.open("wb") as sink:
                while True:
                    chunk = await file.read(UPLOAD_CHUNK)
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
            await file.close()

        if written == 0:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="El archivo está vacío")

        duration_ms = _probe_duration_ms(target)
        thumbnail = folder / "thumbnail.jpg"
        has_thumb = _write_thumbnail(target, thumbnail)

        db.run(
            conn,
            """
            INSERT INTO projects (id, user_id, title, source_language, target_languages, status,
                                  media_path, created_at, duration_ms, thumbnail_path, updated_at)
            VALUES (?, ?, ?, ?, ?, 'READY', ?, ?, ?, ?, ?)
            """,
            (
                project_id, row["id"], filename, source, ",".join(targets), str(target),
                _now(), duration_ms, str(thumbnail) if has_thumb else "", _now(),
            ),
        )
        # El job queda listo pero NO arranca: el usuario pulsa «Producir».
        job_id = _create_job(conn, project_id, row["id"], "progressive", start=False)
        return {"id": project_id, "job_id": job_id, "title": filename, "duration_ms": duration_ms}

    @app.get("/api/videos/{project_id}")
    def get_video(project_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        project = _ensure_project_metadata(conn, paths["input"], project)
        jobs = db.many(conn, "SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC", (project_id,))
        return {"video": _project(project), "jobs": [_job(j) for j in jobs]}

    @app.delete("/api/videos/{project_id}")
    def delete_video(project_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        media = Path(project["media_path"])
        if media.is_file():
            media.unlink()
        db.run(conn, "DELETE FROM subtitles WHERE project_id=?", (project_id,))
        db.run(conn, "DELETE FROM jobs WHERE project_id=?", (project_id,))
        db.run(conn, "DELETE FROM projects WHERE id=?", (project_id,))
        return {"ok": True}

    @app.put("/api/videos/{project_id}")
    async def rename_video(project_id: str, request: Request, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        body = await request.json()
        title = str(body.get("title") or "").strip()
        if not title or len(title) > 180:
            raise HTTPException(status_code=422, detail="Título inválido")
        db.run(conn, "UPDATE projects SET title=? WHERE id=?", (title, project_id))
        return {"title": title}

    @app.get("/api/videos/{project_id}/media")
    def media(project_id: str, ia_session: str | None = Cookie(default=None)) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        path = Path(project["media_path"]).resolve()
        if not path.is_file() or not str(path).startswith(str(paths["input"].resolve())):
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        return FileResponse(path)

    @app.get("/api/subtitles/{project_id}/cues")
    def get_cues(
        project_id: str,
        lang: str = "",
        start_ms: int = -1,
        end_ms: int = -1,
        since_index: int = 0,
        ia_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        """Cues de un idioma, opcionalmente sólo el tramo pedido.

        Permite al reproductor pedir únicamente lo que necesita mientras el
        procesamiento sigue: `start_ms`/`end_ms` acotan por tiempo y
        `since_index` devuelve sólo lo nuevo.
        """
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        if not lang:
            raise HTTPException(status_code=422, detail="Falta el idioma")
        item = db.one(
            conn,
            "SELECT * FROM subtitles WHERE project_id=? AND language=?",
            (project_id, lang),
        )
        if item is None:
            return {"language": lang, "cues": [], "total": 0, "complete": False}
        try:
            cues = json.loads(item["cues_json"])
        except json.JSONDecodeError:
            cues = []
        total = len(cues)
        if since_index > 0:
            cues = [cue for cue in cues if int(cue.get("index", 0)) > since_index]
        if start_ms >= 0:
            cues = [cue for cue in cues if int(cue.get("end_ms", 0)) > start_ms]
        if end_ms >= 0:
            cues = [cue for cue in cues if int(cue.get("start_ms", 0)) < end_ms]
        job = db.one(
            conn,
            "SELECT status, last_cue_ms FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        return {
            "language": lang,
            "cues": cues,
            "total": total,
            "complete": bool(job is not None and job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}),
            "last_cue_ms": int(job["last_cue_ms"] or 0) if job is not None else 0,
        }

    @app.get("/api/videos/{project_id}/audio-check")
    def audio_check(project_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        """FFprobe del original y del procesado: certifica que hay audio.

        El motor nunca reescribe el vídeo; esta comprobación lo demuestra en
        lugar de darlo por supuesto.
        """
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        media = Path(str(project["media_path"])).resolve()
        root = paths["input"].resolve()
        if not str(media).startswith(str(root)) or not media.is_file():
            raise HTTPException(status_code=404, detail="Archivo original no encontrado")
        report = _probe_streams(media)
        audio_ok = report["audio_streams"] >= 1 and report["video_streams"] >= 1
        return {
            "project_id": project_id,
            "original": report,
            "preserved": audio_ok,
            "duration_ms": int(project["duration_ms"] or 0),
        }

    @app.get("/api/videos/{project_id}/thumbnail")
    def thumbnail(project_id: str, ia_session: str | None = Cookie(default=None)) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        raw = str(project["thumbnail_path"] or "")
        if not raw:
            raise HTTPException(status_code=404, detail="Sin miniatura")
        path = Path(raw).resolve()
        root = paths["input"].resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            raise HTTPException(status_code=404, detail="Sin miniatura")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/jobs")
    async def create_job(request: Request, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        body = await request.json()
        project = _own_project(conn, str(body.get("project_id") or ""), row["id"])
        mode = str(body.get("mode") or "progressive")
        return {"id": _create_job(conn, project["id"], row["id"], mode)}

    @app.post("/api/videos/{project_id}/produce")
    async def produce_video(
        project_id: str,
        request: Request,
        ia_session: str | None = Cookie(default=None),
    ) -> dict[str, Any]:
        """Arranca la producción de subtítulos en segundo plano.

        Es el botón «Producir»: separa la carga del vídeo del procesamiento.
        Si ya hay un job activo para el proyecto, devuelve ese mismo en lugar
        de lanzar otro (evita duplicar trabajo).
        """
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])

        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - cuerpo opcional
            body = {}
        mode = str(body.get("mode") or "progressive")
        if mode not in {"progressive", "full"}:
            mode = "progressive"

        # Trabajo realmente en curso: no se duplica.
        running = db.one(
            conn,
            "SELECT id FROM jobs WHERE project_id=? AND status IN "
            "('QUEUED','PROCESSING','TRANSCRIBING','TRANSLATING',"
            "'GENERATING_SUBTITLES') ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        if running is not None:
            return {"id": running["id"], "reused": True, "mode": mode}

        # Job preparado al subir pero nunca arrancado: se encola ESTE, sin crear
        # otro. Un PENDING no está en curso, así que no puede darse por activo.
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

    @app.websocket("/api/jobs/{job_id}/events")
    async def job_events(socket: WebSocket, job_id: str) -> None:
        """Eventos del job en tiempo real: progreso y subtítulos nuevos.

        No sustituye a la base de datos (fuente de verdad): si el cliente se
        conecta tarde o pierde eventos, puede reconstruir el estado con
        `/api/jobs/{id}/progress` y `/api/subtitles/{id}`.

        IMPORTANTE: toda consulta a SQLite se hace en un hilo aparte. El acceso
        es síncrono y con lock global; hacerlo aquí bloquearía el bucle de
        eventos y, con muchas conexiones, dejaría al worker sin poder avanzar.
        """
        import asyncio

        await socket.accept()
        try:
            user = await asyncio.to_thread(user_from_cookie, socket.cookies.get(COOKIE))
        except HTTPException:
            await socket.send_json({"type": "error", "message": "No autenticado"})
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
            # Estado inicial: el cliente no necesita sondear para arrancar.
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
                # Sin eventos: se mantiene viva y se detecta el fin desde la BD.
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

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        job = _own_job(conn, job_id, row["id"])
        return _job(job)

    @app.get("/api/jobs/{job_id}/progress")
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

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        _own_job(conn, job_id, row["id"])
        worker.cancel(conn, job_id)
        return {"id": job_id, "status": "CANCELLED"}

    @app.get("/api/subtitles/{project_id}")
    def get_subtitles(project_id: str, lang: str = "", ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        if lang:
            item = db.one(conn, "SELECT * FROM subtitles WHERE project_id=? AND language=?", (project_id, lang))
            rows = [item] if item else []
        else:
            rows = db.many(conn, "SELECT * FROM subtitles WHERE project_id=?", (project_id,))
        return {
            "subtitles": [
                {"language": r["language"], "cues": json.loads(r["cues_json"])} for r in rows if r is not None
            ]
        }

    @app.put("/api/subtitles/{project_id}")
    async def put_subtitles(project_id: str, request: Request, ia_session: str | None = Cookie(default=None)) -> dict[str, int]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        body = await request.json()
        language = str(body.get("language") or "")
        if language not in OFFICIAL and language != "und":
            raise HTTPException(status_code=422, detail="Idioma inválido")
        cues = [cue_to_dict(cue_from_dict(item)) for item in body.get("cues") or []]
        db.run(
            conn,
            """
            INSERT INTO subtitles (project_id, language, cues_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, language) DO UPDATE SET cues_json=excluded.cues_json, updated_at=excluded.updated_at
            """,
            (project_id, language, json.dumps(cues, ensure_ascii=False), _now()),
        )
        return {"count": len(cues)}

    def _export(project_id: str, user_id: str, lang: str, kind: str) -> Response:
        _own_project(conn, project_id, user_id)
        row = db.one(conn, "SELECT * FROM subtitles WHERE project_id=? AND language=?", (project_id, lang))
        if row is None:
            raise HTTPException(status_code=404, detail="Subtítulos no encontrados")
        cues = [cue_from_dict(item) for item in json.loads(row["cues_json"])]
        if kind == "srt":
            body, media = to_srt(cues), "application/x-subrip"
            filename = f"{project_id}.{lang}.srt"
        elif kind == "vtt":
            body, media = to_vtt(cues), "text/vtt"
            filename = f"{project_id}.{lang}.vtt"
        else:
            body, media = to_ass(cues, language=lang), "text/x-ssa"
            filename = f"{project_id}.{lang}.ass"
        return Response(content=body.encode("utf-8"), media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/api/export/{project_id}/srt")
    def export_srt(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        return _export(project_id, user_from_cookie(ia_session)["id"], lang, "srt")

    @app.get("/api/export/{project_id}/vtt")
    def export_vtt(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        return _export(project_id, user_from_cookie(ia_session)["id"], lang, "vtt")

    @app.get("/api/export/{project_id}/ass")
    def export_ass(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        return _export(project_id, user_from_cookie(ia_session)["id"], lang, "ass")

    @app.get("/api/export/{project_id}/zip")
    def export_zip(project_id: str, ia_session: str | None = Cookie(default=None)) -> Response:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        rows = db.many(conn, "SELECT * FROM subtitles WHERE project_id=?", (project_id,))
        files: dict[str, str] = {}
        for item in rows:
            cues = [cue_from_dict(c) for c in json.loads(item["cues_json"])]
            lang = item["language"]
            files[f"{lang}.srt"] = to_srt(cues)
            files[f"{lang}.vtt"] = to_vtt(cues)
            files[f"{lang}.ass"] = to_ass(cues, language=lang)
        if not files:
            raise HTTPException(status_code=404, detail="No hay subtítulos")
        return Response(content=zip_exports(files), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{project_id}.zip"'})

    @app.websocket("/api/realtime")
    async def realtime(socket: WebSocket) -> None:
        await socket.accept()
        try:
            user_from_cookie(socket.cookies.get(COOKIE))
        except HTTPException:
            await socket.send_json({"type": "error", "message": "No autenticado"})
            await socket.close()
            return
        try:
            from app.infrastructure.audio.faster_whisper_asr_adapter import FasterWhisperSmallASRAdapter

            session = RealtimeSession(
                whisper_transcriber(FasterWhisperSmallASRAdapter(compute_type="int8"))
            )
            session.translator = _live_translator(session)
        except Exception as exc:  # noqa: BLE001
            await socket.send_json({"type": "error", "message": f"ASR en vivo no disponible: {exc}"})
            await socket.close()
            return
        try:
            while True:
                payload = await socket.receive_json()
                for event in session.handle(payload):
                    await socket.send_json(event)
        except WebSocketDisconnect:
            return

    return app


def _project(row: Any) -> dict[str, Any]:
    keys = row.keys()
    duration_ms = int(row["duration_ms"] or 0) if "duration_ms" in keys else 0
    thumbnail_path = str(row["thumbnail_path"] or "") if "thumbnail_path" in keys else ""
    return {
        "id": row["id"],
        "title": row["title"],
        "source_language": row["source_language"],
        "target_languages": row["target_languages"].split(",") if row["target_languages"] else [],
        "status": row["status"],
        "created_at": row["created_at"],
        "duration_ms": duration_ms,
        "thumbnail_url": f"/api/videos/{row['id']}/thumbnail" if thumbnail_path else "",
    }


def _create_job(
    conn: Any, project_id: str, user_id: str, mode: str = "progressive", *, start: bool = True
) -> str:
    """Crea el job y, opcionalmente, lo encola.

    `mode='progressive'` (por defecto) publica subtítulos mientras procesa;
    `mode='full'` ejecuta el pipeline completo T01→T15.

    Con `start=False` el job queda listo en `PENDING` sin arrancar: es lo que
    permite separar «el vídeo está cargado» de «producir subtítulos».
    """
    if mode not in {"progressive", "full"}:
        mode = "progressive"
    job_id = str(uuid.uuid4())
    status = "PENDING" if not start else "QUEUED"
    db.run(
        conn,
        """
        INSERT INTO jobs (id, project_id, user_id, status, progress, stage, error,
                          created_at, updated_at, mode)
        VALUES (?, ?, ?, ?, 0, ?, '', ?, ?, ?)
        """,
        (job_id, project_id, user_id, status, status, _now(), _now(), mode),
    )
    if start:
        worker.enqueue(conn, job_id)
    return job_id


def _job(row: Any) -> dict[str, Any]:
    keys = row.keys()
    raw_history = str(row["stage_history"] or "[]") if "stage_history" in keys else "[]"
    try:
        history = json.loads(raw_history)
    except json.JSONDecodeError:
        history = []
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "status": row["status"],
        "progress": row["progress"],
        "stage": row["stage"],
        "error": row["error"],
        "updated_at": row["updated_at"],
        "stage_history": history,
        "mode": str(row["mode"]) if "mode" in keys and row["mode"] else "full",
        "last_cue_ms": int(row["last_cue_ms"] or 0) if "last_cue_ms" in keys else 0,
    }


def _own_project(conn: Any, project_id: str, user_id: str) -> Any:
    row = db.one(conn, "SELECT * FROM projects WHERE id=? AND user_id=?", (project_id, user_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Proyecto no encontrado")
    return row


def _own_job(conn: Any, job_id: str, user_id: str) -> Any:
    row = db.one(conn, "SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return row


def _live_translator(session: RealtimeSession):
    holder: dict[str, Any] = {}

    def _run(text: str, partial: bool) -> dict[str, str]:
        if "adapter" not in holder:
            from app.infrastructure.translation.m2m100_translation_adapter import (
                M2M100TranslatorAdapter,
            )

            holder["adapter"] = M2M100TranslatorAdapter()
        return m2m_translator(
            holder["adapter"], session.source_language, list(session.targets)
        )(text, partial)

    return _run


sqlite3_row = Any
