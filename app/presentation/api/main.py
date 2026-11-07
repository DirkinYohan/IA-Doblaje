from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from app.application.use_cases.build_subtitles import cue_from_dict, cue_to_dict, zip_exports
from app.core.config import PROJECT_ROOT, get_settings
from app.infrastructure.subtitles.exporters import to_ass, to_srt, to_vtt
from app.presentation.api import db, mailer, media_store, worker
from app.presentation.api.schemas import (
    ChangeEmailBody,
    ChangePasswordBody,
    CreateJobBody,
    EmailBody,
    LoginBody,
    ProduceBody,
    RegisterBody,
    RenameBody,
    RenderBody,
    ResetPasswordBody,
    SubtitlesBody,
    TokenBody,
    UpdateMeBody,
    is_valid_email,
)
from app.presentation.api.realtime import RealtimeSession, m2m_translator, whisper_transcriber
from app.presentation.api.security import RateLimiter, hash_password, new_token, token_hash, verify_password

OFFICIAL = {"es", "en", "fr", "it", "pt"}
COOKIE = "ia_session"
WS_TICKET_TTL_S = 60
UPLOAD_CHUNK = 1024 * 1024
MAX_NAME = 80
MAX_TITLE = 180
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")

_MIME_BY_EXT = {
    "mp4": "video/mp4", "m4v": "video/mp4", "mov": "video/quicktime",
    "mkv": "video/x-matroska", "webm": "video/webm", "avi": "video/x-msvideo",
    "wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4",
    "flac": "audio/flac", "ogg": "audio/ogg",
}


def _guess_mime(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _MIME_BY_EXT.get(ext, "application/octet-stream")


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
        "temp": Path(settings.paths.data_temp_dir),
    }


def _safe_filename(raw: str) -> str:
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


def _write_storyboard(media_path: Path, target: Path, *, frames: int = 14) -> bool:
    try:
        from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver

        ok, ffmpeg_bin, _ = SubprocessFFmpegBinaryResolver().resolve("ffmpeg")
        if not ok:
            return False
        import subprocess

        count = max(4, min(24, int(frames)))
        width, height = 176, 99
        filtro = (
            f"fps={count}/{max(1.0, _probe_duration_ms(media_path) / 1000):.6f},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},tile={count}x1"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(media_path),
                "-vf", filtro, "-frames:v", "1", "-q:v", "6", str(target),
            ],
            capture_output=True,
            timeout=180,
            check=False,
        )
        return completed.returncode == 0 and target.is_file() and target.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def _raise_engine_error(prefix: str, exc: Exception) -> None:
    from app.core.exceptions import EngineBaseError

    detail = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, EngineBaseError):
        raise HTTPException(status_code=422, detail=f"{prefix}: {detail}") from exc
    raise HTTPException(status_code=500, detail=f"{prefix}: {detail}") from exc


def _ensure_project_metadata(conn: Any, input_root: Path, project: Any) -> Any:
    keys = project.keys()
    duration_ms = int(project["duration_ms"] or 0) if "duration_ms" in keys else 0
    thumbnail_path = str(project["thumbnail_path"] or "") if "thumbnail_path" in keys else ""
    thumb_ok = bool(thumbnail_path) and (
        media_store.is_remote(thumbnail_path) or Path(thumbnail_path).is_file()
    )
    if duration_ms > 0 and thumb_ok:
        return project

    remote = media_store.is_remote(project["media_path"])
    media: Path | None = None
    try:
        if remote:
            media = media_store.materialize(
                project["media_path"], input_root.parent / "temporary",
                label=f"meta-{project['id']}",
            )
        else:
            media = Path(project["media_path"]).resolve()
            if not str(media).startswith(str(input_root.resolve())) or not media.is_file():
                return project
    except (OSError, ValueError, TypeError, Exception):  # noqa: BLE001
        return project

    new_duration = duration_ms
    new_thumbnail = thumbnail_path if thumb_ok else ""
    try:
        if new_duration <= 0:
            new_duration = _probe_duration_ms(media)
        if not new_thumbnail:
            candidate = Path(media).parent / "thumbnail.jpg"
            if _write_thumbnail(media, candidate):
                if remote:
                    bucket = os.environ.get("SUPABASE_BUCKET", "media")
                    key = f"{project['user_id']}/{project['id']}/thumbnail.jpg"
                    media_store.upload_bytes(candidate.read_bytes(), key, "image/jpeg")
                    new_thumbnail = media_store.make_uri(bucket, key)
                    candidate.unlink(missing_ok=True)
                else:
                    new_thumbnail = str(candidate)
    except Exception:  # noqa: BLE001
        return project
    finally:
        if remote:
            media_store.release(media, input_root.parent / "temporary")

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

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": "Datos inválidos"})

    app.state.paths = paths
    app.state.login_limit = RateLimiter(10, 60)
    app.state.register_limit = RateLimiter(5, 60)
    app.state.forgot_limit = RateLimiter(5, 300)
    app.state.verify_limit = RateLimiter(5, 600)
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

    live_warm = {"started": os.environ.get("IA_LIVE_PRELOAD", "1") != "1"}

    def _warm_live_models() -> None:
        if live_warm["started"]:
            return
        live_warm["started"] = True
        import threading

        from app.presentation.api import live

        threading.Thread(
            target=live.warm, kwargs={"translation": True}, name="live-preload", daemon=True
        ).start()

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

    ws_tickets: dict[str, tuple[str, float]] = {}
    ws_tickets_lock = threading.Lock()

    def issue_ws_ticket(user_id: str) -> str:
        ticket = new_token()
        now = time.monotonic()
        with ws_tickets_lock:
            for key in [k for k, (_u, exp) in ws_tickets.items() if exp < now]:
                ws_tickets.pop(key, None)
            ws_tickets[ticket] = (user_id, now + WS_TICKET_TTL_S)
        return ticket

    def user_from_socket(socket: WebSocket) -> sqlite3_row:
        ticket = socket.query_params.get("ticket") or ""
        if ticket:
            with ws_tickets_lock:
                entry = ws_tickets.pop(ticket, None)
            if entry is None or entry[1] < time.monotonic():
                raise HTTPException(status_code=401, detail="Ticket inválido o caducado")
            row = db.one(conn, "SELECT * FROM users WHERE id=?", (entry[0],))
            if row is None:
                raise HTTPException(status_code=401, detail="Sesión inválida")
            return row
        return user_from_cookie(socket.cookies.get(COOKIE))

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

    def public_user(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "created_at": row["created_at"],
            "email_verified": bool(row["email_verified_at"]),
        }

    def send_verification(user_id: str, name: str, email: str, locale: str) -> None:
        token = new_token()
        expires = datetime.now(timezone.utc) + timedelta(hours=48)
        db.run(
            conn,
            "INSERT INTO email_tokens (token_hash, user_id, email, expires_at, used) VALUES (?, ?, ?, ?, 0)",
            (token_hash(token), user_id, email, expires.isoformat()),
        )
        mail = mailer.compose(
            "verify", to=email, name=name, token=token, expires=expires.isoformat(), locale=locale,
            link=f"{mailer.public_url()}/verify-email?token={token}",
        )
        mailer.send_in_background(app.state.mailer, mail)

    @app.post("/api/auth/register")
    def register(request: Request, body: RegisterBody) -> dict[str, Any]:
        if not app.state.register_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiados registros")
        name = body.name.strip()
        email = body.email.strip().lower()
        password = body.password
        if not 2 <= len(name) <= MAX_NAME or not is_valid_email(email) or len(password) < 8 or password != body.confirm:
            raise HTTPException(status_code=422, detail="Datos de registro inválidos")
        locale = mailer.locale_of(body.locale)
        existing = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if existing is not None and existing["email_verified_at"]:
            raise HTTPException(status_code=409, detail="El email ya está registrado")
        if existing is not None:
            send_verification(existing["id"], existing["name"], email, locale)
            return {"status": "verify", "email": email}
        user_id = str(uuid.uuid4())
        db.run(
            conn,
            "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, email, hash_password(password), _now()),
        )
        send_verification(user_id, name, email, locale)
        return {"status": "verify", "email": email}

    @app.post("/api/auth/login")
    def login(request: Request, response: Response, body: LoginBody) -> dict[str, Any]:
        if not app.state.login_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiados intentos")
        email = body.email.strip().lower()
        password = body.password
        remember = body.remember
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if row is None or not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
        if not row["email_verified_at"]:
            raise HTTPException(status_code=403, detail="Confirma tu correo para iniciar sesión")
        set_session(response, row["id"], remember)
        return public_user(row)

    @app.post("/api/auth/logout")
    def logout(response: Response, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        if ia_session:
            db.run(conn, "DELETE FROM sessions WHERE token_hash=?", (token_hash(ia_session),))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.post("/api/auth/forgot-password")
    def forgot(request: Request, body: EmailBody) -> dict[str, str]:
        if not app.state.forgot_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiadas solicitudes")
        email = body.email.strip().lower()
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if row is not None:
            token = new_token()
            expires = datetime.now(timezone.utc) + timedelta(hours=1)
            db.run(
                conn,
                "INSERT INTO reset_tokens (token_hash, user_id, expires_at, used) VALUES (?, ?, ?, 0)",
                (token_hash(token), row["id"], expires.isoformat()),
            )
            mail = mailer.compose(
                "reset", to=email, name=row["name"], token=token, expires=expires.isoformat(),
                locale=mailer.locale_of(body.locale),
                link=f"{mailer.public_url()}/reset-password?token={token}",
            )
            mailer.send_in_background(app.state.mailer, mail)
        return {"status": "sent"}

    @app.post("/api/auth/verify-email")
    def verify_email(body: TokenBody) -> dict[str, Any]:
        token = body.token
        row = db.one(
            conn,
            "SELECT * FROM email_tokens WHERE token_hash=? AND used=0 AND expires_at > ?",
            (token_hash(token), _now()),
        ) if token else None
        user = db.one(conn, "SELECT * FROM users WHERE id=?", (row["user_id"],)) if row else None
        if row is None or user is None:
            raise HTTPException(status_code=400, detail="Enlace inválido o vencido")
        email = row["email"]
        if email != user["email"]:
            taken = db.one(conn, "SELECT id FROM users WHERE email=? AND id<>?", (email, user["id"]))
            if taken is not None:
                raise HTTPException(status_code=409, detail="El email ya está registrado")
        db.run(
            conn,
            "UPDATE users SET email=?, email_verified_at=? WHERE id=?",
            (email, user["email_verified_at"] or _now(), user["id"]),
        )
        db.run(conn, "UPDATE email_tokens SET used=1 WHERE user_id=?", (user["id"],))
        return {"ok": True, "email": email}

    @app.post("/api/auth/resend-verification")
    def resend_verification(request: Request, body: EmailBody) -> dict[str, str]:
        email = body.email.strip().lower()
        if not app.state.verify_limit.allow(client_key(request)) or not app.state.verify_limit.allow(f"mail:{email}"):
            raise HTTPException(status_code=429, detail="Demasiadas solicitudes")
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,)) if email else None
        if row is not None and not row["email_verified_at"]:
            send_verification(row["id"], row["name"], row["email"], mailer.locale_of(body.locale))
        return {"status": "sent"}

    @app.post("/api/auth/reset-password")
    def reset_password(body: ResetPasswordBody) -> dict[str, bool]:
        token = body.token
        password = body.password
        if len(password) < 8 or password != body.confirm or not token:
            raise HTTPException(status_code=422, detail="Contraseña inválida")
        row = db.one(
            conn,
            "SELECT * FROM reset_tokens WHERE token_hash=? AND used=0 AND expires_at > ?",
            (token_hash(token), _now()),
        )
        if row is None:
            raise HTTPException(status_code=400, detail="Token inválido o vencido")
        db.run(conn, "UPDATE users SET password_hash=? WHERE id=?", (hash_password(password), row["user_id"]))
        db.run(
            conn, "UPDATE users SET email_verified_at=? WHERE id=? AND email_verified_at=''", (_now(), row["user_id"])
        )
        db.run(conn, "UPDATE reset_tokens SET used=1 WHERE token_hash=?", (token_hash(token),))
        db.run(conn, "DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        return {"ok": True}

    @app.post("/api/ws-ticket")
    def ws_ticket(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        return {"ticket": issue_ws_ticket(row["id"]), "expires_in": WS_TICKET_TTL_S}

    @app.get("/api/auth/me")
    def me(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        return public_user(row)

    @app.post("/api/auth/change-email")
    def change_email(body: ChangeEmailBody, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        email = body.email.strip().lower()
        password = body.password
        if not is_valid_email(email):
            raise HTTPException(status_code=422, detail="Correo inválido")
        if not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        taken = db.one(conn, "SELECT id FROM users WHERE email=? AND id<>?", (email, row["id"]))
        if taken is not None:
            raise HTTPException(status_code=409, detail="El email ya está registrado")
        if email == row["email"]:
            return {"email": email, "pending_email": ""}
        send_verification(row["id"], row["name"], email, mailer.locale_of(body.locale))
        return {"email": row["email"], "pending_email": email}

    @app.post("/api/auth/change-password")
    def change_password(body: ChangePasswordBody, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        row = user_from_cookie(ia_session)
        password = body.password
        if not verify_password(body.current, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        if len(password) < 8 or password != body.confirm:
            raise HTTPException(status_code=422, detail="Contraseña inválida")
        db.run(conn, "UPDATE users SET password_hash=? WHERE id=?", (hash_password(password), row["id"]))
        return {"ok": True}

    @app.patch("/api/users/me")
    def update_me(body: UpdateMeBody, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        if body.name is None:
            raise HTTPException(status_code=422, detail="Nada que actualizar")
        name = body.name.strip()
        if len(name) < 2 or len(name) > MAX_NAME:
            raise HTTPException(status_code=422, detail=f"El nombre debe tener entre 2 y {MAX_NAME} caracteres")
        db.run(conn, "UPDATE users SET name=? WHERE id=?", (name, row["id"]))
        return public_user(db.one(conn, "SELECT * FROM users WHERE id=?", (row["id"],)))

    @app.get("/api/system/preflight")
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

    @app.get("/api/videos")
    def list_videos(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        _warm_live_models()
        projects = db.many(conn, "SELECT * FROM projects WHERE user_id=? ORDER BY created_at DESC", (row["id"],))
        projects = [_ensure_project_metadata(conn, paths["input"], p) for p in projects]
        jobs = db.many(conn, "SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT 20", (row["id"],))
        return {"videos": [_project(p) for p in projects], "jobs": [_job(j) for j in jobs]}

    @app.post("/api/videos")
    def upload_video(
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

        if use_storage:
            media_key = f"{row['id']}/{project_id}/{filename}"
            thumb_key = f"{row['id']}/{project_id}/thumbnail.jpg"
            thumb_temp = paths["temp"] / "uploads" / f"{project_id}-thumbnail.jpg"
            has_thumb = _write_thumbnail(target, thumb_temp)
            try:
                media_store.upload_file(target, media_key, _guess_mime(filename))
                if has_thumb:
                    media_store.upload_bytes(
                        thumb_temp.read_bytes(), thumb_key, "image/jpeg"
                    )
            except Exception as exc:  # noqa: BLE001
                target.unlink(missing_ok=True)
                thumb_temp.unlink(missing_ok=True)
                if "413" in str(exc):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "El almacenamiento rechaza el archivo por tamaño "
                            "(límite del plan de Supabase). Prueba con un vídeo "
                            "más ligero o amplía el plan."
                        ),
                    ) from exc
                _raise_engine_error("No se pudo subir el archivo al almacenamiento", exc)
            finally:
                thumb_temp.unlink(missing_ok=True)

            try:
                suffix = target.suffix
                pending = media_store.pending_path(paths["temp"], project_id, suffix)
                pending.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(pending))
            except Exception:  # noqa: BLE001
                target.unlink(missing_ok=True)

            media_value = media_store.make_uri(
                os.environ.get("SUPABASE_BUCKET", "media"), media_key
            )
            thumb_value = (
                media_store.make_uri(os.environ.get("SUPABASE_BUCKET", "media"), thumb_key)
                if has_thumb
                else ""
            )
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
        job_id = _create_job(conn, project_id, row["id"], "progressive", start=app.state.worker_enabled)
        uploaded = db.one(conn, "SELECT * FROM projects WHERE id=?", (project_id,))
        if uploaded is not None and app.state.worker_enabled:
            from app.presentation.api import live

            live.start_for_project(conn, uploaded, job_id=job_id)
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
        from app.presentation.api import live

        running = live.get_runner(project_id)
        if running is not None:
            running.stop()
        from app.presentation.api import render

        render.forget(paths["output"], project_id)

        media_path = project["media_path"]
        if media_store.is_remote(media_path):
            key = media_store.object_key(media_path)
            if key:
                try:
                    media_store.delete(key)
                except Exception:  # noqa: BLE001 - borrar el proyecto manda
                    pass
        else:
            media = Path(media_path)
            if media.is_file():
                media.unlink()

        thumb_path = str(project["thumbnail_path"] or "")
        if media_store.is_remote(thumb_path):
            thumb_key = media_store.object_key(thumb_path)
            if thumb_key:
                try:
                    media_store.delete(thumb_key)
                except Exception:  # noqa: BLE001
                    pass

        db.run(conn, "DELETE FROM subtitles WHERE project_id=?", (project_id,))
        db.run(conn, "DELETE FROM jobs WHERE project_id=?", (project_id,))
        db.run(conn, "DELETE FROM projects WHERE id=?", (project_id,))
        return {"ok": True}

    @app.put("/api/videos/{project_id}")
    def rename_video(project_id: str, body: RenameBody, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        title = body.title.strip()
        if not title or len(title) > 180:
            raise HTTPException(status_code=422, detail="Título inválido")
        db.run(conn, "UPDATE projects SET title=? WHERE id=?", (title, project_id))
        return {"title": title}

    @app.get("/api/videos/{project_id}/media")
    def media(project_id: str, ia_session: str | None = Cookie(default=None)) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        media_path = project["media_path"]

        if media_store.is_remote(media_path):
            url = media_store.signed_url(media_path, expires_in=3600)
            if url:
                return RedirectResponse(url, status_code=307)
            raise HTTPException(status_code=503, detail="Almacenamiento no disponible")

        path = Path(media_path).resolve()
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

    @app.get("/api/videos/{project_id}/thumbnail")
    def thumbnail(project_id: str, ia_session: str | None = Cookie(default=None)) -> FileResponse:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, project_id, row["id"])
        raw = str(project["thumbnail_path"] or "")
        if not raw:
            raise HTTPException(status_code=404, detail="Sin miniatura")

        if media_store.is_remote(raw):
            url = media_store.signed_url(raw, expires_in=3600)
            if not url:
                raise HTTPException(status_code=503, detail="Almacenamiento no disponible")
            return RedirectResponse(url, status_code=307)

        path = Path(raw).resolve()
        root = paths["input"].resolve()
        if not str(path).startswith(str(root)) or not path.is_file():
            raise HTTPException(status_code=404, detail="Sin miniatura")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/videos/{project_id}/storyboard")
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

    @app.post("/api/jobs")
    def create_job(body: CreateJobBody, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        project = _own_project(conn, body.project_id, row["id"])
        mode = body.mode or "progressive"
        return {"id": _create_job(conn, project["id"], row["id"], mode)}

    @app.post("/api/videos/{project_id}/produce")
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

    @app.websocket("/api/jobs/{job_id}/events")
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
        job = db.one(
            conn,
            "SELECT status FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1",
            (project_id,),
        )
        processing = job is not None and _job_is_running(str(job["status"]))
        return {
            "subtitles": [_track_view(project_id, r, processing=processing) for r in rows if r is not None]
        }

    @app.put("/api/subtitles/{project_id}")
    def put_subtitles(project_id: str, body: SubtitlesBody, ia_session: str | None = Cookie(default=None)) -> dict[str, int]:
        row = user_from_cookie(ia_session)
        _own_project(conn, project_id, row["id"])
        language = body.language
        if language not in OFFICIAL and language != "und":
            raise HTTPException(status_code=422, detail="Idioma inválido")
        try:
            cues = [cue_to_dict(cue_from_dict(item)) for item in body.cues]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Subtítulos inválidos") from exc
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

    def _render_track(project_id: str, user_id: str, lang: str) -> tuple[Any, list[Any], str]:
        project = _own_project(conn, project_id, user_id)
        row = db.one(conn, "SELECT * FROM subtitles WHERE project_id=? AND language=?", (project_id, lang))
        if row is None:
            raise HTTPException(status_code=404, detail="Subtítulos no encontrados")
        cues = [cue_from_dict(item) for item in json.loads(row["cues_json"])]
        if not cues:
            raise HTTPException(status_code=409, detail="La pista todavía no tiene subtítulos")
        return project, cues, str(row["updated_at"] or "")

    @app.post("/api/videos/{project_id}/render")
    def start_render(
        project_id: str, body: RenderBody | None = None, ia_session: str | None = Cookie(default=None)
    ) -> dict[str, Any]:
        from app.presentation.api import live, render

        row = user_from_cookie(ia_session)
        lang = (body.lang if body else "").strip().lower()
        if not lang:
            raise HTTPException(status_code=422, detail="Falta el idioma")
        project, cues, stamp = _render_track(project_id, row["id"], lang)
        return render.start(
            output_root=paths["output"], project=project, language=lang, cues=cues,
            stamp=stamp, source_of=live.media_source,
        )

    @app.get("/api/videos/{project_id}/render")
    def render_status(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        from app.presentation.api import render

        row = user_from_cookie(ia_session)
        _project, _cues, stamp = _render_track(project_id, row["id"], lang)
        return render.status(paths["output"], project_id, lang, stamp)

    @app.get("/api/videos/{project_id}/render/file")
    def render_file(project_id: str, lang: str, ia_session: str | None = Cookie(default=None)) -> Response:
        from urllib.parse import quote

        from app.presentation.api import render

        row = user_from_cookie(ia_session)
        project, _cues, _stamp = _render_track(project_id, row["id"], lang)
        video = render.ready_file(paths["output"], project_id, lang)
        if video is None:
            raise HTTPException(status_code=404, detail="El vídeo todavía no está listo")
        base = Path(str(project["title"] or "video")).stem.replace("_", " ").strip() or "video"
        name = f"{base} ({lang}).mp4"
        ascii_name = name.encode("ascii", "ignore").decode() or f"video-{lang}.mp4"
        return FileResponse(
            video,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"
            },
        )

    @app.websocket("/api/videos/{project_id}/live")
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

    @app.websocket("/api/realtime")
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
            from app.infrastructure.audio.faster_whisper_asr_adapter import FasterWhisperSmallASRAdapter

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


_SETTLED_STATUSES = {"PENDING", "COMPLETED", "FAILED", "CANCELLED", "PARTIAL", "SUCCESS"}


def _job_is_running(status: str) -> bool:
    return status.upper() not in _SETTLED_STATUSES


def _track_view(project_id: str, row: Any, *, processing: bool) -> dict[str, Any]:
    language = str(row["language"])
    try:
        raw = json.loads(row["cues_json"])
    except (TypeError, json.JSONDecodeError):
        raw = []
    cues = []
    for position, cue in enumerate(raw if isinstance(raw, list) else [], start=1):
        index = int(cue.get("index") or position)
        cues.append(
            {
                **cue,
                "id": str(cue.get("id") or f"{project_id}:{language}:{index}"),
                "project_id": project_id,
                "index": index,
                "language": language,
                "start_ms": int(cue.get("start_ms", 0)),
                "end_ms": int(cue.get("end_ms", 0)),
                "text": str(cue.get("text", "")),
                "state": str(cue.get("state") or "final"),
            }
        )
    keys = row.keys() if hasattr(row, "keys") else []
    covered = int(row["covered_ms"] or 0) if "covered_ms" in keys else 0
    return {
        "project_id": project_id,
        "language": language,
        "status": "processing" if processing else "complete",
        "covered_ms": max(covered, max((cue["end_ms"] for cue in cues), default=0)),
        "cues": cues,
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
    import threading

    holder: dict[str, Any] = {}
    lock = threading.Lock()

    def _run(text: str, partial: bool) -> dict[str, str]:
        with lock:
            if "adapter" not in holder:
                from app.infrastructure.translation.m2m100_translation_adapter import (
                    M2M100TranslatorAdapter,
                )

                holder["adapter"] = M2M100TranslatorAdapter()
            return m2m_translator(
                holder["adapter"], session.source_language, list(session.targets)
            )(text, partial)

    def _warm() -> None:
        try:
            _run("ok", False)
        except Exception:  # noqa: BLE001 - optimización: la traducción real informará
            pass

    _run.warm = _warm  # type: ignore[attr-defined]
    return _run


sqlite3_row = Any
