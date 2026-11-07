from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.core.config import PROJECT_ROOT
from app.core.config import get_settings
from app.presentation.api import db
from app.presentation.api import media_store
from app.presentation.api import worker
from app.presentation.api.realtime import RealtimeSession
from app.presentation.api.realtime import m2m_translator


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
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver
    from app.infrastructure.audio.ffmpeg_adapters import _run_ffprobe_json

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
        from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver
        from app.infrastructure.audio.ffmpeg_adapters import _run_ffprobe_json

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
