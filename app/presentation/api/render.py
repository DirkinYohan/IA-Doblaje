from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("ia.render")

_RENDER_SLOT = threading.Semaphore(1)
ACTIVE = ("queued", "running")


def _view(state: str, progress: int, error: str, language: str) -> dict[str, Any]:
    return {"state": state, "progress": progress, "error": error, "language": language}


def _row(conn: Any, project_id: str, language: str) -> Any:
    from app.presentation.api import db

    return db.one(conn, "SELECT * FROM renders WHERE project_id=? AND language=?", (project_id, language))


def _is_dead(row: Any) -> bool:
    from app.presentation.api import cluster

    return row["state"] in ACTIVE and float(row["heartbeat_at"] or 0) < cluster.stale_before()


def stored_file(conn: Any, project_id: str, language: str, stamp: str | None = None) -> str | None:
    row = _row(conn, project_id, language)
    if row is None or row["state"] != "done" or not row["storage_key"]:
        return None
    return str(row["storage_key"]) if stamp is None or row["stamp"] == stamp else None


def render_dir(output_root: Path, project_id: str) -> Path:
    return Path(output_root) / "renders" / project_id


def _paths(output_root: Path, project_id: str, language: str) -> tuple[Path, Path]:
    folder = render_dir(output_root, project_id)
    return folder / f"{language}.mp4", folder / f"{language}.json"


def ready_file(output_root: Path, project_id: str, language: str, stamp: str | None = None) -> Path | None:
    video, meta = _paths(output_root, project_id, language)
    if not video.is_file() or not meta.is_file():
        return None
    try:
        saved = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return video if stamp is None or saved.get("stamp") == stamp else None


def status(conn: Any, output_root: Path, project_id: str, language: str, stamp: str) -> dict[str, Any]:
    row = _row(conn, project_id, language)
    if row is not None and _is_dead(row):
        return _view("failed", int(row["progress"]), "El proceso que preparaba el vídeo se detuvo", language)
    if row is not None and row["state"] in (*ACTIVE, "failed"):
        return _view(row["state"], int(row["progress"]), row["error"], language)
    if ready_file(output_root, project_id, language, stamp) or stored_file(conn, project_id, language, stamp):
        return _view("done", 100, "", language)
    if ready_file(output_root, project_id, language) or stored_file(conn, project_id, language):
        return _view("stale", 100, "", language)
    return _view("idle", 0, "", language)


def _resolve(binary: str) -> str:
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver

    ok, path, _version = SubprocessFFmpegBinaryResolver().resolve(binary)
    if not ok:
        raise RuntimeError(path)
    return path


def probe(source: Any) -> tuple[int, int, int]:
    done = subprocess.run(
        [
            _resolve("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height:format=duration", "-of", "json", str(source),
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"no se pudo leer el vídeo: {done.stderr.strip()[:200]}")
    data = json.loads(done.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("el archivo no tiene pista de vídeo")
    width, height = int(streams[0].get("width") or 0), int(streams[0].get("height") or 0)
    duration_ms = int(float((data.get("format") or {}).get("duration") or 0) * 1000)
    return width, height, duration_ms


_HARDWARE_ENCODERS: list[tuple[str, list[str]]] = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "22", "-b:v", "0"]),
    ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"]),
    ("h264_qsv", ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "22"]),
]
_SOFTWARE_ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]

_MAX_PARTS = 4
_MIN_PART_MS = 20_000


@lru_cache(maxsize=1)
def video_encoder() -> tuple[str, ...]:
    forced = os.environ.get("IA_RENDER_ENCODER", "").strip().lower()
    candidates = [enc for enc in _HARDWARE_ENCODERS if not forced or enc[0] == forced]
    ffmpeg = _resolve("ffmpeg")
    for name, args in candidates:
        try:
            done = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 "color=c=gray:s=320x240:d=0.2", *args, "-pix_fmt", "yuv420p", "-f", "null", "-"],
                capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0:
            log.info("render: codificador %s", name)
            return tuple(args)
    log.info("render: codificador libx264")
    return tuple(_SOFTWARE_ENCODER)


def local_source(source: Any, cache: Path, on_progress: Callable[[float], None] | None = None) -> Path:
    if not isinstance(source, str) or not source.startswith(("http://", "https://")):
        return Path(str(source))
    if cache.is_file() and cache.stat().st_size > 0:
        return cache
    import httpx

    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_name(f"{cache.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        with httpx.stream("GET", source, timeout=httpx.Timeout(60, read=120), follow_redirects=True) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            received = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 20):
                    handle.write(chunk)
                    received += len(chunk)
                    if total and on_progress:
                        on_progress(min(1.0, received / total))
        partial.replace(cache)
    finally:
        partial.unlink(missing_ok=True)
    return cache


def _has_audio(source: Any) -> bool:
    done = subprocess.run(
        [_resolve("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(source)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    return bool(done.stdout.strip())


def _parts(duration_ms: int) -> list[tuple[float, float | None]]:
    count = max(1, min(_MAX_PARTS, duration_ms // _MIN_PART_MS)) if duration_ms > 0 else 1
    step = duration_ms / 1000 / count
    return [(round(i * step, 3), round((i + 1) * step, 3) if i < count - 1 else None) for i in range(count)]


def _run_ffmpeg(command: list[str], cwd: Path, log_path: Path, on_time: Callable[[int], None]) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file, subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=log_file,
        text=True, encoding="utf-8", errors="replace",
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                on_time(int(value))
        code = process.wait()
    if code != 0:
        detail = log_path.read_text(encoding="utf-8", errors="replace").strip()[-300:]
        raise RuntimeError(f"FFmpeg falló: {detail}")


def burn(
    source: Any,
    ass_text: str,
    target: Path,
    *,
    duration_ms: int,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    work = target.parent / f".work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        (work / "subs.ass").write_text(ass_text, encoding="utf-8")
        ffmpeg = _resolve("ffmpeg")
        encoder = list(video_encoder())
        parts = _parts(duration_ms)
        done_us = [0] * len(parts)
        lock = threading.Lock()

        def report(index: int, value: int) -> None:
            with lock:
                done_us[index] = value
                if duration_ms > 0 and on_progress:
                    on_progress(min(98, int(sum(done_us) / 1000 * 98 / duration_ms)))

        def encode_part(index: int, begin: float, end: float | None) -> None:
            window = ["-ss", f"{begin:.3f}"] + (["-to", f"{end:.3f}"] if end is not None else [])
            command = [
                ffmpeg, "-hide_banner", "-nostats", "-y", *window, "-i", str(source),
                "-vf", f"setpts=PTS+{begin:.3f}/TB,subtitles=subs.ass,setpts=PTS-STARTPTS",
                "-map", "0:v:0", "-an", *encoder, "-pix_fmt", "yuv420p",
                "-progress", "pipe:1", f"part{index}.mp4",
            ]
            _run_ffmpeg(command, work, work / f"part{index}.log", lambda us: report(index, us))

        audio = _has_audio(source)

        def encode_audio() -> None:
            _run_ffmpeg(
                [ffmpeg, "-hide_banner", "-nostats", "-y", "-i", str(source), "-map", "0:a:0", "-vn",
                 "-c:a", "aac", "-b:a", "192k", "-progress", "pipe:1", "audio.m4a"],
                work, work / "audio.log", lambda _us: None,
            )

        tasks: list[Callable[[], None]] = [
            (lambda i=i, begin=begin, end=end: encode_part(i, begin, end)) for i, (begin, end) in enumerate(parts)
        ]
        if audio:
            tasks.append(encode_audio)
        errors: list[BaseException] = []

        def guarded(task: Callable[[], None]) -> None:
            try:
                task()
            except BaseException as exc:  # noqa: BLE001 - se relanza abajo
                errors.append(exc)

        threads = [threading.Thread(target=guarded, args=(task,), daemon=True) for task in tasks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]

        (work / "parts.txt").write_text(
            "".join(f"file 'part{i}.mp4'\n" for i in range(len(parts))), encoding="utf-8"
        )
        _run_ffmpeg(
            [
                ffmpeg, "-hide_banner", "-nostats", "-y",
                "-f", "concat", "-safe", "0", "-i", "parts.txt", *(["-i", "audio.m4a"] if audio else []),
                "-map", "0:v:0", *(["-map", "1:a:0"] if audio else []), "-c", "copy",
                "-movflags", "+faststart", "-progress", "pipe:1", "out.mp4",
            ],
            work, work / "join.log", lambda _us: None,
        )
        partial = work / "out.mp4"
        if not partial.is_file():
            raise RuntimeError("FFmpeg no generó el vídeo")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


class _Progress:
    def __init__(self, conn: Any, project_id: str, language: str, owner: str) -> None:
        self.conn = conn
        self.project_id = project_id
        self.language = language
        self.owner = owner
        self.value = 0
        self._written = 0.0

    def save(self, *, state: str | None = None, error: str | None = None, storage_key: str | None = None,
             force: bool = False) -> None:
        from app.presentation.api import cluster, db

        moment = cluster.now()
        if not force and moment - self._written < 1.0:
            return
        self._written = moment
        db.run(
            self.conn,
            "UPDATE renders SET progress=?, heartbeat_at=?, state=COALESCE(?, state), error=COALESCE(?, error), "
            "storage_key=COALESCE(?, storage_key) WHERE project_id=? AND language=? AND owner=?",
            (self.value, moment, state, error, storage_key, self.project_id, self.language, self.owner),
        )

    def set(self, value: int) -> None:
        self.value = value
        self.save()


def _claim(conn: Any, project_id: str, language: str, stamp: str) -> str | None:
    from app.presentation.api import cluster, db

    owner = f"{cluster.INSTANCE_ID}:{uuid.uuid4().hex[:8]}"
    moment = cluster.now()
    with db.transaction(conn) as tx:
        tx.run(
            """
            INSERT INTO renders (project_id, language, state, progress, error, stamp, owner, heartbeat_at, storage_key)
            VALUES (?, ?, 'queued', 0, '', ?, ?, ?, '')
            ON CONFLICT(project_id, language) DO UPDATE SET
                state='queued', progress=0, error='', stamp=excluded.stamp, owner=excluded.owner,
                heartbeat_at=excluded.heartbeat_at, storage_key=''
            WHERE renders.state NOT IN ('queued', 'running') OR renders.heartbeat_at < ?
            """,
            (project_id, language, stamp, owner, moment, cluster.stale_before()),
        )
        row = tx.one("SELECT owner FROM renders WHERE project_id=? AND language=?", (project_id, language))
    return owner if row is not None and row["owner"] == owner else None


def _beat(progress: _Progress, stop: threading.Event) -> None:
    from app.presentation.api import cluster

    while not stop.wait(cluster.HEARTBEAT_S):
        try:
            progress.save(force=True)
        except Exception:  # noqa: BLE001 - un fallo puntual de la base no corta el render
            pass


def _store_remote(project: Any, language: str, video: Path, duration_ms: int) -> str | None:
    from app.presentation.api import media_store
    from app.presentation.api import storage_fit

    if not media_store.configured():
        return None
    try:
        fitted = storage_fit.fit(video, duration_ms, video.with_name(f"{language}.fit.mp4"))
    except storage_fit.TooLongForStorage:
        log.warning("render[%s/%s]: no cabe en el almacenamiento; queda solo en local", str(project["id"])[:8], language)
        return None
    key = f"{project['user_id']}/{project['id']}/renders/{language}.mp4"
    try:
        media_store.upload_file(fitted, key, "video/mp4")
    finally:
        if fitted != video:
            fitted.unlink(missing_ok=True)
    return media_store.make_uri(media_store.bucket(), key)


def start(
    *,
    conn: Any,
    output_root: Path,
    project: Any,
    language: str,
    cues: list[Any],
    stamp: str,
    source_of: Callable[[Any], Any],
) -> dict[str, Any]:
    project_id = str(project["id"])
    row = _row(conn, project_id, language)
    if row is not None and row["state"] in ACTIVE and not _is_dead(row):
        return _view(row["state"], int(row["progress"]), "", language)
    if ready_file(output_root, project_id, language, stamp) or stored_file(conn, project_id, language, stamp):
        return _view("done", 100, "", language)
    owner = _claim(conn, project_id, language, stamp)
    if owner is None:
        row = _row(conn, project_id, language)
        return _view(row["state"], int(row["progress"]), row["error"], language)

    progress = _Progress(conn, project_id, language, owner)
    started = time.monotonic()

    def run() -> None:
        from app.infrastructure.subtitles.exporters import to_ass_burn

        video, meta = _paths(output_root, project_id, language)
        stop = threading.Event()
        threading.Thread(target=_beat, args=(progress, stop), name=f"render-beat-{project_id[:8]}", daemon=True).start()
        try:
            with _RENDER_SLOT:
                progress.save(state="running", force=True)
                source = local_source(
                    source_of(project), render_dir(output_root, project_id) / "source.media",
                    lambda fraction: progress.set(int(fraction * 10)),
                )
                width, height, duration_ms = probe(source)
                ass = to_ass_burn(cues, language=language, width=width, height=height)
                base = progress.value
                burn(source, ass, video, duration_ms=duration_ms,
                     on_progress=lambda value: progress.set(base + value * (100 - base) // 100))
                meta.write_text(json.dumps({"stamp": stamp, "language": language}), encoding="utf-8")
                remote = _store_remote(project, language, video, duration_ms)
                progress.value = 100
                progress.save(state="done", storage_key=remote or "", force=True)
                log.info("render[%s/%s]: listo en %.1fs", project_id[:8], language, time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
            progress.save(state="failed", error=str(exc), force=True)
            log.warning("render[%s/%s]: falló: %s", project_id[:8], language, exc)
        finally:
            stop.set()

    threading.Thread(target=run, name=f"render-{project_id[:8]}-{language}", daemon=True).start()
    return _view("queued", 0, "", language)


def forget(conn: Any, output_root: Path, project_id: str) -> None:
    from app.presentation.api import db, media_store

    for row in db.many(conn, "SELECT storage_key FROM renders WHERE project_id=?", (project_id,)):
        key = media_store.object_key(row["storage_key"]) if row["storage_key"] else ""
        if key:
            try:
                media_store.delete(key)
            except Exception:  # noqa: BLE001 - borrar el proyecto manda
                pass
    db.run(conn, "DELETE FROM renders WHERE project_id=?", (project_id,))
    shutil.rmtree(render_dir(output_root, project_id), ignore_errors=True)


def reset() -> None:
    return None


__all__ = ["burn", "forget", "probe", "ready_file", "render_dir", "reset", "start", "status", "stored_file"]
