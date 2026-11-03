"""Vídeo con los subtítulos INCRUSTADOS (quemados en la imagen).

Se renderiza con FFmpeg + libass a partir del original (sin descargarlo
entero: ver `live.media_source`) y de la pista del idioma pedido, escrita como
ASS a la medida de la resolución del vídeo (ver `to_ass_burn`).

* Un render a la vez: libx264 usa todos los núcleos y competiría con la
  generación de subtítulos. Los demás esperan en cola.
* Progreso real: FFmpeg informa del tiempo procesado (`-progress`).
* El resultado se reutiliza mientras la pista no cambie: la marca
  `updated_at` de los subtítulos se guarda junto al archivo. Si la pista
  cambia después (se editó, o seguía generándose), el MP4 sigue disponible
  pero se marca como desactualizado (`stale`) para poder regenerarlo.
* Se escribe a un temporal y sólo se publica al terminar bien: nunca se
  descarga un MP4 a medias.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("ia.render")

_RENDER_SLOT = threading.Semaphore(1)
_LOCK = threading.Lock()
_JOBS: dict[tuple[str, str], "RenderJob"] = {}


@dataclass
class RenderJob:
    project_id: str
    language: str
    state: str = "queued"  # queued | running | done | failed
    progress: int = 0
    error: str = ""
    started: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {"state": self.state, "progress": self.progress, "error": self.error, "language": self.language}


def render_dir(output_root: Path, project_id: str) -> Path:
    return Path(output_root) / "renders" / project_id


def _paths(output_root: Path, project_id: str, language: str) -> tuple[Path, Path]:
    folder = render_dir(output_root, project_id)
    return folder / f"{language}.mp4", folder / f"{language}.json"


def ready_file(output_root: Path, project_id: str, language: str, stamp: str | None = None) -> Path | None:
    """El MP4 ya renderizado, o None.

    Con `stamp`, sólo si sigue al día con esa versión de la pista.
    """
    video, meta = _paths(output_root, project_id, language)
    if not video.is_file() or not meta.is_file():
        return None
    try:
        saved = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return video if stamp is None or saved.get("stamp") == stamp else None


def status(output_root: Path, project_id: str, language: str, stamp: str) -> dict[str, Any]:
    with _LOCK:
        job = _JOBS.get((project_id, language))
    if job is not None and job.state in {"queued", "running", "failed"}:
        return job.as_dict()
    if ready_file(output_root, project_id, language, stamp) is not None:
        return {"state": "done", "progress": 100, "error": "", "language": language}
    if ready_file(output_root, project_id, language) is not None:
        return {"state": "stale", "progress": 100, "error": "", "language": language}
    return {"state": "idle", "progress": 0, "error": "", "language": language}


def _resolve(binary: str) -> str:
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver

    ok, path, _version = SubprocessFFmpegBinaryResolver().resolve(binary)
    if not ok:
        raise RuntimeError(path)
    return path


def probe(source: Any) -> tuple[int, int, int]:
    """(ancho, alto, duración en ms) del vídeo."""
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


def burn(
    source: Any,
    ass_text: str,
    target: Path,
    *,
    duration_ms: int,
    on_progress: Callable[[int], None] | None = None,
) -> None:
    """Incrusta el ASS en el vídeo y escribe un MP4 (H.264 + AAC) en `target`."""
    work = target.parent / f".work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        # El filtro `subtitles` interpreta «:» y «\» de las rutas de Windows:
        # se ejecuta DENTRO de la carpeta y se le pasa sólo el nombre.
        (work / "subs.ass").write_text(ass_text, encoding="utf-8")
        partial = work / "out.mp4"
        command = [
            _resolve("ffmpeg"), "-hide_banner", "-nostats", "-y",
            "-i", str(source),
            "-vf", "subtitles=subs.ass",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            "-progress", "pipe:1",
            "out.mp4",
        ]
        # El diagnóstico va a un ARCHIVO, no a una tubería: si stderr fuera otra
        # tubería sin leer, al llenarse (~64 KB) FFmpeg se bloquearía esperando
        # y este bucle, esperando el progreso, también: un interbloqueo.
        log_path = work / "ffmpeg.log"
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file, subprocess.Popen(
            command, cwd=work, stdout=subprocess.PIPE, stderr=log_file,
            text=True, encoding="utf-8", errors="replace",
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and value.isdigit() and duration_ms > 0 and on_progress:
                    on_progress(min(99, int(int(value) / 1000 * 100 / duration_ms)))
            code = process.wait()
        if code != 0 or not partial.is_file():
            detail = log_path.read_text(encoding="utf-8", errors="replace").strip()[-300:]
            raise RuntimeError(f"FFmpeg falló: {detail}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial.replace(target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def start(
    *,
    output_root: Path,
    project: Any,
    language: str,
    cues: list[Any],
    stamp: str,
    source_of: Callable[[Any], Any],
) -> dict[str, Any]:
    """Arranca (o devuelve) el render de un idioma. Nunca bloquea la petición."""
    project_id = str(project["id"])
    key = (project_id, language)
    with _LOCK:
        current = _JOBS.get(key)
        if current is not None and current.state in {"queued", "running"}:
            return current.as_dict()
        if ready_file(output_root, project_id, language, stamp) is not None:
            return {"state": "done", "progress": 100, "error": "", "language": language}
        job = RenderJob(project_id, language)
        _JOBS[key] = job

    def run() -> None:
        from app.infrastructure.subtitles.exporters import to_ass_burn

        video, meta = _paths(output_root, project_id, language)
        with _RENDER_SLOT:
            job.state = "running"
            try:
                source = source_of(project)
                width, height, duration_ms = probe(source)
                ass = to_ass_burn(cues, language=language, width=width, height=height)

                def progress(value: int) -> None:
                    job.progress = value

                burn(source, ass, video, duration_ms=duration_ms, on_progress=progress)
                meta.write_text(json.dumps({"stamp": stamp, "language": language}), encoding="utf-8")
                job.progress = 100
                job.state = "done"
                log.info(
                    "render[%s/%s]: listo en %.1fs", project_id[:8], language, time.monotonic() - job.started
                )
            except Exception as exc:  # noqa: BLE001 - se informa al usuario, no se oculta
                job.state = "failed"
                job.error = str(exc)
                log.warning("render[%s/%s]: falló: %s", project_id[:8], language, exc)

    threading.Thread(target=run, name=f"render-{project_id[:8]}-{language}", daemon=True).start()
    return job.as_dict()


def forget(output_root: Path, project_id: str) -> None:
    """Al borrar el proyecto: fuera sus renders y su estado."""
    with _LOCK:
        for key in [k for k in _JOBS if k[0] == project_id]:
            _JOBS.pop(key, None)
    shutil.rmtree(render_dir(output_root, project_id), ignore_errors=True)


def reset() -> None:
    """Pruebas: olvida todos los renders en memoria."""
    with _LOCK:
        _JOBS.clear()


__all__ = ["burn", "forget", "probe", "ready_file", "render_dir", "reset", "start", "status"]
