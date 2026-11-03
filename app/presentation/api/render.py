"""Vídeo con los subtítulos INCRUSTADOS (quemados en la imagen).

Se renderiza con FFmpeg + libass a partir del original y de la pista del
idioma pedido, escrita como ASS a la medida de la resolución del vídeo (ver
`to_ass_burn`).

Rapidez (1 min de vídeo 1080p: de ~7,7 s a ~3,2 s en un portátil):

* El original se descarga UNA vez a disco y sirve para todos los idiomas:
  leerlo de la URL firmada mientras se codifica obliga a FFmpeg a saltar por
  la red (el índice del MP4 suele ir al final).
* Codificador por hardware si lo hay (NVENC, AMF, Quick Sync); si no, libx264.
* El vídeo se parte en tramos que se procesan en paralelo: libass dibuja en un
  solo hilo y era el cuello de botella. Los tramos se unen sin recodificar.

* Un render a la vez: compite con la generación de subtítulos. Los demás
  esperan en cola.
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
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
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


# Calidad parecida en todos: visualmente sin pérdidas a 1080p.
_HARDWARE_ENCODERS: list[tuple[str, list[str]]] = [
    ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "22", "-b:v", "0"]),
    ("h264_amf", ["-c:v", "h264_amf", "-quality", "balanced", "-rc", "cqp", "-qp_i", "20", "-qp_p", "22"]),
    ("h264_qsv", ["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "22"]),
]
_SOFTWARE_ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]

# Tramos en paralelo: como mucho 4 y de al menos 20 s (por debajo, arrancar
# cada FFmpeg cuesta más de lo que se gana).
_MAX_PARTS = 4
_MIN_PART_MS = 20_000


@lru_cache(maxsize=1)
def video_encoder() -> tuple[str, ...]:
    """Argumentos del codificador H.264 más rápido que funcione en este equipo.

    Que FFmpeg lo liste no basta (puede faltar la GPU o el controlador): se
    prueba con una codificación diminuta. `IA_RENDER_ENCODER` fuerza uno
    (`libx264`, `h264_amf`…).
    """
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
    """El original en disco: si es una URL, se descarga a `cache` (una vez)."""
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
    """Tramos (inicio, fin) en segundos; el último sin fin: llega al final."""
    count = max(1, min(_MAX_PARTS, duration_ms // _MIN_PART_MS)) if duration_ms > 0 else 1
    step = duration_ms / 1000 / count
    return [(round(i * step, 3), round((i + 1) * step, 3) if i < count - 1 else None) for i in range(count)]


def _run_ffmpeg(command: list[str], cwd: Path, log_path: Path, on_time: Callable[[int], None]) -> None:
    """Ejecuta FFmpeg informando del tiempo procesado (µs).

    El diagnóstico va a un ARCHIVO, no a una tubería: si stderr fuera otra
    tubería sin leer, al llenarse (~64 KB) FFmpeg se bloquearía esperando y
    este bucle, esperando el progreso, también: un interbloqueo.
    """
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
    """Incrusta el ASS en el vídeo y escribe un MP4 (H.264 + AAC) en `target`."""
    work = target.parent / f".work-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        # El filtro `subtitles` interpreta «:» y «\» de las rutas de Windows:
        # se ejecuta DENTRO de la carpeta y se le pasa sólo el nombre.
        (work / "subs.ass").write_text(ass_text, encoding="utf-8")
        ffmpeg = _resolve("ffmpeg")
        encoder = list(video_encoder())
        parts = _parts(duration_ms)
        done_us = [0] * len(parts)
        lock = threading.Lock()

        def report(index: int, value: int) -> None:
            # Bajo el candado: los tramos informan a la vez y el progreso no
            # debe retroceder.
            with lock:
                done_us[index] = value
                if duration_ms > 0 and on_progress:
                    # El 2 % final es la unión de los tramos y el audio.
                    on_progress(min(98, int(sum(done_us) / 1000 * 98 / duration_ms)))

        def encode_part(index: int, begin: float, end: float | None) -> None:
            # Cada tramo empieza en 0: se lleva a su hora real para que libass
            # dibuje el subtítulo que toca, y se devuelve a 0 después.
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
            # A la vez que el vídeo: en la unión final sólo se copia.
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

        # Unión: sólo se copian los tramos y el audio, nada se recodifica.
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
                # Descargar el original es el primer 10 % del progreso.
                def downloaded(fraction: float) -> None:
                    job.progress = int(fraction * 10)

                source = local_source(
                    source_of(project), render_dir(output_root, project_id) / "source.media", downloaded
                )
                width, height, duration_ms = probe(source)
                ass = to_ass_burn(cues, language=language, width=width, height=height)

                base = job.progress

                def progress(value: int) -> None:
                    job.progress = base + value * (100 - base) // 100

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
