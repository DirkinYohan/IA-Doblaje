"""Ajusta un vídeo al tamaño máximo por archivo del almacenamiento.

El plan gratuito de Supabase rechaza archivos de más de 50 MB. Si el vídeo lo
supera, se recodifica a H.264 con la tasa de bits justa para caber.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


AUDIO_KBPS = 128
MIN_VIDEO_KBPS = 150
SAFETY = 0.95
ENCODE_TIMEOUT_S = 7200


class TooLongForStorage(ValueError):
    pass


def limit_bytes() -> int:
    return int(float(os.environ.get("IA_STORAGE_MAX_MB", "50")) * 1024 * 1024)


def max_minutes(_limit: int | None = None) -> int:
    """Duración máxima de un vídeo que hay que comprimir para que quepa."""
    return int(float(os.environ.get("IA_MAX_VIDEO_MINUTES", "25")))


def _total_kbps(limit: int, duration_ms: int) -> float:
    seconds = max(duration_ms / 1000, 1.0)
    return limit * 8 * SAFETY / seconds / 1000


def audio_kbps(limit: int, duration_ms: int) -> int:
    """En vídeos largos el audio cede bits: 64 kbps bastan para la voz."""
    total = _total_kbps(limit, duration_ms)
    if total >= 900:
        return AUDIO_KBPS
    if total >= 500:
        return 96
    return 64


def video_kbps(limit: int, duration_ms: int) -> int:
    return int(_total_kbps(limit, duration_ms) - audio_kbps(limit, duration_ms))


def _too_long(limit: int) -> TooLongForStorage:
    return TooLongForStorage(
        "El vídeo es demasiado largo para el almacenamiento gratuito: "
        f"admite unos {max_minutes(limit)} minutos como máximo."
    )


def _height(kbps: int) -> int:
    if kbps < 450:
        return 480
    return 720 if kbps < 2500 else 1080


def _command(ffmpeg: str, source: Path, target: Path, kbps: int, audio: int = AUDIO_KBPS) -> list[str]:
    height = _height(kbps)
    nice = shutil.which("nice") if os.name == "posix" else None
    # Prioridad mínima: la transcripción en vivo comparte las mismas CPU.
    return [
        *([nice, "-n", "19"] if nice else []),
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        # Un solo hilo: con uno por CPU visible le quitaba la CPU a la transcripción en vivo.
        "-threads", "1", "-filter_threads", "1", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale=-2:'min({height},ih)'",
        "-c:v", "libx264", "-preset", "superfast", "-threads", "1",
        "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", f"{audio}k", "-ac", "2",
        "-movflags", "+faststart",
        str(target),
    ]


def _ffmpeg() -> str:
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver

    ok, path, _version = SubprocessFFmpegBinaryResolver().resolve("ffmpeg")
    if not ok:
        raise RuntimeError(path)
    return path


def needs_fit(size: int, duration_ms: int, *, limit: int | None = None) -> bool:
    """True si hay que comprimir; lanza TooLongForStorage si ni comprimido cabe."""
    limit = limit or limit_bytes()
    if size <= limit:
        return False
    # «25 minutos» incluye 25:59: se rechaza a partir del minuto siguiente.
    if duration_ms <= 0 or duration_ms >= (max_minutes() + 1) * 60_000:
        raise _too_long(limit)
    if video_kbps(limit, duration_ms) < MIN_VIDEO_KBPS:
        raise _too_long(limit)
    return True


def fit(
    source: Path,
    duration_ms: int,
    target: Path,
    *,
    limit: int | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Devuelve `source` si ya cabe; si no, `target` con el vídeo comprimido."""
    limit = limit or limit_bytes()
    if not needs_fit(source.stat().st_size, duration_ms, limit=limit):
        return source
    kbps = video_kbps(limit, duration_ms)
    audio = audio_kbps(limit, duration_ms)
    ffmpeg = _ffmpeg()
    for share in (1.0, 0.85):
        run(_command(ffmpeg, source, target, int(kbps * share), audio),
            check=True, capture_output=True, timeout=ENCODE_TIMEOUT_S)
        if target.is_file() and target.stat().st_size <= limit:
            return target
    target.unlink(missing_ok=True)
    raise _too_long(limit)
