"""Ajusta un vídeo al tamaño máximo por archivo del almacenamiento.

El plan gratuito de Supabase rechaza archivos de más de 50 MB. Si el vídeo lo
supera, se recodifica a H.264 con la tasa de bits justa para caber.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


AUDIO_KBPS = 128
MIN_VIDEO_KBPS = 250
SAFETY = 0.92
ENCODE_TIMEOUT_S = 1800


class TooLongForStorage(ValueError):
    pass


def limit_bytes() -> int:
    return int(float(os.environ.get("IA_STORAGE_MAX_MB", "50")) * 1024 * 1024)


def max_minutes(limit: int) -> int:
    seconds = limit * 8 * SAFETY / ((MIN_VIDEO_KBPS + AUDIO_KBPS) * 1000)
    return max(1, int(seconds // 60))


def video_kbps(limit: int, duration_ms: int) -> int:
    seconds = max(duration_ms / 1000, 1.0)
    return int(limit * 8 * SAFETY / seconds / 1000 - AUDIO_KBPS)


def _too_long(limit: int) -> TooLongForStorage:
    return TooLongForStorage(
        "El vídeo es demasiado largo para el almacenamiento gratuito: "
        f"admite unos {max_minutes(limit)} minutos como máximo."
    )


def _command(ffmpeg: str, source: Path, target: Path, kbps: int) -> list[str]:
    height = 720 if kbps < 2500 else 1080
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale=-2:'min({height},ih)'",
        "-c:v", "libx264", "-preset", "superfast",
        "-b:v", f"{kbps}k", "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", f"{AUDIO_KBPS}k", "-ac", "2",
        "-movflags", "+faststart",
        str(target),
    ]


def _ffmpeg() -> str:
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver

    ok, path, _version = SubprocessFFmpegBinaryResolver().resolve("ffmpeg")
    if not ok:
        raise RuntimeError(path)
    return path


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
    if source.stat().st_size <= limit:
        return source
    kbps = video_kbps(limit, duration_ms) if duration_ms > 0 else 0
    if kbps < MIN_VIDEO_KBPS:
        raise _too_long(limit)
    ffmpeg = _ffmpeg()
    for share in (1.0, 0.8):
        run(_command(ffmpeg, source, target, int(kbps * share)),
            check=True, capture_output=True, timeout=ENCODE_TIMEOUT_S)
        if target.is_file() and target.stat().st_size <= limit:
            return target
    target.unlink(missing_ok=True)
    raise _too_long(limit)
