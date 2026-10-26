"""Lectura de audio POR SEGMENTOS directamente del medio original.

El procesamiento progresivo no puede esperar a extraer y normalizar el audio
completo (loudnorm EBU R128 a dos pasadas, VAD y LID del archivo entero): en un
vídeo largo eso son minutos sin un solo subtítulo. Este lector decodifica sólo
el tramo pedido con FFmpeg (`-ss` de entrada, búsqueda precisa) y lo devuelve
como PCM float32 16 kHz mono.

Expone la misma interfaz que ``ProgressiveAudioWindowReader`` (``read``,
``duration_ms``, ``total_frames``, ``sample_rate``), así que el
``ProgressiveTranscriber`` lo usa sin cambios.

El archivo original sólo se LEE: nunca se reescribe ni se copia.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.core.exceptions import AudioExtractionError

SAMPLE_RATE = 16000
# Mismo filtrado básico que el preprocesado del pipeline completo (subpaso A):
# quita rumble y continua sin la normalización a dos pasadas del archivo entero.
SEGMENT_FILTER = "highpass=f=80:poles=2"

_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0


class FFmpegSegmentReader:
    """Decodifica tramos ``[start_ms, end_ms)`` del medio bajo demanda."""

    def __init__(
        self,
        media_path: str | Path,
        *,
        ffmpeg_bin: str,
        ffprobe_bin: str = "",
        duration_ms: int = 0,
        sample_rate: int = SAMPLE_RATE,
        timeout_s: int = 300,
    ) -> None:
        # Acepta una URL (p. ej. la firmada de Storage): FFmpeg lee por rangos
        # y no hace falta descargar el archivo entero antes de empezar.
        self.source = str(media_path)
        self.remote = self.source.startswith(("http://", "https://"))
        self.path = None if self.remote else Path(media_path)
        if not self.remote and (self.path is None or not self.path.is_file()):
            raise AudioExtractionError(f"medio original no encontrado: {self.source}")
        if not ffmpeg_bin:
            raise AudioExtractionError("FFmpeg no disponible para leer el audio por segmentos")
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.sample_rate = int(sample_rate)
        self.timeout_s = int(timeout_s)
        probed = self._probe_duration_ms() if ffprobe_bin else 0
        self.duration_ms = int(probed or duration_ms or 0)
        if self.duration_ms <= 0:
            name = "el medio remoto" if self.remote else (self.path.name if self.path else "el medio")
            raise AudioExtractionError(f"no se pudo medir la duración de {name}")
        self.total_frames = int(self.duration_ms * self.sample_rate / 1000)

    def _probe_duration_ms(self) -> int:
        """Duración del PRIMER flujo de audio (o del contenedor)."""
        command = [
            self.ffprobe_bin, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=duration:format=duration",
            "-of", "json", self.source,
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, text=True, timeout=60,
                check=False, creationflags=_NO_WINDOW,
            )
            payload = json.loads(done.stdout or "{}")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return 0
        streams = payload.get("streams") or []
        if not streams:
            raise AudioExtractionError("el medio no tiene pista de audio")
        for raw in (streams[0].get("duration"), (payload.get("format") or {}).get("duration")):
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return int(round(value * 1000))
        return 0

    def read(self, start_ms: int, end_ms: int) -> Any:
        """PCM float32 mono [-1, 1] del tramo pedido (recortado a la duración)."""
        import numpy as np

        start = max(0, int(start_ms))
        stop = min(self.duration_ms, int(end_ms))
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        command = [
            self.ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error",
            "-ss", f"{start / 1000:.3f}",
            "-i", self.source,
            "-t", f"{(stop - start) / 1000:.3f}",
            "-map", "0:a:0", "-vn", "-sn", "-dn",
            "-ac", "1", "-ar", str(self.sample_rate),
            "-af", SEGMENT_FILTER,
            "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1",
        ]
        try:
            done = subprocess.run(
                command, capture_output=True, timeout=self.timeout_s,
                check=False, creationflags=_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AudioExtractionError(f"FFmpeg no pudo leer el tramo {start}-{stop} ms: {exc}") from exc
        if done.returncode != 0:
            detail = done.stderr.decode("utf-8", "replace").strip()[:300]
            raise AudioExtractionError(f"FFmpeg falló en el tramo {start}-{stop} ms: {detail}")
        data = done.stdout
        if len(data) < 2:
            return np.zeros(0, dtype=np.float32)
        pcm = np.frombuffer(data[: len(data) - (len(data) % 2)], dtype="<i2")
        return np.clip(pcm.astype(np.float32) / 32768.0, -1.0, 1.0)


__all__ = ["FFmpegSegmentReader", "SAMPLE_RATE", "SEGMENT_FILTER"]
