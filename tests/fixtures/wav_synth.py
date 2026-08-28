"""Fixtures sintéticas WAV — stdlib `wave` + `struct` solamente.

0 numpy, 0 soundfile, 0 scipy, 0 torch, 0 externas.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from typing import Tuple


def create_synthetic_wav_pcm16(
    output_path: Path,
    *,
    duration_sec: float = 0.2,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    freq_hz: float = 440.0,
    amplitude: float = 0.7,
) -> Tuple[Path, int, int, int, float]:
    """Crea un archivo WAV PCM 16-bit little-endian con una onda senoidal pura.

    Args:
        output_path: Path absoluto donde escribir el WAV.
        duration_sec: duración segundos.
        sample_rate_hz: 8000–96000 (16000 default).
        channels: 1 mono | 2 stereo.
        freq_hz: Frecuencia tono (440Hz por defecto: A4).
        amplitude: 0.0–1.0 (default 0.7 para evitar clipping cuando peak norm).

    Returns:
        (path, n_samples_total_written, sample_rate_hz, channels, duration_sec)
    """
    # Input validation básicos
    if not (0.01 <= float(duration_sec) <= 60.0):
        raise ValueError(f"duration_sec fuera [0.01,60.0]: {duration_sec}")
    if not (8000 <= int(sample_rate_hz) <= 96000):
        raise ValueError(f"sample_rate_hz fuera [8000,96000]: {sample_rate_hz}")
    ch = int(channels)
    if ch not in (1, 2):
        raise ValueError(f"channels invalido: {ch}")
    amp = float(amplitude)
    if amp <= 0.0 or amp > 1.0:
        raise ValueError(f"amplitude fuera (0,1]: {amp}")
    sr = int(sample_rate_hz)
    dur = float(duration_sec)
    total_samples = int(math.ceil(sr * dur))
    if total_samples < 1:
        raise ValueError("0 samples generados, revisar duration/sample_rate.")
    freq = float(freq_hz)
    if freq <= 0.0:
        raise ValueError(f"freq_hz invalido: {freq}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escribir WAV con wave stdlib
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)  # 16-bit = 2 bytes PCM little-endian
        wf.setframerate(sr)
        # Escribir en chunks para no agotar memoria (aunque duration pequeña)
        chunk = 1024
        two_pi_f_over_sr = 2.0 * math.pi * freq / float(sr)
        max_int = 32767
        written = 0
        while written < total_samples:
            end = min(written + chunk, total_samples)
            frames: list[bytes] = []
            for i in range(written, end):
                sample = math.sin(two_pi_f_over_sr * float(i))
                int16_val = int(round(max_int * amp * sample))
                # clamp just in case math sin round issues
                if int16_val > max_int:
                    int16_val = max_int
                if int16_val < -32768:
                    int16_val = -32768
                frame_bytes = struct.pack("<h", int16_val)
                if ch == 1:
                    frames.append(frame_bytes)
                else:
                    # Stereo: mismo tono en ambos canales (test simple)
                    frames.append(frame_bytes)
                    frames.append(frame_bytes)
            written = end
            wf.writeframes(b"".join(frames))

    if not output_path.is_file() or output_path.stat().st_size <= 44:
        raise RuntimeError(f"WAV no generado correctamente: {output_path}")
    return (output_path, total_samples, sr, ch, dur)


__all__ = ["create_synthetic_wav_pcm16"]
