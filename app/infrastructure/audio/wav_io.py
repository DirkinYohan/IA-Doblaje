from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.exceptions import AudioExtractionError


def read_wav_as_tensor(
    wav_path: str | Path,
    *,
    expected_sample_rate: int | None = None,
    mono: bool = True,
    scale_int16: bool = True,
) -> tuple[Any, int]:
    import soundfile as sf
    import torch

    path = Path(wav_path)
    if not path.is_file():
        raise AudioExtractionError(f"WAV no encontrado: {path!s}")
    try:
        data, sr = sf.read(path, always_2d=True, dtype="int16")
    except Exception as exc:  # noqa: BLE001
        raise AudioExtractionError(f"soundfile.read falló en {path.name}: {exc!r}") from exc

    sr_hz = int(sr)
    if expected_sample_rate is not None and sr_hz != int(expected_sample_rate):
        raise AudioExtractionError(
            f"Se esperaba {expected_sample_rate} Hz y el WAV tiene {sr_hz} Hz: {path.name}"
        )
    if data.size == 0:
        raise AudioExtractionError(f"WAV vacío: {path.name}")

    tensor = torch.from_numpy(data.T.astype("float32", copy=True))
    if mono and tensor.size(0) > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    if scale_int16:
        tensor = tensor / 32768.0
    return tensor.contiguous(), sr_hz


def read_wav_as_float_mono(wav_path: str | Path) -> tuple[Any, int]:
    return read_wav_as_tensor(wav_path, mono=True, scale_int16=True)


__all__ = ["read_wav_as_float_mono", "read_wav_as_tensor"]
