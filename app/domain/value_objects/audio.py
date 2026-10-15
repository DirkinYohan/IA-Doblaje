"""Value Objects de audio/multimedia — Capa DOMAIN. Pydantic V2 native Annotated.

Todos son inmutables vía Pydantic frozen=True en las entidades.
Sin dependencias de Application ni Infrastructure.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, NewType

from pydantic import Field, field_validator
from pydantic.functional_validators import AfterValidator


# ---------------------------------------------------------------------------
# Helpers validadores
# ---------------------------------------------------------------------------


_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def _validate_sample_rate(v: Any) -> int:
    try:
        value = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SampleRate invalido, esperado int: {v!r}") from exc
    if value < 8000 or value > 96000:
        raise ValueError(
            f"SampleRate fuera de rango [8000, 96000] Hz: {value}"
        )
    return value


def _validate_channel_count(v: Any) -> int:
    try:
        value = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ChannelCount invalido: {v!r}") from exc
    if value not in (1, 2):
        raise ValueError(f"ChannelCount debe ser 1 (mono) o 2 (stereo): {value}")
    return value


def _validate_bit_depth(v: Any) -> int:
    _ALLOWED = frozenset({8, 16, 24, 32})
    try:
        value = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BitDepth invalido: {v!r}") from exc
    if value not in _ALLOWED:
        raise ValueError(
            f"BitDepth no permitido {value}. Validos: {sorted(_ALLOWED)}"
        )
    return value


def _validate_audio_duration(v: Any) -> float:
    try:
        value = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AudioDuration invalido: {v!r}") from exc
    if value < 0.0:
        raise ValueError(f"AudioDuration no puede ser negativa: {value}")
    return value


def _validate_sha256_hash(v: Any) -> str:
    s = str(v or "").strip().lower()
    if not _SHA256_RE.match(s):
        raise ValueError(
            "Sha256Hash no cumple pattern 64 chars hex: "
            f"{str(v)[:20]!r}..."
        )
    return s


def _validate_file_bytes(v: Any) -> int:
    try:
        value = int(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FileBytes invalido: {v!r}") from exc
    if value < 0:
        raise ValueError(f"FileBytes no puede ser negativo: {value}")
    return value


# ---------------------------------------------------------------------------
# Tipos públicos (alias + validators). Pydantic V2 usa Annotated[BaseType, AfterValidator(...)]
# ---------------------------------------------------------------------------


SampleRate = Annotated[int, AfterValidator(_validate_sample_rate)]
ChannelCount = Annotated[int, AfterValidator(_validate_channel_count)]
BitDepth = Annotated[int, AfterValidator(_validate_bit_depth)]
AudioDuration = Annotated[float, AfterValidator(_validate_audio_duration)]
Sha256Hash = Annotated[str, AfterValidator(_validate_sha256_hash)]
FileBytes = Annotated[int, AfterValidator(_validate_file_bytes)]

# Exports para runtime helpers (tests)
__all__ = [
    "SampleRate",
    "ChannelCount",
    "BitDepth",
    "AudioDuration",
    "Sha256Hash",
    "FileBytes",
    "_SHA256_RE",
    "_validate_sample_rate",
    "_validate_channel_count",
    "_validate_bit_depth",
    "_validate_audio_duration",
    "_validate_sha256_hash",
    "_validate_file_bytes",
]
