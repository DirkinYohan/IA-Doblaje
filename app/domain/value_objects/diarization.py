"""Value Objects Diarization Step 08 — Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (sin IA).
"""
from __future__ import annotations

import math
import re
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator

from app.core.constants import (
    SPEAKER_LABEL_MAX_INDEX,
    SPEAKER_LABEL_MIN_INDEX,
    SPEAKER_LABEL_REGEX,
)


# ---------------------------------------------------------------------------
# 1) Value Objects escalares
# ---------------------------------------------------------------------------

_SPEAKER_LABEL_RE = re.compile(SPEAKER_LABEL_REGEX)


def _validate_ms_ge_zero(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"DiarizationMs requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"DiarizationMs no admite negativos: {v}")
    return int(v)


DiarizationMs = Annotated[int, BeforeValidator(_validate_ms_ge_zero)]


def _validate_confidence_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("DiarizationConfidence no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(
            f"DiarizationConfidence requiere float/int, recibido {type(v).__name__}"
        )
    x = float(v)
    if math.isnan(x):
        raise ValueError("DiarizationConfidence no admite NaN")
    if math.isinf(x):
        raise ValueError("DiarizationConfidence no admite Inf")
    if x < 0.0:
        raise ValueError(f"DiarizationConfidence < 0.0: {x}")
    if x > 1.0:
        raise ValueError(f"DiarizationConfidence > 1.0: {x}")
    return x


DiarizationConfidence = Annotated[float, BeforeValidator(_validate_confidence_0_1)]


def _validate_speaker_label(v: object) -> str:
    if isinstance(v, bool) or not isinstance(v, str):
        raise ValueError(
            f"SpeakerLabel requiere str, recibido {type(v).__name__}"
        )
    s = str(v).strip()
    if not _SPEAKER_LABEL_RE.match(s):
        raise ValueError(
            f"SpeakerLabel invalido: {v!r}. Debe cumplir {SPEAKER_LABEL_REGEX!r} "
            f"(case-sensitive, mayúsculas)."
        )
    idx = int(s.split("_")[1])
    if idx < SPEAKER_LABEL_MIN_INDEX or idx > SPEAKER_LABEL_MAX_INDEX:
        raise ValueError(
            f"SpeakerLabel indice fuera de rango "
            f"[{SPEAKER_LABEL_MIN_INDEX}, {SPEAKER_LABEL_MAX_INDEX}]: {s!r}"
        )
    return s


SpeakerLabel = Annotated[str, BeforeValidator(_validate_speaker_label)]


# ---------------------------------------------------------------------------
# 2) SpeakerTurn
# ---------------------------------------------------------------------------


class SpeakerTurn(BaseModel):
    """Turno de hablante (intervalo temporal atribuido a un speaker).

    Garantias:
      start_ms >= 0
      end_ms > start_ms
      duration_ms == end_ms - start_ms (single source of truth)
      confidence en [0,1], sin NaN/Inf/bool
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    speaker_label: SpeakerLabel = Field(..., description="Label normalizado SPEAKER_NN.")
    start_ms: DiarizationMs = Field(..., description="Inicio del turno (inclusivo, ms).")
    end_ms: DiarizationMs = Field(..., description="Fin del turno (exclusivo, ms).")
    duration_ms: DiarizationMs = Field(
        0, description="Duración del turno (ms). Se recomputa = end_ms - start_ms."
    )
    confidence: DiarizationConfidence = Field(
        1.0, description="Confianza del turno en [0,1]. Default 1.0 (sin score del modelo)."
    )

    @model_validator(mode="after")
    def _validate(self) -> "SpeakerTurn":
        if not self.end_ms > self.start_ms:
            raise ValueError(
                f"SpeakerTurn requiere end_ms > start_ms: "
                f"start_ms={self.start_ms}, end_ms={self.end_ms}"
            )
        expected = int(self.end_ms) - int(self.start_ms)
        if self.duration_ms != expected:
            object.__setattr__(self, "duration_ms", expected)
        return self

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


# ---------------------------------------------------------------------------
# 3) DiarizationThresholds (configuracion centralizada T08)
# ---------------------------------------------------------------------------


class DiarizationThresholds(BaseModel):
    """Configuracion T08. Defaults estrictos.

    min_speaker_duration_ms = 200
    min_gap_ms              = 0
    max_num_speakers        = 10
    allow_merge_proximal    = True
    allow_silent_fallback   = True
    strict_validity         = True
    max_turns_safety        = 100000
    max_overlap_tolerance_ms = 0
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_speaker_duration_ms: int = Field(
        200, ge=1, le=60_000, description="Duracion minima (ms) de un turno valido."
    )
    min_gap_ms: int = Field(
        0, ge=0, le=60_000, description="Gap maximo (ms) permitido para merge proximal."
    )
    max_num_speakers: int = Field(
        10, ge=1, le=100, description="Numero maximo de speakers permitido."
    )
    allow_merge_proximal: bool = Field(
        True, description="Permitir merge de turnos cortos del mismo speaker."
    )
    allow_silent_fallback: bool = Field(
        True, description="Permitir silent fallback cuando no hay intervalos de voz."
    )
    strict_validity: bool = Field(
        True, description="Si True, el UseCase exige post-condiciones estrictas."
    )
    max_turns_safety: int = Field(
        100_000, ge=1, le=10_000_000, description="Limite de seguridad de turnos."
    )
    max_overlap_tolerance_ms: int = Field(
        0, ge=0, le=10_000, description="Tolerancia de borde (ms) al intersectar con VAD."
    )

    @field_validator(
        "min_speaker_duration_ms",
        "min_gap_ms",
        "max_num_speakers",
        "max_turns_safety",
        "max_overlap_tolerance_ms",
        mode="before",
    )
    @classmethod
    def _reject_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("No se permite bool donde se espera int (T08).")
        return v


__all__ = [
    "DiarizationMs",
    "DiarizationConfidence",
    "SpeakerLabel",
    "SpeakerTurn",
    "DiarizationThresholds",
]
