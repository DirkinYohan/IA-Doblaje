"""Value Objects Alignment Step 09 — ASR ↔ Speaker Alignment. Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (sin IA).
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator


# ---------------------------------------------------------------------------
# 1) Value Objects escalares
# ---------------------------------------------------------------------------


def _validate_ms_ge_zero(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"AlignmentMs requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"AlignmentMs no admite negativos: {v}")
    return int(v)


AlignmentMs = Annotated[int, BeforeValidator(_validate_ms_ge_zero)]


def _validate_segment_index_ge_zero(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"AlignmentSegmentIndex requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"AlignmentSegmentIndex debe ser >=0: {v!r}")
    return int(v)


AlignmentSegmentIndex = Annotated[int, BeforeValidator(_validate_segment_index_ge_zero)]


def _validate_confidence_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("AlignmentConfidence no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(
            f"AlignmentConfidence requiere float/int, recibido {type(v).__name__}"
        )
    x = float(v)
    if math.isnan(x):
        raise ValueError("AlignmentConfidence no admite NaN")
    if math.isinf(x):
        raise ValueError("AlignmentConfidence no admite Inf")
    if x < 0.0:
        raise ValueError(f"AlignmentConfidence < 0.0: {x}")
    if x > 1.0:
        raise ValueError(f"AlignmentConfidence > 1.0: {x}")
    return x


AlignmentConfidence = Annotated[float, BeforeValidator(_validate_confidence_0_1)]


AlignmentStrategyName = Literal[
    "weighted_overlap_majority_vote",
    "alignment_silent_fallback",
]


# ---------------------------------------------------------------------------
# 2) DialogueSegment (segmento ASR alineado a un speaker)
# ---------------------------------------------------------------------------


class DialogueSegment(BaseModel):
    """Segmento de diálogo: timestamp oficial T07 + texto ASR T06 + speaker T08.

    Garantías:
      - segment_index >= 0
      - start_ms < end_ms
      - duration_ms == end_ms - start_ms (recomputado)
      - speaker_label válido (SPEAKER_NN)
      - alignment_confidence en [0,1]
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_index: AlignmentSegmentIndex = Field(..., description="Índice 0-based contiguo.")
    start_ms: AlignmentMs = Field(..., description="Inicio oficial T07 (ms).")
    end_ms: AlignmentMs = Field(..., description="Fin oficial T07 (ms).")
    duration_ms: AlignmentMs = Field(
        0, description="Duración (ms). Se recomputa = end_ms - start_ms."
    )
    text: str = Field(..., max_length=1_000_000, description="Texto ASR T06 del segmento.")
    speaker_label: str = Field(
        ..., description="Label SPEAKER_NN asignado por majority vote (T08)."
    )
    alignment_confidence: AlignmentConfidence = Field(
        1.0, description="Confianza de alineación [0,1]. Default 1.0."
    )
    ts_confidence: AlignmentConfidence = Field(
        1.0, description="Confianza del timestamp oficial T07 (ts_confidence)."
    )

    @field_validator("speaker_label")
    @classmethod
    def _speaker_label_valid(cls, v: object) -> str:
        if isinstance(v, bool) or not isinstance(v, str):
            raise ValueError("speaker_label requiere str")
        s = str(v)
        import re

        if not re.fullmatch(r"^SPEAKER_[0-9]{2}$", s):
            raise ValueError(f"speaker_label inválido: {s!r}")
        return s

    @model_validator(mode="after")
    def _validate(self) -> "DialogueSegment":
        if not self.end_ms > self.start_ms:
            raise ValueError(
                f"DialogueSegment requiere end_ms > start_ms: "
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
# 3) AlignmentThresholds (configuración centralizada T09)
# ---------------------------------------------------------------------------


class AlignmentThresholds(BaseModel):
    """Configuración T09. Defaults estrictos.

    min_overlap_ms          = 1        (overlap > 0 requerido; se usa > 0)
    allow_silent_fallback   = True
    strict_validity         = True
    max_segments_safety     = 1000000
    max_speakers_safety     = 100
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_overlap_ms: int = Field(
        1, ge=0, le=1_000_000, description="Overlap mínimo (ms) para considerar un candidato."
    )
    allow_silent_fallback: bool = Field(
        True, description="Permitir silent fallback cuando no hay speaker turns."
    )
    strict_validity: bool = Field(
        True, description="Si True, el UseCase exige post-condiciones estrictas."
    )
    max_segments_safety: int = Field(
        1_000_000, ge=1, le=10_000_000, description="Límite de seguridad de segmentos."
    )
    max_speakers_safety: int = Field(
        100, ge=1, le=10_000, description="Límite de seguridad de speakers."
    )

    @field_validator("min_overlap_ms", "max_segments_safety", "max_speakers_safety", mode="before")
    @classmethod
    def _reject_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("No se permite bool donde se espera int (T09).")
        return v


__all__ = [
    "AlignmentMs",
    "AlignmentSegmentIndex",
    "AlignmentConfidence",
    "AlignmentStrategyName",
    "DialogueSegment",
    "AlignmentThresholds",
]
