"""Value Objects VAD Step 04 — Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (no torch/silero/soundfile).
"""
from __future__ import annotations

from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import AfterValidator


# ---------------------------------------------------------------------------
# 1) Value Objects escalares
# ---------------------------------------------------------------------------


def _validate_ms_ge_zero(v: int) -> int:
    if not isinstance(v, int):
        raise TypeError(f"MsTimestamp requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"MsTimestamp no admite valores negativos: {v}")
    return int(v)


MsTimestamp = Annotated[int, AfterValidator(_validate_ms_ge_zero)]


def _validate_confidence_0_1(v: float) -> float:
    if isinstance(v, bool):
        raise TypeError("SpeechConfidence no admite bool")
    if not isinstance(v, (int, float)):
        raise TypeError(
            f"SpeechConfidence requiere float/int, recibido {type(v).__name__}"
        )
    x = float(v)
    if x < 0.0:
        raise ValueError(f"SpeechConfidence < 0.0: {x}")
    if x > 1.0:
        raise ValueError(f"SpeechConfidence > 1.0: {x}")
    return x


SpeechConfidence = Annotated[float, AfterValidator(_validate_confidence_0_1)]


# ---------------------------------------------------------------------------
# 2) VoiceInterval + SilenceSegment
# ---------------------------------------------------------------------------


class VoiceInterval(BaseModel):
    """Segmento de voz detectado. Validación estricta. Frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: MsTimestamp = Field(..., description="Inclusivo.")
    end_ms: MsTimestamp = Field(..., description="Exclusivo.")
    max_confidence: SpeechConfidence = Field(
        ..., description="Confidence máxima observada en este intervalo."
    )
    duration_ms: MsTimestamp = Field(0)
    sample_count: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _validate(self) -> "VoiceInterval":
        if not self.end_ms > self.start_ms:
            raise ValueError(
                f"VoiceInterval requiere end_ms>start_ms: [{self.start_ms}, {self.end_ms})"
            )
        expected_dur = int(self.end_ms) - int(self.start_ms)
        if self.duration_ms <= 0 or self.duration_ms != expected_dur:
            # Recompute forzando la regla single source of truth.
            object.__setattr__(self, "duration_ms", expected_dur)
        if self.sample_count < 0:
            raise ValueError(
                f"VoiceInterval.sample_count < 0: {self.sample_count}"
            )
        return self

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


class SilenceSegment(BaseModel):
    """Segmento de silencio (complementario de VoiceIntervals). Frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_ms: MsTimestamp
    end_ms: MsTimestamp
    speech_index_before: int = Field(-1, ge=-1)
    speech_index_after: int = Field(-1, ge=-1)

    @model_validator(mode="after")
    def _validate(self) -> "SilenceSegment":
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"SilenceSegment end_ms<start_ms: [{self.start_ms}, {self.end_ms}]"
            )
        return self

    @property
    def duration_ms(self) -> int:
        return int(self.end_ms) - int(self.start_ms)


# ---------------------------------------------------------------------------
# 3) VadThresholds (configuración centralizada VAD)
# ---------------------------------------------------------------------------


class VadThresholds(BaseModel):
    """Configuración VAD centralizada. Defaults estrictos T04.

    speech_threshold           = 0.5
    min_speech_duration_ms     = 200
    min_silence_between_ms     = 300
    merge_proximal_ms          = 200
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    speech_threshold: SpeechConfidence = Field(
        0.5, description="Probabilidad mínima por frame para considerar voz. [0,1]"
    )
    min_speech_duration_ms: int = Field(
        200, ge=20, le=10_000, description="Duración mínima (ms) para ser intervalo."
    )
    min_silence_between_ms: int = Field(
        300,
        ge=0,
        le=20_000,
        description="Silencio mínimo (ms) entre intervalos para NO unirlos pre-filter.",
    )
    merge_proximal_ms: int = Field(
        200,
        ge=0,
        le=10_000,
        description="Separación máxima (ms) para MREGAR dos intervalos vecinos post-filter.",
    )

    @field_validator("merge_proximal_ms")
    @classmethod
    def _merge_must_be_sensible(cls, v: int) -> int:
        if v > 5_000:
            raise ValueError(
                f"merge_proximal_ms excesivo {v}ms (risk merge everything). Máx 5000ms recomendado."
            )
        return v


__all__ = [
    "MsTimestamp",
    "SpeechConfidence",
    "VoiceInterval",
    "SilenceSegment",
    "VadThresholds",
]
