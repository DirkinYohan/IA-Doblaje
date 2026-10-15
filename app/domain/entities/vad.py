"""Entidad VadResult Step 04 VAD — Capa DOMAIN.

Frozen + extra=forbid. Chain of custody via source_preprocessed_sha256.
Sin dependencias infraestructura.
"""
from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.domain.value_objects.audio import SampleRate, ChannelCount, Sha256Hash
from app.domain.value_objects.vad import (
    MsTimestamp,
    SilenceSegment,
    VadThresholds,
    VoiceInterval,
)


MODEL_LABEL_SILERO_V5_1_LITERAL: Literal["silero-vad:v5.1"] = "silero-vad:v5.1"


class VadResult(BaseModel):
    """Resultado Step 04 Voice Activity Detection. Inmutable. Chain of Custody."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),  # permite field name = model_label (no conflict pydantic internal model_ prefix)
    )

    source_preprocessed_sha256: Sha256Hash = Field(
        ..., description="Enlace T03 → T04, chain of custody SHA256 WAV."
    )
    voice_intervals: tuple[VoiceInterval, ...] = Field(
        default_factory=tuple, description="Ordenados por start_ms ASC. Sin overlaps."
    )
    silence_segments: tuple[SilenceSegment, ...] = Field(
        default_factory=tuple, description="Complementario (0..N). Ordenados ASC."
    )
    speech_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="total_speech_ms / duration_ms."
    )
    total_speech_ms: MsTimestamp = Field(0)
    total_silence_ms: MsTimestamp = Field(0)
    num_intervals: int = Field(0, ge=0)
    num_silences: int = Field(0, ge=0)
    thresholds: VadThresholds = Field(default_factory=VadThresholds)
    model_label: Literal["silero-vad:v5.1"] = Field(
        MODEL_LABEL_SILERO_V5_1_LITERAL, description="Etiqueta de modelo aplicado (fix T04)."
    )
    sample_rate: SampleRate = Field(16000)
    channels: ChannelCount = Field(1)
    duration_ms: MsTimestamp = Field(0, description="Duración total audio de entrada (ms).")

    @model_validator(mode="after")
    def _validate(self) -> "VadResult":
        # (A) intervals sorted asc, start < end, no overlap
        prev_end: int = -1
        for i, vi in enumerate(self.voice_intervals):
            if not int(vi.start_ms) >= 0:
                raise ValueError(f"voice_intervals[{i}].start_ms<0")
            if not int(vi.end_ms) > int(vi.start_ms):
                raise ValueError(f"voice_intervals[{i}] end<=start")
            if int(vi.start_ms) < prev_end:
                raise ValueError(
                    f"voice_intervals overlap at [{i}] start={vi.start_ms} < prev_end={prev_end}"
                )
            if self.duration_ms > 0 and int(vi.end_ms) > int(self.duration_ms):
                raise ValueError(
                    f"voice_intervals[{i}].end_ms={vi.end_ms} > duration_ms={self.duration_ms}"
                )
            prev_end = int(vi.end_ms)

        # (B) silence segments sorted asc, no overlap con speech intervals
        prev_s_end: int = -1
        for j, si in enumerate(self.silence_segments):
            if int(si.start_ms) < prev_s_end:
                raise ValueError(f"silence_segments[{j}] overlap/unsorted")
            if int(si.end_ms) > int(self.duration_ms) and self.duration_ms > 0:
                raise ValueError(
                    f"silence_segments[{j}].end={si.end_ms} > duration={self.duration_ms}"
                )
            prev_s_end = int(si.end_ms)

        # (C) total speech + silence debe concordar (~ duración ms, tolerancia 2 ms por redondeos)
        if int(self.duration_ms) > 0:
            diff = abs(
                (int(self.total_speech_ms) + int(self.total_silence_ms))
                - int(self.duration_ms)
            )
            if diff > 2:
                raise ValueError(
                    "total_speech+total_silence no coincide con duration_ms: "
                    f"{self.total_speech_ms}+{self.total_silence_ms} != {self.duration_ms} diff={diff}"
                )

        # (D) totals congruentes
        ts = sum(int(v.duration_ms) for v in self.voice_intervals)
        if ts != int(self.total_speech_ms):
            raise ValueError(
                f"total_speech_ms={self.total_speech_ms} != suma intervals={ts}"
            )
        tsil = sum(int(s.duration_ms) for s in self.silence_segments)
        if tsil != int(self.total_silence_ms):
            raise ValueError(
                f"total_silence_ms={self.total_silence_ms} != suma silences={tsil}"
            )

        # (E) speech_ratio
        if int(self.duration_ms) > 0:
            expected = round(ts / int(self.duration_ms), 6)
            actual = round(float(self.speech_ratio), 6)
            if abs(expected - actual) > 0.02:
                raise ValueError(
                    f"speech_ratio incorrecto actual={actual}, esperado aprox={expected}"
                )

        if len(self.voice_intervals) != int(self.num_intervals):
            raise ValueError("num_intervals != len(voice_intervals)")
        if len(self.silence_segments) != int(self.num_silences):
            raise ValueError("num_silences != len(silence_segments)")

        # (F) source sha 64 hex lower
        sha = str(self.source_preprocessed_sha256).lower().strip()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(
                f"source_preprocessed_sha256 invalido 64-hex lower: {sha!r}"
            )
        object.__setattr__(self, "source_preprocessed_sha256", sha)
        return self


__all__ = ["VadResult", "MODEL_LABEL_SILERO_V5_1_LITERAL"]
