from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import BeforeValidator


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def _validate_asr_text_nonbinary(v: object) -> str:
    if isinstance(v, bool):
        raise TypeError("ASRText no admite bool")
    if not isinstance(v, str):
        raise TypeError(f"ASRText requiere str, recibido {type(v).__name__}")
    if _CONTROL_CHARS_RE.search(v):
        raise ValueError("ASRText contiene caracteres de control no permitidos")
    if len(v) > 1_000_000:
        raise ValueError("ASRText excede longitud máxima 1.000.000 caracteres")
    return v


ASRText = Annotated[str, BeforeValidator(_validate_asr_text_nonbinary)]


def _validate_transcript_confidence_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise TypeError("TranscriptConfidence no admite bool")
    if not isinstance(v, (int, float)):
        raise TypeError(
            f"TranscriptConfidence requiere float/int, recibido {type(v).__name__}"
        )
    f = float(v)
    import math

    if math.isnan(f) or math.isinf(f):
        raise ValueError("TranscriptConfidence no admite NaN o Inf")
    if f < 0.0 or f > 1.0:
        raise ValueError(f"TranscriptConfidence fuera de rango [0,1]: {f!r}")
    return f


TranscriptConfidence = Annotated[float, BeforeValidator(_validate_transcript_confidence_0_1)]


def _validate_asr_segment_index_ge0(v: object) -> int:
    if isinstance(v, bool):
        raise TypeError("ASRSegmentIndex no admite bool")
    if not isinstance(v, int):
        raise TypeError(f"ASRSegmentIndex requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"ASRSegmentIndex debe ser >=0, recibido {v!r}")
    return v


ASRSegmentIndex = Annotated[int, BeforeValidator(_validate_asr_segment_index_ge0)]


def _validate_minutes_positive_nonbool(v: object) -> int:
    if isinstance(v, bool):
        raise TypeError("max_audio_duration_minutes no admite bool")
    if not isinstance(v, int):
        raise TypeError(f"max_audio_duration_minutes requiere int, recibido {type(v).__name__}")
    if v <= 0:
        raise ValueError(f"max_audio_duration_minutes debe ser >0, recibido {v!r}")
    return v


AsrMaxDurationMinutes = Annotated[int, BeforeValidator(_validate_minutes_positive_nonbool)]


class AsrThresholds(BaseModel):
    """Thresholds configurables ASR T06 (Step 06)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_transcript_confidence: TranscriptConfidence = Field(
        default=0.0,
        description=(
            "Confianza mínima [0,1] para aceptar resultado ASR cuando strict=True. "
            "strict=True + confidence<valor → ValidationFailedError."
        ),
    )

    beam_size: ASRSegmentIndex = Field(
        default=1,
        description="Beam size decoder PERF=1 greedy, BAL/QLT >=3.",
    )

    max_audio_duration_minutes: AsrMaxDurationMinutes = Field(
        default=480,
        description="Duración máxima audio en minutos para anti memory exhaustion.",
    )

    allow_empty_transcript_when_no_voice: bool = Field(
        default=True,
        description="Cuando VadResult.num_intervals==0, retornar silent fallback (True).",
    )


ASRStrategyName = Literal["full_audio", "only_voice_concat", "asr_silent_fallback"]
