"""Entidad AlignmentResult Step 09 — ASR ↔ Speaker Alignment. Capa DOMAIN.

Frozen + extra=forbid. Chain of custody 6-SHA:
  source_preprocessed_sha256 (T03)
  vad_reference_sha256 (T04)
  lid_reference_sha256 (T05)
  asr_reference_sha256 (T06)
  ts_reference_sha256 (T07)
  diarization_reference_sha256 (T08)

Sin dependencias infraestructura (no torch/pyannote/numpy).
"""
from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.core.constants import PipelineStep
from app.domain.value_objects.audio import Sha256Hash
from app.domain.value_objects.alignment import (
    AlignmentMs,
    AlignmentStrategyName,
    DialogueSegment,
)

ALIGNMENT_STRATEGY_WOMV: Literal["weighted_overlap_majority_vote"] = (
    "weighted_overlap_majority_vote"
)
ALIGNMENT_STRATEGY_SILENT_FALLBACK: Literal["alignment_silent_fallback"] = (
    "alignment_silent_fallback"
)

ALIGNMENT_STRATEGIES_TYPE = (
    Literal["weighted_overlap_majority_vote"]
    | Literal["alignment_silent_fallback"]
)

MODEL_LABEL_ALIGNMENT_V1_LITERAL: Literal[
    "t09:asr-speaker-alignment:v1"
] = "t09:asr-speaker-alignment:v1"


class AlignmentResult(BaseModel):
    """Resultado Step 09. Inmutable. Chain of Custody 6-SHA."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    # ---- Step identifiers ------------------------------------------------
    step: PipelineStep = PipelineStep.ASR_SPEAKER_ALIGNMENT
    model_label: Literal["t09:asr-speaker-alignment:v1"] = MODEL_LABEL_ALIGNMENT_V1_LITERAL
    strategy: AlignmentStrategyName

    # ---- Chain of Custody 6-SHA ------------------------------------------
    source_preprocessed_sha256: Sha256Hash
    vad_reference_sha256: Sha256Hash
    lid_reference_sha256: Sha256Hash
    asr_reference_sha256: Sha256Hash
    ts_reference_sha256: Sha256Hash
    diarization_reference_sha256: Sha256Hash

    # ---- Metadata T03/T07 passthrough ------------------------------------
    total_duration_ms: AlignmentMs
    transcript_language_code: str = Field(min_length=2, max_length=16)
    job_id: str | None = Field(default=None, min_length=8, max_length=128)

    # ---- Outputs ---------------------------------------------------------
    dialogue_segments: tuple[DialogueSegment, ...] = Field(
        default_factory=tuple, description="Segmentos alineados ASC, sin speaker artificial."
    )
    num_segments: int = Field(0, ge=0, description="Total de TimestampedSegment T07.")
    num_segments_aligned: int = Field(0, ge=0, description="Segmentos con speaker asignado.")
    num_unassigned: int = Field(0, ge=0, description="Segmentos sin speaker (sin overlap).")

    # ---- Resumen auditoría -----------------------------------------------
    validation_errors: tuple[str, ...] = Field(default_factory=tuple)
    strict_ok: bool = True

    # ======================================================================
    # VALIDADORES
    # ======================================================================

    @field_validator(
        "source_preprocessed_sha256",
        "vad_reference_sha256",
        "lid_reference_sha256",
        "asr_reference_sha256",
        "ts_reference_sha256",
        "diarization_reference_sha256",
    )
    @classmethod
    def _sha_64hex_lower(cls, v: object) -> str:
        s = str(v or "").strip().lower()
        if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
            raise ValueError(f"SHA inválido 64-hex lower: {s[:20]!r}...")
        return s

    @model_validator(mode="after")
    def _validate_all(self) -> "AlignmentResult":
        # (A) Chain of custody: las 5 primeras referencias deben coincidir con source (T03)
        source = str(self.source_preprocessed_sha256)
        refs = {
            "vad_reference_sha256": str(self.vad_reference_sha256),
            "lid_reference_sha256": str(self.lid_reference_sha256),
            "asr_reference_sha256": str(self.asr_reference_sha256),
            "ts_reference_sha256": str(self.ts_reference_sha256),
        }
        for name, value in refs.items():
            if value != source:
                raise ValueError(
                    f"Chain of Custody rota T09: {name} != source_preprocessed_sha256."
                )

        # (B) num consistency
        if self.num_segments_aligned != len(self.dialogue_segments):
            raise ValueError(
                f"num_segments_aligned={self.num_segments_aligned} != "
                f"len(dialogue_segments)={len(self.dialogue_segments)}"
            )
        if self.num_segments != self.num_segments_aligned + self.num_unassigned:
            raise ValueError(
                f"num_segments={self.num_segments} != "
                f"num_segments_aligned({self.num_segments_aligned}) + "
                f"num_unassigned({self.num_unassigned})"
            )

        # (C) Orden ASC por segment_index contiguo y por start_ms, sin overlaps
        last_idx = -1
        prev_end = -1
        for i, seg in enumerate(self.dialogue_segments):
            if int(seg.segment_index) != last_idx + 1:
                raise ValueError(
                    f"DialogueSegment index discontinuo: esperado {last_idx + 1}, "
                    f"encontrado {seg.segment_index}"
                )
            if int(seg.end_ms) > int(self.total_duration_ms):
                raise ValueError(
                    f"DialogueSegment[{i}] end_ms={seg.end_ms} > total_duration_ms={self.total_duration_ms}"
                )
            if int(seg.start_ms) < prev_end:
                raise ValueError(
                    f"DialogueSegment overlap/unsorted en idx={i}: start={seg.start_ms} < prev_end={prev_end}"
                )
            last_idx = int(seg.segment_index)
            prev_end = int(seg.end_ms)

        # (D) Silent fallback consistency
        if self.strategy == ALIGNMENT_STRATEGY_SILENT_FALLBACK:
            if len(self.dialogue_segments) != 0:
                raise ValueError(
                    "alignment_silent_fallback requiere dialogue_segments=()"
                )
            if self.num_segments_aligned != 0:
                raise ValueError(
                    "alignment_silent_fallback requiere num_segments_aligned=0"
                )
            if self.num_unassigned != self.num_segments:
                raise ValueError(
                    "alignment_silent_fallback requiere num_unassigned == num_segments"
                )

        # (E) strict_ok implica validation_errors == ()
        if self.strict_ok and self.validation_errors:
            raise ValueError(
                f"strict_ok=True con validation_errors={list(self.validation_errors)}"
            )

        return self


__all__ = [
    "AlignmentResult",
    "ALIGNMENT_STRATEGY_WOMV",
    "ALIGNMENT_STRATEGY_SILENT_FALLBACK",
    "ALIGNMENT_STRATEGIES_TYPE",
    "MODEL_LABEL_ALIGNMENT_V1_LITERAL",
]
