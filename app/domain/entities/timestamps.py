from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.constants import PipelineStep
from app.domain.entities.vad import VadResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.asr import ASRResult
from app.domain.value_objects.audio import Sha256Hash
from app.domain.value_objects.timestamps import (
    DurationMs,
    TimestampConfidence01,
    TimestampSourceKind,
    TimestampStrategyName,
    TimestampedSegmentIndex,
    TimestampedWordIndex,
    TimedEndMs,
    TimedStartMs,
    WordTimestampSourceKind,
)


class TimestampedWord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_index_ref: TimestampedSegmentIndex
    word_index_in_segment: TimestampedWordIndex
    word_text: str = Field(min_length=0, max_length=512)
    start_ms: TimedStartMs | None
    end_ms: TimedEndMs | None
    source: WordTimestampSourceKind = "none"


class TimestampedSegment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    segment_index: TimestampedSegmentIndex
    start_ms: TimedStartMs
    end_ms: TimedEndMs
    duration_ms: DurationMs
    source: TimestampSourceKind
    overlap_normalized: bool = False
    gap_filled: bool = False
    end_clamped_to_total_duration: bool = False
    vad_interpolated: bool = False
    ts_confidence: TimestampConfidence01 = 1.0

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"TimestampedSegment idx={self.segment_index} end<=start: "
                f"end_ms={self.end_ms} start_ms={self.start_ms}"
            )
        expected_dur = self.end_ms - self.start_ms
        if self.duration_ms != expected_dur:
            raise ValueError(
                f"TimestampedSegment idx={self.segment_index} duration_ms mismatch: "
                f"declared={self.duration_ms}, computed={expected_dur}"
            )
        return self


class TimestampGenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    step: PipelineStep = PipelineStep.TIMESTAMPS_GENERATION
    model_label: Literal["t07:timestamp-normalizer:v1"] = "t07:timestamp-normalizer:v1"
    strategy: TimestampStrategyName

    source_preprocessed_sha256: Sha256Hash
    vad_reference_sha256: Sha256Hash
    lid_reference_sha256: Sha256Hash
    asr_reference_sha256: Sha256Hash

    total_duration_ms: DurationMs
    transcript_language_code: str = Field(min_length=2, max_length=16)
    job_id: str | None = Field(default=None, min_length=8, max_length=128)

    segments: tuple[TimestampedSegment, ...] = Field(default_factory=tuple)
    words: tuple[TimestampedWord, ...] = Field(default_factory=tuple)

    num_segments_ts: int = Field(ge=0, default=0)
    num_words_ts: int = Field(ge=0, default=0)
    num_overlaps_resolved: int = Field(ge=0, default=0)
    num_gaps_filled: int = Field(ge=0, default=0)
    num_end_clamped: int = Field(ge=0, default=0)
    num_vad_interpolated: int = Field(ge=0, default=0)
    num_fw_none_startms_fixed: int = Field(ge=0, default=0)
    validation_errors: tuple[str, ...] = Field(default_factory=tuple)
    strict_ok: bool = True

    transcript_text: str = Field(default="", max_length=1_000_000)
    confidence: TimestampConfidence01 = 0.0

    @model_validator(mode="after")
    def _post_validate(self) -> Self:
        if self.num_segments_ts != len(self.segments):
            raise ValueError(
                f"num_segments_ts={self.num_segments_ts} != len(segments)={len(self.segments)}"
            )
        if self.num_words_ts != len(self.words):
            raise ValueError(
                f"num_words_ts={self.num_words_ts} != len(words)={len(self.words)}"
            )
        last_idx = -1
        for seg in self.segments:
            if seg.segment_index != last_idx + 1:
                raise ValueError(
                    f"TimestampedSegment index discontinuo en idx={seg.segment_index}, "
                    f"esperado {last_idx + 1}"
                )
            last_idx = int(seg.segment_index)
        for i, seg in enumerate(self.segments):
            if int(seg.end_ms) > int(self.total_duration_ms):
                raise ValueError(
                    f"segment idx={i} end_ms={seg.end_ms} > total_duration_ms={self.total_duration_ms}"
                )
            if i > 0:
                prev_end = int(self.segments[i - 1].end_ms)
                curr_start = int(seg.start_ms)
                if curr_start < prev_end:
                    raise ValueError(
                        f"Overlap idx={i-1, i}: prev_end={prev_end} curr_start={curr_start} (post-normalización)"
                    )
        if self.strategy == "ts_silent_fallback":
            if self.segments:
                raise ValueError("ts_silent_fallback debe tener segments=()")
            if self.words:
                raise ValueError("ts_silent_fallback debe tener words=()")
            if self.transcript_text:
                raise ValueError("ts_silent_fallback debe tener transcript_text=''")
            if self.confidence != 0.0:
                raise ValueError("ts_silent_fallback debe tener confidence=0.0")
        if self.strict_ok and self.validation_errors:
            raise ValueError(
                f"strict_ok=True con validation_errors={list(self.validation_errors)}"
            )
        return self

    @classmethod
    def build_chain(
        cls,
        *,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        language_detection_result: LanguageDetectionResult,
        asr_result: ASRResult,
        strategy: TimestampStrategyName,
        transcript_language_code: str,
        segments: tuple[TimestampedSegment, ...],
        words: tuple[TimestampedWord, ...],
        job_id: str | None,
        num_overlaps_resolved: int = 0,
        num_gaps_filled: int = 0,
        num_end_clamped: int = 0,
        num_vad_interpolated: int = 0,
        num_fw_none_startms_fixed: int = 0,
        validation_errors: tuple[str, ...] = (),
        strict_ok: bool = True,
        transcript_text: str = "",
        confidence: float = 0.0,
    ) -> "TimestampGenerationResult":
        return cls(
            strategy=strategy,
            source_preprocessed_sha256=preprocessed_audio.sha256,
            vad_reference_sha256=vad_result.source_preprocessed_sha256,
            lid_reference_sha256=language_detection_result.source_preprocessed_sha256,
            asr_reference_sha256=asr_result.source_preprocessed_sha256,
            total_duration_ms=int(preprocessed_audio.duration_sec * 1000),
            transcript_language_code=transcript_language_code,
            job_id=job_id,
            segments=segments,
            words=words,
            num_segments_ts=len(segments),
            num_words_ts=len(words),
            num_overlaps_resolved=num_overlaps_resolved,
            num_gaps_filled=num_gaps_filled,
            num_end_clamped=num_end_clamped,
            num_vad_interpolated=num_vad_interpolated,
            num_fw_none_startms_fixed=num_fw_none_startms_fixed,
            validation_errors=tuple(validation_errors),
            strict_ok=strict_ok,
            transcript_text=transcript_text,
            confidence=confidence,
        )
