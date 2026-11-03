from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import ValidationFailedError
from app.domain.entities.asr import ASRResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.timestamps import (
    TimestampGenerationResult,
    TimestampedSegment,
    TimestampedWord,
)
from app.domain.entities.vad import VadResult
from app.domain.interfaces.timestamp_ports import TimestampNormalizerPort
from app.domain.value_objects.timestamps import (
    AsrTimestampThresholds,
)


@dataclass(frozen=True, slots=True)
class GenerateTimestampsUseCase:
    timestamp_normalizer: TimestampNormalizerPort

    def run(
        self,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        language_detection_result: LanguageDetectionResult,
        asr_result: ASRResult,
        *,
        thresholds: AsrTimestampThresholds | None = None,
        job_id: str | None = None,
    ) -> TimestampGenerationResult:
        if preprocessed_audio is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: preprocessed_audio es None")
        if vad_result is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: vad_result es None")
        if language_detection_result is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: language_detection_result es None")
        if asr_result is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: asr_result es None")

        effective_thresholds: AsrTimestampThresholds
        if thresholds is None:
            effective_thresholds = AsrTimestampThresholds()
        else:
            if not isinstance(thresholds, AsrTimestampThresholds):
                raise ValidationFailedError(
                    "GenerateTimestampsUseCase: thresholds no es AsrTimestampThresholds"
                )
            effective_thresholds = thresholds

        prep_sha = str(preprocessed_audio.sha256)
        vad_prep_sha = str(vad_result.source_preprocessed_sha256)
        lid_prep_sha = str(language_detection_result.source_preprocessed_sha256)
        asr_prep_sha = str(asr_result.source_preprocessed_sha256)
        if prep_sha != vad_prep_sha:
            raise ValidationFailedError(
                "Chain of custody T04→T03 mismatch: Prep.sha256 != VadResult.source_preprocessed_sha256"
            )
        if prep_sha != lid_prep_sha:
            raise ValidationFailedError(
                "Chain of custody T05→T03 mismatch: Prep.sha256 != LID.source_preprocessed_sha256"
            )
        if prep_sha != asr_prep_sha:
            raise ValidationFailedError(
                "Chain of custody T06→T03 mismatch: Prep.sha256 != ASR.source_preprocessed_sha256"
            )
        if vad_prep_sha != lid_prep_sha:
            raise ValidationFailedError(
                "Chain of custody T04-T05 mismatch: vad vs lid source_preprocessed_sha256"
            )
        if vad_prep_sha != asr_prep_sha:
            raise ValidationFailedError(
                "Chain of custody T04-T06 mismatch: vad vs asr source_preprocessed_sha256"
            )
        if lid_prep_sha != asr_prep_sha:
            raise ValidationFailedError(
                "Chain of custody T05-T06 mismatch: lid vs asr source_preprocessed_sha256"
            )

        total_segments = int(asr_result.num_segments)
        if total_segments > int(effective_thresholds.max_segments_safety):
            raise ValidationFailedError(
                f"T07 safety: num_segments={total_segments} > "
                f"max_segments_safety={effective_thresholds.max_segments_safety}"
            )

        is_silent = (
            int(vad_result.num_intervals) == 0
            or str(asr_result.strategy) == "asr_silent_fallback"
            or len(asr_result.segments) == 0
        )
        if is_silent:
            if not effective_thresholds.allow_silent_fallback:
                raise ValidationFailedError(
                    "T07 silent fallback detectado pero allow_silent_fallback=False"
                )
            return TimestampGenerationResult.build_chain(
                preprocessed_audio=preprocessed_audio,
                vad_result=vad_result,
                language_detection_result=language_detection_result,
                asr_result=asr_result,
                strategy="ts_silent_fallback",
                transcript_language_code=asr_result.transcript_language_code,
                segments=(),
                words=(),
                job_id=job_id,
                num_overlaps_resolved=0,
                num_gaps_filled=0,
                num_end_clamped=0,
                num_vad_interpolated=0,
                num_fw_none_startms_fixed=0,
                validation_errors=(),
                strict_ok=True,
                transcript_text="",
                confidence=0.0,
            )

        result: TimestampGenerationResult = self.timestamp_normalizer.normalize(
            preprocessed_audio,
            vad_result=vad_result,
            lid_result=language_detection_result,
            asr_result=asr_result,
            thresholds=effective_thresholds,
            job_id=job_id,
        )

        if not isinstance(result, TimestampGenerationResult):
            raise ValidationFailedError(
                "T07 Adapter devolvió tipo no esperado: no es TimestampGenerationResult"
            )
        if effective_thresholds.strict_validity:
            if not bool(result.strict_ok):
                raise ValidationFailedError(
                    f"T07 strict_validity=True con strict_ok=False; "
                    f"errors={list(result.validation_errors)}"
                )
            if tuple(result.validation_errors) != ():
                raise ValidationFailedError(
                    f"T07 strict_validity=True con validation_errors={list(result.validation_errors)}"
                )

        if len(result.words) != 0:
            raise ValidationFailedError(
                "T07 D#1 violation (WORD_TS_APLAZADO): len(words)!=0; "
                "word timestamps aplazados a BAL/QLT."
            )

        return result
