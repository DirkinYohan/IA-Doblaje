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
    """Step 07 Use Case.
    Clean Architecture:
      - sin faster_whisper / torch / numpy
      - sin acceso a WAV
      - depende ÚNICAMENTE de TimestampNormalizerPort

    Firma conceptual:
      run(preprocessed_audio, vad_result, lid_result, asr_result,
          thresholds=None, job_id=None) -> TimestampGenerationResult

    Responsabilidades:
      1) None inputs → ValidationFailedError
      2) 6-sha chain of custody mismatch → ValidationFailedError
      3) Safety límites (max_segments_safety) → ValidationFailedError
      4) Silent fallback (num_intervals==0 o ASR silent strategy) →
         NO invocar Adapter, construir directamente ts_silent_fallback
      5) Delegar normalización a TimestampNormalizerPort.normalize()
      6) Post-conditions: start<end / orden cronológico validados en Entity validator,
         no es necesario re-validar en UseCase.
      """

    timestamp_normalizer: TimestampNormalizerPort

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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
        # 1) None inputs sanity
        if preprocessed_audio is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: preprocessed_audio es None")
        if vad_result is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: vad_result es None")
        if language_detection_result is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: language_detection_result es None")
        if asr_result is None:
            raise ValidationFailedError("GenerateTimestampsUseCase: asr_result es None")

        # 2) Thresholds defaults
        effective_thresholds: AsrTimestampThresholds
        if thresholds is None:
            effective_thresholds = AsrTimestampThresholds()
        else:
            if not isinstance(thresholds, AsrTimestampThresholds):
                raise ValidationFailedError(
                    "GenerateTimestampsUseCase: thresholds no es AsrTimestampThresholds"
                )
            effective_thresholds = thresholds

        # 3) Chain of Custody 4-SHA (T03→T06). T07 añadirá asr_reference al binding.
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
        # 3-way adicional: vad/lid/asr deben referenciar mismo prep
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

        # 4) Safety: max_segments
        total_segments = int(asr_result.num_segments)
        if total_segments > int(effective_thresholds.max_segments_safety):
            raise ValidationFailedError(
                f"T07 safety: num_segments={total_segments} > "
                f"max_segments_safety={effective_thresholds.max_segments_safety}"
            )

        # 5) Silent fallback D#2 (NO invocar adapter)
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

        # 6) Delegar en TimestampNormalizerPort
        result: TimestampGenerationResult = self.timestamp_normalizer.normalize(
            preprocessed_audio,
            vad_result=vad_result,
            lid_result=language_detection_result,
            asr_result=asr_result,
            thresholds=effective_thresholds,
            job_id=job_id,
        )

        # 7) Post-conditions básicas (construcción frozen ya valida el resto)
        if not isinstance(result, TimestampGenerationResult):
            raise ValidationFailedError(
                "T07 Adapter devolvió tipo no esperado: no es TimestampGenerationResult"
            )
        # 7.1) Strict mode: si strict_validity y strict_ok False / validation_errors not ()
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

        # 7.2) D#1 compliance: T07 SIEMPRE retorna words=()
        if len(result.words) != 0:
            raise ValidationFailedError(
                "T07 D#1 violation (WORD_TS_APLAZADO): len(words)!=0; "
                "word timestamps aplazados a BAL/QLT."
            )

        # 7.3) Determinismo: no es posible verificar sin rerun; pero se exige que
        #      Adapter sea puro. Se confía.
        return result
