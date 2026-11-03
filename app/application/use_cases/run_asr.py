from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.core.exceptions import (
    ASRProcessingError,
    ModelLoadError,
    ValidationFailedError,
)
from app.domain.entities.asr import ASRResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.interfaces.asr_ports import ASRPort
from app.domain.value_objects.asr import (
    ASRStrategyName,
    AsrThresholds,
)
AUDIO_16KHZ_SAMPLE_RATE: int = 16000
CHANNELS_MONO: int = 1
BIT_DEPTH_16: int = 16


@dataclass(frozen=True, slots=True, weakref_slot=False)
class RunASRUseCase:
    asr_port: ASRPort
    strict: bool = True
    default_thresholds: AsrThresholds | None = None

    def run(
        self,
        preprocessed_audio: PreprocessedAudio | None,
        vad_result: VadResult | None,
        language_detection_result: LanguageDetectionResult | None,
        *,
        thresholds: AsrThresholds | None = None,
        job_id: str | None = None,
    ) -> ASRResult:
        if preprocessed_audio is None:
            raise ValidationFailedError("RunASRUseCase: preprocessed_audio es None")
        if vad_result is None:
            raise ValidationFailedError("RunASRUseCase: vad_result es None")
        if language_detection_result is None:
            raise ValidationFailedError(
                "RunASRUseCase: language_detection_result es None"
            )

        if preprocessed_audio.sample_rate != AUDIO_16KHZ_SAMPLE_RATE:
            raise ValidationFailedError(
                f"RunASRUseCase requiere sample_rate={AUDIO_16KHZ_SAMPLE_RATE}, "
                f"recibido {preprocessed_audio.sample_rate}"
            )
        if preprocessed_audio.channels != CHANNELS_MONO:
            raise ValidationFailedError(
                f"RunASRUseCase requiere channels={CHANNELS_MONO}, "
                f"recibido {preprocessed_audio.channels}"
            )
        if preprocessed_audio.bit_depth != BIT_DEPTH_16:
            raise ValidationFailedError(
                f"RunASRUseCase requiere bit_depth={BIT_DEPTH_16}, "
                f"recibido {preprocessed_audio.bit_depth}"
            )

        effective_thresholds = self._resolve_thresholds(thresholds)

        prep_sha = preprocessed_audio.sha256
        vad_sha = vad_result.source_preprocessed_sha256
        lid_sha = language_detection_result.source_preprocessed_sha256
        if prep_sha != vad_sha or prep_sha != lid_sha:
            raise ValidationFailedError(
                "RunASRUseCase chain of custody SHA256 roto: "
                f"preprocessed={prep_sha} vad_ref={vad_sha} lid_ref={lid_sha}"
            )

        total_ms = int(preprocessed_audio.duration_sec * 1000)
        max_ms = int(effective_thresholds.max_audio_duration_minutes * 60 * 1000)
        if total_ms > max_ms:
            raise ValidationFailedError(
                f"RunASRUseCase audio duración {total_ms}ms > "
                f"máximo configurado {max_ms}ms ({effective_thresholds.max_audio_duration_minutes}min)"
            )

        if (
            vad_result.num_intervals == 0
            and effective_thresholds.allow_empty_transcript_when_no_voice
        ):
            return self._build_silent_fallback(
                preprocessed_audio=preprocessed_audio,
                vad_result=vad_result,
                lid_result=language_detection_result,
                thresholds=effective_thresholds,
                job_id=job_id,
            )

        try:
            asr_result = self.asr_port.transcribe(
                preprocessed_audio,
                vad_result=vad_result,
                lid_result=language_detection_result,
                thresholds=effective_thresholds,
                job_id=job_id,
            )
        except (ModelLoadError, ASRProcessingError, ValidationFailedError):
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ASRProcessingError(
                f"RunASRUseCase error ASR sin clasificar: {type(exc).__name__}: {exc}"
            ) from exc

        self._post_validate_result(
            asr_result,
            prep_sha=prep_sha,
            vad_sha=vad_sha,
            lid_sha=lid_sha,
            thresholds_used=effective_thresholds,
        )

        if self.strict and asr_result.confidence < effective_thresholds.min_transcript_confidence:
            if len(asr_result.segments) > 0 or asr_result.num_segments > 0:
                raise ValidationFailedError(
                    f"RunASRUseCase strict=True: confidence={asr_result.confidence:.4f} < "
                    f"min_transcript_confidence={effective_thresholds.min_transcript_confidence:.4f}"
                )

        return asr_result

    def _resolve_thresholds(self, incoming: AsrThresholds | None) -> AsrThresholds:
        if incoming is not None:
            return incoming
        if self.default_thresholds is not None:
            return self.default_thresholds
        return AsrThresholds()

    def _build_silent_fallback(
        self,
        *,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        lid_result: LanguageDetectionResult,
        thresholds: AsrThresholds,
        job_id: str | None,
    ) -> ASRResult:
        fallback_model_label = "silent-fallback:no-voice:v1"
        strategy: ASRStrategyName = "asr_silent_fallback"
        return ASRResult.build_chain(
            preprocessed_audio=preprocessed_audio,
            vad_result=vad_result,
            lid_result=lid_result,
            transcript_text="",
            transcript_language_code="und",
            confidence=0.0,
            segments=(),
            num_segments=0,
            strategy=strategy,
            model_label=fallback_model_label,
            thresholds_used=thresholds,
            job_id=job_id,
            analysis_metadata={
                "silent_fallback_reason": "vad_result.num_intervals==0",
                "vad_num_intervals": vad_result.num_intervals,
            },
        )

    def _post_validate_result(
        self,
        result: ASRResult,
        *,
        prep_sha: str,
        vad_sha: str,
        lid_sha: str,
        thresholds_used: AsrThresholds,
    ) -> None:
        if (
            result.source_preprocessed_sha256 != prep_sha
            or result.vad_reference_sha256 != vad_sha
            or result.lid_reference_sha256 != lid_sha
        ):
            raise ASRProcessingError(
                "RunASRUseCase resultado ASR chain of custody corrupto "
                "(adapter no respetó hashes)."
            )
        if result.strategy not in ("full_audio", "only_voice_concat", "asr_silent_fallback"):
            raise ASRProcessingError(
                f"RunASRUseCase strategy desconocida: {result.strategy!r}"
            )
        if result.thresholds_used is not thresholds_used:
            pass
