"""UseCase RunMetricsAggregationUseCase — Capa APPLICATION (T12 Step 12 Metrics).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa psutil/torch/pyannote/numpy/speechbrain (clean architecture).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import ValidationFailedError
from app.core.logging import bind_context, get_logger
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.metrics import (
    MODEL_LABEL_METRICS_V1_LITERAL,
    MetricsResult,
)
from app.domain.entities.quality import QualityResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.metrics import RuntimeMetricsSnapshot


log = get_logger("app.application.use_cases.aggregate_metrics")


@dataclass(frozen=True, slots=True)
class RunMetricsAggregationUseCase:
    """Orquesta Step 12 Metrics Aggregation. Inmutable.

    Input: snapshot + entidades upstream. Output: MetricsResult frozen 2-SHA.
    """

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        snapshot: RuntimeMetricsSnapshot,
        language: LanguageDetectionResult,
        asr_result: ASRResult,
        vad: VadResult,
        diarization: DiarizationResult,
        alignment: AlignmentResult,
        quality: QualityResult,
        validation: ValidationResult,
        *,
        job_id: str | None = None,
        metrics_enabled: bool = True,
    ) -> MetricsResult:
        # 1) None inputs
        for name, obj in (
            ("snapshot", snapshot),
            ("language", language),
            ("asr_result", asr_result),
            ("vad", vad),
            ("diarization", diarization),
            ("alignment", alignment),
            ("quality", quality),
            ("validation", validation),
        ):
            if obj is None:
                raise ValidationFailedError(f"RunMetricsAggregationUseCase: {name} es None")

        bind_context(job_id=job_id, phase="METRICS", model=MODEL_LABEL_METRICS_V1_LITERAL)
        t0 = time.perf_counter()

        # 2) Chain of custody
        source = str(alignment.source_preprocessed_sha256).lower()
        if str(validation.source_preprocessed_sha256).lower() != source:
            raise ValidationFailedError(
                "Chain of Custody rota T11↔T12: ValidationResult.source != alignment.source."
            )

        # 3) Disabled metrics
        if not metrics_enabled:
            return self._build_disabled(source, job_id)

        # 4) Runtime obligatorio
        if snapshot.processing_wall_time_sec <= 0:
            raise ValidationFailedError(
                "RunMetricsAggregationUseCase: processing_wall_time_sec obligatorio y > 0."
            )

        # 5) Métricas derivadas
        audio_duration_sec = round(float(alignment.total_duration_ms) / 1000.0, 6)
        rtf = (
            round(float(snapshot.processing_wall_time_sec) / audio_duration_sec, 6)
            if audio_duration_sec > 0
            else None
        )

        segment_count_total = int(alignment.num_segments)
        segment_count_with_text = sum(
            1 for seg in alignment.dialogue_segments if str(seg.text).strip()
        )

        word_count_total = 0
        for seg in asr_result.segments:
            word_count_total += len(str(seg.text).strip().split())

        speaker_count_detected = int(diarization.num_speakers)
        silence_segment_count = int(len(vad.silence_segments))

        average_asr_confidence = round(float(asr_result.confidence), 6)
        turns = diarization.speaker_turns
        average_diarization_confidence = (
            round(sum(float(t.confidence) for t in turns) / len(turns), 6)
            if turns
            else 0.0
        )
        language_detection_confidence = round(float(language.confidence), 6)
        overall_quality_score = round(float(quality.overall_quality_score), 6)

        # word_count_confidence_ge_08: no disponible (WORD_TS_APLAZADO)
        word_count_confidence_ge_08 = 0
        metadata: dict[str, object] = {"word_confidence_unavailable": True}

        result = MetricsResult(
            source_preprocessed_sha256=source,
            validation_reference_sha256=source,
            job_id=job_id,
            metrics_enabled=True,
            processing_wall_time_sec=float(snapshot.processing_wall_time_sec),
            audio_duration_sec=audio_duration_sec,
            real_time_factor_RTF=rtf,
            step_timings_sec=dict(snapshot.step_timings_sec),
            peak_cpu_rss_mb=float(snapshot.peak_cpu_rss_mb),
            peak_gpu_vram_used_mb=float(snapshot.peak_gpu_vram_used_mb),
            peak_gpu_vram_total_mb=float(snapshot.peak_gpu_vram_total_mb),
            cpu_utilization_avg_percent=float(snapshot.cpu_utilization_avg_percent),
            gpu_utilization_avg_percent=float(snapshot.gpu_utilization_avg_percent),
            segment_count_total=segment_count_total,
            segment_count_with_text=segment_count_with_text,
            word_count_total=word_count_total,
            word_count_confidence_ge_08=word_count_confidence_ge_08,
            speaker_count_detected=speaker_count_detected,
            silence_segment_count=silence_segment_count,
            average_asr_confidence=average_asr_confidence,
            average_diarization_confidence=average_diarization_confidence,
            language_detection_confidence=language_detection_confidence,
            overall_quality_score=overall_quality_score,
            profile_requested=str(snapshot.profile_requested),
            profile_applied=str(snapshot.profile_applied),
            profile_downgrade_applied=bool(snapshot.profile_downgrade_applied),
            profile_downgrade_reason=str(snapshot.profile_downgrade_reason),
            device_requested=str(snapshot.device_requested),
            device_used=str(snapshot.device_used),
            analysis_metadata=metadata,
        )

        t1 = time.perf_counter()
        log.info(
            "aggregate_metrics_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "rtf": rtf,
                "speakers": speaker_count_detected,
                "segments": segment_count_total,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_disabled(source: str, job_id: str | None) -> MetricsResult:
        """metrics_enabled=False → resultado deshabilitado (sin inventar métricas)."""
        return MetricsResult(
            source_preprocessed_sha256=source,
            validation_reference_sha256=source,
            job_id=job_id,
            metrics_enabled=False,
            analysis_metadata={"disabled": True},
        )


__all__ = ["RunMetricsAggregationUseCase"]
