"""UseCase RunStructuredJsonOutputUseCase — Capa APPLICATION (T13 Step 13).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO escribe filesystem directamente (delega al JsonOutputWriterPort).
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
from app.domain.entities.metrics import MetricsResult
from app.domain.entities.output import (
    MODEL_LABEL_OUTPUT_V1_LITERAL,
    StructuredJsonOutputResult,
)
from app.domain.entities.quality import QualityResult
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.entities.validation import ValidationResult
from app.domain.interfaces.output_ports import JsonOutputWriterPort
from app.domain.value_objects.output import OutputFileRecord

OUTPUT_FILENAMES: tuple[str, ...] = (
    "analysis.json",
    "transcript.json",
    "speakers.json",
    "segments.json",
    "metadata.json",
)


log = get_logger("app.application.use_cases.write_structured_json")


@dataclass(frozen=True, slots=True)
class RunStructuredJsonOutputUseCase:
    """Orquesta Step 13 Structured JSON Output. Inmutable."""

    writer: JsonOutputWriterPort

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        asr_result: ASRResult,
        timestamps: TimestampGenerationResult,
        diarization: DiarizationResult,
        alignment: AlignmentResult,
        quality: QualityResult,
        validation: ValidationResult,
        metrics: MetricsResult,
        *,
        output_directory: str,
        schema_version: str,
        job_id: str | None = None,
        indent: int = 2,
        ensure_ascii: bool = False,
        force_save: bool = False,
    ) -> StructuredJsonOutputResult:
        # 1) None inputs
        for name, obj in (
            ("asr_result", asr_result),
            ("timestamps", timestamps),
            ("diarization", diarization),
            ("alignment", alignment),
            ("quality", quality),
            ("validation", validation),
            ("metrics", metrics),
        ):
            if obj is None:
                raise ValidationFailedError(f"RunStructuredJsonOutputUseCase: {name} es None")

        bind_context(job_id=job_id, phase="OUTPUT", model=MODEL_LABEL_OUTPUT_V1_LITERAL)
        t0 = time.perf_counter()

        # 2) Construir los 5 documentos en orden determinista
        documents: list[tuple[str, dict[str, Any]]] = [
            ("analysis.json", self._build_analysis(validation, quality, alignment, diarization, schema_version, job_id)),
            ("transcript.json", self._build_transcript(asr_result, timestamps, schema_version, job_id)),
            ("speakers.json", self._build_speakers(diarization, schema_version, job_id)),
            ("segments.json", self._build_segments(alignment, schema_version, job_id)),
            ("metadata.json", self._build_metadata(metrics, schema_version, job_id)),
        ]

        # 3) Delegar escritura al port (sin filesystem directo)
        records: list[OutputFileRecord] = []
        sha_map: dict[str, str] = {}
        for filename, doc in documents:
            sha = self.writer.write_json(
                output_directory=output_directory,
                filename=filename,
                document=doc,
                indent=indent,
                ensure_ascii=ensure_ascii,
            )
            records.append(OutputFileRecord(filename=filename, sha256=sha))
            sha_map[filename] = sha

        result = StructuredJsonOutputResult(
            job_id=job_id,
            output_directory=output_directory,
            written_files=tuple(records),
            file_sha256=sha_map,
            files_written_count=len(records),
            output_success=True,
            schema_version=schema_version,
            analysis_metadata={"force_save": bool(force_save)},
        )

        t1 = time.perf_counter()
        log.info(
            "write_structured_json_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "files_written": len(records),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Documentos
    # ------------------------------------------------------------------

    @staticmethod
    def _build_analysis(validation, quality, alignment, diarization, schema_version, job_id) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "step": "STRUCTURED_JSON_OUTPUT",
            "job_id": job_id,
            "validation_status": str(validation.validation_status),
            "quality_level": str(quality.quality_level.value) if hasattr(quality.quality_level, "value") else str(quality.quality_level),
            "overall_quality_score": float(quality.overall_quality_score),
            "validation_passed": bool(validation.validation_passed),
            "validation_errors": list(validation.validation_errors),
            "validation_warnings": list(validation.validation_warnings),
            "num_checks": int(validation.num_checks),
            "num_errors": int(validation.num_errors),
            "num_warnings": int(validation.num_warnings),
            "num_passed": int(validation.num_passed),
            "num_segments": int(alignment.num_segments),
            "num_segments_aligned": int(alignment.num_segments_aligned),
            "num_unassigned": int(alignment.num_unassigned),
            "num_speakers": int(diarization.num_speakers),
            "transcript_language_code": str(alignment.transcript_language_code),
            "total_duration_ms": int(alignment.total_duration_ms),
            "analysis_metadata": {},
        }

    @staticmethod
    def _build_transcript(asr_result, timestamps, schema_version, job_id) -> dict[str, Any]:
        # timestamps oficiales T07 para cada segmento; texto desde ASR T06.
        ts_by_index = {int(s.segment_index): s for s in timestamps.segments}
        asr_by_index = {int(s.segment_index): s for s in asr_result.segments}
        segments = []
        for idx, ts_seg in sorted(ts_by_index.items()):
            asr_seg = asr_by_index.get(idx)
            segments.append({
                "segment_index": idx,
                "start_ms": int(ts_seg.start_ms),
                "end_ms": int(ts_seg.end_ms),
                "duration_ms": int(ts_seg.duration_ms),
                "text": str(asr_seg.text) if asr_seg is not None else "",
                "words": [],
            })
        return {
            "schema_version": schema_version,
            "job_id": job_id,
            "transcript_text": str(asr_result.transcript_text),
            "transcript_language_code": str(asr_result.transcript_language_code),
            "asr_confidence": float(asr_result.confidence),
            "words": [],
            "segments": segments,
        }

    @staticmethod
    def _build_speakers(diarization, schema_version, job_id) -> dict[str, Any]:
        speakers = [
            {
                "speaker_label": str(t.speaker_label),
                "start_ms": int(t.start_ms),
                "end_ms": int(t.end_ms),
                "duration_ms": int(t.duration_ms),
                "confidence": float(t.confidence),
            }
            for t in diarization.speaker_turns
        ]
        return {
            "schema_version": schema_version,
            "job_id": job_id,
            "num_speakers": int(diarization.num_speakers),
            "num_turns": int(diarization.num_turns),
            "speakers": speakers,
        }

    @staticmethod
    def _build_segments(alignment, schema_version, job_id) -> dict[str, Any]:
        segments = [
            {
                "segment_index": int(s.segment_index),
                "start_ms": int(s.start_ms),
                "end_ms": int(s.end_ms),
                "duration_ms": int(s.duration_ms),
                "text": str(s.text),
                "speaker_label": str(s.speaker_label),
                "alignment_confidence": float(s.alignment_confidence),
                "ts_confidence": float(s.ts_confidence),
            }
            for s in alignment.dialogue_segments
        ]
        return {
            "schema_version": schema_version,
            "job_id": job_id,
            "num_segments": int(alignment.num_segments),
            "num_segments_aligned": int(alignment.num_segments_aligned),
            "num_unassigned": int(alignment.num_unassigned),
            "segments": segments,
        }

    @staticmethod
    def _build_metadata(metrics, schema_version, job_id) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "job_id": job_id,
            "processing_wall_time_sec": float(metrics.processing_wall_time_sec),
            "audio_duration_sec": float(metrics.audio_duration_sec),
            "real_time_factor_RTF": (
                float(metrics.real_time_factor_RTF)
                if metrics.real_time_factor_RTF is not None
                else None
            ),
            "step_timings_sec": dict(metrics.step_timings_sec),
            "peak_cpu_rss_mb": float(metrics.peak_cpu_rss_mb),
            "peak_gpu_vram_used_mb": float(metrics.peak_gpu_vram_used_mb),
            "peak_gpu_vram_total_mb": float(metrics.peak_gpu_vram_total_mb),
            "cpu_utilization_avg_percent": float(metrics.cpu_utilization_avg_percent),
            "gpu_utilization_avg_percent": float(metrics.gpu_utilization_avg_percent),
            "segment_count_total": int(metrics.segment_count_total),
            "segment_count_with_text": int(metrics.segment_count_with_text),
            "word_count_total": int(metrics.word_count_total),
            "word_count_confidence_ge_08": int(metrics.word_count_confidence_ge_08),
            "speaker_count_detected": int(metrics.speaker_count_detected),
            "silence_segment_count": int(metrics.silence_segment_count),
            "average_asr_confidence": float(metrics.average_asr_confidence),
            "average_diarization_confidence": float(metrics.average_diarization_confidence),
            "language_detection_confidence": float(metrics.language_detection_confidence),
            "overall_quality_score": float(metrics.overall_quality_score),
            "profile_requested": str(metrics.profile_requested),
            "profile_applied": str(metrics.profile_applied),
            "profile_downgrade_applied": bool(metrics.profile_downgrade_applied),
            "profile_downgrade_reason": str(metrics.profile_downgrade_reason),
            "device_requested": str(metrics.device_requested),
            "device_used": str(metrics.device_used),
        }


__all__ = ["RunStructuredJsonOutputUseCase", "OUTPUT_FILENAMES"]
