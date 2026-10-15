"""UseCase RunValidationUseCase — Capa APPLICATION (T11 Step 11 Validation).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa torch/pyannote/numpy/speechbrain (clean architecture).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.constants import QualityLevel
from app.core.exceptions import ValidationFailedError
from app.core.logging import bind_context, get_logger
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import (
    MODEL_LABEL_VALIDATION_V1_LITERAL,
    ValidationResult,
)
from app.domain.value_objects.validation import (
    ValidationCheckResult,
    ValidationThresholds,
)


log = get_logger("app.application.use_cases.validate_results")


@dataclass(frozen=True, slots=True)
class RunValidationUseCase:
    """Orquesta Step 11 Validation. Inmutable.

    Input: (QualityResult, AlignmentResult, VadResult, ASRResult, DiarizationResult).
    Output: ValidationResult frozen con chain of custody 8-SHA.
    """

    strict_quality: bool = True

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        quality: QualityResult,
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        *,
        thresholds: ValidationThresholds | None = None,
        job_id: str | None = None,
        force_save: bool = False,
    ) -> ValidationResult:
        # 1) None inputs
        for name, obj in (
            ("quality", quality),
            ("alignment", alignment),
            ("vad", vad),
            ("asr_result", asr_result),
            ("diarization", diarization),
        ):
            if obj is None:
                raise ValidationFailedError(f"RunValidationUseCase: {name} es None")

        bind_context(job_id=job_id, phase="VALIDATION", model=MODEL_LABEL_VALIDATION_V1_LITERAL)
        t0 = time.perf_counter()

        # 2) Thresholds
        eff: ValidationThresholds = thresholds or ValidationThresholds()
        if not isinstance(eff, ValidationThresholds):
            raise ValidationFailedError("RunValidationUseCase: thresholds no es ValidationThresholds")

        # 3) Política Quality FAILED (B5)
        if (
            quality.quality_level == QualityLevel.FAILED
            and eff.strict_quality
            and not force_save
        ):
            raise ValidationFailedError(
                f"RunValidationUseCase: Quality FAILED (score={quality.overall_quality_score:.4f}) "
                "y strict_quality=True sin force_save."
            )

        # 4) Ejecutar checks VC01..VC11 en orden determinista
        checks = self._evaluate_checks(
            quality=quality,
            alignment=alignment,
            vad=vad,
            asr_result=asr_result,
            diarization=diarization,
            thresholds=eff,
        )

        # 5) Agregar
        errors: list[str] = []
        warnings: list[str] = []
        for c in checks:
            if c.outcome == "error":
                errors.append(f"{c.check_code}: {c.message}")
            elif c.outcome == "warning":
                warnings.append(f"{c.check_code}: {c.message}")

        num_errors = len(errors)
        num_warnings = len(warnings)
        num_passed = sum(1 for c in checks if c.outcome == "passed")

        # status
        if num_errors > 0 or quality.quality_level == QualityLevel.FAILED:
            status: str = "failed"
        elif num_warnings > 0:
            status = "warning"
        else:
            status = "passed"

        # 6) Errores estructurales siempre abortan (force_save NO los ignora).
        #    Se lanzan ANTES de construir ValidationResult para evitar construir
        #    una entidad con cadena SHA rota (que produciría ValidationError).
        if num_errors > 0:
            raise ValidationFailedError(
                f"RunValidationUseCase: {num_errors} errores estructurales de validación: "
                f"{errors[:3]}"
            )

        result = ValidationResult(
            source_preprocessed_sha256=str(alignment.source_preprocessed_sha256),
            vad_reference_sha256=str(alignment.vad_reference_sha256),
            lid_reference_sha256=str(alignment.lid_reference_sha256),
            asr_reference_sha256=str(alignment.asr_reference_sha256),
            ts_reference_sha256=str(alignment.ts_reference_sha256),
            diarization_reference_sha256=str(alignment.diarization_reference_sha256),
            alignment_reference_sha256=str(alignment.source_preprocessed_sha256),
            quality_reference_sha256=str(quality.source_preprocessed_sha256),
            job_id=job_id,
            validation_passed=num_errors == 0,
            validation_status=status,  # type: ignore[arg-type]
            quality_level=quality.quality_level,
            overall_quality_score=float(quality.overall_quality_score),
            validation_errors=tuple(errors),
            validation_warnings=tuple(warnings),
            num_checks=11,
            num_errors=num_errors,
            num_warnings=num_warnings,
            num_passed=num_passed,
            check_results=tuple(checks),
            analysis_metadata={},
        )

        t1 = time.perf_counter()
        log.info(
            "validate_results_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "status": status,
                "errors": num_errors,
                "warnings": num_warnings,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Checks VC01..VC11
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_checks(
        *,
        quality: QualityResult,
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        thresholds: ValidationThresholds,
    ) -> list[ValidationCheckResult]:
        return [
            RunValidationUseCase._vc01(quality, alignment, vad, asr_result, diarization),
            RunValidationUseCase._vc02(quality, alignment, vad, asr_result, diarization),
            RunValidationUseCase._vc03(quality, alignment, vad),
            RunValidationUseCase._vc04(alignment),
            RunValidationUseCase._vc05(alignment),
            RunValidationUseCase._vc06(alignment, diarization),
            RunValidationUseCase._vc07(alignment, asr_result),
            RunValidationUseCase._vc08(alignment, asr_result),
            RunValidationUseCase._vc09(quality, thresholds),
            RunValidationUseCase._vc10(quality),
            RunValidationUseCase._vc11(quality, alignment),
        ]

    # VC01: SHA chain coherente
    @staticmethod
    def _vc01(quality, alignment, vad, asr_result, diarization) -> ValidationCheckResult:
        source = str(alignment.source_preprocessed_sha256).lower()
        mismatches = []
        if str(quality.source_preprocessed_sha256).lower() != source:
            mismatches.append("QualityResult")
        if str(vad.source_preprocessed_sha256).lower() != source:
            mismatches.append("VadResult")
        if str(asr_result.source_preprocessed_sha256).lower() != source:
            mismatches.append("ASRResult")
        if str(diarization.source_preprocessed_sha256).lower() != source:
            mismatches.append("DiarizationResult")
        if mismatches:
            return ValidationCheckResult(
                check_code="VC01", outcome="error",
                message=f"SHA chain mismatch: {', '.join(mismatches)}",
            )
        return ValidationCheckResult(check_code="VC01", outcome="passed", message="sha_chain_ok")

    # VC02: mismo job_id
    @staticmethod
    def _vc02(quality, alignment, vad, asr_result, diarization) -> ValidationCheckResult:
        ids = {
            "QualityResult": getattr(quality, "job_id", None),
            "AlignmentResult": getattr(alignment, "job_id", None),
            "ASRResult": getattr(asr_result, "job_id", None),
            "DiarizationResult": getattr(diarization, "job_id", None),
        }
        present = {k: v for k, v in ids.items() if v is not None}
        if present and len(set(present.values())) != 1:
            return ValidationCheckResult(
                check_code="VC02", outcome="error",
                message=f"job_id mismatch: {present}",
            )
        return ValidationCheckResult(check_code="VC02", outcome="passed", message="job_id_ok")

    # VC03: duración coherente y no negativa
    @staticmethod
    def _vc03(quality, alignment, vad) -> ValidationCheckResult:
        durations = [
            int(alignment.total_duration_ms),
            int(quality.total_duration_ms),
        ]
        if int(vad.duration_ms) > 0:
            durations.append(int(vad.duration_ms))
        if any(d < 0 for d in durations):
            return ValidationCheckResult(
                check_code="VC03", outcome="error", message="negative_duration"
            )
        if len(set(durations)) > 1:
            return ValidationCheckResult(
                check_code="VC03", outcome="error",
                message=f"duration mismatch: {durations}",
            )
        return ValidationCheckResult(check_code="VC03", outcome="passed", message="duration_ok")

    # VC04: segment counts coherentes
    @staticmethod
    def _vc04(alignment) -> ValidationCheckResult:
        ns = int(alignment.num_segments)
        na = int(alignment.num_segments_aligned)
        nu = int(alignment.num_unassigned)
        if ns < 0 or na < 0 or nu < 0:
            return ValidationCheckResult(
                check_code="VC04", outcome="error", message="negative_segment_count"
            )
        if na + nu > ns:
            return ValidationCheckResult(
                check_code="VC04", outcome="error",
                message=f"aligned({na})+unassigned({nu}) > segments({ns})",
            )
        return ValidationCheckResult(check_code="VC04", outcome="passed", message="segment_counts_ok")

    # VC05: cada DialogueSegment válido
    @staticmethod
    def _vc05(alignment) -> ValidationCheckResult:
        total = int(alignment.total_duration_ms)
        for seg in alignment.dialogue_segments:
            if int(seg.start_ms) < 0:
                return ValidationCheckResult(check_code="VC05", outcome="error", message="start_ms<0")
            if int(seg.end_ms) <= int(seg.start_ms):
                return ValidationCheckResult(check_code="VC05", outcome="error", message="end<=start")
            if int(seg.duration_ms) != int(seg.end_ms) - int(seg.start_ms):
                return ValidationCheckResult(check_code="VC05", outcome="error", message="duration_mismatch")
            if int(seg.end_ms) > total:
                return ValidationCheckResult(check_code="VC05", outcome="error", message="end>total")
            if int(seg.segment_index) < 0:
                return ValidationCheckResult(check_code="VC05", outcome="error", message="invalid_index")
        return ValidationCheckResult(check_code="VC05", outcome="passed", message="dialogue_ok")

    # VC06: speakers existentes en DiarizationResult
    @staticmethod
    def _vc06(alignment, diarization) -> ValidationCheckResult:
        valid = {str(t.speaker_label) for t in diarization.speaker_turns}
        for seg in alignment.dialogue_segments:
            if str(seg.speaker_label) not in valid:
                return ValidationCheckResult(
                    check_code="VC06", outcome="error",
                    message=f"unknown_speaker={seg.speaker_label}",
                )
        return ValidationCheckResult(check_code="VC06", outcome="passed", message="speakers_ok")

    # VC07: ASR segments coherentes
    @staticmethod
    def _vc07(alignment, asr_result) -> ValidationCheckResult:
        if int(asr_result.num_segments) != len(asr_result.segments):
            return ValidationCheckResult(
                check_code="VC07", outcome="error",
                message=f"asr num_segments={asr_result.num_segments} != len={len(asr_result.segments)}",
            )
        return ValidationCheckResult(check_code="VC07", outcome="passed", message="transcription_ok")

    # VC08: idioma coherente
    @staticmethod
    def _vc08(alignment, asr_result) -> ValidationCheckResult:
        if str(alignment.transcript_language_code) != str(asr_result.transcript_language_code):
            return ValidationCheckResult(
                check_code="VC08", outcome="error",
                message=f"language mismatch: alignment={alignment.transcript_language_code} asr={asr_result.transcript_language_code}",
            )
        return ValidationCheckResult(check_code="VC08", outcome="passed", message="language_ok")

    # VC09: score en [0,1] y level coherente
    @staticmethod
    def _vc09(quality, thresholds) -> ValidationCheckResult:
        score = float(quality.overall_quality_score)
        if score < 0.0 or score > 1.0:
            return ValidationCheckResult(check_code="VC09", outcome="error", message="score_out_of_range")
        if score >= float(thresholds.pass_threshold):
            expected = QualityLevel.PASSED
        elif score >= float(thresholds.warn_threshold):
            expected = QualityLevel.WARNING
        else:
            expected = QualityLevel.FAILED
        if quality.quality_level != expected:
            return ValidationCheckResult(
                check_code="VC09", outcome="error",
                message=f"level mismatch: {quality.quality_level} != {expected}",
            )
        return ValidationCheckResult(check_code="VC09", outcome="passed", message="quality_ok")

    # VC10: contadores de rule_results coherentes (derivados)
    @staticmethod
    def _vc10(quality) -> ValidationCheckResult:
        results = quality.rule_results
        if quality.num_rules_evaluated != 11:
            return ValidationCheckResult(check_code="VC10", outcome="error", message="num_rules_evaluated!=11")
        if len(results) != 11:
            return ValidationCheckResult(check_code="VC10", outcome="error", message="rule_results!=11")
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if r.severity == "ERROR" and not r.passed)
        warning = sum(1 for r in results if r.severity == "WARNING" and not r.passed)
        not_applicable = sum(1 for r in results if not r.applicable)
        if quality.num_rules_passed != passed or quality.num_rules_failed != failed or quality.num_rules_warning != warning:
            return ValidationCheckResult(
                check_code="VC10", outcome="error",
                message=f"counters mismatch: passed={passed} failed={failed} warning={warning}",
            )
        return ValidationCheckResult(
            check_code="VC10", outcome="passed",
            message=f"rule_counts_ok not_applicable={not_applicable}",
        )

    # VC11: quality_reference coherente con cadena T09
    @staticmethod
    def _vc11(quality, alignment) -> ValidationCheckResult:
        if str(quality.source_preprocessed_sha256).lower() != str(alignment.source_preprocessed_sha256).lower():
            return ValidationCheckResult(
                check_code="VC11", outcome="error", message="quality_reference mismatch",
            )
        return ValidationCheckResult(check_code="VC11", outcome="passed", message="quality_reference_ok")


__all__ = ["RunValidationUseCase"]
