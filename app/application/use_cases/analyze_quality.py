"""UseCase RunQualityAnalysisUseCase — Capa APPLICATION (T10 Step 10 Quality).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa torch/pyannote/numpy/speechbrain (clean architecture).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import QualityFailedError, ValidationFailedError
from app.core.logging import bind_context, get_logger
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.quality import (
    QUALITY_STRATEGY_RULES,
    QUALITY_STRATEGY_SILENT_FALLBACK,
    MODEL_LABEL_QUALITY_V1_LITERAL,
    QualityResult,
)
from app.domain.entities.vad import VadResult
from app.domain.value_objects.quality import (
    QualityRuleResult,
    QualityThresholds,
)


log = get_logger("app.application.use_cases.analyze_quality")


@dataclass(frozen=True, slots=True)
class RunQualityAnalysisUseCase:
    """Orquesta Step 10 Quality Analysis. Inmutable.

    Input: (AlignmentResult, VadResult, ASRResult, DiarizationResult) — frozen, NO mutados.
    Output: QualityResult frozen con chain of custody 7-SHA.
    """

    strict: bool = True

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        *,
        thresholds: QualityThresholds | None = None,
        job_id: str | None = None,
        allow_silent_fallback: bool = True,
    ) -> QualityResult:
        # 1) None inputs
        if alignment is None:
            raise ValidationFailedError("RunQualityAnalysisUseCase: alignment es None")
        if vad is None:
            raise ValidationFailedError("RunQualityAnalysisUseCase: vad es None")
        if asr_result is None:
            raise ValidationFailedError("RunQualityAnalysisUseCase: asr_result es None")
        if diarization is None:
            raise ValidationFailedError("RunQualityAnalysisUseCase: diarization es None")

        bind_context(job_id=job_id, phase="QUALITY", model=MODEL_LABEL_QUALITY_V1_LITERAL)
        t0 = time.perf_counter()

        # 2) Thresholds defaults
        effective_thresholds: QualityThresholds = thresholds or QualityThresholds()
        if not isinstance(effective_thresholds, QualityThresholds):
            raise ValidationFailedError(
                "RunQualityAnalysisUseCase: thresholds no es QualityThresholds"
            )

        # 3) Chain of Custody T03→T09
        self._validate_chain(alignment, vad, asr_result, diarization)

        # 4) Silent fallback: sin voz
        if int(vad.num_intervals) == 0:
            if not allow_silent_fallback:
                raise ValidationFailedError(
                    "RunQualityAnalysisUseCase: sin voz y allow_silent_fallback=False"
                )
            return self._build_silent_fallback(alignment, job_id=job_id)

        # 5) Evaluar las 11 reglas en orden determinista
        rule_results = self._evaluate_rules(
            alignment=alignment,
            vad=vad,
            asr_result=asr_result,
            diarization=diarization,
            thresholds=effective_thresholds,
        )

        # 6) Cálculo de score
        score = self._compute_score(rule_results)
        from app.core.constants import QualityLevel

        if score >= 0.80:
            level = QualityLevel.PASSED
        elif score >= 0.30:
            level = QualityLevel.WARNING
        else:
            level = QualityLevel.FAILED

        passed = sum(1 for r in rule_results if r.passed)
        failed = sum(1 for r in rule_results if r.severity == "ERROR" and not r.passed)
        warning = sum(1 for r in rule_results if r.severity == "WARNING" and not r.passed)

        result = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=str(alignment.source_preprocessed_sha256),
            vad_reference_sha256=str(alignment.vad_reference_sha256),
            lid_reference_sha256=str(alignment.lid_reference_sha256),
            asr_reference_sha256=str(alignment.asr_reference_sha256),
            ts_reference_sha256=str(alignment.ts_reference_sha256),
            diarization_reference_sha256=str(alignment.diarization_reference_sha256),
            alignment_reference_sha256=str(alignment.source_preprocessed_sha256),
            quality_level=level,
            overall_quality_score=score,
            num_rules_evaluated=11,
            num_rules_passed=passed,
            num_rules_warning=warning,
            num_rules_failed=failed,
            rule_results=tuple(rule_results),
            average_alignment_confidence=self._avg_alignment_conf(alignment),
            average_ts_confidence=self._avg_ts_conf(alignment),
            average_asr_confidence=float(asr_result.confidence),
            num_segments=int(alignment.num_segments),
            num_segments_aligned=int(alignment.num_segments_aligned),
            num_segments_unassigned=int(alignment.num_unassigned),
            total_duration_ms=int(alignment.total_duration_ms),
            transcript_language_code=str(alignment.transcript_language_code),
            job_id=job_id,
            analysis_metadata={},
        )

        # 7) Strict mode
        if self.strict and level == QualityLevel.FAILED:
            raise QualityFailedError(
                f"Quality Analysis FAILED: score={score:.4f}, "
                f"failed={failed}, warning={warning}."
            )

        t1 = time.perf_counter()
        log.info(
            "analyze_quality_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "score": round(score, 4),
                "level": str(level),
                "passed": passed,
                "warning": warning,
                "failed": failed,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Reglas QR01..QR11
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_rules(
        *,
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        thresholds: QualityThresholds,
    ) -> list[QualityRuleResult]:
        segs = list(alignment.dialogue_segments)
        results: list[QualityRuleResult] = []

        results.append(RunQualityAnalysisUseCase._qr01(segs, alignment.total_duration_ms))
        results.append(RunQualityAnalysisUseCase._qr02(segs))
        results.append(RunQualityAnalysisUseCase._qr03(segs, alignment.total_duration_ms))
        results.append(RunQualityAnalysisUseCase._qr04(segs, thresholds.max_overlap_ms))
        results.append(RunQualityAnalysisUseCase._qr05(alignment))
        results.append(RunQualityAnalysisUseCase._qr06(diarization, alignment))
        results.append(RunQualityAnalysisUseCase._qr07(asr_result, thresholds))
        results.append(RunQualityAnalysisUseCase._qr08(asr_result, thresholds))
        results.append(RunQualityAnalysisUseCase._qr09(segs, thresholds.max_segment_sec))
        results.append(RunQualityAnalysisUseCase._qr10(vad, thresholds.max_silence_sec))
        results.append(RunQualityAnalysisUseCase._qr11(segs, vad))

        return results

    # QR01: start >= end en cualquier segmento (ERROR, score 0.0)
    @staticmethod
    def _qr01(segs, total_ms) -> QualityRuleResult:
        occ = 0
        for s in segs:
            if int(s.start_ms) >= int(s.end_ms):
                occ += 1
        return QualityRuleResult(
            rule_code="QR01",
            severity="ERROR",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=0.0 if occ == 0 else 1.0,
            reason=f"QR01 start>=end occurrences={occ}",
        )

    # QR02: texto vacío/whitespace (ERROR, score 0.0)
    @staticmethod
    def _qr02(segs) -> QualityRuleResult:
        occ = 0
        for s in segs:
            if not str(s.text).strip():
                occ += 1
        return QualityRuleResult(
            rule_code="QR02",
            severity="ERROR",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=0.0 if occ == 0 else 1.0,
            reason=f"QR02 empty_text occurrences={occ}",
        )

    # QR03: timestamps imposibles (negativos o > duración) (ERROR, score 0.0)
    @staticmethod
    def _qr03(segs, total_ms) -> QualityRuleResult:
        occ = 0
        for s in segs:
            if int(s.start_ms) < 0 or int(s.end_ms) > int(total_ms):
                occ += 1
        return QualityRuleResult(
            rule_code="QR03",
            severity="ERROR",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=0.0 if occ == 0 else 1.0,
            reason=f"QR03 impossible_timestamps occurrences={occ}",
        )

    # QR04: overlap > max_overlap_ms (ERROR, -0.05/ocurrencia) — defensiva
    @staticmethod
    def _qr04(segs, max_overlap_ms: int) -> QualityRuleResult:
        occ = 0
        for i in range(len(segs) - 1):
            a = segs[i]
            b = segs[i + 1]
            overlap = min(int(a.end_ms), int(b.end_ms)) - max(int(a.start_ms), int(b.start_ms))
            if overlap > int(max_overlap_ms):
                occ += 1
        penalty = 0.05 * occ
        return QualityRuleResult(
            rule_code="QR04",
            severity="ERROR",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=penalty,
            reason=f"QR04 overlap> {max_overlap_ms}ms occurrences={occ}",
        )

    # QR05: diálogo sin speaker asignado (WARNING, -0.03/ocurrencia)
    @staticmethod
    def _qr05(alignment: AlignmentResult) -> QualityRuleResult:
        occ = int(alignment.num_unassigned)
        penalty = 0.03 * occ
        return QualityRuleResult(
            rule_code="QR05",
            severity="WARNING",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=penalty,
            reason=f"QR05 unassigned_segments occurrences={occ}",
        )

    # QR06: speaker detectado pero 0 segmentos alineados (WARNING, -0.02)
    @staticmethod
    def _qr06(diarization: DiarizationResult, alignment: AlignmentResult) -> QualityRuleResult:
        num_speakers = int(diarization.num_speakers)
        occ = 1 if (num_speakers > 0 and int(alignment.num_segments_aligned) == 0) else 0
        penalty = 0.02 * occ
        return QualityRuleResult(
            rule_code="QR06",
            severity="WARNING",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=penalty,
            reason=f"QR06 speakers={num_speakers} aligned={alignment.num_segments_aligned} occurrences={occ}",
        )

    # QR07: ASR segment confidence < 0.4 (WARNING, -0.04/segmento)
    # No hay confianza normalizada por segmento → not applicable.
    @staticmethod
    def _qr07(asr_result: ASRResult, thresholds: QualityThresholds) -> QualityRuleResult:
        return QualityRuleResult(
            rule_code="QR07",
            severity="WARNING",
            passed=True,
            applicable=False,
            occurrences=0,
            penalty=0.0,
            reason="segment_level_asr_confidence_unavailable",
        )

    # QR08: palabra confidence < 0.30 (WARNING, -0.005/palabra)
    # T07 WORD_TS_APLAZADO → not applicable.
    @staticmethod
    def _qr08(asr_result: ASRResult, thresholds: QualityThresholds) -> QualityRuleResult:
        return QualityRuleResult(
            rule_code="QR08",
            severity="WARNING",
            passed=True,
            applicable=False,
            occurrences=0,
            penalty=0.0,
            reason="word_level_confidence_unavailable",
        )

    # QR09: segmentos > max_segment_sec (WARNING, -0.02/caso)
    @staticmethod
    def _qr09(segs, max_segment_sec: float) -> QualityRuleResult:
        occ = 0
        for s in segs:
            if float(s.duration_ms) / 1000.0 > float(max_segment_sec):
                occ += 1
        penalty = 0.02 * occ
        return QualityRuleResult(
            rule_code="QR09",
            severity="WARNING",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=penalty,
            reason=f"QR09 long_segments occurrences={occ}",
        )

    # QR10: silencios inter-speech > max_silence_sec (WARNING, -0.015/caso)
    @staticmethod
    def _qr10(vad: VadResult, max_silence_sec: float) -> QualityRuleResult:
        occ = 0
        for sil in vad.silence_segments:
            if float(sil.duration_ms) / 1000.0 > float(max_silence_sec):
                occ += 1
        penalty = 0.015 * occ
        return QualityRuleResult(
            rule_code="QR10",
            severity="WARNING",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=penalty,
            reason=f"QR10 long_silences occurrences={occ}",
        )

    # QR11: ASR detecta voz en segmento de silencio (WARNING, -0.03/caso)
    @staticmethod
    def _qr11(segs, vad: VadResult) -> QualityRuleResult:
        voice = [(int(v.start_ms), int(v.end_ms)) for v in vad.voice_intervals]
        occ = 0
        for s in segs:
            # Un segmento alineado está "en silencio" si no intersecta ningún voice_interval
            overlaps_voice = False
            for vs, ve in voice:
                if int(s.end_ms) > vs and int(s.start_ms) < ve:
                    overlaps_voice = True
                    break
            if not overlaps_voice:
                occ += 1
        penalty = 0.03 * occ
        return QualityRuleResult(
            rule_code="QR11",
            severity="WARNING",
            passed=occ == 0,
            applicable=True,
            occurrences=occ,
            penalty=penalty,
            reason=f"QR11 asr_in_silence occurrences={occ}",
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(rule_results: list[QualityRuleResult]) -> float:
        total_penalty = 0.0
        for r in rule_results:
            if not r.applicable:
                continue
            if r.severity == "ERROR" and not r.passed:
                # Las reglas ERROR con violación ya imponen score 0.0 en QUALITY_RULES.md
                total_penalty += 1.0
            else:
                total_penalty += float(r.penalty)
        score = 1.0 - total_penalty
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Promedios
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_alignment_conf(alignment: AlignmentResult) -> float:
        segs = alignment.dialogue_segments
        if not segs:
            return 0.0
        return round(sum(float(s.alignment_confidence) for s in segs) / len(segs), 6)

    @staticmethod
    def _avg_ts_conf(alignment: AlignmentResult) -> float:
        segs = alignment.dialogue_segments
        if not segs:
            return 0.0
        return round(sum(float(s.ts_confidence) for s in segs) / len(segs), 6)

    # ------------------------------------------------------------------
    # Chain + silent fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_chain(
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
    ) -> None:
        source = str(alignment.source_preprocessed_sha256).lower()
        if str(vad.source_preprocessed_sha256).lower() != source:
            raise ValidationFailedError("Chain of Custody rota T04→T09 (VAD).")
        if str(asr_result.source_preprocessed_sha256).lower() != source:
            raise ValidationFailedError("Chain of Custody rota T06→T09 (ASR).")
        if str(diarization.source_preprocessed_sha256).lower() != source:
            raise ValidationFailedError("Chain of Custody rota T08→T09 (Diarization).")

    @staticmethod
    def _build_silent_fallback(alignment: AlignmentResult, *, job_id: str | None) -> QualityResult:
        from app.core.constants import QualityLevel

        return QualityResult(
            strategy=QUALITY_STRATEGY_SILENT_FALLBACK,
            source_preprocessed_sha256=str(alignment.source_preprocessed_sha256),
            vad_reference_sha256=str(alignment.vad_reference_sha256),
            lid_reference_sha256=str(alignment.lid_reference_sha256),
            asr_reference_sha256=str(alignment.asr_reference_sha256),
            ts_reference_sha256=str(alignment.ts_reference_sha256),
            diarization_reference_sha256=str(alignment.diarization_reference_sha256),
            alignment_reference_sha256=str(alignment.source_preprocessed_sha256),
            quality_level=QualityLevel.PASSED,
            overall_quality_score=1.0,
            num_rules_evaluated=11,
            num_rules_passed=11,
            num_rules_warning=0,
            num_rules_failed=0,
            rule_results=tuple(
                QualityRuleResult(
                    rule_code=code,
                    severity="WARNING",
                    passed=True,
                    applicable=False,
                    occurrences=0,
                    penalty=0.0,
                    reason="silent_fallback_no_voice",
                )
                for code in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
            ),
            average_alignment_confidence=0.0,
            average_ts_confidence=0.0,
            average_asr_confidence=0.0,
            num_segments=0,
            num_segments_aligned=0,
            num_segments_unassigned=0,
            total_duration_ms=int(alignment.total_duration_ms),
            transcript_language_code=str(alignment.transcript_language_code),
            job_id=job_id,
            analysis_metadata={"fallback_reason": "vad_no_intervals"},
        )


__all__ = ["RunQualityAnalysisUseCase"]
