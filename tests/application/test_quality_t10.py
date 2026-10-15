"""Tests unitarios T10 Application RunQualityAnalysisUseCase.

No requiere IA. Entidades sintéticas T04/T06/T08/T09. Cubre QR01..QR11,
score, niveles, strict mode, silent fallback y AST.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.constants import QualityLevel
from app.core.exceptions import QualityFailedError, ValidationFailedError
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.quality import (
    QUALITY_STRATEGY_RULES,
    QUALITY_STRATEGY_SILENT_FALLBACK,
    QualityResult,
)
from app.domain.entities.vad import VadResult
from app.domain.value_objects.alignment import DialogueSegment
from app.domain.value_objects.diarization import SpeakerTurn
from app.domain.value_objects.quality import (
    QualityRuleResult,
    QualityThresholds,
)
from app.domain.value_objects.vad import SilenceSegment, VadThresholds, VoiceInterval
from app.application.use_cases.analyze_quality import RunQualityAnalysisUseCase


_SHA = "a" * 64


def _seg(idx=0, start=0, end=1000, text="hola", speaker="SPEAKER_00", align_conf=1.0, ts_conf=1.0) -> DialogueSegment:
    return DialogueSegment(
        segment_index=idx, start_ms=start, end_ms=end, duration_ms=end - start,
        text=text, speaker_label=speaker, alignment_confidence=align_conf, ts_confidence=ts_conf,
    )


def _seg_raw(idx=0, start=0, end=1000, text="hola", speaker="SPEAKER_00", align_conf=1.0, ts_conf=1.0) -> DialogueSegment:
    """Construye DialogueSegment sin validación (para reglas defensivas QR01/QR03)."""
    return DialogueSegment.model_construct(
        segment_index=idx, start_ms=start, end_ms=end, duration_ms=end - start,
        text=text, speaker_label=speaker, alignment_confidence=align_conf, ts_confidence=ts_conf,
    )


def _alignment(segments=None, num_segments=0, num_aligned=0, num_unassigned=0, total_ms=10000):
    segs = tuple(segments or [])
    return AlignmentResult(
        strategy="weighted_overlap_majority_vote",
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        total_duration_ms=total_ms,
        transcript_language_code="es",
        dialogue_segments=segs,
        num_segments=num_segments if num_segments else len(segs),
        num_segments_aligned=num_aligned if num_aligned is not None else len(segs),
        num_unassigned=num_unassigned,
    )


def _alignment_raw(segments=None, num_segments=0, num_aligned=0, num_unassigned=0, total_ms=10000):
    """Construye AlignmentResult sin validación (para reglas defensivas QR01/QR03)."""
    segs = tuple(segments or [])
    return AlignmentResult.model_construct(
        strategy="weighted_overlap_majority_vote",
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        total_duration_ms=total_ms,
        transcript_language_code="es",
        dialogue_segments=segs,
        num_segments=num_segments if num_segments else len(segs),
        num_segments_aligned=num_aligned if num_aligned is not None else len(segs),
        num_unassigned=num_unassigned,
    )


def _vad(intervals=None, silences=None, num_intervals=1):
    vis = tuple(VoiceInterval(start_ms=s, end_ms=e, max_confidence=0.9) for s, e in (intervals or [(0, 10000)]))
    sil = tuple(
        SilenceSegment(start_ms=s, end_ms=e, speech_index_before=-1, speech_index_after=-1)
        for s, e in (silences or [])
    )
    speech_ms = sum(v.duration_ms for v in vis)
    return VadResult.model_construct(
        source_preprocessed_sha256=_SHA,
        voice_intervals=vis,
        silence_segments=sil,
        speech_ratio=1.0 if vis else 0.0,
        total_speech_ms=speech_ms,
        total_silence_ms=sum(s.duration_ms for s in sil),
        num_intervals=len(vis) if num_intervals is None else num_intervals,
        num_silences=len(sil),
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",
        sample_rate=16000,
        channels=1,
        duration_ms=10000,
    )


def _asr(confidence=0.9):
    segs = tuple(ASRSegment.model_construct(segment_index=i, text="t", start_ms=None, end_ms=None, avg_logprob=None) for i in range(1))
    return ASRResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        total_duration_ms=10000,
        transcript_text="t",
        transcript_language_code="es",
        confidence=confidence,
        segments=segs,
        num_segments=1,
        strategy="full_audio",
        model_label="faster-whisper:small",
    )


def _diarization(num_speakers=1, turns=None):
    t = tuple(turns or [SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=0.9)])
    return DiarizationResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        total_duration_ms=10000,
        speaker_turns=t,
        num_speakers=num_speakers,
        num_turns=len(t),
    )


def _run(alignment, vad, asr, diar, strict=False, thresholds=None):
    return RunQualityAnalysisUseCase(strict=strict).run(alignment, vad, asr, diar, thresholds=thresholds)


# =============================================================================
# Reglas individuales
# =============================================================================


class TestRules:
    def test_qr01_start_ge_end_fails(self):
        a = _alignment_raw([_seg_raw(start=1000, end=1000)], num_segments=1, num_aligned=1)
        with pytest.raises(QualityFailedError):
            _run(a, _vad(), _asr(), _diarization(), strict=True)

    def test_qr02_empty_text_fails(self):
        a = _alignment([_seg(text="  ")], num_segments=1, num_aligned=1)
        with pytest.raises(QualityFailedError):
            _run(a, _vad(), _asr(), _diarization(), strict=True)

    def test_qr03_impossible_timestamp_fails(self):
        a = _alignment_raw([_seg_raw(start=-5, end=100)], num_segments=1, num_aligned=1)
        with pytest.raises(QualityFailedError):
            _run(a, _vad(), _asr(), _diarization(), strict=True)

    def test_qr04_overlap_defensive_zero(self):
        # con entradas T09 no solapadas, QR04 debe dar 0 occurrences
        a = _alignment([_seg(0, 0, 1000), _seg(1, 1000, 2000)], num_segments=2, num_aligned=2)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        qr04 = [x for x in r.rule_results if x.rule_code == "QR04"][0]
        assert qr04.occurrences == 0
        assert qr04.passed is True

    def test_qr05_unassigned_warning(self):
        a = _alignment([_seg(0, 0, 1000)], num_segments=2, num_aligned=1, num_unassigned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        qr05 = [x for x in r.rule_results if x.rule_code == "QR05"][0]
        assert qr05.occurrences == 1
        assert qr05.passed is False

    def test_qr06_speaker_no_segments(self):
        a = _alignment([], num_segments=0, num_aligned=0, num_unassigned=0)
        r = _run(a, _vad(), _asr(), _diarization(num_speakers=1), strict=False)
        qr06 = [x for x in r.rule_results if x.rule_code == "QR06"][0]
        assert qr06.occurrences == 1
        assert qr06.passed is False

    def test_qr07_not_applicable(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        qr07 = [x for x in r.rule_results if x.rule_code == "QR07"][0]
        assert qr07.applicable is False
        assert qr07.reason == "segment_level_asr_confidence_unavailable"

    def test_qr08_not_applicable(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        qr08 = [x for x in r.rule_results if x.rule_code == "QR08"][0]
        assert qr08.applicable is False
        assert qr08.reason == "word_level_confidence_unavailable"

    def test_qr09_long_segment(self):
        a = _alignment([_seg(0, 0, 70000)], num_segments=1, num_aligned=1, total_ms=70000)
        r = _run(a, _vad(intervals=[(0, 70000)]), _asr(), _diarization(), strict=False)
        qr09 = [x for x in r.rule_results if x.rule_code == "QR09"][0]
        assert qr09.occurrences == 1

    def test_qr10_long_silence(self):
        a = _alignment([_seg(0, 0, 1000)], num_segments=1, num_aligned=1)
        vad = _vad(intervals=[(0, 1000)], silences=[(1000, 20000)])
        r = _run(a, vad, _asr(), _diarization(), strict=False)
        qr10 = [x for x in r.rule_results if x.rule_code == "QR10"][0]
        assert qr10.occurrences == 1

    def test_qr11_asr_in_silence(self):
        # segmento alineado fuera de voice_intervals → voz detectada en silencio
        a = _alignment([_seg(0, 5000, 6000)], num_segments=1, num_aligned=1)
        vad = _vad(intervals=[(0, 1000)])
        r = _run(a, vad, _asr(), _diarization(), strict=False)
        qr11 = [x for x in r.rule_results if x.rule_code == "QR11"][0]
        assert qr11.occurrences == 1


# =============================================================================
# Score / niveles / strict
# =============================================================================


class TestScoreLevel:
    def test_score_passed(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        assert r.quality_level == QualityLevel.PASSED
        assert r.overall_quality_score == 1.0

    def test_score_warning(self):
        # 10 unassigned → QR05: 10*0.03 = 0.30 → score 0.70 (warning)
        a2 = _alignment([], num_segments=10, num_aligned=0, num_unassigned=10)
        r = _run(a2, _vad(), _asr(), _diarization(num_speakers=1), strict=False)
        # QR05: 10*0.03 = 0.30; QR06: speakers=1 aligned=0 → -0.02 → total 0.32 → score 0.68 (warning)
        assert r.quality_level == QualityLevel.WARNING
        assert r.overall_quality_score < 0.80

    def test_score_failed(self):
        # QR01 ERROR con violación → score 0.0 → failed
        a = _alignment_raw([_seg_raw(start=1000, end=1000)], num_segments=1, num_aligned=1)
        with pytest.raises(QualityFailedError):
            _run(a, _vad(), _asr(), _diarization(), strict=True)

    def test_strict_mode_raises_quality_failed(self):
        a = _alignment_raw([_seg_raw(text="")], num_segments=1, num_aligned=1)
        with pytest.raises(QualityFailedError):
            _run(a, _vad(), _asr(), _diarization(), strict=True)

    def test_non_strict_returns_failed_result(self):
        a = _alignment_raw([_seg_raw(text="")], num_segments=1, num_aligned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        assert r.quality_level == QualityLevel.FAILED
        assert r.overall_quality_score == 0.0


class TestSilent:
    def test_silent_fallback(self):
        a = _alignment([], num_segments=0, num_aligned=0, num_unassigned=0)
        vad = _vad(intervals=[], num_intervals=0)
        r = _run(a, vad, _asr(), _diarization(num_speakers=0), strict=False)
        assert r.strategy == QUALITY_STRATEGY_SILENT_FALLBACK
        assert r.num_segments == 0

    def test_silent_allow_false_raises(self):
        a = _alignment([], num_segments=0, num_aligned=0, num_unassigned=0)
        vad = _vad(intervals=[], num_intervals=0)
        with pytest.raises(ValidationFailedError):
            RunQualityAnalysisUseCase(strict=False).run(a, vad, _asr(), _diarization(num_speakers=0), allow_silent_fallback=False)


class TestChainSHA:
    def test_chain_sha_ok(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        assert r.source_preprocessed_sha256 == _SHA
        assert r.alignment_reference_sha256 == _SHA

    def test_chain_sha_mismatch_raises(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        bad_asr = _asr()
        bad_asr = ASRResult.model_construct(
            source_preprocessed_sha256="b" * 64,
            vad_reference_sha256="b" * 64,
            lid_reference_sha256="b" * 64,
            sample_rate=16000, channels=1, bit_depth=16, total_duration_ms=10000,
            transcript_text="t", transcript_language_code="es", confidence=0.9,
            segments=(), num_segments=0, strategy="full_audio", model_label="x",
        )
        with pytest.raises(ValidationFailedError):
            _run(a, _vad(), bad_asr, _diarization(), strict=False)


class TestNoMutation:
    def test_inputs_not_mutated(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        vad = _vad()
        asr = _asr()
        diar = _diarization()
        a_d = a.model_dump(); vad_d = vad.model_dump(); asr_d = asr.model_dump(); diar_d = diar.model_dump()
        _run(a, vad, asr, diar, strict=False)
        assert a.model_dump() == a_d
        assert vad.model_dump() == vad_d
        assert asr.model_dump() == asr_d
        assert diar.model_dump() == diar_d


class TestDeterminism:
    def test_same_input_same_output(self):
        a = _alignment([_seg(0, 0, 1000), _seg(1, 1000, 2000)], num_segments=2, num_aligned=2)
        r1 = _run(a, _vad(), _asr(), _diarization(), strict=False)
        r2 = _run(a, _vad(), _asr(), _diarization(), strict=False)
        assert r1.model_dump() == r2.model_dump()


class TestInputValidation:
    def test_none_alignment_raises(self):
        with pytest.raises(ValidationFailedError):
            RunQualityAnalysisUseCase().run(None, _vad(), _asr(), _diarization())  # type: ignore[arg-type]

    def test_none_vad_raises(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        with pytest.raises(ValidationFailedError):
            RunQualityAnalysisUseCase().run(a, None, _asr(), _diarization())  # type: ignore[arg-type]

    def test_thresholds_wrong_type(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        with pytest.raises(ValidationFailedError):
            _run(a, _vad(), _asr(), _diarization(), thresholds="x")  # type: ignore[arg-type]


class TestASTAudit:
    def test_no_ai_imports_in_domain_app(self):
        paths = [
            "app/domain/value_objects/quality.py",
            "app/domain/entities/quality.py",
            "app/domain/interfaces/quality_ports.py",
            "app/application/use_cases/analyze_quality.py",
        ]
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess"}
        for p in paths:
            tree = ast.parse(Path(p).read_text(encoding="utf-8"))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imports.add(a.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
            assert not (imports & forbidden), f"{p} importa prohibido: {imports & forbidden}"


# =============================================================================
# Cobertura adicional: límites, transiciones, acumulaciones
# =============================================================================


class TestScoreBoundaries:
    def test_score_exactly_080_passed(self):
        # score = 0.80 exacto → passed
        r = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA,
            asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA,
            diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level=QualityLevel.PASSED,
            overall_quality_score=0.80,
            num_rules_evaluated=11,
            num_rules_passed=11,
            rule_results=tuple(
                QualityRuleResult(rule_code=c, severity="WARNING", passed=True, applicable=True)
                for c in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
            ),
            total_duration_ms=0,
            transcript_language_code="es",
        )
        assert r.quality_level == QualityLevel.PASSED

    def test_score_079_warning(self):
        # score 0.79 → warning
        r = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA,
            asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA,
            diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level=QualityLevel.WARNING,
            overall_quality_score=0.79,
            num_rules_evaluated=11,
            num_rules_passed=11,
            rule_results=tuple(
                QualityRuleResult(rule_code=c, severity="WARNING", passed=True, applicable=True)
                for c in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
            ),
            total_duration_ms=0,
            transcript_language_code="es",
        )
        assert r.quality_level == QualityLevel.WARNING

    def test_score_030_warning(self):
        # score 0.30 exacto → warning (límite inferior)
        r = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA,
            asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA,
            diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level=QualityLevel.WARNING,
            overall_quality_score=0.30,
            num_rules_evaluated=11,
            num_rules_passed=11,
            rule_results=tuple(
                QualityRuleResult(rule_code=c, severity="WARNING", passed=True, applicable=True)
                for c in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
            ),
            total_duration_ms=0,
            transcript_language_code="es",
        )
        assert r.quality_level == QualityLevel.WARNING

    def test_score_029_failed(self):
        # score 0.29 → failed
        r = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA,
            asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA,
            diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level=QualityLevel.FAILED,
            overall_quality_score=0.29,
            num_rules_evaluated=11,
            num_rules_passed=11,
            rule_results=tuple(
                QualityRuleResult(rule_code=c, severity="WARNING", passed=True, applicable=True)
                for c in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
            ),
            total_duration_ms=0,
            transcript_language_code="es",
        )
        assert r.quality_level == QualityLevel.FAILED


class TestPenaltyAccumulation:
    def test_multiple_silences_penalty_accumulates(self):
        # 2 silencios >15s → QR10: 2 * 0.015 = 0.03
        a = _alignment([_seg(0, 0, 1000)], num_segments=1, num_aligned=1)
        vad = _vad(intervals=[(0, 1000), (17000, 18000)], silences=[(1000, 17000), (18000, 34000)])
        r = _run(a, vad, _asr(), _diarization(), strict=False)
        qr10 = [x for x in r.rule_results if x.rule_code == "QR10"][0]
        assert qr10.occurrences == 2
        assert abs(qr10.penalty - 0.03) < 1e-9

    def test_multiple_long_segments(self):
        a = _alignment(
            [_seg(0, 0, 70000), _seg(1, 70000, 140000)],
            num_segments=2, num_aligned=2, total_ms=140000,
        )
        vad = _vad(intervals=[(0, 140000)])
        r = _run(a, vad, _asr(), _diarization(), strict=False)
        qr09 = [x for x in r.rule_results if x.rule_code == "QR09"][0]
        assert qr09.occurrences == 2

    def test_score_clamp_to_zero(self):
        # un ERROR con violación → score 0.0 (clamp inferior)
        a = _alignment_raw([_seg_raw(text="")], num_segments=1, num_aligned=1)
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        assert r.overall_quality_score == 0.0
        assert r.quality_level == QualityLevel.FAILED

    def test_mixed_assigned_unassigned_segments(self):
        a = _alignment(
            [_seg(0, 0, 1000, text="a")],
            num_segments=2, num_aligned=1, num_unassigned=1,
        )
        r = _run(a, _vad(), _asr(), _diarization(), strict=False)
        qr05 = [x for x in r.rule_results if x.rule_code == "QR05"][0]
        assert qr05.occurrences == 1
        assert r.num_segments == 2
        assert r.num_segments_aligned == 1
        assert r.num_segments_unassigned == 1

    def test_partial_asr_in_silence(self):
        # segmento parcialmente en voz, parcialmente en silencio → sin violación QR11
        a = _alignment([_seg(0, 0, 500)], num_segments=1, num_aligned=1)
        vad = _vad(intervals=[(0, 1000)])
        r = _run(a, vad, _asr(), _diarization(), strict=False)
        qr11 = [x for x in r.rule_results if x.rule_code == "QR11"][0]
        assert qr11.occurrences == 0

    def test_determinism_with_multiple_violations(self):
        a = _alignment(
            [_seg(0, 0, 1000, text=""), _seg(1, 1000, 2000, text="b")],
            num_segments=2, num_aligned=2, num_unassigned=0,
        )
        r1 = _run(a, _vad(), _asr(), _diarization(), strict=False)
        r2 = _run(a, _vad(), _asr(), _diarization(), strict=False)
        assert r1.model_dump() == r2.model_dump()

