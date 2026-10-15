"""Tests unitarios T11 Application RunValidationUseCase.

No requiere IA. Entidades sintéticas T04/T06/T08/T09/T10. Cubre VC01..VC11,
Quality FAILED/strict/force-save y AST.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.constants import QualityLevel
from app.core.exceptions import ValidationFailedError
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.quality import (
    QUALITY_STRATEGY_RULES,
    QualityResult,
)
from app.domain.entities.vad import VadResult
from app.domain.value_objects.alignment import DialogueSegment
from app.domain.value_objects.diarization import SpeakerTurn
from app.domain.value_objects.quality import QualityRuleResult
from app.domain.value_objects.validation import ValidationThresholds
from app.domain.value_objects.vad import SilenceSegment, VadThresholds, VoiceInterval
from app.application.use_cases.validate_results import RunValidationUseCase


_SHA = "a" * 64


def _seg(idx=0, start=0, end=1000, text="hola", speaker="SPEAKER_00") -> DialogueSegment:
    return DialogueSegment(
        segment_index=idx, start_ms=start, end_ms=end, duration_ms=end - start,
        text=text, speaker_label=speaker, alignment_confidence=1.0, ts_confidence=1.0,
    )


def _alignment(segments=None, num_segments=0, num_aligned=0, num_unassigned=0, total_ms=10000, lang="es"):
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
        transcript_language_code=lang,
        dialogue_segments=segs,
        num_segments=num_segments if num_segments else len(segs),
        num_segments_aligned=num_aligned if num_aligned is not None else len(segs),
        num_unassigned=num_unassigned,
    )


def _alignment_raw(segments=None, num_segments=0, num_aligned=0, num_unassigned=0, total_ms=10000, lang="es"):
    """AlignmentResult sin validación (para checks defensivos)."""
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
        transcript_language_code=lang,
        dialogue_segments=segs,
        num_segments=num_segments if num_segments else len(segs),
        num_segments_aligned=num_aligned if num_aligned is not None else len(segs),
        num_unassigned=num_unassigned,
    )


def _vad(duration_ms=10000, intervals=None):
    vis = tuple(VoiceInterval(start_ms=s, end_ms=e, max_confidence=0.9) for s, e in (intervals or [(0, 10000)]))
    return VadResult.model_construct(
        source_preprocessed_sha256=_SHA,
        voice_intervals=vis,
        silence_segments=(),
        speech_ratio=1.0,
        total_speech_ms=sum(v.duration_ms for v in vis),
        total_silence_ms=0,
        num_intervals=len(vis),
        num_silences=0,
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",
        sample_rate=16000,
        channels=1,
        duration_ms=duration_ms,
    )


def _asr(lang="es", num_segments=1):
    segs = tuple(ASRSegment.model_construct(segment_index=i, text="t", start_ms=None, end_ms=None, avg_logprob=None) for i in range(num_segments))
    return ASRResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        sample_rate=16000, channels=1, bit_depth=16,
        total_duration_ms=10000,
        transcript_text="t",
        transcript_language_code=lang,
        confidence=0.9,
        segments=segs,
        num_segments=num_segments,
        strategy="full_audio",
        model_label="faster-whisper:small",
    )


def _diarization(speakers=("SPEAKER_00",)):
    turns = tuple(SpeakerTurn(speaker_label=s, start_ms=0, end_ms=1000, confidence=0.9) for s in speakers)
    return DiarizationResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        total_duration_ms=10000,
        speaker_turns=turns,
        num_speakers=len(speakers),
        num_turns=len(turns),
    )


def _all_rules_passed():
    return tuple(
        QualityRuleResult(rule_code=c, severity="WARNING", passed=True, applicable=True, occurrences=0, penalty=0.0, reason="")
        for c in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
    )


def _quality(level=QualityLevel.PASSED, score=1.0, rules=None):
    return QualityResult(
        strategy=QUALITY_STRATEGY_RULES,
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA,
        quality_level=level,
        overall_quality_score=score,
        num_rules_evaluated=11,
        num_rules_passed=11,
        num_rules_warning=0,
        num_rules_failed=0,
        rule_results=rules or _all_rules_passed(),
        total_duration_ms=10000,
        transcript_language_code="es",
    )


def _quality_raw(level=QualityLevel.PASSED, score=1.0, rules=None, total_ms=10000, num_passed=11):
    """QualityResult sin validación (para checks defensivos)."""
    return QualityResult.model_construct(
        strategy=QUALITY_STRATEGY_RULES,
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA,
        quality_level=level,
        overall_quality_score=score,
        num_rules_evaluated=11,
        num_rules_passed=num_passed,
        num_rules_warning=0,
        num_rules_failed=0,
        rule_results=rules or _all_rules_passed(),
        total_duration_ms=total_ms,
        transcript_language_code="es",
    )


def _run(quality, alignment, vad, asr, diar, force_save=False, thresholds=None, strict_quality=True):
    return RunValidationUseCase(strict_quality=strict_quality).run(
        quality, alignment, vad, asr, diar, force_save=force_save, thresholds=thresholds
    )


# =============================================================================
# Checks VC01..VC11 (PASS/FAIL)
# =============================================================================


class TestChecks:
    def test_all_pass(self):
        r = _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())
        assert r.validation_passed is True
        assert r.validation_status == "passed"
        assert r.num_errors == 0

    def test_vc01_sha_mismatch(self):
        bad_vad = VadResult.model_construct(
            source_preprocessed_sha256="b" * 64,
            voice_intervals=(), silence_segments=(), speech_ratio=0.0,
            total_speech_ms=0, total_silence_ms=0, num_intervals=0, num_silences=0,
            thresholds=VadThresholds(), model_label="silero-vad:v5.1",
            sample_rate=16000, channels=1, duration_ms=0,
        )
        with pytest.raises(ValidationFailedError):
            _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), bad_vad, _asr(), _diarization())

    def test_vc03_duration_mismatch(self):
        q = _quality_raw(total_ms=9999)
        with pytest.raises(ValidationFailedError):
            _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())

    def test_vc04_segment_count_overflow(self):
        a = _alignment_raw([_seg()], num_segments=1, num_aligned=2, num_unassigned=0)
        with pytest.raises(ValidationFailedError):
            _run(_quality(), a, _vad(), _asr(), _diarization())

    def test_vc05_invalid_dialogue(self):
        seg = DialogueSegment.model_construct(
            segment_index=0, start_ms=100, end_ms=50, duration_ms=-50,
            text="x", speaker_label="SPEAKER_00", alignment_confidence=1.0, ts_confidence=1.0,
        )
        a = AlignmentResult.model_construct(
            strategy="weighted_overlap_majority_vote",
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            total_duration_ms=10000, transcript_language_code="es",
            dialogue_segments=(seg,), num_segments=1, num_segments_aligned=1, num_unassigned=0,
        )
        with pytest.raises(ValidationFailedError):
            _run(_quality(), a, _vad(), _asr(), _diarization())

    def test_vc06_unknown_speaker(self):
        a = _alignment([_seg(speaker="SPEAKER_99")], num_segments=1, num_aligned=1)
        with pytest.raises(ValidationFailedError):
            _run(_quality(), a, _vad(), _asr(), _diarization(speakers=("SPEAKER_00",)))

    def test_vc07_asr_count_mismatch(self):
        asr = ASRResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA, lid_reference_sha256=_SHA,
            sample_rate=16000, channels=1, bit_depth=16, total_duration_ms=10000,
            transcript_text="t", transcript_language_code="es", confidence=0.9,
            segments=(), num_segments=5, strategy="full_audio", model_label="x",
        )
        with pytest.raises(ValidationFailedError):
            _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), asr, _diarization())

    def test_vc08_language_mismatch(self):
        with pytest.raises(ValidationFailedError):
            _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1, lang="en"), _vad(), _asr(lang="es"), _diarization())

    def test_vc09_level_mismatch(self):
        # score 1.0 pero level failed → inconsistente (usar raw para saltar validación upstream)
        q = _quality_raw(level=QualityLevel.FAILED, score=1.0)
        with pytest.raises(ValidationFailedError):
            _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization(), force_save=True)

    def test_vc10_rule_count_mismatch(self):
        rules = list(_all_rules_passed())
        rules[0] = QualityRuleResult(rule_code="QR01", severity="ERROR", passed=False, applicable=True, occurrences=1, penalty=1.0, reason="x")
        q = _quality_raw(level=QualityLevel.FAILED, score=0.0, rules=tuple(rules), num_passed=11)
        with pytest.raises(ValidationFailedError):
            _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization(), force_save=True)

    def test_vc11_quality_reference_mismatch(self):
        q = QualityResult.model_construct(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256="b" * 64, vad_reference_sha256="b" * 64,
            lid_reference_sha256="b" * 64, asr_reference_sha256="b" * 64,
            ts_reference_sha256="b" * 64, diarization_reference_sha256="b" * 64,
            alignment_reference_sha256="b" * 64,
            quality_level=QualityLevel.PASSED, overall_quality_score=1.0,
            num_rules_evaluated=11, num_rules_passed=11, num_rules_warning=0, num_rules_failed=0,
            rule_results=_all_rules_passed(),
            total_duration_ms=10000, transcript_language_code="es",
        )
        with pytest.raises(ValidationFailedError):
            _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())


class TestQualityFailed:
    def test_quality_failed_strict_raises(self):
        q = _quality(level=QualityLevel.FAILED, score=0.1)
        with pytest.raises(ValidationFailedError):
            _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization(), force_save=False)

    def test_quality_failed_force_save_returns_failed(self):
        q = _quality(level=QualityLevel.FAILED, score=0.1)
        r = _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization(), force_save=True)
        assert r.validation_status == "failed"
        assert r.validation_passed is True  # sin errores estructurales

    def test_structural_error_force_save_raises(self):
        # error estructural (VC03) NO es ignorado por force_save
        q = _quality()
        q = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level=QualityLevel.PASSED, overall_quality_score=1.0,
            num_rules_evaluated=11, num_rules_passed=11, num_rules_warning=0, num_rules_failed=0,
            rule_results=_all_rules_passed(), total_duration_ms=9999, transcript_language_code="es",
        )
        with pytest.raises(ValidationFailedError):
            _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization(), force_save=True)


class TestNoMutation:
    def test_inputs_not_mutated(self):
        q = _quality()
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        v = _vad(); asr = _asr(); d = _diarization()
        q_d = q.model_dump(); a_d = a.model_dump(); v_d = v.model_dump(); asr_d = asr.model_dump(); d_d = d.model_dump()
        _run(q, a, v, asr, d)
        assert q.model_dump() == q_d
        assert a.model_dump() == a_d
        assert v.model_dump() == v_d
        assert asr.model_dump() == asr_d
        assert d.model_dump() == d_d


class TestDeterminism:
    def test_same_input_same_output(self):
        q = _quality()
        a = _alignment([_seg(0, 0, 1000), _seg(1, 1000, 2000)], num_segments=2, num_aligned=2)
        r1 = _run(q, a, _vad(), _asr(), _diarization())
        r2 = _run(q, a, _vad(), _asr(), _diarization())
        assert r1.model_dump() == r2.model_dump()


class TestInputValidation:
    def test_none_quality_raises(self):
        with pytest.raises(ValidationFailedError):
            RunValidationUseCase().run(None, _alignment(), _vad(), _asr(), _diarization())  # type: ignore[arg-type]

    def test_thresholds_wrong_type(self):
        q = _quality()
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        with pytest.raises(ValidationFailedError):
            _run(q, a, _vad(), _asr(), _diarization(), thresholds="x")  # type: ignore[arg-type]


class TestSHAChain:
    def test_8_sha_chain_ok(self):
        r = _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())
        assert r.source_preprocessed_sha256 == _SHA
        assert r.quality_reference_sha256 == _SHA
        assert r.alignment_reference_sha256 == _SHA


class TestASTAudit:
    def test_no_ai_imports_in_domain_app(self):
        paths = [
            "app/domain/value_objects/validation.py",
            "app/domain/entities/validation.py",
            "app/domain/interfaces/validation_ports.py",
            "app/application/use_cases/validate_results.py",
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
# Cobertura adicional: casos límite y passthrough
# =============================================================================


class TestAdditional:
    def test_vc02_job_id_present_consistent(self):
        q = _quality()
        a = _alignment([_seg()], num_segments=1, num_aligned=1)
        r = _run(q, a, _vad(), _asr(), _diarization())
        # Todos los job_id son None → consistente (VC02 passed)
        assert r.validation_passed is True

    def test_vc04_exact_equality(self):
        a = _alignment([_seg()], num_segments=1, num_aligned=1, num_unassigned=0)
        r = _run(_quality(), a, _vad(), _asr(), _diarization())
        assert r.validation_passed is True

    def test_vc05_valid_multiple_segments(self):
        a = _alignment([_seg(0, 0, 1000), _seg(1, 1000, 2000)], num_segments=2, num_aligned=2)
        r = _run(_quality(), a, _vad(), _asr(), _diarization())
        assert r.validation_passed is True

    def test_vc06_multiple_speakers(self):
        a = _alignment([_seg(0, 0, 1000, speaker="SPEAKER_00"), _seg(1, 1000, 2000, speaker="SPEAKER_01")], num_segments=2, num_aligned=2)
        d = _diarization(speakers=("SPEAKER_00", "SPEAKER_01"))
        r = _run(_quality(), a, _vad(), _asr(), d)
        assert r.validation_passed is True

    def test_vc08_ok(self):
        r = _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1, lang="es"), _vad(), _asr(lang="es"), _diarization())
        assert r.validation_passed is True

    def test_vc09_score_boundaries(self):
        # score 0.80 exacto → passed (nivel correcto)
        q = _quality(level=QualityLevel.PASSED, score=0.80)
        r = _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())
        assert r.validation_passed is True

    def test_vc10_not_applicable_derived(self):
        # rule_results con 2 not_applicable, contadores correctos → VC10 passed
        rules = list(_all_rules_passed())
        rules[6] = QualityRuleResult(rule_code="QR07", severity="WARNING", passed=True, applicable=False, occurrences=0, penalty=0.0, reason="x")
        rules[7] = QualityRuleResult(rule_code="QR08", severity="WARNING", passed=True, applicable=False, occurrences=0, penalty=0.0, reason="x")
        q = QualityResult(
            strategy=QUALITY_STRATEGY_RULES,
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level=QualityLevel.PASSED, overall_quality_score=1.0,
            num_rules_evaluated=11, num_rules_passed=11, num_rules_warning=0, num_rules_failed=0,
            rule_results=tuple(rules), total_duration_ms=10000, transcript_language_code="es",
        )
        r = _run(q, _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())
        assert r.validation_passed is True

    def test_8_sha_passthrough_complete(self):
        r = _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())
        assert r.vad_reference_sha256 == _SHA
        assert r.lid_reference_sha256 == _SHA
        assert r.asr_reference_sha256 == _SHA
        assert r.ts_reference_sha256 == _SHA
        assert r.diarization_reference_sha256 == _SHA
        assert r.alignment_reference_sha256 == _SHA
        assert r.quality_reference_sha256 == _SHA

    def test_thresholds_custom_pass(self):
        r = _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization(), thresholds=ValidationThresholds(strict_quality=True))
        assert r.validation_passed is True

    def test_num_checks_is_11(self):
        r = _run(_quality(), _alignment([_seg()], num_segments=1, num_aligned=1), _vad(), _asr(), _diarization())
        assert r.num_checks == 11
        assert len(r.check_results) == 11

