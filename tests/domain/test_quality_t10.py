"""Tests unitarios T10 — Domain Value Objects y Entity Quality.

Solo DOMAIN, sin IA/infraestructura. Pydantic + patrón DUT.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.constants import QualityLevel
from app.domain.entities.quality import (
    QUALITY_STRATEGY_RULES,
    QUALITY_STRATEGY_SILENT_FALLBACK,
    QualityResult,
)
from app.domain.value_objects.quality import (
    QualityRuleResult,
    QualityThresholds,
)


_SHA = "a" * 64
_SHA_B = "b" * 64


def _rule(code: str, severity: str = "WARNING", passed: bool = True, applicable: bool = True):
    return QualityRuleResult(
        rule_code=code, severity=severity, passed=passed, applicable=applicable,
        occurrences=0, penalty=0.0, reason="",
    )


def _all_rules_passed():
    return tuple(
        _rule(code) for code in
        ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
    )


def _result(**kw):
    base = dict(
        strategy=QUALITY_STRATEGY_RULES,
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA,
        quality_level=QualityLevel.PASSED,
        overall_quality_score=1.0,
        num_rules_evaluated=11,
        num_rules_passed=11,
        num_rules_warning=0,
        num_rules_failed=0,
        rule_results=_all_rules_passed(),
        average_alignment_confidence=0.0,
        average_ts_confidence=0.0,
        average_asr_confidence=0.0,
        num_segments=0,
        num_segments_aligned=0,
        num_segments_unassigned=0,
        total_duration_ms=0,
        transcript_language_code="es",
    )
    base.update(kw)
    return QualityResult(**base)


# =============================================================================
# QualityRuleResult
# =============================================================================


class TestQualityRuleResult:
    def test_valid(self):
        r = _rule("QR01", severity="ERROR", passed=True)
        assert r.rule_code == "QR01"
        assert r.severity == "ERROR"
        assert r.applicable is True

    def test_severity_literal(self):
        with pytest.raises(ValidationError):
            QualityRuleResult(rule_code="QR01", severity="INFO")

    def test_rule_code_literal(self):
        with pytest.raises(ValidationError):
            QualityRuleResult(rule_code="QR99", severity="WARNING")

    def test_not_applicable_forces_zero(self):
        with pytest.raises(ValidationError):
            QualityRuleResult(
                rule_code="QR01", severity="WARNING", passed=True,
                applicable=False, occurrences=3, penalty=0.0, reason="",
            )

    def test_not_applicable_forces_zero_penalty(self):
        with pytest.raises(ValidationError):
            QualityRuleResult(
                rule_code="QR01", severity="WARNING", passed=True,
                applicable=False, occurrences=0, penalty=0.5, reason="",
            )

    def test_penalty_rejects_nan(self):
        with pytest.raises(ValidationError):
            QualityRuleResult(
                rule_code="QR01", severity="WARNING", passed=True,
                applicable=True, occurrences=0, penalty=float("nan"), reason="",
            )

    def test_frozen(self):
        r = _rule("QR01")
        with pytest.raises(ValidationError):
            r.occurrences = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            QualityRuleResult(rule_code="QR01", severity="WARNING", x=1)  # type: ignore[call-arg]


# =============================================================================
# QualityThresholds
# =============================================================================


class TestQualityThresholds:
    def test_defaults(self):
        th = QualityThresholds()
        assert th.pass_threshold == 0.80
        assert th.warn_threshold == 0.30
        assert th.max_overlap_ms == 100
        assert th.min_asr_segment_conf == 0.40
        assert th.min_word_conf == 0.30
        assert th.max_segment_sec == 60.0
        assert th.max_silence_sec == 15.0

    def test_pass_gt_warn(self):
        with pytest.raises(ValidationError):
            QualityThresholds(pass_threshold=0.2, warn_threshold=0.5)

    def test_frozen(self):
        th = QualityThresholds()
        with pytest.raises(ValidationError):
            th.pass_threshold = 0.9  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            QualityThresholds(x=1)  # type: ignore[call-arg]

    def test_bool_rejected(self):
        with pytest.raises(ValidationError):
            QualityThresholds(max_overlap_ms=True)  # type: ignore[arg-type]


# =============================================================================
# QualityResult
# =============================================================================


class TestQualityResult:
    def test_valid_passed(self):
        r = _result()
        assert r.quality_level == QualityLevel.PASSED
        assert r.model_label == "t10:quality-analysis:v1"
        assert r.num_rules_evaluated == 11

    def test_sha_chain_mismatch(self):
        with pytest.raises(ValidationError):
            _result(vad_reference_sha256=_SHA_B)

    def test_alignment_ref_mismatch(self):
        with pytest.raises(ValidationError):
            _result(alignment_reference_sha256=_SHA_B)

    def test_sha_invalid(self):
        with pytest.raises(ValidationError):
            _result(source_preprocessed_sha256="bad")

    def test_num_rules_evaluated_must_be_11(self):
        with pytest.raises(ValidationError):
            _result(num_rules_evaluated=10)

    def test_rule_results_order(self):
        rules = list(_all_rules_passed())
        rules[0], rules[1] = rules[1], rules[0]
        with pytest.raises(ValidationError):
            _result(rule_results=tuple(rules))

    def test_level_consistent_with_score(self):
        # score 1.0 → passed; forzar failed sería inconsistente
        with pytest.raises(ValidationError):
            _result(overall_quality_score=1.0, quality_level=QualityLevel.FAILED)

    def test_level_warning(self):
        rules = list(_all_rules_passed())
        # una regla WARNING no passed
        rules[4] = QualityRuleResult(
            rule_code="QR05", severity="WARNING", passed=False, applicable=True,
            occurrences=1, penalty=0.03, reason="x",
        )
        r = _result(
            rule_results=tuple(rules),
            overall_quality_score=0.97,
            quality_level=QualityLevel.PASSED,
            num_rules_passed=10,
            num_rules_warning=1,
        )
        assert r.num_rules_warning == 1

    def test_counters_consistent(self):
        rules = list(_all_rules_passed())
        rules[0] = QualityRuleResult(
            rule_code="QR01", severity="ERROR", passed=False, applicable=True,
            occurrences=1, penalty=1.0, reason="x",
        )
        with pytest.raises(ValidationError):
            _result(
                rule_results=tuple(rules),
                overall_quality_score=0.0,
                quality_level=QualityLevel.FAILED,
                num_rules_passed=11,  # incorrecto: QR01 failed
                num_rules_failed=0,
                num_rules_warning=0,
            )

    def test_silent_fallback(self):
        r = _result(
            strategy=QUALITY_STRATEGY_SILENT_FALLBACK,
            rule_results=tuple(
                QualityRuleResult(
                    rule_code=code, severity="WARNING", passed=True, applicable=False,
                    occurrences=0, penalty=0.0, reason="silent_fallback_no_voice",
                )
                for code in ("QR01", "QR02", "QR03", "QR04", "QR05", "QR06", "QR07", "QR08", "QR09", "QR10", "QR11")
            ),
            num_rules_passed=11,
        )
        assert r.strategy == QUALITY_STRATEGY_SILENT_FALLBACK

    def test_silent_fallback_requires_zero_segments(self):
        with pytest.raises(ValidationError):
            _result(strategy=QUALITY_STRATEGY_SILENT_FALLBACK, num_segments=5)

    def test_frozen(self):
        r = _result()
        with pytest.raises(ValidationError):
            r.overall_quality_score = 0.5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _result(extra=1)  # type: ignore[call-arg]

    def test_metadata_no_bytes(self):
        with pytest.raises(ValidationError):
            _result(analysis_metadata={"raw": b"bytes"})  # type: ignore[dict-item]

    def test_determinism(self):
        a = _result()
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()
