"""Tests unitarios T11 — Domain Value Objects y Entity Validation.

Solo DOMAIN, sin IA/infraestructura. Pydantic + patrón DUT.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.constants import QualityLevel
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.validation import (
    ValidationCheckResult,
    ValidationThresholds,
)


_SHA = "a" * 64
_SHA_B = "b" * 64


def _check(code: str, outcome: str = "passed"):
    return ValidationCheckResult(check_code=code, outcome=outcome, message="")


def _all_checks():
    return tuple(
        _check(c) for c in
        ("VC01", "VC02", "VC03", "VC04", "VC05", "VC06", "VC07", "VC08", "VC09", "VC10", "VC11")
    )


def _result(**kw):
    base = dict(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA,
        quality_reference_sha256=_SHA,
        validation_passed=True,
        validation_status="passed",
        quality_level=QualityLevel.PASSED,
        overall_quality_score=1.0,
        validation_errors=(),
        validation_warnings=(),
        num_checks=11,
        num_errors=0,
        num_warnings=0,
        num_passed=11,
        check_results=_all_checks(),
    )
    base.update(kw)
    return ValidationResult(**base)


# =============================================================================
# ValidationCheckResult
# =============================================================================


class TestValidationCheckResult:
    def test_valid(self):
        r = _check("VC01", "passed")
        assert r.check_code == "VC01"
        assert r.outcome == "passed"

    def test_check_code_literal(self):
        with pytest.raises(ValidationError):
            ValidationCheckResult(check_code="VC99", outcome="passed")

    def test_outcome_literal(self):
        with pytest.raises(ValidationError):
            ValidationCheckResult(check_code="VC01", outcome="invalid")

    def test_frozen(self):
        r = _check("VC01")
        with pytest.raises(ValidationError):
            r.outcome = "error"  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            ValidationCheckResult(check_code="VC01", outcome="passed", x=1)  # type: ignore[call-arg]


# =============================================================================
# ValidationThresholds
# =============================================================================


class TestValidationThresholds:
    def test_defaults(self):
        th = ValidationThresholds()
        assert th.strict_quality is True
        assert th.pass_threshold == 0.80
        assert th.warn_threshold == 0.30

    def test_pass_gt_warn(self):
        with pytest.raises(ValidationError):
            ValidationThresholds(pass_threshold=0.2, warn_threshold=0.5)

    def test_frozen(self):
        th = ValidationThresholds()
        with pytest.raises(ValidationError):
            th.strict_quality = False  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            ValidationThresholds(x=1)  # type: ignore[call-arg]

    def test_bool_rejected(self):
        with pytest.raises(ValidationError):
            ValidationThresholds(pass_threshold=True)  # type: ignore[arg-type]


# =============================================================================
# ValidationResult
# =============================================================================


class TestValidationResult:
    def test_valid_passed(self):
        r = _result()
        assert r.validation_status == "passed"
        assert r.model_label == "t11:validation:v1"
        assert r.num_checks == 11

    def test_sha_chain_mismatch(self):
        with pytest.raises(ValidationError):
            _result(vad_reference_sha256=_SHA_B)

    def test_quality_ref_mismatch(self):
        with pytest.raises(ValidationError):
            _result(quality_reference_sha256=_SHA_B)

    def test_sha_invalid(self):
        with pytest.raises(ValidationError):
            _result(source_preprocessed_sha256="bad")

    def test_num_checks_must_be_11(self):
        with pytest.raises(ValidationError):
            _result(num_checks=10)

    def test_check_results_order(self):
        checks = list(_all_checks())
        checks[0], checks[1] = checks[1], checks[0]
        with pytest.raises(ValidationError):
            _result(check_results=tuple(checks))

    def test_counters_consistent(self):
        checks = list(_all_checks())
        checks[0] = _check("VC01", "error")
        with pytest.raises(ValidationError):
            _result(
                check_results=tuple(checks),
                validation_passed=True,  # inconsistente: hay error
                validation_status="passed",
                num_errors=0,
                num_passed=11,
            )

    def test_validation_passed_consistent(self):
        checks = list(_all_checks())
        checks[0] = _check("VC01", "error")
        with pytest.raises(ValidationError):
            _result(check_results=tuple(checks), validation_passed=True, num_errors=1)

    def test_status_failed_requires_error(self):
        with pytest.raises(ValidationError):
            _result(validation_status="failed", validation_passed=True, num_errors=0)

    def test_errors_length_consistency(self):
        with pytest.raises(ValidationError):
            _result(validation_errors=("x",), num_errors=0)

    def test_frozen(self):
        r = _result()
        with pytest.raises(ValidationError):
            r.num_checks = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _result(extra=1)  # type: ignore[call-arg]

    def test_metadata_no_bytes(self):
        with pytest.raises(ValidationError):
            _result(analysis_metadata={"raw": b"bytes"})  # type: ignore[dict-item]

    def test_score_rejects_nan(self):
        with pytest.raises(ValidationError):
            _result(overall_quality_score=float("nan"))

    def test_determinism(self):
        a = _result()
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()


class TestValidationResultAdditional:
    def test_status_warning_with_warnings(self):
        checks = list(_all_checks())
        checks[10] = _check("VC11", "warning")
        r = _result(
            check_results=tuple(checks),
            validation_status="warning",
            validation_warnings=("VC11: quality_reference mismatch",),
            num_warnings=1,
            num_passed=10,
        )
        assert r.validation_status == "warning"
        assert r.num_warnings == 1

    def test_status_passed_rejects_warnings_inconsistent(self):
        # passed pero con warning contado → inconsistente
        checks = list(_all_checks())
        checks[10] = _check("VC11", "warning")
        with pytest.raises(ValidationError):
            _result(
                check_results=tuple(checks),
                validation_status="passed",
                num_warnings=1,
                num_passed=10,
                validation_passed=True,
            )

    def test_quality_level_passthrough(self):
        r = _result(quality_level=QualityLevel.WARNING, overall_quality_score=0.5)
        assert r.quality_level == QualityLevel.WARNING
        assert r.overall_quality_score == 0.5

    def test_score_rejects_inf(self):
        with pytest.raises(ValidationError):
            _result(overall_quality_score=float("inf"))

    def test_score_rejects_bool(self):
        with pytest.raises(ValidationError):
            _result(overall_quality_score=True)  # type: ignore[arg-type]

    def test_count_rejects_bool(self):
        with pytest.raises(ValidationError):
            _result(num_checks=True)  # type: ignore[arg-type]
