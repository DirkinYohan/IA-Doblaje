"""Tests unitarios T14 — Domain Entity TempCleanupResult.

Solo DOMAIN, sin filesystem/IA. Pydantic + patrón DUT.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.core.constants import PipelineStep
from app.domain.entities.cleanup import TempCleanupResult


def _result(**kw):
    base = dict(
        job_id="job-12345678",
        cleaned=True,
        removed_paths=("job-12345678",),
        errors=(),
        success=True,
    )
    base.update(kw)
    return TempCleanupResult(**base)


# =============================================================================
# TempCleanupResult
# =============================================================================


class TestTempCleanupResult:
    def test_valid(self):
        r = _result()
        assert r.step == PipelineStep.TEMP_CLEANUP
        assert r.cleaned is True
        assert r.success is True

    def test_step_always_temp_cleanup(self):
        r = _result()
        assert r.step == PipelineStep.TEMP_CLEANUP

    def test_success_true_no_errors(self):
        r = _result(errors=(), success=True)
        assert r.success is True

    def test_success_true_with_errors_raises(self):
        with pytest.raises(ValidationError):
            _result(errors=("boom",), success=True)

    def test_idempotent_cleaned_true_no_paths(self):
        r = _result(cleaned=True, removed_paths=(), success=True)
        assert r.cleaned is True
        assert r.removed_paths == ()

    def test_skipped_cleaned_false(self):
        r = _result(cleaned=False, removed_paths=(), success=True)
        assert r.cleaned is False
        assert r.success is True

    def test_job_id_none_allowed(self):
        r = _result(job_id=None, cleaned=False, removed_paths=(), success=True)
        assert r.job_id is None

    def test_job_id_empty_raises(self):
        with pytest.raises(ValidationError):
            _result(job_id="")

    def test_removed_paths_must_be_tuple(self):
        with pytest.raises(ValidationError):
            _result(removed_paths="not-a-tuple")  # type: ignore[arg-type]

    def test_errors_must_be_tuple(self):
        with pytest.raises(ValidationError):
            _result(errors="not-a-tuple")  # type: ignore[arg-type]

    def test_cleaned_bool_required(self):
        with pytest.raises(ValidationError):
            _result(cleaned="yes")  # type: ignore[arg-type]

    def test_frozen(self):
        r = _result()
        with pytest.raises(ValidationError):
            r.success = False  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _result(x=1)  # type: ignore[call-arg]

    def test_determinism(self):
        a = _result()
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()

    def test_removed_paths_order_preserved(self):
        r = _result(removed_paths=("a", "b", "c"))
        assert r.removed_paths == ("a", "b", "c")

    def test_errors_order_preserved(self):
        r = _result(errors=("e1", "e2"), success=False, cleaned=False, removed_paths=())
        assert r.errors == ("e1", "e2")
        assert r.success is False

    def test_success_false_with_errors_ok(self):
        r = _result(cleaned=False, removed_paths=(), errors=("x",), success=False)
        assert r.success is False
        assert r.errors == ("x",)


class TestTempCleanupResultAdditional:
    def test_defaults(self):
        r = TempCleanupResult()
        assert r.step == PipelineStep.TEMP_CLEANUP
        assert r.cleaned is False
        assert r.success is False
        assert r.removed_paths == ()
        assert r.errors == ()

    def test_cleaned_requires_success_false_ok(self):
        # cleaned True pero success False con errores → válido (parcial)
        r = _result(cleaned=True, removed_paths=("x",), errors=("e",), success=False)
        assert r.cleaned is True
        assert r.success is False

    def test_no_path_handles(self):
        # removed_paths debe ser strings, no Path
        r = _result(removed_paths=("job-12345678/sub",))
        assert isinstance(r.removed_paths[0], str)
