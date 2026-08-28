"""Tests unitarios T12 — Domain Value Objects y Entity Metrics.

Solo DOMAIN, sin IA/infraestructura. Pydantic + patrón DUT.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.core.constants import PipelineStep
from app.domain.entities.metrics import MetricsResult
from app.domain.value_objects.metrics import (
    PIPELINE_STEP_NAMES,
    RuntimeMetricsSnapshot,
)


_SHA = "a" * 64
_SHA_B = "b" * 64


def _snapshot(**kw):
    base = dict(processing_wall_time_sec=10.0)
    base.update(kw)
    return RuntimeMetricsSnapshot(**base)


def _result(**kw):
    base = dict(
        source_preprocessed_sha256=_SHA,
        validation_reference_sha256=_SHA,
        metrics_enabled=True,
        processing_wall_time_sec=10.0,
        audio_duration_sec=10.0,
        real_time_factor_RTF=1.0,
    )
    base.update(kw)
    return MetricsResult(**base)


# =============================================================================
# RuntimeMetricsSnapshot
# =============================================================================


class TestRuntimeMetricsSnapshot:
    def test_valid_minimal(self):
        s = _snapshot()
        assert s.processing_wall_time_sec == 10.0

    def test_step_timings_valid_keys(self):
        s = _snapshot(step_timings_sec={"MEDIA_VALIDATION": 1.0, "TEMP_CLEANUP": 2.0})
        assert s.step_timings_sec["MEDIA_VALIDATION"] == 1.0

    def test_step_timings_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(step_timings_sec={"UNKNOWN_STEP": 1.0})

    def test_step_timings_negative_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(step_timings_sec={"MEDIA_VALIDATION": -1.0})

    def test_step_timings_nan_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(step_timings_sec={"MEDIA_VALIDATION": float("nan")})

    def test_wall_time_negative_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(processing_wall_time_sec=-1.0)

    def test_wall_time_bool_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(processing_wall_time_sec=True)  # type: ignore[arg-type]

    def test_percent_range(self):
        with pytest.raises(ValidationError):
            _snapshot(cpu_utilization_avg_percent=101.0)

    def test_frozen(self):
        s = _snapshot()
        with pytest.raises(ValidationError):
            s.processing_wall_time_sec = 5.0  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _snapshot(x=1)  # type: ignore[call-arg]

    def test_all_15_step_names_exist(self):
        assert len(PIPELINE_STEP_NAMES) == 15
        assert set(PIPELINE_STEP_NAMES) == {s.name for s in PipelineStep}


# =============================================================================
# MetricsResult
# =============================================================================


class TestMetricsResult:
    def test_valid(self):
        r = _result()
        assert r.step == PipelineStep.METRICS_AGGREGATION
        assert r.model_label == "t12:metrics-aggregation:v1"

    def test_sha_mismatch_raises(self):
        with pytest.raises(ValidationError):
            _result(validation_reference_sha256=_SHA_B)

    def test_sha_invalid(self):
        with pytest.raises(ValidationError):
            _result(source_preprocessed_sha256="bad")

    def test_rtf_consistency(self):
        with pytest.raises(ValidationError):
            _result(
                processing_wall_time_sec=10.0,
                audio_duration_sec=5.0,
                real_time_factor_RTF=999.0,  # incorrecto
            )

    def test_rtf_none_allowed(self):
        r = _result(audio_duration_sec=0.0, real_time_factor_RTF=None)
        assert r.real_time_factor_RTF is None

    def test_counts_non_negative(self):
        with pytest.raises(ValidationError):
            _result(segment_count_total=-1)

    def test_count_rejects_bool(self):
        with pytest.raises(ValidationError):
            _result(segment_count_total=True)  # type: ignore[arg-type]

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            _result(average_asr_confidence=1.5)

    def test_confidence_rejects_nan(self):
        with pytest.raises(ValidationError):
            _result(average_asr_confidence=float("nan"))

    def test_percent_range(self):
        with pytest.raises(ValidationError):
            _result(cpu_utilization_avg_percent=150.0)

    def test_frozen(self):
        r = _result()
        with pytest.raises(ValidationError):
            r.segment_count_total = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _result(x=1)  # type: ignore[call-arg]

    def test_metadata_no_bytes(self):
        with pytest.raises(ValidationError):
            _result(analysis_metadata={"raw": b"x"})  # type: ignore[dict-item]

    def test_metrics_disabled_valid(self):
        r = _result(metrics_enabled=False, processing_wall_time_sec=0.0, real_time_factor_RTF=None)
        assert r.metrics_enabled is False

    def test_determinism(self):
        a = _result()
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()
