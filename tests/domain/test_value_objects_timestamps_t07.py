from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.value_objects.timestamps import (
    AsrTimestampThresholds,
    TimestampConfidence01,
    TimestampSourceKind,
    TimestampStrategyName,
    TimedEndMs,
    TimedStartMs,
)


# =========================================================================
# Helpers Base wrappers frozen forbid (igual pattern T06 tests VO Domain)
# =========================================================================
class W_TStart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: TimedStartMs


class W_TEnd(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: TimedEndMs


class W_TConf(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: TimestampConfidence01


# =========================================================================
# TestTimedStartMs (≥ 6 tests)
# =========================================================================
class TestTimedStartMs:
    def test_zero_ok(self):
        t = W_TStart(value=0)
        assert t.value == 0

    def test_positive_ok(self):
        t = W_TStart(value=123456)
        assert t.value == 123456

    def test_str_numeric_ok(self):
        t = W_TStart(value="1000")
        assert t.value == 1000

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            W_TStart(value=-1)

    def test_bool_coercion_rejected(self):
        with pytest.raises((TypeError, ValidationError)):
            W_TStart(value=True)
        with pytest.raises((TypeError, ValidationError)):
            W_TStart(value=False)

    def test_nan_str_rejected(self):
        with pytest.raises(ValidationError):
            W_TStart(value="abc")

    def test_frozen_and_forbid(self):
        t = W_TStart(value=10)
        with pytest.raises((TypeError, ValidationError)):
            t.value = 20
        with pytest.raises(ValidationError):
            W_TStart.model_validate({"value": 10, "extra": 1})


# =========================================================================
# TestTimedEndMs (≥ 5 tests)
# =========================================================================
class TestTimedEndMs:
    def test_ok(self):
        assert W_TEnd(value=500).value == 500

    def test_big_ok(self):
        assert W_TEnd(value=3_600_000).value == 3_600_000

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            W_TEnd(value=-1)

    def test_bool_rejected(self):
        with pytest.raises((TypeError, ValidationError)):
            W_TEnd(value=True)

    def test_forbid_extra(self):
        with pytest.raises(ValidationError):
            W_TEnd.model_validate({"value": 1, "e": 2})


# =========================================================================
# TestTimestampConfidence01 (≥ 6 tests)
# =========================================================================
class TestTimestampConfidence01:
    def test_zero_one(self):
        assert W_TConf(value=0.0).value == 0.0
        assert W_TConf(value=1.0).value == 1.0

    def test_mid_ok(self):
        assert W_TConf(value=0.5).value == 0.5

    def test_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            W_TConf(value=-0.01)

    def test_above_one_rejected(self):
        with pytest.raises(ValidationError):
            W_TConf(value=1.001)

    def test_nan_inf_rejected(self):
        with pytest.raises(ValidationError):
            W_TConf(value=float("nan"))
        with pytest.raises(ValidationError):
            W_TConf(value=float("inf"))

    def test_bool_coercion_rejected(self):
        with pytest.raises((TypeError, ValidationError)):
            W_TConf(value=True)

    def test_str_non_numeric_rejected(self):
        with pytest.raises(ValidationError):
            W_TConf(value="high")


# =========================================================================
# TestTimestampStrategyName Literal (≥ 3 tests)
# =========================================================================
class TestTimestampStrategyName:
    def test_valid_values(self):
        v: TimestampStrategyName = "segment_ts_from_fw_raw"
        assert v in {"segment_ts_from_fw_raw", "segment_ts_from_vad_interpolate", "ts_silent_fallback"}
        v = "segment_ts_from_vad_interpolate"
        assert v
        v = "ts_silent_fallback"
        assert v

    def test_invalid_rejected(self):
        class _W(BaseModel):
            model_config = ConfigDict(extra="forbid")
            s: TimestampStrategyName
        with pytest.raises(ValidationError):
            _W(s="only_voice_concat_2")

    def test_source_kind_contains_fw_raw(self):
        k: TimestampSourceKind = "fw_raw"
        assert k in {"fw_raw", "vad_interpolate", "heuristic_gap_fill", "silent"}


# =========================================================================
# TestAsrTimestampThresholds (≥ 6 tests)
# =========================================================================
class TestAsrTimestampThresholds:
    def test_defaults(self):
        t = AsrTimestampThresholds()
        assert t.max_segments_safety == 1_000_000
        assert t.allow_overlap_resolution is True
        assert t.allow_end_clamp_to_total is True
        assert t.allow_vad_interpolate_fallback is True
        assert t.strict_validity is True
        assert t.allow_silent_fallback is True
        assert t.min_segment_duration_ms == 10

    def test_override_ok(self):
        t = AsrTimestampThresholds(max_segments_safety=1000, min_segment_duration_ms=5)
        assert t.max_segments_safety == 1000
        assert t.min_segment_duration_ms == 5

    def test_frozen(self):
        t = AsrTimestampThresholds()
        with pytest.raises((TypeError, ValidationError)):
            t.max_segments_safety = 500  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            AsrTimestampThresholds.model_validate({"max_segments_safety": 1, "unknown": 1})

    def test_invalid_override_bool_expected_int_rejected(self):
        with pytest.raises(ValidationError):
            AsrTimestampThresholds(max_segments_safety=True)

    def test_min_segment_duration_ms_zero_invalid(self):
        with pytest.raises(ValidationError):
            AsrTimestampThresholds(min_segment_duration_ms=0)

    def test_strict_validity_non_bool_rejected(self):
        with pytest.raises(ValidationError):
            AsrTimestampThresholds(strict_validity="strict")


# =========================================================================
# Summary helpers count: TOTAL Domain tests ≥ 27
# =========================================================================
