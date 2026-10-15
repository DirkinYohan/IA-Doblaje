"""Tests unitarios Value Objects VAD T04 Step 04.

Domain VO. Sin torch/silero/soundfile. pydantic only.
"""
from __future__ import annotations

import re

import pytest
from pydantic import BaseModel, ValidationError

from app.domain.value_objects.vad import (
    SilenceSegment,
    SpeechConfidence,
    VadThresholds,
    VoiceInterval,
    MsTimestamp,
)


class _MsDut(BaseModel):
    v: MsTimestamp


class _ConfDut(BaseModel):
    v: SpeechConfidence


def test_ms_timestamp_zero_and_positive_ok() -> None:
    for x in (0, 1, 1_000, 10_000_000):
        assert _MsDut(v=x).v == int(x)


@pytest.mark.parametrize(
    "bad", [-1, -100, None, "abc_noentero", None, [], object()]
)
def test_ms_timestamp_invalid_raises(bad) -> None:
    with pytest.raises((ValidationError, TypeError)):
        _MsDut(v=bad)


def test_speech_confidence_valid() -> None:
    for x in (0.0, 0.5, 1.0, 0.999, 1e-6):
        obj = _ConfDut(v=x)
        assert 0.0 <= float(obj.v) <= 1.0


@pytest.mark.parametrize("bad", [-0.01, 1.0001, 1.5, None, [], 2.0])
def test_speech_confidence_invalid_raises(bad) -> None:
    with pytest.raises((ValidationError, TypeError)):
        _ConfDut(v=bad)


def test_voice_interval_valid_ok() -> None:
    vi = VoiceInterval(
        start_ms=100, end_ms=250, max_confidence=0.8, duration_ms=150, sample_count=2400
    )
    assert vi.start_ms == 100
    assert vi.end_ms == 250
    assert vi.duration_ms == 150
    assert vi.duration_seconds == 0.15
    assert vi.sample_count == 2400


def test_voice_interval_duration_autocorrect_wrong() -> None:
    # duration_ms != end-start → autocorrect single-source
    vi = VoiceInterval(
        start_ms=100, end_ms=250, max_confidence=0.8, duration_ms=9999, sample_count=1
    )
    assert vi.duration_ms == 150


def test_voice_interval_end_lte_start_raises() -> None:
    with pytest.raises(ValidationError):
        VoiceInterval(
            start_ms=200, end_ms=200, max_confidence=0.1, sample_count=0
        )


def test_voice_interval_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        VoiceInterval(start_ms=0, end_ms=100, max_confidence=1.1, sample_count=0)


def test_voice_interval_sample_count_neg_raises() -> None:
    with pytest.raises(ValidationError):
        VoiceInterval(
            start_ms=0, end_ms=100, max_confidence=0.5, sample_count=-1
        )


def test_voice_interval_is_frozen() -> None:
    vi = VoiceInterval(
        start_ms=0, end_ms=100, max_confidence=0.5, duration_ms=100, sample_count=1600
    )
    with pytest.raises(ValidationError):
        vi.start_ms = 50  # type: ignore[misc]


def test_silence_segment_valid() -> None:
    s = SilenceSegment(
        start_ms=0, end_ms=250, speech_index_before=-1, speech_index_after=0
    )
    assert s.duration_ms == 250


def test_silence_segment_end_less_start_raises() -> None:
    with pytest.raises(ValidationError):
        SilenceSegment(start_ms=500, end_ms=300, speech_index_before=0, speech_index_after=1)


def test_vad_thresholds_defaults_ok() -> None:
    t = VadThresholds()
    assert t.speech_threshold == pytest.approx(0.5)
    assert t.min_speech_duration_ms == 200
    assert t.min_silence_between_ms == 300
    assert t.merge_proximal_ms == 200


def test_vad_thresholds_valid_overrides() -> None:
    t = VadThresholds(
        speech_threshold=0.3,
        min_speech_duration_ms=50,
        min_silence_between_ms=0,
        merge_proximal_ms=50,
    )
    assert float(t.speech_threshold) == 0.3


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"speech_threshold": 1.1},
        {"speech_threshold": -0.1},
        {"speech_threshold": None},
        {"min_speech_duration_ms": 10},
        {"merge_proximal_ms": 1000000},
    ],
)
def test_vad_thresholds_invalid_raises(bad_kwargs) -> None:
    with pytest.raises(ValidationError):
        VadThresholds(**bad_kwargs)


def test_vad_thresholds_extra_forbid_raises() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        VadThresholds(inventado=True)
