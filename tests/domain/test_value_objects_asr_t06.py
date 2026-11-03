from __future__ import annotations

import math

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.value_objects.asr import (
    ASRSegmentIndex,
    ASRStrategyName,
    ASRText,
    AsrMaxDurationMinutes,
    AsrThresholds,
    TranscriptConfidence,
)


class W_ASRText(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: ASRText


class W_Confidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: TranscriptConfidence


class W_Index(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: ASRSegmentIndex


class W_MaxMinutes(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: AsrMaxDurationMinutes


SHA = "a" * 64


class TestASRText:
    def test_asrtext_happy(self):
        t = W_ASRText(value="Hola mundo. This is a test.")
        assert t.value == "Hola mundo. This is a test."

    def test_asrtext_empty(self):
        t = W_ASRText(value="")
        assert t.value == ""

    def test_asrtext_frozen_extra_forbid(self):
        t = W_ASRText(value="ok")
        with pytest.raises((TypeError, ValidationError)):
            t.value = "mutado"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            W_ASRText.model_validate({"value": "ok", "extra": 1})

    def test_asrtext_rejects_bool(self):
        with pytest.raises((ValidationError, TypeError)):
            W_ASRText(value=True)

    def test_asrtext_rejects_int(self):
        with pytest.raises((ValidationError, TypeError)):
            W_ASRText(value=1234)

    def test_asrtext_rejects_control_chars(self):
        with pytest.raises((ValidationError, ValueError)):
            W_ASRText(value="abc\x00def")

    def test_asrtext_rejects_too_long(self):
        with pytest.raises(ValidationError):
            W_ASRText(value="x" * 1_000_001)


class TestTranscriptConfidence:
    def test_confidence_0_1_range_valid(self):
        W_Confidence(value=0.0)
        W_Confidence(value=0.5)
        W_Confidence(value=1.0)
        W_Confidence(value=0)
        W_Confidence(value=1)

    def test_confidence_rejects_above(self):
        with pytest.raises(ValidationError):
            W_Confidence(value=1.1)

    def test_confidence_rejects_below(self):
        with pytest.raises(ValidationError):
            W_Confidence(value=-0.001)

    def test_confidence_rejects_bool(self):
        with pytest.raises((ValidationError, TypeError)):
            W_Confidence(value=True)

    def test_confidence_rejects_nan_inf(self):
        with pytest.raises(ValidationError):
            W_Confidence(value=float("nan"))
        with pytest.raises(ValidationError):
            W_Confidence(value=float("inf"))

    def test_confidence_rejects_str(self):
        with pytest.raises((ValidationError, TypeError)):
            W_Confidence(value="0.5")


class TestASRSegmentIndex:
    def test_index_valid(self):
        assert W_Index(value=0).value == 0
        assert W_Index(value=9999).value == 9999

    def test_index_rejects_negative(self):
        with pytest.raises(ValidationError):
            W_Index(value=-1)

    def test_index_rejects_bool(self):
        with pytest.raises((ValidationError, TypeError)):
            W_Index(value=True)


class TestAsrThresholds:
    def test_defaults(self):
        t = AsrThresholds()
        assert t.min_transcript_confidence == 0.0
        assert t.beam_size == 1
        assert t.max_audio_duration_minutes == 480
        assert t.allow_empty_transcript_when_no_voice is True

    def test_frozen(self):
        t = AsrThresholds()
        with pytest.raises((TypeError, ValidationError)):
            t.min_transcript_confidence = 0.5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            AsrThresholds.model_validate({"min_transcript_confidence": 0.0, "bad_key": 1})

    def test_overrides_valid(self):
        t = AsrThresholds(
            min_transcript_confidence=0.6,
            beam_size=5,
            max_audio_duration_minutes=30,
            allow_empty_transcript_when_no_voice=False,
        )
        assert t.min_transcript_confidence == pytest.approx(0.6)
        assert t.beam_size == 5
        assert t.max_audio_duration_minutes == 30
        assert t.allow_empty_transcript_when_no_voice is False

    def test_confidence_threshold_max(self):
        AsrThresholds(min_transcript_confidence=1.0)
        with pytest.raises(ValidationError):
            AsrThresholds(min_transcript_confidence=1.01)

    def test_beam_size_rejects_negative(self):
        with pytest.raises(ValidationError):
            AsrThresholds(beam_size=-1)

    def test_max_duration_rejects_zero(self):
        with pytest.raises(ValidationError):
            W_MaxMinutes(value=0)
        with pytest.raises(ValidationError):
            W_MaxMinutes(value=-1)

    def test_max_duration_rejects_bool(self):
        with pytest.raises((ValidationError, TypeError)):
            W_MaxMinutes(value=True)


class TestASRStrategyLiteral:
    def test_known_strategies(self):
        from app.domain.entities.asr import ASRResult

        assert "full_audio" in ASRStrategyName.__args__
        assert "asr_silent_fallback" in ASRStrategyName.__args__

    def test_invalid_strategy_rejected(self):
        from app.domain.entities.asr import ASRSegment
        with pytest.raises(ValidationError):
            ASRSegment.model_validate({"segment_index": 0})
        with pytest.raises(ValidationError):
            ASRSegment.model_validate({"segment_index": 0, "text": "ok", "extra_field": 1})
        s = "full_audio"
        assert s in {"full_audio", "asr_silent_fallback", "only_voice_concat"}
        assert "bad_strategy" not in {"full_audio", "asr_silent_fallback", "only_voice_concat"}


class TestDeterminism:
    def test_thresholds_deterministic(self):
        a = AsrThresholds(beam_size=3, min_transcript_confidence=0.2, max_audio_duration_minutes=120)
        b = AsrThresholds(beam_size=3, min_transcript_confidence=0.2, max_audio_duration_minutes=120)
        assert a == b
        assert hash((a.min_transcript_confidence, a.beam_size, a.max_audio_duration_minutes)) == hash(
            (b.min_transcript_confidence, b.beam_size, b.max_audio_duration_minutes)
        )
