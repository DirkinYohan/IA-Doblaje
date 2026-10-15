"""Tests unitarios T08 — Domain Value Objects y Entity Diarization.

Solo DOMAIN, sin torch/pyannote/infrastructure. Pydantic + patrón DUT T04/T05.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.entities.diarization import (
    DIARIZATION_STRATEGY_PYANNOTE,
    DIARIZATION_STRATEGY_SILENT_FALLBACK,
    DiarizationResult,
)
from app.domain.value_objects.diarization import (
    DiarizationThresholds,
    SpeakerLabel,
    SpeakerTurn,
)


_SHA = "a" * 64
_SHA_B = "b" * 64


class _LabelDut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    v: SpeakerLabel


class _TurnDut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    v: SpeakerTurn


# =============================================================================
# SpeakerLabel
# =============================================================================


class TestSpeakerLabel:
    @pytest.mark.parametrize(
        "label", ["SPEAKER_00", "SPEAKER_01", "SPEAKER_09", "SPEAKER_99"]
    )
    def test_valid(self, label: str):
        assert _LabelDut(v=label).v == label

    @pytest.mark.parametrize(
        "bad",
        ["SPEAKER_0", "SPEAKER_000", "SPEAKER_100", "speaker_00", "speaker-00", "SPK_00"],
    )
    def test_invalid(self, bad: str):
        with pytest.raises(ValidationError):
            _LabelDut(v=bad)

    def test_strip_spaces_but_case_sensitive(self):
        assert _LabelDut(v=" SPEAKER_01 ").v == "SPEAKER_01"

    def test_lowercase_is_invalid_case_sensitive(self):
        with pytest.raises(ValidationError):
            _LabelDut(v="speaker_00")

    def test_regex_is_strict_no_partial(self):
        with pytest.raises(ValidationError):
            _LabelDut(v="SPEAKER_00_extra")

    def test_reject_bool(self):
        with pytest.raises(ValidationError):
            _LabelDut(v=True)  # type: ignore[arg-type]

    def test_frozen_no_mutation(self):
        d = _LabelDut(v="SPEAKER_00")
        with pytest.raises(ValidationError):
            d.v = "SPEAKER_01"  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _LabelDut(v="SPEAKER_00", extra=1)  # type: ignore[call-arg]


# =============================================================================
# SpeakerTurn
# =============================================================================


class TestSpeakerTurn:
    def test_valid_turn(self):
        t = SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000)
        assert t.duration_ms == 1000
        assert t.confidence == 1.0

    def test_duration_recomputed(self):
        t = SpeakerTurn(speaker_label="SPEAKER_00", start_ms=100, end_ms=300, duration_ms=999)
        assert t.duration_ms == 200

    def test_end_must_be_greater_than_start(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=1000, end_ms=1000)

    def test_start_ge_zero(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=-1, end_ms=100)

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=100, confidence=1.5)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_confidence_rejects_nan_inf(self, bad: float):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=100, confidence=bad)

    def test_confidence_rejects_bool(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=100, confidence=True)  # type: ignore[arg-type]

    def test_frozen(self):
        t = SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=100)
        with pytest.raises(ValidationError):
            t.start_ms = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=100, x=1)  # type: ignore[call-arg]

    def test_duration_seconds_property(self):
        t = SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1500)
        assert t.duration_seconds == 1.5


# =============================================================================
# DiarizationThresholds
# =============================================================================


class TestDiarizationThresholds:
    def test_defaults(self):
        th = DiarizationThresholds()
        assert th.min_speaker_duration_ms == 200
        assert th.min_gap_ms == 0
        assert th.max_num_speakers == 10
        assert th.allow_merge_proximal is True
        assert th.allow_silent_fallback is True
        assert th.strict_validity is True
        assert th.max_turns_safety == 100000
        assert th.max_overlap_tolerance_ms == 0

    def test_frozen(self):
        th = DiarizationThresholds()
        with pytest.raises(ValidationError):
            th.min_gap_ms = 10  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            DiarizationThresholds(unknown=1)  # type: ignore[call-arg]

    def test_type_strict_int(self):
        with pytest.raises(ValidationError):
            DiarizationThresholds(min_gap_ms="abc")  # type: ignore[arg-type]

    def test_type_strict_bool_rejected_as_int(self):
        with pytest.raises(ValidationError):
            DiarizationThresholds(min_gap_ms=True)  # type: ignore[arg-type]

    def test_bounds_min_speaker(self):
        with pytest.raises(ValidationError):
            DiarizationThresholds(min_speaker_duration_ms=0)
        with pytest.raises(ValidationError):
            DiarizationThresholds(min_speaker_duration_ms=1_000_000)


# =============================================================================
# DiarizationResult
# =============================================================================


def _turn(label: str, s: int, e: int) -> SpeakerTurn:
    return SpeakerTurn(speaker_label=label, start_ms=s, end_ms=e)


class TestDiarizationResult:
    def test_happy_path_two_speakers(self):
        r = DiarizationResult(
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            total_duration_ms=10000,
            speaker_turns=(
                _turn("SPEAKER_00", 0, 1000),
                _turn("SPEAKER_01", 1000, 2000),
                _turn("SPEAKER_00", 2000, 3000),
            ),
            num_speakers=2,
            num_turns=3,
        )
        assert r.strategy == DIARIZATION_STRATEGY_PYANNOTE
        assert r.model_label == "pyannote:speaker-diarization:3.1:local-v1"
        assert len(r.speaker_turns) == 3

    def test_sha_mismatch_raises(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA_B,
                total_duration_ms=1000,
                speaker_turns=(),
                num_speakers=0,
                num_turns=0,
            )

    def test_sha_invalid_format(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256="not-a-sha",
                vad_reference_sha256="not-a-sha",
                total_duration_ms=1000,
                speaker_turns=(),
                num_speakers=0,
                num_turns=0,
            )

    def test_num_turns_consistency(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(_turn("SPEAKER_00", 0, 100),),
                num_speakers=1,
                num_turns=5,
            )

    def test_num_speakers_consistency(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(
                    _turn("SPEAKER_00", 0, 100),
                    _turn("SPEAKER_01", 100, 200),
                ),
                num_speakers=1,
                num_turns=2,
            )

    def test_order_asc_required(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(
                    _turn("SPEAKER_00", 500, 600),
                    _turn("SPEAKER_01", 0, 100),
                ),
                num_speakers=2,
                num_turns=2,
            )

    def test_overlap_raises(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(
                    _turn("SPEAKER_00", 0, 100),
                    _turn("SPEAKER_01", 50, 200),
                ),
                num_speakers=2,
                num_turns=2,
            )

    def test_end_le_total(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(_turn("SPEAKER_00", 0, 1500),),
                num_speakers=1,
                num_turns=1,
            )

    def test_silent_fallback_requires_empty(self):
        r = DiarizationResult(
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            total_duration_ms=1000,
            speaker_turns=(),
            num_speakers=0,
            num_turns=0,
            strategy=DIARIZATION_STRATEGY_SILENT_FALLBACK,
        )
        assert r.strategy == DIARIZATION_STRATEGY_SILENT_FALLBACK

    def test_silent_fallback_with_turns_raises(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(_turn("SPEAKER_00", 0, 100),),
                num_speakers=1,
                num_turns=1,
                strategy=DIARIZATION_STRATEGY_SILENT_FALLBACK,
            )

    def test_frozen(self):
        r = DiarizationResult(
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            total_duration_ms=1000,
            speaker_turns=(),
            num_speakers=0,
            num_turns=0,
        )
        with pytest.raises(ValidationError):
            r.num_turns = 1  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(),
                num_speakers=0,
                num_turns=0,
                extra_field=1,  # type: ignore[call-arg]
            )

    def test_metadata_no_bytes(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(),
                num_speakers=0,
                num_turns=0,
                analysis_metadata={"raw": b"bytes"},  # type: ignore[dict-item]
            )

    def test_determinism_equal_instances(self):
        a = DiarizationResult(
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            total_duration_ms=1000,
            speaker_turns=(_turn("SPEAKER_00", 0, 100),),
            num_speakers=1,
            num_turns=1,
        )
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()

    def test_strategy_literal_only(self):
        with pytest.raises(ValidationError):
            DiarizationResult(
                source_preprocessed_sha256=_SHA,
                vad_reference_sha256=_SHA,
                total_duration_ms=1000,
                speaker_turns=(),
                num_speakers=0,
                num_turns=0,
                strategy="other_strategy",  # type: ignore[arg-type]
            )
