"""Tests unitarios T09 — Domain Value Objects y Entity Alignment.

Solo DOMAIN, sin IA/infraestructura. Pydantic + patrón DUT T04/T05/T08.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.entities.alignment import (
    ALIGNMENT_STRATEGY_SILENT_FALLBACK,
    ALIGNMENT_STRATEGY_WOMV,
    AlignmentResult,
)
from app.domain.value_objects.alignment import (
    AlignmentThresholds,
    DialogueSegment,
)


_SHA = "a" * 64
_SHA_B = "b" * 64


class _SegDut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    v: DialogueSegment


def _seg(**kw):
    base = dict(
        segment_index=0,
        start_ms=0,
        end_ms=1000,
        duration_ms=1000,
        text="hola",
        speaker_label="SPEAKER_00",
        alignment_confidence=1.0,
        ts_confidence=1.0,
    )
    base.update(kw)
    return DialogueSegment(**base)


# =============================================================================
# DialogueSegment
# =============================================================================


class TestDialogueSegment:
    def test_valid(self):
        s = _seg()
        assert s.duration_ms == 1000
        assert s.duration_seconds == 1.0

    def test_duration_recomputed(self):
        s = _seg(duration_ms=999)
        assert s.duration_ms == 1000

    def test_end_must_be_greater_than_start(self):
        with pytest.raises(ValidationError):
            _seg(start_ms=1000, end_ms=1000)

    def test_start_ge_zero(self):
        with pytest.raises(ValidationError):
            _seg(start_ms=-1, end_ms=100)

    def test_segment_index_ge_zero(self):
        with pytest.raises(ValidationError):
            _seg(segment_index=-1)

    def test_segment_index_rejects_bool(self):
        with pytest.raises(ValidationError):
            _seg(segment_index=True)  # type: ignore[arg-type]

    def test_speaker_label_valid(self):
        assert _seg(speaker_label="SPEAKER_07").speaker_label == "SPEAKER_07"

    @pytest.mark.parametrize(
        "bad", ["speaker_00", "SPEAKER_0", "SPEAKER_100", "SPK_00", "SPEAKER_UNKNOWN"]
    )
    def test_speaker_label_invalid(self, bad: str):
        with pytest.raises(ValidationError):
            _seg(speaker_label=bad)

    def test_alignment_confidence_range(self):
        with pytest.raises(ValidationError):
            _seg(alignment_confidence=1.5)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_confidence_rejects_nan_inf(self, bad: float):
        with pytest.raises(ValidationError):
            _seg(alignment_confidence=bad)

    def test_confidence_rejects_bool(self):
        with pytest.raises(ValidationError):
            _seg(alignment_confidence=True)  # type: ignore[arg-type]

    def test_frozen(self):
        s = _seg()
        with pytest.raises(ValidationError):
            s.start_ms = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _seg(x=1)  # type: ignore[call-arg]


# =============================================================================
# AlignmentThresholds
# =============================================================================


class TestAlignmentThresholds:
    def test_defaults(self):
        th = AlignmentThresholds()
        assert th.min_overlap_ms == 1
        assert th.allow_silent_fallback is True
        assert th.strict_validity is True
        assert th.max_segments_safety == 1_000_000
        assert th.max_speakers_safety == 100

    def test_frozen(self):
        th = AlignmentThresholds()
        with pytest.raises(ValidationError):
            th.min_overlap_ms = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            AlignmentThresholds(unknown=1)  # type: ignore[call-arg]

    def test_bool_rejected(self):
        with pytest.raises(ValidationError):
            AlignmentThresholds(min_overlap_ms=True)  # type: ignore[arg-type]


# =============================================================================
# AlignmentResult
# =============================================================================


def _result(**kw):
    base = dict(
        strategy=ALIGNMENT_STRATEGY_WOMV,
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA,
        diarization_reference_sha256=_SHA,
        total_duration_ms=10000,
        transcript_language_code="es",
        dialogue_segments=(),
        num_segments=0,
        num_segments_aligned=0,
        num_unassigned=0,
    )
    base.update(kw)
    return AlignmentResult(**base)


class TestAlignmentResult:
    def test_valid_empty(self):
        r = _result()
        assert r.strategy == ALIGNMENT_STRATEGY_WOMV
        assert r.model_label == "t09:asr-speaker-alignment:v1"

    def test_valid_with_segments(self):
        r = _result(
            dialogue_segments=(_seg(segment_index=0, start_ms=0, end_ms=1000),),
            num_segments=1,
            num_segments_aligned=1,
            num_unassigned=0,
        )
        assert r.num_segments_aligned == 1

    def test_sha_chain_mismatch_raises(self):
        with pytest.raises(ValidationError):
            _result(vad_reference_sha256=_SHA_B)

    def test_sha_invalid_format(self):
        with pytest.raises(ValidationError):
            _result(source_preprocessed_sha256="bad")

    def test_num_consistency_aligned(self):
        with pytest.raises(ValidationError):
            _result(
                dialogue_segments=(_seg(),),
                num_segments=1,
                num_segments_aligned=0,  # incorrecto
                num_unassigned=0,
            )

    def test_num_consistency_total(self):
        with pytest.raises(ValidationError):
            _result(
                dialogue_segments=(),
                num_segments=5,
                num_segments_aligned=0,
                num_unassigned=0,  # 0 + 0 != 5
            )

    def test_segment_index_contiguous(self):
        with pytest.raises(ValidationError):
            _result(
                dialogue_segments=(_seg(segment_index=3, start_ms=0, end_ms=1000),),
                num_segments=1,
                num_segments_aligned=1,
                num_unassigned=0,
            )

    def test_segments_asc_no_overlap(self):
        with pytest.raises(ValidationError):
            _result(
                dialogue_segments=(
                    _seg(segment_index=0, start_ms=500, end_ms=1000),
                    _seg(segment_index=1, start_ms=0, end_ms=600),
                ),
                num_segments=2,
                num_segments_aligned=2,
                num_unassigned=0,
            )

    def test_segment_end_le_total(self):
        with pytest.raises(ValidationError):
            _result(
                dialogue_segments=(_seg(segment_index=0, start_ms=0, end_ms=20000),),
                num_segments=1,
                num_segments_aligned=1,
                num_unassigned=0,
                total_duration_ms=10000,
            )

    def test_silent_fallback_valid(self):
        r = _result(
            strategy=ALIGNMENT_STRATEGY_SILENT_FALLBACK,
            dialogue_segments=(),
            num_segments=5,
            num_segments_aligned=0,
            num_unassigned=5,
        )
        assert r.strategy == ALIGNMENT_STRATEGY_SILENT_FALLBACK
        assert r.num_unassigned == 5

    def test_silent_fallback_with_segments_raises(self):
        with pytest.raises(ValidationError):
            _result(
                strategy=ALIGNMENT_STRATEGY_SILENT_FALLBACK,
                dialogue_segments=(_seg(),),
                num_segments=1,
                num_segments_aligned=1,
                num_unassigned=0,
            )

    def test_frozen(self):
        r = _result()
        with pytest.raises(ValidationError):
            r.num_segments = 1  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _result(extra=1)  # type: ignore[call-arg]

    def test_determinism(self):
        a = _result(dialogue_segments=(_seg(),), num_segments=1, num_segments_aligned=1)
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()
