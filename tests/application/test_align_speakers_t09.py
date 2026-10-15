"""Tests unitarios T09 Application RunSpeakerAlignmentUseCase.

No requiere IA. Entidades sintéticas T06/T07/T08. Cubre Weighted Overlap
Majority Vote, tie-break, silent fallback y no-speaker.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.exceptions import ValidationFailedError
from app.domain.entities.alignment import (
    ALIGNMENT_STRATEGY_SILENT_FALLBACK,
    ALIGNMENT_STRATEGY_WOMV,
    AlignmentResult,
)
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.timestamps import (
    TimestampGenerationResult,
    TimestampedSegment,
)
from app.domain.value_objects.alignment import AlignmentThresholds
from app.domain.value_objects.diarization import SpeakerTurn
from app.application.use_cases.align_speakers import RunSpeakerAlignmentUseCase


_SHA = "a" * 64


def _ts_seg(idx: int, start: int, end: int, ts_conf: float = 1.0) -> TimestampedSegment:
    return TimestampedSegment.model_construct(
        segment_index=idx,
        start_ms=start,
        end_ms=end,
        duration_ms=end - start,
        source="fw_raw",
        ts_confidence=ts_conf,
    )


def _ts_result(segments: list[TimestampedSegment], total_ms: int = 10000) -> TimestampGenerationResult:
    return TimestampGenerationResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        asr_reference_sha256=_SHA,
        total_duration_ms=total_ms,
        transcript_language_code="es",
        segments=tuple(segments),
        words=(),
        num_segments_ts=len(segments),
        num_words_ts=0,
        strategy="segment_ts_from_fw_raw",
        transcript_text="",
        confidence=1.0,
    )


def _asr_result(texts: list[str]) -> ASRResult:
    segs = tuple(
        ASRSegment.model_construct(segment_index=i, text=t, start_ms=None, end_ms=None, avg_logprob=None)
        for i, t in enumerate(texts)
    )
    return ASRResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        total_duration_ms=10000,
        transcript_text=" ".join(texts),
        transcript_language_code="es",
        confidence=0.9,
        segments=segs,
        num_segments=len(segs),
        strategy="full_audio",
        model_label="faster-whisper:small",
    )


def _turn(label: str, start: int, end: int, conf: float = 1.0) -> SpeakerTurn:
    return SpeakerTurn(speaker_label=label, start_ms=start, end_ms=end, confidence=conf)


def _diarization(turns: list[SpeakerTurn]) -> DiarizationResult:
    return DiarizationResult.model_construct(
        source_preprocessed_sha256=_SHA,
        vad_reference_sha256=_SHA,
        total_duration_ms=10000,
        speaker_turns=tuple(turns),
        num_speakers=len({t.speaker_label for t in turns}),
        num_turns=len(turns),
    )


def _align(ts, asr, diar, thresholds=None):
    return RunSpeakerAlignmentUseCase().run(ts, asr, diar, thresholds=thresholds)


# =============================================================================
# Algoritmo
# =============================================================================


class TestOverlap:
    def test_no_overlap(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 1000, 2000)])
        r = _align(ts, asr, diar)
        assert r.num_segments_aligned == 0
        assert r.num_unassigned == 1

    def test_partial_overlap(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 500, 1500)])
        r = _align(ts, asr, diar)
        assert r.num_segments_aligned == 1
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_00"

    def test_total_overlap(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1000)])
        r = _align(ts, asr, diar)
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_00"

    def test_speaker_contained(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 300, 700)])
        r = _align(ts, asr, diar)
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_00"


class TestWeightedVote:
    def test_weighted_score_beats_overlap(self):
        # A: overlap 500, conf 0.8 -> 400 ; B: overlap 600, conf 0.5 -> 300
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([
            _turn("SPEAKER_00", 0, 500, conf=0.8),
            _turn("SPEAKER_01", 400, 1000, conf=0.5),
        ])
        r = _align(ts, asr, diar)
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_00"

    def test_tie_break_by_confidence(self):
        # A: overlap 500, conf 0.6 ; B: overlap 500, conf 0.9 -> B
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([
            _turn("SPEAKER_00", 0, 500, conf=0.6),
            _turn("SPEAKER_01", 500, 1000, conf=0.9),
        ])
        r = _align(ts, asr, diar)
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_01"

    def test_tie_break_final_earliest_start(self):
        # mismo weighted_score y misma confidence -> menor start_ms gana
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([
            _turn("SPEAKER_01", 0, 500, conf=0.7),
            _turn("SPEAKER_00", 500, 1000, conf=0.7),
        ])
        r = _align(ts, asr, diar)
        # ambos weighted 350, conf 0.7 -> gana SPEAKER_01 (start 0 < 500)
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_01"


class TestSilentNoSpeaker:
    def test_empty_diarization_silent_fallback(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([])
        r = _align(ts, asr, diar)
        assert r.strategy == ALIGNMENT_STRATEGY_SILENT_FALLBACK
        assert r.dialogue_segments == ()
        assert r.num_unassigned == 1
        assert r.num_segments_aligned == 0

    def test_mixed_assigned_unassigned(self):
        ts = _ts_result([_ts_seg(0, 0, 1000), _ts_seg(1, 1000, 2000)])
        asr = _asr_result(["a", "b"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1000)])
        r = _align(ts, asr, diar)
        assert r.num_segments_aligned == 1
        assert r.num_unassigned == 1
        assert r.dialogue_segments[0].speaker_label == "SPEAKER_00"
        assert r.dialogue_segments[0].text == "a"


class TestTextAndTimestamps:
    def test_text_from_asr_not_ts(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["texto real del ASR"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1000)])
        r = _align(ts, asr, diar)
        assert r.dialogue_segments[0].text == "texto real del ASR"

    def test_timestamps_from_ts(self):
        ts = _ts_result([_ts_seg(0, 100, 900, ts_conf=0.8)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 100, 900)])
        r = _align(ts, asr, diar)
        seg = r.dialogue_segments[0]
        assert seg.start_ms == 100
        assert seg.end_ms == 900
        assert seg.duration_ms == 800
        assert seg.ts_confidence == 0.8

    def test_segment_index_preserved(self):
        # índices contiguos 0..5 (6 segmentos); el índice 5 se preserva y mapea texto 'f'
        segs = [_ts_seg(i, i * 1000, (i + 1) * 1000) for i in range(6)]
        ts = _ts_result(segs, total_ms=6000)
        asr = _asr_result(["a", "b", "c", "d", "e", "f"])
        turns = [_turn("SPEAKER_00", i * 1000, (i + 1) * 1000) for i in range(6)]
        diar = _diarization(turns)
        r = _align(ts, asr, diar)
        last = r.dialogue_segments[-1]
        assert last.segment_index == 5
        assert last.text == "f"

    def test_missing_asr_segment_raises(self):
        ts = _ts_result([_ts_seg(9, 0, 1000)])
        asr = _asr_result(["solo uno"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1000)])
        with pytest.raises(ValidationFailedError):
            _align(ts, asr, diar)


class TestChainSHA:
    def test_chain_sha_ok(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1000)])
        r = _align(ts, asr, diar)
        assert r.source_preprocessed_sha256 == _SHA
        assert r.ts_reference_sha256 == _SHA
        assert r.diarization_reference_sha256 == _SHA

    def test_chain_sha_mismatch_raises(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = DiarizationResult.model_construct(
            source_preprocessed_sha256="b" * 64,
            vad_reference_sha256="b" * 64,
            total_duration_ms=10000,
            speaker_turns=(_turn("SPEAKER_00", 0, 1000),),
            num_speakers=1,
            num_turns=1,
        )
        with pytest.raises(ValidationFailedError):
            _align(ts, asr, diar)


class TestNoMutation:
    def test_inputs_not_mutated(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        asr = _asr_result(["hola"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1000)])
        ts_d = ts.model_dump()
        asr_d = asr.model_dump()
        diar_d = diar.model_dump()
        _align(ts, asr, diar)
        assert ts.model_dump() == ts_d
        assert asr.model_dump() == asr_d
        assert diar.model_dump() == diar_d


class TestDeterminism:
    def test_same_input_same_output(self):
        ts = _ts_result([_ts_seg(0, 0, 1000), _ts_seg(1, 1000, 2000)])
        asr = _asr_result(["a", "b"])
        diar = _diarization([_turn("SPEAKER_00", 0, 1500), _turn("SPEAKER_01", 1500, 2000)])
        r1 = _align(ts, asr, diar)
        r2 = _align(ts, asr, diar)
        assert r1.model_dump() == r2.model_dump()


class TestInputValidation:
    def test_none_ts_raises(self):
        with pytest.raises(ValidationFailedError):
            RunSpeakerAlignmentUseCase().run(None, _asr_result(["a"]), _diarization([]))  # type: ignore[arg-type]

    def test_none_asr_raises(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        with pytest.raises(ValidationFailedError):
            RunSpeakerAlignmentUseCase().run(ts, None, _diarization([]))  # type: ignore[arg-type]

    def test_thresholds_wrong_type(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        with pytest.raises(ValidationFailedError):
            _align(ts, _asr_result(["a"]), _diarization([_turn("SPEAKER_00", 0, 1000)]), thresholds="x")  # type: ignore[arg-type]

    def test_allow_silent_fallback_false(self):
        ts = _ts_result([_ts_seg(0, 0, 1000)])
        with pytest.raises(ValidationFailedError):
            _align(ts, _asr_result(["a"]), _diarization([]), thresholds=AlignmentThresholds(allow_silent_fallback=False))


class TestASTAudit:
    def test_no_ai_imports_in_domain_app(self):
        paths = [
            "app/domain/value_objects/alignment.py",
            "app/domain/entities/alignment.py",
            "app/domain/interfaces/alignment_ports.py",
            "app/application/use_cases/align_speakers.py",
        ]
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx"}
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
            assert not (imports & forbidden), f"{p} importa IA: {imports & forbidden}"
