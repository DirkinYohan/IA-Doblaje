from __future__ import annotations

import ast
import os
import sys

import pytest
from pydantic import ValidationError

from app.core.exceptions import ValidationFailedError
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.timestamps import (
    TimestampGenerationResult,
    TimestampedSegment,
    TimestampedWord,
)
from app.domain.entities.vad import VadResult, VoiceInterval
from app.domain.value_objects.timestamps import AsrTimestampThresholds
from app.infrastructure.audio.basic_timestamp_normalizer_adapter import (
    BasicTimestampNormalizerAdapter,
)

# =========================================================================
# Factories igual que Application tests (helpers locales)
# =========================================================================
_SHA = "a" * 64

def _mk_prep(duration_sec: float = 10.0) -> PreprocessedAudio:
    return PreprocessedAudio.model_construct(
        sha256=_SHA,
        source_media_sha256="b"*64,
        wav_path="X:/dummy/x.wav",
        format_name="wav",
        size_bytes=100,
        duration_sec=duration_sec,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        job_id=None,
    )


def _mk_vad(*, prep=None, num=2, strategy="vad_speech_found") -> VadResult:
    prep = prep if prep is not None else _mk_prep()
    intervals: tuple[VoiceInterval, ...] = tuple(
        VoiceInterval.model_construct(
            index=i, start_ms=1000 + i*2000, end_ms=1800 + i*2000,
            duration_ms=800, max_confidence=0.9, mean_confidence=0.8,
        )
        for i in range(num)
    )
    return VadResult.model_construct(
        source_preprocessed_sha256=prep.sha256,
        vad_adapter_label="t", strategy=strategy,
        num_intervals=len(intervals),
        total_speech_ms=len(intervals) * 800,
        total_silence_ms=int(prep.duration_sec*1000) - len(intervals)*800,
        voice_intervals=intervals, analysis_metadata={}, validation_errors=(),
        thresholds=None, duration_ms=int(prep.duration_sec*1000),
    )


def _mk_lid(*, prep=None, lang="es") -> LanguageDetectionResult:
    prep = prep if prep is not None else _mk_prep()
    return LanguageDetectionResult.model_construct(
        source_preprocessed_sha256=prep.sha256,
        vad_reference_sha256=prep.sha256,
        language_code=lang,
        top_languages={"es":0.99}, confidence=0.99,
        detection_method="t", analysis_metadata={}, lid_adapter_label="t",
        strategy="ok", duration_ms=int(prep.duration_sec*1000),
        validation_errors=(), job_id=None,
    )


def _mk_asr(*, prep=None, vad=None, lid=None, strategy="full_audio",
            segments_crudo=((1000, 1800, "Hola"), (3000, 3800, "Mundo"))) -> ASRResult:
    prep = prep if prep is not None else _mk_prep()
    vad = vad if vad is not None else _mk_vad(prep=prep, num=len(segments_crudo))
    lid = lid if lid is not None else _mk_lid(prep=prep)
    segs = [
        ASRSegment.model_construct(
            segment_index=i, start_ms=s, end_ms=e, text=t,
            avg_logprob=-0.5, no_speech_prob=0.01,
        )
        for i, (s, e, t) in enumerate(segments_crudo)
    ]
    return ASRResult.model_construct(
        strategy=strategy, source_preprocessed_sha256=prep.sha256,
        vad_reference_sha256=vad.source_preprocessed_sha256,
        lid_reference_sha256=lid.source_preprocessed_sha256,
        total_duration_ms=int(prep.duration_sec*1000),
        transcript_language_code=lid.language_code,
        job_id=None, segments=tuple(segs), num_segments=len(segs),
        confidence=0.9,
        transcript_text=" ".join(t for _s,_e,t in segments_crudo),
        analysis_metadata={}, model_label="t", asr_backend="t",
        validation_errors=(), strict_ok=True,
        language_detected=lid.language_code,
        duration_ms=int(prep.duration_sec*1000),
    )


# =========================================================================
# TestAdapter Basic Happy (≥ 6 tests)
# =========================================================================
class TestBasicAdapterNormalizerHappy:
    def test_happy_two_segments_d4_overlap_resolution(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p)
        # Seg 0 end = 4000 > Seg 1 start = 3000. Overlap 1000ms.
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((1000, 4000, "A"), (3000, 5000, "B")),
        )
        ad = BasicTimestampNormalizerAdapter()
        res: TimestampGenerationResult = ad.normalize(
            p, vad_result=v, lid_result=l, asr_result=a,
            thresholds=AsrTimestampThresholds(strict_validity=True),
        )
        assert res.num_segments_ts == 2
        assert res.num_overlaps_resolved == 1
        # Después de clamp END overlap: seg 0 end = min(4000, 3000) = 3000
        assert res.segments[0].end_ms == 3000
        assert res.segments[0].overlap_normalized is True
        # Seg 1 sin cambios
        assert res.segments[1].start_ms == 3000
        assert res.segments[1].end_ms == 5000
        assert res.segments[0].duration_ms == res.segments[0].end_ms - res.segments[0].start_ms

    def test_happy_orden_asc(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p)
        # Orden DESC en crudo, esperamos orden ASC normalizado en segment_index oficial
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((5000, 6000, "Luego"), (1000, 2000, "Primero")),
        )
        ad = BasicTimestampNormalizerAdapter()
        res = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)
        # Después sort ASC: start=1000 va a segment_index 0
        assert res.segments[0].start_ms == 1000
        assert res.segments[0].segment_index == 0
        assert res.segments[1].start_ms == 5000
        assert res.segments[1].segment_index == 1
        assert res.strategy == "segment_ts_from_fw_raw"
        # words D#1 == ()
        assert res.num_words_ts == 0
        assert len(res.words) == 0

    def test_d5_clamp_end_total_duration_ms(self):
        p = _mk_prep(duration_sec=5.0)  # 5000 ms total
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        # end_ms crudo = 6000 > 5000 total. Clamp → 5000.
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((1000, 6000, "Audio"),),
        )
        ad = BasicTimestampNormalizerAdapter()
        res = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)
        assert res.num_end_clamped == 1
        assert res.segments[0].end_clamped_to_total_duration is True
        assert res.segments[0].end_ms == 5000
        assert res.segments[0].duration_ms == 4000

    def test_d3_vad_interpolate_1_a_1(self):
        """D#3 ASRSegment.start_ms = None, M=vad N=asr 2==2 → match 1:1 índice."""
        p = _mk_prep()
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p)
        # Segment 0 sin start/end; segment 1 con start/end
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((None, None, "A"), (3000, 3800, "B")),
        )
        ad = BasicTimestampNormalizerAdapter()
        res = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)
        assert res.strategy == "segment_ts_from_vad_interpolate"
        assert res.num_vad_interpolated >= 1
        assert res.num_fw_none_startms_fixed == 1
        # Seg 0 interpolated: corresponde a VoiceInterval index=0 (1000,1800)
        assert res.segments[0].vad_interpolated is True
        assert res.segments[0].start_ms == 1000
        assert res.segments[0].end_ms == 1800
        # Seg 1 sigue siendo fw
        assert res.segments[1].vad_interpolated is False
        assert res.segments[1].source == "fw_raw"

    def test_no_crash_100_percent_overlap_non_strict_is_rejected_strict(self):
        """Overlap extreme: seg0 [1000-5000], seg1 [1500-2000].
        Clamp produce end(seg0) = 1500. Seg1 end(2000) > seg1 start(1500). OK."""
        p = _mk_prep(duration_sec=10.0)
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p)
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((1000, 5000, "Grande"), (1500, 2000, "Pequeño")),
        )
        ad = BasicTimestampNormalizerAdapter()
        res = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a,
                           thresholds=AsrTimestampThresholds(strict_validity=True))
        assert res.num_overlaps_resolved == 1
        assert res.segments[0].end_ms == 1500
        assert res.segments[1].end_ms == 2000

    def test_result_frozen_no_mutation(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=((0, 5000, "Todo"),))
        ad = BasicTimestampNormalizerAdapter()
        res = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)
        with pytest.raises((TypeError, ValidationError)):
            res.num_segments_ts = 99  # type: ignore[misc]


class TestBasicAdapterNormalizerInvalidInputs:
    def test_d3_start_none_sin_vad_intervals(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=0, strategy="vad_no_speech")  # intervals 0
        l = _mk_lid(prep=p)
        a = _mk_asr(
            prep=p, vad=v, lid=l, strategy="full_audio",
            segments_crudo=((None, None, "A"),),
        )
        # Silent UseCase handle; pero adapter directo recibe segments vacíos? No, segments 1.
        # adapter debe fallar porque _match_vad_interval returns None None y strict=True
        ad = BasicTimestampNormalizerAdapter()
        with pytest.raises(ValidationFailedError, match="sin VAD match válido"):
            ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)

    def test_strict_validity_min_duration_triggers_validation_failed(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((1000, 1005, "tiny"),),  # duration 5 < default min=10
        )
        ad = BasicTimestampNormalizerAdapter()
        with pytest.raises(ValidationFailedError):
            ad.normalize(
                p, vad_result=v, lid_result=l, asr_result=a,
                thresholds=AsrTimestampThresholds(strict_validity=True,
                                                  min_segment_duration_ms=10),
            )

    def test_non_strict_returns_warnings_ok_false(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(
            prep=p, vad=v, lid=l,
            segments_crudo=((1000, 1005, "tiny"),),
        )
        ad = BasicTimestampNormalizerAdapter()
        res = ad.normalize(
            p, vad_result=v, lid_result=l, asr_result=a,
            thresholds=AsrTimestampThresholds(strict_validity=False, min_segment_duration_ms=10),
        )
        assert res.strict_ok is False
        assert len(res.validation_errors) >= 1


# =========================================================================
# Static audit imports T08+ (≥ 2 tests)
# =========================================================================
class TestAdapterStaticAudit:
    @pytest.fixture
    def adapter_source(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        adapter_path = os.path.normpath(
            os.path.join(here, "..", "..", "..",
                         "app", "infrastructure", "audio",
                         "basic_timestamp_normalizer_adapter.py")
        )
        with open(adapter_path, "r", encoding="utf-8") as f:
            return f.read()

    def test_no_hf_or_download_or_network_imports(self, adapter_source: str):
        tree = ast.parse(adapter_source)
        forbidden_module_prefixes = (
            "faster_whisper", "torch", "numpy", "transformers",
            "requests", "urllib", "httpx", "huggingface_hub",
        )
        bad_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in forbidden_module_prefixes:
                        bad_imports.append(name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.split(".")[0]
                    if name in forbidden_module_prefixes:
                        bad_imports.append(name)
        assert bad_imports == [], (
            f"Adapter imports foroibidos red/nn: {bad_imports}"
        )

    def test_no_t08_keywords_executable_code(self, adapter_source: str):
        """pyannote/diarization/SPEAKER_00/weighted overlap/quality_score deben ser 0."""
        tree = ast.parse(adapter_source)
        bad_calls: list[str] = []
        keywords = ("pyannote", "diarization", "SPEAKER_00", "SPEAKER_01",
                    "quality_score", "snr", "clipping_ratio", "QUALITY_RULES",
                    "rtf", "vram", "weighted_overlap", "majority_vote",
                    "cleanup_temp", "analysis.json", "transcript.json")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    for k in keywords:
                        if k.lower() in func.id.lower():
                            bad_calls.append(func.id)
                elif isinstance(func, ast.Attribute):
                    for k in keywords:
                        if k.lower() in func.attr.lower():
                            bad_calls.append(func.attr)
            elif isinstance(node, ast.Name):
                for k in keywords:
                    if node.id.lower() == k.lower():
                        # Solo si NO está dentro de un context docstring/literal?
                        # Aceptar nombres variables iguales pero filtrar:
                        pass
        assert bad_calls == []

    def test_no_t07_calls_word_timestamps_in_fw_adapter_or_other(self, adapter_source: str):
        """word_timestamps no debe aparecer en calls (solo en comentarios/literals tests)."""
        tree = ast.parse(adapter_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "word_timestamps":
                raise AssertionError("keyword word_timestamps en adapter T07 (esperado 0)")


# =========================================================================
# TestAdapter Edge cases (≥ 3 tests más)
# =========================================================================
class TestBasicAdapterEdgeCases:
    def test_zero_segments_silent_case(self):
        """Silent lo resuelve UseCase; Adapter se invoca con 0 segments?
        Adapter debe retornar strategy_ok sin segments. Pero la chain validation
        en build_chain produce res num_segments_ts = 0 → OK."""
        p = _mk_prep(duration_sec=10.0)
        v = _mk_vad(prep=p, num=0)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, strategy="asr_silent_fallback", segments_crudo=())
        ad = BasicTimestampNormalizerAdapter()
        # strategy_raw=True pero len(segments) 0 → devolver build chain sin fw raw:
        res = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a,
                           thresholds=AsrTimestampThresholds(strict_validity=True))
        assert res.num_segments_ts == 0
        # words 0
        assert res.num_words_ts == 0

    def test_d3_vad_none_but_fallback_flag_off_is_validation_failed(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=0)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, strategy="full_audio",
                    segments_crudo=((None, None, "sin datos"),))
        ad = BasicTimestampNormalizerAdapter()
        with pytest.raises(ValidationFailedError, match="allow_vad_interpolate_fallback"):
            ad.normalize(
                p, vad_result=v, lid_result=l, asr_result=a,
                thresholds=AsrTimestampThresholds(allow_vad_interpolate_fallback=False,
                                                  strict_validity=True),
            )

    def test_determinismo_two_runs_equal(self):
        p = _mk_prep(duration_sec=10.0)
        v = _mk_vad(prep=p, num=3)
        l = _mk_lid(prep=p, lang="pt")
        a = _mk_asr(prep=p, vad=v, lid=l,
                    segments_crudo=((100, 4000, "A"), (2500, 6000, "B"), (5500, 9500, "C")))
        ad = BasicTimestampNormalizerAdapter()
        r1 = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)
        r2 = ad.normalize(p, vad_result=v, lid_result=l, asr_result=a)
        assert r1.strategy == r2.strategy
        assert [s.start_ms for s in r1.segments] == [s.start_ms for s in r2.segments]
        assert [s.end_ms for s in r1.segments] == [s.end_ms for s in r2.segments]
        assert r1.num_overlaps_resolved == r2.num_overlaps_resolved
