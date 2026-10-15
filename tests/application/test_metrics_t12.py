"""Tests unitarios T12 Application RunMetricsAggregationUseCase.

No requiere IA. Entidades sintéticas + RuntimeMetricsSnapshot. Cubre todas las
métricas de METRICS.md, disabled, runtime incompleto, SHA, determinismo y AST.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.constants import QualityLevel
from app.core.exceptions import ValidationFailedError
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.metrics import MetricsResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.alignment import DialogueSegment
from app.domain.value_objects.diarization import SpeakerTurn
from app.domain.value_objects.metrics import RuntimeMetricsSnapshot
from app.domain.value_objects.vad import SilenceSegment, VadThresholds, VoiceInterval
from app.application.use_cases.aggregate_metrics import RunMetricsAggregationUseCase


_SHA = "a" * 64


def _snapshot(**kw):
    base = dict(
        processing_wall_time_sec=10.0,
        peak_cpu_rss_mb=500.0,
        peak_gpu_vram_used_mb=2000.0,
        peak_gpu_vram_total_mb=8000.0,
        cpu_utilization_avg_percent=40.0,
        gpu_utilization_avg_percent=60.0,
        profile_requested="balanced",
        profile_applied="balanced",
        profile_downgrade_applied=False,
        profile_downgrade_reason="",
        device_requested="auto",
        device_used="cpu",
    )
    base.update(kw)
    return RuntimeMetricsSnapshot(**base)


def _seg(idx=0, start=0, end=1000, text="hola mundo") -> DialogueSegment:
    return DialogueSegment(
        segment_index=idx, start_ms=start, end_ms=end, duration_ms=end - start,
        text=text, speaker_label="SPEAKER_00", alignment_confidence=0.9, ts_confidence=0.9,
    )


def _alignment(segments=None, total_ms=10000):
    segs = tuple(segments or [])
    return AlignmentResult(
        strategy="weighted_overlap_majority_vote",
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
        total_duration_ms=total_ms, transcript_language_code="es",
        dialogue_segments=segs, num_segments=len(segs), num_segments_aligned=len(segs), num_unassigned=0,
    )


def _lid(confidence=0.8):
    return LanguageDetectionResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        language_code="es", language_name="Spanish", confidence=confidence,
        alternatives=(), total_duration_ms=10000, analyzed_duration_ms=10000,
        analyzed_strategy="only_voice_intervals", num_speech_intervals_considered=1,
        model_label="whisper-encoder-small-lid:v1", sample_rate=16000, channels=1, bit_depth=16,
    )


def _asr(confidence=0.85, texts=None):
    texts = texts or ["hola mundo"]
    segs = tuple(ASRSegment.model_construct(segment_index=i, text=t, start_ms=None, end_ms=None, avg_logprob=None) for i, t in enumerate(texts))
    return ASRResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA, lid_reference_sha256=_SHA,
        sample_rate=16000, channels=1, bit_depth=16, total_duration_ms=10000,
        transcript_text=" ".join(texts), transcript_language_code="es", confidence=confidence,
        segments=segs, num_segments=len(segs), strategy="full_audio", model_label="x",
    )


def _vad(silences=None):
    sil = tuple(SilenceSegment(start_ms=s, end_ms=e, speech_index_before=-1, speech_index_after=-1) for s, e in (silences or []))
    return VadResult.model_construct(
        source_preprocessed_sha256=_SHA, voice_intervals=(), silence_segments=sil,
        speech_ratio=0.0, total_speech_ms=0, total_silence_ms=0,
        num_intervals=0, num_silences=len(sil), thresholds=VadThresholds(),
        model_label="silero-vad:v5.1", sample_rate=16000, channels=1, duration_ms=10000,
    )


def _diarization(num_speakers=1, turns=None):
    if turns is None:
        turns = [SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=0.7)]
    t = tuple(turns)
    return DiarizationResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA, total_duration_ms=10000,
        speaker_turns=t, num_speakers=num_speakers, num_turns=len(t),
    )


def _quality(score=0.9):
    return QualityResult.model_construct(
        strategy="quality_rules_qr01_qr11",
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA,
        quality_level=QualityLevel.PASSED, overall_quality_score=score,
        num_rules_evaluated=11, num_rules_passed=11, num_rules_warning=0, num_rules_failed=0,
        rule_results=(), total_duration_ms=10000, transcript_language_code="es",
    )


def _validation():
    return ValidationResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA, quality_reference_sha256=_SHA,
        validation_passed=True, validation_status="passed",
        quality_level=QualityLevel.PASSED, overall_quality_score=0.9,
        validation_errors=(), validation_warnings=(),
        num_checks=11, num_errors=0, num_warnings=0, num_passed=11,
        check_results=(),
    )


def _run(snapshot, lid, asr, vad, diar, align, quality, validation, metrics_enabled=True):
    return RunMetricsAggregationUseCase().run(
        snapshot, lid, asr, vad, diar, align, quality, validation,
        metrics_enabled=metrics_enabled,
    )


# =============================================================================
# Métricas individuales
# =============================================================================


class TestMetrics:
    def test_rtf(self):
        r = _run(_snapshot(processing_wall_time_sec=5.0), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.real_time_factor_RTF == 0.5
        assert r.audio_duration_sec == 10.0

    def test_audio_duration_zero_rtf_none(self):
        r = _run(_snapshot(processing_wall_time_sec=5.0), _lid(), _asr(), _vad(), _diarization(), _alignment([], total_ms=0), _quality(), _validation())
        assert r.real_time_factor_RTF is None

    def test_segment_counts(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.segment_count_total == 1
        assert r.segment_count_with_text == 1

    def test_segment_count_with_empty_text(self):
        a = _alignment([_seg(text="   ")])
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), a, _quality(), _validation())
        assert r.segment_count_with_text == 0

    def test_word_count(self):
        r = _run(_snapshot(), _lid(), _asr(texts=["uno dos tres"]), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.word_count_total == 3

    def test_word_confidence_unavailable(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.word_count_confidence_ge_08 == 0
        assert r.analysis_metadata.get("word_confidence_unavailable") is True

    def test_speaker_count(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(num_speakers=2), _alignment([_seg()]), _quality(), _validation())
        assert r.speaker_count_detected == 2

    def test_silence_count(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(silences=[(0, 1000), (2000, 3000)]), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.silence_segment_count == 2

    def test_average_asr_confidence(self):
        r = _run(_snapshot(), _lid(), _asr(confidence=0.77), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.average_asr_confidence == 0.77

    def test_average_diarization_confidence(self):
        turns = [
            SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=0.5),
            SpeakerTurn(speaker_label="SPEAKER_01", start_ms=1000, end_ms=2000, confidence=0.9),
        ]
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(num_speakers=2, turns=turns), _alignment([_seg()]), _quality(), _validation())
        assert r.average_diarization_confidence == 0.7

    def test_language_detection_confidence(self):
        r = _run(_snapshot(), _lid(confidence=0.6), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.language_detection_confidence == 0.6

    def test_overall_quality_score(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(score=0.95), _validation())
        assert r.overall_quality_score == 0.95

    def test_compliance_passthrough(self):
        r = _run(_snapshot(profile_requested="quality", profile_applied="balanced", profile_downgrade_applied=True, profile_downgrade_reason="vram", device_requested="cuda", device_used="cpu"), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.profile_requested == "quality"
        assert r.profile_applied == "balanced"
        assert r.profile_downgrade_applied is True
        assert r.device_used == "cpu"

    def test_memory_utilization_passthrough(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.peak_cpu_rss_mb == 500.0
        assert r.peak_gpu_vram_used_mb == 2000.0
        assert r.cpu_utilization_avg_percent == 40.0

    def test_step_timings_passthrough(self):
        r = _run(_snapshot(step_timings_sec={"MEDIA_VALIDATION": 1.5}), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.step_timings_sec["MEDIA_VALIDATION"] == 1.5


class TestDisabled:
    def test_metrics_disabled(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation(), metrics_enabled=False)
        assert r.metrics_enabled is False
        assert r.analysis_metadata.get("disabled") is True
        assert r.processing_wall_time_sec == 0.0


class TestRuntimeMissing:
    def test_wall_time_zero_raises(self):
        with pytest.raises(ValidationFailedError):
            _run(_snapshot(processing_wall_time_sec=0.0), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())


class TestSHA:
    def test_sha_chain_ok(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.source_preprocessed_sha256 == _SHA
        assert r.validation_reference_sha256 == _SHA

    def test_sha_mismatch_raises(self):
        bad_validation = ValidationResult.model_construct(
            source_preprocessed_sha256="b" * 64, vad_reference_sha256="b" * 64,
            lid_reference_sha256="b" * 64, asr_reference_sha256="b" * 64,
            ts_reference_sha256="b" * 64, diarization_reference_sha256="b" * 64,
            alignment_reference_sha256="b" * 64, quality_reference_sha256="b" * 64,
            validation_passed=True, validation_status="passed",
            quality_level=QualityLevel.PASSED, overall_quality_score=0.9,
            validation_errors=(), validation_warnings=(),
            num_checks=11, num_errors=0, num_warnings=0, num_passed=11, check_results=(),
        )
        with pytest.raises(ValidationFailedError):
            _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), bad_validation)


class TestNoMutation:
    def test_inputs_not_mutated(self):
        sn = _snapshot(); lid = _lid(); asr = _asr(); vad = _vad(); diar = _diarization(); align = _alignment([_seg()]); q = _quality(); v = _validation()
        d = [x.model_dump() for x in (sn, lid, asr, vad, diar, align, q, v)]
        _run(sn, lid, asr, vad, diar, align, q, v)
        assert [x.model_dump() for x in (sn, lid, asr, vad, diar, align, q, v)] == d


class TestDeterminism:
    def test_same_input_same_output(self):
        a = _alignment([_seg(0, 0, 1000), _seg(1, 1000, 2000, text="dos palabras")])
        r1 = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), a, _quality(), _validation())
        r2 = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), a, _quality(), _validation())
        assert r1.model_dump() == r2.model_dump()


class TestInputValidation:
    def test_none_snapshot_raises(self):
        with pytest.raises(ValidationFailedError):
            RunMetricsAggregationUseCase().run(None, _lid(), _asr(), _vad(), _diarization(), _alignment(), _quality(), _validation())  # type: ignore[arg-type]

    def test_none_alignment_raises(self):
        with pytest.raises(ValidationFailedError):
            RunMetricsAggregationUseCase().run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), None, _quality(), _validation())  # type: ignore[arg-type]


class TestASTAudit:
    def test_no_forbidden_imports(self):
        paths = [
            "app/domain/value_objects/metrics.py",
            "app/domain/entities/metrics.py",
            "app/domain/interfaces/metrics_ports.py",
            "app/application/use_cases/aggregate_metrics.py",
        ]
        forbidden = {"psutil", "torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess"}
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
            assert not (imports & forbidden), f"{p} importa prohibido: {imports & forbidden}"


# =============================================================================
# Cobertura adicional: casos límite
# =============================================================================


class TestAdditional:
    def test_rtf_gt_1(self):
        r = _run(_snapshot(processing_wall_time_sec=20.0), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.real_time_factor_RTF == 2.0

    def test_multiple_segments_count(self):
        a = _alignment([_seg(0, 0, 1000, text="a"), _seg(1, 1000, 2000, text="b"), _seg(2, 2000, 3000, text="c")])
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), a, _quality(), _validation())
        assert r.segment_count_total == 3
        assert r.segment_count_with_text == 3

    def test_word_count_multiple_segments(self):
        r = _run(_snapshot(), _lid(), _asr(texts=["uno dos", "tres"]), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.word_count_total == 3

    def test_no_speakers(self):
        d = _diarization(num_speakers=0, turns=[])
        r = _run(_snapshot(), _lid(), _asr(), _vad(), d, _alignment([], total_ms=0), _quality(), _validation())
        assert r.speaker_count_detected == 0
        assert r.average_diarization_confidence == 0.0

    def test_step_timings_full_14(self):
        timings = {name: 0.1 for name in [
            "MEDIA_VALIDATION", "AUDIO_EXTRACTION", "AUDIO_PREPROCESSING",
            "VOICE_ACTIVITY_DETECTION", "LANGUAGE_DETECTION", "SPEECH_TO_TEXT",
            "TIMESTAMPS_GENERATION", "SPEAKER_DIARIZATION", "ASR_SPEAKER_ALIGNMENT",
            "QUALITY_ANALYSIS", "VALIDATION", "METRICS_AGGREGATION",
            "STRUCTURED_JSON_OUTPUT", "TEMP_CLEANUP",
        ]}
        r = _run(_snapshot(step_timings_sec=timings), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert len(r.step_timings_sec) == 14

    def test_unknown_step_timing_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(step_timings_sec={"NOT_A_STEP": 1.0})

    def test_snapshot_negative_memory_raises(self):
        with pytest.raises(ValidationError):
            _snapshot(peak_cpu_rss_mb=-1.0)

    def test_snapshot_inf_rejected(self):
        with pytest.raises(ValidationError):
            _snapshot(processing_wall_time_sec=float("inf"))

    def test_percent_bool_rejected(self):
        with pytest.raises(ValidationError):
            _snapshot(cpu_utilization_avg_percent=True)  # type: ignore[arg-type]

    def test_profile_downgrade_reason_passthrough(self):
        r = _run(_snapshot(profile_downgrade_applied=True, profile_downgrade_reason="vram_insuficiente"), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.profile_downgrade_applied is True
        assert r.profile_downgrade_reason == "vram_insuficiente"

    def test_disabled_no_runtime_required(self):
        # con disabled, no se exige wall time > 0
        r = _run(_snapshot(processing_wall_time_sec=0.0), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation(), metrics_enabled=False)
        assert r.metrics_enabled is False

    def test_job_id_passthrough(self):
        r = RunMetricsAggregationUseCase().run(
            _snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation(),
            job_id="job-12345678",
        )
        assert r.job_id == "job-12345678"

    def test_metrics_enabled_true_full(self):
        r = _run(_snapshot(), _lid(), _asr(), _vad(), _diarization(), _alignment([_seg()]), _quality(), _validation())
        assert r.metrics_enabled is True
        assert isinstance(r, MetricsResult)


class TestRuntimeSnapshotContract:
    def test_snapshot_valid_full(self):
        s = _snapshot(step_timings_sec={"METRICS_AGGREGATION": 0.5})
        assert s.step_timings_sec["METRICS_AGGREGATION"] == 0.5

    def test_snapshot_step_timings_bool_rejected(self):
        with pytest.raises(ValidationError):
            _snapshot(step_timings_sec={"METRICS_AGGREGATION": True})

    def test_snapshot_step_timings_inf_rejected(self):
        with pytest.raises(ValidationError):
            _snapshot(step_timings_sec={"METRICS_AGGREGATION": float("inf")})

    def test_snapshot_gpu_percent_range(self):
        with pytest.raises(ValidationError):
            _snapshot(gpu_utilization_avg_percent=-1.0)

    def test_snapshot_frozen(self):
        s = _snapshot()
        with pytest.raises(ValidationError):
            s.profile_requested = "quality"  # type: ignore[misc]

    def test_snapshot_extra_forbid(self):
        with pytest.raises(ValidationError):
            RuntimeMetricsSnapshot(processing_wall_time_sec=1.0, extra=1)  # type: ignore[call-arg]

    def test_snapshot_defaults(self):
        s = _snapshot()
        assert s.peak_cpu_rss_mb == 500.0
        assert s.profile_downgrade_applied is False

    def test_snapshot_determinism(self):
        a = _snapshot()
        b = _snapshot()
        assert a.model_dump() == b.model_dump()
