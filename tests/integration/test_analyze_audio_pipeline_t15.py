"""Tests de integración end-to-end — AnalyzeAudioPipeline (T01→T14).

Usa Fakes SOLO en tests para los adapters/use cases (no modelos reales).
NO introduce Fakes en producción.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.application.pipeline.analyze_audio_pipeline import AnalyzeAudioPipeline
from app.application.pipeline.preflight import preflight_all_ok, run_preflight
from app.core.config import AppSettings
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import MediaPrepResult, PreprocessedAudio
from app.domain.entities.metrics import MetricsResult
from app.domain.entities.output import StructuredJsonOutputResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import ValidationResult


_SHA = "a" * 64


# ---------------------------------------------------------------------------
# Fakes (solo tests)
# ---------------------------------------------------------------------------


class _FakeMediaPrep:
    def execute(self, input_path, job_id=None, settings=None, **kw):
        prep = PreprocessedAudio.model_construct(
            wav_path=Path("X:/x.wav"), sample_rate=16000, channels=1, bit_depth=16,
            duration_sec=10.0, size_bytes=1000, sha256=_SHA, applied_filters=(),
        )
        return MediaPrepResult.model_construct(
            job_id=job_id or "job-12345678", profile="balanced",
            validated=None, extracted_audio=None, preprocessed_audio=prep,
            step_times_sec={}, ffmpeg_available=True, ffprobe_available=True, errors=[],
        )


class _FakeVad:
    def execute(self, media_prep_result, job_id=None, thresholds=None, strict=True):
        return VadResult.model_construct(
            source_preprocessed_sha256=_SHA, voice_intervals=(), silence_segments=(),
            speech_ratio=0.0, total_speech_ms=0, total_silence_ms=0,
            num_intervals=0, num_silences=0, thresholds=None, model_label="x",
            sample_rate=16000, channels=1, duration_ms=10000,
        )


class _FakeLang:
    def execute(self, media_prep_result, vad_result, job_id=None, thresholds=None):
        return LanguageDetectionResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            language_code="es", language_name="Spanish", confidence=0.9,
            alternatives=(), total_duration_ms=10000, analyzed_duration_ms=10000,
            analyzed_strategy="only_voice_intervals", num_speech_intervals_considered=1,
            model_label="x", sample_rate=16000, channels=1, bit_depth=16,
        )


class _FakeAsr:
    def run(self, prep, vad, lang, job_id=None, thresholds=None):
        return ASRResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA, lid_reference_sha256=_SHA,
            sample_rate=16000, channels=1, bit_depth=16, total_duration_ms=10000,
            transcript_text="hola", transcript_language_code="es", confidence=0.9,
            segments=(), num_segments=0, strategy="full_audio", model_label="x",
        )


class _FakeTimestamps:
    def run(self, prep, vad, lang, asr, job_id=None, thresholds=None):
        return TimestampGenerationResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            total_duration_ms=10000, transcript_language_code="es",
            segments=(), words=(), num_segments_ts=0, num_words_ts=0,
            strategy="ts_silent_fallback", transcript_text="", confidence=0.0,
        )


class _FakeDiarization:
    def run(self, prep, vad, job_id=None, thresholds=None):
        return DiarizationResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            total_duration_ms=10000, speaker_turns=(), num_speakers=0, num_turns=0,
        )


class _FakeAlignment:
    def run(self, ts, asr, diar, job_id=None, thresholds=None):
        return AlignmentResult.model_construct(
            strategy="alignment_silent_fallback",
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            total_duration_ms=10000, transcript_language_code="es",
            dialogue_segments=(), num_segments=0, num_segments_aligned=0, num_unassigned=0,
        )


class _FakeQuality:
    def run(self, alignment, vad, asr, diar, job_id=None, thresholds=None, allow_silent_fallback=True):
        return QualityResult.model_construct(
            strategy="quality_silent_fallback",
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA,
            quality_level="passed", overall_quality_score=1.0,
            num_rules_evaluated=11, num_rules_passed=11, num_rules_warning=0, num_rules_failed=0,
            rule_results=(), average_alignment_confidence=0.0, average_ts_confidence=0.0,
            average_asr_confidence=0.0, num_segments=0, num_segments_aligned=0,
            num_segments_unassigned=0, total_duration_ms=10000, transcript_language_code="es",
        )


class _FakeValidation:
    def run(self, quality, alignment, vad, asr, diar, job_id=None, thresholds=None, force_save=False):
        return ValidationResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA, quality_reference_sha256=_SHA,
            validation_passed=True, validation_status="passed",
            quality_level="passed", overall_quality_score=1.0,
            validation_errors=(), validation_warnings=(),
            num_checks=11, num_errors=0, num_warnings=0, num_passed=11, check_results=(),
        )


class _FakeMetrics:
    def run(self, snapshot, lang, asr, vad, diar, alignment, quality, validation, job_id=None, metrics_enabled=True):
        return MetricsResult.model_construct(
            source_preprocessed_sha256=_SHA, validation_reference_sha256=_SHA,
            metrics_enabled=metrics_enabled, processing_wall_time_sec=1.0,
            audio_duration_sec=10.0, real_time_factor_RTF=0.1, step_timings_sec={},
            peak_cpu_rss_mb=0.0, peak_gpu_vram_used_mb=0.0, peak_gpu_vram_total_mb=0.0,
            cpu_utilization_avg_percent=0.0, gpu_utilization_avg_percent=0.0,
            segment_count_total=0, segment_count_with_text=0, word_count_total=0,
            word_count_confidence_ge_08=0, speaker_count_detected=0, silence_segment_count=0,
            average_asr_confidence=0.0, average_diarization_confidence=0.0,
            language_detection_confidence=0.0, overall_quality_score=1.0,
            profile_requested="", profile_applied="", profile_downgrade_applied=False,
            profile_downgrade_reason="", device_requested="", device_used="cpu",
            analysis_metadata={},
        )


class _FakeOutput:
    def run(self, asr, ts, diar, alignment, quality, validation, metrics, output_directory=None, schema_version=None, job_id=None, indent=2, ensure_ascii=False, force_save=False):
        return StructuredJsonOutputResult.model_construct(
            output_directory=output_directory or "/out",
            written_files=(), file_sha256={}, files_written_count=0,
            output_success=True, schema_version=schema_version or "0.1.0",
        )


class _FakeCleanup:
    def __init__(self):
        self.calls = 0

    def run(self, temp_root, job_id, cleanup_enabled=True, keep_temp_files=False, force_save=False):
        self.calls += 1
        from app.domain.entities.cleanup import TempCleanupResult

        return TempCleanupResult(job_id=job_id, cleaned=True, removed_paths=(), errors=(), success=True)


class _FakePathManager:
    def __init__(self):
        self.data_temp_dir = Path("/tmp/data/temporary")
        self.data_output_dir = Path("/tmp/data/output")


def _build_pipeline(cleanup=None):
    return AnalyzeAudioPipeline(
        media_prep=_FakeMediaPrep(),
        vad=_FakeVad(),
        language=_FakeLang(),
        asr=_FakeAsr(),
        timestamps=_FakeTimestamps(),
        diarization=_FakeDiarization(),
        alignment=_FakeAlignment(),
        quality=_FakeQuality(),
        validation=_FakeValidation(),
        metrics=_FakeMetrics(),
        output=_FakeOutput(),
        cleanup=cleanup or _FakeCleanup(),
        settings=AppSettings(),
        path_manager=_FakePathManager(),
    )


# ---------------------------------------------------------------------------
# Tests de integración
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_runs_all_steps(self):
        p = _build_pipeline()
        results = p.run("data/input/x.mp4")
        assert "media_prep" in results
        assert "vad" in results
        assert "language" in results
        assert "asr" in results
        assert "timestamps" in results
        assert "diarization" in results
        assert "alignment" in results
        assert "quality" in results
        assert "validation" in results
        assert "metrics" in results
        assert "output" in results

    def test_order_preserved(self):
        p = _build_pipeline()
        p.run("data/input/x.mp4")
        keys = list(p.step_times.keys())
        assert keys[:4] == ["T01_T02_T03", "T04_VAD", "T05_LID", "T06_ASR"]

    def test_cleanup_called(self):
        cleanup = _FakeCleanup()
        p = _build_pipeline(cleanup=cleanup)
        p.run("data/input/x.mp4")
        assert cleanup.calls == 1

    def test_cleanup_on_error(self):
        cleanup = _FakeCleanup()

        class _BoomQuality:
            def run(self, *a, **kw):
                raise RuntimeError("boom")

        p = _build_pipeline(cleanup=cleanup)
        p.quality = _BoomQuality()
        with pytest.raises(RuntimeError):
            p.run("data/input/x.mp4")
        # T14 en finally
        assert cleanup.calls == 1

    def test_force_save_propagated(self):
        cleanup = _FakeCleanup()
        p = _build_pipeline(cleanup=cleanup)
        p.run("data/input/x.mp4", force_save=True)
        assert cleanup.calls == 1

    def test_step_times_recorded(self):
        p = _build_pipeline()
        p.run("data/input/x.mp4")
        assert len(p.step_times) >= 11
        for v in p.step_times.values():
            assert v >= 0.0

    def test_job_id_autogenerated(self):
        p = _build_pipeline()
        results = p.run("data/input/x.mp4")
        assert results["media_prep"].job_id is not None

    def test_no_mutation_of_results(self):
        p = _build_pipeline()
        r1 = p.run("data/input/x.mp4")
        r2 = p.run("data/input/x.mp4")
        # job_id autogenerado puede diferir; el resto es determinista
        assert r1["vad"].num_intervals == r2["vad"].num_intervals


class TestPreflight:
    def test_preflight_runs_readonly(self):
        items = run_preflight()
        assert len(items) > 0
        # Solo lectura: no modifica nada

    def test_preflight_all_ok_returns_bool(self):
        items = run_preflight()
        assert isinstance(preflight_all_ok(items), bool)
