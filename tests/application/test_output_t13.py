"""Tests unitarios T13 Application RunStructuredJsonOutputUseCase.

No requiere filesystem (Fake writer). Cubre los 5 documentos, orden, words=[],
None/NaN/Inf, force_save, no mutación, determinismo y AST.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from app.core.constants import QualityLevel
from app.core.exceptions import ValidationFailedError
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.metrics import MetricsResult
from app.domain.entities.output import StructuredJsonOutputResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.timestamps import TimestampGenerationResult, TimestampedSegment
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.alignment import DialogueSegment
from app.domain.value_objects.diarization import SpeakerTurn
from app.application.use_cases.write_structured_json import RunStructuredJsonOutputUseCase


_SHA = "a" * 64


class _FakeWriter:
    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.order: list[str] = []

    def write_json(self, *, output_directory, filename, document, indent, ensure_ascii):
        # Serializa con allow_nan=False para replicar el contrato (rechaza NaN/Inf)
        raw = json.dumps(document, indent=indent, ensure_ascii=ensure_ascii, allow_nan=False, sort_keys=False)
        self.documents[filename] = json.loads(raw)
        self.order.append(filename)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _seg(idx=0, start=0, end=1000, text="hola", speaker="SPEAKER_00") -> DialogueSegment:
    return DialogueSegment(
        segment_index=idx, start_ms=start, end_ms=end, duration_ms=end - start,
        text=text, speaker_label=speaker, alignment_confidence=0.9, ts_confidence=0.9,
    )


def _alignment(segments=None, num_speakers=1):
    segs = tuple(segments or [_seg()])
    return AlignmentResult(
        strategy="weighted_overlap_majority_vote",
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
        total_duration_ms=10000, transcript_language_code="es",
        dialogue_segments=segs, num_segments=len(segs), num_segments_aligned=len(segs), num_unassigned=0,
    )


def _asr():
    segs = tuple(ASRSegment.model_construct(segment_index=i, text=f"texto {i}", start_ms=None, end_ms=None, avg_logprob=None) for i in range(1))
    return ASRResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA, lid_reference_sha256=_SHA,
        sample_rate=16000, channels=1, bit_depth=16, total_duration_ms=10000,
        transcript_text="texto 0", transcript_language_code="es", confidence=0.85,
        segments=segs, num_segments=1, strategy="full_audio", model_label="x",
    )


def _timestamps():
    segs = (TimestampedSegment.model_construct(segment_index=0, start_ms=0, end_ms=1000, duration_ms=1000, source="fw_raw", ts_confidence=0.9),)
    return TimestampGenerationResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
        total_duration_ms=10000, transcript_language_code="es",
        segments=segs, words=(), num_segments_ts=1, num_words_ts=0,
        strategy="segment_ts_from_fw_raw", transcript_text="", confidence=1.0,
    )


def _diarization():
    turns = (SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=0.7),)
    return DiarizationResult.model_construct(
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA, total_duration_ms=10000,
        speaker_turns=turns, num_speakers=1, num_turns=1,
    )


def _quality(level="passed", score=0.9):
    return QualityResult.model_construct(
        strategy="quality_rules_qr01_qr11",
        source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
        lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
        ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
        alignment_reference_sha256=_SHA,
        quality_level=QualityLevel(level), overall_quality_score=score,
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
        num_checks=11, num_errors=0, num_warnings=0, num_passed=11, check_results=(),
    )


def _metrics():
    return MetricsResult.model_construct(
        source_preprocessed_sha256=_SHA, validation_reference_sha256=_SHA,
        metrics_enabled=True,
        processing_wall_time_sec=5.0, audio_duration_sec=10.0, real_time_factor_RTF=0.5,
        step_timings_sec={"MEDIA_VALIDATION": 1.0},
        peak_cpu_rss_mb=500.0, peak_gpu_vram_used_mb=2000.0, peak_gpu_vram_total_mb=8000.0,
        cpu_utilization_avg_percent=40.0, gpu_utilization_avg_percent=60.0,
        segment_count_total=1, segment_count_with_text=1, word_count_total=2,
        word_count_confidence_ge_08=0, speaker_count_detected=1, silence_segment_count=0,
        average_asr_confidence=0.85, average_diarization_confidence=0.7,
        language_detection_confidence=0.8, overall_quality_score=0.9,
        profile_requested="balanced", profile_applied="balanced",
        profile_downgrade_applied=False, profile_downgrade_reason="",
        device_requested="auto", device_used="cpu",
        analysis_metadata={"word_confidence_unavailable": True},
    )


def _run(writer=None):
    w = writer or _FakeWriter()
    result = RunStructuredJsonOutputUseCase(writer=w).run(
        _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
        output_directory="/out", schema_version="0.1.0",
    )
    return w, result


# =============================================================================
# Documentos
# =============================================================================


class TestDocuments:
    def test_five_documents_written(self):
        w, r = _run()
        assert len(w.documents) == 5
        assert set(w.documents.keys()) == {"analysis.json", "transcript.json", "speakers.json", "segments.json", "metadata.json"}
        assert r.files_written_count == 5
        assert r.output_success is True

    def test_deterministic_order(self):
        w, _ = _run()
        assert w.order == ["analysis.json", "transcript.json", "speakers.json", "segments.json", "metadata.json"]

    def test_analysis_fields(self):
        w, _ = _run()
        a = w.documents["analysis.json"]
        assert a["validation_status"] == "passed"
        assert a["quality_level"] == "passed"
        assert a["num_speakers"] == 1
        assert a["total_duration_ms"] == 10000
        assert a["overall_quality_score"] == 0.9

    def test_transcript_words_empty(self):
        w, _ = _run()
        t = w.documents["transcript.json"]
        assert t["words"] == []
        assert t["transcript_text"] == "texto 0"
        assert t["segments"][0]["words"] == []
        assert t["segments"][0]["start_ms"] == 0
        assert t["segments"][0]["end_ms"] == 1000

    def test_speakers_fields(self):
        w, _ = _run()
        s = w.documents["speakers.json"]
        assert s["num_speakers"] == 1
        assert s["num_turns"] == 1
        assert s["speakers"][0]["speaker_label"] == "SPEAKER_00"
        assert s["speakers"][0]["confidence"] == 0.7

    def test_segments_fields(self):
        w, _ = _run()
        s = w.documents["segments.json"]
        seg = s["segments"][0]
        assert seg["segment_index"] == 0
        assert seg["speaker_label"] == "SPEAKER_00"
        assert seg["alignment_confidence"] == 0.9
        assert seg["ts_confidence"] == 0.9

    def test_metadata_fields(self):
        w, _ = _run()
        m = w.documents["metadata.json"]
        assert m["processing_wall_time_sec"] == 5.0
        assert m["real_time_factor_RTF"] == 0.5
        assert m["profile_applied"] == "balanced"
        assert m["average_asr_confidence"] == 0.85
        assert m["word_count_confidence_ge_08"] == 0

    def test_result_sha_map(self):
        w, r = _run()
        assert set(r.file_sha256.keys()) == {"analysis.json", "transcript.json", "speakers.json", "segments.json", "metadata.json"}
        for name, sha in r.file_sha256.items():
            assert len(sha) == 64


class TestSerialization:
    def test_allow_nan_false_rejects_nan(self):
        # Fake writer serializa con allow_nan=False; documento con NaN debe fallar.
        writer = _FakeWriter()
        with pytest.raises(ValueError):
            writer.write_json(output_directory="/out", filename="x.json", document={"v": float("nan")}, indent=2, ensure_ascii=False)

    def test_allow_nan_false_rejects_inf(self):
        writer = _FakeWriter()
        with pytest.raises(ValueError):
            writer.write_json(output_directory="/out", filename="x.json", document={"v": float("inf")}, indent=2, ensure_ascii=False)

    def test_none_serializes_as_null(self):
        writer = _FakeWriter()
        sha = writer.write_json(output_directory="/out", filename="x.json", document={"v": None}, indent=2, ensure_ascii=False)
        assert len(sha) == 64


class TestForceSave:
    def test_quality_failed_force_save_writes(self):
        writer = _FakeWriter()
        result = RunStructuredJsonOutputUseCase(writer=writer).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(level="failed", score=0.1), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0", force_save=True,
        )
        assert writer.documents["analysis.json"]["quality_level"] == "failed"
        assert result.analysis_metadata.get("force_save") is True


class TestNoMutation:
    def test_inputs_not_mutated(self):
        asr = _asr(); ts = _timestamps(); diar = _diarization(); align = _alignment(); q = _quality(); v = _validation(); m = _metrics()
        d = [x.model_dump() for x in (asr, ts, diar, align, q, v, m)]
        RunStructuredJsonOutputUseCase(writer=_FakeWriter()).run(asr, ts, diar, align, q, v, m, output_directory="/out", schema_version="0.1.0")
        assert [x.model_dump() for x in (asr, ts, diar, align, q, v, m)] == d


class TestDeterminism:
    def test_byte_identical_output(self):
        w1, r1 = _run(_FakeWriter())
        w2, r2 = _run(_FakeWriter())
        assert r1.file_sha256 == r2.file_sha256
        assert w1.documents == w2.documents


class TestInputValidation:
    def test_none_asr_raises(self):
        with pytest.raises(ValidationFailedError):
            RunStructuredJsonOutputUseCase(writer=_FakeWriter()).run(None, _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(), output_directory="/out", schema_version="0.1.0")  # type: ignore[arg-type]

    def test_none_metrics_raises(self):
        with pytest.raises(ValidationFailedError):
            RunStructuredJsonOutputUseCase(writer=_FakeWriter()).run(_asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), None, output_directory="/out", schema_version="0.1.0")  # type: ignore[arg-type]


class TestASTAudit:
    def test_application_no_filesystem_no_ai(self):
        p = Path("app/application/use_cases/write_structured_json.py")
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess", "pathlib"}
        assert not (imports & forbidden), f"importa prohibido: {imports & forbidden}"

    def test_domain_no_ai(self):
        for p in ["app/domain/value_objects/output.py", "app/domain/entities/output.py", "app/domain/interfaces/output_ports.py"]:
            tree = ast.parse(Path(p).read_text(encoding="utf-8"))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imports.add(a.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
            forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess"}
            assert not (imports & forbidden), f"{p} importa prohibido: {imports & forbidden}"


# =============================================================================
# Cobertura adicional
# =============================================================================


class TestAdditionalDocuments:
    def test_analysis_validation_errors_passthrough(self):
        v = _validation()
        v = ValidationResult.model_construct(
            source_preprocessed_sha256=_SHA, vad_reference_sha256=_SHA,
            lid_reference_sha256=_SHA, asr_reference_sha256=_SHA,
            ts_reference_sha256=_SHA, diarization_reference_sha256=_SHA,
            alignment_reference_sha256=_SHA, quality_reference_sha256=_SHA,
            validation_passed=False, validation_status="failed",
            quality_level=QualityLevel.FAILED, overall_quality_score=0.1,
            validation_errors=("VC01: sha mismatch",), validation_warnings=(),
            num_checks=11, num_errors=1, num_warnings=0, num_passed=10, check_results=(),
        )
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), v, _metrics(),
            output_directory="/out", schema_version="0.1.0", force_save=True,
        )
        a = w.documents["analysis.json"]
        assert a["validation_errors"] == ["VC01: sha mismatch"]
        assert a["num_errors"] == 1
        assert a["validation_passed"] is False

    def test_transcript_segment_text_from_asr(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        t = w.documents["transcript.json"]
        assert t["segments"][0]["text"] == "texto 0"

    def test_transcript_timestamp_from_t07(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        t = w.documents["transcript.json"]
        assert t["segments"][0]["start_ms"] == 0
        assert t["segments"][0]["duration_ms"] == 1000

    def test_metadata_word_confidence_flag(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        m = w.documents["metadata.json"]
        assert m["word_count_confidence_ge_08"] == 0
        assert m["speaker_count_detected"] == 1

    def test_metadata_step_timings_passthrough(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        m = w.documents["metadata.json"]
        assert m["step_timings_sec"]["MEDIA_VALIDATION"] == 1.0

    def test_schema_version_in_all_documents(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="9.9.9",
        )
        for doc in w.documents.values():
            assert doc["schema_version"] == "9.9.9"

    def test_speakers_no_identity(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        s = w.documents["speakers.json"]
        assert s["speakers"][0]["speaker_label"] == "SPEAKER_00"
        assert "name" not in s["speakers"][0]
        assert "identity" not in s["speakers"][0]

    def test_output_result_schema_version(self):
        w = _FakeWriter()
        r = RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        assert r.schema_version == "0.1.0"
        assert r.output_directory == "/out"

    def test_segments_no_timestamp_alteration(self):
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        s = w.documents["segments.json"]["segments"][0]
        assert s["start_ms"] == 0
        assert s["end_ms"] == 1000

    def test_metadata_rtf_none(self):
        m = _metrics()
        m = MetricsResult.model_construct(
            source_preprocessed_sha256=_SHA, validation_reference_sha256=_SHA,
            metrics_enabled=True,
            processing_wall_time_sec=5.0, audio_duration_sec=0.0, real_time_factor_RTF=None,
            step_timings_sec={}, peak_cpu_rss_mb=0.0, peak_gpu_vram_used_mb=0.0, peak_gpu_vram_total_mb=0.0,
            cpu_utilization_avg_percent=0.0, gpu_utilization_avg_percent=0.0,
            segment_count_total=0, segment_count_with_text=0, word_count_total=0,
            word_count_confidence_ge_08=0, speaker_count_detected=0, silence_segment_count=0,
            average_asr_confidence=0.0, average_diarization_confidence=0.0,
            language_detection_confidence=0.0, overall_quality_score=0.9,
            profile_requested="", profile_applied="", profile_downgrade_applied=False,
            profile_downgrade_reason="", device_requested="", device_used="",
            analysis_metadata={},
        )
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), m,
            output_directory="/out", schema_version="0.1.0",
        )
        assert w.documents["metadata.json"]["real_time_factor_RTF"] is None

    def test_job_id_passthrough(self):
        w = _FakeWriter()
        r = RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0", job_id="job-12345678",
        )
        assert r.job_id == "job-12345678"
        assert w.documents["analysis.json"]["job_id"] == "job-12345678"

    def test_analysis_no_transcript_leak(self):
        # analysis.json no debe incluir transcript_text (no secretos/texto completo)
        w = _FakeWriter()
        RunStructuredJsonOutputUseCase(writer=w).run(
            _asr(), _timestamps(), _diarization(), _alignment(), _quality(), _validation(), _metrics(),
            output_directory="/out", schema_version="0.1.0",
        )
        a = w.documents["analysis.json"]
        assert "transcript_text" not in a
