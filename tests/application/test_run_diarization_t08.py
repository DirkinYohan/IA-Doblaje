"""Tests unitarios T08 Application RunSpeakerDiarizationUseCase — con FAKE Port.

No requiere pyannote. No toca adapter. Solo FAKE Port.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.exceptions import DiarizationError, ModelLoadError, ValidationFailedError
from app.domain.entities.diarization import (
    DIARIZATION_STRATEGY_PYANNOTE,
    DIARIZATION_STRATEGY_SILENT_FALLBACK,
    DiarizationResult,
)
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.diarization import (
    DiarizationThresholds,
    SpeakerTurn,
)
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.application.use_cases.run_diarization import RunSpeakerDiarizationUseCase


_SHA = "a" * 64
_SHA_B = "b" * 64


def _mk_prep(sha: str = _SHA, duration_sec: float = 10.0, wav: str = "X:/d/x.wav") -> PreprocessedAudio:
    return PreprocessedAudio.model_construct(
        wav_path=Path(wav),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=duration_sec,
        size_bytes=1000,
        sha256=sha,
        applied_filters=(),
    )


def _mk_vad(
    sha: str = _SHA,
    intervals: list[tuple[int, int, float]] | None = None,
) -> VadResult:
    if intervals is None:
        intervals = [(0, 10000, 0.9)]
    vis = tuple(
        VoiceInterval(start_ms=s, end_ms=e, max_confidence=c)
        for s, e, c in intervals
    )
    speech_ms = sum(v.duration_ms for v in vis)
    return VadResult.model_construct(
        source_preprocessed_sha256=sha,
        voice_intervals=vis,
        silence_segments=(),
        speech_ratio=1.0 if intervals else 0.0,
        total_speech_ms=speech_ms,
        total_silence_ms=0,
        num_intervals=len(vis),
        num_silences=0,
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",
        sample_rate=16000,
        channels=1,
        duration_ms=10000,
    )


class _FakeDiarizer:
    """FAKE port. Registra llamadas."""

    def __init__(self, result: DiarizationResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls = 0
        self.last_args: tuple | None = None

    def diarize(self, preprocessed_audio, vad_result, *, thresholds=None, job_id=None, logger=None):
        self.calls += 1
        self.last_args = (preprocessed_audio, vad_result, thresholds, job_id)
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        return DiarizationResult(
            source_preprocessed_sha256=str(preprocessed_audio.sha256),
            vad_reference_sha256=str(vad_result.source_preprocessed_sha256),
            total_duration_ms=int(round(float(preprocessed_audio.duration_sec) * 1000)),
            speaker_turns=(SpeakerTurn(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000),),
            num_speakers=1,
            num_turns=1,
        )


# =============================================================================
# Tests
# =============================================================================


class TestInputs:
    def test_none_prep_raises(self):
        uc = RunSpeakerDiarizationUseCase(diarizer=_FakeDiarizer())
        with pytest.raises(ValidationFailedError):
            uc.run(None, _mk_vad())  # type: ignore[arg-type]

    def test_none_vad_raises(self):
        uc = RunSpeakerDiarizationUseCase(diarizer=_FakeDiarizer())
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(), None)  # type: ignore[arg-type]


class TestChainSHA:
    def test_sha_match_ok(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        r = uc.run(_mk_prep(_SHA), _mk_vad(_SHA))
        assert r.strategy == DIARIZATION_STRATEGY_PYANNOTE

    def test_sha_mismatch_raises(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA_B))
        assert fake.calls == 0

    def test_result_sha_mismatch_raises(self):
        bad_result = DiarizationResult(
            source_preprocessed_sha256=_SHA_B,
            vad_reference_sha256=_SHA_B,
            total_duration_ms=10000,
            speaker_turns=(),
            num_speakers=0,
            num_turns=0,
        )
        fake = _FakeDiarizer(result=bad_result)
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA))


class TestThresholds:
    def test_default_thresholds(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        uc.run(_mk_prep(_SHA), _mk_vad(_SHA))
        args = fake.last_args
        assert args is not None
        assert isinstance(args[2], DiarizationThresholds)
        assert args[2].max_num_speakers == 10

    def test_custom_thresholds_passed(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        th = DiarizationThresholds(max_num_speakers=3)
        uc.run(_mk_prep(_SHA), _mk_vad(_SHA), thresholds=th)
        assert fake.last_args[2] is th

    def test_thresholds_wrong_type_raises(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA), thresholds="x")  # type: ignore[arg-type]


class TestSilentFallback:
    def test_zero_intervals_returns_silent(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        r = uc.run(_mk_prep(_SHA), _mk_vad(_SHA, intervals=[]))
        assert r.strategy == DIARIZATION_STRATEGY_SILENT_FALLBACK
        assert r.speaker_turns == ()
        assert r.num_speakers == 0
        assert r.num_turns == 0

    def test_zero_intervals_does_not_call_adapter(self):
        mock_diarizer = Mock()
        uc = RunSpeakerDiarizationUseCase(diarizer=mock_diarizer)
        uc.run(_mk_prep(_SHA), _mk_vad(_SHA, intervals=[]))
        mock_diarizer.diarize.assert_not_called()

    def test_allow_silent_fallback_false_raises(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        th = DiarizationThresholds(allow_silent_fallback=False)
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA, intervals=[]), thresholds=th)
        assert fake.calls == 0


class TestErrors:
    def test_model_load_error_propagates(self):
        fake = _FakeDiarizer(error=ModelLoadError("modelo local no encontrado"))
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(ModelLoadError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA))

    def test_diarization_error_propagates(self):
        fake = _FakeDiarizer(error=DiarizationError("fallo inferencia"))
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(DiarizationError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA))

    def test_generic_error_wrapped(self):
        fake = _FakeDiarizer(error=RuntimeError("boom"))
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(DiarizationError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA))

    def test_wrong_type_result_raises(self):
        fake = _FakeDiarizer(result="not-a-result")  # type: ignore[arg-type]
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA))


class TestPostConditions:
    def test_strict_validity_max_turns(self):
        big_result = DiarizationResult(
            source_preprocessed_sha256=_SHA,
            vad_reference_sha256=_SHA,
            total_duration_ms=10000,
            speaker_turns=tuple(
                SpeakerTurn(speaker_label="SPEAKER_00", start_ms=i * 10, end_ms=i * 10 + 5)
                for i in range(500)
            ),
            num_speakers=1,
            num_turns=500,
        )
        fake = _FakeDiarizer(result=big_result)
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        th = DiarizationThresholds(max_turns_safety=100)
        with pytest.raises(ValidationFailedError):
            uc.run(_mk_prep(_SHA), _mk_vad(_SHA), thresholds=th)


class TestNoMutation:
    def test_inputs_not_mutated(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        prep = _mk_prep(_SHA)
        vad = _mk_vad(_SHA)
        prep_dump = prep.model_dump()
        vad_dump = vad.model_dump()
        uc.run(prep, vad)
        assert prep.model_dump() == prep_dump
        assert vad.model_dump() == vad_dump


class TestDeterminism:
    def test_same_input_same_output(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        r1 = uc.run(_mk_prep(_SHA), _mk_vad(_SHA))
        r2 = uc.run(_mk_prep(_SHA), _mk_vad(_SHA))
        assert r1.model_dump() == r2.model_dump()


class TestDependencyOnlyPort:
    def test_no_infra_imports_in_module_source(self):
        import ast
        from app.application.use_cases.run_diarization import __file__ as _f

        tree = ast.parse(Path(_f).read_text(encoding="utf-8"))
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx"}
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        assert not (imports & forbidden), f"UseCase T08 importa infraestructura: {imports & forbidden}"


class TestJobId:
    def test_job_id_passthrough(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        uc.run(_mk_prep(_SHA), _mk_vad(_SHA), job_id="job-12345678")
        assert fake.last_args[3] == "job-12345678"

    def test_no_job_id_ok(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        r = uc.run(_mk_prep(_SHA), _mk_vad(_SHA))
        assert r.job_id is None

    def test_silent_fallback_job_id_preserved(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        r = uc.run(_mk_prep(_SHA), _mk_vad(_SHA, intervals=[]), job_id="job-12345678")
        assert r.job_id == "job-12345678"


class TestSilentFallbackMeta:
    def test_silent_fallback_has_metadata(self):
        fake = _FakeDiarizer()
        uc = RunSpeakerDiarizationUseCase(diarizer=fake)
        r = uc.run(_mk_prep(_SHA), _mk_vad(_SHA, intervals=[]))
        assert r.analysis_metadata.get("fallback_reason") == "vad_no_intervals"
