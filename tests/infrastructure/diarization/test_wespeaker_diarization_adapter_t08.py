from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.exceptions import DiarizationError, ModelLoadError
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.diarization import DiarizationThresholds, SpeakerTurn
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.infrastructure.diarization.wespeaker_diarization_adapter import (
    MODEL_FOLDERNAME,
    WESPEAKER_CHECKPOINT_FILENAME,
    WESPEAKER_SUBFOLDER,
    WespeakerDiarizationModelLoader,
    WespeakerSpeakerDiarizationAdapter,
)


_SHA = "a" * 64


def _mk_prep(duration_sec: float = 10.0, wav: str = "X:/d/x.wav") -> PreprocessedAudio:
    return PreprocessedAudio.model_construct(
        wav_path=Path(wav),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=duration_sec,
        size_bytes=1000,
        sha256=_SHA,
        applied_filters=(),
    )


def _mk_vad(intervals: list[tuple[int, int, float]] | None = None) -> VadResult:
    if intervals is None:
        intervals = [(0, 10000, 0.9)]
    vis = tuple(
        VoiceInterval(start_ms=s, end_ms=e, max_confidence=c) for s, e, c in intervals
    )
    speech_ms = sum(v.duration_ms for v in vis)
    return VadResult.model_construct(
        source_preprocessed_sha256=_SHA,
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


class _FakeLoader:
    def __init__(self, checkpoint: Path | None = None, model=None, error: Exception | None = None):
        self.checkpoint = checkpoint or Path("X:/models/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin")
        self.model = model
        self.error = error

    def validate(self) -> Path:
        if self.error is not None:
            raise self.error
        return self.checkpoint

    def load(self, checkpoint_path: Path):
        return self.model


class TestWespeakerDiarizationModelLoader:
    def test_modelo_ausente(self, tmp_path: Path):
        loader = WespeakerDiarizationModelLoader(
            settings_getter=lambda: Mock(paths=Mock(models_cache_dir=tmp_path))
        )
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_modelo_invalido(self, tmp_path: Path):
        d = tmp_path / MODEL_FOLDERNAME / WESPEAKER_SUBFOLDER
        d.mkdir(parents=True)
        cp = d / WESPEAKER_CHECKPOINT_FILENAME
        cp.write_bytes(b"tiny")
        loader = WespeakerDiarizationModelLoader(
            settings_getter=lambda: Mock(paths=Mock(models_cache_dir=tmp_path))
        )
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_carga_correcta(self, tmp_path: Path):
        d = tmp_path / MODEL_FOLDERNAME / WESPEAKER_SUBFOLDER
        d.mkdir(parents=True)
        cp = d / WESPEAKER_CHECKPOINT_FILENAME
        cp.write_bytes(b"x" * 2048)
        loader = WespeakerDiarizationModelLoader(
            settings_getter=lambda: Mock(paths=Mock(models_cache_dir=tmp_path))
        )
        assert loader.validate() == cp


class TestWespeakerSpeakerDiarizationAdapter:
    def test_audio_valido(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(
            return_value=[(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
        )
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        assert isinstance(res, DiarizationResult)
        assert res.num_speakers == 2
        assert res.num_turns == 2

    def test_audio_vacio_sin_habla(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(return_value=[])
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        assert res.num_speakers == 0
        assert res.num_turns == 0
        assert res.speaker_turns == ()

    def test_multiples_hablantes(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(
            return_value=[
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 2.0, "SPEAKER_01"),
                (2.0, 3.0, "SPEAKER_00"),
            ]
        )
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        assert res.num_speakers == 2
        labels = [t.speaker_label for t in res.speaker_turns]
        assert labels == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]

    def test_timestamps_ordenados(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(
            return_value=[
                (2.0, 3.0, "SPEAKER_00"),
                (0.0, 1.0, "SPEAKER_01"),
                (1.0, 2.0, "SPEAKER_00"),
            ]
        )
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        starts = [t.start_ms for t in res.speaker_turns]
        assert starts == sorted(starts)

    def test_speaker_labels_validos(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(return_value=[(0.0, 1.0, "SPEAKER_00")])
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        assert res.speaker_turns[0].speaker_label == "SPEAKER_00"

    def test_output_compatible_diarization_result(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(return_value=[(0.0, 1.0, "SPEAKER_00")])
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        assert isinstance(res.speaker_turns[0], SpeakerTurn)
        assert res.source_preprocessed_sha256 == _SHA
        assert res.vad_reference_sha256 == _SHA
        assert res.total_duration_ms == 10000

    def test_diarization_error_se_propaga(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(side_effect=RuntimeError("boom"))
        with pytest.raises(DiarizationError):
            adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")

    def test_duracion_minima_respetada(self):
        adapter = WespeakerSpeakerDiarizationAdapter(
            model_loader=_FakeLoader(model=Mock())
        )
        adapter._run_diarization = Mock(return_value=[(0.0, 0.1, "SPEAKER_00")])
        res = adapter.diarize(_mk_prep(), _mk_vad(), job_id="test-1234")
        assert res.num_turns == 0


class TestWespeakerOfflineSafety:
    def test_ausencia_tokens_urls_en_codigo(self):
        src = Path(
            "app/infrastructure/diarization/wespeaker_diarization_adapter.py"
        ).read_text(encoding="utf-8")
        assert "use_auth_token" not in src
        assert "auth_token" not in src
        assert "hf_hub_download" not in src
        for line in src.splitlines():
            if "http" in line.lower():
                stripped = line.strip()
                assert stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'")

    def test_constantes_ruta_local(self):
        assert MODEL_FOLDERNAME == "pyannote"
        assert WESPEAKER_SUBFOLDER == "wespeaker-voxceleb-resnet34-LM"
        assert WESPEAKER_CHECKPOINT_FILENAME == "pytorch_model.bin"
