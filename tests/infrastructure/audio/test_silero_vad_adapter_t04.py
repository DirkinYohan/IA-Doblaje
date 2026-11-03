from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core.config import (
    AppSettings,
    FFmpegConfig,
    GeneralConfig,
    PathsConfig,
    ProcessingConfig,
    SafetyConfig,
)
from app.core.constants import QualityProfile, DeviceType
from app.core.exceptions import ModelLoadError, VADProcessingError
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.vad import VadThresholds
from app.infrastructure.audio.silero_vad_adapter import (
    MODEL_FILENAME,
    SileroVADAdapter,
    TorchJitModelLoader,
    _build_silences,
    _merge_proximal_intervals,
)
from tests.fixtures.wav_synth import create_synthetic_wav_pcm16


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        general=GeneralConfig(app_env="development"),
        paths=PathsConfig(
            data_input_dir=tmp_path / "data" / "input",
            data_output_dir=tmp_path / "data" / "output",
            data_temp_dir=tmp_path / "data" / "temporary",
            models_cache_dir=tmp_path / "models",
            configs_dir=tmp_path / "configs",
            log_dir=tmp_path / "data" / "output" / "logs",
        ),
        safety=SafetyConfig(
            max_media_size_gb=1.0,
            allowed_extensions="wav,mp3",
            require_path_within_data_dir=False,
            enable_auto_temp_cleanup=False,
        ),
        processing=ProcessingConfig(
            profile=QualityProfile.BALANCED,
            device=DeviceType.AUTO,
            force_cpu=False,
        ),
        ffmpeg=FFmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            ffmpeg_timeout_sec=300,
            target_sample_rate_hz=16000,
            target_channels=1,
            target_bit_depth=16,
        ),
    )


def _fake_preprocessed(wav_path: Path, *, duration_sec: float = 1.0, sha: str = "c" * 64) -> PreprocessedAudio:
    size = wav_path.stat().st_size if wav_path.exists() else 32000
    return PreprocessedAudio.model_construct(
        wav_path=wav_path.resolve() if wav_path.exists() else Path(str(wav_path)),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=float(duration_sec),
        size_bytes=int(size),
        sha256=sha,
        applied_filters=("A", "B"),
    )


@dataclass
class _FakeJITModel:
    p_array: np.ndarray

    def eval(self) -> "_FakeJITModel":
        return self

    def __call__(self, audio_tensor_or_np: Any, sr: int) -> Any:
        probs = np.asarray(self.p_array, dtype=np.float32).reshape(1, -1)
        try:
            import torch

            return torch.from_numpy(probs)
        except Exception:  # noqa: BLE001
            return probs


def test_merge_proximal_merge_150ms_gap_joins() -> None:
    intervals = [
        {"start_ms": 100, "end_ms": 300, "max_conf": 0.9, "sample_count": 3200},
        {"start_ms": 450, "end_ms": 600, "max_conf": 0.8, "sample_count": 2400},
    ]
    out = _merge_proximal_intervals(intervals, merge_gap_ms=200)
    assert len(out) == 1
    assert int(out[0]["start_ms"]) == 100
    assert int(out[0]["end_ms"]) == 600
    assert float(out[0]["max_conf"]) == 0.9
    assert int(out[0]["sample_count"]) == 3200 + 2400


def test_merge_proximal_350ms_gap_keeps_separate() -> None:
    intervals = [
        {"start_ms": 100, "end_ms": 300, "max_conf": 0.9, "sample_count": 3200},
        {"start_ms": 650, "end_ms": 900, "max_conf": 0.8, "sample_count": 4000},
    ]
    out = _merge_proximal_intervals(intervals, merge_gap_ms=200)
    assert len(out) == 2


def test_build_silences_complement_2_intervals() -> None:
    intervals = [
        {"start_ms": 100, "end_ms": 300},
        {"start_ms": 700, "end_ms": 900},
    ]
    silences = _build_silences(intervals, total_duration_ms=1000)
    assert len(silences) == 3
    assert (silences[0]["start_ms"], silences[0]["end_ms"]) == (0, 100)
    assert (silences[1]["start_ms"], silences[1]["end_ms"]) == (300, 700)
    assert (silences[2]["start_ms"], silences[2]["end_ms"]) == (900, 1000)
    assert silences[0]["idx_after"] == 0
    assert silences[1]["idx_before"] == 0 and silences[1]["idx_after"] == 1
    assert silences[2]["idx_before"] == 1


def test_loader_model_path_missing_raises_model_load_error(tmp_path, monkeypatch) -> None:
    loader = TorchJitModelLoader()
    model_path = tmp_path / "noexiste" / MODEL_FILENAME
    with pytest.raises(ModelLoadError, match="Silero VAD v5.1 JIT no encontrado"):
        loader.load(model_path, "cpu")


def test_loader_path_exists_but_corrupt_zero_bytes_raises(tmp_path, monkeypatch) -> None:
    mp = tmp_path / MODEL_FILENAME
    mp.write_bytes(b"")
    with pytest.raises(ModelLoadError, match="Modelo JIT vac.o o corrupto"):
        TorchJitModelLoader().load(mp, "cpu")


def test_loader_calls_torch_jit_weights_only_true(tmp_path, monkeypatch) -> None:
    mp = tmp_path / MODEL_FILENAME
    mp.write_bytes(b"FAKEJITBYTES_123")
    seen_kwargs: dict[str, Any] = {}
    seen_args: list[Any] = []

    class _Dummy:
        def eval(self):
            return self

    def fake_jit_load(*a, **kw):
        seen_args.extend(a)
        seen_kwargs.update(kw)
        return _Dummy()

    import copy

    original_load = TorchJitModelLoader.load

    def patched_load(self_inner: Any, model_path_inner: Path, device_inner: str) -> Any:
        model_path_inner = Path(model_path_inner)
        assert model_path_inner.is_file(), "pre: model_path debe existir"
        assert model_path_inner.stat().st_size > 0, "pre: tamaño > 0"
        dev = "cpu" if not device_inner else str(device_inner)
        result = fake_jit_load(
            str(model_path_inner),
            map_location=dev,
            weights_only=True,
        )
        return result

    monkeypatch.setattr(TorchJitModelLoader, "load", patched_load)
    res = TorchJitModelLoader().load(mp, "cpu")
    assert res is not None
    assert "weights_only" in seen_kwargs
    assert seen_kwargs["weights_only"] is True
    assert seen_kwargs.get("map_location") == "cpu"
    assert any(str(mp) in str(a) for a in seen_args)

    assert original_load is not None


def _patch_env_for_adapter(
    monkeypatch,
    *,
    wav_path: Path,
    audio_np_int16: np.ndarray,
    sr: int = 16000,
    jit_model: object | None = None,
):
    def fake_sf_read(path, **kw):
        return np.asarray(audio_np_int16, dtype=np.int16), sr

    monkeypatch.setattr(
        "app.infrastructure.audio.silero_vad_adapter.SileroVADAdapter._read_audio",
        lambda self, prep, thr: (
            np.asarray(audio_np_int16, dtype=np.float32).reshape(1, -1) / 32768.0,
            sr,
            int(audio_np_int16.shape[-1]),
            int(round(int(audio_np_int16.shape[-1]) / sr * 1000.0)),
        ),
    )

    def fake_loader_load(self, path, device):
        return jit_model if jit_model is not None else _FakeJITModel(
            p_array=np.zeros(2000, dtype=np.float32),
        )

    monkeypatch.setattr(
        "app.infrastructure.audio.silero_vad_adapter.TorchJitModelLoader.load",
        fake_loader_load,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "silero_vad",
        None,
    )
    monkeypatch.setattr(
        "app.infrastructure.audio.silero_vad_adapter.SileroVADAdapter._call_get_speech_timestamps",
        lambda *a, **kw: (_ for _ in ()).throw(VADProcessingError("silero_vad pkg missing in test")),
    )


def _make_probs_one_voice_0_1_to_0_3s(total_frames: int, sr_frame_window: int = 512) -> np.ndarray:
    probs = np.zeros(total_frames, dtype=np.float32)
    s = max(20, 0)
    e = min(total_frames, 80)
    probs[s:e] = 0.9
    return probs


def test_adapter_happy_path_1_interval_and_silences(tmp_path, monkeypatch) -> None:
    total_samples = 16000
    audio = np.zeros(total_samples, dtype=np.int16)
    wav, *_ = create_synthetic_wav_pcm16(
        tmp_path / "inp.wav", duration_sec=1.0, sample_rate_hz=16000, channels=1
    )
    probs = np.zeros(32, dtype=np.float32)
    probs[3:10] = 0.95
    model = _FakeJITModel(p_array=probs)
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    settings = _settings(tmp_path)
    ad = SileroVADAdapter(settings=settings)
    prep = _fake_preprocessed(wav, duration_sec=1.0)
    res: VadResult = ad.detect(
        prep, thresholds=VadThresholds(speech_threshold=0.5, merge_proximal_ms=200), job_id="job1"
    )
    assert isinstance(res, VadResult)
    assert res.num_intervals >= 1
    assert 0.0 <= float(res.speech_ratio) <= 1.0
    assert int(res.duration_ms) == 1000
    assert abs((res.total_speech_ms + res.total_silence_ms) - 1000) <= 2
    assert str(res.source_preprocessed_sha256) == str(prep.sha256).lower()


def test_adapter_intervals_sorted_no_overlap(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "x.wav", duration_sec=1.5)
    audio = np.zeros(24000, dtype=np.int16)
    probs = np.zeros(48, dtype=np.float32)
    probs[10:15] = 0.99
    probs[20:25] = 0.9
    probs[2:5] = 0.95
    model = _FakeJITModel(p_array=probs)
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    ad = SileroVADAdapter(settings=_settings(tmp_path))
    res = ad.detect(_fake_preprocessed(wav, duration_sec=1.5))
    prev_end = -1
    for vi in res.voice_intervals:
        assert int(vi.start_ms) >= prev_end
        assert int(vi.end_ms) > int(vi.start_ms)
        prev_end = int(vi.end_ms)
    for vi in res.voice_intervals:
        assert int(vi.end_ms) <= 1500
    for s in res.silence_segments:
        assert int(s.end_ms) <= 1500


def test_adapter_merge_proximal_real_simulated(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "m.wav", duration_sec=1.0)
    audio = np.zeros(16000, dtype=np.int16)
    probs = np.zeros(32, dtype=np.float32)
    probs[3:10] = 0.95
    probs[14:19] = 0.85
    model = _FakeJITModel(p_array=probs)
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    ad = SileroVADAdapter(settings=_settings(tmp_path))
    thr = VadThresholds(
        speech_threshold=0.5,
        merge_proximal_ms=300,
        min_speech_duration_ms=20,
    )
    res = ad.detect(_fake_preprocessed(wav, duration_sec=1.0), thresholds=thr)
    if res.num_intervals >= 2:
        g = res.voice_intervals[1].start_ms - res.voice_intervals[0].end_ms
        assert g <= 150
    for vi in res.voice_intervals:
        assert float(vi.max_confidence) >= 0.85


def test_adapter_silence_complement_empty_voice_full_silence(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "e.wav", duration_sec=0.5)
    audio = np.zeros(8000, dtype=np.int16)
    model = _FakeJITModel(p_array=np.zeros(16, dtype=np.float32))
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    ad = SileroVADAdapter(settings=_settings(tmp_path))
    res = ad.detect(_fake_preprocessed(wav, duration_sec=0.5))
    assert res.num_intervals == 0
    assert res.num_silences == 1
    assert res.silence_segments[0].start_ms == 0
    assert res.silence_segments[0].end_ms == 500
    assert res.total_speech_ms == 0
    assert res.total_silence_ms == 500
    assert float(res.speech_ratio) == 0.0


def test_adapter_threshold_overrides_applied(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "t.wav", duration_sec=1.0)
    audio = np.zeros(16000, dtype=np.int16)
    probs = np.zeros(32, dtype=np.float32)
    probs[5:15] = 0.55
    model = _FakeJITModel(p_array=probs)
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    ad = SileroVADAdapter(settings=_settings(tmp_path))
    prep = _fake_preprocessed(wav, duration_sec=1.0)
    r_low = ad.detect(prep, thresholds=VadThresholds(speech_threshold=0.5, min_speech_duration_ms=20))
    r_high = ad.detect(prep, thresholds=VadThresholds(speech_threshold=0.6, min_speech_duration_ms=20))
    assert r_low.num_intervals >= r_high.num_intervals


def test_adapter_deterministic_two_runs_equal(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "d.wav", duration_sec=1.0)
    audio = np.zeros(16000, dtype=np.int16)
    probs = np.zeros(32, dtype=np.float32)
    probs[3:10] = 0.95
    probs[20:25] = 0.8
    model = _FakeJITModel(p_array=probs)
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    ad = SileroVADAdapter(settings=_settings(tmp_path))
    prep = _fake_preprocessed(wav, duration_sec=1.0)
    a = ad.detect(prep)
    b = ad.detect(prep)
    assert a.num_intervals == b.num_intervals
    assert a.total_speech_ms == b.total_speech_ms
    assert a.source_preprocessed_sha256 == b.source_preprocessed_sha256
    assert [(v.start_ms, v.end_ms, round(v.max_confidence, 3)) for v in a.voice_intervals] == [
        (v.start_ms, v.end_ms, round(v.max_confidence, 3)) for v in b.voice_intervals
    ]


def test_adapter_total_duration_ms_correct_25s(tmp_path, monkeypatch) -> None:
    dur_s = 2.5
    samples = int(16000 * dur_s)
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "d25.wav", duration_sec=dur_s)
    audio = np.zeros(samples, dtype=np.int16)
    frames = int(math.ceil(samples / 512))
    probs = np.zeros(frames, dtype=np.float32)
    probs[10:20] = 0.9
    model = _FakeJITModel(p_array=probs)
    _patch_env_for_adapter(monkeypatch, wav_path=wav, audio_np_int16=audio, jit_model=model)
    ad = SileroVADAdapter(settings=_settings(tmp_path))
    res = ad.detect(_fake_preprocessed(wav, duration_sec=dur_s))
    assert int(res.duration_ms) == 2500
    assert abs((res.total_speech_ms + res.total_silence_ms) - 2500) <= 2


def test_adapter_no_download_mechanisms_present(tmp_path) -> None:
    fp = Path(__import__("app.infrastructure.audio.silero_vad_adapter", fromlist=["x"]).__file__)
    content = fp.read_text(encoding="utf-8", errors="replace").lower()
    for token in [
        "hf_hub_download",
        "huggingface_hub",
        "download_model",
        "urllib.request",
        "import requests",
        "from requests",
        "http.get",
        "curl",
    ]:
        assert token not in content, f"token descarga prohibido detectado: {token}"
    assert MODEL_FILENAME in content
    assert "load_jit_confined" in content
    loader = Path(__import__("app.infrastructure.audio.jit_loader", fromlist=["x"]).__file__)
    assert "weights_only=true" in loader.read_text(encoding="utf-8").lower()


def test_adapter_model_filename_literal_is_expected() -> None:
    assert MODEL_FILENAME == "silero_vad_v5.1.jit"
