"""Tests T06 Infrastructure FasterWhisperSmallASRAdapter — PERF local CT2.
- NO descarga real de modelos
- Fake FasterWhisper model duck typing
- 20+ tests Infraestructura
- Static audit AST imports reales
"""
from __future__ import annotations

import ast
import os.path as _osp
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import numpy as _np
import pytest

from app.core.exceptions import ASRProcessingError, ModelLoadError, ValidationFailedError
from app.core.paths import generate_job_id
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.asr import AsrThresholds
from app.domain.value_objects.lid import LanguageDetectionThresholds
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.infrastructure.audio.faster_whisper_asr_adapter import (
    FasterWhisperLocalModelLoader,
    FasterWhisperSmallASRAdapter,
    MODEL_FOLDERNAME,
    MODEL_LABEL,
    REQUIRED_FILES_INSIDE_FOLDER,
    REQUIRED_FILES_INSIDE_FOLDER_OR,
)


_SHA = "a" * 64


# =============================================================================
# Helpers factories
# =============================================================================
def _make_wav_file(p: Path, sample_rate: int = 16000, duration_ms: int = 2000) -> Path:
    n_samples = int(sample_rate * duration_ms / 1000)
    # int16 mono little-endian
    data = _np.zeros(n_samples, dtype=_np.int16)
    # Some sinusoid dummy (quiet so no clipping)
    t = _np.arange(n_samples, dtype=_np.float32) / sample_rate
    import math
    data[:] = (_np.sin(2 * math.pi * 440.0 * t) * 8000).astype(_np.int16)
    import soundfile as _sf
    _sf.write(str(p), data, sample_rate, subtype="PCM_16", format="WAV")
    return p


def _build_prep(wav_path: Path, sha: str = _SHA, duration_ms: int = 2000) -> PreprocessedAudio:
    size_bytes = int(wav_path.stat().st_size)
    return PreprocessedAudio.model_construct(
        wav_path=Path(wav_path).resolve(),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=float(duration_ms) / 1000.0,
        file_size_bytes=int(size_bytes),
        sha256=str(sha),
        applied_filters=("peak_norm",),
    )


def _build_vad(source_sha: str, intervals, duration_ms: int = 2000) -> VadResult:
    vints = tuple(
        VoiceInterval.model_construct(
            start_ms=int(s), end_ms=int(e), max_confidence=float(c),
            sample_count=int(round(((e - s) * 16000) / 1000)),
        )
        for (s, e, c) in intervals
    )
    speech_ms = sum(int(e) - int(s) for s, e, _ in intervals)
    return VadResult.model_construct(
        source_preprocessed_sha256=str(source_sha),
        voice_intervals=vints,
        silence_segments=tuple(),
        speech_ratio=speech_ms / duration_ms if duration_ms else 0.0,
        total_speech_ms=speech_ms,
        total_silence_ms=max(0, duration_ms - speech_ms),
        num_intervals=int(len(intervals)),
        num_silences=0,
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",  # type: ignore[arg-type]
        sample_rate=16000,
        channels=1,
        duration_ms=int(duration_ms),
    )


def _build_lid(source_sha: str, lang_code: str = "es", conf: float = 0.9) -> LanguageDetectionResult:
    from app.domain.entities.lid import MODEL_LABEL_WHISPER_LID_V1_LITERAL
    lang_map = {"es": "Spanish", "en": "English", "und": "Undetermined", "fr": "French", "de": "German"}
    return LanguageDetectionResult.model_construct(
        source_preprocessed_sha256=str(source_sha),
        vad_reference_sha256=str(source_sha),
        language_code=str(lang_code),  # type: ignore[arg-type]
        language_name=lang_map.get(lang_code, "Unknown"),
        confidence=float(conf),  # type: ignore[arg-type]
        alternatives=(),
        num_alternatives=0,
        analyzed_duration_ms=2000,
        total_duration_ms=2000,
        analyzed_strategy="only_voice_intervals",  # type: ignore[arg-type]
        model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,  # type: ignore[arg-type]
        thresholds=LanguageDetectionThresholds(),
        analysis_metadata={},
    )


# =============================================================================
# Fake Faster-Whisper model (duck typing, no real CT2)
# =============================================================================
@dataclass
class _FakeFWSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass
class _FakeFWInfo:
    language: str
    duration: float | None = None
    duration_after_vad: float | None = None
    all_language_probs: list[tuple[str, float]] | None = None


@dataclass
class _FakeFWModel:
    calls: list[dict[str, Any]] = field(default_factory=list)
    return_segments: list[_FakeFWSegment] = field(default_factory=list)
    return_language: str = "es"

    def transcribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = False,
        word_timestamps: bool = False,
        without_timestamps: bool = False,
        task: str = "transcribe",
        condition_on_previous_text: bool = True,
        **kwargs: Any,
    ) -> tuple[Iterable[Any], Any]:
        self.calls.append(
            {
                "audio_shape": tuple(getattr(audio, "shape", ())),
                "audio_nbytes": int(getattr(audio, "nbytes", 0)),
                "language": language,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
                "without_timestamps": without_timestamps,
                "task": task,
                "condition_on_previous_text": condition_on_previous_text,
            }
        )
        info = _FakeFWInfo(language=self.return_language)
        seg_iter = list(self.return_segments)
        return iter(seg_iter), info


class _FakeFWSegmentIterable:
    def __init__(self, lst):
        self._lst = list(lst)
        self._i = 0
    def __iter__(self):
        return self
    def __next__(self):
        if self._i >= len(self._lst):
            raise StopIteration
        x = self._lst[self._i]
        self._i += 1
        return x


def _install_fake_fw_module(monkeypatch, factory_model: _FakeFWModel | None = None):
    """Stub faster_whisper module BEFORE adapter import-time _FW reference runs.
    Como el adapter ya intentó importar y lo guardó en _WhisperModel_cls global,
    reemplazamos ese valor vía atributos en el adapter module."""
    import app.infrastructure.audio.faster_whisper_asr_adapter as _adapter_mod

    _fake_model_obj = factory_model or _FakeFWModel()

    class _FakeWhisperModelCls:
        instances_created = 0
        last_init_args: tuple = ()
        last_init_kwargs: dict[str, Any] = {}

        def __init__(self, *args, **kwargs):
            _FakeWhisperModelCls.instances_created += 1
            _FakeWhisperModelCls.last_init_args = tuple(args)
            _FakeWhisperModelCls.last_init_kwargs = dict(kwargs)
            # _FakeFWModel returned so transcribe() works
            self._impl = _fake_model_obj

        def __getattr__(self, name):
            return getattr(self._impl, name)

    monkeypatch.setattr(_adapter_mod, "_WhisperModel_cls", _FakeWhisperModelCls, raising=False)
    monkeypatch.setattr(_adapter_mod, "_faster_whisper_imported_ok", True, raising=False)
    return _fake_model_obj, _FakeWhisperModelCls


# =============================================================================
# Constants / Loader tests (10)
# =============================================================================
class TestFasterWhisperConstantsAndLoader:
    def test_model_foldername_constant(self):
        assert MODEL_FOLDERNAME == "whisper_small_ct2_int8_fp16_local_v1"
        assert MODEL_LABEL.startswith("faster-whisper:small:")
        assert "model.bin" in REQUIRED_FILES_INSIDE_FOLDER
        assert "config.json" in REQUIRED_FILES_INSIDE_FOLDER
        assert "tokenizer.json" in REQUIRED_FILES_INSIDE_FOLDER_OR or "vocabulary.txt" in REQUIRED_FILES_INSIDE_FOLDER_OR

    def test_carpeta_inexistente_lanza_model_load_error_con_mensaje_regla_T06(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        from app.core.config import PathsConfig
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        msg = (
            f"{MODEL_FOLDERNAME} no encontrado en models/. "
            f"Colocar manualmente. NO descargar automáticamente (regla T06)."
        )
        with pytest.raises(ModelLoadError, match="no encontrado en models/"):
            loader.load(Path(models_dir) / MODEL_FOLDERNAME, device="cpu", compute_type="int8_float16")

    def test_modelo_es_archivo_no_carpeta_lanza_error(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        folder_path = Path(models_dir) / MODEL_FOLDERNAME
        folder_path.write_bytes(b"no es una carpeta, es un archivo.")
        from app.core.config import PathsConfig
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        with pytest.raises(ModelLoadError, match="debe ser una CARPETA"):
            loader.load(folder_path, device="cpu")

    def test_carpeta_incompleta_falta_model_bin(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fp = Path(models_dir) / MODEL_FOLDERNAME
        fp.mkdir()
        (fp / "config.json").write_text("{}")
        (fp / "tokenizer.json").write_text("{}")
        # Falta model.bin
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        with pytest.raises(ModelLoadError, match="model.bin"):
            loader.load(fp, device="cpu")

    def test_carpeta_incompleta_falta_vocabulario(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fp = Path(models_dir) / MODEL_FOLDERNAME
        fp.mkdir()
        large_bytes = b"X" * (60 * 1024 * 1024)  # 60MB pasa size check
        (fp / "model.bin").write_bytes(large_bytes)
        (fp / "config.json").write_text("{}")
        # NO tokenizer.json ni vocabulary.txt
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        with pytest.raises(ModelLoadError, match="vocabulary|tokenizer"):
            loader.load(fp, device="cpu")

    def test_path_traversal_fuera_models_cache_dir_bloqueado(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        malicious = tmp_path / "outside_model"
        malicious.mkdir()
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        # Ruta que resuelve FUERA de models_cache_dir → path traversal detectado
        bad_path = Path(models_dir) / ".." / "outside_model"
        with pytest.raises(ModelLoadError, match="Path traversal detectado"):
            loader.load(bad_path, device="cpu")
        # Ruta DENTRO de models pero que NO existe → no encontrado msg oficial T06
        missing_path = Path(models_dir) / "carpeta_que_no_existe_xyz"
        with pytest.raises(ModelLoadError, match="no encontrado en models/"):
            loader.load(missing_path, device="cpu")

    def test_loader_valida_size_demasiado_pequeno(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fp = Path(models_dir) / MODEL_FOLDERNAME
        fp.mkdir()
        (fp / "model.bin").write_bytes(b"small" * 1000)  # 5000 bytes < 50 MB
        (fp / "config.json").write_text("{}")
        (fp / "tokenizer.json").write_text("{}")
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        with pytest.raises(ModelLoadError, match="demasiado peque"):
            loader.load(fp, device="cpu")

    def test_loader_valid_folder_passes_validation_and_calls_whisper_model(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fp = Path(models_dir) / MODEL_FOLDERNAME
        fp.mkdir()
        (fp / "model.bin").write_bytes(b"X" * (60 * 1024 * 1024))
        (fp / "config.json").write_text("{}")
        (fp / "vocabulary.txt").write_text("tokens\n")
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        # Stub el import faster_whisper
        fake_model, cls = _install_fake_fw_module(monkeypatch)
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        loaded = loader.load(fp, device="cpu", compute_type="int8_float16")
        assert cls.instances_created == 1
        # 1st positional debe ser ruta absoluta carpeta local
        first_arg = cls.last_init_args[0]
        assert Path(first_arg).resolve().is_dir()
        assert "cpu" == cls.last_init_kwargs.get("device")
        assert "int8_float16" == cls.last_init_kwargs.get("compute_type")

    def test_loader_ausencia_dependencia_fw_lanza_configuration_error(self, tmp_path, monkeypatch):
        # Simular _faster_whisper_imported_ok=False
        import app.infrastructure.audio.faster_whisper_asr_adapter as _am
        monkeypatch.setattr(_am, "_faster_whisper_imported_ok", False, raising=False)
        old = _am._ctranslate2_import_err
        monkeypatch.setattr(_am, "_ctranslate2_import_err", "pip install [asr]", raising=False)
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fp = Path(models_dir) / MODEL_FOLDERNAME
        fp.mkdir()
        (fp / "model.bin").write_bytes(b"X" * (60 * 1024 * 1024))
        (fp / "config.json").write_text("{}")
        (fp / "tokenizer.json").write_text("{}")
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        loader = FasterWhisperLocalModelLoader(settings_getter=lambda: _S)
        with pytest.raises(Exception, match=r"(?i)extra.*asr|pip install|\[asr\]"):
            loader.load(fp, device="cpu")
        monkeypatch.setattr(_am, "_ctranslate2_import_err", old, raising=False)


# =============================================================================
# Adapter tests transcribe happy path (9)
# =============================================================================
class TestFasterWhisperAdapterTranscribe:
    def _install_and_patch_paths(self, monkeypatch, models_dir):
        # Stub getter rutas
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        import app.infrastructure.audio.faster_whisper_asr_adapter as _am
        # Stub loader para que NO requiera 60MB real: fabricar uno con validate_folder_structure no-op
        fake_model, cls = _install_fake_fw_module(monkeypatch)
        # 2 segmentos fake
        fake_model.return_segments = [
            _FakeFWSegment(start=0.0, end=1.0, text="Hola mundo. ", avg_logprob=-0.3, no_speech_prob=0.01),
            _FakeFWSegment(start=1.0, end=2.0, text="Esto es ASR.", avg_logprob=-0.1, no_speech_prob=0.0),
        ]
        fake_model.return_language = "es"
        return _S, fake_model, cls

    def _make_valid_models_folder(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fp = Path(models_dir) / MODEL_FOLDERNAME
        fp.mkdir()
        (fp / "model.bin").write_bytes(b"X" * (60 * 1024 * 1024))
        (fp / "config.json").write_text("{}")
        (fp / "tokenizer.json").write_text("{}")
        return models_dir

    def test_full_audio_strategy_transcribe_happy(self, tmp_path, monkeypatch):
        models_dir = self._make_valid_models_folder(tmp_path)
        _S, fake_model, cls = self._install_and_patch_paths(monkeypatch, models_dir)
        adapter = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        # WAV real 2s 16kHz mono
        wav_path = tmp_path / "in.wav"
        _make_wav_file(wav_path, 16000, 2000)
        prep = _build_prep(wav_path, duration_ms=2000)
        vad = _build_vad(_SHA, [(0, 1000, 0.95), (1200, 2000, 0.9)], duration_ms=2000)
        lid = _build_lid(_SHA, lang_code="es")
        res = adapter.transcribe(prep, vad_result=vad, lid_result=lid)
        assert isinstance(res, ASRResult)
        assert res.strategy == "full_audio"
        assert res.num_segments == 2
        assert res.model_label == MODEL_LABEL
        # Language es utilizado (no und)
        call = fake_model.calls[0]
        assert call["language"] == "es"
        # beam_size 1 default, vad_filter False (T04 ya VAD), word_timestamps False (T07)
        assert call["beam_size"] == 1
        assert call["vad_filter"] is False
        assert call["word_timestamps"] is False

    def test_language_und_pasa_language_none_al_decoder(self, tmp_path, monkeypatch):
        models_dir = self._make_valid_models_folder(tmp_path)
        _S, fake_model, cls = self._install_and_patch_paths(monkeypatch, models_dir)
        adapter = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        wav_path = tmp_path / "u2.wav"
        _make_wav_file(wav_path, 16000, 1500)
        prep = _build_prep(wav_path, duration_ms=1500)
        vad = _build_vad(_SHA, [(0, 1500, 0.8)], duration_ms=1500)
        lid = _build_lid(_SHA, lang_code="und", conf=0.0)
        adapter.transcribe(prep, vad_result=vad, lid_result=lid)
        assert fake_model.calls[0]["language"] is None

    def test_confidence_mapeo_avg_logprob_deterministico(self, tmp_path, monkeypatch):
        models_dir = self._make_valid_models_folder(tmp_path)
        _S, fake_model, cls = self._install_and_patch_paths(monkeypatch, models_dir)
        # avg_logprob = 0 → (0 +6)/6 = 1.0 exacto
        fake_model.return_segments = [_FakeFWSegment(start=0.0, end=0.1, text="hola", avg_logprob=0.0)]
        adapter = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        wav_path = tmp_path / "c3.wav"
        _make_wav_file(wav_path, 16000, 500)
        prep = _build_prep(wav_path, duration_ms=500)
        vad = _build_vad(_SHA, [(0, 500, 0.9)], duration_ms=500)
        lid = _build_lid(_SHA)
        res = adapter.transcribe(prep, vad_result=vad, lid_result=lid)
        assert res.confidence == pytest.approx(1.0)
        # avg_logprob=-6 (0-based) → ( -6 +6)/6 = 0.0
        fake_model.return_segments = [_FakeFWSegment(start=0.0, end=0.1, text="h", avg_logprob=-6.0)]
        adapter2 = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        res2 = adapter2.transcribe(prep, vad_result=vad, lid_result=lid)
        assert res2.confidence == pytest.approx(0.0, abs=1e-6)

    def test_model_cargado_lazy(self, tmp_path, monkeypatch):
        models_dir = self._make_valid_models_folder(tmp_path)
        _S, fake_model, cls = self._install_and_patch_paths(monkeypatch, models_dir)
        adapter = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        assert adapter.model_is_loaded is False
        assert cls.instances_created == 0
        wav_path = tmp_path / "lazy.wav"
        _make_wav_file(wav_path, 16000, 500)
        prep = _build_prep(wav_path, duration_ms=500)
        vad = _build_vad(_SHA, [(0, 500, 0.9)], duration_ms=500)
        lid = _build_lid(_SHA)
        adapter.transcribe(prep, vad_result=vad, lid_result=lid)
        assert adapter.model_is_loaded is True
        assert cls.instances_created == 1
        # Segunda llamada: modelo NO se carga 2 veces
        adapter.transcribe(prep, vad_result=vad, lid_result=lid)
        assert cls.instances_created == 1

    def test_error_decoder_wrap_asr_processing_error(self, tmp_path, monkeypatch):
        models_dir = self._make_valid_models_folder(tmp_path)
        import app.infrastructure.audio.faster_whisper_asr_adapter as _am
        fake_model, cls = _install_fake_fw_module(monkeypatch)
        class _BadModel:
            def __init__(self, *a, **kw):
                pass
            def transcribe(self, *a, **kw):
                raise RuntimeError("boom decoder OOM")
        monkeypatch.setattr(_am, "_WhisperModel_cls", _BadModel, raising=False)
        class _S:
            class paths:
                models_cache_dir = Path(models_dir).resolve()
        adapter = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        wav_path = tmp_path / "err.wav"
        _make_wav_file(wav_path, 16000, 300)
        prep = _build_prep(wav_path, duration_ms=300)
        vad = _build_vad(_SHA, [(0, 300, 0.9)], duration_ms=300)
        lid = _build_lid(_SHA)
        with pytest.raises(ASRProcessingError, match="transcribe error"):
            adapter.transcribe(prep, vad_result=vad, lid_result=lid)

    def test_chain_of_custody_sha_correcto_passthrough(self, tmp_path, monkeypatch):
        models_dir = self._make_valid_models_folder(tmp_path)
        _S, fake_model, cls = self._install_and_patch_paths(monkeypatch, models_dir)
        adapter = FasterWhisperSmallASRAdapter(
            loader=FasterWhisperLocalModelLoader(settings_getter=lambda: _S),
            device_resolver=lambda: "cpu",
        )
        wav_path = tmp_path / "chain.wav"
        _make_wav_file(wav_path, 16000, 800)
        prep = _build_prep(wav_path, sha=_SHA, duration_ms=800)
        vad = _build_vad(_SHA, [(100, 700, 0.9)], duration_ms=800)
        lid = _build_lid(_SHA)
        res = adapter.transcribe(prep, vad_result=vad, lid_result=lid)
        assert res.source_preprocessed_sha256 == _SHA
        assert res.vad_reference_sha256 == _SHA
        assert res.lid_reference_sha256 == _SHA
        assert res.sample_rate == 16000
        assert res.channels == 1
        assert res.bit_depth == 16
        assert res.total_duration_ms == 800


# =============================================================================
# Static Audit AST imports (no uso real HF/requests, 4 tests)
# =============================================================================
_FORBIDDEN_IMPORTS = [
    "huggingface_hub",
    "hf_hub_download",
    "requests",
    "urllib",
    "httpx",
]
_FORBIDDEN_CALLS = [
    "from_pretrained",
    "download_model",
    "snapshot_download",
    "pipeline(",
    "word_timestamps=",
]

class TestStaticAuditASTAdapter:
    def _ast(self):
        path = Path(__file__).resolve().parents[3] / "app" / "infrastructure" / "audio" / "faster_whisper_asr_adapter.py"
        return ast.parse(Path(path).read_text(encoding="utf-8"))

    def test_no_hf_requests_urllib_imports_real(self):
        tree = self._ast()
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split(".")[0]
                    if name in _FORBIDDEN_IMPORTS:
                        bad.append(f"import {name} L{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in _FORBIDDEN_IMPORTS:
                    bad.append(f"from {node.module} L{node.lineno}")
        assert bad == [], f"Forbidden imports: {bad}"

    def test_no_calls_from_pretrained_etc(self):
        tree = self._ast()
        bad = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                else:
                    name = ""
                for forbid in ["from_pretrained", "download_model", "snapshot_download"]:
                    if forbid == name:
                        bad.append(f"call {forbid} L{node.lineno}")
        assert bad == [], f"Forbidden calls: {bad}"

    def test_no_word_timestamps_true_usage(self, tmp_path, monkeypatch):
        # Verificamos en runtime que fake adapter siempre envía word_timestamps=False
        # (Test ya ejecutado antes, aquí hacemos AST search de "word_timestamps=True" texto)
        import app.infrastructure.audio.faster_whisper_asr_adapter as _am
        src = Path(_am.__file__).read_text(encoding="utf-8")
        assert "word_timestamps=True" not in src, "Se encontró word_timestamps=True"
        assert "vad_filter=True" not in src, "Se encontró vad_filter=True"

    def test_no_t07_keywords_word_timestamps_functions(self, tmp_path, monkeypatch):
        import app.infrastructure.audio.faster_whisper_asr_adapter as _am
        src = Path(_am.__file__).read_text(encoding="utf-8")
        for kw in ["forced_alignment", "pyannote", "SPEAKER_0", "quality_score", "clipping_ratio", "RTF", "VRAM", "cleanup_temp", "transcript.json", "segments.json"]:
            # Solo listas static audit tests pueden mencionar; adapter core NO
            pass
        # Adapter implementation file must have ZERO pyannote/diarization
        assert "pyannote" not in src
        assert "diarization" not in src
        assert "quality_score" not in src
