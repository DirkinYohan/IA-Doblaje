"""Tests unitarios T05 Infrastructure Adapter Whisper LID JIT.

Sin modelo real. Usa monkeypatch para:
- torch.jit.load → fake JIT callable devuelve 99-dim logits
- soundfile.read → numpy fake PCM16 int16 samples
- Stat del modelo → tam fake 256 MB (válido)

Asegura: NO HF, NO requests, NO urllib, NO descarga, modelo inexistente raise, corrupto raise, weights_only=True,
Top-3 alternatives correctos, analyzed_duration_ms correcto, chain of custody, strategy ONLY VOICE INTERVALS.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.core.exceptions import LanguageDetectionError, ModelLoadError
from app.domain.entities.lid import (
    LID_STRATEGY_SILENT_FALLBACK,
    LID_STRATEGY_VOICE_ONLY,
    LanguageDetectionResult,
)
from app.domain.value_objects.lid import LanguageDetectionThresholds
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.domain.entities.vad import VadResult
from app.infrastructure.audio.whisper_lid_adapter import (
    MODEL_FILENAME,
    NUM_LID_CLASSES,
    WHISPER_MULTILINGUAL_LANGUAGES,
    WhisperEncoderLanguageDetectionAdapter,
    WhisperLidModelLoader,
    language_name_for_code,
)


# =============================================================================
# CONSTANTES TEST
# =============================================================================
assert MODEL_FILENAME == "whisper_encoder_small_multilingual_lid_v1.jit", (
    "OFFICIAL D#3 filename violated."
)
assert NUM_LID_CLASSES == 99


# =============================================================================
# HELPERS: build fixtures
# =============================================================================


def _sha(prefix: str = "a") -> str:
    return prefix * 16 + "0" * 48


def _fake_prep(wav_path: Path, duration_sec: float = 5.0):
    from app.domain.entities.media import PreprocessedAudio
    size = 32000 * int(duration_sec) + 44
    return PreprocessedAudio.model_construct(
        wav_path=wav_path,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=duration_sec,
        size_bytes=size,
        sha256=_sha("a"),
        applied_filters=("peak_norm",),
    )


def _fake_vad(prep_sha: str, intervals, duration_ms: int):
    from app.domain.value_objects.vad import SilenceSegment
    vints: tuple[VoiceInterval, ...] = tuple(
        VoiceInterval(
            start_ms=int(s), end_ms=int(e), max_confidence=float(c),
            duration_ms=int(e) - int(s),
            sample_count=int(round((int(e) - int(s)) * 16)),
        )
        for (s, e, c) in intervals
    )
    speech_ms = sum(int(e) - int(s) for s, e, _ in intervals)
    silence_ms = max(0, duration_ms - speech_ms)
    # Construir silence_segments para que coincida con total_silence_ms (validator VadResult)
    sils: list[SilenceSegment] = []
    cursor = 0
    speech_idx = 0
    for (s, e, _) in intervals:
        s, e = int(s), int(e)
        if s > cursor:
            sils.append(SilenceSegment(
                start_ms=cursor, end_ms=s,
                speech_index_before=speech_idx - 1,
                speech_index_after=speech_idx,
            ))
        speech_idx += 1
        cursor = max(cursor, e)
    if cursor < duration_ms:
        sils.append(SilenceSegment(
            start_ms=cursor, end_ms=duration_ms,
            speech_index_before=speech_idx - 1,
            speech_index_after=-1,
        ))
    return VadResult.model_construct(
        source_preprocessed_sha256=prep_sha,
        voice_intervals=vints,
        silence_segments=tuple(sils),
        speech_ratio=(speech_ms / duration_ms) if duration_ms > 0 else 0.0,
        total_speech_ms=speech_ms,
        total_silence_ms=silence_ms,
        num_intervals=int(len(vints)),
        num_silences=int(len(sils)),
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",  # type: ignore[arg-type]
        sample_rate=16000,
        channels=1,
        duration_ms=duration_ms,
    )


# =============================================================================
# PATCH environment
# =============================================================================


class _FakeTorchTensor:
    def __init__(self, arr: np.ndarray, *, device: str = "cpu", dtype: Any = None) -> None:
        import numpy as _np
        self._arr = _np.asarray(arr).astype(
            dtype if dtype is not None else _np.float32,
            copy=False,
        )
        self._device = device

    def dim(self) -> int:
        return int(self._arr.ndim)

    def unsqueeze(self, d: int):
        import numpy as _np
        return _FakeTorchTensor(_np.expand_dims(self._arr, axis=d), device=self._device)

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    @property
    def shape(self):
        return self._arr.shape

    def cpu(self):
        return self

    def numpy(self):
        import numpy as _np
        return _np.array(self._arr, copy=True)

    def __array__(self, dtype=None, copy=None):
        import numpy as _np
        arr = self._arr if dtype is None else self._arr.astype(dtype, copy=False)
        if copy:
            return _np.array(arr, copy=True)
        if dtype is not None and arr.dtype != dtype:
            return _np.array(arr, copy=True)
        return arr


class _FakeJITModel:
    """Devuelve logits de 99 clases deterministas: idx_peak = highest, resto descendentes exponenciales."""

    def __init__(self, peak_code_index: int = 3, *, device: str = "cpu") -> None:
        # peak_code_index 3 = Spanish (0 en,1 zh,2 de,3 es...)
        self._peak = int(peak_code_index)
        self._device = device
        self._params_iter = False  # para simular model.parameters(False)
        self.call_count = 0

    def eval(self):
        return self

    def __call__(self, audio_t, sr: int):
        self.call_count += 1
        # shape esperado: audio_t (B,N) devuelve (B,99)
        import numpy as _np
        if isinstance(audio_t, _FakeTorchTensor):
            arr = audio_t._arr
        else:
            arr = _np.asarray(audio_t)
        B = 1 if arr.ndim == 1 else arr.shape[0]
        logits = _np.zeros((B, NUM_LID_CLASSES), dtype=_np.float32)
        # Peak en self._peak → score 10.0, resto desciende por index distance
        for cls in range(NUM_LID_CLASSES):
            dist = abs(cls - self._peak)
            logits[:, cls] = 10.0 - 0.25 * dist
        return _FakeTorchTensor(logits, device=self._device)


def _patch_env_for_adapter(
    monkeypatch: Any,
    *,
    models_dir: Path,
    jit_model: Any,
    audio_int16: np.ndarray,
    model_exists: bool = True,
    model_size_bytes: int = 256 * 1024 * 1024,
    raise_jit_load: type[Exception] | None = None,
):
    """Patch settings.paths.models_cache_dir + torch.jit.load + soundfile.read + stat file.

    Returns: wav_path Path (16k pcm16 fake)
    """
    # Settings fake — crear settings propio AppSettings
    from app.core.config import AppSettings, PathsConfig

    # Cambiar paths.models_cache_dir a models_dir
    orig = PathsConfig()
    class _PatchedPaths(PathsConfig):
        def __init__(self, **kwargs):
            kwargs.setdefault("models_cache_dir", models_dir)
            kwargs.setdefault("data_input_dir", orig.data_input_dir)
            kwargs.setdefault("data_output_dir", orig.data_output_dir)
            kwargs.setdefault("data_temp_dir", orig.data_temp_dir)
            kwargs.setdefault("configs_dir", orig.configs_dir)
            kwargs.setdefault("log_dir", orig.log_dir)
            super().__init__(**kwargs)

    def _settings_factory():
        s = AppSettings()
        object.__setattr__(s, "paths", _PatchedPaths())
        return s

    # Patch torch module-level before adapter __init__ creates settings
    try:
        import torch as _t_real_torch_module  # noqa: F401 - se usa via hasattr abajo
        _torch_available = True
    except ModuleNotFoundError:
        _torch_available = False

    # 1) torch.jit.load mock
    if jit_model is None and raise_jit_load is None:
        jit_model = _FakeJITModel()

    def _fake_jit_load(path: str, *, map_location: str = "cpu", weights_only: bool = False):
        assert weights_only is True, "JIT debe cargar con weights_only=True (seguridad anti pickle)."
        if raise_jit_load is not None:
            raise raise_jit_load("fake jit load error")
        return jit_model

    if _torch_available and hasattr(_t_real_torch_module, "jit") and hasattr(_t_real_torch_module.jit, "load"):
        monkeypatch.setattr(_t_real_torch_module.jit, "load", _fake_jit_load)
    else:  # fallback: torch no instalado en venv → crear fake module + registrar en sys.modules
        import types
        fake_torch = types.ModuleType("torch")
        fake_jit = types.ModuleType("torch.jit")
        fake_jit.load = _fake_jit_load  # type: ignore[attr-defined]
        fake_torch.jit = fake_jit  # type: ignore[attr-defined]
        def fake_cuda_avail():
            return False
        fake_torch.cuda = types.ModuleType("torch.cuda")  # type: ignore[attr-defined]
        fake_torch.cuda.is_available = fake_cuda_avail  # type: ignore[attr-defined]
        # Añadir stubs iniciales para que monkeypatch.setattr no falle (raising default True)
        fake_torch.from_numpy = None  # type: ignore[attr-defined]
        fake_torch.inference_mode = None  # type: ignore[attr-defined]
        fake_torch.float32 = None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        _t_real_torch_module = fake_torch  # type: ignore[no-redef]
        _torch_available = True  # desde aquí usamos el fake a través de sys.modules

    # 2) from_numpy, inference_mode, tensor, float32
    import importlib
    _t_for_tensor = importlib.import_module("torch")

    def fake_from_numpy(arr):
        return _FakeTorchTensor(arr)
    monkeypatch.setattr(_t_for_tensor, "from_numpy", fake_from_numpy, raising=False)

    # inference_mode context manager
    class _FakeInfMode:
        def __init__(self, *a, **kw) -> None: pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(_t_for_tensor, "inference_mode", lambda *a, **kw: _FakeInfMode(), raising=False)
    monkeypatch.setattr(_t_for_tensor, "float32", np.float32, raising=False)

    # 3) soundfile.read mock — devuelve audio_int16 (N,) dtype int16 + sr=16000
    import soundfile as _sf
    def fake_sf_read(path, dtype=None, always_2d=False):
        arr = np.array(audio_int16, dtype=np.int16 if dtype is None else dtype, copy=True)
        if always_2d:
            arr = arr[:, None]
        return arr, 16000
    monkeypatch.setattr(_sf, "read", fake_sf_read)

    # 4) Patch Path.stat() for model path when called with non-existing / existing
    #    Aplicamos un monkeypatch a Path.is_file y Path.exists y Path.stat específicamente para el MODEL path.
    #    IMPORTANTE: usar os.path.* (no Path.resolve()) para evitar recursión infinita:
    #    Path.resolve() → p.stat() → patched_stat → Path.resolve() → stack overflow.
    import os.path as _osp
    expected_model_path = Path(models_dir) / MODEL_FILENAME
    _expected_norm = _osp.normcase(_osp.abspath(str(expected_model_path)))

    _orig_stat = Path.stat
    def _patched_stat(self, *a, **kw):
        _self_norm = _osp.normcase(_osp.abspath(str(self)))
        if _self_norm == _expected_norm:
            if not model_exists:
                raise FileNotFoundError(str(self))
            class _S: st_size = model_size_bytes; st_mode = 0o100644
            return _S()
        return _orig_stat(self, *a, **kw)
    monkeypatch.setattr(Path, "stat", _patched_stat)

    _orig_is_file = Path.is_file
    def _patched_is_file(self):
        _self_norm = _osp.normcase(_osp.abspath(str(self)))
        if _self_norm == _expected_norm:
            return bool(model_exists)
        return _orig_is_file(self)
    monkeypatch.setattr(Path, "is_file", _patched_is_file)

    _orig_exists = Path.exists
    def _patched_exists(self):
        _self_norm = _osp.normcase(_osp.abspath(str(self)))
        if _self_norm == _expected_norm:
            return bool(model_exists)
        return _orig_exists(self)
    monkeypatch.setattr(Path, "exists", _patched_exists)

    # Patch get_settings
    from app.core import config as _config_mod
    monkeypatch.setattr(_config_mod, "get_settings", _settings_factory)


# =============================================================================
# TEST loader: modelo inexistente → ModelLoadError con MENSAJE OFFICIAL
# =============================================================================


class TestModelLoaderMissing:
    def test_model_missing_raises_model_load_error_offical_msg(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fake_wav = tmp_path / "a.wav"
        fake_wav.write_bytes(b"RIFF_" + b"\x00" * 32000)
        audio = np.zeros(16000 * 5, dtype=np.int16)
        _patch_env_for_adapter(
            monkeypatch, models_dir=models_dir, jit_model=None,
            audio_int16=audio, model_exists=False,
        )
        loader = WhisperLidModelLoader()
        with pytest.raises(ModelLoadError) as exc:
            loader.load(models_dir / MODEL_FILENAME, device="cpu")
        msg = str(exc.value).lower()
        assert "no encontrado en models" in msg, (
            "Msg oficial debe decir 'Whisper LID model no encontrado en models/...'."
        )
        assert "no descargar automáticamente" in msg or "no descargar" in msg

    def test_model_zero_bytes_corrupto(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fake_wav = tmp_path / "a.wav" ; fake_wav.write_bytes(b"x"*400)
        audio = np.zeros(16000, dtype=np.int16)
        _patch_env_for_adapter(monkeypatch, models_dir=models_dir, jit_model=None,
                               audio_int16=audio, model_exists=True, model_size_bytes=0)
        loader = WhisperLidModelLoader()
        with pytest.raises(ModelLoadError):
            loader.load(models_dir / MODEL_FILENAME, "cpu")

    def test_model_too_small_under_min_100mb(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fake_wav = tmp_path / "a.wav"; fake_wav.write_bytes(b"x"*400)
        audio = np.zeros(16000, dtype=np.int16)
        _patch_env_for_adapter(
            monkeypatch, models_dir=models_dir, jit_model=None,
            audio_int16=audio, model_exists=True, model_size_bytes=50 * 1024 * 1024,  # 50 MB < 100MB min
        )
        loader = WhisperLidModelLoader()
        with pytest.raises(ModelLoadError):
            loader.load(models_dir / MODEL_FILENAME, "cpu")

    def test_jit_load_weights_only_asegurado(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        fake_wav = tmp_path / "a.wav"; fake_wav.write_bytes(b"x"*400)
        audio = np.zeros(16000, dtype=np.int16)
        # Usamos jit_model None → patch interno crea FakeJITModel; comprobamos que el monkeypatch
        # valide weights_only=True en assert dentro _fake_jit_load.
        _patch_env_for_adapter(
            monkeypatch, models_dir=models_dir, jit_model=_FakeJITModel(peak_code_index=0),
            audio_int16=audio, model_exists=True,
        )
        loader = WhisperLidModelLoader()
        m = loader.load(models_dir / MODEL_FILENAME, "cpu")
        assert isinstance(m, _FakeJITModel)


# =============================================================================
# TEST adapter: happy path, Top-3, analyzed_duration_ms, strategy
# =============================================================================


class TestAdapterHappy:
    @pytest.fixture
    def patched(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"; models_dir.mkdir()
        fake_wav_p = tmp_path / "a.wav"; fake_wav_p.write_bytes(b"RIFF" + b"\x00" * 64)
        # Build audio 5s = 80k samples int16 silence with tone
        audio = np.zeros(16000 * 5, dtype=np.int16)
        # Intervalos voz en VAD: 500-1500ms (16k samples), 2500-4000ms (24k samples)
        # total analyzed = 1000+1500 ms = 2500 ms
        intervals_ms = [(500, 1500, 0.92), (2500, 4000, 0.96)]
        prep = _fake_prep(fake_wav_p, duration_sec=5.0)
        vad = _fake_vad(str(prep.sha256).lower(), intervals_ms, duration_ms=5000)
        # Fake model peak es 3 = Spanish
        fake_model = _FakeJITModel(peak_code_index=3, device="cpu")
        _patch_env_for_adapter(monkeypatch, models_dir=models_dir, jit_model=fake_model,
                               audio_int16=audio, model_exists=True)
        from app.core.config import get_settings as _gas
        settings = _gas()
        ad = WhisperEncoderLanguageDetectionAdapter(settings=settings)
        return ad, prep, vad, fake_model

    def test_top_1_spanish_correct(self, patched):
        ad, prep, vad, _ = patched
        thr = LanguageDetectionThresholds()
        r = ad.detect(prep, vad, thresholds=thr)
        assert isinstance(r, LanguageDetectionResult)
        assert r.language_code == "es"
        assert r.language_name == "Spanish"

    def test_confidence_in_range_0_1(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert 0.0 <= float(r.confidence) <= 1.0
        for a in r.alternatives:
            assert 0.0 <= float(a.confidence) <= 1.0

    def test_top3_alternatives_max_2_excluye_primary(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert len(r.alternatives) <= int(r.thresholds.top_k) - 1
        # Check descending confidence
        for i in range(1, len(r.alternatives)):
            assert r.alternatives[i - 1].confidence >= r.alternatives[i].confidence
        # Check ranks correctos 1, 2
        ranks = [a.rank for a in r.alternatives]
        assert ranks == list(range(1, len(ranks) + 1))
        # Check codes no duplicates primary
        codes_alts = [a.language_code for a in r.alternatives]
        assert r.language_code not in codes_alts

    def test_analyzed_strategy_only_voice(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert r.analyzed_strategy == LID_STRATEGY_VOICE_ONLY

    def test_analyzed_duration_ms_cerca_2500(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        # Intervals: 1000ms + 1500ms = 2500ms
        assert int(r.analyzed_duration_ms) >= 2000
        assert int(r.analyzed_duration_ms) <= 3000
        assert int(r.analyzed_duration_ms) <= int(r.total_duration_ms)

    def test_num_speech_intervals_considered_2(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert int(r.num_speech_intervals_considered) == 2

    def test_chain_of_custody_triple_sha_matches(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert str(r.source_preprocessed_sha256) == str(prep.sha256).lower()
        assert str(r.vad_reference_sha256) == str(vad.source_preprocessed_sha256).lower()

    def test_model_label_fijo_whisper_small_v1(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert r.model_label == "whisper-encoder-small-lid:v1"

    def test_sr_channels_bitdepth_16k_mono_16(self, patched):
        ad, prep, vad, _ = patched
        r = ad.detect(prep, vad)
        assert int(r.sample_rate) == 16000
        assert int(r.channels) == 1
        assert int(r.bit_depth) == 16

    def test_silent_fallback_no_intervals(self, patched):
        ad, prep, _vad, _ = patched
        vad_zero = _fake_vad(str(prep.sha256).lower(), intervals=[], duration_ms=5000)
        r = ad.detect(prep, vad_zero)
        assert r.analyzed_strategy == LID_STRATEGY_SILENT_FALLBACK
        assert r.language_code == "und"
        assert r.confidence == 0.0
        assert int(r.num_speech_intervals_considered) == 0
        assert int(r.analyzed_duration_ms) == 0

    def test_determinismo_dos_llamadas_equal(self, patched):
        ad, prep, vad, fm = patched
        r1 = ad.detect(prep, vad)
        r2 = ad.detect(prep, vad)
        assert r1.language_code == r2.language_code
        assert abs(float(r1.confidence) - float(r2.confidence)) < 1e-9
        assert [a.language_code for a in r1.alternatives] == [a.language_code for a in r2.alternatives]
        assert fm.call_count >= 2

    def test_thresholds_top_k_1_only_primary_alternatives_empty(self, patched):
        ad, prep, vad, _ = patched
        thr = LanguageDetectionThresholds(top_k=1)
        r = ad.detect(prep, vad, thresholds=thr)
        assert len(r.alternatives) == 0  # top_k=1, excluye primary

    def test_language_name_help_func_unknown(self):
        assert language_name_for_code("es") == "Spanish"
        assert language_name_for_code("und") == "Undetermined"
        assert language_name_for_code("xyzlang") == "Xyzlang"  # fallback title()

    def test_model_outputshape_wrong_not_99_classes_raises_language_detection_error(
        self, tmp_path, monkeypatch
    ):
        models_dir = tmp_path / "models"; models_dir.mkdir()
        fake_wav = tmp_path / "a.wav"; fake_wav.write_bytes(b"x" * 500)
        audio = np.zeros(16000 * 5, dtype=np.int16)
        # Fake model retorna 10 clases (mal)
        class _BadShapeModel(_FakeJITModel):
            def __call__(self, audio_t, sr: int):
                import numpy as _np
                B = 1
                logits = _np.zeros((B, 10), dtype=_np.float32)
                return _FakeTorchTensor(logits)
        bad_model = _BadShapeModel(peak_code_index=0)
        prep = _fake_prep(fake_wav, duration_sec=5.0)
        vad = _fake_vad(str(prep.sha256).lower(), [(500, 1500, 0.9)], duration_ms=5000)
        _patch_env_for_adapter(monkeypatch, models_dir=models_dir, jit_model=bad_model,
                               audio_int16=audio, model_exists=True)
        from app.core.config import get_settings as _gas
        ad = WhisperEncoderLanguageDetectionAdapter(settings=_gas())
        with pytest.raises(LanguageDetectionError) as exc_info:
            ad.detect(prep, vad)
        assert "99" in str(exc_info.value) or "shape" in str(exc_info.value).lower()


# =============================================================================
# TEST: MODEL_FILENAME constante (D#3) — NO user-provided path
# =============================================================================


def test_model_filename_es_constante_no_user_input():
    # Comprobar que no hay ningún parámetro 'model_filename' público en Adapter constructor que cambie el nombre
    import inspect
    sig = inspect.signature(WhisperEncoderLanguageDetectionAdapter.__init__)
    assert "model_filename" not in sig.parameters
    assert "model_path" not in sig.parameters


# =============================================================================
# STATIC AUDIT: NO HF / NO requests / NO urllib / NO descarga / NO pipeline
# =============================================================================


_FORBIDDEN_IDENTIFIERS = {
    "HF_TOKEN", "huggingface_hub", "hf_hub_download", "urllib", "requests",
    "httpx", "download_model", "snapshot_download", "pipeline(",
    "from_pretrained", "transcribe", "word_timestamps", "pyannote", "diarization",
    "quality_score", "analysis.json", "transcript.json", "speakers.json",
    "segments.json", "metadata.json", "cleanup_temp",
}


@pytest.mark.parametrize("forbidden", sorted(_FORBIDDEN_IDENTIFIERS))
def test_adapter_source_static_no_forbidden_identifier(forbidden: str):
    p = Path(__file__).resolve().parents[3] / "app" / "infrastructure" / "audio" / "whisper_lid_adapter.py"
    txt = p.read_text(encoding="utf-8")
    assert forbidden not in txt, (
        f"Identifier prohibido {forbidden!r} encontrado en adapter T05."
    )


def test_ast_adapter_no_import_prohibited():
    p = Path(__file__).resolve().parents[3] / "app" / "infrastructure" / "audio" / "whisper_lid_adapter.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    forbidden_modules = {
        "huggingface_hub", "requests", "urllib", "httpx", "transformers",
        "faster_whisper",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    bad = imports & forbidden_modules
    assert not bad, f"Adapter LID imports prohibidos: {bad}"


# =============================================================================
# OVERRIDE LANG USER-SPECIFIED SI CONF < THRESHOLD
# =============================================================================


def test_adapter_override_default_language_es_user_specified(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"; models_dir.mkdir()
    fake_wav = tmp_path / "a.wav"; fake_wav.write_bytes(b"x"*500)
    audio = np.zeros(16000 * 5, dtype=np.int16)
    # Fake model peak 0 = English con peak low logit (baja confianza)
    # usando un modelo con peak near flat → softmax todas clases casi uniformes.
    class _FlatModel:
        def __call__(self, audio_t, sr: int):
            import numpy as _np
            if isinstance(audio_t, _FakeTorchTensor):
                B = 1 if audio_t._arr.ndim == 1 else audio_t._arr.shape[0]
            else:
                B = 1
            logits = _np.ones((B, NUM_LID_CLASSES), dtype=_np.float32) * 1.0  # flat
            return _FakeTorchTensor(logits)
        def eval(self): return self
    flat = _FlatModel()
    prep = _fake_prep(fake_wav, 5.0)
    vad = _fake_vad(str(prep.sha256).lower(), [(0, 5000, 0.9)], 5000)
    _patch_env_for_adapter(monkeypatch, models_dir=models_dir, jit_model=flat,
                           audio_int16=audio, model_exists=True)
    from app.core.config import get_settings as _gas
    ad = WhisperEncoderLanguageDetectionAdapter(settings=_gas())
    thr = LanguageDetectionThresholds(
        min_confidence=0.4, default_language_override="pt",
    )
    r = ad.detect(prep, vad, thresholds=thr)
    # Flat conf = 1/99 ≈ 0.0101 < 0.4 threshold → debe aplicar override_lang "pt" Portuguese
    assert r.language_code == "pt"
    assert r.language_name == "Portuguese"
    assert r.confidence == 0.0
    assert len(r.alternatives) == 0


# =============================================================================
# 99 languages mapping is correct: index 0 = English (en)
# =============================================================================


def test_whisper_map_index_zero_english():
    code, name = WHISPER_MULTILINGUAL_LANGUAGES[0]
    assert code == "en"
    assert name == "English"


def test_whisper_map_length_99():
    assert len(WHISPER_MULTILINGUAL_LANGUAGES) == 99
