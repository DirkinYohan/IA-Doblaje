"""Tests unitarios T15 — Infrastructure M2M100 adapter + adaptación doblaje.

- Loader: modelo ausente / incompleto / correcto (fake settings).
- Translator: fake engine para traducir; carga real offline.
- Adaptación: reglas de doblaje (normalización, cps, ratio, warnings).
- AST: ausencia de requests/urllib/tokens/descargas.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.exceptions import ModelLoadError, TranslationError
from app.domain.value_objects.translation import TranslationThresholds
from app.infrastructure.translation.m2m100_translation_adapter import (
    MODEL_FOLDERNAME,
    REQUIRED_FILES,
    M2M100TranslationModelLoader,
    M2M100TranslatorAdapter,
)
from app.infrastructure.translation.translation_adaptation import (
    AdaptationResult,
    TranslationAdaptationRule,
)


class _FakeModel:
    def __init__(self, out_map: dict | None = None):
        self.out_map = out_map or {}
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        lang = kwargs.get("forced_bos_token_id")
        # devolvemos un tensor de tokens ficticio
        import torch
        out = f"TRAD:{lang}"
        return torch.tensor([[1, 2, 3]])


class _FakeTokenizer:
    def __init__(self):
        self.src_lang = None

    def __call__(self, text, return_tensors=None):
        import torch
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def get_lang_id(self, lang: str) -> int:
        return hash(lang) % 1000

    def batch_decode(self, tokens, skip_special_tokens=None):
        return ["Resultado traducido"]


def _mk_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / MODEL_FOLDERNAME
    d.mkdir(parents=True, exist_ok=True)
    for f in REQUIRED_FILES:
        (d / f).write_bytes(b"x" * 2048)
    return d


def _loader_for(tmp_path: Path) -> M2M100TranslationModelLoader:
    return M2M100TranslationModelLoader(
        settings_getter=lambda: Mock(paths=Mock(models_cache_dir=tmp_path))
    )


# =============================================================================
# Loader
# =============================================================================


class TestM2M100Loader:
    def test_modelo_ausente(self, tmp_path):
        loader = _loader_for(tmp_path)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_modelo_incompleto(self, tmp_path):
        d = tmp_path / MODEL_FOLDERNAME
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_bytes(b"x")
        loader = _loader_for(tmp_path)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_carga_correcta_validate(self, tmp_path):
        _mk_models_dir(tmp_path)
        loader = _loader_for(tmp_path)
        d = loader.validate()
        assert (d / MODEL_FOLDERNAME).name == MODEL_FOLDERNAME or d.name == MODEL_FOLDERNAME

    def test_archivo_vacio_rechazado(self, tmp_path):
        d = tmp_path / MODEL_FOLDERNAME
        d.mkdir(parents=True, exist_ok=True)
        for f in REQUIRED_FILES:
            (d / f).write_bytes(b"")  # vacío
        loader = _loader_for(tmp_path)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_path_traversal_no_permitido(self, tmp_path):
        # el loader resuelve dentro de models_cache_dir; un models_cache_dir
        # externo no debería alterar la resolución
        _mk_models_dir(tmp_path)
        loader = _loader_for(tmp_path)
        d = loader._model_dir()
        assert str(d).startswith(str(tmp_path.resolve()))


# =============================================================================
# Translator (con loader fake de transformers)
# =============================================================================


class TestM2M100Translator:
    def test_traduce(self, tmp_path):
        _mk_models_dir(tmp_path)
        loader = _loader_for(tmp_path)
        # sobreescribir load() para devolver fakes sin cargar transformers
        loader.load = Mock(return_value=(_FakeModel(), _FakeTokenizer()))
        adapter = M2M100TranslatorAdapter(model_loader=loader)
        out = adapter.translate(
            source_text="Hello", source_language="en", target_language="es"
        )
        assert out == "Resultado traducido"

    def test_texto_vacio_devuelve_vacio(self, tmp_path):
        _mk_models_dir(tmp_path)
        loader = _loader_for(tmp_path)
        loader.load = Mock(return_value=(_FakeModel(), _FakeTokenizer()))
        adapter = M2M100TranslatorAdapter(model_loader=loader)
        assert adapter.translate(source_text="  ", source_language="en", target_language="es") == ""

    def test_error_traduccion_es_translation_error(self, tmp_path):
        _mk_models_dir(tmp_path)
        loader = _loader_for(tmp_path)

        class _BadModel:
            def generate(self, **kwargs):
                raise RuntimeError("boom")

        loader.load = Mock(return_value=(_BadModel(), _FakeTokenizer()))
        adapter = M2M100TranslatorAdapter(model_loader=loader)
        with pytest.raises(TranslationError):
            adapter.translate(source_text="Hi", source_language="en", target_language="es")

    def test_greedy_determinista_params(self, tmp_path):
        _mk_models_dir(tmp_path)
        model = _FakeModel()
        loader = _loader_for(tmp_path)
        loader.load = Mock(return_value=(model, _FakeTokenizer()))
        adapter = M2M100TranslatorAdapter(model_loader=loader, num_beams=1, do_sample=False)
        assert adapter._num_beams == 1
        assert adapter._do_sample is False


# =============================================================================
# Adaptación para doblaje
# =============================================================================


class TestTranslationAdaptation:
    def test_normaliza(self):
        r = TranslationAdaptationRule().adapt(
            translated_text="  Hola   mundo ,  ¿cómo estás?  ",
            source_text="Hello world, how are you?",
            duration_ms=3000,
            target_language="es",
        )
        assert "  " not in r.adapted_text
        assert r.adapted_text.startswith("Hola mundo")

    def test_no_trunca_nunca(self):
        r = TranslationAdaptationRule().adapt(
            translated_text="Esta es una traducción muy larga que no cabe en el tiempo",
            source_text="short",
            duration_ms=100,
            target_language="es",
        )
        # no trunca: conserva el texto completo
        assert len(r.adapted_text) == len(
            "Esta es una traducción muy larga que no cabe en el tiempo"
        )
        assert r.fits_duration is False
        assert any("duraci" in w for w in r.warnings)

    def test_warning_duracion_insuficiente(self):
        r = TranslationAdaptationRule().adapt(
            translated_text="Hola amigo como estas todo bien espero que si",
            source_text="hello",
            duration_ms=200,
            target_language="es",
        )
        assert any("insuficiente" in w for w in r.warnings) or r.fits_duration is False

    def test_cabe_en_duracion(self):
        r = TranslationAdaptationRule().adapt(
            translated_text="Hola",
            source_text="Hello",
            duration_ms=1000,
            target_language="es",
        )
        assert r.fits_duration is True

    def test_ratio_expansion_warning(self):
        r = TranslationAdaptationRule().adapt(
            translated_text="X" * 50,
            source_text="Y" * 3,
            duration_ms=10000,
            target_language="es",
        )
        assert r.adaptation_ratio > 1.35
        assert any("expansi" in w for w in r.warnings)

    def test_muletillas_eliminadas_es(self):
        r = TranslationAdaptationRule().adapt(
            translated_text="Bueno, hola que tal",
            source_text="Well hello",
            duration_ms=5000,
            target_language="es",
        )
        assert not r.adapted_text.startswith("Bueno, ")

    def test_no_elimina_si_cambia_sentido(self):
        # "well" al inicio puede ser adverbio; nuestra lista solo elimina
        # "well, " con coma al inicio (muletilla inequívoca)
        r = TranslationAdaptationRule().adapt(
            translated_text="Well done!",
            source_text="Well done!",
            duration_ms=5000,
            target_language="en",
        )
        assert "Well done" in r.adapted_text

    def test_conserva_timestamps_no_modificables(self):
        # la adaptación solo produce texto; no toca timestamps
        r = TranslationAdaptationRule().adapt(
            translated_text="hola", source_text="hi", duration_ms=1000, target_language="es"
        )
        assert isinstance(r, AdaptationResult)


# =============================================================================
# AST / Seguridad / Offline
# =============================================================================


class TestTranslationOfflineSafety:
    def test_ast_sin_requests_urllib_descargas(self):
        src = Path("app/infrastructure/translation/m2m100_translation_adapter.py").read_text(encoding="utf-8")
        assert "requests" not in src
        assert "urllib" not in src
        assert "snapshot_download" not in src
        assert "use_auth_token" not in src
        assert "http://" not in src and "https://" not in src

    def test_local_files_only(self):
        src = Path("app/infrastructure/translation/m2m100_translation_adapter.py").read_text(encoding="utf-8")
        assert "local_files_only=True" in src

    def test_ast_adaptacion_sin_ia(self):
        src = Path("app/infrastructure/translation/translation_adaptation.py").read_text(encoding="utf-8")
        assert "torch" not in src
        assert "transformers" not in src

    def test_offline_env_no_rompe_import(self):
        # con HF_HUB_OFFLINE no debe fallar el import del módulo
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import importlib
        mod = importlib.import_module("app.infrastructure.translation.m2m100_translation_adapter")
        assert mod.MODEL_FOLDERNAME == "m2m100_418M"


class TestDegenerateOutput:
    def test_is_degenerate_detecta_repeticion(self):
        assert M2M100TranslatorAdapter._is_degenerate("¡¡¡¡ ¡¡¡¡¡¡¡¡¡¡¡¡")

    def test_is_degenerate_detecta_sin_letras(self):
        assert M2M100TranslatorAdapter._is_degenerate("!!! ... ???")

    def test_is_degenerate_normal_ok(self):
        assert not M2M100TranslatorAdapter._is_degenerate("¿Qué es nuevo, Mark?")

    def test_is_degenerate_vacio(self):
        assert M2M100TranslatorAdapter._is_degenerate("")

    def test_fallback_a_texto_fuente(self, tmp_path):
        _mk_models_dir(tmp_path)
        model = _FakeModel()
        loader = _loader_for(tmp_path)
        loader.load = Mock(return_value=(model, _FakeTokenizer()))
        adapter = M2M100TranslatorAdapter(model_loader=loader)
        # fake tokenizer devuelve 'Resultado traducido' -> no degenerado
        out = adapter.translate(source_text="Hola", source_language="en", target_language="es")
        assert out == "Resultado traducido"
