from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import TranslationError
from app.domain.value_objects.translation import TranslationLanguageCode as Lang
from app.infrastructure.translation import m2m100_ct2_adapter
from app.infrastructure.translation import m2m100_translation_adapter as mod


class _Loader:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _get_models_dir(self) -> Path:
        return self.root

    def _model_dir(self) -> Path:
        return self.root / mod.MODEL_FOLDERNAME


class _FakeCT2:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[str], str, str]] = []

    def load(self) -> None:
        pass

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        self.calls.append((texts, source, target))
        return self.outputs[: len(texts)]


def _ct2_models(root: Path) -> None:
    (root / m2m100_ct2_adapter.CT2_FOLDERNAME).mkdir(parents=True)
    (root / m2m100_ct2_adapter.CT2_FOLDERNAME / "model.bin").write_bytes(b"x")
    (root / mod.MODEL_FOLDERNAME).mkdir(parents=True)
    (root / mod.MODEL_FOLDERNAME / "sentencepiece.bpe.model").write_bytes(b"x")


def test_sin_pesos_pytorch_usa_ct2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ct2_models(tmp_path)
    fake = _FakeCT2(["Hola", "Adiós"])
    monkeypatch.setattr(m2m100_ct2_adapter, "shared", lambda: fake)
    assert mod._ct2_fallback(_Loader(tmp_path)) is fake


def test_con_pesos_y_torch_no_cambia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ct2_models(tmp_path)
    (tmp_path / mod.MODEL_FOLDERNAME / "pytorch_model.bin").write_bytes(b"x")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert mod._ct2_fallback(_Loader(tmp_path)) is None


def test_sin_ningun_modelo_no_hay_alternativa(tmp_path: Path) -> None:
    assert mod._ct2_fallback(_Loader(tmp_path)) is None


def test_traduce_con_ct2() -> None:
    adapter = mod.M2M100TranslatorAdapter(model_loader=object())
    fake = _FakeCT2(["Hola mundo", "Buenos días"])
    adapter._ct2 = fake
    assert adapter.translate(source_text=" Hello world ", source_language=Lang("en"), target_language=Lang("es")) == "Hola mundo"
    assert adapter.translate_batch(["a", "b"], source_language=Lang("en"), target_language=Lang("es")) == ["Hola mundo", "Buenos días"]
    assert fake.calls[0] == (["Hello world"], "en", "es")


def test_ct2_salida_degenerada_falla() -> None:
    adapter = mod.M2M100TranslatorAdapter(model_loader=object())
    adapter._ct2 = _FakeCT2(["aaaaaa"])
    with pytest.raises(TranslationError):
        adapter.translate(source_text="hi", source_language=Lang("en"), target_language=Lang("es"))
