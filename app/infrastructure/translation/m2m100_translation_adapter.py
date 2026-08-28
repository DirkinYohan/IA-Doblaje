"""Infrastructure — Adapter M2M100 (facebook/m2m100_418M) local offline.

Carga exclusivamente desde models/m2m100_418M/. SIN descargas en runtime.
Implementa TranslationModelLoaderPort + TranslatorPort.

Determinismo: do_sample=False, num_beams=1 (greedy). Sin seed manual
(greedy no muestrea; la salida es determinista por construcción).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.core.exceptions import ModelLoadError
from app.domain.interfaces.translation_ports import (
    TranslationModelLoaderPort,
    TranslatorPort,
)
from app.domain.value_objects.translation import TranslationLanguageCode

MODEL_FOLDERNAME = "m2m100_418M"

# Archivos físicos requeridos del modelo local (verificación pre-carga)
REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "pytorch_model.bin",
    "sentencepiece.bpe.model",
    "tokenizer_config.json",
    "vocab.json",
    "special_tokens_map.json",
)

# Lazy singleton global para no recargar el modelo entre llamadas
_LAZY_MODEL: Any = None
_LAZY_TOKENIZER: Any = None
_LAZY_LOADED_PATH: str | None = None


class M2M100TranslationModelLoader(TranslationModelLoaderPort):
    """Valida y carga el modelo M2M100 local (offline)."""

    def __init__(self, settings_getter: Any | None = None) -> None:
        self._settings_getter = settings_getter

    def _get_models_dir(self) -> Path:
        if self._settings_getter is not None:
            settings = self._settings_getter()
            models_dir = settings.paths.models_cache_dir
        else:
            from app.core.config import get_settings

            models_dir = get_settings().paths.models_cache_dir
        return Path(models_dir).expanduser().resolve()

    def _model_dir(self) -> Path:
        return self._get_models_dir() / MODEL_FOLDERNAME

    def validate(self) -> Path:
        """Valida que el modelo local exista y esté completo."""
        d = self._model_dir()
        if not d.is_dir():
            raise ModelLoadError(
                f"Modelo de traducción no encontrado: {d!s}. "
                "Colocar M2M100 en models/m2m100_418M (regla T15: NO descargar "
                "automáticamente)."
            )
        missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
        if missing:
            raise ModelLoadError(
                f"Modelo M2M100 incompleto en {d!s}: faltan {missing}."
            )
        for f in REQUIRED_FILES:
            if (d / f).stat().st_size <= 0:
                raise ModelLoadError(f"Archivo M2M100 vacío/corrupto: {f}")
        return d

    def load(self) -> Any:
        """Carga modelo + tokenizer local. Devuelve (model, tokenizer)."""
        global _LAZY_MODEL, _LAZY_TOKENIZER, _LAZY_LOADED_PATH
        d = self.validate()
        d_str = str(d)
        if (
            _LAZY_MODEL is not None
            and _LAZY_TOKENIZER is not None
            and _LAZY_LOADED_PATH == d_str
        ):
            return (_LAZY_MODEL, _LAZY_TOKENIZER)
        try:
            from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"transformers no disponible: {exc!r}. Instalar transformers>=4.41,<4.45."
            ) from exc
        try:
            model = M2M100ForConditionalGeneration.from_pretrained(d_str, local_files_only=True)
            tokenizer = M2M100Tokenizer.from_pretrained(d_str, local_files_only=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"No se pudo cargar M2M100 desde {d_str!s}: {exc!r}"
            ) from exc
        model.eval()
        _LAZY_MODEL, _LAZY_TOKENIZER, _LAZY_LOADED_PATH = model, tokenizer, d_str
        return (model, tokenizer)


class M2M100TranslatorAdapter(TranslatorPort):
    """Traduce el segmento actual con M2M100. Offline. Determinista.

    El contexto (anterior/siguiente/speaker/duración) NO se inyecta al modelo
    (M2M100 no es conversacional). El contexto queda para la lógica de
    adaptación/validación del UseCase.
    """

    def __init__(
        self,
        *,
        model_loader: Any | None = None,
        settings_getter: Any | None = None,
        max_length: int = 128,
        num_beams: int = 1,
        do_sample: bool = False,
    ) -> None:
        self._loader: Any = model_loader or M2M100TranslationModelLoader(
            settings_getter=settings_getter
        )
        self._max_length = int(max_length)
        self._num_beams = int(num_beams)
        self._do_sample = bool(do_sample)

    def _get_engine(self) -> Any:
        return self._loader.load()

    def translate(
        self,
        *,
        source_text: str,
        source_language: TranslationLanguageCode,
        target_language: TranslationLanguageCode,
        context_prev: str | None = None,
        context_next: str | None = None,
        speaker_label: str | None = None,
        duration_ms: int | None = None,
    ) -> str:
        """Traduce source_text al target_language (1:1, no usa contexto en el prompt).

        Si la salida del modelo es degenerada (M2M100 falla con textos muy
        cortos/interjecciones), devuelve el texto fuente como traducción segura
        en lugar de emitir salida corrupta (regla: nunca sacrificar significado).
        """
        text = str(source_text).strip()
        if not text:
            return ""

        model, tokenizer = self._get_engine()
        try:
            tokenizer.src_lang = str(source_language)
            encoded = tokenizer(text, return_tensors="pt")
            lang_id = tokenizer.get_lang_id(str(target_language))
            generated = model.generate(
                **encoded,
                forced_bos_token_id=lang_id,
                max_length=self._max_length,
                num_beams=self._num_beams,
                do_sample=self._do_sample,
                no_repeat_ngram_size=3,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            out = (decoded[0] if decoded else "").strip()
        except Exception as exc:  # noqa: BLE001
            from app.core.exceptions import TranslationError

            raise TranslationError(
                f"Error de traducción M2M100 ({source_language}->{target_language}): {exc!r}"
            ) from exc

        # Detección de salida degenerada (M2M100 con inputs ultra-cortos):
        # repetición de un mismo carácter o ausencia total de letras.
        if self._is_degenerate(out):
            return text  # fallback seguro: conserva el significado del original

        return out

    @staticmethod
    def _is_degenerate(text: str) -> bool:
        import re

        s = str(text).strip()
        if not s:
            return True
        # >3 repeticiones del mismo carácter (p. ej. '¡¡¡¡ ¡¡¡¡...')
        if re.search(r"(.)\1{3,}", s):
            return True
        # Sin letras y con longitud > 3 (solo puntuación/ruido)
        if len(s) > 3 and not re.search(r"[A-Za-z\u00C0-\u017F]", s):
            return True
        return False


__all__ = [
    "MODEL_FOLDERNAME",
    "REQUIRED_FILES",
    "M2M100TranslationModelLoader",
    "M2M100TranslatorAdapter",
]
