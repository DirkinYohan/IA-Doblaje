"""Ports abstractos Translation Step 15 — Capa DOMAIN.

Clean Architecture. Domain NO usa transformers/torch/filesystem. Solo ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.value_objects.translation import TranslationLanguageCode


class TranslationModelLoaderPort(ABC):
    """Puerto abstracto de carga del modelo de traducción local."""

    @abstractmethod
    def validate(self) -> Any:
        """Valida que el modelo local exista. Raise ModelLoadError si falta."""
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def load(self) -> Any:
        """Carga el modelo local (offline). Raise ModelLoadError si falla."""
        raise NotImplementedError  # pragma: no cover


class TranslatorPort(ABC):
    """Puerto abstracto del motor de traducción. Application depende solo de esto.

    translate recibe el texto del segmento actual + contexto opcional
    (anterior/siguiente/speaker/duración) pero SIEMPRE produce 1:1 el
    translated_text del segmento actual.
    """

    @abstractmethod
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
        """Traduce source_text al target_language. Devuelve translated_text."""
        raise NotImplementedError  # pragma: no cover


__all__ = ["TranslationModelLoaderPort", "TranslatorPort"]
