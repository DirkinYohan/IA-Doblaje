from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.value_objects.translation import TranslationLanguageCode


class TranslationModelLoaderPort(ABC):
    @abstractmethod
    def validate(self) -> Any:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def load(self) -> Any:
        raise NotImplementedError  # pragma: no cover


class TranslatorPort(ABC):
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
        raise NotImplementedError  # pragma: no cover


__all__ = ["TranslationModelLoaderPort", "TranslatorPort"]
