from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.lid import LanguageDetectionThresholds


class LanguageDetectorPort(ABC):
    @abstractmethod
    def detect(
        self,
        preprocessed: PreprocessedAudio,
        vad: VadResult,
        *,
        thresholds: LanguageDetectionThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> LanguageDetectionResult:
        raise NotImplementedError  # pragma: no cover


@runtime_checkable
class LidModelLoaderPort(Protocol):
    def load(self, model_path: Path, device: str) -> Any:
        ...


__all__ = ["LanguageDetectorPort", "LidModelLoaderPort"]
