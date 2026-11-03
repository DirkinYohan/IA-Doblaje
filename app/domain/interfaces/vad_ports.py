from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.vad import VadThresholds


class VoiceActivityDetectorPort(ABC):
    @abstractmethod
    def detect(
        self,
        preprocessed: PreprocessedAudio,
        *,
        thresholds: VadThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> VadResult:
        raise NotImplementedError  # pragma: no cover


@runtime_checkable
class VadModelLoaderPort(Protocol):
    def load(self, model_path: Path, device: str) -> Any:
        ...


__all__ = ["VoiceActivityDetectorPort", "VadModelLoaderPort"]
