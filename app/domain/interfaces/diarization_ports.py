from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.diarization import DiarizationThresholds


class SpeakerDiarizerPort(ABC):
    @abstractmethod
    def diarize(
        self,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        *,
        thresholds: DiarizationThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> DiarizationResult:
        raise NotImplementedError  # pragma: no cover


__all__ = ["SpeakerDiarizerPort"]
