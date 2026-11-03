from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.value_objects.alignment import AlignmentThresholds


class SpeakerAlignmentPort(ABC):
    @abstractmethod
    def align(
        self,
        timestamps: TimestampGenerationResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        *,
        thresholds: AlignmentThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> AlignmentResult:
        raise NotImplementedError  # pragma: no cover


__all__ = ["SpeakerAlignmentPort"]
