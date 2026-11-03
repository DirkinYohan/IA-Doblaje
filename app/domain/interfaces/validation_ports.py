from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.validation import ValidationThresholds


class ValidationPort(ABC):
    @abstractmethod
    def validate(
        self,
        quality: QualityResult,
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        *,
        thresholds: ValidationThresholds | None = None,
        job_id: str | None = None,
        force_save: bool = False,
        logger: Any | None = None,
    ) -> ValidationResult:
        raise NotImplementedError  # pragma: no cover


__all__ = ["ValidationPort"]
