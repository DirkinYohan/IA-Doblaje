"""Ports abstractos Validation Step 11 — Capa DOMAIN.

Clean Architecture. Domain NO usa IA. Solo ABC.
"""
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
    """Puerto abstracto Validation. Application depende solo de esto.

    Implementación concreta: el propio UseCase (lógica pura, sin infraestructura).
    """

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
        """Ejecutar los checks VC01..VC11 y producir ValidationResult.

        Regla Quality FAILED:
          quality_level == failed + strict_quality == True + force_save == False
          → ValidationFailedError.

        Errores estructurales (VC01..VC11) siempre → ValidationFailedError.
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["ValidationPort"]
