"""Ports abstractos Quality Analysis Step 10 — Capa DOMAIN.

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
from app.domain.value_objects.quality import QualityThresholds


class QualityAnalyzerPort(ABC):
    """Puerto abstracto Quality Analysis. Application depende solo de esto.

    Implementación concreta: el propio UseCase (lógica pura, sin infraestructura).
    """

    @abstractmethod
    def analyze(
        self,
        alignment: AlignmentResult,
        vad: VadResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        *,
        thresholds: QualityThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> QualityResult:
        """Evaluar las reglas QR01..QR11 y producir QualityResult.

        Precondición (garantizada por el UseCase):
          cadena SHA consistente T03→T09.

        Reglas: QUALITY_RULES.md (QR01..QR11), ninguna eliminada.
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["QualityAnalyzerPort"]
