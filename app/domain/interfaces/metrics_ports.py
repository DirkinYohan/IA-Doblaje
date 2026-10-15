"""Ports abstractos Metrics Aggregation Step 12 — Capa DOMAIN.

Clean Architecture. Domain NO usa IA. Solo ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.metrics import MetricsResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.metrics import RuntimeMetricsSnapshot


class MetricsAggregationPort(ABC):
    """Puerto abstracto Metrics Aggregation. Application depende solo de esto."""

    @abstractmethod
    def aggregate(
        self,
        snapshot: RuntimeMetricsSnapshot,
        language: LanguageDetectionResult,
        asr_result: ASRResult,
        vad: VadResult,
        diarization: DiarizationResult,
        alignment: AlignmentResult,
        quality: QualityResult,
        validation: ValidationResult,
        *,
        job_id: str | None = None,
        metrics_enabled: bool = True,
        logger: Any | None = None,
    ) -> MetricsResult:
        """Agregar métricas desde entidades upstream + snapshot de runtime.

        metrics_enabled=False → MetricsResult deshabilitado (sin inventar métricas).
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["MetricsAggregationPort"]
