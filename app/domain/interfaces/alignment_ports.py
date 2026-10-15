"""Ports abstractos Alignment Step 09 — ASR ↔ Speaker Alignment. Capa DOMAIN.

Clean Architecture. Domain NO usa IA. Solo ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.value_objects.alignment import AlignmentThresholds


class SpeakerAlignmentPort(ABC):
    """Puerto abstracto ASR ↔ Speaker Alignment. Application depende solo de esto.

    Implementación concreta: el propio UseCase (lógica pura, sin infraestructura).
    Se declara como ABC por coherencia con el patrón DIP del proyecto.
    """

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
        """Alinear segmentos ASR (T07) con speaker turns (T08).

        Precondición (garantizada por el UseCase):
          cadena SHA consistente T03→T08.

        Reglas:
          - Weighted Overlap Majority Vote por speaker.
          - tie-break: mayor diarization confidence.
          - empate final: menor start_ms del SpeakerTurn (primera aparición cronológica).
          - segmento sin overlap → NO alineado (num_unassigned), sin speaker artificial.
          - diarización vacía → alignment_silent_fallback.
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["SpeakerAlignmentPort"]
