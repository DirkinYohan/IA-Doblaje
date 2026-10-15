"""Ports abstractos VAD Step 04 — Capa DOMAIN.

Clean Architecture. Domain NO usa torch/silero/soundfile. Solo ABC/Protocol.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.vad import VadThresholds


class VoiceActivityDetectorPort(ABC):
    """Puerto abstracto VAD. Application depende solo de esto."""

    @abstractmethod
    def detect(
        self,
        preprocessed: PreprocessedAudio,
        *,
        thresholds: VadThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> VadResult:
        """Detectar actividad de voz. Retorna VadResult con chain of custody.

        Args:
            preprocessed: Salida Step 03. Garantizado 16kHz/mono/16-bit.
            thresholds: Opcional override VadThresholds. None → defaults T04.
            job_id: Para logging estructurado.
            logger: Optional app.core.logging logger.
        """
        raise NotImplementedError  # pragma: no cover


@runtime_checkable
class VadModelLoaderPort(Protocol):
    """Carga modelo JIT. Dependency Inversion: Domain no importa torch.

    Firma: load(model_path: Path, device: str) -> object (callable JIT/module).
    """

    def load(self, model_path: Path, device: str) -> Any:
        ...


__all__ = ["VoiceActivityDetectorPort", "VadModelLoaderPort"]
