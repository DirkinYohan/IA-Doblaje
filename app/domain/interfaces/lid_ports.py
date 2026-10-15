"""Ports abstractos LID Step 05 — Language Detection. Capa DOMAIN.

Clean Architecture. Domain NO usa torch/transformers. Solo ABC/Protocol.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.lid import LanguageDetectionThresholds


class LanguageDetectorPort(ABC):
    """Puerto abstracto Language Detection. Application depende solo de esto."""

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
        """Detectar idioma predominante sobre PreprocessedAudio + VadResult.

        Args:
            preprocessed: Salida Step 03. 16kHz/mono/16-bit. Chain of custody sha.
            vad: Salida Step 04 VAD. Voice intervals D#7 strategy only-voice.
            thresholds: Opcional override LanguageDetectionThresholds. None → defaults T05 (0.4, top3).
            job_id: Para logging estructurado.
            logger: Optional app.core.logging logger.

        Returns:
            LanguageDetectionResult frozen con chain of custody triple bind.
        """
        raise NotImplementedError  # pragma: no cover


@runtime_checkable
class LidModelLoaderPort(Protocol):
    """Carga modelo WhisperEncoder+LIDHead TorchScript JIT.

    Dependency Inversion: Domain no importa torch/jit.

    Firma: load(model_path: Path, device: str) -> object (JIT ScriptModule callable).
    """

    def load(self, model_path: Path, device: str) -> Any:
        ...


__all__ = ["LanguageDetectorPort", "LidModelLoaderPort"]
