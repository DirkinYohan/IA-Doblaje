"""Ports abstractos Diarization Step 08 — Speaker Diarization. Capa DOMAIN.

Clean Architecture. Domain NO usa pyannote/torch/speechbrain. Solo ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.diarization import DiarizationThresholds


class SpeakerDiarizerPort(ABC):
    """Puerto abstracto Speaker Diarization. Application depende solo de esto.

    Implementación concreta (Infrastructure): pyannote_diarization_adapter.
    """

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
        """Ejecutar diarización sobre PreprocessedAudio + VadResult.

        Precondición (garantizada por el UseCase):
          prep.sha256 == vad.source_preprocessed_sha256

        Responsabilidades del adapter:
          1. Resolver models_cache_dir y validar containment.
          2. Validar config.yaml local (sin referencias remotas).
          3. Cargar Pipeline.from_pretrained(ruta_local).
          4. Ejecutar apply() sobre PreprocessedAudio.wav_path.
          5. Convertir Annotation → SpeakerTurn (ms).
          6. Normalizar labels, resolver overlaps, intersectar con VAD.
          7. Aplicar duración mínima.
          8. Retornar DiarizationResult frozen forbid.

        Si falta modelo → ModelLoadError (sin descarga).
        Si falla inferencia → DiarizationError.
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["SpeakerDiarizerPort"]
