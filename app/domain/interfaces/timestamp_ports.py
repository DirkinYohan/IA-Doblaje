from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.entities.asr import ASRResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.entities.vad import VadResult
from app.domain.value_objects.timestamps import AsrTimestampThresholds


@runtime_checkable
class TimestampNormalizerPort(Protocol):
    """Puerto T07: normaliza segment timestamps metadata cruda ASR → oficial T07.

    Pure logic adapter. NO toca audio WAV. NO carga modelos.
    NO ejecuta ASR nuevamente.
    """

    def normalize(
        self,
        preprocessed_audio: PreprocessedAudio,
        *,
        vad_result: VadResult,
        lid_result: LanguageDetectionResult,
        asr_result: ASRResult,
        thresholds: AsrTimestampThresholds | None = None,
        job_id: str | None = None,
    ) -> TimestampGenerationResult:
        """Produce TimestampGenerationResult frozen forbid.

        Precondición (garantizada por UseCase):
          - prep.sha256 == vad.source_preprocessed_sha256
            == lid.source_preprocessed_sha256
            == asr.source_preprocessed_sha256
          - thresholds NO es None (UseCase instancia defaults si viene None)

        Reglas implementadas por el Adapter normalizador (D#1 a D#5 aprobadas):
          D#1: words=() — WORD_TS_APLAZADO
          D#2: si strategy silent fallback retornar sin lógica.
          D#3: ASRSegment.start_ms None → VAD_INTERPOLATE con VadResult.voice_intervals
          D#4: sort ASC start_ms + clamp end[i]=min(end[i], start[i+1]) overlap normalization
          D#5: clamp end_ms <= total_duration_ms.

        Postcondición: Return TimestampGenerationResult pasa model_validator frozen.
        """
        ...
