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
        ...
