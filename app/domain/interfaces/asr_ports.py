from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.domain.entities.asr import ASRResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.asr import AsrThresholds


@runtime_checkable
class ASRModelLoaderPort(Protocol):
    def load(
        self,
        model_folder: Path,
        *,
        device: str,
        compute_type: str | None = None,
    ) -> Any:
        ...


@runtime_checkable
class ASRPort(Protocol):
    def transcribe(
        self,
        preprocessed_audio: PreprocessedAudio,
        *,
        vad_result: VadResult,
        lid_result: LanguageDetectionResult,
        thresholds: AsrThresholds | None = None,
        job_id: str | None = None,
    ) -> ASRResult:
        ...
