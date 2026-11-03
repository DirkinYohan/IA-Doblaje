from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from app.core.config import FFmpegConfig, SafetyConfig
from app.domain.entities.media import (
    ExtractedAudio,
    PreprocessedAudio,
    ValidatedMedia,
)


@runtime_checkable
class BinaryResolverPort(Protocol):
    def resolve(
        self,
        bin_name: str,
        timeout_s: int = 5,
    ) -> tuple[bool, str, str]:
        ...


class MediaValidatorPort(ABC):
    @abstractmethod
    def validate(
        self,
        input_path: Path,
        job_id: str,
        safety_cfg: SafetyConfig,
        allowed_formats: set[str],
        probe_bin_path: str,
        probe_timeout_s: int = 30,
        logger: Any | None = None,
    ) -> ValidatedMedia:
        ...


class AudioExtractorPort(ABC):
    @abstractmethod
    def extract(
        self,
        validated: ValidatedMedia,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg: FFmpegConfig,
        logger: Any | None = None,
    ) -> ExtractedAudio:
        ...


class AudioPreprocessorPort(ABC):
    @abstractmethod
    def preprocess(
        self,
        extracted: ExtractedAudio,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg: FFmpegConfig,
        logger: Any | None = None,
    ) -> PreprocessedAudio:
        ...


__all__ = [
    "BinaryResolverPort",
    "MediaValidatorPort",
    "AudioExtractorPort",
    "AudioPreprocessorPort",
]
