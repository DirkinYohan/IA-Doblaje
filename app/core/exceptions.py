from __future__ import annotations

from typing import Any


class EngineBaseError(Exception):
    code: str = "ENGINE_ERROR"
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:  # pragma: no cover
        return f"[{self.code}] {self.message}"


class ConfigurationError(EngineBaseError):
    code = "CONFIG_ERROR"
    status_code = 500


class DeviceDetectionError(EngineBaseError):
    code = "DEVICE_DETECT_ERROR"
    status_code = 500


class MediaError(EngineBaseError):
    code = "MEDIA_ERROR"


class UnsupportedMediaError(MediaError):
    code = "UNSUPPORTED_MEDIA"
    status_code = 400


class MediaTooLargeError(MediaError):
    code = "MEDIA_TOO_LARGE"
    status_code = 413


class InvalidMediaPathError(MediaError):
    code = "INVALID_MEDIA_PATH"
    status_code = 400


class MediaProcessingError(MediaError):
    code = "MEDIA_PROCESSING_FAILED"
    status_code = 500


class AudioExtractionError(MediaProcessingError):
    code = "AUDIO_EXTRACTION_FAILED"


class ModelError(EngineBaseError):
    code = "MODEL_ERROR"
    status_code = 500


class ModelLoadError(ModelError):
    code = "MODEL_LOAD_FAILED"


class ModelAdapterNotRegistered(ModelError):
    code = "MODEL_ADAPTER_MISSING"
    status_code = 400


class ASRProcessingError(ModelError):
    code = "ASR_FAILED"


class DiarizationError(ModelError):
    code = "DIARIZATION_FAILED"


class VADProcessingError(ModelError):
    code = "VAD_FAILED"


class LanguageDetectionError(ModelError):
    code = "LANG_DETECT_FAILED"


class TranslationError(ModelError):
    code = "TRANSLATION_FAILED"


class UnsupportedLanguageError(TranslationError):
    code = "UNSUPPORTED_LANGUAGE"
    status_code = 422


class PipelineError(EngineBaseError):
    code = "PIPELINE_ERROR"


class AlignmentError(PipelineError):
    code = "ALIGNMENT_FAILED"


class QualityFailedError(PipelineError):
    code = "QUALITY_FAILED"


class ValidationFailedError(PipelineError):
    code = "VALIDATION_FAILED"
    status_code = 422


class PipelineCancelled(PipelineError):
    code = "PIPELINE_CANCELLED"
    status_code = 409


class ProfileDowngradeRequired(PipelineError):
    code = "PROFILE_DOWNGRADE_REQUIRED"
    status_code = 412

    def __init__(
        self,
        requested_profile: str,
        required_vram_gb: float,
        available_vram_gb: float,
        alternatives: list[str],
        message: str | None = None,
    ) -> None:
        msg = message or (
            f"El perfil '{requested_profile}' requiere "
            f"{required_vram_gb:.1f} GB VRAM. Disponible: {available_vram_gb:.1f} GB."
        )
        super().__init__(
            msg,
            details={
                "requested_profile": requested_profile,
                "required_vram_gb": required_vram_gb,
                "available_vram_gb": available_vram_gb,
                "alternative_profiles": alternatives,
            },
        )
        self.requested_profile = requested_profile
        self.required_vram_gb = required_vram_gb
        self.available_vram_gb = available_vram_gb
        self.alternatives = alternatives


class StorageError(EngineBaseError):
    code = "STORAGE_ERROR"


class OutputWriteError(StorageError):
    code = "OUTPUT_WRITE_FAILED"
