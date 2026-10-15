"""Constantes globales del proyecto.

Enums y valores inmutables compartidos por todas las capas.

IMPORTANTE: No importar dependencias pesadas (torch/etc.) aqui.
            Solo stdlib + typing.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import FrozenSet, Literal


# =============================================================================
# VERSIONADO
# =============================================================================
ENGINE_VERSION: str = "0.1.0"
PIPELINE_VERSION: int = 1
JSON_SCHEMA_VERSION: str = "0.1.0"
PYPROJECT_MIN_PYTHON: tuple[int, int] = (3, 11)
PYPROJECT_MAX_PYTHON: tuple[int, int] = (3, 12)

# =============================================================================
# CALIDAD / PERFILES DE PROCESAMIENTO (PUNTO 6)
# =============================================================================
QualityProfileName = Literal["quality", "balanced", "performance"]


class QualityProfile(str, Enum):
    """Perfil de procesamiento.

    +----------------------+------------------+-------------------+--------------------+
    | Perfil               | Adapter ASR      | Modelo            | VRAM aproximada    |
    +======================+==================+===================+====================+
    | QUALITY              | Whisper-PyTorch  | large-v3 FP16     | 8 GB               |
    +----------------------+------------------+-------------------+--------------------+
    | BALANCED             | Whisper-PyTorch  | medium FP16       | 5 GB               |
    +----------------------+------------------+-------------------+--------------------+
    | PERFORMANCE          | Faster-Whisper   | large-v3 int8     | 3 GB               |
    +----------------------+------------------+-------------------+--------------------+
    """

    QUALITY = "quality"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

    @classmethod
    def _missing_(cls, value: object) -> "QualityProfile | None":
        if isinstance(value, str):
            lower = value.strip().lower()
            for member in cls:
                if member.value == lower:
                    return member
        return None


# =============================================================================
# DEVICE / DISPOSITIVOS
# =============================================================================
DeviceTypeName = Literal["auto", "cpu", "cuda", "mps"]


class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon

    @property
    def is_accelerator(self) -> bool:
        return self in (DeviceType.CUDA, DeviceType.MPS)


# =============================================================================
# FORMATOS / MEDIA
# =============================================================================
class MediaFormat(str, Enum):
    MP4 = "mp4"
    MKV = "mkv"
    MOV = "mov"
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    UNKNOWN = "unknown"


VIDEO_FORMATS: FrozenSet[MediaFormat] = frozenset({
    MediaFormat.MP4,
    MediaFormat.MKV,
    MediaFormat.MOV,
})

AUDIO_ONLY_FORMATS: FrozenSet[MediaFormat] = frozenset({
    MediaFormat.WAV,
    MediaFormat.MP3,
    MediaFormat.M4A,
})

SUPPORTED_MEDIA_FORMATS: FrozenSet[MediaFormat] = VIDEO_FORMATS | AUDIO_ONLY_FORMATS
SUPPORTED_MEDIA_EXTENSIONS: FrozenSet[str] = frozenset(
    f".{fmt.value}" for fmt in SUPPORTED_MEDIA_FORMATS
)

# =============================================================================
# AUDIO PRESETS (targets de normalizacion)
# =============================================================================
DEFAULT_SAMPLE_RATE_HZ: int = 16_000
DEFAULT_CHANNELS: int = 1
DEFAULT_BIT_DEPTH: int = 16
DEFAULT_AUDIO_FORMAT: str = "wav"
DEFAULT_AUDIO_CODEC: str = "pcm_s16le"

# =============================================================================
# PIPELINE STEPS - 14 pasos (PUNTO 9 del diseño actualizado)
# =============================================================================
class PipelineStep(IntEnum):
    """Orden de ejecucion del pipeline AnalyzeAudioUseCase."""

    MEDIA_VALIDATION = 1
    AUDIO_EXTRACTION = 2
    AUDIO_PREPROCESSING = 3
    VOICE_ACTIVITY_DETECTION = 4
    LANGUAGE_DETECTION = 5
    SPEECH_TO_TEXT = 6
    TIMESTAMPS_GENERATION = 7
    SPEAKER_DIARIZATION = 8
    ASR_SPEAKER_ALIGNMENT = 9
    QUALITY_ANALYSIS = 10
    VALIDATION = 11
    METRICS_AGGREGATION = 12
    STRUCTURED_JSON_OUTPUT = 13
    TEMP_CLEANUP = 14
    TRANSLATION = 15

    @property
    def display_name(self) -> str:
        return _STEP_DISPLAY_NAMES.get(self, self.name.replace("_", " ").title())

    @property
    def emoji(self) -> str:
        return _STEP_EMOJI.get(self, "🔹")


_STEP_DISPLAY_NAMES: dict[PipelineStep, str] = {
    PipelineStep.MEDIA_VALIDATION: "Media Validation",
    PipelineStep.AUDIO_EXTRACTION: "Audio Extraction",
    PipelineStep.AUDIO_PREPROCESSING: "Audio Preprocessing",
    PipelineStep.VOICE_ACTIVITY_DETECTION: "Voice Activity Detection",
    PipelineStep.LANGUAGE_DETECTION: "Language Detection",
    PipelineStep.SPEECH_TO_TEXT: "Speech-to-Text",
    PipelineStep.TIMESTAMPS_GENERATION: "Timestamps Generation",
    PipelineStep.SPEAKER_DIARIZATION: "Speaker Diarization",
    PipelineStep.ASR_SPEAKER_ALIGNMENT: "ASR ↔ Speaker Alignment",
    PipelineStep.QUALITY_ANALYSIS: "Quality Analysis",
    PipelineStep.VALIDATION: "Validation",
    PipelineStep.METRICS_AGGREGATION: "Metrics Aggregation",
    PipelineStep.STRUCTURED_JSON_OUTPUT: "Structured JSON Output",
    PipelineStep.TEMP_CLEANUP: "Temp Cleanup",
    PipelineStep.TRANSLATION: "Translation Engine",
}

_STEP_EMOJI: dict[PipelineStep, str] = {
    PipelineStep.MEDIA_VALIDATION: "📂",
    PipelineStep.AUDIO_EXTRACTION: "🔊",
    PipelineStep.AUDIO_PREPROCESSING: "🧹",
    PipelineStep.VOICE_ACTIVITY_DETECTION: "🎤",
    PipelineStep.LANGUAGE_DETECTION: "🌍",
    PipelineStep.SPEECH_TO_TEXT: "📝",
    PipelineStep.TIMESTAMPS_GENERATION: "🔖",
    PipelineStep.SPEAKER_DIARIZATION: "🎭",
    PipelineStep.ASR_SPEAKER_ALIGNMENT: "🔗",
    PipelineStep.QUALITY_ANALYSIS: "🔍",
    PipelineStep.VALIDATION: "✅",
    PipelineStep.METRICS_AGGREGATION: "📊",
    PipelineStep.STRUCTURED_JSON_OUTPUT: "💾",
    PipelineStep.TEMP_CLEANUP: "🧹",
    PipelineStep.TRANSLATION: "🌐",
}

# =============================================================================
# QUALITY ANALYSIS - NIVELES
# =============================================================================
QualityLevelName = Literal["passed", "warning", "failed"]


class QualityLevel(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"

    @property
    def blocking(self) -> bool:
        """Si el nivel debe abortar el save en modo estricto."""
        return self is QualityLevel.FAILED


# =============================================================================
# RESULT STATUS DE UN STEP
# =============================================================================
class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


# =============================================================================
# FILES OUTPUT (5 archivos)
# =============================================================================
OUTPUT_FILES_DEFAULT: tuple[str, ...] = (
    "analysis",
    "transcript",
    "speakers",
    "segments",
    "metadata",
)

# =============================================================================
# SPEAKER DIARIZATION (PUNTO 3)
# =============================================================================
# Pattern estricto del label para Fase 1: SOLO SPEAKER_NN
SPEAKER_LABEL_REGEX: str = r"^SPEAKER_[0-9]{2}$"
SPEAKER_LABEL_MIN_INDEX: int = 0
SPEAKER_LABEL_MAX_INDEX: int = 99
