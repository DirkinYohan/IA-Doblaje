"""Entidades de multimedia y audio — Capa DOMAIN.

Son ValueObjects grandes y estructurados: validados, frozen=True.
Sin dependencias de Application ni Infrastructure.
"""
from __future__ import annotations

from typing import Any

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.core.constants import QualityProfile, MediaFormat
from app.domain.value_objects.audio import (
    AudioDuration,
    BitDepth,
    ChannelCount,
    FileBytes,
    SampleRate,
    Sha256Hash,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers — tipos validados sin depender de BaseModel custom ints ya validados
# ---------------------------------------------------------------------------


def _as_abs_path(p: Path | str) -> Path:
    return Path(p).resolve()


# ---------------------------------------------------------------------------
# Step 01 — ValidatedMedia
# ---------------------------------------------------------------------------


class ValidatedMedia(BaseModel):
    """Resultado de Media Validation (Pipeline Step 01).

    Garantiza:
        - path absoluto y sanitizado (via PathManager antes de llamar al Port)
        - extension permitida (whitelist SafetyConfig)
        - size > 0, size <= MAX_MEDIA_SIZE_GB
        - SHA-256 file hash
        - ffprobe devuelve al menos 1 stream de audio
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    original_path: Path = Field(
        ...,
        description="Ruta original absoluta ya sanitizada (no traversal).",
    )
    safe_name: str = Field(
        ...,
        min_length=1,
        max_length=220,
        description="safe_filename() version para logs/UI.",
    )
    media_format: MediaFormat = Field(
        ..., description="Enum MediaFormat detectado por extensión."
    )
    extension: str = Field(
        ..., min_length=2, max_length=10, description="Extension sin punto, lower."
    )
    size_bytes: FileBytes = Field(
        ..., description="Tamaño del archivo en bytes (FileBytes >= 0)."
    )
    sha256: Sha256Hash = Field(
        ..., description="SHA-256 hex 64 chars del archivo original."
    )
    probed_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Salida JSON de ffprobe -show_format -show_streams (sin leak tokens).",
    )
    duration_guess_sec: AudioDuration | None = Field(
        None,
        description="Duración estimada por ffprobe format.duration (None si no se pudo).",
    )
    probed_stream_count: int = Field(0, ge=0, description="N streams que devolvió ffprobe.")

    @field_validator("original_path", mode="before")
    @classmethod
    def _abs_path(cls, v: Any) -> Path:
        return _as_abs_path(v)

    @field_validator("extension", mode="after")
    @classmethod
    def _norm_ext(cls, v: str) -> str:
        s = v.strip().lower().lstrip(".")
        if not s or len(s) > 10:
            raise ValueError(f"Extension invalida: {v!r}")
        return s


# ---------------------------------------------------------------------------
# Step 02 — ExtractedAudio
# ---------------------------------------------------------------------------


class ExtractedAudio(BaseModel):
    """Resultado de Audio Extraction (Pipeline Step 02).

    Salida WAV PCM 16-bit, sample_rate arbitrario (lo que detecta ffprobe antes del
    preprocessing), canales arbitrarios. El Preprocessing (Step03) posterior normaliza a
    16 kHz / 16-bit / mono target.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    wav_path: Path = Field(
        ...,
        description="Path absoluto WAV PCM extraído (normalmente dentro data/temp/{jid}/workspace).",
    )
    sample_rate: SampleRate = Field(..., description="Sample rate WAV detectado.")
    channels: ChannelCount = Field(1, description="Número de canales detectados.")
    bit_depth: BitDepth = Field(16, description="Bit depth PCM detectado.")
    duration_sec: AudioDuration = Field(
        ..., description="Duración en segundos (por ffprobe del nuevo WAV)."
    )
    size_bytes: FileBytes = Field(..., description="Tamaño bytes WAV.")
    sha256: Sha256Hash = Field(..., description="SHA-256 del archivo WAV extraído.")
    ffmpeg_duration_ms: int = Field(
        0,
        ge=0,
        description="Duración ms reportada por ffmpeg stdout progress (0 si no se leyó).",
    )
    probed_duration_ms: int = Field(
        0, ge=0, description="Duración ms por ffprobe del WAV resultante."
    )
    sanity_duration_pct_diff_ok: bool = Field(
        True,
        description="Sanity check: abs(diff)% <= 5% original vs extracted.",
    )

    @field_validator("wav_path", mode="before")
    @classmethod
    def _abs_path(cls, v: Any) -> Path:
        return _as_abs_path(v)


# ---------------------------------------------------------------------------
# Step 03 — PreprocessedAudio
# ---------------------------------------------------------------------------


class PreprocessedAudio(BaseModel):
    """Resultado de Audio Preprocessing (Pipeline Step 03).

    Garantizado: WAV PCM 16-bit, 16 kHz, mono, con filtros deterministas aplicados.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    wav_path: Path = Field(
        ...,
        description="Path absoluto WAV PCM preprocesado. Normalmente {jid}_preprocessed.wav.",
    )
    sample_rate: SampleRate = Field(
        16000, description="16 kHz target (obligatorio para ASR Fase1)."
    )
    channels: ChannelCount = Field(1, description="1 (mono).")
    bit_depth: BitDepth = Field(16, description="16-bit PCM little-endian.")
    duration_sec: AudioDuration = Field(..., description="Duración segundos.")
    size_bytes: FileBytes = Field(..., description="Tamaño bytes WAV.")
    sha256: Sha256Hash = Field(..., description="SHA-256 del WAV preprocesado.")
    applied_filters: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Filtros aplicados en orden (reproducibilidad). Siempre mismo orden determinístico.",
    )

    @field_validator("wav_path", mode="before")
    @classmethod
    def _abs_path(cls, v: Any) -> Path:
        return _as_abs_path(v)

    @field_validator("sample_rate")
    @classmethod
    def _enforce_target_sr(cls, v: Any) -> SampleRate:
        # Normalizamos a SampleRate, pero Step03 siempre debe ser 16kHz target.
        i = int(v)
        if i != 16000:
            raise ValueError(f"PreprocessedAudio sample_rate debe ser 16000, got {i}")
        return SampleRate(i)

    @field_validator("channels")
    @classmethod
    def _enforce_mono(cls, v: Any) -> ChannelCount:
        i = int(v)
        if i != 1:
            raise ValueError(f"PreprocessedAudio channels debe ser 1 (mono), got {i}")
        return ChannelCount(i)

    @field_validator("bit_depth")
    @classmethod
    def _enforce_16bit(cls, v: Any) -> BitDepth:
        i = int(v)
        if i != 16:
            raise ValueError(f"PreprocessedAudio bit_depth debe ser 16, got {i}")
        return BitDepth(i)


# ---------------------------------------------------------------------------
# UseCase — MediaPrepResult agregado Steps 01-03
# ---------------------------------------------------------------------------


class MediaPrepResult(BaseModel):
    """Resultado final de AnalyzeMediaPrepUseCase.execute().

    Representa Steps 01-03 del pipeline listos para pasar a Step 04 (VAD).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    job_id: str = Field(..., min_length=10, max_length=128)
    profile: QualityProfile = Field(
        ..., description="Perfil usado para el pipeline (QualityProfile enum)."
    )
    validated: ValidatedMedia = Field(..., description="Salida Step 01.")
    extracted_audio: ExtractedAudio | None = Field(
        None, description="Salida Step 02 (None si falló extracción, no strict mode)."
    )
    preprocessed_audio: PreprocessedAudio | None = Field(
        None, description="Salida Step 03 (None si falló o no extracto)."
    )
    step_times_sec: dict[str, float] = Field(
        default_factory=dict,
        description="Tiempos por paso: validate, extract, preprocess en segundos (float).",
    )
    ffmpeg_available: bool = Field(False, description="True si ffmpeg está en PATH.")
    ffprobe_available: bool = Field(False, description="True si ffprobe está en PATH.")
    errors: list[str] = Field(
        default_factory=list,
        description="Lista de errores si strict=False y continuamos. Default strict raise exc directamente.",
    )


__all__ = [
    "ValidatedMedia",
    "ExtractedAudio",
    "PreprocessedAudio",
    "MediaPrepResult",
]
