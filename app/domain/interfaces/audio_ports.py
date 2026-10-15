"""Ports (interfaces abstractas) para audio/media — Capa DOMAIN.

Dependency Inversion Principle: las capas Application/Infrastructure dependen de
estos ABCs/Protocol, NO al revés. Domain NO importa nada de Infra/Application.
"""
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


# ---------------------------------------------------------------------------
# BinaryResolver: detectar si ffmpeg/ffprobe están disponibles
# ---------------------------------------------------------------------------


@runtime_checkable
class BinaryResolverPort(Protocol):
    """Resolver: dónde está un binario (ffmpeg/ffprobe) y si funciona.

    Implementación concreta típica: SubprocessFFmpegBinaryResolver.
    """

    def resolve(
        self,
        bin_name: str,
        timeout_s: int = 5,
    ) -> tuple[bool, str, str]:
        """Comprueba si un binario es ejecutable.

        Args:
            bin_name: Nombre binario ("ffmpeg", "ffprobe").
            timeout_s: Timeout en segundos para el subprocess test.

        Returns:
            (available: bool, path_or_error_msg: str, version: str)
                - Si available=True → path_or_error_msg = resolved path absoluto, version[:80]
                - Si available=False → path_or_error_msg = descripción amigable user
                  (ej: "No se encontró ffmpeg en PATH. Instalar FFmpeg LGPL ..."),
                  version=""
        """
        ...


# ---------------------------------------------------------------------------
# MediaValidatorPort: Step 01 — VALIDATION
# ---------------------------------------------------------------------------


class MediaValidatorPort(ABC):
    """Pipeline Step 01: Media Validation.

    El caller debe garantizar que input_path viene de PathManager.resolve_input_path
    (ya anti path-traversal). El Port no debería repetir eso, pero de todas formas
    puede comprobar que exista y no sea un directorio.
    """

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
        """Valida archivo multimedia de entrada.

        Fail-fast order (para no ejecutar ffprobe en entradas obviamente inválidas):
            1. Path.is_file() → raise ValidationFailedError / FileNotFoundError no silenciado
            2. st_size == 0 bytes → raise ValidationFailedError("EmptyMediaFile: 0 bytes")
            3. st_size > (max_gb * 1024**3) → raise MediaTooLargeError
            4. suffix.lower() not in allowed_formats → raise UnsupportedMediaError
            5. SHA-256 hash file (chunk 4MB)
            6. ffprobe -print_format json -show_format -show_streams
               - exit != 0 → raise ValidationFailedError(stderr[:200] truncado, NO leak)
               - streams sin audio → raise ValidationFailedError("No_audio_stream_found")
            7. Return ValidatedMedia.

        Args:
            input_path: Path (absoluto). Ya fue anti-traversal validado.
            job_id: Para contexto de logging (bind_context de structlog).
            safety_cfg: SafetyConfig (max_media_size_gb, allowed_extensions, etc.)
            allowed_formats: allowed_extensions_set lower-case sin punto.
            probe_bin_path: path resolved (por BinaryResolver) a ffprobe.
            probe_timeout_s: Timeout ffprobe.
            logger: Optional structlog BoundLogger (si None se crea get_logger default).
        """
        ...


# ---------------------------------------------------------------------------
# AudioExtractorPort: Step 02 — AUDIO EXTRACTION via FFmpeg
# ---------------------------------------------------------------------------


class AudioExtractorPort(ABC):
    """Pipeline Step 02: Audio Extraction.

    Output: WAV PCM 16-bit (o bit depth según ffmpeg_cfg), sample rate / channels según
    config (target_sample_rate 16k, channels 1).
    """

    @abstractmethod
    def extract(
        self,
        validated: ValidatedMedia,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg: FFmpegConfig,
        logger: Any | None = None,
    ) -> ExtractedAudio:
        """Extrae pista de audio como WAV.

        Implementación típica:
            ffmpeg -hide_banner -nostdin -y -i INPUT -vn -sn -dn \
                   -acodec pcm_s16le -ar {target_sr} -ac {target_ch} OUT_WAV

        Post-extract sanity:
            - Output file size > 0 bytes?
            - Segunda llamada ffprobe para confirmar sr/channels/bit_depth/duration exactos
            - Sanity: diff duration probed original vs extracted (%) ≤ 5%

        Args:
            validated: Salida Step 01 (ValidatedMedia).
            job_temp_root: {data_temp}/{jid}/ directorio workspace ya creado.
            ffmpeg_bin_path: resolved path FFmpeg via BinaryResolver.
            ffmpeg_cfg: FFmpegConfig con target_sample_rate/channels/bit_depth/timeout.
            logger: Optional BoundLogger.

        Returns:
            ExtractedAudio con sha256, paths abs, sanity_pct_diff_ok.

        Raises:
            AudioExtractionError (EngineBaseError status 500?):
                - ffmpeg exit code != 0
                - timeout
                - output no creado
                - output no tiene streams de audio tras nueva llamada probe.
        """
        ...


# ---------------------------------------------------------------------------
# AudioPreprocessorPort: Step 03 — PREPROCESSING DETERMINISTICO
# ---------------------------------------------------------------------------


class AudioPreprocessorPort(ABC):
    """Pipeline Step 03: Audio Preprocessing normalizado.

    Garantiza 16 kHz / 16-bit / mono + filtros orden fijo determinista.
    Aplicamos siempre los mismos filtros independientemente de la entrada para
    máxima reproducibilidad.
    """

    @abstractmethod
    def preprocess(
        self,
        extracted: ExtractedAudio,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg: FFmpegConfig,
        logger: Any | None = None,
    ) -> PreprocessedAudio:
        """Preprocess. Filters aplicados siempre en el MISMO ORDEN.

        Filters AF ffmpeg (-af "..."):
            1. highpass=f=80:order=4
            2. dcshift=0  (remueve DC offset)
            3. aresample=16000:resampler=swr  (resampler incluido en FFmpeg LGPL)
            4. volume=... peak normalize a -1.5dBTP (approx)
                Simplificado: af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=none

        Post:
            - WAV output size > 0?
            - Nueva llamada ffprobe, sample_rate DEBE ser 16000, channels 1, bit_depth 16
              (raise AudioExtractionError si no — debería ser imposible, pero safety check).
            - applied_filters = ("highpass_80hz", "dc_removal", "resample_16k_si_necesario",
                                 "peak_norm_loudnorm")

        Returns:
            PreprocessedAudio frozen.
        """
        ...


__all__ = [
    "BinaryResolverPort",
    "MediaValidatorPort",
    "AudioExtractorPort",
    "AudioPreprocessorPort",
]
