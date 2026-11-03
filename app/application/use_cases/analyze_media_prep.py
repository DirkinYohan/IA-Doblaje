from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import AppSettings, FFmpegConfig, SafetyConfig, get_settings
from app.core.constants import QualityProfile, SUPPORTED_MEDIA_FORMATS
from app.core.exceptions import (
    AudioExtractionError,
    ConfigurationError,
    ValidationFailedError,
)
from app.core.logging import bind_context, get_logger
from app.core.paths import PathManager, generate_job_id
from app.domain.entities.media import (
    ExtractedAudio,
    MediaPrepResult,
    PreprocessedAudio,
    ValidatedMedia,
)
from app.domain.interfaces.audio_ports import (
    AudioExtractorPort,
    AudioPreprocessorPort,
    BinaryResolverPort,
    MediaValidatorPort,
)


@dataclass
class AnalyzeMediaPrepUseCase:
    binary_resolver: BinaryResolverPort
    validator: MediaValidatorPort
    extractor: AudioExtractorPort
    preprocessor: AudioPreprocessorPort


    def execute(
        self,
        input_path: str | Path,
        job_id: str | None = None,
        settings: AppSettings | None = None,
        logger: Any | None = None,
        force_keep_temp: bool = False,
        strict: bool = True,
        progress_cb: Any | None = None,
    ) -> MediaPrepResult:
        def _emit(step: str) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(step)
            except Exception:  # noqa: BLE001
                pass

        if settings is None:
            try:
                from app.core.config import clear_settings_cache

                clear_settings_cache()
            except Exception:  # noqa: BLE001
                pass
            settings = get_settings()

        processing_cfg = settings.processing
        ffmpeg_cfg: FFmpegConfig = settings.ffmpeg
        safety_cfg: SafetyConfig = settings.safety
        profile: QualityProfile = processing_cfg.profile

        log = logger or get_logger("ia-doblaje.t03.prep")

        jid = str(job_id) if job_id and isinstance(job_id, str) else generate_job_id()
        pm: PathManager = PathManager.from_settings(settings)
        bind_context(
            phase="media_prep_steps01_03",
            job_id=jid,
            profile=profile.value,
            logger=log,
        )

        ffmpeg_ok, ffmpeg_path_or_err, ffmpeg_ver = self.binary_resolver.resolve(
            ffmpeg_cfg.ffmpeg_bin,
            timeout_s=5,
        )
        ffprobe_ok, ffprobe_path_or_err, ffprobe_ver = self.binary_resolver.resolve(
            ffmpeg_cfg.ffprobe_bin,
            timeout_s=5,
        )

        errors_collected: list[str] = []
        if not ffmpeg_ok:
            errors_collected.append(f"ffmpeg_missing::{ffmpeg_path_or_err}")
        if not ffprobe_ok:
            errors_collected.append(f"ffprobe_missing::{ffprobe_path_or_err}")

        if strict and errors_collected:
            missing_msg = "; ".join(errors_collected)
            raise ConfigurationError(
                "T03 requiere binarios FFmpeg/FFprobe disponibles. "
                f"Errores: {missing_msg}"
            )
        if not ffmpeg_ok or not ffprobe_ok:
            log.warning(
                "saltando_steps_01_03_binarios_ausentes",
                errors=errors_collected,
                job_id=jid,
            )
            if strict:
                raise ConfigurationError(
                    "Binarios FFmpeg/FFprobe ausentes y strict=True."
                )
            log.info(
                "continuando_sin_binarios_strict_false",
                job_id=jid,
                ffmpeg=ffmpeg_ok,
                ffprobe=ffprobe_ok,
            )
        ffmpeg_bin: str = ffmpeg_path_or_err if ffmpeg_ok else ""
        ffprobe_bin: str = ffprobe_path_or_err if ffprobe_ok else ""
        ffmpeg_cfg = ffmpeg_cfg.model_copy(
            update={
                "ffmpeg_bin": ffmpeg_bin or ffmpeg_cfg.ffmpeg_bin,
                "ffprobe_bin": ffprobe_bin or ffmpeg_cfg.ffprobe_bin,
            }
        )

        t0 = time.perf_counter()
        times: dict[str, float] = {}

        job_temp_root: Path
        _, job_temp_root = pm.create_job_temp_dir(jid)

        validated: ValidatedMedia
        extracted: ExtractedAudio | None = None
        preprocessed: PreprocessedAudio | None = None
        try:
            t1_start = time.perf_counter()
            _emit("T01_MEDIA_VALIDATION")

            safe_resolved: Path = pm.resolve_input_path(input_path)

            allowed = set(getattr(safety_cfg, "allowed_extensions_set", set()))
            if not allowed:
                allowed = {f.value.lower() for f in SUPPORTED_MEDIA_FORMATS}

            try:
                validated = self.validator.validate(
                    safe_resolved,
                    job_id=jid,
                    safety_cfg=safety_cfg,
                    allowed_formats=allowed,
                    probe_bin_path=ffprobe_bin or ffmpeg_cfg.ffprobe_bin,
                    probe_timeout_s=30,
                    logger=log,
                )
            except (
                ValidationFailedError,
                AudioExtractionError,
                ConfigurationError,
            ):
                raise

            times["validate"] = round(time.perf_counter() - t1_start, 6)

            t2_start = time.perf_counter()
            _emit("T02_AUDIO_EXTRACTION")
            extracted = self.extractor.extract(
                validated=validated,
                job_temp_root=job_temp_root,
                ffmpeg_bin_path=ffmpeg_bin or ffmpeg_cfg.ffmpeg_bin,
                ffmpeg_cfg=ffmpeg_cfg,
                logger=log,
            )
            times["extract"] = round(time.perf_counter() - t2_start, 6)

            t3_start = time.perf_counter()
            _emit("T03_AUDIO_PREPROCESSING")
            preprocessed = self.preprocessor.preprocess(
                extracted=extracted,
                job_temp_root=job_temp_root,
                ffmpeg_bin_path=ffmpeg_bin or ffmpeg_cfg.ffmpeg_bin,
                ffmpeg_cfg=ffmpeg_cfg,
                logger=log,
            )
            times["preprocess"] = round(time.perf_counter() - t3_start, 6)

        finally:
            log.debug(
                "media_prep_finished_keep_temp_for_downstream",
                job_id=jid,
                force_keep_temp=bool(force_keep_temp),
            )

        total = round(time.perf_counter() - t0, 6)
        times.setdefault("total_wall", total)

        return MediaPrepResult(
            job_id=jid,
            profile=profile,
            validated=validated,
            extracted_audio=extracted,
            preprocessed_audio=preprocessed,
            step_times_sec=dict(times),
            ffmpeg_available=bool(ffmpeg_ok),
            ffprobe_available=bool(ffprobe_ok),
            errors=list(errors_collected),
        )


__all__ = [
    "AnalyzeMediaPrepUseCase",
]
