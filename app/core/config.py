"""Configuracion central del motor IA-Doblaje (T02 implementacion completa).

Implementa AppSettings (Pydantic BaseSettings v2) con 9 sub-modelos de configuracion,
validators estrictos, paths relativos resueltos desde PROJECT_ROOT, SecretStr para HF_TOKEN
y la politica CRITICA PUNTO 2: GPU_AUTO_DOWNGRADE_ENABLED = False por defecto.

Regla de No Sobreingenieria:
  - Solo Pydantic + Pydantic-Settings + stdlib.
  - No dependencias de YAML en este modulo (perfiles YAML se cargan via PathManager T02.6
    si el usuario lo explicita, pero AppSettings solo lee .env/env vars).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import (
    DEFAULT_BIT_DEPTH,
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ,
    DeviceType,
    OUTPUT_FILES_DEFAULT,
    QualityProfile,
    SPEAKER_LABEL_REGEX,
)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Sub-modelos de configuracion (ordenados por dominio funcional)
# ---------------------------------------------------------------------------


class GeneralConfig(BaseModel):
    """Variables generales de la aplicacion."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = Field("IA Doblaje Engine", description="Nombre mostrado en banners/logs.")
    app_env: Literal["development", "staging", "production"] = Field(
        "development", description="Entorno de ejecucion."
    )
    app_debug: bool = Field(True, description="Modo debug (mostrar tracebacks extendidos).")
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO")
    app_log_format: Literal["human", "json"] = Field(
        "human", description="Formato de salida de logs: human (colored) o json (ndjson)."
    )
    app_job_id_prefix: str = Field(
        "job", description="Prefijo para job IDs autogenerados: job-<uuid7>..."
    )
    dev_fix_random_seed: int | None = Field(
        42, description="Seed determinista para tests/dev. None = no fijar."
    )
    dev_keep_temp_files: bool = Field(
        False, description="Si True, limpieza automatica de temp dirs se desactiva (debug)."
    )
    dev_mock_asr_enabled: bool = Field(False)
    dev_mock_diarization_enabled: bool = Field(False)
    dev_mock_lang_detect_code: str = Field("es")


class PathsConfig(BaseModel):
    """Directorios de trabajo del proyecto.

    Todos los paths relativos se resuelven contra PROJECT_ROOT.
    """

    model_config = ConfigDict(extra="forbid")

    data_input_dir: Path = Field(Path("data/input"))
    data_output_dir: Path = Field(Path("data/output"))
    data_temp_dir: Path = Field(Path("data/temporary"))
    models_cache_dir: Path = Field(Path("models"))
    configs_dir: Path = Field(Path("configs"))
    log_dir: Path = Field(Path("data/output/logs"))

    @model_validator(mode="after")
    def _resolve_all_absolute(self) -> "PathsConfig":
        for fname in self.model_fields.keys():
            raw = getattr(self, fname)
            p = Path(raw) if not isinstance(raw, Path) else raw
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            else:
                p = p.resolve()
            object.__setattr__(self, fname, p)
        return self

    @property
    def as_absolute_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.model_dump(mode="python").items()}


class ProcessingConfig(BaseModel):
    """Perfil + dispositivo + variables de procesamiento general."""

    model_config = ConfigDict(extra="forbid")

    profile: QualityProfile = Field(
        QualityProfile.BALANCED,
        description="Perfil calidad/velocidad (quality | balanced | performance).",
    )
    device: DeviceType = Field(
        DeviceType.AUTO, description="Dispositivo inferencia (auto | cpu | cuda | mps)."
    )
    force_cpu: bool = Field(
        False,
        description="Si True, DeviceDetector ignora CUDA/MPS y retorna CPU siempre (debug).",
    )

    @field_validator("profile", mode="before")
    @classmethod
    def _normalize_profile(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return v  # dejar que pydantic maneje vacío (default)
            return s.lower()
        return v

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, v: Any) -> Any:
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return v
            return s.lower()
        return v


class GPUConfig(BaseModel):
    """Politicas GPU / VRAM.

    POLITICA CRITICA PUNTO 2:
      - GPU_AUTO_DOWNGRADE_ENABLED default = False  (NO downgrade silencioso NUNCA).
      - validate_profile_vram_capability() en DeviceDetector raisea ProfileDowngradeRequired
        status_code=412 si la GPU no cumple VRAM del perfil.
    """

    model_config = ConfigDict(extra="forbid")

    cuda_visible_devices: str = Field("")
    allow_mps: bool = Field(True, description="Permitir MPS backend en macOS Apple Silicon.")
    detect_device_verbose: bool = Field(True)
    vram_headroom_mb: int = Field(
        800, gt=0, description="Reserva de seguridad VRAM para evitar OOM en MB."
    )
    gpu_auto_downgrade_enabled: bool = Field(
        False,
        description=(
            "POLITICA PUNTO 2 (DEFECTO FALSE). "
            "Si False -> aborta con ProfileDowngradeRequired(412) cuando la GPU "
            "no tiene VRAM suficiente para el perfil solicitado. "
            "Si True  -> permite downgrade automatico al perfil de menor precision "
            "que quepa en VRAM. SIEMPRE se registra warning + downgrade flag en Metrics."
        ),
    )


class ProfileVramConfig(BaseModel):
    """VRAM minima requerida (GB) por cada perfil de calidad.

    Diseño PUNTO 6:
      QUALITY = 8 GB   (Whisper-PyTorch large-v3 FP16)
      BALANCED = 5 GB  (Whisper-PyTorch medium FP16)
      PERFORMANCE = 3 GB (Faster-Whisper large-v3 int8)
    """

    model_config = ConfigDict(extra="forbid")

    quality_required_gb: float = Field(8.0, gt=0)
    balanced_required_gb: float = Field(5.0, gt=0)
    performance_required_gb: float = Field(3.0, gt=0)

    def required_for(self, profile: QualityProfile | str) -> float:
        p = QualityProfile(profile) if isinstance(profile, str) else profile
        if p is QualityProfile.QUALITY:
            return self.quality_required_gb
        if p is QualityProfile.BALANCED:
            return self.balanced_required_gb
        return self.performance_required_gb


class FFmpegConfig(BaseModel):
    """Binarios FFmpeg/FFprobe + audio extraction defaults."""

    model_config = ConfigDict(extra="forbid")

    ffmpeg_bin: str = Field("ffmpeg")
    ffprobe_bin: str = Field("ffprobe")
    ffmpeg_timeout_sec: int = Field(7200, ge=10, description="Timeout max para procesar 2h de video.")
    ffmpeg_build_conf_label: Literal["LGPL", "GPL", "UNKNOWN"] = Field(
        "LGPL",
        description="Etiqueta documentando build usado. REVISION LEGAL si se redistribuye binario GPL.",
    )
    target_sample_rate_hz: int = Field(DEFAULT_SAMPLE_RATE_HZ, ge=8000, le=96000)
    target_channels: int = Field(DEFAULT_CHANNELS, ge=1, le=2)
    target_bit_depth: int = Field(DEFAULT_BIT_DEPTH, ge=8, le=32)
    target_format: Literal["wav", "flac", "mp3"] = Field("wav")


class SafetyConfig(BaseModel):
    """Validaciones seguridad de entrada multimedia."""

    model_config = ConfigDict(extra="forbid")

    max_media_size_gb: float = Field(10.0, gt=0.1, description="Tamaño maximo de input en GB.")
    allowed_extensions: str = Field(
        "mp4,mkv,mov,wav,mp3,m4a",
        description="Extensiones permitidas (coma separadas, sin punto).",
    )
    require_path_within_data_dir: bool = Field(
        True, description="Bloquear paths que escapen del directorio data/ (anti traversal)."
    )
    enable_auto_temp_cleanup: bool = Field(
        True, description="Borrar job temp dirs al finalizar pipeline (exito o error)."
    )

    @property
    def allowed_extensions_set(self) -> set[str]:
        return {ext.strip().lower().lstrip(".") for ext in self.allowed_extensions.split(",") if ext.strip()}


class HFConfig(BaseModel):
    """HuggingFace Hub (modelos gated pyannote / whisper weights en cache)."""

    model_config = ConfigDict(extra="forbid")

    hf_token: SecretStr = Field(
        SecretStr(""), description="Token HuggingFace. USAR SecretStr, NO imprimir nunca en logs."
    )


class QualityConfig(BaseModel):
    """Quality Analysis Module (implementado T08). Aqui solo thresholds default.

    Reglas QR01-QR11 en docs/QUALITY_RULES.md.
    """

    model_config = ConfigDict(extra="forbid")

    quality_enabled: bool = Field(True)
    quality_strict_mode: bool = Field(
        True, description="Si True, cualquier status=failed bloquea guardado (a menos --force-save)."
    )
    quality_min_score_to_save: float = Field(0.30, ge=0.0, le=1.0)
    quality_penalty_low_conf_word: float = Field(0.005, ge=0.0, le=1.0)
    quality_penalty_low_conf_segment: float = Field(0.04, ge=0.0, le=1.0)


class MetricsConfig(BaseModel):
    """Metricas T09. Flags default para tracking de rendimiento/memoria."""

    model_config = ConfigDict(extra="forbid")

    metrics_enabled: bool = Field(True)
    metrics_track_step_timings: bool = Field(True)
    metrics_track_memory_peak: bool = Field(True)
    metrics_track_gpu_util: bool = Field(True)
    metrics_poll_interval_ms: int = Field(250, gt=0)


class OutputConfig(BaseModel):
    """Formato y archivos de salida JSON por pipeline."""

    model_config = ConfigDict(extra="forbid")

    output_json_indent: int = Field(2, ge=0, le=8)
    output_json_ensure_ascii: bool = Field(
        False, description="False = unicode (acentos español preservado en JSON)."
    )
    output_files: tuple[str, ...] = Field(OUTPUT_FILES_DEFAULT)
    output_include_schema_version: bool = Field(True)

    @field_validator("output_files", mode="before")
    @classmethod
    def _split_csv_if_str(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, str):
            return tuple(x.strip() for x in v.split(",") if x.strip())
        if isinstance(v, (list, tuple)):
            return tuple(x for x in v if x)
        return OUTPUT_FILES_DEFAULT


class PipelineConfig(BaseModel):
    """Fail-safes globales del pipeline 14 pasos."""

    model_config = ConfigDict(extra="forbid")

    pipeline_max_duration_sec: int = Field(
        28_800, ge=60, description="Timeout absoluto pipeline (8h default)."
    )
    pipeline_abort_on_quality_failed: bool = Field(True)
    pipeline_abort_on_asr_failed: bool = Field(True)
    pipeline_skip_diarization_on_error: bool = Field(False)


class TranslationConfig(BaseModel):
    """Configuración T15 — Translation Engine (Step 15).

    Motor: facebook/m2m100_418M local, offline.
    Idiomas oficiales: es, en, fr, de, it, pt, ja, zh.
    """

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    enabled: bool = Field(True, description="Activar T15 en el pipeline.")
    target_language: str = Field(
        "es", description="Idioma destino por defecto (uno de los 8 oficiales)."
    )
    model_dir: str = Field(
        "models/m2m100_418M", description="Ruta relativa al modelo M2M100 local."
    )
    max_length: int = Field(128, ge=16, le=1024, description="Máx tokens generados.")
    num_beams: int = Field(1, ge=1, le=8, description="Beams (1 = greedy).")
    do_sample: bool = Field(False, description="Sampling (False = determinista).")
    max_chars_per_second: float = Field(
        18.0, gt=0.0, le=100.0, description="Velocidad de locución doblada (cps)."
    )
    max_adaptation_ratio: float = Field(
        1.35, gt=0.0, le=5.0, description="Ratio máx expansión adapted/source."
    )

    @field_validator("target_language")
    @classmethod
    def _target_valid(cls, v: object) -> str:
        s = str(v).strip().lower()
        if s not in {"es", "en", "fr", "de", "it", "pt", "ja", "zh"}:
            raise ValueError(
                f"translation.target_language inválido: {s!r}. "
                "Oficiales: es,en,fr,de,it,pt,ja,zh."
            )
        return s


class LoggingConfig(BaseModel):
    """StructLog logging (T02.3 implementacion) thresholds + rotacion."""

    model_config = ConfigDict(extra="forbid")

    log_dir: Path = Field(Path("data/output/logs"))
    log_rotation: str = Field("100 MB", description="Rotacion archivo: 100 MB / 1 day / ...")
    log_retention: str = Field("30 days")
    log_include_process_id: bool = Field(True)
    log_include_job_id: bool = Field(True)

    @field_validator("log_dir", mode="before")
    @classmethod
    def _resolve(cls, v: str | os.PathLike[str] | None) -> Path:
        p = Path(v) if v else Path("data/output/logs")
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()


# ---------------------------------------------------------------------------
# AppSettings principal
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Configuracion central completa del motor.

    Lee variables de:
      1. Argumentos constructor (tests inyeccion)
      2. .env file en PROJECT_ROOT si existe
      3. OS environment variables

    SENSIBLE DEFAULTS:
      - profile = balanced
      - device = auto
      - gpu_auto_downgrade_enabled = False  (POLITICA PUNTO 2 CRITICA)
      - quality_strict_mode = True
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # ---- sub-configs ----
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    profile_vram: ProfileVramConfig = Field(default_factory=ProfileVramConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    hf: HFConfig = Field(default_factory=HFConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    logging_cfg: LoggingConfig = Field(default_factory=LoggingConfig)

    # ---- backward compat flat validators: algunos usuarios pasan vars flat por env ----

    @model_validator(mode="before")
    @classmethod
    def _flatten_env_to_nested(cls, data: dict[str, object]) -> dict[str, object]:
        """Acepta variables flat estilo PROFILE o DATA_INPUT_DIR y las mapea a nested:
        ej:  PROFILE=quality  -> processing.profile
        """
        if not isinstance(data, dict):
            return data

        flat_map: dict[str, tuple[str, str]] = {
            "APP_NAME": ("general", "app_name"),
            "APP_ENV": ("general", "app_env"),
            "APP_DEBUG": ("general", "app_debug"),
            "APP_LOG_LEVEL": ("general", "app_log_level"),
            "APP_LOG_FORMAT": ("general", "app_log_format"),
            "APP_JOB_ID_PREFIX": ("general", "app_job_id_prefix"),
            "DEV_FIX_RANDOM_SEED": ("general", "dev_fix_random_seed"),
            "DEV_KEEP_TEMP_FILES": ("general", "dev_keep_temp_files"),
            "DEV_MOCK_ASR_ENABLED": ("general", "dev_mock_asr_enabled"),
            "DEV_MOCK_DIARIZATION_ENABLED": ("general", "dev_mock_diarization_enabled"),
            "DEV_MOCK_LANG_DETECT_CODE": ("general", "dev_mock_lang_detect_code"),
            "DATA_INPUT_DIR": ("paths", "data_input_dir"),
            "DATA_OUTPUT_DIR": ("paths", "data_output_dir"),
            "DATA_TEMP_DIR": ("paths", "data_temp_dir"),
            "MODELS_CACHE_DIR": ("paths", "models_cache_dir"),
            "CONFIGS_DIR": ("paths", "configs_dir"),
            "PROFILE": ("processing", "profile"),
            "DEVICE": ("processing", "device"),
            "FORCE_CPU": ("processing", "force_cpu"),
            "CUDA_VISIBLE_DEVICES": ("gpu", "cuda_visible_devices"),
            "ALLOW_MPS": ("gpu", "allow_mps"),
            "DETECT_DEVICE_VERBOSE": ("gpu", "detect_device_verbose"),
            "VRAM_HEADROOM_MB": ("gpu", "vram_headroom_mb"),
            "GPU_AUTO_DOWNGRADE_ENABLED": ("gpu", "gpu_auto_downgrade_enabled"),
            "FFMPEG_BIN": ("ffmpeg", "ffmpeg_bin"),
            "FFPROBE_BIN": ("ffmpeg", "ffprobe_bin"),
            "FFMPEG_TIMEOUT_SEC": ("ffmpeg", "ffmpeg_timeout_sec"),
            "FFMPEG_BUILD_CONF_LABEL": ("ffmpeg", "ffmpeg_build_conf_label"),
            "TARGET_SAMPLE_RATE_HZ": ("ffmpeg", "target_sample_rate_hz"),
            "TARGET_CHANNELS": ("ffmpeg", "target_channels"),
            "TARGET_BIT_DEPTH": ("ffmpeg", "target_bit_depth"),
            "TARGET_FORMAT": ("ffmpeg", "target_format"),
            "MAX_MEDIA_SIZE_GB": ("safety", "max_media_size_gb"),
            "ALLOWED_EXTENSIONS": ("safety", "allowed_extensions"),
            "REQUIRE_PATH_WITHIN_DATA_DIR": ("safety", "require_path_within_data_dir"),
            "ENABLE_AUTO_TEMP_CLEANUP": ("safety", "enable_auto_temp_cleanup"),
            "HF_TOKEN": ("hf", "hf_token"),
            "QUALITY_ENABLED": ("quality", "quality_enabled"),
            "QUALITY_STRICT_MODE": ("quality", "quality_strict_mode"),
            "QUALITY_MIN_SCORE_TO_SAVE": ("quality", "quality_min_score_to_save"),
            "QUALITY_PENALTY_LOW_CONF_WORD": ("quality", "quality_penalty_low_conf_word"),
            "QUALITY_PENALTY_LOW_CONF_SEGMENT": ("quality", "quality_penalty_low_conf_segment"),
            "METRICS_ENABLED": ("metrics", "metrics_enabled"),
            "METRICS_TRACK_STEP_TIMINGS": ("metrics", "metrics_track_step_timings"),
            "METRICS_TRACK_MEMORY_PEAK": ("metrics", "metrics_track_memory_peak"),
            "METRICS_TRACK_GPU_UTIL": ("metrics", "metrics_track_gpu_util"),
            "METRICS_POLL_INTERVAL_MS": ("metrics", "metrics_poll_interval_ms"),
            "OUTPUT_JSON_INDENT": ("output", "output_json_indent"),
            "OUTPUT_JSON_ENSURE_ASCII": ("output", "output_json_ensure_ascii"),
            "OUTPUT_FILES": ("output", "output_files"),
            "OUTPUT_INCLUDE_SCHEMA_VERSION": ("output", "output_include_schema_version"),
            "PIPELINE_MAX_DURATION_SEC": ("pipeline", "pipeline_max_duration_sec"),
            "PIPELINE_ABORT_ON_QUALITY_FAILED": ("pipeline", "pipeline_abort_on_quality_failed"),
            "PIPELINE_ABORT_ON_ASR_FAILED": ("pipeline", "pipeline_abort_on_asr_failed"),
            "PIPELINE_SKIP_DIARIZATION_ON_ERROR": ("pipeline", "pipeline_skip_diarization_on_error"),
            "TRANSLATION_ENABLED": ("translation", "enabled"),
            "TRANSLATION_TARGET_LANGUAGE": ("translation", "target_language"),
            "TRANSLATION_MODEL_DIR": ("translation", "model_dir"),
            "TRANSLATION_MAX_LENGTH": ("translation", "max_length"),
            "TRANSLATION_NUM_BEAMS": ("translation", "num_beams"),
            "TRANSLATION_DO_SAMPLE": ("translation", "do_sample"),
            "TRANSLATION_MAX_CHARS_PER_SECOND": ("translation", "max_chars_per_second"),
            "TRANSLATION_MAX_ADAPTATION_RATIO": ("translation", "max_adaptation_ratio"),
            "LOG_DIR": ("logging_cfg", "log_dir"),
            "LOG_ROTATION": ("logging_cfg", "log_rotation"),
            "LOG_RETENTION": ("logging_cfg", "log_retention"),
            "LOG_INCLUDE_PROCESS_ID": ("logging_cfg", "log_include_process_id"),
            "LOG_INCLUDE_JOB_ID": ("logging_cfg", "log_include_job_id"),
        }

        nested: dict[str, dict[str, object]] = {}
        # Garantizamos lectura de OS env vars flat independientemente de
        # si SettingsConfigDict las entrega a este validator.
        import os as _os
        for key in list(dict(data).keys()) + list(_os.environ.keys()):
            up = str(key).upper()
            if up in flat_map:
                group_name, field_name = flat_map[up]
                # Si ya vino en data y esta nested -> no sobrescribir
                if isinstance(data.get(group_name), dict):
                    if field_name in data[group_name]:  # type: ignore[operator]
                        continue
                # Si data ya contenia flat key -> usarla, sino OS env
                if key in data and data[key] not in (None, ""):
                    raw_value: object = data[key]
                else:
                    raw_value = _os.environ.get(key, "")
                if raw_value in (None, "") and up not in {k.upper() for k in data.keys()}:
                    continue
                if raw_value == "" and up not in {
                    "HF_TOKEN",
                }:
                    continue
                nested.setdefault(group_name, {})[field_name] = raw_value  # type: ignore[assignment,index]
        for k, v in nested.items():
            if k not in data or data[k] in (None, ""):
                data[k] = v  # type: ignore[assignment]
            elif isinstance(data[k], dict):
                data[k].update(v)  # type: ignore[union-attr]
        return data

    def dump_safe(self) -> dict[str, object]:
        """Dump config sin secrets (HF_TOKEN) / passwords. Usar para logging/diagnose."""
        d = self.model_dump(mode="python")
        if isinstance(d.get("hf"), dict) and "hf_token" in d["hf"]:
            tok: SecretStr | None = d["hf"].get("hf_token")
            if isinstance(tok, SecretStr):
                d["hf"]["hf_token"] = "***HIDDEN_SECRET_STR***" if tok.get_secret_value() else ""
            else:
                d["hf"]["hf_token"] = (
                    "***HIDDEN_SECRET_STR***" if bool(tok) else ""
                )
        return d


# ---------------------------------------------------------------------------
# Publico: accessor singleton con caching (testeable via monkeypatch env)
# ---------------------------------------------------------------------------


def get_settings() -> AppSettings:
    """Retorna settings singleton cargado desde .env / env vars.

    Nota: No usamos @lru_cache() global para tests (monkeypatch.setenv no
    refresca cache). En tests llamamos AppSettings() directamente o limpiamos
    cache via `get_settings.cache_clear()`.
    """
    return _get_settings_cached()


@lru_cache(maxsize=1)
def _get_settings_cached() -> AppSettings:
    return AppSettings()


def clear_settings_cache() -> None:
    """Helper para tests / reload manual de configuracion."""
    _get_settings_cached.cache_clear()


__all__ = [
    "PROJECT_ROOT",
    "SPEAKER_LABEL_REGEX",
    "AppSettings",
    "GeneralConfig",
    "PathsConfig",
    "ProcessingConfig",
    "GPUConfig",
    "ProfileVramConfig",
    "FFmpegConfig",
    "SafetyConfig",
    "HFConfig",
    "QualityConfig",
    "MetricsConfig",
    "OutputConfig",
    "PipelineConfig",
    "LoggingConfig",
    "get_settings",
    "clear_settings_cache",
]
