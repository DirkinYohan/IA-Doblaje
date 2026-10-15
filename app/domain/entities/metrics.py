"""Entidad MetricsResult Step 12 — Metrics Aggregation. Capa DOMAIN.

Frozen + extra=forbid. Chain of custody 2-SHA:
  source_preprocessed_sha256 (T03)
  validation_reference_sha256 (T11)

Sin dependencias infraestructura (no psutil/torch/numpy).
"""
from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.core.constants import PipelineStep
from app.domain.value_objects.audio import Sha256Hash
from app.domain.value_objects.metrics import (
    MetricConfidence,
    MetricCount,
    MetricNonNegativeFloat,
    MetricPercent,
)

MODEL_LABEL_METRICS_V1_LITERAL: Literal["t12:metrics-aggregation:v1"] = (
    "t12:metrics-aggregation:v1"
)


class MetricsResult(BaseModel):
    """Resultado Step 12 Metrics Aggregation. Inmutable. Chain of Custody 2-SHA."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    # ---- Step identifiers ------------------------------------------------
    step: PipelineStep = PipelineStep.METRICS_AGGREGATION
    model_label: Literal["t12:metrics-aggregation:v1"] = MODEL_LABEL_METRICS_V1_LITERAL

    # ---- Chain of Custody 2-SHA ------------------------------------------
    source_preprocessed_sha256: Sha256Hash
    validation_reference_sha256: Sha256Hash

    # ---- Estado ----------------------------------------------------------
    job_id: str | None = Field(default=None, min_length=8, max_length=128)
    metrics_enabled: bool = Field(True, description="False si la agregación fue deshabilitada.")

    # ---- TIEMPOS ---------------------------------------------------------
    processing_wall_time_sec: MetricNonNegativeFloat = Field(0.0)
    audio_duration_sec: MetricNonNegativeFloat = Field(0.0)
    real_time_factor_RTF: MetricNonNegativeFloat | None = Field(
        None, description="processing/audio. None si audio==0."
    )
    step_timings_sec: dict[str, float] = Field(default_factory=dict)

    # ---- MEMORIA ---------------------------------------------------------
    peak_cpu_rss_mb: MetricNonNegativeFloat = Field(0.0)
    peak_gpu_vram_used_mb: MetricNonNegativeFloat = Field(0.0)
    peak_gpu_vram_total_mb: MetricNonNegativeFloat = Field(0.0)

    # ---- UTILIZACIÓN -----------------------------------------------------
    cpu_utilization_avg_percent: MetricPercent = Field(0.0)
    gpu_utilization_avg_percent: MetricPercent = Field(0.0)

    # ---- CONTEOS ---------------------------------------------------------
    segment_count_total: MetricCount = Field(0)
    segment_count_with_text: MetricCount = Field(0)
    word_count_total: MetricCount = Field(0)
    word_count_confidence_ge_08: MetricCount = Field(0)
    speaker_count_detected: MetricCount = Field(0)
    silence_segment_count: MetricCount = Field(0)

    # ---- CALIDAD / CONFIANZA ---------------------------------------------
    average_asr_confidence: MetricConfidence = Field(0.0)
    average_diarization_confidence: MetricConfidence = Field(0.0)
    language_detection_confidence: MetricConfidence = Field(0.0)
    overall_quality_score: MetricConfidence = Field(0.0)

    # ---- COMPLIANCE ------------------------------------------------------
    profile_requested: str = Field("")
    profile_applied: str = Field("")
    profile_downgrade_applied: bool = Field(False)
    profile_downgrade_reason: str = Field("")
    device_requested: str = Field("")
    device_used: str = Field("")

    # ---- Auditoría -------------------------------------------------------
    analysis_metadata: dict[str, object] = Field(default_factory=dict)

    # ======================================================================
    # VALIDADORES
    # ======================================================================

    @field_validator("source_preprocessed_sha256", "validation_reference_sha256")
    @classmethod
    def _sha_64hex_lower(cls, v: object) -> str:
        s = str(v or "").strip().lower()
        if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
            raise ValueError(f"SHA inválido 64-hex lower: {s[:20]!r}...")
        return s

    @field_validator("analysis_metadata")
    @classmethod
    def _metadata_no_bytes(cls, v: object) -> dict[str, object]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("analysis_metadata debe ser dict")
        for k, val in v.items():
            if isinstance(val, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"analysis_metadata[{k!r}] contiene bytes/tensores prohibidos."
                )
        return v

    @field_validator("step_timings_sec", mode="before")
    @classmethod
    def _validate_step_timings(cls, v: object) -> dict[str, float]:
        from app.domain.value_objects.metrics import _ALLOWED_STEP_NAMES

        import math as _math

        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("step_timings_sec debe ser dict")
        out: dict[str, float] = {}
        for k, val in v.items():
            if k not in _ALLOWED_STEP_NAMES:
                raise ValueError(f"step_timings_sec key desconocida: {k!r}")
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"step_timings_sec[{k!r}] requiere float/int")
            x = float(val)
            if _math.isnan(x) or _math.isinf(x):
                raise ValueError(f"step_timings_sec[{k!r}] no admite NaN/Inf")
            if x < 0.0:
                raise ValueError(f"step_timings_sec[{k!r}] negativo")
            out[k] = x
        return out

    @model_validator(mode="after")
    def _validate_all(self) -> "MetricsResult":
        # (A) Chain of custody
        if str(self.source_preprocessed_sha256) != str(self.validation_reference_sha256):
            raise ValueError(
                "Chain of Custody rota T12: source != validation_reference."
            )
        # (B) RTF coherencia (si audio>0 y RTF presente, debe coincidir aprox)
        if self.audio_duration_sec > 0 and self.real_time_factor_RTF is not None:
            expected = round(self.processing_wall_time_sec / self.audio_duration_sec, 6)
            actual = round(float(self.real_time_factor_RTF), 6)
            if abs(expected - actual) > 1e-6:
                raise ValueError(
                    f"real_time_factor_RTF={actual} no coincide con processing/audio={expected}"
                )
        return self


__all__ = ["MetricsResult", "MODEL_LABEL_METRICS_V1_LITERAL"]
