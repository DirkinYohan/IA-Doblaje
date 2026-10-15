"""Value Objects Metrics Step 12 — Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (sin IA, sin psutil/torch).
"""
from __future__ import annotations

import math
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator

from app.core.constants import PipelineStep

# 14 keys exactas de PipelineStep (identificadores .name)
PIPELINE_STEP_NAMES: tuple[str, ...] = tuple(s.name for s in PipelineStep)
_ALLOWED_STEP_NAMES = frozenset(PIPELINE_STEP_NAMES)


# ---------------------------------------------------------------------------
# 1) Value Objects escalares
# ---------------------------------------------------------------------------


def _validate_non_negative_float(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("MetricFloat no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(f"MetricFloat requiere float/int, recibido {type(v).__name__}")
    x = float(v)
    if math.isnan(x):
        raise ValueError("MetricFloat no admite NaN")
    if math.isinf(x):
        raise ValueError("MetricFloat no admite Inf")
    if x < 0.0:
        raise ValueError(f"MetricFloat no admite negativos: {x}")
    return x


MetricNonNegativeFloat = Annotated[float, BeforeValidator(_validate_non_negative_float)]


def _validate_percent_0_100(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("Percent no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(f"Percent requiere float/int, recibido {type(v).__name__}")
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("Percent no admite NaN/Inf")
    if x < 0.0 or x > 100.0:
        raise ValueError(f"Percent fuera de rango [0,100]: {x}")
    return x


MetricPercent = Annotated[float, BeforeValidator(_validate_percent_0_100)]


def _validate_non_negative_int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"MetricCount requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"MetricCount no admite negativos: {v}")
    return int(v)


MetricCount = Annotated[int, BeforeValidator(_validate_non_negative_int)]


def _validate_score_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("MetricConfidence no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(f"MetricConfidence requiere float/int, recibido {type(v).__name__}")
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("MetricConfidence no admite NaN/Inf")
    if x < 0.0 or x > 1.0:
        raise ValueError(f"MetricConfidence fuera de [0,1]: {x}")
    return x


MetricConfidence = Annotated[float, BeforeValidator(_validate_score_0_1)]


# ---------------------------------------------------------------------------
# 2) RuntimeMetricsSnapshot (snapshot YA medido por Infrastructure/orquestador)
# ---------------------------------------------------------------------------


class RuntimeMetricsSnapshot(BaseModel):
    """Snapshot de métricas de runtime, medido por Infrastructure/orquestador.

    T12 NO mide CPU/GPU/RAM; solo agrega lo ya medido.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    processing_wall_time_sec: MetricNonNegativeFloat = Field(..., description="Tiempo wall-clock total.")
    step_timings_sec: dict[str, float] = Field(
        default_factory=dict, description="Tiempos por cada uno de los 14 PipelineStep (keys exactas)."
    )
    peak_cpu_rss_mb: MetricNonNegativeFloat = Field(0.0, description="Peak RAM (MB).")
    peak_gpu_vram_used_mb: MetricNonNegativeFloat = Field(0.0, description="Peak VRAM usado (MB).")
    peak_gpu_vram_total_mb: MetricNonNegativeFloat = Field(0.0, description="VRAM total (MB).")
    cpu_utilization_avg_percent: MetricPercent = Field(0.0, description="CPU promedio %.")
    gpu_utilization_avg_percent: MetricPercent = Field(0.0, description="GPU promedio %.")

    # Compliance (del orquestador, no deducido)
    profile_requested: str = Field("", description="Perfil solicitado.")
    profile_applied: str = Field("", description="Perfil aplicado.")
    profile_downgrade_applied: bool = Field(False, description="True si hubo downgrade explícito.")
    profile_downgrade_reason: str = Field("", description="Razón del downgrade (si aplica).")
    device_requested: str = Field("", description="Dispositivo solicitado.")
    device_used: str = Field("", description="Dispositivo usado.")

    @field_validator("step_timings_sec", mode="before")
    @classmethod
    def _validate_step_timings(cls, v: object) -> dict[str, float]:
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
            if math.isnan(x) or math.isinf(x):
                raise ValueError(f"step_timings_sec[{k!r}] no admite NaN/Inf")
            if x < 0.0:
                raise ValueError(f"step_timings_sec[{k!r}] negativo")
            out[k] = x
        return out


__all__ = [
    "PIPELINE_STEP_NAMES",
    "MetricNonNegativeFloat",
    "MetricPercent",
    "MetricCount",
    "MetricConfidence",
    "RuntimeMetricsSnapshot",
]
