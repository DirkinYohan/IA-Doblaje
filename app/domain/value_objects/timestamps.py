from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, ConfigDict, model_validator
from pydantic import BaseModel as _PydanticBase

# ---------------------------------------------------------------------------
# Helpers reutilizables
# ---------------------------------------------------------------------------
def _reject_bool(v: Any) -> Any:
    if isinstance(v, bool):
        raise TypeError("No se permite coerción bool → int/float en VO T07.")
    return v


def _validate_non_negative_int(v: Any) -> int:
    _reject_bool(v)
    if isinstance(v, int):
        value = v
    else:
        try:
            value = int(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Se esperaba int >=0, got {type(v).__name__}: {v!r}") from exc
    if value < 0:
        raise ValueError(f"Valor negativo no permitido: {value}")
    return value


def _validate_non_negative_float01(v: Any) -> float:
    _reject_bool(v)
    try:
        value = float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Se esperaba float [0,1], got {type(v).__name__}: {v!r}") from exc
    import math
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"No se permite NaN/Inf: {v!r}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Valor fuera de [0,1]: {value}")
    return value


# ---------------------------------------------------------------------------
# Timestamp Value Objects frozen forbid defaults
# ---------------------------------------------------------------------------
TimedStartMs = Annotated[int, BeforeValidator(_validate_non_negative_int)]
TimedEndMs = Annotated[int, BeforeValidator(_validate_non_negative_int)]
DurationMs = Annotated[int, BeforeValidator(_validate_non_negative_int)]
TimestampConfidence01 = Annotated[float, BeforeValidator(_validate_non_negative_float01)]

TimestampStrategyName = Literal[
    "segment_ts_from_fw_raw",
    "segment_ts_from_vad_interpolate",
    "ts_silent_fallback",
]

TimestampSourceKind = Literal[
    "fw_raw",
    "vad_interpolate",
    "heuristic_gap_fill",
    "silent",
]

WordTimestampSourceKind = Literal["none"]

TimestampedWordIndex = Annotated[int, BeforeValidator(_validate_non_negative_int)]
TimestampedSegmentIndex = Annotated[int, BeforeValidator(_validate_non_negative_int)]


# ---------------------------------------------------------------------------
# AsrTimestampThresholds: VO frozen defaults configurables T07
# ---------------------------------------------------------------------------
class AsrTimestampThresholds(_PydanticBase):
    """Umbrales configurables para Step 07 Timestamp Generation.
    frozen + forbid. Sin coerción bool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_segments_safety: int = 1_000_000
    allow_overlap_resolution: bool = True
    allow_end_clamp_to_total: bool = True
    allow_vad_interpolate_fallback: bool = True
    strict_validity: bool = True
    allow_silent_fallback: bool = True
    min_segment_duration_ms: int = 10

    @model_validator(mode="before")
    @classmethod
    def _pre(cls, v: Any) -> Any:
        if isinstance(v, dict):
            for k in (
                "max_segments_safety",
                "allow_overlap_resolution",
                "allow_end_clamp_to_total",
                "allow_vad_interpolate_fallback",
                "strict_validity",
                "allow_silent_fallback",
                "min_segment_duration_ms",
            ):
                if k not in v:
                    continue
                val = v[k]
                if k.startswith("allow_") or k == "strict_validity":
                    if not isinstance(val, bool):
                        raise ValueError(f"{k} debe ser bool, got {type(val).__name__}")
                elif k == "max_segments_safety" or k == "min_segment_duration_ms":
                    if isinstance(val, bool) or not isinstance(val, int):
                        raise ValueError(f"{k} debe ser int no-bool")
                    if val <= 0:
                        raise ValueError(f"{k} debe ser >0")
        return v
