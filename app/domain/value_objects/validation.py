"""Value Objects Validation Step 11 — Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (sin IA).
"""
from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.functional_validators import BeforeValidator


# ---------------------------------------------------------------------------
# 1) Value Objects escalares
# ---------------------------------------------------------------------------


def _validate_score_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("ValidationScore no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(f"ValidationScore requiere float/int, recibido {type(v).__name__}")
    x = float(v)
    if math.isnan(x):
        raise ValueError("ValidationScore no admite NaN")
    if math.isinf(x):
        raise ValueError("ValidationScore no admite Inf")
    if x < 0.0:
        raise ValueError(f"ValidationScore < 0.0: {x}")
    if x > 1.0:
        raise ValueError(f"ValidationScore > 1.0: {x}")
    return x


ValidationScore = Annotated[float, BeforeValidator(_validate_score_0_1)]


def _validate_non_negative_int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"ValidationCount requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"ValidationCount no admite negativos: {v}")
    return int(v)


ValidationCount = Annotated[int, BeforeValidator(_validate_non_negative_int)]


ValidationStatus = Literal["passed", "warning", "failed"]

ValidationCheckCode = Literal[
    "VC01", "VC02", "VC03", "VC04", "VC05", "VC06",
    "VC07", "VC08", "VC09", "VC10", "VC11",
]

ValidationCheckOutcome = Literal["passed", "warning", "error"]


# ---------------------------------------------------------------------------
# 2) ValidationCheckResult (resultado individual de un check)
# ---------------------------------------------------------------------------


class ValidationCheckResult(BaseModel):
    """Resultado individual de un check VC01..VC11."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_code: ValidationCheckCode = Field(..., description="Código VC01..VC11.")
    outcome: ValidationCheckOutcome = Field(..., description="passed | warning | error.")
    message: str = Field("", max_length=512, description="Mensaje determinista.")

    @field_validator("message", mode="before")
    @classmethod
    def _message_str(cls, v: object) -> str:
        return str(v or "")


# ---------------------------------------------------------------------------
# 3) ValidationThresholds (configuración centralizada T11)
# ---------------------------------------------------------------------------


class ValidationThresholds(BaseModel):
    """Umbrales de validación T11.

    strict_quality: si True y quality_level == failed → aborta salvo force_save.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    strict_quality: bool = Field(
        True, description="Si True, quality failed (sin force_save) aborta validación."
    )
    pass_threshold: ValidationScore = Field(0.80, description="Umbral passed (coherencia con T10).")
    warn_threshold: ValidationScore = Field(0.30, description="Umbral warning (coherencia con T10).")

    @field_validator("pass_threshold", "warn_threshold", mode="before")
    @classmethod
    def _reject_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("No se permite bool donde se espera float (T11).")
        return v

    @model_validator(mode="after")
    def _pass_gt_warn(self) -> "ValidationThresholds":
        if not self.pass_threshold > self.warn_threshold:
            raise ValueError("pass_threshold debe ser > warn_threshold")
        return self


__all__ = [
    "ValidationScore",
    "ValidationCount",
    "ValidationStatus",
    "ValidationCheckCode",
    "ValidationCheckOutcome",
    "ValidationCheckResult",
    "ValidationThresholds",
]
