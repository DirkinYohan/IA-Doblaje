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


def _validate_score_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("QualityScore no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(
            f"QualityScore requiere float/int, recibido {type(v).__name__}"
        )
    x = float(v)
    if math.isnan(x):
        raise ValueError("QualityScore no admite NaN")
    if math.isinf(x):
        raise ValueError("QualityScore no admite Inf")
    if x < 0.0:
        raise ValueError(f"QualityScore < 0.0: {x}")
    if x > 1.0:
        raise ValueError(f"QualityScore > 1.0: {x}")
    return x


QualityScore = Annotated[float, BeforeValidator(_validate_score_0_1)]


def _validate_non_negative_int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"QualityCount requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"QualityCount no admite negativos: {v}")
    return int(v)


QualityCount = Annotated[int, BeforeValidator(_validate_non_negative_int)]


def _validate_non_negative_float(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("QualityFloat no admite bool")
    if not isinstance(v, (int, float)):
        raise ValueError(f"QualityFloat requiere float/int, recibido {type(v).__name__}")
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("QualityFloat no admite NaN/Inf")
    if x < 0.0:
        raise ValueError(f"QualityFloat < 0.0: {x}")
    return x


QualityNonNegativeFloat = Annotated[float, BeforeValidator(_validate_non_negative_float)]


QualitySeverity = Literal["ERROR", "WARNING"]

QualityRuleCode = Literal[
    "QR01", "QR02", "QR03", "QR04", "QR05", "QR06",
    "QR07", "QR08", "QR09", "QR10", "QR11",
]

QualityStrategyName = Literal["quality_rules_qr01_qr11", "quality_silent_fallback"]


class QualityRuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_code: QualityRuleCode = Field(..., description="Código de regla QR01..QR11.")
    severity: QualitySeverity = Field(..., description="ERROR o WARNING.")
    passed: bool = Field(..., description="True si la regla no reporta violación.")
    applicable: bool = Field(
        True, description="False si el dato requerido no está disponible."
    )
    occurrences: QualityCount = Field(0, description="Número de violaciones detectadas.")
    penalty: QualityNonNegativeFloat = Field(
        0.0, description="Penalización aplicada al score (0 si not applicable)."
    )
    reason: str = Field(
        "", max_length=512, description="Mensaje/razón determinista."
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _reason_str(cls, v: object) -> str:
        return str(v or "")

    @model_validator(mode="after")
    def _consistency(self) -> "QualityRuleResult":
        if not self.applicable:
            if self.occurrences != 0:
                raise ValueError("not applicable implica occurrences=0")
            if self.penalty != 0.0:
                raise ValueError("not applicable implica penalty=0")
        return self


class QualityThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_threshold: QualityScore = Field(0.80, description="Umbral para status=passed.")
    warn_threshold: QualityScore = Field(0.30, description="Umbral para status=warning.")
    max_overlap_ms: int = Field(100, ge=0, le=1_000_000, description="QR04 solape máximo permitido.")
    min_asr_segment_conf: QualityScore = Field(0.40, description="QR07 umbral confianza segmento.")
    min_word_conf: QualityScore = Field(0.30, description="QR08 umbral confianza palabra.")
    max_segment_sec: float = Field(60.0, gt=0, description="QR09 duración máxima segmento.")
    max_silence_sec: float = Field(15.0, gt=0, description="QR10 silencio máximo inter-speech.")
    penalty_low_conf_segment: QualityScore = Field(
        0.04, description="Penalización QR07 por segmento de baja confianza (config.py default 0.04)."
    )
    penalty_low_conf_word: QualityScore = Field(
        0.005, description="Penalización QR08 por palabra de baja confianza (config.py default 0.005)."
    )

    @field_validator("max_overlap_ms", mode="before")
    @classmethod
    def _reject_bool_int(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("No se permite bool donde se espera int (T10).")
        return v

    @field_validator(
        "max_segment_sec",
        "max_silence_sec",
        "penalty_low_conf_segment",
        "penalty_low_conf_word",
        mode="before",
    )
    @classmethod
    def _reject_bool_float(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("No se permite bool donde se espera float (T10).")
        return v

    @model_validator(mode="after")
    def _pass_gt_warn(self) -> "QualityThresholds":
        if not self.pass_threshold > self.warn_threshold:
            raise ValueError("pass_threshold debe ser > warn_threshold")
        return self


__all__ = [
    "QualityScore",
    "QualityCount",
    "QualityNonNegativeFloat",
    "QualitySeverity",
    "QualityRuleCode",
    "QualityStrategyName",
    "QualityRuleResult",
    "QualityThresholds",
]
