"""Value Objects Translation Step 15 — Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura
(no transformers/torch/filesystem).

Idiomas oficiales T15 (exactamente 8):
    es, en, fr, de, it, pt, ja, zh
"""
from __future__ import annotations

import math
import re
from enum import Enum
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
# 1) Idiomas oficiales
# ---------------------------------------------------------------------------


class TranslationLanguage(str, Enum):
    """Los 8 idiomas oficiales de T15."""

    ES = "es"
    EN = "en"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    JA = "ja"
    ZH = "zh"

    @property
    def display_name(self) -> str:
        return _LANGUAGE_NAMES[self]


_LANGUAGE_NAMES: dict["TranslationLanguage", str] = {
    TranslationLanguage.ES: "Español",
    TranslationLanguage.EN: "Inglés",
    TranslationLanguage.FR: "Francés",
    TranslationLanguage.DE: "Alemán",
    TranslationLanguage.IT: "Italiano",
    TranslationLanguage.PT: "Portugués",
    TranslationLanguage.JA: "Japonés",
    TranslationLanguage.ZH: "Chino mandarín",
}

TRANSLATION_LANGUAGE_CODES: frozenset[str] = frozenset(
    lang.value for lang in TranslationLanguage
)


def _validate_language_code(v: object) -> str:
    if isinstance(v, bool) or not isinstance(v, str):
        raise TypeError(f"TranslationLanguageCode requiere str, recibido {type(v).__name__}")
    s = str(v).strip().lower()
    if s not in TRANSLATION_LANGUAGE_CODES:
        raise ValueError(
            f"Idioma no soportado T15: {s!r}. Idiomas oficiales: "
            f"{sorted(TRANSLATION_LANGUAGE_CODES)}."
        )
    return s


TranslationLanguageCode = Annotated[str, BeforeValidator(_validate_language_code)]


# ---------------------------------------------------------------------------
# 2) Value objects escalares
# ---------------------------------------------------------------------------


def _validate_ms_ge_zero(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"TranslationMs requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"TranslationMs no admite negativos: {v}")
    return int(v)


TranslationMs = Annotated[int, BeforeValidator(_validate_ms_ge_zero)]


def _validate_nonempty_text(v: object) -> str:
    if isinstance(v, bool) or not isinstance(v, str):
        raise TypeError(f"TranslationText requiere str, recibido {type(v).__name__}")
    if len(v) > 1_000_000:
        raise ValueError("TranslationText excede longitud máxima 1.000.000")
    return v


TranslationText = Annotated[str, BeforeValidator(_validate_nonempty_text)]


def _validate_float_ge0(v: object) -> float:
    if isinstance(v, bool):
        raise TypeError("TranslationFloat no admite bool")
    if not isinstance(v, (int, float)):
        raise TypeError(f"TranslationFloat requiere float/int, recibido {type(v).__name__}")
    x = float(v)
    if math.isnan(x) or math.isinf(x):
        raise ValueError("TranslationFloat no admite NaN/Inf")
    if x < 0.0:
        raise ValueError(f"TranslationFloat < 0: {x}")
    return x


TranslationFloat = Annotated[float, BeforeValidator(_validate_float_ge0)]


# ---------------------------------------------------------------------------
# 3) TranslationSegmentValidation (por segmento)
# ---------------------------------------------------------------------------


class TranslationSegmentValidation(BaseModel):
    """Validación por segmento de traducción."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = Field(True)
    errors: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


# ---------------------------------------------------------------------------
# 4) TranslationSegment (1:1 con segments.json)
# ---------------------------------------------------------------------------


class TranslationSegment(BaseModel):
    """Segmento traducido/adaptado. 1:1 con segments.json.

    Garantías:
      - segment_index >= 0
      - start_ms < end_ms
      - duration_ms == end_ms - start_ms (recomputado)
      - source_text conservado
      - translated_text no vacío si source_text no vacío
      - adapted_text no vacío si source_text no vacío
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_index: int = Field(..., ge=0)
    speaker_label: str = Field(...)
    start_ms: TranslationMs = Field(...)
    end_ms: TranslationMs = Field(...)
    duration_ms: TranslationMs = Field(0)
    source_text: str = Field(...)
    translated_text: str = Field(...)
    adapted_text: str = Field(...)
    validation: TranslationSegmentValidation = Field(
        default_factory=TranslationSegmentValidation
    )

    @field_validator("speaker_label")
    @classmethod
    def _speaker_label_valid(cls, v: object) -> str:
        if isinstance(v, bool) or not isinstance(v, str):
            raise ValueError("speaker_label requiere str")
        s = str(v)
        if not re.fullmatch(r"^SPEAKER_[0-9]{2}$", s):
            raise ValueError(f"speaker_label inválido: {s!r}")
        return s

    @model_validator(mode="after")
    def _validate(self) -> "TranslationSegment":
        if not self.end_ms > self.start_ms:
            raise ValueError(
                f"TranslationSegment requiere end_ms > start_ms: "
                f"start_ms={self.start_ms}, end_ms={self.end_ms}"
            )
        expected = int(self.end_ms) - int(self.start_ms)
        if self.duration_ms != expected:
            object.__setattr__(self, "duration_ms", expected)
        # 1:1: si hay texto fuente, debe haber traducción y adaptación
        if str(self.source_text).strip() and not str(self.translated_text).strip():
            raise ValueError("translated_text vacío con source_text no vacío")
        if str(self.source_text).strip() and not str(self.adapted_text).strip():
            raise ValueError("adapted_text vacío con source_text no vacío")
        return self

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


# ---------------------------------------------------------------------------
# 5) TranslationValidation (global)
# ---------------------------------------------------------------------------


class TranslationValidation(BaseModel):
    """Validación global T15."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool = Field(True)
    errors: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    num_segments_input: int = Field(0, ge=0)
    num_segments_output: int = Field(0, ge=0)
    translation_performed: bool = Field(True)
    deterministic: bool = Field(True)

    @model_validator(mode="after")
    def _validate(self) -> "TranslationValidation":
        if self.num_segments_input != self.num_segments_output:
            raise ValueError(
                "TranslationValidation: num_segments_input != num_segments_output "
                "(1:1 violado)."
            )
        return self


# ---------------------------------------------------------------------------
# 6) TranslationThresholds (adaptación para doblaje)
# ---------------------------------------------------------------------------


class TranslationThresholds(BaseModel):
    """Umbrales configurables T15 (adaptación para doblaje).

    max_chars_per_second = 18.0   (velocidad de locución doblada)
    max_adaptation_ratio   = 1.35 (expansión máx traducción/original)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_chars_per_second: float = Field(
        18.0, gt=0.0, le=100.0, description="Máx caracteres por segundo para doblaje."
    )
    max_adaptation_ratio: float = Field(
        1.35, gt=0.0, le=5.0, description="Ratio máx adapted/source (expansión)."
    )

    @field_validator("max_chars_per_second", "max_adaptation_ratio", mode="before")
    @classmethod
    def _reject_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("No se permite bool en TranslationThresholds.")
        return v


__all__ = [
    "TranslationLanguage",
    "TRANSLATION_LANGUAGE_CODES",
    "TranslationLanguageCode",
    "TranslationMs",
    "TranslationText",
    "TranslationFloat",
    "TranslationSegmentValidation",
    "TranslationSegment",
    "TranslationValidation",
    "TranslationThresholds",
]
