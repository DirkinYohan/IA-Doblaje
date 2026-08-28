"""Value Objects LID Step 05 — Language ID Detection. Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (no torch/transformers/soundfile).
"""
from __future__ import annotations

from typing import Annotated, Optional

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


def _validate_duration_ms_ge_zero(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f"LidDurationMs requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"LidDurationMs no admite negativos: {v}")
    return int(v)


LidDurationMs = Annotated[int, BeforeValidator(_validate_duration_ms_ge_zero)]


def _validate_confidence_0_1(v: object) -> float:
    if isinstance(v, bool):
        raise TypeError("LanguageConfidence no admite bool")
    if not isinstance(v, (int, float)):
        raise TypeError(
            f"LanguageConfidence requiere float/int, recibido {type(v).__name__}"
        )
    x = float(v)
    if x < 0.0:
        raise ValueError(f"LanguageConfidence < 0.0: {x}")
    if x > 1.0:
        raise ValueError(f"LanguageConfidence > 1.0: {x}")
    return x


LanguageConfidence = Annotated[float, BeforeValidator(_validate_confidence_0_1)]


_ISO_2_WHITELIST = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "la", "mi", "ml", "cy", "sk", "te",
    "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
    "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl",
    "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka",
    "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht",
    "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
    "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "und",
}


def _validate_language_code(v: object) -> str:
    if not isinstance(v, str):
        raise TypeError(
            f"LanguageCode requiere str, recibido {type(v).__name__}"
        )
    s = v.strip().lower()
    if len(s) < 2 or len(s) > 3:
        raise ValueError(
            f"LanguageCode length inválida: {s!r} (esperado 2 o 3 chars ISO)."
        )
    if not s.isalpha():
        raise ValueError(f"LanguageCode no alfabético: {s!r}")
    if s not in _ISO_2_WHITELIST:
        if len(s) == 3 and s.isalpha():
            return s
        raise ValueError(
            f"LanguageCode {s!r} no permitido (ISO 639-1/639-3 Whisper multilingual)."
        )
    return s


LanguageCode = Annotated[str, BeforeValidator(_validate_language_code)]


# ---------------------------------------------------------------------------
# 2) LanguageAlternative (Top-3 alternatives; rank 1 = primary ya incluido).
# ---------------------------------------------------------------------------


class LanguageAlternative(BaseModel):
    """Alternativa de idioma detectada. Frozen + extra forbid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language_code: LanguageCode = Field(..., description="ISO 639-1 lowercase (o 639-3 si rare).")
    language_name: str = Field(..., min_length=1, max_length=64, description="Nombre EN INGLÉS.")
    confidence: LanguageConfidence = Field(..., description="Confianza [0,1] para esta alternativa.")
    rank: int = Field(..., ge=1, le=3, description="Posición en Top-3 (1 = mejor alternativa tras primary).")

    @field_validator("language_name")
    @classmethod
    def _name_stripped(cls, v: str) -> str:
        s = str(v).strip()
        if len(s) == 0:
            raise ValueError("LanguageAlternative.language_name vacío.")
        return s


# ---------------------------------------------------------------------------
# 3) LanguageDetectionThresholds (configuración centralizada T05).
#    NO tocar ProcessingConfig.
# ---------------------------------------------------------------------------


class LanguageDetectionThresholds(BaseModel):
    """Configuración LID centralizada. Defaults oficiales PIPELINE.md + user rules.

    min_confidence             = 0.4  (PIPELINE.md Umbral mínimo)
    top_k                      = 3    (PIPELINE.md Top-3 alternativas)
    min_speech_ratio_to_analyze = 0.01 (al menos 1% speech para analizar)
    default_language_override  = None (fallback user-specified si < threshold strict=False)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_confidence: LanguageConfidence = Field(
        0.4,
        description="Umbral mínimo PIPELINE.md para aceptar una predicción como válida. Default 0.4.",
    )
    top_k: int = Field(
        3, ge=1, le=3, description="Número máximo de alternativas. PIPELINE.md requiere Top-3."
    )
    min_speech_ratio_to_analyze: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description="Ratio mínimo VAD speech sobre audio total para correr LID. Default 1%.",
    )
    default_language_override: Optional[LanguageCode] = Field(
        None,
        description="ISO code user-specified fallback si conf < min_confidence (PIPELINE.md: user-specified). PERTENECE AQUÍ, NO a ProcessingConfig.",
    )

    @field_validator("min_speech_ratio_to_analyze")
    @classmethod
    def _ratio_range(cls, v: float) -> float:
        x = float(v)
        if x < 0.0 or x > 1.0:
            raise ValueError(
                f"min_speech_ratio_to_analyze fuera [0,1]: {x}"
            )
        return x


# ---------------------------------------------------------------------------
# 4) Helpers validación de alternatives (usados desde Entity).
# ---------------------------------------------------------------------------


def validate_alternatives_sorted_desc(
    alternatives: tuple[LanguageAlternative, ...], top_k: int
) -> tuple[LanguageAlternative, ...]:
    """Validar y normalizar alternatives: len<=top_k, order DESC confidence, rank consecutivos."""
    if len(alternatives) > top_k:
        raise ValueError(
            f"alternatives length {len(alternatives)} > top_k={top_k}"
        )
    seen_ranks: set[int] = set()
    seen_codes: set[str] = set()
    prev_conf: float = 1.0 + 1e-9
    for alt in alternatives:
        if alt.rank in seen_ranks:
            raise ValueError(f"alternatives rank duplicado: {alt.rank}")
        seen_ranks.add(int(alt.rank))
        code = str(alt.language_code)
        if code in seen_codes:
            raise ValueError(f"alternatives language_code duplicado: {code}")
        seen_codes.add(code)
        if float(alt.confidence) > prev_conf + 1e-9:
            raise ValueError(
                "alternatives no están ORDENADAS DESC por confidence: "
                f"{alt.confidence} > prev {prev_conf}"
            )
        prev_conf = float(alt.confidence)
    return alternatives


__all__ = [
    "LidDurationMs",
    "LanguageConfidence",
    "LanguageCode",
    "LanguageAlternative",
    "LanguageDetectionThresholds",
    "validate_alternatives_sorted_desc",
]
