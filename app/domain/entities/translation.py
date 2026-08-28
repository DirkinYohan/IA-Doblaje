"""Entidad TranslationResult Step 15 — Capa DOMAIN.

Frozen + extra=forbid. Sin filesystem (rutas como strings). Sin IA.
Chain of custody vía source_reference (job_id + source_language).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.value_objects.translation import (
    TranslationLanguageCode,
    TranslationSegment,
    TranslationValidation,
)

MODEL_LABEL_TRANSLATION_V1_LITERAL: Literal["t15:translation:v1"] = "t15:translation:v1"


class TranslationResult(BaseModel):
    """Resultado Step 15. Inmutable. Correspondencia 1:1 garantizada."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    model_label: Literal["t15:translation:v1"] = MODEL_LABEL_TRANSLATION_V1_LITERAL
    job_id: str | None = Field(default=None, min_length=8, max_length=128)
    source_language: TranslationLanguageCode = Field(...)
    target_language: TranslationLanguageCode = Field(...)
    segments: tuple[TranslationSegment, ...] = Field(
        default_factory=tuple, description="1:1 con segments.json, orden conservado."
    )
    validation: TranslationValidation = Field(default_factory=TranslationValidation)

    # =====================================================================
    # VALIDADORES
    # =====================================================================

    @model_validator(mode="after")
    def _validate_all(self) -> "TranslationResult":
        # (A) 1:1 — segment_index contiguo desde 0, sin duplicados ni huecos
        expected = 0
        seen: set[int] = set()
        for seg in self.segments:
            idx = int(seg.segment_index)
            if idx in seen:
                raise ValueError(f"TranslationResult segment_index duplicado: {idx}")
            seen.add(idx)
            if idx != expected:
                raise ValueError(
                    f"TranslationResult segment_index discontinuo: esperado "
                    f"{expected}, encontrado {idx} (1:1 violado)."
                )
            expected += 1

        # (B) Coherencia con validation global
        if self.validation.num_segments_input != len(self.segments):
            raise ValueError(
                "TranslationResult validation.num_segments_input != len(segments)."
            )
        if self.validation.num_segments_output != len(self.segments):
            raise ValueError(
                "TranslationResult validation.num_segments_output != len(segments)."
            )
        # (C) validation.ok coherente con errores
        if self.validation.ok and self.validation.errors:
            raise ValueError("TranslationResult validation.ok=True con errors no vacío.")
        return self


__all__ = ["TranslationResult", "MODEL_LABEL_TRANSLATION_V1_LITERAL"]
