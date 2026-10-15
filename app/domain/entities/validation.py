"""Entidad ValidationResult Step 11 — Validation. Capa DOMAIN.

Frozen + extra=forbid. Chain of custody 8-SHA:
  source_preprocessed_sha256 (T03)
  vad_reference_sha256 (T04)
  lid_reference_sha256 (T05)
  asr_reference_sha256 (T06)
  ts_reference_sha256 (T07)
  diarization_reference_sha256 (T08)
  alignment_reference_sha256 (T09)
  quality_reference_sha256 (T10)

Sin dependencias infraestructura (no torch/pyannote/numpy).
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

from app.core.constants import PipelineStep, QualityLevel
from app.domain.value_objects.audio import Sha256Hash
from app.domain.value_objects.validation import (
    ValidationCheckCode,
    ValidationCheckResult,
    ValidationCount,
    ValidationScore,
    ValidationStatus,
)

VALIDATION_STRATEGY_LITERAL: Literal["validation_vc01_vc11"] = "validation_vc01_vc11"

MODEL_LABEL_VALIDATION_V1_LITERAL: Literal[
    "t11:validation:v1"
] = "t11:validation:v1"


class ValidationResult(BaseModel):
    """Resultado Step 11 Validation. Inmutable. Chain of Custody 8-SHA."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    # ---- Step identifiers ------------------------------------------------
    step: PipelineStep = PipelineStep.VALIDATION
    model_label: Literal["t11:validation:v1"] = MODEL_LABEL_VALIDATION_V1_LITERAL

    # ---- Chain of Custody 8-SHA ------------------------------------------
    source_preprocessed_sha256: Sha256Hash
    vad_reference_sha256: Sha256Hash
    lid_reference_sha256: Sha256Hash
    asr_reference_sha256: Sha256Hash
    ts_reference_sha256: Sha256Hash
    diarization_reference_sha256: Sha256Hash
    alignment_reference_sha256: Sha256Hash
    quality_reference_sha256: Sha256Hash

    # ---- Validación ------------------------------------------------------
    job_id: str | None = Field(default=None, min_length=8, max_length=128)
    validation_passed: bool = Field(..., description="True si no hay errores.")
    validation_status: ValidationStatus = Field(..., description="passed | warning | failed.")
    quality_level: QualityLevel = Field(..., description="Nivel de calidad heredado de T10.")
    overall_quality_score: ValidationScore = Field(..., description="Score heredado de T10.")
    validation_errors: tuple[str, ...] = Field(default_factory=tuple)
    validation_warnings: tuple[str, ...] = Field(default_factory=tuple)
    num_checks: ValidationCount = Field(11, description="Siempre 11 checks.")
    num_errors: ValidationCount = Field(0)
    num_warnings: ValidationCount = Field(0)
    num_passed: ValidationCount = Field(0)

    # ---- Auditoría -------------------------------------------------------
    check_results: tuple[ValidationCheckResult, ...] = Field(
        default_factory=tuple, description="11 resultados en orden VC01..VC11."
    )
    analysis_metadata: dict[str, object] = Field(default_factory=dict)

    # ======================================================================
    # VALIDADORES
    # ======================================================================

    @field_validator(
        "source_preprocessed_sha256",
        "vad_reference_sha256",
        "lid_reference_sha256",
        "asr_reference_sha256",
        "ts_reference_sha256",
        "diarization_reference_sha256",
        "alignment_reference_sha256",
        "quality_reference_sha256",
    )
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

    @model_validator(mode="after")
    def _validate_all(self) -> "ValidationResult":
        # (A) Chain of custody: todas las refs == source (T03)
        source = str(self.source_preprocessed_sha256)
        for name in (
            "vad_reference_sha256",
            "lid_reference_sha256",
            "asr_reference_sha256",
            "ts_reference_sha256",
            "diarization_reference_sha256",
            "alignment_reference_sha256",
            "quality_reference_sha256",
        ):
            if str(getattr(self, name)) != source:
                raise ValueError(
                    f"Chain of Custody rota T11: {name} != source_preprocessed_sha256."
                )

        # (B) num_checks == 11 y == len(check_results)
        if self.num_checks != 11:
            raise ValueError(f"num_checks debe ser 11, got {self.num_checks}")
        if len(self.check_results) != 11:
            raise ValueError(
                f"check_results debe tener 11 elementos, got {len(self.check_results)}"
            )

        # (C) Orden determinista VC01..VC11
        expected: tuple[ValidationCheckCode, ...] = (
            "VC01", "VC02", "VC03", "VC04", "VC05", "VC06",
            "VC07", "VC08", "VC09", "VC10", "VC11",
        )
        actual = tuple(r.check_code for r in self.check_results)
        if actual != expected:
            raise ValueError(f"check_results deben estar en orden VC01..VC11, got {actual!r}")

        # (D) contadores consistentes
        errors = sum(1 for r in self.check_results if r.outcome == "error")
        warnings = sum(1 for r in self.check_results if r.outcome == "warning")
        passed = sum(1 for r in self.check_results if r.outcome == "passed")
        if self.num_errors != errors:
            raise ValueError(f"num_errors={self.num_errors} != computed={errors}")
        if self.num_warnings != warnings:
            raise ValueError(f"num_warnings={self.num_warnings} != computed={warnings}")
        if self.num_passed != passed:
            raise ValueError(f"num_passed={self.num_passed} != computed={passed}")

        # (E) validation_passed coherente
        if self.validation_passed != (errors == 0):
            raise ValueError(
                f"validation_passed={self.validation_passed} inconsistente con errors={errors}"
            )

        # (F) validation_status coherente
        if self.validation_status == "failed":
            # "failed" puede provenir de quality FAILED (force-save) sin errores
            # estructurales, o de errores estructurales.
            if errors == 0 and self.quality_level != QualityLevel.FAILED:
                raise ValueError(
                    "validation_status=failed requiere errors>0 o quality_level=failed"
                )
        elif self.validation_status == "passed":
            if errors != 0:
                raise ValueError("validation_status=passed requiere errors==0")
            if self.quality_level == QualityLevel.FAILED:
                raise ValueError("validation_status=passed incompatible con quality_level=failed")

        # (G) validation_errors/warnings consistencia con contadores
        if self.num_errors != len(self.validation_errors):
            raise ValueError(
                f"num_errors={self.num_errors} != len(validation_errors)={len(self.validation_errors)}"
            )
        if self.num_warnings != len(self.validation_warnings):
            raise ValueError(
                f"num_warnings={self.num_warnings} != len(validation_warnings)={len(self.validation_warnings)}"
            )

        return self


__all__ = [
    "ValidationResult",
    "VALIDATION_STRATEGY_LITERAL",
    "MODEL_LABEL_VALIDATION_V1_LITERAL",
]
