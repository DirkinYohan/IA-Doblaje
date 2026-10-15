"""Entidad QualityResult Step 10 — Quality Analysis. Capa DOMAIN.

Frozen + extra=forbid. Chain of custody 7-SHA:
  source_preprocessed_sha256 (T03)
  vad_reference_sha256 (T04)
  lid_reference_sha256 (T05)
  asr_reference_sha256 (T06)
  ts_reference_sha256 (T07)
  diarization_reference_sha256 (T08)
  alignment_reference_sha256 (T09 — referencia de trazabilidad)

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
from app.domain.value_objects.quality import (
    QualityCount,
    QualityNonNegativeFloat,
    QualityRuleCode,
    QualityRuleResult,
    QualityScore,
    QualityStrategyName,
)

QUALITY_STRATEGY_RULES: Literal["quality_rules_qr01_qr11"] = "quality_rules_qr01_qr11"
QUALITY_STRATEGY_SILENT_FALLBACK: Literal["quality_silent_fallback"] = (
    "quality_silent_fallback"
)

QUALITY_STRATEGIES_TYPE = (
    Literal["quality_rules_qr01_qr11"] | Literal["quality_silent_fallback"]
)

MODEL_LABEL_QUALITY_V1_LITERAL: Literal[
    "t10:quality-analysis:v1"
] = "t10:quality-analysis:v1"


class QualityResult(BaseModel):
    """Resultado Step 10 Quality Analysis. Inmutable. Chain of Custody 7-SHA."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    # ---- Step identifiers ------------------------------------------------
    step: PipelineStep = PipelineStep.QUALITY_ANALYSIS
    model_label: Literal["t10:quality-analysis:v1"] = MODEL_LABEL_QUALITY_V1_LITERAL
    strategy: QualityStrategyName

    # ---- Chain of Custody 7-SHA ------------------------------------------
    source_preprocessed_sha256: Sha256Hash
    vad_reference_sha256: Sha256Hash
    lid_reference_sha256: Sha256Hash
    asr_reference_sha256: Sha256Hash
    ts_reference_sha256: Sha256Hash
    diarization_reference_sha256: Sha256Hash
    alignment_reference_sha256: Sha256Hash

    # ---- Quality ---------------------------------------------------------
    quality_level: QualityLevel = Field(..., description="passed | warning | failed.")
    overall_quality_score: QualityScore = Field(..., description="Score [0,1].")
    num_rules_evaluated: QualityCount = Field(11, description="Siempre 11 reglas.")
    num_rules_passed: QualityCount = Field(0)
    num_rules_warning: QualityCount = Field(0)
    num_rules_failed: QualityCount = Field(0)
    rule_results: tuple[QualityRuleResult, ...] = Field(
        default_factory=tuple, description="11 resultados en orden QR01..QR11."
    )

    # ---- Confianzas promedio ---------------------------------------------
    average_alignment_confidence: QualityScore = Field(0.0)
    average_ts_confidence: QualityScore = Field(0.0)
    average_asr_confidence: QualityScore = Field(0.0)

    # ---- Conteos / metadata T03/T07 passthrough --------------------------
    num_segments: QualityCount = Field(0)
    num_segments_aligned: QualityCount = Field(0)
    num_segments_unassigned: QualityCount = Field(0)
    total_duration_ms: QualityCount = Field(0)
    transcript_language_code: str = Field(min_length=2, max_length=16)
    job_id: str | None = Field(default=None, min_length=8, max_length=128)

    # ---- Auditoría -------------------------------------------------------
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
    def _validate_all(self) -> "QualityResult":
        # (A) Chain of custody: 6 primeras referencias == source (T03)
        source = str(self.source_preprocessed_sha256)
        for name in (
            "vad_reference_sha256",
            "lid_reference_sha256",
            "asr_reference_sha256",
            "ts_reference_sha256",
            "diarization_reference_sha256",
            "alignment_reference_sha256",
        ):
            if str(getattr(self, name)) != source:
                raise ValueError(
                    f"Chain of Custody rota T10: {name} != source_preprocessed_sha256."
                )

        # (B) num_rules_evaluated == 11 y == len(rule_results)
        if self.num_rules_evaluated != 11:
            raise ValueError(f"num_rules_evaluated debe ser 11, got {self.num_rules_evaluated}")
        if len(self.rule_results) != 11:
            raise ValueError(
                f"rule_results debe tener 11 elementos, got {len(self.rule_results)}"
            )

        # (C) Orden determinista QR01..QR11 sin duplicados
        expected_codes: tuple[QualityRuleCode, ...] = (
            "QR01", "QR02", "QR03", "QR04", "QR05", "QR06",
            "QR07", "QR08", "QR09", "QR10", "QR11",
        )
        actual_codes = tuple(r.rule_code for r in self.rule_results)
        if actual_codes != expected_codes:
            raise ValueError(
                f"rule_results deben estar en orden QR01..QR11, got {actual_codes!r}"
            )

        # (D) contadores passed/warning/failed consistentes con rule_results
        passed = sum(1 for r in self.rule_results if r.passed)
        failed = sum(
            1 for r in self.rule_results if r.severity == "ERROR" and not r.passed
        )
        warning = sum(
            1 for r in self.rule_results if r.severity == "WARNING" and not r.passed
        )
        if self.num_rules_passed != passed:
            raise ValueError(
                f"num_rules_passed={self.num_rules_passed} != computed={passed}"
            )
        if self.num_rules_failed != failed:
            raise ValueError(
                f"num_rules_failed={self.num_rules_failed} != computed={failed}"
            )
        if self.num_rules_warning != warning:
            raise ValueError(
                f"num_rules_warning={self.num_rules_warning} != computed={warning}"
            )

        # (E) quality_level coherente con score
        score = float(self.overall_quality_score)
        if score >= 0.80:
            expected_level = QualityLevel.PASSED
        elif score >= 0.30:
            expected_level = QualityLevel.WARNING
        else:
            expected_level = QualityLevel.FAILED
        if self.quality_level != expected_level:
            raise ValueError(
                f"quality_level={self.quality_level} no coincide con score={score} "
                f"(esperado {expected_level})."
            )

        # (F) silent fallback consistency
        if self.strategy == QUALITY_STRATEGY_SILENT_FALLBACK:
            if self.num_segments != 0:
                raise ValueError(
                    "quality_silent_fallback requiere num_segments=0"
                )

        return self


__all__ = [
    "QualityResult",
    "QUALITY_STRATEGY_RULES",
    "QUALITY_STRATEGY_SILENT_FALLBACK",
    "QUALITY_STRATEGIES_TYPE",
    "MODEL_LABEL_QUALITY_V1_LITERAL",
]
