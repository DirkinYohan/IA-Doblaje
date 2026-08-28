"""Entidad TempCleanupResult Step 14 — Temp Cleanup. Capa DOMAIN.

Frozen + extra=forbid. Sin Path/handles/objetos Infrastructure. Determinista.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.constants import PipelineStep


class TempCleanupResult(BaseModel):
    """Resultado Step 14 Temp Cleanup. Inmutable. Sin SHA artificial."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    step: PipelineStep = PipelineStep.TEMP_CLEANUP
    job_id: str | None = Field(default=None, min_length=1, max_length=128)
    cleaned: bool = Field(False)
    removed_paths: tuple[str, ...] = Field(default_factory=tuple)
    errors: tuple[str, ...] = Field(default_factory=tuple)
    success: bool = Field(False)

    @field_validator("removed_paths", "errors", mode="before")
    @classmethod
    def _must_be_tuple(cls, v: object) -> object:
        if v is None:
            return ()
        if not isinstance(v, tuple):
            raise ValueError("removed_paths/errors debe ser tuple")
        return v

    @field_validator("cleaned", "success", mode="before")
    @classmethod
    def _must_be_bool(cls, v: object) -> object:
        if not isinstance(v, bool):
            raise ValueError("cleaned/success debe ser bool")
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "TempCleanupResult":
        # step debe ser siempre TEMP_CLEANUP
        if self.step != PipelineStep.TEMP_CLEANUP:
            raise ValueError("step debe ser PipelineStep.TEMP_CLEANUP")

        # success coherente con errors
        if self.success and self.errors:
            raise ValueError("success=True no puede tener errors no vacíos")

        return self


__all__ = ["TempCleanupResult"]
