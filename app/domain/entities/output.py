"""Entidad StructuredJsonOutputResult Step 13 — Capa DOMAIN.

Frozen + extra=forbid. Sin filesystem (rutas como strings). Determinista.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.constants import PipelineStep
from app.domain.value_objects.output import OutputCount, OutputFileRecord

MODEL_LABEL_OUTPUT_V1_LITERAL: Literal["t13:structured-json-output:v1"] = (
    "t13:structured-json-output:v1"
)


class StructuredJsonOutputResult(BaseModel):
    """Resultado Step 13. Inmutable. Referencias lógicas, sin handles de filesystem."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    step: PipelineStep = PipelineStep.STRUCTURED_JSON_OUTPUT
    model_label: Literal["t13:structured-json-output:v1"] = MODEL_LABEL_OUTPUT_V1_LITERAL
    job_id: str | None = Field(default=None, min_length=8, max_length=128)
    output_directory: str = Field(..., min_length=1, max_length=1024)
    written_files: tuple[OutputFileRecord, ...] = Field(
        default_factory=tuple, description="Registros de archivos escritos (nombre + SHA)."
    )
    file_sha256: dict[str, str] = Field(
        default_factory=dict, description="{filename: sha256} de los bytes escritos."
    )
    files_written_count: OutputCount = Field(0)
    output_success: bool = Field(False)
    schema_version: str = Field(..., min_length=1, max_length=32)
    analysis_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_all(self) -> "StructuredJsonOutputResult":
        n = int(self.files_written_count)
        if n != len(self.written_files):
            raise ValueError(
                f"files_written_count={n} != len(written_files)={len(self.written_files)}"
            )
        if n != len(self.file_sha256):
            raise ValueError(
                f"files_written_count={n} != len(file_sha256)={len(self.file_sha256)}"
            )
        # Cada written_file debe tener su SHA en file_sha256
        for rec in self.written_files:
            if rec.filename not in self.file_sha256:
                raise ValueError(f"written_file {rec.filename!r} sin SHA en file_sha256")
        # Sin claves extras en file_sha256
        written_names = {r.filename for r in self.written_files}
        if set(self.file_sha256.keys()) != written_names:
            raise ValueError("file_sha256 claves no coinciden con written_files")
        # output_success coherente con conteo
        if self.output_success and n == 0:
            raise ValueError("output_success=True con files_written_count=0")
        return self


__all__ = ["StructuredJsonOutputResult", "MODEL_LABEL_OUTPUT_V1_LITERAL"]
