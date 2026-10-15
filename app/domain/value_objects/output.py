"""Value Objects Structured JSON Output Step 13 — Capa DOMAIN.

Inmutables, validados. Sin dependencias de infraestructura (sin IA, sin filesystem).
"""
from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.functional_validators import BeforeValidator


_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def _validate_sha256(v: object) -> str:
    if isinstance(v, bool) or not isinstance(v, str):
        raise ValueError(f"OutputSha256 requiere str, recibido {type(v).__name__}")
    s = str(v).strip().lower()
    if not _SHA256_RE.match(s):
        raise ValueError(f"SHA256 inválido 64-hex: {s[:20]!r}...")
    return s


OutputSha256 = Annotated[str, BeforeValidator(_validate_sha256)]


def _validate_non_negative_int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"OutputCount requiere int, recibido {type(v).__name__}")
    if v < 0:
        raise ValueError(f"OutputCount no admite negativos: {v}")
    return int(v)


OutputCount = Annotated[int, BeforeValidator(_validate_non_negative_int)]


def _validate_output_filename(v: object) -> str:
    if isinstance(v, bool) or not isinstance(v, str):
        raise ValueError("OutputFilename requiere str")
    s = str(v).strip()
    if not s.endswith(".json"):
        raise ValueError(f"OutputFilename debe terminar en .json: {s!r}")
    return s


OutputFilename = Annotated[str, BeforeValidator(_validate_output_filename)]


class OutputFileRecord(BaseModel):
    """Registro de un archivo JSON escrito: nombre + SHA256."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: OutputFilename = Field(..., description="Nombre del archivo (ej. analysis.json).")
    sha256: OutputSha256 = Field(..., description="SHA256 de los bytes escritos.")


__all__ = [
    "OutputSha256",
    "OutputCount",
    "OutputFilename",
    "OutputFileRecord",
]
