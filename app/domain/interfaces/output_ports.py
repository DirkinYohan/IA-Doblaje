"""Ports abstractos Structured JSON Output Step 13 — Capa DOMAIN.

Clean Architecture. Domain NO usa filesystem. Solo ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class JsonOutputWriterPort(ABC):
    """Puerto abstracto de escritura JSON. Application depende solo de esto.

    Implementación concreta (Infrastructure): filesystem + stdlib json.
    """

    @abstractmethod
    def write_json(
        self,
        *,
        output_directory: str,
        filename: str,
        document: dict[str, Any],
        indent: int,
        ensure_ascii: bool,
    ) -> str:
        """Serializa (allow_nan=False) y escribe el documento JSON.

        Retorna el SHA256 (64-hex) de los bytes exactos escritos.

        Raises:
            OutputWriteError si la escritura falla.
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["JsonOutputWriterPort"]
