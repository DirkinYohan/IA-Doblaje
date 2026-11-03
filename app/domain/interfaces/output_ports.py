from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class JsonOutputWriterPort(ABC):
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
        raise NotImplementedError  # pragma: no cover


__all__ = ["JsonOutputWriterPort"]
