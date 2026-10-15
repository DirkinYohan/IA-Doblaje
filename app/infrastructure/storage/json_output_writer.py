"""Infrastructure Adapter Step 13 — JsonOutputWriter (filesystem + stdlib json).

INFRASTRUCTURE ONLY. Aquí sí se accede al filesystem. Sin IA, sin red.
Serializa con allow_nan=False (rechaza NaN/Inf) y calcula SHA256 de los bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.exceptions import OutputWriteError
from app.domain.interfaces.output_ports import JsonOutputWriterPort


class JsonOutputWriter(JsonOutputWriterPort):
    """Implementación concreta del JsonOutputWriterPort.

    Usa únicamente filesystem + stdlib. allow_nan=False (estricto).
    """

    def write_json(
        self,
        *,
        output_directory: str,
        filename: str,
        document: dict[str, Any],
        indent: int,
        ensure_ascii: bool,
    ) -> str:
        # Serializar con allow_nan=False (rechaza NaN/Inf)
        try:
            raw = json.dumps(
                document,
                indent=indent,
                ensure_ascii=ensure_ascii,
                allow_nan=False,
                sort_keys=False,
            )
        except (ValueError, TypeError) as exc:
            raise OutputWriteError(
                f"No se pudo serializar {filename!r}: {exc!r}"
            ) from exc

        data = raw.encode("utf-8")

        # Escribir bytes exactos
        try:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / filename
            target.write_bytes(data)
        except OSError as exc:
            raise OutputWriteError(
                f"No se pudo escribir {filename!r} en {output_directory!r}: {exc!r}"
            ) from exc

        # SHA256 sobre los bytes exactos escritos
        return hashlib.sha256(data).hexdigest()


__all__ = ["JsonOutputWriter"]
