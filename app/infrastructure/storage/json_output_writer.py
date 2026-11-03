from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.core.exceptions import OutputWriteError
from app.domain.interfaces.output_ports import JsonOutputWriterPort


class JsonOutputWriter(JsonOutputWriterPort):
    def write_json(
        self,
        *,
        output_directory: str,
        filename: str,
        document: dict[str, Any],
        indent: int,
        ensure_ascii: bool,
    ) -> str:
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

        try:
            out_dir = Path(output_directory)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / filename
            target.write_bytes(data)
        except OSError as exc:
            raise OutputWriteError(
                f"No se pudo escribir {filename!r} en {output_directory!r}: {exc!r}"
            ) from exc

        return hashlib.sha256(data).hexdigest()


__all__ = ["JsonOutputWriter"]
