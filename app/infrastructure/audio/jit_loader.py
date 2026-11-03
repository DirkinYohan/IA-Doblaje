from __future__ import annotations

from pathlib import Path
from typing import Any


def _inside_models(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    parts = {p.lower() for p in resolved.parts}
    return "models" in parts


def load_jit_confined(torch_mod: Any, model_path: Path, device: str) -> Any:
    path = Path(model_path).expanduser().resolve()
    if not _inside_models(path):
        from app.core.exceptions import ModelLoadError

        raise ModelLoadError(
            f"JIT rechazado: {path} no está dentro del directorio models/."
        )
    try:
        return torch_mod.jit.load(str(path), map_location=device, weights_only=True)
    except TypeError as exc:
        message = str(exc).lower()
        if "weights_only" not in message and "unexpected keyword" not in message:
            raise
        return torch_mod.jit.load(str(path), map_location=device)
