"""Carga del `.env` del proyecto en el entorno del proceso.

`pydantic-settings` lee `.env` para su propio modelo, pero **no lo exporta** a
`os.environ`. Cualquier módulo que lea variables de entorno directamente
(la base de datos, el almacenamiento) necesita esta carga explícita.

Es idempotente y no pisa lo que ya venga del entorno real.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from app.core.config import PROJECT_ROOT

_LOCK = threading.Lock()
_LOADED = False


def load_project_env() -> None:
    """Vuelca `.env` en `os.environ` una sola vez."""
    global _LOADED
    if _LOADED:
        return
    with _LOCK:
        if _LOADED:
            return
        _LOADED = True
        path = Path(PROJECT_ROOT) / ".env"
        if not path.is_file():
            return
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover - dependencia opcional
            return
        load_dotenv(path, override=False)


def env(name: str, default: str = "") -> str:
    """Variable de entorno ya con el `.env` cargado."""
    load_project_env()
    return (os.environ.get(name) or default).strip()
