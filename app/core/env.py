from __future__ import annotations

import os
import threading
from pathlib import Path

from app.core.config import PROJECT_ROOT

_LOCK = threading.Lock()
_LOADED = False


def load_project_env() -> None:
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
    load_project_env()
    return (os.environ.get(name) or default).strip()
