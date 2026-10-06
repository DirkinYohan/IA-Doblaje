"""Caché JSON de traducciones. La clave la calcula el caso de uso."""
from __future__ import annotations

import json
from pathlib import Path


class JsonTranslationCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data = {str(k): str(v) for k, v in loaded.items() if isinstance(v, str)}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        self._data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
