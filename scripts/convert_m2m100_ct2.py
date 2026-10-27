"""Convierte el M2M100 418M local a CTranslate2 int8 (sin red).

Uso:
    .venv\\Scripts\\python.exe scripts\\convert_m2m100_ct2.py

Genera ``models/m2m100_418M_ct2_int8`` (~470 MB). El modo en vivo lo usa si
existe; si no, traduce con el modelo PyTorch, unas cuatro veces más lento.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from ctranslate2.converters import TransformersConverter

    source = ROOT / "models" / "m2m100_418M"
    target = ROOT / "models" / "m2m100_418M_ct2_int8"
    if not (source / "config.json").is_file():
        print(f"No encuentro el modelo de origen: {source}", file=sys.stderr)
        return 1
    TransformersConverter(str(source)).convert(str(target), quantization="int8", force=True)
    print(f"Listo: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
