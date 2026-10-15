"""Fixtures de audio sintetico.

T01: solo stubs.
Los generadores reales se implementan en T04 junto a la extraccion de audio
y tests de integracion de 30s / 1min.
"""

from __future__ import annotations


def get_fixture_clip_path(duration_label: str) -> str:
    """Devuelve ruta (no garantiza existencia) a un fixture de duracion conocida.

    Labels soportados en Fase 1 (validacion progresiva PUNTO 5):
        - "30s"
        - "1min"
        - "5min"
        - "30min"
    """
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "integration" / "fixtures_clips"
    return str(p / f"clip_{duration_label}.wav")
