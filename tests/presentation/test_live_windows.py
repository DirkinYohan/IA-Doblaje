from __future__ import annotations

import numpy as np

from app.presentation.api import live


def _subtitler(reads: list[tuple[int, int]]) -> live.LiveSubtitler:
    def read(start: int, end: int) -> np.ndarray:
        reads.append((start, end))
        return np.zeros(0, dtype=np.float32)

    callbacks = live.LiveCallbacks(
        read=read, transcribe=lambda *a: [], translate=lambda texts, *_a: list(texts),
        detect=lambda _audio: "en", emit=lambda _e: None, persist=lambda *a, **k: None,
    )
    return live.LiveSubtitler(callbacks=callbacks, source_language="en", duration_ms=600_000, targets=["en"])


def test_con_poco_margen_usa_tramos_cortos() -> None:
    reads: list[tuple[int, int]] = []
    sub = _subtitler(reads)
    sub._ranges("en").add(0, 15_000)
    sub.set_position(5_000)
    sub.step()
    assert reads[-1] == (15_000, 15_000 + live.NEAR_WINDOW_MS)


def test_con_margen_usa_tramos_largos() -> None:
    reads: list[tuple[int, int]] = []
    sub = _subtitler(reads)
    sub._ranges("en").add(0, 40_000)
    sub.set_position(5_000)
    sub.step()
    assert reads[-1] == (40_000, 40_000 + live.WINDOW_MS)


def test_al_empezar_el_primer_tramo_es_corto() -> None:
    reads: list[tuple[int, int]] = []
    sub = _subtitler(reads)
    sub.step()
    assert reads[0] == (0, live.NEAR_WINDOW_MS)
