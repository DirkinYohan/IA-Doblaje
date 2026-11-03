from __future__ import annotations

import pytest

from app.infrastructure.translation import m2m100_ct2_adapter

pytestmark = pytest.mark.skipif(
    not m2m100_ct2_adapter.is_available(),
    reason="falta models/m2m100_418M_ct2_int8 (scripts/convert_m2m100_ct2.py)",
)


def test_traduce_un_lote_conservando_el_orden() -> None:
    translator = m2m100_ct2_adapter.M2M100CT2Translator()
    out = translator.translate_batch(
        ["Hello, and welcome.", "See you tomorrow."], source="en", target="es"
    )
    assert len(out) == 2
    assert "bienvenid" in out[0].lower()
    assert "mañana" in out[1].lower()
    assert translator.translate_batch([], source="en", target="es") == []
