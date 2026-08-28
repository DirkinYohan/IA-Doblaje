"""Tests unitarios T15 — Domain Value Objects y Entity Translation.

Solo DOMAIN, sin transformers/torch/infrastructure. Pydantic.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.domain.entities.translation import TranslationResult
from app.domain.value_objects.translation import (
    TRANSLATION_LANGUAGE_CODES,
    TranslationLanguage,
    TranslationLanguageCode,
    TranslationSegment,
    TranslationSegmentValidation,
    TranslationThresholds,
    TranslationValidation,
)


class _LangDut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    v: TranslationLanguageCode


class _SegDut(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    v: TranslationSegment


_SHA = "a" * 64


def _seg(idx: int = 0, start: int = 0, end: int = 1000) -> TranslationSegment:
    return TranslationSegment(
        segment_index=idx,
        speaker_label="SPEAKER_00",
        start_ms=start,
        end_ms=end,
        duration_ms=end - start,
        source_text="Hello",
        translated_text="Hola",
        adapted_text="Hola",
    )


# =============================================================================
# TranslationLanguage / TranslationLanguageCode
# =============================================================================


class TestTranslationLanguage:
    def test_8_idiomas(self):
        codes = {lang.value for lang in TranslationLanguage}
        assert codes == {"es", "en", "fr", "de", "it", "pt", "ja", "zh"}

    def test_constantes_coinciden(self):
        assert TRANSLATION_LANGUAGE_CODES == {
            "es", "en", "fr", "de", "it", "pt", "ja", "zh"
        }

    def test_display_names(self):
        assert TranslationLanguage.ES.display_name == "Español"
        assert TranslationLanguage.ZH.display_name == "Chino mandarín"

    def test_cada_idioma_valido(self):
        for code in ("es", "en", "fr", "de", "it", "pt", "ja", "zh"):
            assert _LangDut(v=code).v == code

    @pytest.mark.parametrize("bad", ["xx", "ru", "ko", "und", "", 5, None, True])
    def test_idioma_invalido_rechazado(self, bad):
        with pytest.raises((ValidationError, TypeError, ValueError)):
            _LangDut(v=bad)

    def test_normaliza_minusculas_y_strip(self):
        assert _LangDut(v="ES").v == "es"
        assert _LangDut(v=" EN ").v == "en"
        assert _LangDut(v="Es").v == "es"


# =============================================================================
# TranslationSegment
# =============================================================================


class TestTranslationSegment:
    def test_valido(self):
        s = _seg()
        assert s.segment_index == 0
        assert s.duration_ms == 1000
        assert s.validation.ok is True

    def test_duration_recomputada(self):
        s = TranslationSegment(
            segment_index=0, speaker_label="SPEAKER_00",
            start_ms=100, end_ms=400, duration_ms=999,
            source_text="hi", translated_text="hola", adapted_text="hola",
        )
        assert s.duration_ms == 300

    def test_end_must_be_greater_than_start(self):
        with pytest.raises(ValidationError):
            _SegDut(v=TranslationSegment(
                segment_index=0, speaker_label="SPEAKER_00",
                start_ms=1000, end_ms=1000,
                source_text="x", translated_text="y", adapted_text="z"))

    def test_speaker_label_invalido(self):
        with pytest.raises(ValidationError):
            TranslationSegment(
                segment_index=0, speaker_label="SPK_00",
                start_ms=0, end_ms=100,
                source_text="x", translated_text="y", adapted_text="z")

    def test_translated_vacio_con_source_no(self):
        with pytest.raises(ValidationError):
            TranslationSegment(
                segment_index=0, speaker_label="SPEAKER_00",
                start_ms=0, end_ms=100,
                source_text="hello", translated_text="", adapted_text="hola")

    def test_adapted_vacio_con_source_no(self):
        with pytest.raises(ValidationError):
            TranslationSegment(
                segment_index=0, speaker_label="SPEAKER_00",
                start_ms=0, end_ms=100,
                source_text="hello", translated_text="hola", adapted_text="")

    def test_vacio_permite_vacio(self):
        s = TranslationSegment(
            segment_index=0, speaker_label="SPEAKER_00",
            start_ms=0, end_ms=100,
            source_text="", translated_text="", adapted_text="")
        assert s.duration_ms == 100

    def test_frozen_no_mutation(self):
        s = _seg()
        with pytest.raises(ValidationError):
            s.segment_index = 5  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            TranslationSegment(
                segment_index=0, speaker_label="SPEAKER_00",
                start_ms=0, end_ms=100, source_text="x",
                translated_text="y", adapted_text="z", extra=1)  # type: ignore[call-arg]


# =============================================================================
# TranslationValidation
# =============================================================================


class TestTranslationValidation:
    def test_ok_default(self):
        v = TranslationValidation()
        assert v.ok is True
        assert v.translation_performed is True

    def test_1a1_requerido(self):
        with pytest.raises(ValidationError):
            TranslationValidation(num_segments_input=3, num_segments_output=2)

    def test_ok_true_con_errors_rechaza(self):
        with pytest.raises(ValidationError):
            TranslationResult(
                source_language="en",
                target_language="es",
                segments=(_seg(0),),
                validation=TranslationValidation(
                    ok=True, errors=("algo",),
                    num_segments_input=1, num_segments_output=1,
                ),
            )


# =============================================================================
# TranslationResult
# =============================================================================


class TestTranslationResult:
    def test_valido(self):
        r = TranslationResult(
            source_language="en",
            target_language="es",
            segments=(_seg(0), _seg(1, start=1000, end=2000)),
            validation=TranslationValidation(
                num_segments_input=2, num_segments_output=2
            ),
        )
        assert r.source_language == "en"
        assert r.target_language == "es"

    def test_segment_index_discontinuo_rechaza(self):
        with pytest.raises(ValidationError):
            TranslationResult(
                source_language="en",
                target_language="es",
                segments=(_seg(0), _seg(2, start=1000, end=2000)),
                validation=TranslationValidation(
                    num_segments_input=2, num_segments_output=2
                ),
            )

    def test_segment_index_duplicado_rechaza(self):
        with pytest.raises(ValidationError):
            TranslationResult(
                source_language="en",
                target_language="es",
                segments=(_seg(0), _seg(0, start=1000, end=2000)),
                validation=TranslationValidation(
                    num_segments_input=2, num_segments_output=2
                ),
            )

    def test_num_segments_input_no_coincide_rechaza(self):
        with pytest.raises(ValidationError):
            TranslationResult(
                source_language="en",
                target_language="es",
                segments=(_seg(0),),
                validation=TranslationValidation(
                    num_segments_input=2, num_segments_output=1
                ),
            )

    def test_orden_conservado(self):
        r = TranslationResult(
            source_language="en",
            target_language="es",
            segments=(_seg(0), _seg(1, start=1000, end=2000)),
            validation=TranslationValidation(
                num_segments_input=2, num_segments_output=2
            ),
        )
        assert [s.segment_index for s in r.segments] == [0, 1]


# =============================================================================
# TranslationThresholds
# =============================================================================


class TestTranslationThresholds:
    def test_defaults(self):
        t = TranslationThresholds()
        assert t.max_chars_per_second == 18.0
        assert t.max_adaptation_ratio == 1.35

    def test_rechaza_bool(self):
        with pytest.raises(ValidationError):
            TranslationThresholds(max_chars_per_second=True)  # type: ignore[arg-type]

    def test_rechaza_negativos(self):
        with pytest.raises(ValidationError):
            TranslationThresholds(max_chars_per_second=-1.0)

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            TranslationThresholds(otro=1)  # type: ignore[call-arg]
