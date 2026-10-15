"""Tests unitarios T05 — Domain Value Objects LID.

Solo DOMAIN, sin torch/transformers/infrastructure. pydantic + _Dut pattern T04.
"""
from __future__ import annotations

import copy
import pickle  # noqa: S403 - solo para comprobar frozen, no usa pickle inseguro de red

import pytest
from pydantic import BaseModel, ValidationError

from app.domain.value_objects.lid import (
    LanguageAlternative,
    LanguageCode,
    LanguageConfidence,
    LanguageDetectionThresholds,
    LidDurationMs,
    validate_alternatives_sorted_desc,
)


# DUTs para Annotated VOs (igual que T04 pattern)
class _CodeDut(BaseModel):
    v: LanguageCode


class _ConfDut(BaseModel):
    v: LanguageConfidence


class _DurDut(BaseModel):
    v: LidDurationMs


_VALID_CODES = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "haw", "und", "su", "jw", "ba", "tt", "as", "mg", "tl",
]


# =============================================================================
# 1) LanguageCode
# =============================================================================


class TestLanguageCode:
    @pytest.mark.parametrize("c", _VALID_CODES)
    def test_iso_2_whisper_known_ok(self, c: str):
        assert _CodeDut(v=c).v == c.lower()

    def test_case_normalized_to_lower(self):
        assert _CodeDut(v="ES").v == "es"
        assert _CodeDut(v="EN ").v == "en"

    @pytest.mark.parametrize("bad", ["", "  "])
    def test_invalid_empty(self, bad: str):
        with pytest.raises(ValidationError):
            _CodeDut(v=bad)

    def test_invalid_nonalpha(self):
        with pytest.raises(ValidationError):
            _CodeDut(v="e1")

    def test_invalid_too_short_1char(self):
        with pytest.raises(ValidationError):
            _CodeDut(v="e")

    def test_invalid_wrong_not_str_int(self):
        with pytest.raises((ValidationError, TypeError)):
            _CodeDut(v=42)

    @pytest.mark.parametrize("bad_bcp", ["es-ES", "pt-BR", "en-US"])
    def test_invalid_spaces_and_bcp47(self, bad_bcp: str):
        with pytest.raises(ValidationError):
            _CodeDut(v=bad_bcp)


# =============================================================================
# 2) LanguageConfidence
# =============================================================================


class TestLanguageConfidence:
    @pytest.mark.parametrize("x", [0.0, 0.25, 0.5, 0.99, 1.0])
    def test_valid_0_1_ok(self, x: float):
        assert abs(float(_ConfDut(v=x).v) - float(x)) < 1e-9

    def test_int_cast_ok(self):
        assert float(_ConfDut(v=0).v) == 0.0
        assert float(_ConfDut(v=1).v) == 1.0

    def test_invalid_below_zero(self):
        with pytest.raises((ValidationError, TypeError)):
            _ConfDut(v=-0.01)

    def test_invalid_above_one(self):
        with pytest.raises((ValidationError, TypeError)):
            _ConfDut(v=1.000001)

    def test_bool_rejected_not_float(self):
        with pytest.raises((ValidationError, TypeError)):
            _ConfDut(v=True)

    def test_non_numeric_rejected(self):
        with pytest.raises((ValidationError, TypeError)):
            _ConfDut(v="0.5")


# =============================================================================
# 3) LidDurationMs
# =============================================================================


class TestLidDurationMs:
    def test_ge0_ok(self):
        assert _DurDut(v=0).v == 0
        assert _DurDut(v=1_234_567).v == 1234567

    def test_negative_raises(self):
        with pytest.raises((ValidationError, TypeError)):
            _DurDut(v=-1)

    def test_nonint_raises_type(self):
        with pytest.raises((ValidationError, TypeError)):
            _DurDut(v="100")


# =============================================================================
# 4) LanguageAlternative
# =============================================================================


class TestLanguageAlternative:
    @staticmethod
    def _ok():
        return LanguageAlternative(
            language_code="es",
            language_name="Spanish",
            confidence=0.8,
            rank=1,
        )

    def test_happy_ok(self):
        a = self._ok()
        assert a.language_code == "es"
        assert a.confidence == 0.8
        assert a.rank == 1

    @pytest.mark.parametrize("r", [1, 2, 3])
    def test_rank_1_to_3_ok(self, r: int):
        LanguageAlternative(
            language_code="en", language_name="English", confidence=0.5, rank=r,
        )

    def test_rank_zero_or_4_invalid(self):
        with pytest.raises(ValidationError):
            LanguageAlternative(language_code="en", language_name="English", confidence=0.5, rank=0)
        with pytest.raises(ValidationError):
            LanguageAlternative(language_code="en", language_name="English", confidence=0.5, rank=4)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            LanguageAlternative(language_code="en", language_name="English", confidence=1.5, rank=1)

    def test_language_code_invalid(self):
        with pytest.raises(ValidationError):
            LanguageAlternative(language_code="en-US", language_name="English", confidence=0.5, rank=1)

    def test_language_name_empty_or_strip(self):
        with pytest.raises(ValidationError):
            LanguageAlternative(language_code="en", language_name="  ", confidence=0.5, rank=1)
        a = LanguageAlternative(language_code="en", language_name="  English  ", confidence=0.5, rank=1)
        assert a.language_name == "English"

    def test_frozen_no_mutation(self):
        a = self._ok()
        with pytest.raises(ValidationError):
            a.confidence = 0.9  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            LanguageAlternative(
                language_code="es", language_name="Spanish",
                confidence=0.5, rank=1,
                extra_field="X",
            )

    def test_determinismo_same_equals(self):
        a1 = self._ok()
        a2 = self._ok()
        assert a1 == a2
        assert a1.model_dump_json() == a2.model_dump_json()


# =============================================================================
# 5) LanguageDetectionThresholds
# =============================================================================


class TestLanguageDetectionThresholds:
    def test_defaults_pipeline_min_confidence_04(self):
        t = LanguageDetectionThresholds()
        assert abs(float(t.min_confidence) - 0.4) < 1e-9
        assert int(t.top_k) == 3
        assert abs(float(t.min_speech_ratio_to_analyze) - 0.01) < 1e-9
        assert t.default_language_override is None

    def test_override_ok(self):
        t = LanguageDetectionThresholds(
            min_confidence=0.7, top_k=2,
            min_speech_ratio_to_analyze=0.1,
            default_language_override="es",
        )
        assert float(t.min_confidence) == 0.7
        assert int(t.top_k) == 2
        assert t.default_language_override == "es"

    def test_frozen_no_assign(self):
        t = LanguageDetectionThresholds()
        with pytest.raises(ValidationError):
            t.min_confidence = 0.9  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(unknown_field=123)

    def test_min_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(min_confidence=1.5)
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(min_confidence=-0.1)

    def test_top_k_invalid_values(self):
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(top_k=0)
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(top_k=4)

    def test_speech_ratio_range(self):
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(min_speech_ratio_to_analyze=-0.01)
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(min_speech_ratio_to_analyze=1.5)

    def test_override_code_invalid(self):
        with pytest.raises(ValidationError):
            LanguageDetectionThresholds(default_language_override="es-ES")

    def test_deepcopy_identico(self):
        t1 = LanguageDetectionThresholds(
            min_confidence=0.6, default_language_override="pt",
        )
        t2 = copy.deepcopy(t1)
        assert t1 == t2

    def test_pickle_roundtrip_frozen(self):
        t = LanguageDetectionThresholds()
        b = pickle.dumps(t)
        t2 = pickle.loads(b)  # noqa: S301 - ok es objeto local
        assert t == t2

    def test_determinismo_defaults_equals(self):
        assert LanguageDetectionThresholds() == LanguageDetectionThresholds()


# =============================================================================
# 6) validate_alternatives_sorted_desc
# =============================================================================


class TestValidateAlternatives:
    @staticmethod
    def _alt(code: str, name: str, conf: float, rank: int) -> LanguageAlternative:
        return LanguageAlternative(language_code=code, language_name=name, confidence=conf, rank=rank)

    def test_empty_ok(self):
        res = validate_alternatives_sorted_desc(tuple(), 3)
        assert len(res) == 0

    def test_desc_ok(self):
        alts = (
            self._alt("en", "English", 0.5, 1),
            self._alt("pt", "Portuguese", 0.3, 2),
        )
        res = validate_alternatives_sorted_desc(alts, 3)
        assert len(res) == 2

    def test_ascending_raises_not_sorted(self):
        alts = (
            self._alt("en", "English", 0.3, 1),
            self._alt("pt", "Portuguese", 0.5, 2),
        )
        with pytest.raises(ValueError):
            validate_alternatives_sorted_desc(alts, 3)

    def test_duplicate_rank_raises(self):
        alts = (
            self._alt("en", "English", 0.5, 1),
            self._alt("pt", "Portuguese", 0.3, 1),
        )
        with pytest.raises(ValueError):
            validate_alternatives_sorted_desc(alts, 3)

    def test_duplicate_code_raises(self):
        alts = (
            self._alt("en", "English", 0.5, 1),
            self._alt("en", "English2", 0.3, 2),
        )
        with pytest.raises(ValueError):
            validate_alternatives_sorted_desc(alts, 3)

    def test_topk_exceed_raises(self):
        # Usar códigos ISO válidos. rank ≤3 en Field, así que 4 items con ranks 1,2,3,1;
        # validate_alternatives_sorted_desc primero detecta len>top_k antes que duplicate rank.
        alts = (
            self._alt("en", "English", 0.5, 1),
            self._alt("es", "Spanish", 0.3, 2),
            self._alt("pt", "Portuguese", 0.15, 3),
            self._alt("fr", "French", 0.05, 1),
        )
        with pytest.raises(ValueError):
            validate_alternatives_sorted_desc(alts, 3)


# =============================================================================
# 7) VO globales: sin imports infraestructura
# =============================================================================


def test_todos_los_vo_no_tienen_dependencias_infraestructura():
    import ast
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "app" / "domain" / "value_objects" / "lid.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    forbidden = {
        "torch", "transformers", "numpy", "soundfile", "faster_whisper",
        "onnxruntime", "requests", "urllib", "huggingface_hub",
    }
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    bad = imports & forbidden
    assert not bad, f"Imports prohibidos en Domain VO LID: {bad}"
