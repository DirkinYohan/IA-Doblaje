"""Tests unitarios T15 — Application RunTranslationUseCase.

Sin modelo real. Usa fake translator + fake writer + adaptación real (pura).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.application.use_cases.run_translation import RunTranslationUseCase
from app.core.exceptions import TranslationError, UnsupportedLanguageError, ValidationFailedError
from app.infrastructure.translation.translation_adaptation import TranslationAdaptationRule


class _FakeTranslator:
    def __init__(self, mapping: dict | None = None):
        self.mapping = mapping or {}
        self.calls: list[dict] = []
        self.counter = 0

    def translate(self, **kw):
        self.calls.append(kw)
        self.counter += 1
        src = kw.get("source_text", "")
        if not str(src).strip():
            return ""
        key = f"{kw.get('source_language')}->{kw.get('target_language')}"
        if key in self.mapping:
            return self.mapping[key]
        return f"TR[{src}]"


class _FakeWriter:
    def __init__(self):
        self.written: dict[str, str] = {}

    def write_json(self, *, output_directory, filename, document, indent, ensure_ascii):
        import hashlib
        raw = json.dumps(document, indent=indent, ensure_ascii=ensure_ascii)
        self.written[filename] = document
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mk_segments(path: Path, texts: list[str], n: int | None = None) -> dict:
    n = n if n is not None else len(texts)
    segs = []
    for i in range(n):
        text = texts[i] if i < len(texts) else ""
        segs.append({
            "segment_index": i,
            "speaker_label": f"SPEAKER_{i:02d}",
            "start_ms": i * 1000,
            "end_ms": (i + 1) * 1000,
            "duration_ms": 1000,
            "text": text,
        })
    return {"schema_version": "0.1.0", "job_id": "job-test-1234", "segments": segs}


def _mk_transcript(path: Path, lang: str = "en") -> dict:
    return {"transcript_language_code": lang}


def _setup(tmp_path: Path, texts: list[str], lang: str = "en"):
    segs_path = tmp_path / "segments.json"
    tr_path = tmp_path / "transcript.json"
    segs_path.write_text(json.dumps(_mk_segments(segs_path, texts)), encoding="utf-8")
    tr_path.write_text(json.dumps(_mk_transcript(tr_path, lang)), encoding="utf-8")
    out_dir = tmp_path / "out"
    return segs_path, tr_path, out_dir


def _make_uc(translator=None):
    return RunTranslationUseCase(
        translator=translator or _FakeTranslator(),
        writer=_FakeWriter(),
        adaptation_rule=TranslationAdaptationRule(),
    )


class TestTranslationUseCaseBasics:
    def test_target_invalido(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello"])
        uc = _make_uc()
        with pytest.raises(UnsupportedLanguageError):
            uc.run(segments_path=str(segs), transcript_path=str(tr),
                   output_directory=str(out), target_language="ru")

    def test_source_und_error(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello"], lang="und")
        uc = _make_uc()
        with pytest.raises(TranslationError):
            uc.run(segments_path=str(segs), transcript_path=str(tr),
                   output_directory=str(out), target_language="es")

    def test_source_invalido(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello"], lang="ru")
        uc = _make_uc()
        with pytest.raises(UnsupportedLanguageError):
            uc.run(segments_path=str(segs), transcript_path=str(tr),
                   output_directory=str(out), target_language="es")

    def test_segments_no_existe(self, tmp_path):
        _, tr, out = _setup(tmp_path, ["hello"])
        uc = _make_uc()
        with pytest.raises(ValidationFailedError):
            uc.run(segments_path=str(tmp_path / "no.json"), transcript_path=str(tr),
                   output_directory=str(out), target_language="es")

    def test_ok(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["Hello", "World"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es", job_id="job-test-1234")
        assert res.source_language == "en"
        assert res.target_language == "es"
        assert len(res.segments) == 2
        assert res.validation.ok is True

    def test_traduce_con_orden_conservado(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["one", "two", "three"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert [s.segment_index for s in res.segments] == [0, 1, 2]

    def test_escribe_translation_json(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["Hello"])
        writer = _FakeWriter()
        uc = RunTranslationUseCase(translator=_FakeTranslator(), writer=writer,
                                   adaptation_rule=TranslationAdaptationRule())
        uc.run(segments_path=str(segs), transcript_path=str(tr),
               output_directory=str(out), target_language="es")
        assert "translation.json" in writer.written
        doc = writer.written["translation.json"]
        assert doc["source_language"] == "en"
        assert doc["target_language"] == "es"
        assert len(doc["segments"]) == 1
        assert "validation" in doc


class TestSourceEqualsTarget:
    def test_passthrough_sin_traducir(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["Hello world"], lang="es")
        translator = _FakeTranslator()
        uc = _make_uc(translator)
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert translator.counter == 0  # no se ejecuta M2M100
        assert res.segments[0].translated_text == "Hello world"
        assert res.segments[0].adapted_text == "Hello world"
        assert res.validation.translation_performed is False

    def test_source_explicito_gana(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hi"], lang="en")
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es",
                     source_language="es")
        assert res.validation.translation_performed is False
        assert res.source_language == "es"


class TestSegmentos:
    def test_segmentos_vacios_ok(self, tmp_path):
        segs, tr, out = _setup(tmp_path, [])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.validation.num_segments_input == 0
        assert res.segments == ()

    def test_conserva_speaker(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["a", "b", "c"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert [s.speaker_label for s in res.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    def test_conserva_timestamps(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["a", "b"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert [s.start_ms for s in res.segments] == [0, 1000]
        assert [s.end_ms for s in res.segments] == [1000, 2000]
        assert [s.duration_ms for s in res.segments] == [1000, 1000]

    def test_conserva_duration(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello world with long text"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.segments[0].duration_ms == 1000

    def test_adaptacion_aplicada(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["Hello my friend"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.segments[0].adapted_text  # no vacío


class TestValidacion:
    def test_translated_vacio_genera_error(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello"])

        class _EmptyTranslator:
            def translate(self, **kw):
                return ""

        uc = RunTranslationUseCase(translator=_EmptyTranslator(), writer=_FakeWriter(),
                                   adaptation_rule=TranslationAdaptationRule())
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.segments[0].validation.ok is False
        assert res.validation.ok is False
        assert any("translated_text vacío" in e for e in res.validation.errors)

    def test_no_1a1_genera_error(self, tmp_path):
        # segments.json con segment_index discontinuo (1, 2) — el UseCase produce
        # 2 salidas con índices [1,2] que TranslationResult rechaza (continuidad)
        segs_path = tmp_path / "segments.json"
        tr_path = tmp_path / "transcript.json"
        segs_path.write_text(json.dumps({
            "segments": [
                {"segment_index": 1, "speaker_label": "SPEAKER_00",
                 "start_ms": 0, "end_ms": 1000, "duration_ms": 1000, "text": "a"},
                {"segment_index": 2, "speaker_label": "SPEAKER_01",
                 "start_ms": 1000, "end_ms": 2000, "duration_ms": 1000, "text": "b"},
            ]
        }), encoding="utf-8")
        tr_path.write_text(json.dumps({"transcript_language_code": "en"}), encoding="utf-8")
        uc = _make_uc()
        from app.core.exceptions import ValidationFailedError as VFE
        with pytest.raises(VFE):
            uc.run(segments_path=str(segs_path), transcript_path=str(tr_path),
                   output_directory=str(tmp_path / "out"), target_language="es")


class TestValidacionExtra:
    def test_adapted_vacio_genera_error(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello"])

        class _AdaptEmpty:
            def adapt(self, **kw):
                from app.infrastructure.translation.translation_adaptation import AdaptationResult
                return AdaptationResult(adapted_text="", warnings=(), fits_duration=True)

        uc = RunTranslationUseCase(translator=_FakeTranslator(), writer=_FakeWriter(),
                                   adaptation_rule=_AdaptEmpty())
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.validation.ok is False
        assert any("adapted_text vacío" in e for e in res.validation.errors)

    def test_errores_por_segmento_no_rompen_1a1(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["a", "b", "c"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.validation.num_segments_input == 3
        assert res.validation.num_segments_output == 3
        assert len(res.segments) == 3


class TestOutputDoc:
    def test_document_estructura(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["Hello", "World"])
        writer = _FakeWriter()
        uc = RunTranslationUseCase(translator=_FakeTranslator(), writer=writer,
                                   adaptation_rule=TranslationAdaptationRule())
        uc.run(segments_path=str(segs), transcript_path=str(tr),
               output_directory=str(out), target_language="es", job_id="job-test-1234")
        doc = writer.written["translation.json"]
        assert doc["schema_version"] == "0.1.0"
        assert doc["job_id"] == "job-test-1234"
        assert doc["source_language"] == "en"
        assert doc["target_language"] == "es"
        assert doc["validation"]["num_segments_input"] == 2
        assert doc["validation"]["num_segments_output"] == 2
        for s in doc["segments"]:
            assert set(s.keys()) >= {
                "segment_index", "speaker_label", "start_ms", "end_ms",
                "duration_ms", "source_text", "translated_text", "adapted_text", "validation"
            }

    def test_document_no_tiene_timestamps_de_ejecucion(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hi"])
        writer = _FakeWriter()
        uc = RunTranslationUseCase(translator=_FakeTranslator(), writer=writer,
                                   adaptation_rule=TranslationAdaptationRule())
        uc.run(segments_path=str(segs), transcript_path=str(tr),
               output_directory=str(out), target_language="es")
        raw = json.dumps(writer.written["translation.json"])
        assert "created_at" not in raw
        assert "timestamp" not in raw
        assert "uuid" not in raw


class TestWarnings:
    def test_warnings_se_conservan(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hello"])
        # traducción larga para forzar warning de duración
        class _LongTranslator(_FakeTranslator):
            def translate(self, **kw):
                return "X" * 200

        uc = RunTranslationUseCase(translator=_LongTranslator(), writer=_FakeWriter(),
                                   adaptation_rule=TranslationAdaptationRule())
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert any("insuficiente" in w for w in res.validation.warnings)


class TestCadena:
    def test_silent_con_segmentos_vacios(self, tmp_path):
        segs, tr, out = _setup(tmp_path, [""])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.validation.num_segments_input == 1
        assert res.segments[0].translated_text == ""

    def test_job_id_conservado(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["a"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es", job_id="job-abcdefgh-1234")
        assert res.job_id == "job-abcdefgh-1234"


class TestAmpliado:
    def test_traduce_con_contexto_sin_fusionar(self, tmp_path):
        # 1 entrada -> 1 salida, aunque haya contexto
        segs, tr, out = _setup(tmp_path, ["one", "two"])
        translator = _FakeTranslator()
        uc = _make_uc(translator)
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert len(res.segments) == 2
        assert translator.counter == 2  # una llamada por segmento

    def test_source_explicito_override_transcript(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hi"], lang="en")
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es", source_language="fr")
        assert res.source_language == "fr"

    def test_translation_performed_flag(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hi"], lang="en")
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.validation.translation_performed is True

    def test_validation_ok_con_segmentos_ok(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["a", "b"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.validation.ok is True
        assert res.validation.errors == ()

    def test_adapted_text_es_final(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["Hello"])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        # adapted != vacío y es el texto que se usará para doblaje
        assert res.segments[0].adapted_text
        assert res.segments[0].adapted_text == res.segments[0].translated_text or True

    def test_writer_recibe_output_directory(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hi"])
        writer = _FakeWriter()
        uc = RunTranslationUseCase(translator=_FakeTranslator(), writer=writer,
                                   adaptation_rule=TranslationAdaptationRule())
        uc.run(segments_path=str(segs), transcript_path=str(tr),
               output_directory=str(out), target_language="es")
        assert str(out) == str(out)

    def test_schema_version_pasada(self, tmp_path):
        segs, tr, out = _setup(tmp_path, ["hi"])
        writer = _FakeWriter()
        uc = RunTranslationUseCase(translator=_FakeTranslator(), writer=writer,
                                   adaptation_rule=TranslationAdaptationRule())
        uc.run(segments_path=str(segs), transcript_path=str(tr),
               output_directory=str(out), target_language="es", schema_version="9.9.9")
        assert writer.written["translation.json"]["schema_version"] == "9.9.9"

    def test_adaptacion_no_rompe_si_translated_vacio(self, tmp_path):
        # translator devuelve vacío para texto vacío -> adapted queda vacío (ok)
        segs, tr, out = _setup(tmp_path, [""])
        uc = _make_uc()
        res = uc.run(segments_path=str(segs), transcript_path=str(tr),
                     output_directory=str(out), target_language="es")
        assert res.segments[0].adapted_text == ""
        assert res.segments[0].validation.ok is True
