"""Tests del modelo de progreso por etapas (presentation/api/progress.py)."""
from __future__ import annotations

import pytest

from app.presentation.api.progress import (
    PIPELINE_STAGES,
    ProgressTracker,
    canonical_key,
    stage_percent,
    stage_spec,
    stage_status,
)


class TestPipelineStages:
    def test_percentages_estrictamente_crecientes(self) -> None:
        percents = [spec.percent for spec in PIPELINE_STAGES]
        assert percents == sorted(percents), "el catálogo debe ir en orden creciente"
        assert len(set(percents)) == len(percents), "no puede haber dos etapas con el mismo %"

    def test_rango_valido(self) -> None:
        for spec in PIPELINE_STAGES:
            assert 0 <= spec.percent <= 100, spec
            assert spec.key, spec
            assert spec.status, spec
            assert spec.label_es and spec.label_en, spec

    def test_empieza_en_cola_y_termina_en_100(self) -> None:
        assert PIPELINE_STAGES[0].key == "QUEUED"
        assert PIPELINE_STAGES[0].percent == 0
        assert PIPELINE_STAGES[-1].percent == 100

    def test_cada_etapa_activa_tiene_nombre_propio(self) -> None:
        """Ninguna etapa del pipeline puede quedar oculta tras un genérico."""
        keys = {spec.key for spec in PIPELINE_STAGES}
        for required in (
            "MEDIA_PREP", "VAD", "LID", "ASR", "TIMESTAMPS", "DIARIZATION",
            "ALIGNMENT", "QUALITY", "VALIDATION", "METRICS", "OUTPUT",
            "TRANSLATING", "GENERATING_SUBTITLES",
        ):
            assert required in keys, f"falta la etapa {required}"


class TestCanonicalKey:
    @pytest.mark.parametrize(
        ("step", "expected"),
        [
            ("T01_T02_T03", "MEDIA_PREP"),
            ("T01_MEDIA_VALIDATION", "MEDIA_PREP"),
            ("T02_AUDIO_EXTRACTION", "MEDIA_PREP"),
            ("T03_AUDIO_PREPROCESSING", "MEDIA_PREP"),
            ("T04_VAD", "VAD"),
            ("T05_LID", "LID"),
            ("T06_ASR", "ASR"),
            ("T07_TIMESTAMPS", "TIMESTAMPS"),
            ("T08_DIARIZATION", "DIARIZATION"),
            ("T09_ALIGNMENT", "ALIGNMENT"),
            ("T10_QUALITY", "QUALITY"),
            ("T11_VALIDATION", "VALIDATION"),
            ("T12_METRICS", "METRICS"),
            ("T13_OUTPUT", "OUTPUT"),
            ("T15_TRANSLATION_es", "TRANSLATING"),
            ("T15_TRANSLATION_en", "TRANSLATING"),
            ("T15", "TRANSLATING"),
            ("GENERATING_SUBTITLES", "GENERATING_SUBTITLES"),
            ("T14_CLEANUP", "COMPLETED"),
        ],
    )
    def test_mapeo_de_pasos_reales(self, step: str, expected: str) -> None:
        assert canonical_key(step) == expected

    def test_desconocido_cae_a_generico(self) -> None:
        assert canonical_key("PASO_RARO") == "PROCESSING"
        assert canonical_key("") == "PROCESSING"

    def test_orden_de_ejecucion_es_creciente(self) -> None:
        """La secuencia real del pipeline nunca hace retroceder el porcentaje."""
        sequence = [
            "QUEUED", "PREPARING", "T01_MEDIA_VALIDATION", "T02_AUDIO_EXTRACTION",
            "T03_AUDIO_PREPROCESSING", "T04_VAD", "T05_LID", "T06_ASR",
            "T07_TIMESTAMPS", "T08_DIARIZATION", "T09_ALIGNMENT", "T10_QUALITY",
            "T11_VALIDATION", "T12_METRICS", "T13_OUTPUT", "T15_TRANSLATION_es",
            "GENERATING_SUBTITLES", "T14_CLEANUP",
        ]
        percents = [stage_percent(canonical_key(step)) for step in sequence]
        assert percents == sorted(percents), f"retroceso detectado: {percents}"


class TestProgressTracker:
    def test_monotono_no_decreciente(self) -> None:
        tracker = ProgressTracker()
        sequence = [
            "QUEUED", "PREPARING", "MEDIA_PREP", "VAD", "LID", "ASR",
            "TIMESTAMPS", "DIARIZATION", "ALIGNMENT", "QUALITY", "VALIDATION",
            "METRICS", "OUTPUT", "TRANSLATING", "GENERATING_SUBTITLES",
        ]
        last = -1
        for key in sequence:
            update = tracker.announce(key)
            assert update is not None, key
            _status, percent, _stage = update
            assert percent >= last, f"{key} retrocedió de {last} a {percent}"
            last = percent

    def test_repetir_etapa_no_reporta_nada(self) -> None:
        tracker = ProgressTracker()
        assert tracker.announce("VAD") is not None
        assert tracker.announce("VAD") is None, "una etapa repetida no debe reescribirse"

    def test_volver_a_una_etapa_previa_mantiene_el_porcentaje(self) -> None:
        tracker = ProgressTracker()
        tracker.announce("ASR")
        high = tracker.percent
        update = tracker.announce("VAD")
        assert update is not None
        _status, percent, stage = update
        assert percent >= high, "nunca puede bajar"
        assert stage == "VAD", "la etapa sí se reporta para diagnóstico"

    def test_fin_es_idempotente_y_llega_a_100(self) -> None:
        tracker = ProgressTracker()
        tracker.announce("GENERATING_SUBTITLES")
        status, percent, stage = tracker.finish()
        assert percent == 100
        assert status == "COMPLETED"
        assert stage == "COMPLETED"
        again = tracker.finish()
        assert again[1] == 100

    def test_tracker_nuevo_empieza_en_cero(self) -> None:
        assert ProgressTracker().percent == 0


class TestStageLookup:
    def test_spec_desconocida_devuelve_generico(self) -> None:
        spec = stage_spec("NO_EXISTE")
        assert spec.key == "PROCESSING"
        assert stage_status("NO_EXISTE") == "PROCESSING"
        assert 0 < stage_percent("NO_EXISTE") < 100
