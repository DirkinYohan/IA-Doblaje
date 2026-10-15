"""Tests unitarios T08 Infrastructure — Pyannote adapter + Loader.

No requiere pesos reales. Usa mocks de pyannote (Fake pipeline/loader).
AST security audit incluido.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.exceptions import ModelLoadError
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.diarization import (
    DiarizationThresholds,
    SpeakerTurn,
)
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.infrastructure.diarization.pyannote_diarization_adapter import (
    PIPELINE_CONFIG_FILENAME,
    MODEL_FOLDERNAME,
    PyannoteDiarizationModelLoader,
    PyannoteSpeakerDiarizationAdapter,
    _apply_domain_rules,
    _normalize_labels,
    _RawTurn,
)


_SHA = "a" * 64


def _mk_prep(duration_sec: float = 10.0, wav: str = "X:/d/x.wav") -> PreprocessedAudio:
    return PreprocessedAudio.model_construct(
        wav_path=Path(wav),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=duration_sec,
        size_bytes=1000,
        sha256=_SHA,
        applied_filters=(),
    )


def _mk_vad(
    intervals: list[tuple[int, int, float]] | None = None,
) -> VadResult:
    if intervals is None:
        intervals = [(0, 10000, 0.9)]
    vis = tuple(
        VoiceInterval(start_ms=s, end_ms=e, max_confidence=c) for s, e, c in intervals
    )
    speech_ms = sum(v.duration_ms for v in vis)
    return VadResult.model_construct(
        source_preprocessed_sha256=_SHA,
        voice_intervals=vis,
        silence_segments=(),
        speech_ratio=1.0 if intervals else 0.0,
        total_speech_ms=speech_ms,
        total_silence_ms=0,
        num_intervals=len(vis),
        num_silences=0,
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",
        sample_rate=16000,
        channels=1,
        duration_ms=10000,
    )


class _FakeSegment:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, turns: list[tuple[float, float, str]]):
        self.turns = turns

    def itertracks(self, yield_label: bool = False):
        for s, e, l in self.turns:
            yield (_FakeSegment(s, e), None, l)


class _FakePipeline:
    def __init__(self, annotation):
        self.annotation = annotation
        self.calls = 0

    def apply(self, wav_path, min_speakers=None, max_speakers=None):
        self.calls += 1
        return self.annotation


class _FakeLoader:
    def __init__(self, config_path: Path | None = None, pipeline=None, error: Exception | None = None):
        self.config_path = config_path
        self.pipeline = pipeline
        self.error = error
        self.validate_calls = 0
        self.load_calls = 0

    def validate(self) -> Path:
        self.validate_calls += 1
        if self.error is not None:
            raise self.error
        return self.config_path or Path("X:/models/pyannote/config.yaml")

    def load(self, config_path: Path):
        self.load_calls += 1
        return self.pipeline


# =============================================================================
# _normalize_labels (pure)
# =============================================================================


class TestNormalizeLabels:
    def test_valid_labels_kept(self):
        out = _normalize_labels([(0, 100, "SPEAKER_00"), (100, 200, "SPEAKER_01")])
        assert [t.label for t in out] == ["SPEAKER_00", "SPEAKER_01"]

    def test_invalid_labels_remapped_by_first_appearance(self):
        out = _normalize_labels([(0, 100, "X"), (100, 200, "Y"), (200, 300, "X")])
        assert [t.label for t in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]

    def test_lowercase_invalid_remapped(self):
        out = _normalize_labels([(0, 100, "speaker_00")])
        assert out[0].label == "SPEAKER_00"

    def test_empty(self):
        assert _normalize_labels([]) == []


# =============================================================================
# _apply_domain_rules (pure, determinista)
# =============================================================================


def _t(label: str, s: int, e: int) -> _RawTurn:
    return _RawTurn(s, e, label)


class TestDomainRules:
    def test_clamp_end_to_total(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 20000)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert out[0].end_ms == 10000

    def test_clamp_start_ge_zero(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", -50, 1000)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert out[0].start_ms == 0

    def test_overlap_trim(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 1000), _t("SPEAKER_01", 800, 1500)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert [(t.start_ms, t.end_ms) for t in out] == [(0, 1000), (1000, 1500)]

    def test_contained_turn_discarded(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 1000), _t("SPEAKER_01", 500, 700)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert [(t.start_ms, t.end_ms, t.label) for t in out] == [(0, 1000, "SPEAKER_00")]

    def test_contiguous_kept(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 1000), _t("SPEAKER_01", 1000, 1500)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert len(out) == 2

    def test_vad_intersection_outside_discarded(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 5000), _t("SPEAKER_01", 6000, 9000)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 5500)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert len(out) == 1
        assert out[0].label == "SPEAKER_00"

    def test_min_duration_discards_short(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 50)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=200),
        )
        assert out == []

    def test_merge_proximal_same_label(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 100), _t("SPEAKER_00", 120, 300)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(
                min_speaker_duration_ms=200, min_gap_ms=50
            ),
        )
        assert [(t.start_ms, t.end_ms) for t in out] == [(0, 300)]

    def test_dedupe(self):
        out = _apply_domain_rules(
            [_t("SPEAKER_00", 0, 1000), _t("SPEAKER_00", 0, 1000)],
            total_duration_ms=10000,
            vad_voice_intervals=[(0, 10000)],
            thresholds=DiarizationThresholds(min_speaker_duration_ms=10),
        )
        assert len(out) == 1

    def test_determinism_same_input(self):
        turns = [_t("SPEAKER_01", 800, 1500), _t("SPEAKER_00", 0, 1000)]
        a = _apply_domain_rules(turns, total_duration_ms=10000, vad_voice_intervals=[(0, 10000)], thresholds=DiarizationThresholds(min_speaker_duration_ms=10))
        b = _apply_domain_rules(turns, total_duration_ms=10000, vad_voice_intervals=[(0, 10000)], thresholds=DiarizationThresholds(min_speaker_duration_ms=10))
        assert a == b


# =============================================================================
# Loader (errores sin modelos reales)
# =============================================================================


class TestLoader:
    def _loader(self, tmp_path, config: str | None = None):
        models = tmp_path / "models"
        (models / MODEL_FOLDERNAME).mkdir(parents=True)
        if config is not None:
            (models / MODEL_FOLDERNAME / PIPELINE_CONFIG_FILENAME).write_text(config, encoding="utf-8")
        loader = PyannoteDiarizationModelLoader(settings_getter=lambda: _Settings(models))
        return loader, models

    def test_missing_dir_raises(self, tmp_path):
        loader = PyannoteDiarizationModelLoader(
            settings_getter=lambda: _Settings(tmp_path / "models")
        )
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_missing_config_raises(self, tmp_path):
        loader, _ = self._loader(tmp_path)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_config_remote_url_raises(self, tmp_path):
        cfg = "pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n  params:\n    segmentation: https://evil/x.ckpt\n"
        loader, _ = self._loader(tmp_path, config=cfg)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_config_use_auth_token_raises(self, tmp_path):
        cfg = "pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n  params:\n    use_auth_token: abc\n"
        loader, _ = self._loader(tmp_path, config=cfg)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_config_path_traversal_raises(self, tmp_path):
        cfg = "pipeline:\n  name: pyannote.audio.pipelines.SpeakerDiarization\n  params:\n    segmentation: ../../outside/x.ckpt\n"
        loader, _ = self._loader(tmp_path, config=cfg)
        with pytest.raises(ModelLoadError):
            loader.validate()

    def test_config_missing_pipeline_name_raises(self, tmp_path):
        cfg = "params: {}\n"
        loader, _ = self._loader(tmp_path, config=cfg)
        with pytest.raises(ModelLoadError):
            loader.validate()


class _Settings:
    def __init__(self, models_dir: Path):
        self.paths = _Paths(models_dir)


class _Paths:
    def __init__(self, models_dir: Path):
        self.models_cache_dir = models_dir


# =============================================================================
# Adapter (con mock pipeline)
# =============================================================================


class TestAdapter:
    def test_happy_path(self):
        ann = _FakeAnnotation([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])
        pipeline = _FakePipeline(ann)
        loader = _FakeLoader(pipeline=pipeline)
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        r = adapter.diarize(_mk_prep(), _mk_vad())
        assert isinstance(r, DiarizationResult)
        assert r.num_turns == 2
        assert r.num_speakers == 2
        assert pipeline.calls == 1

    def test_conversion_seconds_to_ms(self):
        ann = _FakeAnnotation([(0.0, 0.25, "SPEAKER_00")])
        pipeline = _FakePipeline(ann)
        loader = _FakeLoader(pipeline=pipeline)
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        r = adapter.diarize(_mk_prep(duration_sec=10.0), _mk_vad())
        assert r.speaker_turns[0].start_ms == 0
        assert r.speaker_turns[0].end_ms == 250

    def test_invalid_label_remapped(self):
        ann = _FakeAnnotation([(0.0, 1.0, "whatever")])
        pipeline = _FakePipeline(ann)
        loader = _FakeLoader(pipeline=pipeline)
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        r = adapter.diarize(_mk_prep(), _mk_vad())
        assert r.speaker_turns[0].speaker_label == "SPEAKER_00"

    def test_model_load_error_propagates(self):
        loader = _FakeLoader(error=ModelLoadError("modelo no encontrado"))
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        with pytest.raises(ModelLoadError):
            adapter.diarize(_mk_prep(), _mk_vad())

    def test_pipeline_singleton_cached(self):
        ann = _FakeAnnotation([(0.0, 1.0, "SPEAKER_00")])
        pipeline = _FakePipeline(ann)
        loader = _FakeLoader(pipeline=pipeline)
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        adapter.diarize(_mk_prep(), _mk_vad())
        adapter.diarize(_mk_prep(), _mk_vad())
        assert loader.load_calls == 1  # cached

    def test_determinism_two_runs(self):
        ann = _FakeAnnotation([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])
        loader = _FakeLoader(pipeline=_FakePipeline(ann))
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        r1 = adapter.diarize(_mk_prep(), _mk_vad())
        r2 = adapter.diarize(_mk_prep(), _mk_vad())
        assert r1.model_dump() == r2.model_dump()

    def test_does_not_mutate_inputs(self):
        ann = _FakeAnnotation([(0.0, 1.0, "SPEAKER_00")])
        loader = _FakeLoader(pipeline=_FakePipeline(ann))
        adapter = PyannoteSpeakerDiarizationAdapter(model_loader=loader)
        prep = _mk_prep()
        vad = _mk_vad()
        prep_dump = prep.model_dump()
        vad_dump = vad.model_dump()
        adapter.diarize(prep, vad)
        assert prep.model_dump() == prep_dump
        assert vad.model_dump() == vad_dump


# =============================================================================
# AST SECURITY AUDIT
# =============================================================================


_ADAPTER_PATH = Path(__file__).resolve().parents[3] / "app" / "infrastructure" / "diarization" / "pyannote_diarization_adapter.py"
_USECASE_PATH = Path(__file__).resolve().parents[3] / "app" / "application" / "use_cases" / "run_diarization.py"
_DOMAIN_VO_PATH = Path(__file__).resolve().parents[3] / "app" / "domain" / "value_objects" / "diarization.py"
_DOMAIN_ENT_PATH = Path(__file__).resolve().parents[3] / "app" / "domain" / "entities" / "diarization.py"
_DOMAIN_PORT_PATH = Path(__file__).resolve().parents[3] / "app" / "domain" / "interfaces" / "diarization_ports.py"


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


class TestASTAudit:
    @pytest.mark.parametrize(
        "path",
        [_DOMAIN_VO_PATH, _DOMAIN_ENT_PATH, _DOMAIN_PORT_PATH, _USECASE_PATH],
    )
    def test_domain_app_no_ai_imports(self, path: Path):
        imports = _top_level_imports(path)
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx"}
        assert not (imports & forbidden), f"{path.name} importa IA: {imports & forbidden}"

    def test_adapter_no_forbidden_identifiers(self):
        txt = _ADAPTER_PATH.read_text(encoding="utf-8")
        for kw in ["hf_hub_download", "use_auth_token=", "requests.", "urllib.", "httpx.", "http://", "https://"]:
            assert kw not in txt, f"Identificador prohibido {kw!r} en adapter T08."

    def test_adapter_imports_pyannote_only_lazy(self):
        # El adapter debe importar pyannote solo dentro de funciones (lazy), no a nivel de módulo.
        tree = ast.parse(_ADAPTER_PATH.read_text(encoding="utf-8"))
        module_level_imports: set[str] = set()
        for node in ast.walk(tree):
            # Solo imports en el cuerpo del módulo (no dentro de funciones)
            if isinstance(node, ast.Import) and isinstance(node, ast.Import):
                # distingue: los imports lazy están dentro de FunctionDef; aquí solo módulo-level
                pass
        # Comprobación: no debe haber import pyannote a nivel de módulo
        for node in ast.parse(_ADAPTER_PATH.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    module_level_imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_level_imports.add(node.module.split(".")[0])
        assert "pyannote" not in module_level_imports
        assert "speechbrain" not in module_level_imports
        assert "torch" not in module_level_imports
