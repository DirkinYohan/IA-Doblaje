"""Tests unitarios T05 Application RunLanguageDetectionUseCase — con FAKE Port.

No requiere modelo Whisper. No toca adapter. No requiere torch para UseCase tests (solo Fake Port se usa).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.core.constants import QualityProfile
from app.core.exceptions import (
    LanguageDetectionError,
    ModelLoadError,
    ValidationFailedError,
)
from app.core.paths import generate_job_id
from app.domain.entities.lid import (
    LID_STRATEGY_SILENT_FALLBACK,
    LID_STRATEGY_VOICE_ONLY,
    LanguageDetectionResult,
    MODEL_LABEL_WHISPER_LID_V1_LITERAL,
)
from app.domain.entities.media import (
    ExtractedAudio,
    MediaPrepResult,
    PreprocessedAudio,
    ValidatedMedia,
)
from app.domain.entities.vad import VadResult
from app.domain.value_objects.lid import (
    LanguageAlternative,
    LanguageDetectionThresholds,
)
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.application.use_cases.run_language_detection import RunLanguageDetectionUseCase


_SHA_A = "a" * 16 + "0" * 48
_SHA_B = "b" * 16 + "0" * 48


# =============================================================================
# Factory helpers (model_construct sin validation compleja T01 ValidatedMedia fields required
# =============================================================================


class _FakeMediaPrepFactory:
    @staticmethod
    def build(
        *,
        wav_path: Path,
        sha256: str,
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
        duration_sec: float = 5.0,
        size_bytes: int = 32000 * 5 + 44,
        preprocessed_exists: bool = True,
        job_id_override: str | None = None,
    ) -> MediaPrepResult:
        wp = Path(str(wav_path)).resolve()
        validated = ValidatedMedia.model_construct(
            original_path=wp,
            safe_name=wp.name if wp.name else "fake.wav",
            media_format="wav",  # type: ignore[arg-type]
            extension="wav",
            size_bytes=int(size_bytes),
            sha256=str(sha256),
            probed_info={},
            duration_guess_sec=float(duration_sec),
            probed_stream_count=1,
        )
        extracted = ExtractedAudio.model_construct(
            wav_path=wp,
            sample_rate=int(sample_rate),
            channels=int(channels),
            bit_depth=int(bit_depth),
            duration_sec=float(duration_sec),
            size_bytes=int(size_bytes),
            sha256=str(sha256),
            ffmpeg_duration_ms=int(round(float(duration_sec) * 1000)),
            probed_duration_ms=int(round(float(duration_sec) * 1000)),
            sanity_duration_pct_diff_ok=True,
        )
        if not preprocessed_exists:
            prep = None
        else:
            prep = PreprocessedAudio.model_construct(
                wav_path=wp,
                sample_rate=int(sample_rate),
                channels=int(channels),
                bit_depth=int(bit_depth),
                duration_sec=float(duration_sec),
                file_size_bytes=int(size_bytes),
                sha256=str(sha256),
                applied_filters=("peak_norm",),
            )
        return MediaPrepResult.model_construct(
            job_id=job_id_override or generate_job_id(),
            profile=QualityProfile.BALANCED,  # type: ignore[attr-defined]
            validated=validated,
            extracted_audio=extracted,
            preprocessed_audio=prep,
            step_times_sec={"validate": 0.1, "extract": 0.1, "preprocess": 0.1},
            ffmpeg_available=False,
            ffprobe_available=False,
            errors=[],
        )


def _build_vad(
    *,
    source_sha: str,
    intervals,  # list of (start_ms, end_ms, max_conf) or []
    duration_ms: int = 5000,
):
    vints: tuple[VoiceInterval, ...] = tuple(
        VoiceInterval(
            start_ms=int(s),
            end_ms=int(e),
            max_confidence=float(c),
            sample_count=int(round(((e - s) * 16000) / 1000)),
        )
        for (s, e, c) in intervals
    )
    speech_ms = sum(int(e) - int(s) for s, e, _ in intervals)
    silence_ms = max(0, duration_ms - speech_ms)
    ratio = (speech_ms / duration_ms) if duration_ms > 0 else 0.0
    return VadResult.model_construct(
        source_preprocessed_sha256=str(source_sha),
        voice_intervals=vints,
        silence_segments=tuple(),
        speech_ratio=float(ratio),
        total_speech_ms=int(speech_ms),
        total_silence_ms=int(silence_ms),
        num_intervals=int(len(intervals)),
        num_silences=0,
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",  # type: ignore[arg-type]
        sample_rate=16000,
        channels=1,
        duration_ms=int(duration_ms),
    )


# =============================================================================
# Fake Success Detector Port (no toca infraestructura)
# =============================================================================


@dataclass
class FakeSuccessLIDPort:
    code: str = "es"
    name: str = "Spanish"
    confidence: float = 0.92
    alt_codes: tuple[str, ...] = ("en", "pt")
    alt_names: tuple[str, ...] = ("English", "Portuguese")
    alt_confs: tuple[float, ...] = (0.05, 0.02)
    analyzed_ratio: float = 0.6
    raise_exc: type[Exception] | None = None
    raise_msg: str = "fake port error"

    def detect(
        self,
        preprocessed: Any,
        vad: Any,
        *,
        thresholds: Any | None = None,
        job_id: Any | None = None,
        logger: Any | None = None,
    ) -> LanguageDetectionResult:
        if self.raise_exc is not None:
            raise self.raise_exc(self.raise_msg)
        thr = thresholds or LanguageDetectionThresholds()
        alts: list[LanguageAlternative] = []
        rank = 1
        for c, n, cf in zip(self.alt_codes, self.alt_names, self.alt_confs):
            alts.append(LanguageAlternative(
                language_code=c, language_name=n, confidence=float(cf), rank=int(rank),
            ))
            rank += 1
        dur_ms = int(round(float(preprocessed.duration_sec) * 1000.0))
        return LanguageDetectionResult(
            source_preprocessed_sha256=str(preprocessed.sha256),
            vad_reference_sha256=str(vad.source_preprocessed_sha256),
            language_code=str(self.code),
            language_name=str(self.name),
            confidence=float(self.confidence),
            alternatives=tuple(alts),
            total_duration_ms=int(dur_ms),
            analyzed_duration_ms=int(float(dur_ms) * float(self.analyzed_ratio)),
            analyzed_strategy=LID_STRATEGY_VOICE_ONLY,  # type: ignore[arg-type]
            num_speech_intervals_considered=int(vad.num_intervals),
            model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,  # type: ignore[arg-type]
            sample_rate=int(preprocessed.sample_rate),
            channels=int(preprocessed.channels),
            bit_depth=int(preprocessed.bit_depth),
            thresholds=thr,
            analysis_metadata={"fake": True},
        )


# =============================================================================
# FIXTURE env
# =============================================================================


@pytest.fixture
def env(tmp_path):
    wav = tmp_path / "x.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    # Create stub file so Path.exists/is_file True
    wav.write_bytes(b"RIFF\xff\x00\x00\x00WAVEfmt " + b"\x00" * (32000 * 5))
    mp = _FakeMediaPrepFactory.build(wav_path=wav, sha256=_SHA_A, duration_sec=5.0)
    vad = _build_vad(
        source_sha=_SHA_A,
        intervals=[(500, 1500, 0.92), (2500, 4000, 0.96)],
        duration_ms=5000,
    )
    return mp, vad, wav


# =============================================================================
# TESTS HAPPY PATH
# =============================================================================


class TestUseCaseHappyPath:
    def test_happy_path_returns_result(self, env):
        mp, vad, _ = env
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        r = uc.execute(mp, vad)
        assert isinstance(r, LanguageDetectionResult)
        assert r.language_code == "es"
        assert r.language_name == "Spanish"
        assert r.confidence == 0.92
        assert r.analyzed_strategy == LID_STRATEGY_VOICE_ONLY

    def test_default_thresholds_04_applied(self, env):
        mp, vad, _ = env
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        r = uc.execute(mp, vad)
        assert abs(float(r.thresholds.min_confidence) - 0.4) < 1e-9

    def test_override_thresholds_applied(self, env):
        mp, vad, _ = env
        thr = LanguageDetectionThresholds(min_confidence=0.7, top_k=2, min_speech_ratio_to_analyze=0.0)
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        r = uc.execute(mp, vad, thresholds=thr)
        assert abs(float(r.thresholds.min_confidence) - 0.7) < 1e-9
        assert int(r.thresholds.top_k) == 2

    def test_top3_alternatives_ordered_desc(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(
            code="en",
            name="English",
            confidence=0.8,
            alt_codes=("es", "fr"),
            alt_names=("Spanish", "French"),
            alt_confs=(0.15, 0.03),
        )
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        r = uc.execute(mp, vad)
        confs = [float(a.confidence) for a in r.alternatives]
        assert confs == sorted(confs, reverse=True)
        codes = [a.language_code for a in r.alternatives]
        assert "en" not in codes

    def test_determinismo(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort()
        uc1 = RunLanguageDetectionUseCase(detector=port, strict=True)
        uc2 = RunLanguageDetectionUseCase(detector=port, strict=True)
        job = "det-job-" + "a" * 20
        r1 = uc1.execute(mp, vad, job_id=job)
        r2 = uc2.execute(mp, vad, job_id=job)
        assert r1.language_code == r2.language_code
        assert abs(float(r1.confidence) - float(r2.confidence)) < 1e-12
        assert r1.source_preprocessed_sha256 == r2.source_preprocessed_sha256
        assert [a.language_code for a in r1.alternatives] == [a.language_code for a in r2.alternatives]


# =============================================================================
# CHAIN OF CUSTODY
# =============================================================================


class TestChainOfCustody:
    def test_sha_mismatch_t03_t04_raises_validationfailed(self, env):
        mp, _vad_ok, _ = env
        vad_bad = _build_vad(source_sha=_SHA_B, intervals=[(0, 1000, 0.9)], duration_ms=5000)
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        with pytest.raises(ValidationFailedError, match="Chain of Custody"):
            uc.execute(mp, vad_bad)

    def test_sha_matches_triple_t03_t04_t05(self, env):
        mp, vad, _ = env
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        r = uc.execute(mp, vad)
        assert str(r.source_preprocessed_sha256).lower() == str(mp.preprocessed_audio.sha256).lower()
        assert str(r.vad_reference_sha256).lower() == str(vad.source_preprocessed_sha256).lower()


# =============================================================================
# PREP NONE strict/nonstrict
# =============================================================================


class TestPrepNone:
    def test_prep_none_strict_true_raises(self, env):
        _mp_ok, vad, wav = env
        mp_none = _FakeMediaPrepFactory.build(
            wav_path=wav, sha256=_SHA_A, duration_sec=5.0, preprocessed_exists=False,
        )
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        with pytest.raises(ValidationFailedError, match="preprocessed_audio None"):
            uc.execute(mp_none, vad)

    def test_prep_none_strict_false_returns_silent(self, env):
        _mp_ok, vad, wav = env
        mp_none = _FakeMediaPrepFactory.build(
            wav_path=wav, sha256=_SHA_A, duration_sec=5.0, preprocessed_exists=False,
        )
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=False)
        r = uc.execute(mp_none, vad)
        assert r.language_code == "und"
        assert r.language_name == "Undetermined"
        assert r.confidence == 0.0
        assert r.analyzed_strategy == LID_STRATEGY_SILENT_FALLBACK


# =============================================================================
# STRICT THRESHOLD D#8
# =============================================================================


class TestStrictThreshold:
    def test_strict_conf_035_below_04_raises_language_detection_error(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(code="de", name="German", confidence=0.35)
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        with pytest.raises(LanguageDetectionError, match="strict=True"):
            uc.execute(mp, vad)

    def test_nonstrict_conf_035_returns_und_00(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(code="de", name="German", confidence=0.35)
        uc = RunLanguageDetectionUseCase(detector=port, strict=False)
        r = uc.execute(mp, vad)
        assert r.language_code == "und"
        assert r.confidence == 0.0
        assert len(r.alternatives) == 0

    def test_strict_conf_05_above_04_ok(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(confidence=0.5)
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        r = uc.execute(mp, vad)
        assert r.confidence == 0.5

    def test_strict_with_override_lang_allows_pass_no_raise(self, env):
        mp, vad, _ = env
        thr = LanguageDetectionThresholds(min_confidence=0.8, default_language_override="it")
        port = FakeSuccessLIDPort(confidence=0.3)
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        # Con override presente, no debe raise LanguageDetectionError aunque conf < min.
        # Fake devuelve confidence 0.3 code es. UseCase no raisea porque default_language_override != None.
        r = uc.execute(mp, vad, thresholds=thr)
        # Pasamos sin excepción (UseCase no swallows). OK.
        assert r is not None


# =============================================================================
# SILENT FALLBACK vad sin voz
# =============================================================================


class TestSilentFallback:
    def test_zero_intervals_strict_returns_und_no_raise(self, env):
        mp, _vad, wav = env
        vad_zero = _build_vad(source_sha=_SHA_A, intervals=[], duration_ms=5000)
        assert vad_zero.num_intervals == 0
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        r = uc.execute(mp, vad_zero)
        assert r.language_code == "und"
        assert r.confidence == 0.0
        assert r.analyzed_strategy == LID_STRATEGY_SILENT_FALLBACK
        assert int(r.num_speech_intervals_considered) == 0

    def test_zero_intervals_name_undetermined(self, env):
        mp, _vad, wav = env
        vad_zero = _build_vad(source_sha=_SHA_A, intervals=[], duration_ms=5000)
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        r = uc.execute(mp, vad_zero)
        assert r.language_name == "Undetermined"


# =============================================================================
# NO MUTATION
# =============================================================================


class TestNoMutation:
    def test_no_mutate_media_prep(self, env):
        mp, vad, _ = env
        jid_before = str(mp.job_id)
        sha_before = str(mp.validated.sha256)
        prep_ref_before = mp.preprocessed_audio
        prep_sha_before = str(prep_ref_before.sha256) if prep_ref_before else None
        prep_sr_before = int(prep_ref_before.sample_rate) if prep_ref_before else None
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        uc.execute(mp, vad)
        assert str(mp.job_id) == jid_before
        assert str(mp.validated.sha256) == sha_before
        assert mp.preprocessed_audio is prep_ref_before
        if prep_sha_before is not None:
            assert str(mp.preprocessed_audio.sha256) == prep_sha_before
            assert int(mp.preprocessed_audio.sample_rate) == prep_sr_before

    def test_no_mutate_vad(self, env):
        mp, vad, _ = env
        sha_before = str(vad.source_preprocessed_sha256)
        ni_before = int(vad.num_intervals)
        speech_before = int(vad.total_speech_ms)
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort(), strict=True)
        uc.execute(mp, vad)
        assert str(vad.source_preprocessed_sha256) == sha_before
        assert int(vad.num_intervals) == ni_before
        assert int(vad.total_speech_ms) == speech_before


# =============================================================================
# EXCEPCIONES PROPAGACIÓN
# =============================================================================


class TestExceptionsPropagate:
    def test_model_load_error(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(raise_exc=ModelLoadError)
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        with pytest.raises(ModelLoadError):
            uc.execute(mp, vad)

    def test_language_detection_error(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(raise_exc=LanguageDetectionError)
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        with pytest.raises(LanguageDetectionError):
            uc.execute(mp, vad)

    def test_generic_runtime_wrapped_language_detection(self, env):
        mp, vad, _ = env
        port = FakeSuccessLIDPort(raise_exc=RuntimeError, raise_msg="boom unhandled")
        uc = RunLanguageDetectionUseCase(detector=port, strict=True)
        with pytest.raises(LanguageDetectionError):
            uc.execute(mp, vad)


# =============================================================================
# JOB ID
# =============================================================================


class TestJobId:
    def test_no_job_id_passed_ok(self, env):
        mp, vad, _ = env
        uc = RunLanguageDetectionUseCase(detector=FakeSuccessLIDPort())
        r = uc.execute(mp, vad)
        # No raise → ok
        assert isinstance(r, LanguageDetectionResult)


# =============================================================================
# Static audit: Application no imports torch/transformers/whisper/numpy/soundfile
# =============================================================================


def test_application_no_importa_infra():
    import ast
    p = Path(__file__).resolve().parents[2] / "app" / "application" / "use_cases" / "run_language_detection.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    forbidden = {
        "torch", "transformers", "whisper", "soundfile", "onnxruntime",
        "faster_whisper", "numpy", "silero",
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
    assert not bad, f"Application imports infra prohibidos: {bad}"
