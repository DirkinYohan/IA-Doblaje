"""Tests T06 Application RunASRUseCase — FAKE ASRPort.
- 16+ tests
- NO torch / faster_whisper / numpy para UseCase
- Fake adapter que devuelve ASRResult mockeado
- Chain of custody 3 variantes mismatch
- strict / non-strict
- silent fallback num_intervals=0 (NO WhisperModel llamado)
- determinismo
- propagation ModelLoadError / ASRProcessingError / ValidationFailedError
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.core.exceptions import (
    ASRProcessingError,
    ModelLoadError,
    ValidationFailedError,
)
from app.core.paths import generate_job_id
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import (
    ExtractedAudio,
    MediaPrepResult,
    PreprocessedAudio,
    ValidatedMedia,
)
from app.domain.entities.vad import VadResult
from app.domain.value_objects.asr import AsrThresholds
from app.domain.value_objects.lid import LanguageDetectionThresholds
from app.domain.value_objects.vad import VadThresholds, VoiceInterval
from app.application.use_cases.run_asr import RunASRUseCase


_SHA_A = "a" * 16 + "0" * 48
_SHA_B = "b" * 16 + "0" * 48
_SHA_C = "c" * 16 + "0" * 48


# =============================================================================
# Factories
# =============================================================================
def _dummy_wav(tmp_path_factory) -> Path:
    wp = tmp_path_factory.mktemp("asrusecase") / "in.wav"
    wp.write_bytes(
        # 44 byte header wav 16kHz mono 16-bit 50ms = 1600 samples = 3200 bytes data
        b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    return wp


def _build_prep(
    *,
    wav_path: Path,
    sha256: str,
    sample_rate: int = 16000,
    channels: int = 1,
    bit_depth: int = 16,
    duration_ms: int = 5000,
) -> PreprocessedAudio:
    size_bytes = max(44, int(sample_rate * channels * bit_depth / 8 * duration_ms / 1000) + 44)
    return PreprocessedAudio.model_construct(
        wav_path=Path(wav_path).resolve(),
        sample_rate=int(sample_rate),
        channels=int(channels),
        bit_depth=int(bit_depth),
        duration_sec=float(duration_ms) / 1000.0,
        file_size_bytes=int(size_bytes),
        sha256=str(sha256),
        applied_filters=("peak_norm",),
    )


def _build_vad(
    *,
    source_sha: str,
    intervals,  # list[(start_ms, end_ms, conf)] or []
    duration_ms: int = 5000,
):
    vints = tuple(
        VoiceInterval.model_construct(
            start_ms=int(s), end_ms=int(e),
            max_confidence=float(c),
            sample_count=int(round(((e - s) * 16000) / 1000)),
        )
        for (s, e, c) in intervals
    )
    speech_ms = sum(int(e) - int(s) for s, e, _ in intervals)
    ratio = speech_ms / duration_ms if duration_ms > 0 else 0.0
    return VadResult.model_construct(
        source_preprocessed_sha256=str(source_sha),
        voice_intervals=vints,
        silence_segments=tuple(),
        speech_ratio=float(ratio),
        total_speech_ms=int(speech_ms),
        total_silence_ms=max(0, duration_ms - speech_ms),
        num_intervals=int(len(intervals)),
        num_silences=0,
        thresholds=VadThresholds(),
        model_label="silero-vad:v5.1",  # type: ignore[arg-type]
        sample_rate=16000,
        channels=1,
        duration_ms=int(duration_ms),
    )


def _build_lid(
    *,
    source_sha: str,
    language_code: str = "es",
    confidence: float = 0.9,
    strategy: str = "only_voice_intervals",
):
    from app.domain.value_objects.lid import LanguageConfidence, LanguageCode
    from app.domain.entities.lid import MODEL_LABEL_WHISPER_LID_V1_LITERAL

    return LanguageDetectionResult.model_construct(
        source_preprocessed_sha256=str(source_sha),
        vad_reference_sha256=str(source_sha),
        language_code=str(language_code),  # type: ignore[arg-type]
        language_name={"es": "Spanish", "en": "English", "und": "Undetermined"}.get(
            language_code, "Unknown"
        ),
        confidence=float(confidence),  # type: ignore[arg-type]
        alternatives=(),
        num_alternatives=0,
        analyzed_duration_ms=5000,
        total_duration_ms=5000,
        analyzed_strategy=str(strategy),  # type: ignore[arg-type]
        model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,  # type: ignore[arg-type]
        thresholds=LanguageDetectionThresholds(),
        analysis_metadata={},
    )


# =============================================================================
# Fake ASRPort (SUCCESS siempre que no sea error)
# =============================================================================
@dataclass
class FakeOKASRPort:
    model_loaded_called: int = 0
    transcript_text: str = "Hola, esto es una prueba."
    lang_code: str = "es"
    confidence: float = 0.85
    num_segments: int = 2

    def transcribe(
        self,
        preprocessed_audio: Any,
        *,
        vad_result: Any,
        lid_result: Any,
        thresholds: Any | None = None,
        job_id: Any | None = None,
    ) -> ASRResult:
        self.model_loaded_called += 1
        thr = thresholds if isinstance(thresholds, AsrThresholds) else AsrThresholds()
        segs = tuple(
            ASRSegment.model_construct(
                segment_index=i,
                text=self.transcript_text.split()[i] if i < len(self.transcript_text.split()) else f"seg{i}",
                start_ms=1000 * i,
                end_ms=1000 * (i + 1),
                avg_logprob=-0.5,
                no_speech_prob=0.02,
            )
            for i in range(self.num_segments)
        )
        return ASRResult.build_chain(
            preprocessed_audio=preprocessed_audio,
            vad_result=vad_result,
            lid_result=lid_result,
            transcript_text=self.transcript_text,
            transcript_language_code=self.lang_code,
            confidence=self.confidence,
            segments=segs,
            num_segments=self.num_segments,
            strategy="full_audio",
            model_label="fake:asr:v1",
            thresholds_used=thr,
            job_id=job_id,
            analysis_metadata={"fake": True},
        )


@dataclass
class FakeErrorASRPort:
    exc_class: type = ModelLoadError
    exc_msg: str = "modelo no encontrado"

    def transcribe(self, *a, **kw):
        raise self.exc_class(self.exc_msg)


# =============================================================================
# Tests Application — 16+
# =============================================================================
class TestRunASRUseCaseHappyPath:
    def test_happy_path_es(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 2500, 0.9), (3000, 4500, 0.8)])
        lid = _build_lid(source_sha=_SHA_A, language_code="es", confidence=0.9)
        fake = FakeOKASRPort()
        uc = RunASRUseCase(asr_port=fake, strict=False)
        res = uc.run(prep, vad, lid, job_id=generate_job_id())
        assert isinstance(res, ASRResult)
        assert res.source_preprocessed_sha256 == _SHA_A
        assert res.vad_reference_sha256 == _SHA_A
        assert res.lid_reference_sha256 == _SHA_A
        assert res.strategy == "full_audio"
        assert res.confidence == pytest.approx(0.85)
        assert res.num_segments == 2
        assert res.transcript_language_code == "es"
        assert fake.model_loaded_called == 1

    def test_language_und_passes_to_adapter(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 5000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A, language_code="und", confidence=0.0, strategy="silent_fallback_none")
        fake = FakeOKASRPort()
        uc = RunASRUseCase(asr_port=fake)
        res = uc.run(prep, vad, lid)
        # Result OK; und pasa al adapter -> usa auto detect (adapter resuelve)
        assert res.strategy == "full_audio"


class TestRunASRUseCaseChainOfCustodyMismatch:
    def test_prep_vad_mismatch(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_B, intervals=[(0, 1000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A)
        uc = RunASRUseCase(asr_port=FakeOKASRPort())
        with pytest.raises(ValidationFailedError):
            uc.run(prep, vad, lid)

    def test_prep_lid_mismatch(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 1000, 0.9)])
        lid = _build_lid(source_sha=_SHA_B)
        uc = RunASRUseCase(asr_port=FakeOKASRPort())
        with pytest.raises(ValidationFailedError):
            uc.run(prep, vad, lid)

    def test_vad_lid_mismatch(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_B, intervals=[(0, 1000, 0.9)])
        lid = _build_lid(source_sha=_SHA_C)
        uc = RunASRUseCase(asr_port=FakeOKASRPort())
        with pytest.raises(ValidationFailedError):
            uc.run(prep, vad, lid)


class TestRunASRUseCaseNoneInputs:
    def test_prep_none(self):
        uc = RunASRUseCase(asr_port=FakeOKASRPort())
        with pytest.raises(ValidationFailedError):
            uc.run(None, None, None)

    def test_vad_none(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        uc = RunASRUseCase(asr_port=FakeOKASRPort())
        with pytest.raises(ValidationFailedError):
            uc.run(prep, None, None)  # type: ignore[arg-type]


class TestRunASRUseCaseStrictConfidence:
    def test_strict_raises_when_conf_below_threshold(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 5000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A)
        fake = FakeOKASRPort(confidence=0.3, num_segments=1)
        thr = AsrThresholds(min_transcript_confidence=0.6)
        uc = RunASRUseCase(asr_port=fake, strict=True)
        with pytest.raises(ValidationFailedError, match="confidence="):
            uc.run(prep, vad, lid, thresholds=thr)

    def test_nonstrict_no_raise(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 5000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A)
        fake = FakeOKASRPort(confidence=0.1, num_segments=1)
        thr = AsrThresholds(min_transcript_confidence=0.6)
        uc = RunASRUseCase(asr_port=fake, strict=False)
        res = uc.run(prep, vad, lid, thresholds=thr)
        assert res.confidence == pytest.approx(0.1)


class TestRunASRUseCaseSilentFallback:
    def test_silent_fallback_no_model_called(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A, duration_ms=10000)
        vad = _build_vad(source_sha=_SHA_A, intervals=[], duration_ms=10000)
        lid = _build_lid(source_sha=_SHA_A)
        fake = FakeOKASRPort()
        uc = RunASRUseCase(asr_port=fake, strict=True)
        res = uc.run(prep, vad, lid)
        # Model NO debe ser llamado; strategy = silent fallback, conf=0, vacío
        assert fake.model_loaded_called == 0  # KEY: no carga ASR
        assert res.strategy == "asr_silent_fallback"
        assert res.transcript_text == ""
        assert res.num_segments == 0
        assert len(res.segments) == 0
        assert res.confidence == 0.0
        assert res.transcript_language_code == "und"

    def test_silent_fallback_works_regardless_strict(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A, duration_ms=3000)
        vad = _build_vad(source_sha=_SHA_A, intervals=[], duration_ms=3000)
        lid = _build_lid(source_sha=_SHA_A)
        fake = FakeOKASRPort()
        uc_strict = RunASRUseCase(asr_port=fake, strict=True)
        uc_relax = RunASRUseCase(asr_port=fake, strict=False)
        r1 = uc_strict.run(prep, vad, lid)
        r2 = uc_relax.run(prep, vad, lid)
        assert r1.strategy == r2.strategy == "asr_silent_fallback"


class TestRunASRUseCaseErrorPropagation:
    def test_propagate_model_load_error(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 1000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A)
        uc = RunASRUseCase(asr_port=FakeErrorASRPort(ModelLoadError, "modelo no cargado"))
        with pytest.raises(ModelLoadError):
            uc.run(prep, vad, lid)

    def test_propagate_asr_processing_error(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 1000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A)
        uc = RunASRUseCase(asr_port=FakeErrorASRPort(ASRProcessingError, "decoder error"))
        with pytest.raises(ASRProcessingError):
            uc.run(prep, vad, lid)


class TestRunASRUseCaseNoMutationDeterminism:
    def test_no_mutation_inputs(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad = _build_vad(source_sha=_SHA_A, intervals=[(0, 5000, 0.9)])
        lid = _build_lid(source_sha=_SHA_A)
        id_prep = id(prep)
        id_vad = id(vad)
        id_lid = id(lid)
        prep_sha_before = prep.sha256
        vad_n_before = vad.num_intervals
        lid_lang_before = lid.language_code
        fake = FakeOKASRPort()
        uc = RunASRUseCase(asr_port=fake)
        uc.run(prep, vad, lid)
        assert id(prep) == id_prep
        assert id(vad) == id_vad
        assert id(lid) == id_lid
        assert prep.sha256 == prep_sha_before
        assert vad.num_intervals == vad_n_before
        assert lid.language_code == lid_lang_before

    def test_determinism_two_runs_equal(self, tmp_path_factory):
        wp = _dummy_wav(tmp_path_factory)
        prep1 = _build_prep(wav_path=wp, sha256=_SHA_A)
        prep2 = _build_prep(wav_path=wp, sha256=_SHA_A)
        vad1 = _build_vad(source_sha=_SHA_A, intervals=[(0, 1000, 0.9)])
        vad2 = _build_vad(source_sha=_SHA_A, intervals=[(0, 1000, 0.9)])
        lid1 = _build_lid(source_sha=_SHA_A)
        lid2 = _build_lid(source_sha=_SHA_A)
        fake1 = FakeOKASRPort()
        fake2 = FakeOKASRPort()
        uc1 = RunASRUseCase(asr_port=fake1)
        uc2 = RunASRUseCase(asr_port=fake2)
        jid = generate_job_id()
        r1 = uc1.run(prep1, vad1, lid1, job_id=jid)
        r2 = uc2.run(prep2, vad2, lid2, job_id=jid)
        assert r1.transcript_text == r2.transcript_text
        assert r1.confidence == pytest.approx(r2.confidence)
        assert r1.num_segments == r2.num_segments
        assert r1.strategy == r2.strategy
        assert r1.model_label == r2.model_label
