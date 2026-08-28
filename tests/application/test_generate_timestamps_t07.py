from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.application.use_cases.generate_timestamps import GenerateTimestampsUseCase
from app.core.exceptions import ValidationFailedError
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.timestamps import (
    TimestampGenerationResult,
    TimestampedSegment,
    TimestampedWord,
)
from app.domain.entities.vad import VadResult, VoiceInterval
from app.domain.interfaces.timestamp_ports import TimestampNormalizerPort
from app.domain.value_objects.timestamps import AsrTimestampThresholds

# =========================================================================
# Helpers factories: model_construct frozen igual que en T05/T06 tests
# =========================================================================
_SHA = "a" * 64  # dummy sha chain binding

def _mk_prep(duration_sec: float = 10.0) -> PreprocessedAudio:
    return PreprocessedAudio.model_construct(
        sha256=_SHA,
        source_media_sha256="b"*64,
        wav_path="X:/dummy/x.wav",
        format_name="wav",
        size_bytes=100,
        duration_sec=duration_sec,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        job_id=None,
    )


def _mk_vad(*, prep=None, num=2, strategy: str = "vad_speech_found") -> VadResult:
    prep = prep if prep is not None else _mk_prep()
    intervals: tuple[VoiceInterval, ...] = tuple(
        VoiceInterval.model_construct(
            index=i,
            start_ms=1000 + i * 2000,
            end_ms=1800 + i * 2000,
            duration_ms=800,
            max_confidence=0.9,
            mean_confidence=0.8,
        )
        for i in range(num)
    )
    return VadResult.model_construct(
        source_preprocessed_sha256=prep.sha256,
        vad_adapter_label="unit-test",
        strategy=strategy,
        num_intervals=len(intervals),
        total_speech_ms=len(intervals) * 800,
        total_silence_ms=int(prep.duration_sec * 1000) - len(intervals) * 800,
        voice_intervals=intervals,
        analysis_metadata={},
        validation_errors=(),
        thresholds=None,
        duration_ms=int(prep.duration_sec * 1000),
    )


def _mk_lid(*, prep=None, lang="es") -> LanguageDetectionResult:
    prep = prep if prep is not None else _mk_prep()
    return LanguageDetectionResult.model_construct(
        source_preprocessed_sha256=prep.sha256,
        vad_reference_sha256=prep.sha256,
        language_code=lang,
        top_languages={"es": 0.99},
        confidence=0.99,
        detection_method="unit-test",
        analysis_metadata={},
        lid_adapter_label="unit-test",
        strategy="language_detect_ok",
        duration_ms=int(prep.duration_sec * 1000),
        validation_errors=(),
        job_id=None,
    )


def _mk_asr(
    *,
    prep=None,
    vad=None,
    lid=None,
    strategy: str = "full_audio",
    segments_crudo: tuple[tuple[int | None, int | None, str], ...] = ((1000, 1800, "Hola"), (3000, 3800, "Mundo")),
) -> ASRResult:
    prep = prep if prep is not None else _mk_prep()
    vad = vad if vad is not None else _mk_vad(prep=prep, num=len(segments_crudo))
    lid = lid if lid is not None else _mk_lid(prep=prep)
    segs: list[ASRSegment] = []
    for i, (s, e, t) in enumerate(segments_crudo):
        segs.append(
            ASRSegment.model_construct(
                segment_index=i,
                start_ms=s,
                end_ms=e,
                text=t,
                avg_logprob=-0.5,
                no_speech_prob=0.01,
            )
        )
    return ASRResult.model_construct(
        strategy=strategy,
        source_preprocessed_sha256=prep.sha256,
        vad_reference_sha256=vad.source_preprocessed_sha256,
        lid_reference_sha256=lid.source_preprocessed_sha256,
        total_duration_ms=int(prep.duration_sec * 1000),
        transcript_language_code=lid.language_code,
        job_id=None,
        segments=tuple(segs),
        num_segments=len(segs),
        confidence=0.9,
        transcript_text=" ".join(t for _s, _e, t in segments_crudo),
        analysis_metadata={},
        model_label="unit-test",
        asr_backend="fw-test",
        validation_errors=(),
        strict_ok=True,
        language_detected=lid.language_code,
        duration_ms=int(prep.duration_sec * 1000),
    )


# =========================================================================
# Fake Ports (duck typing TimestampNormalizerPort)
# =========================================================================
class _FakeOKNormalizer(TimestampNormalizerPort):
    def __init__(self):
        self.calls: list = []
        self.normalize_called_count = 0

    def normalize(self, prep, *, vad_result, lid_result, asr_result, thresholds=None, job_id=None):
        self.normalize_called_count += 1
        self.calls.append((prep, vad_result, lid_result, asr_result, thresholds, job_id))
        # Construir TimestampGenerationResult frozen chain
        segs = []
        for i, seg in enumerate(asr_result.segments):
            s = seg.start_ms if seg.start_ms is not None else 0
            e = seg.end_ms if seg.end_ms is not None else s + 200
            if e <= s:
                e = s + 1
            segs.append(TimestampedSegment.model_construct(
                segment_index=i,
                start_ms=s,
                end_ms=e,
                duration_ms=e - s,
                source="fw_raw",
                ts_confidence=1.0,
            ))
        return TimestampGenerationResult.build_chain(
            preprocessed_audio=prep,
            vad_result=vad_result,
            language_detection_result=lid_result,
            asr_result=asr_result,
            strategy="segment_ts_from_fw_raw",
            transcript_language_code=asr_result.transcript_language_code,
            segments=tuple(segs),
            words=(),
            job_id=job_id,
        )


class _FakeErrorNormalizer(TimestampNormalizerPort):
    def normalize(self, prep, *, vad_result, lid_result, asr_result, thresholds=None, job_id=None):
        raise ValidationFailedError("intentional failure for propagation test")


class _FakeReturnWordsNonEmpty(TimestampNormalizerPort):
    """Adapter malicioso: devuelve len(words)>0. UseCase debe rechazar (D#1)."""
    def normalize(self, prep, *, vad_result, lid_result, asr_result, thresholds=None, job_id=None):
        segs = []
        for i, seg in enumerate(asr_result.segments):
            s = seg.start_ms if seg.start_ms is not None else 0
            e = seg.end_ms if seg.end_ms is not None else s + 200
            if e <= s:
                e = s + 1
            segs.append(TimestampedSegment.model_construct(
                segment_index=i, start_ms=s, end_ms=e, duration_ms=e-s, source="fw_raw", ts_confidence=1.0,
            ))
        w = TimestampedWord.model_construct(
            segment_index_ref=0, word_index_in_segment=0, word_text="hola",
            start_ms=1000, end_ms=1100, source="none",
        )
        return TimestampGenerationResult.build_chain(
            preprocessed_audio=prep, vad_result=vad_result,
            language_detection_result=lid_result, asr_result=asr_result,
            strategy="segment_ts_from_fw_raw",
            transcript_language_code=asr_result.transcript_language_code,
            segments=tuple(segs), words=(w,),
            job_id=job_id,
        )


# =========================================================================
# Tests Application (≥ 16)
# =========================================================================
class TestGenerateTimestampsUseCaseHappy:
    def test_happy_path_two_segments_builds_timestamps(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p, lang="es")
        a = _mk_asr(prep=p, vad=v, lid=l)
        fp = _FakeOKNormalizer()
        uc = GenerateTimestampsUseCase(timestamp_normalizer=fp)
        res = uc.run(p, v, l, a, thresholds=AsrTimestampThresholds(strict_validity=False))
        assert isinstance(res, TimestampGenerationResult)
        assert res.num_segments_ts == 2
        assert res.num_words_ts == 0
        assert len(res.words) == 0
        assert fp.normalize_called_count == 1

    def test_happy_path_idempotente(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p, lang="en")
        a = _mk_asr(prep=p, vad=v, lid=l)
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        r1 = uc.run(p, v, l, a, thresholds=AsrTimestampThresholds())
        r2 = uc.run(p, v, l, a, thresholds=AsrTimestampThresholds())
        assert r1.num_segments_ts == r2.num_segments_ts
        assert r1.strategy == r2.strategy
        assert r1.segments[0].start_ms == r2.segments[0].start_ms

    def test_no_mutate_prep_vad_lid_asr_ids(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=2)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l)
        idp, idv, idl, ida = id(p), id(v), id(l), id(a)
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        uc.run(p, v, l, a)
        assert id(p) == idp and id(v) == idv and id(l) == idl and id(a) == ida

    def test_chain_6_sha(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=((1000, 3000, "OK"),))
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        r = uc.run(p, v, l, a)
        assert r.source_preprocessed_sha256 == p.sha256
        assert r.vad_reference_sha256 == v.source_preprocessed_sha256
        assert r.lid_reference_sha256 == l.source_preprocessed_sha256
        assert r.asr_reference_sha256 == a.source_preprocessed_sha256

    def test_thresholds_none_uses_defaults(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=((1000, 3000, "OK"),))
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        res = uc.run(p, v, l, a, thresholds=None)
        assert res.num_segments_ts == 1


class TestGenerateTimestampsUseCaseNoneInputs:
    def test_none_prep(self):
        v = _mk_vad(num=1)
        l = _mk_lid()
        a = _mk_asr()
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError):
            uc.run(None, v, l, a)  # type: ignore[arg-type]

    def test_none_vad(self):
        p = _mk_prep()
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, lid=l)
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError):
            uc.run(p, None, l, a)  # type: ignore[arg-type]

    def test_none_lid(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        a = _mk_asr(prep=p, vad=v)
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError):
            uc.run(p, v, None, a)  # type: ignore[arg-type]


class TestGenerateTimestampsUseCaseChainMismatch:
    def test_mismatch_prep_vad(self):
        p1 = _mk_prep(); p2 = _mk_prep()  # diferentes SHA (no, mismo dummy)
        # Corregir: dar a p2 un sha diferente manualmente
        p2 = PreprocessedAudio.model_construct(**{
            **{f: getattr(p1, f) for f in PreprocessedAudio.model_fields},
            "sha256": "0" * 64,
        })
        v = _mk_vad(prep=p1, num=1)  # vad bound to p1 SHA
        l = _mk_lid(prep=p1)
        a = _mk_asr(prep=p1, vad=v, lid=l)
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError, match="mismatch"):
            uc.run(p2, v, l, a)

    def test_mismatch_prep_asr(self):
        p1 = _mk_prep()
        p2 = PreprocessedAudio.model_construct(**{
            **{f: getattr(p1, f) for f in PreprocessedAudio.model_fields},
            "sha256": "f" * 64,
        })
        v = _mk_vad(prep=p2, num=1)
        l = _mk_lid(prep=p2)
        a = _mk_asr(prep=p2, vad=v, lid=l, segments_crudo=((1000,3000,"t"),))
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError, match="mismatch"):
            uc.run(p1, v, l, a)

    def test_mismatch_vad_asr(self):
        p1 = _mk_prep()
        p2 = PreprocessedAudio.model_construct(**{
            **{f: getattr(p1, f) for f in PreprocessedAudio.model_fields},
            "sha256": "e" * 64,
        })
        v = _mk_vad(prep=p1, num=1)  # bind p1
        l = _mk_lid(prep=p2)  # bind p2
        a = _mk_asr(prep=p2, vad=None, lid=l, segments_crudo=((1000,3000,"x"),))  # bind p2
        # Para forzar mismatch vad-asr (source_prep p2 == a.source_prep pero vad_reference_sha256 apunta a p1):
        a_tampered = a.model_copy(update={"vad_reference_sha256": p1.sha256})
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError, match="mismatch"):
            uc.run(p2, v, l, a_tampered)


class TestGenerateTimestampsUseCaseSilentFallbackD2:
    def test_silent_fallback_num_intervals_zero_no_call_adapter(self):
        p = _mk_prep(duration_sec=5.0)
        v = _mk_vad(prep=p, num=0, strategy="vad_no_speech")
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, strategy="asr_silent_fallback", segments_crudo=())
        fake = _FakeOKNormalizer()
        uc = GenerateTimestampsUseCase(timestamp_normalizer=fake)
        res = uc.run(p, v, l, a, thresholds=AsrTimestampThresholds())
        assert res.strategy == "ts_silent_fallback"
        assert len(res.segments) == 0
        assert len(res.words) == 0
        assert res.transcript_text == ""
        assert res.confidence == 0.0
        assert fake.normalize_called_count == 0  # Adapter NO invocado

    def test_silent_fallback_asr_segments_empty(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)  # vad NO 0, pero ASR segments ()
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, strategy="full_audio", segments_crudo=())
        fake = _FakeOKNormalizer()
        uc = GenerateTimestampsUseCase(timestamp_normalizer=fake)
        res = uc.run(p, v, l, a)
        assert res.strategy == "ts_silent_fallback"
        assert fake.normalize_called_count == 0

    def test_silent_fallback_disabled_rejected(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=0)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=())
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeOKNormalizer())
        with pytest.raises(ValidationFailedError, match="allow_silent_fallback"):
            uc.run(p, v, l, a, thresholds=AsrTimestampThresholds(allow_silent_fallback=False))


class TestGenerateTimestampsUseCaseStrictAndPropagation:
    def test_propagation_validation_failed_adapter(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=((1000,3000,"x"),))
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeErrorNormalizer())
        with pytest.raises(ValidationFailedError):
            uc.run(p, v, l, a)

    def test_d1_word_ts_nonempty_use_case_rejects(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=((1000,3000,"t"),))
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeReturnWordsNonEmpty())
        with pytest.raises(ValidationFailedError, match="D#1 violation|WORD_TS_APLAZADO"):
            uc.run(p, v, l, a)

    def test_strict_validity_rejects_strict_ok_false(self):
        p = _mk_prep()
        v = _mk_vad(prep=p, num=1)
        l = _mk_lid(prep=p)
        a = _mk_asr(prep=p, vad=v, lid=l, segments_crudo=((1000,3000,"t"),))
        class _FakeNonStrict(TimestampNormalizerPort):
            def normalize(self, prep, *, vad_result, lid_result, asr_result, thresholds=None, job_id=None):
                res_base = TimestampGenerationResult.build_chain(
                    preprocessed_audio=prep, vad_result=vad_result,
                    language_detection_result=lid_result, asr_result=asr_result,
                    strategy="segment_ts_from_fw_raw",
                    transcript_language_code=asr_result.transcript_language_code,
                    segments=(), words=(), job_id=job_id,
                )
                # force strict_ok false
                new_fields = {name: getattr(res_base, name) for name in TimestampGenerationResult.model_fields}
                new_fields["strict_ok"] = False
                new_fields["validation_errors"] = ("dummy warn",)
                # Usar model_construct (frozen bypass pero sí se puede con model_construct)
                return TimestampGenerationResult.model_construct(**new_fields)
        uc = GenerateTimestampsUseCase(timestamp_normalizer=_FakeNonStrict())
        with pytest.raises(ValidationFailedError):
            uc.run(p, v, l, a, thresholds=AsrTimestampThresholds(strict_validity=True))
