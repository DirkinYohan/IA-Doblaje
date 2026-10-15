"""Tests Application UseCase RunVoiceActivityDetectionUseCase T04.

Usa Fake VoiceActivityDetectorPort (no torch, no silero). Verifica:
- happy path
- preprocessed_audio=None
- sample_rate/channels/bit_depth incorrectos
- chain of custody SHA256
- MediaPrepResult no modificado
- determinismo
- logging structured
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from app.core.constants import QualityProfile
from app.core.exceptions import VADProcessingError, ValidationFailedError
from app.core.paths import generate_job_id
from app.domain.entities.media import (
    ExtractedAudio,
    MediaPrepResult,
    PreprocessedAudio,
    ValidatedMedia,
)
from app.domain.entities.vad import MODEL_LABEL_SILERO_V5_1_LITERAL, VadResult
from app.domain.interfaces.vad_ports import VoiceActivityDetectorPort
from app.domain.value_objects.vad import (
    SilenceSegment,
    VadThresholds,
    VoiceInterval,
)
from app.application.use_cases.run_vad import RunVoiceActivityDetectionUseCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHA = "a" * 64
_SHA2 = "b" * 64


class _FakeMediaPrepResultFactory:
    @staticmethod
    def build(
        *,
        wav_path: Path,
        sha256: str,
        sample_rate: int = 16000,
        channels: int = 1,
        bit_depth: int = 16,
        duration_sec: float = 1.0,
        size_bytes: int = 32000,
        preprocessed_exists: bool = True,
    ) -> MediaPrepResult:
        if preprocessed_exists and not wav_path.exists():
            # Stub create synthetic minimal WAV? No — caller lo pasa.
            pass
        validated = ValidatedMedia.model_construct(
            original_path=Path(str(wav_path)),
            safe_name=wav_path.name,
            media_format="wav",  # type: ignore[arg-type]
            extension="wav",
            size_bytes=size_bytes,
            sha256=sha256,
            probed_info={},
            duration_guess_sec=float(duration_sec),
            probed_stream_count=1,
        )
        extracted = ExtractedAudio.model_construct(
            wav_path=wav_path.resolve() if wav_path.exists() else Path(str(wav_path)).resolve(),
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            duration_sec=float(duration_sec),
            size_bytes=size_bytes,
            sha256=sha256,
            ffmpeg_duration_ms=int(duration_sec * 1000),
            probed_duration_ms=int(duration_sec * 1000),
            sanity_duration_pct_diff_ok=True,
        )
        if not preprocessed_exists:
            prep = None
        else:
            prep = PreprocessedAudio.model_construct(
                wav_path=wav_path.resolve(),
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                duration_sec=float(duration_sec),
                size_bytes=size_bytes,
                sha256=sha256,
                applied_filters=("FAKE_FILTERS_T03_FOR_TESTING",),
            )
        return MediaPrepResult.model_construct(
            job_id=generate_job_id(),
            profile=QualityProfile.BALANCED,
            validated=validated,
            extracted_audio=extracted,
            preprocessed_audio=prep,
            step_times_sec={"validate": 0.1, "extract": 0.2, "preprocess": 0.3},
            ffmpeg_available=True,
            ffprobe_available=True,
            errors=[],
        )


# ---------------------------------------------------------------------------
# Fake VAD Port
# ---------------------------------------------------------------------------


class FakeVADPort(VoiceActivityDetectorPort):
    """Retorna VadResult deterministico. NO toca infraestructura."""

    def __init__(
        self,
        *,
        intervals: list[tuple[int, int, float]] | None = None,
        duration_ms: int = 1000,
        source_sha_override: str | None = None,
        raise_type: type[BaseException] | None = None,
    ) -> None:
        # default: 2 intervals separated 400ms gap (merge > 200 no merges)
        self._intervals = list(intervals) if intervals else [(100, 300, 0.9), (700, 900, 0.8)]
        self._duration_ms = int(duration_ms)
        self._source_sha_override = source_sha_override
        self._raise_type = raise_type

    def detect(
        self,
        preprocessed: PreprocessedAudio,
        *,
        thresholds: VadThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> VadResult:
        if self._raise_type is not None:
            raise self._raise_type("port raise test")
        thr = thresholds or VadThresholds()
        total_dur = int(self._duration_ms)
        # Build intervals (VoiceInterval frozen)
        vis: list[VoiceInterval] = []
        for s, e, c in self._intervals:
            dur_ms = int(e) - int(s)
            samples = int(round(dur_ms / 1000.0 * 16000))
            vis.append(
                VoiceInterval(
                    start_ms=int(s),
                    end_ms=int(e),
                    max_confidence=float(c),
                    duration_ms=dur_ms,
                    sample_count=samples,
                )
            )
        vis_sorted = sorted(vis, key=lambda v: v.start_ms)
        silences: list[SilenceSegment] = []
        cursor = 0
        for i, vi in enumerate(vis_sorted):
            if cursor < vi.start_ms:
                silences.append(
                    SilenceSegment(
                        start_ms=cursor,
                        end_ms=vi.start_ms,
                        speech_index_before=-1 if i == 0 else i - 1,
                        speech_index_after=i,
                    )
                )
            cursor = max(cursor, vi.end_ms)
        if cursor < total_dur:
            silences.append(
                SilenceSegment(
                    start_ms=cursor,
                    end_ms=total_dur,
                    speech_index_before=len(vis_sorted) - 1 if vis_sorted else -1,
                    speech_index_after=-1,
                )
            )
        total_speech = sum(v.duration_ms for v in vis_sorted)
        total_silence = sum(s.duration_ms for s in silences)
        ratio = round(total_speech / total_dur, 6) if total_dur > 0 else 0.0
        sha = self._source_sha_override or str(preprocessed.sha256)
        return VadResult(
            source_preprocessed_sha256=sha,
            voice_intervals=tuple(vis_sorted),
            silence_segments=tuple(silences),
            speech_ratio=ratio,
            total_speech_ms=total_speech,
            total_silence_ms=total_silence,
            num_intervals=len(vis_sorted),
            num_silences=len(silences),
            thresholds=thr,
            model_label=MODEL_LABEL_SILERO_V5_1_LITERAL,
            sample_rate=int(preprocessed.sample_rate),
            channels=int(preprocessed.channels),
            duration_ms=total_dur,
        )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def _fake_prep_result(tmp_path: Path) -> MediaPrepResult:
    wav = tmp_path / "input.wav"
    # Crear archivo vacío existente pero tamaño>0.
    wav.parent.mkdir(parents=True, exist_ok=True)
    wav.write_bytes(b"\x00" * 32000)
    return _FakeMediaPrepResultFactory.build(
        wav_path=wav, sha256=_SHA, duration_sec=1.0, size_bytes=32000
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_usecase_happy_path_ok(_fake_prep_result) -> None:
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=1000))
    r = uc.execute(_fake_prep_result)
    assert isinstance(r, VadResult)
    assert r.num_intervals == 2
    assert r.num_silences == 3
    assert r.voice_intervals[0].start_ms == 100
    assert r.voice_intervals[1].end_ms == 900
    # total speech = (300-100)+(900-700) = 400ms → ratio=0.4
    assert r.speech_ratio == pytest.approx(0.4, abs=0.001)
    assert r.total_speech_ms == 400


def test_usecase_preprocessed_none_strict_true_raises(tmp_path) -> None:
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"\x00" * 1000)
    r = _FakeMediaPrepResultFactory.build(
        wav_path=wav, sha256=_SHA, duration_sec=0.05, preprocessed_exists=False
    )
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=50))
    with pytest.raises(VADProcessingError, match="preprocessed_audio es None"):
        uc.execute(r, strict=True)


def test_usecase_preprocessed_none_strict_false_empty_vad(tmp_path) -> None:
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"\x00" * 1000)
    mp = _FakeMediaPrepResultFactory.build(
        wav_path=wav,
        sha256=_SHA,
        duration_sec=0.05,
        preprocessed_exists=False,
    )
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=50))
    r = uc.execute(mp, strict=False)
    assert r.num_intervals == 0
    assert r.total_speech_ms == 0
    # Chain of custody → validated.sha256 used when prep is None
    assert str(r.source_preprocessed_sha256) == _SHA


@pytest.mark.parametrize(
    "sr,ch,bd,match",
    [
        (48000, 1, 16, "sample_rate=48000"),
        (16000, 2, 16, "channels=2"),
        (16000, 1, 24, "bit_depth=24"),
    ],
)
def test_usecase_sanity_t03_format_wrong_raises(
    tmp_path, sr, ch, bd, match
) -> None:
    wav = tmp_path / "bad.wav"
    wav.write_bytes(b"\x00" * 32000)
    prep = _FakeMediaPrepResultFactory.build(
        wav_path=wav,
        sha256=_SHA,
        sample_rate=sr,
        channels=ch,
        bit_depth=bd,
        duration_sec=1.0,
    )
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=1000))
    with pytest.raises(VADProcessingError, match=re.escape(match)):
        uc.execute(prep, strict=True)


def test_usecase_chain_of_custody_sha_ok(_fake_prep_result) -> None:
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=1000))
    r = uc.execute(_fake_prep_result)
    assert str(r.source_preprocessed_sha256) == str(_fake_prep_result.preprocessed_audio.sha256).lower()


def test_usecase_chain_of_custody_sha_mismatch_raises(_fake_prep_result) -> None:
    fake_port = FakeVADPort(duration_ms=1000, source_sha_override=_SHA2)
    uc = RunVoiceActivityDetectionUseCase(fake_port)
    with pytest.raises(ValidationFailedError, match="Chain of Custody rota"):
        uc.execute(_fake_prep_result)


def test_usecase_does_not_mutate_media_prep_result(_fake_prep_result) -> None:
    # Guardamos snapshots
    prep_id_a = id(_fake_prep_result.preprocessed_audio)
    prep_sha_a = str(_fake_prep_result.preprocessed_audio.sha256)
    prep_sr_a = int(_fake_prep_result.preprocessed_audio.sample_rate)
    prep_ch_a = int(_fake_prep_result.preprocessed_audio.channels)
    prep_bd_a = int(_fake_prep_result.preprocessed_audio.bit_depth)
    prep_wp_a = str(_fake_prep_result.preprocessed_audio.wav_path)
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=1000))
    r = uc.execute(_fake_prep_result)
    # Post assertions
    assert id(_fake_prep_result.preprocessed_audio) == prep_id_a
    assert str(_fake_prep_result.preprocessed_audio.sha256) == prep_sha_a
    assert int(_fake_prep_result.preprocessed_audio.sample_rate) == prep_sr_a
    assert int(_fake_prep_result.preprocessed_audio.channels) == prep_ch_a
    assert int(_fake_prep_result.preprocessed_audio.bit_depth) == prep_bd_a
    assert str(_fake_prep_result.preprocessed_audio.wav_path) == prep_wp_a
    assert r is not None


def test_usecase_thresholds_override_propagated(_fake_prep_result) -> None:
    custom = VadThresholds(
        speech_threshold=0.3,
        min_speech_duration_ms=50,
        min_silence_between_ms=100,
        merge_proximal_ms=200,
    )
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=1000))
    r = uc.execute(_fake_prep_result, thresholds=custom)
    assert r.thresholds is not None
    assert float(r.thresholds.speech_threshold) == 0.3
    assert int(r.thresholds.min_speech_duration_ms) == 50


def test_usecase_result_deterministic_two_runs(_fake_prep_result) -> None:
    uc = RunVoiceActivityDetectionUseCase(FakeVADPort(duration_ms=1000))
    a = uc.execute(_fake_prep_result)
    b = uc.execute(_fake_prep_result)
    assert a.num_intervals == b.num_intervals
    assert a.total_speech_ms == b.total_speech_ms
    assert a.speech_ratio == pytest.approx(b.speech_ratio)
    # Mismos intervals
    assert [(v.start_ms, v.end_ms, v.max_confidence) for v in a.voice_intervals] == [
        (v.start_ms, v.end_ms, v.max_confidence) for v in b.voice_intervals
    ]
    # Chain of custody
    assert a.source_preprocessed_sha256 == b.source_preprocessed_sha256
