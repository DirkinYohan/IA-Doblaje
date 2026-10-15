"""Tests unitarios T03 para Value Objects audio (domain). Pydantic V2 Annotated."""
from __future__ import annotations

import hashlib

import pytest
from pydantic import BaseModel, ValidationError

from app.domain.value_objects.audio import (
    _SHA256_RE,
    _validate_audio_duration,
    _validate_bit_depth,
    _validate_channel_count,
    _validate_file_bytes,
    _validate_sample_rate,
    _validate_sha256_hash,
    AudioDuration,
    BitDepth,
    ChannelCount,
    FileBytes,
    SampleRate,
    Sha256Hash,
)


# Test helper model (porque Annotated se valida dentro de un modelo Pydantic)
class _DutModel(BaseModel):
    model_config = {"extra": "forbid"}
    sample_rate: SampleRate = 16000
    channels: ChannelCount = 1
    bit_depth: BitDepth = 16
    duration: AudioDuration = 0.0
    sha: Sha256Hash = "a" * 64
    bytes_sz: FileBytes = 0


# ---------------------------------------------------------------------------
# SampleRate
# ---------------------------------------------------------------------------


def test_sample_rate_ok() -> None:
    assert _validate_sample_rate(16000) == 16000
    m = _DutModel(sample_rate=8000, channels=1, bit_depth=16, duration=0.0, sha="a" * 64, bytes_sz=0)
    assert m.sample_rate == 8000
    m = _DutModel(sample_rate=96000)
    assert m.sample_rate == 96000


def test_sample_rate_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        _validate_sample_rate(100)
    with pytest.raises(ValidationError):
        _DutModel(sample_rate=200000)
    with pytest.raises(ValueError):
        _validate_sample_rate("hello")


# ---------------------------------------------------------------------------
# ChannelCount
# ---------------------------------------------------------------------------


def test_channelcount_mono_stereo() -> None:
    assert _validate_channel_count(1) == 1
    assert _validate_channel_count(2) == 2


def test_channelcount_invalid() -> None:
    with pytest.raises(ValueError):
        _validate_channel_count(0)
    with pytest.raises(ValueError):
        _validate_channel_count(3)
    with pytest.raises(ValidationError):
        _DutModel(channels=3)


# ---------------------------------------------------------------------------
# BitDepth
# ---------------------------------------------------------------------------


def test_bitdepth_allowed() -> None:
    for bd in (8, 16, 24, 32):
        assert _validate_bit_depth(bd) == bd
    m = _DutModel(bit_depth=32)
    assert m.bit_depth == 32


def test_bitdepth_invalid() -> None:
    for bd in (0, 15, 20, 40):
        with pytest.raises(ValueError):
            _validate_bit_depth(bd)
    with pytest.raises(ValidationError):
        _DutModel(bit_depth=15)


# ---------------------------------------------------------------------------
# AudioDuration
# ---------------------------------------------------------------------------


def test_audio_duration_positive() -> None:
    assert _validate_audio_duration(1.23) == pytest.approx(1.23)
    m = _DutModel(duration=5.5)
    assert m.duration == pytest.approx(5.5)


def test_audio_duration_zero() -> None:
    assert _validate_audio_duration(0.0) == 0.0


def test_audio_duration_negative_invalid() -> None:
    with pytest.raises(ValueError):
        _validate_audio_duration(-1)
    with pytest.raises(ValueError):
        _validate_audio_duration("abc")
    with pytest.raises(ValidationError):
        _DutModel(duration=-5.0)


# ---------------------------------------------------------------------------
# Sha256Hash
# ---------------------------------------------------------------------------


def test_sha256_valid() -> None:
    h = hashlib.sha256(b"hello t03").hexdigest()
    assert _validate_sha256_hash(h) == h.lower()
    m = _DutModel(sha=h)
    assert m.sha == h.lower()


def test_sha256_upper_converted_lower() -> None:
    h = hashlib.sha256(b"abc").hexdigest().upper()
    assert _validate_sha256_hash(h) == h.lower()


def test_sha256_invalid() -> None:
    with pytest.raises(ValueError):
        _validate_sha256_hash("")
    with pytest.raises(ValueError):
        _validate_sha256_hash("zz" * 32)
    with pytest.raises(ValidationError):
        _DutModel(sha="short123")


# ---------------------------------------------------------------------------
# FileBytes
# ---------------------------------------------------------------------------


def test_filebytes_ok() -> None:
    assert _validate_file_bytes(123456789) == 123456789
    assert _validate_file_bytes(0) == 0
    m = _DutModel(bytes_sz=5000)
    assert m.bytes_sz == 5000


def test_filebytes_negative_invalid() -> None:
    with pytest.raises(ValueError):
        _validate_file_bytes(-42)
    with pytest.raises(ValidationError):
        _DutModel(bytes_sz=-1)
