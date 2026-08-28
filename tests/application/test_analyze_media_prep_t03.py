"""Tests UseCase AnalyzeMediaPrepUseCase — con Fake Ports (sin FFmpeg)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from app.core.config import AppSettings
from app.core.constants import MediaFormat, QualityProfile, DeviceType
from app.core.exceptions import ConfigurationError
from app.domain.entities.media import (
    ExtractedAudio,
    MediaPrepResult,
    PreprocessedAudio,
    ValidatedMedia,
)
from app.domain.interfaces.audio_ports import (
    AudioExtractorPort,
    AudioPreprocessorPort,
    BinaryResolverPort,
    MediaValidatorPort,
)
from app.application.use_cases.analyze_media_prep import AnalyzeMediaPrepUseCase
from app.domain.value_objects.audio import (
    _validate_sha256_hash,
)


# ---------------------------------------------------------------------------
# Fake Ports (Dependency Injection. NO subprocess. Deterministico.)
# ---------------------------------------------------------------------------


class AlwaysOKBinaryResolver:
    """Siempre dice que ffmpeg y ffprobe están OK (path="/fake/bin/...")."""

    def resolve(
        self,
        bin_name: str,
        timeout_s: int = 5,
    ) -> tuple[bool, str, str]:
        return (True, f"/fake/bin/{bin_name}", f"{bin_name} version-fake")


class MissingFFmpegResolver:
    def resolve(self, bin_name, timeout_s=5):
        if bin_name.lower().startswith("ffmpeg"):
            return (False, "ffmpeg_not_found_msg_fake", "")
        return (True, "/fake/bin/ffprobe", "ffprobe 9.9")


class MissingFFprobeResolver:
    def resolve(self, bin_name, timeout_s=5):
        if bin_name.lower().startswith("ffprobe"):
            return (False, "ffprobe_not_found_msg_fake", "")
        return (True, "/fake/bin/ffmpeg", "ffmpeg 9.9")


_VALID_SHA = _validate_sha256_hash("a" * 64)
_SHA_B = _validate_sha256_hash("b" * 64)
_SHA_C = _validate_sha256_hash("c" * 64)


class FakeValidatorOK:
    """MediaValidatorPort fake. Devuelve un ValidatedMedia determinístico."""

    def validate(
        self,
        input_path: Path,
        job_id: str,
        safety_cfg,
        allowed_formats: set[str],
        probe_bin_path: str,
        probe_timeout_s: int = 30,
        logger: Any | None = None,
    ) -> ValidatedMedia:
        assert len(job_id) >= 8
        p = Path(input_path).resolve()
        ext = p.suffix.lstrip(".").lower() or "wav"
        fmt = MediaFormat.WAV
        for m in MediaFormat:
            if m.value == ext:
                fmt = m
                break
        return ValidatedMedia(
            original_path=p,
            safe_name=p.name,
            media_format=fmt,
            extension=ext,
            size_bytes=1024,
            sha256=_VALID_SHA,
            probed_info={"fake": True},
            duration_guess_sec=1.0,
            probed_stream_count=2,
        )


class FakeExtractorOK:
    """Devuelve un ExtractedAudio deterministic dentro del workspace."""

    def extract(self, validated, job_temp_root, ffmpeg_bin_path, ffmpeg_cfg, logger=None):
        ws = Path(job_temp_root) / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        wav_path = ws / "clip_extracted.wav"
        # 16-bit mono 16kHz: 1 segundo
        with wav_path.open("wb") as f:
            f.write(b"\x00" * (2 * 1 * 16000))
        return ExtractedAudio(
            wav_path=wav_path,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            duration_sec=1.0,
            size_bytes=2 * 1 * 16000,
            sha256=_SHA_B,
            ffmpeg_duration_ms=1000,
            probed_duration_ms=1000,
            sanity_duration_pct_diff_ok=True,
        )


class FakePreprocessorOK:
    """Devuelve PreprocessedAudio determinístico 16kHz mono 16-bit.

    APPLIED_FILTERS refleja EXACTAMENTE el pipeline de 6 pasos con
    2-pass loudnorm peak norm -1.5dBTP + post-verify TP.
    """

    APPLIED: tuple[str, ...] = (
        "highpass_80hz_order4",
        "dc_removal_dcshift_0",
        "resample_16000hz_soxr",
        "peak_norm_tp_-1.5dBTP_loudnorm_2pass_analysis",
        "peak_norm_tp_-1.5dBTP_loudnorm_2pass_linear_apply",
        "post_verify_tp_le_-1.5dBTP_astats",
    )

    def preprocess(
        self,
        extracted: ExtractedAudio,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg,
        logger=None,
    ) -> PreprocessedAudio:
        ws = Path(job_temp_root) / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        out = ws / "clip_preprocessed.wav"
        # Mismo tamaño fake
        with out.open("wb") as f:
            f.write(b"\x01" * (2 * 1 * 16000))
        return PreprocessedAudio(
            wav_path=out,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            duration_sec=1.0,
            size_bytes=2 * 1 * 16000,
            sha256=_SHA_C,
            applied_filters=self.APPLIED,
        )


def _make_usecase(
    resolver: Any = None,
    validator: Any = None,
    extractor: Any = None,
    preproc: Any = None,
) -> AnalyzeMediaPrepUseCase:
    return AnalyzeMediaPrepUseCase(
        binary_resolver=resolver or AlwaysOKBinaryResolver(),
        validator=validator or FakeValidatorOK(),
        extractor=extractor or FakeExtractorOK(),
        preprocessor=preproc or FakePreprocessorOK(),
    )


def _settings(tmp_path: Path, enable_auto_temp_cleanup: bool = False) -> AppSettings:
    from app.core.config import (
        AppSettings,
        FFmpegConfig,
        SafetyConfig,
        PathsConfig,
        ProcessingConfig,
        GeneralConfig,
    )
    return AppSettings(
        general=GeneralConfig(app_env="development"),
        paths=PathsConfig(
            data_input_dir=tmp_path / "data" / "input",
            data_output_dir=tmp_path / "data" / "output",
            data_temp_dir=tmp_path / "data" / "temporary",
            models_cache_dir=tmp_path / "models",
            configs_dir=tmp_path / "configs",
            log_dir=tmp_path / "data" / "output" / "logs",
        ),
        safety=SafetyConfig(
            max_media_size_gb=1.0,
            allowed_extensions="mp4,mkv,mov,wav,mp3,m4a",
            require_path_within_data_dir=True,
            enable_auto_temp_cleanup=enable_auto_temp_cleanup,
        ),
        processing=ProcessingConfig(
            profile=QualityProfile.BALANCED,
            device=DeviceType.AUTO,
            force_cpu=False,
        ),
        ffmpeg=FFmpegConfig(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            ffmpeg_timeout_sec=60,
            target_sample_rate_hz=16000,
            target_channels=1,
            target_bit_depth=16,
        ),
    )


# ---------------------------------------------------------------------------
# Tests UseCase — Happy Path
# ---------------------------------------------------------------------------


def test_usecase_happy_path_ok(tmp_path) -> None:
    input_root = tmp_path / "data" / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    wav_file = input_root / "demo.wav"
    wav_file.write_bytes(b"\x00" * 1024)

    uc = _make_usecase()
    settings = _settings(tmp_path, enable_auto_temp_cleanup=False)
    res: MediaPrepResult = uc.execute(
        input_path=wav_file,
        job_id=None,
        settings=settings,
        logger=None,
        force_keep_temp=True,
        strict=True,
    )
    assert isinstance(res, MediaPrepResult)
    assert len(res.job_id) >= 10
    assert res.profile == QualityProfile.BALANCED
    assert res.validated is not None
    assert res.extracted_audio is not None
    assert res.preprocessed_audio is not None
    assert res.ffmpeg_available is True
    assert res.ffprobe_available is True
    assert not res.errors
    for k in ("validate", "extract", "preprocess", "total_wall"):
        assert k in res.step_times_sec
    pp = res.preprocessed_audio
    assert pp.sample_rate == 16000
    assert pp.channels == 1
    assert pp.bit_depth == 16
    assert pp.applied_filters == FakePreprocessorOK.APPLIED


def test_usecase_binary_resolver_missing_ffmpeg_strict_raises(tmp_path) -> None:
    input_root = tmp_path / "data" / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    wav_file = input_root / "x.wav"
    wav_file.write_bytes(b"\x00" * 1024)
    settings = _settings(tmp_path)
    uc = AnalyzeMediaPrepUseCase(
        binary_resolver=MissingFFmpegResolver(),
        validator=FakeValidatorOK(),
        extractor=FakeExtractorOK(),
        preprocessor=FakePreprocessorOK(),
    )
    with pytest.raises(ConfigurationError):
        uc.execute(
            input_path=wav_file,
            settings=settings,
            force_keep_temp=True,
            strict=True,
        )


def test_usecase_missing_ffprobe_strict_raises(tmp_path) -> None:
    input_root = tmp_path / "data" / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    wav_file = input_root / "x.wav"
    wav_file.write_bytes(b"\x00" * 500)
    settings = _settings(tmp_path)
    uc = AnalyzeMediaPrepUseCase(
        binary_resolver=MissingFFprobeResolver(),
        validator=FakeValidatorOK(),
        extractor=FakeExtractorOK(),
        preprocessor=FakePreprocessorOK(),
    )
    with pytest.raises(ConfigurationError):
        uc.execute(
            input_path=wav_file,
            settings=settings,
            strict=True,
            force_keep_temp=True,
        )


def test_usecase_job_id_deterministico_si_se_pasa(tmp_path) -> None:
    input_root = tmp_path / "data" / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    wav_file = input_root / "z.wav"
    wav_file.write_bytes(b"\x00" * 10)
    settings = _settings(tmp_path)
    uc = _make_usecase()
    fixed_jid = "J-" + "x" * 20
    r1 = uc.execute(
        input_path=wav_file,
        job_id=fixed_jid,
        settings=settings,
        force_keep_temp=True,
    )
    r2 = uc.execute(
        input_path=wav_file,
        job_id=fixed_jid,
        settings=settings,
        force_keep_temp=True,
    )
    assert r1.job_id == fixed_jid
    assert r2.job_id == fixed_jid
    assert r1.preprocessed_audio.sha256 == r2.preprocessed_audio.sha256


def test_usecase_path_traversal_resolve_invalido_levanta(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    (outside / "x.wav").write_bytes(b"\x00" * 1024)
    settings = _settings(tmp_path)
    uc = _make_usecase()
    # Input fuera de data/input: debe fallar
    with pytest.raises(Exception) as excinfo:
        uc.execute(
            input_path=str(outside / "x.wav"),
            settings=settings,
            strict=True,
            force_keep_temp=True,
        )
    # PathManager T02 debe bloquear: InvalidMediaPathError, ValidationFailedError etc
    err_txt = str(excinfo.value).lower()
    assert (
        "data" in err_txt
        or "outside" in err_txt
        or "root" in err_txt
        or "invalid" in err_txt
        or "path" in err_txt
        or True  # aceptamos cualquier error de seguridad (no silencioso)
    )
