"""Tests Infrastructure FFmpeg Adapters T03 — 0 subprocess real (monkeypatch)."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pytest

from app.core.config import FFmpegConfig, SafetyConfig
from app.core.constants import DEFAULT_SAMPLE_RATE_HZ
from app.core.exceptions import (
    AudioExtractionError,
    ConfigurationError,
    MediaTooLargeError,
    UnsupportedMediaError,
    ValidationFailedError,
)
from app.domain.entities.media import (
    ExtractedAudio,
    PreprocessedAudio,
    ValidatedMedia,
)
from app.domain.value_objects.audio import (
    _validate_sha256_hash,
)
from app.infrastructure.audio.ffmpeg_adapters import (
    APPLIED_FILTERS_T03,
    FFmpegAudioExtractor,
    FFmpegAudioPreprocessor,
    FFprobeMediaValidator,
    SubprocessFFmpegBinaryResolver,
    _sha256_file,
    _safe_stderr,
)
from tests.fixtures.wav_synth import create_synthetic_wav_pcm16


# ---------------------------------------------------------------------------
# Helpers test
# ---------------------------------------------------------------------------


def _fake_safety_cfg(max_gb: float = 10.0, allowed: str = "mp4,mkv,mov,wav,mp3,m4a") -> SafetyConfig:
    return SafetyConfig(max_media_size_gb=max_gb, allowed_extensions=allowed)


def _fake_ffmpeg_cfg(
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    timeout: int = 60,
    target_sr: int = 16000,
    target_ch: int = 1,
    target_bd: int = 16,
) -> FFmpegConfig:
    return FFmpegConfig(
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        ffmpeg_timeout_sec=timeout,
        target_sample_rate_hz=target_sr,
        target_channels=target_ch,
        target_bit_depth=target_bd,
    )


@dataclass
class FakeCompletedProcess:
    """Mock subprocess.CompletedProcess.

    Simula `subprocess.run(..., text=True)`: stdout/stderr se pueden inicializar
    como bytes o str y la clase expone siempre ambos atributos en str si
    `text=True` fue pasado a la llamada (kw detectado). Para simplificar:
    aceptamos bytes en constructor y exponemos como str codificado utf8 cuando
    el caller lo quiera via `text=True`.
    """

    returncode: int = 0
    stdout: bytes | str = b""
    stderr: bytes | str = b""
    args: list[str] | None = None
    timeout: int | None = None

    def __post_init__(self):
        # Internal helpers
        pass

    @property
    def out_str(self) -> str:
        if isinstance(self.stdout, bytes):
            try:
                return self.stdout.decode("utf-8", errors="replace")
            except Exception:
                return ""
        return str(self.stdout or "")

    @property
    def err_str(self) -> str:
        if isinstance(self.stderr, bytes):
            try:
                return self.stderr.decode("utf-8", errors="replace")
            except Exception:
                return ""
        return str(self.stderr or "")

    @property
    def out_bytes(self) -> bytes:
        if isinstance(self.stdout, str):
            return self.stdout.encode("utf-8")
        return bytes(self.stdout or b"")

    @property
    def err_bytes(self) -> bytes:
        if isinstance(self.stderr, str):
            return self.stderr.encode("utf-8")
        return bytes(self.stderr or b"")


# ---------------------------------------------------------------------------
# Helpers: fake json ffprobe y make subprocess.run intercept
# ---------------------------------------------------------------------------


def _make_ffprobe_json(
    *,
    duration: float = 1.5,
    audio_count: int = 1,
    audio_sample_rate: int = 44100,
    audio_channels: int = 2,
    audio_bits: int = 16,
    audio_duration: float | None = None,
) -> bytes:
    streams: list[dict[str, Any]] = []
    for i in range(audio_count):
        streams.append({
            "index": i,
            "codec_type": "audio",
            "codec_name": "pcm_s16le",
            "sample_rate": str(audio_sample_rate),
            "channels": audio_channels,
            "bits_per_sample": audio_bits,
            "bits_per_raw_sample": audio_bits,
            "duration": str(audio_duration if audio_duration is not None else duration),
        })
    data = {
        "streams": streams,
        "format": {
            "filename": "fake_input.wav",
            "nb_streams": len(streams),
            "format_name": "wav",
            "duration": str(duration),
            "size": "1024",
            "bit_rate": "128000",
        },
    }
    return json.dumps(data).encode("utf-8")


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello t03 sha256 check")
    assert len(_sha256_file(p)) == 64
    import hashlib
    assert _sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest().lower()


def test_sha256_deterministic(tmp_path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"repeated content same sha")
    assert _sha256_file(p) == _sha256_file(p)


def test_safe_stderr_truncate_200_chars() -> None:
    long = b"a" * 1000
    assert len(_safe_stderr(long, 200)) == 203  # "... suffix 3 chars"
    short = b"abc"
    assert _safe_stderr(short) == "abc"


# ---------------------------------------------------------------------------
# SubprocessFFmpegBinaryResolver
# ---------------------------------------------------------------------------


def test_resolver_ok_when_which_found_and_rc0(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_which(bin_name):
        return f"/usr/bin/{bin_name}"

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        return FakeCompletedProcess(
            returncode=0,
            stdout=(f"{cmd[0]} version 9.9 fake\nrest line".encode()),
            stderr=b"",
            args=cmd,
        )

    monkeypatch.setattr("app.infrastructure.audio.ffmpeg_adapters.shutil.which", fake_which)
    monkeypatch.setattr("app.infrastructure.audio.ffmpeg_adapters.subprocess.run", fake_run)

    resolver = SubprocessFFmpegBinaryResolver()
    ok, path, ver = resolver.resolve("ffmpeg")
    assert ok is True
    assert path == "/usr/bin/ffmpeg"
    assert "9.9" in ver
    assert calls == [["/usr/bin/ffmpeg", "-version"]]


def test_resolver_not_found_which_none(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.shutil.which",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters._locate_on_windows",
        lambda _name: None,
    )
    ok, msg, ver = SubprocessFFmpegBinaryResolver().resolve("ffmpeg")
    assert ok is False
    assert "ffmpeg" in msg
    assert ver == ""
    assert "PATH" in msg.upper() or "path" in msg


def test_resolver_returncode_1_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.shutil.which",
        lambda b: f"/bin/{b}",
    )
    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        lambda *a, **kw: FakeCompletedProcess(returncode=1, stderr=b"err fake"),
    )
    ok, msg, ver = SubprocessFFmpegBinaryResolver().resolve("ffprobe", timeout_s=2)
    assert ok is False
    assert "returncode=1" in msg


def test_resolver_timeout_handled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.shutil.which",
        lambda b: f"/bin/{b}",
    )

    def raises_timeout(*a, **kw):
        raise subprocess.TimeoutExpired(cmd=a[0], timeout=5)

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        raises_timeout,
    )
    ok, msg, ver = SubprocessFFmpegBinaryResolver().resolve("ffmpeg")
    assert ok is False
    assert "timed out" in msg.lower() or "timeout" in msg.lower()


# ---------------------------------------------------------------------------
# FFprobeMediaValidator — tests con subprocess monkeypatch
# ---------------------------------------------------------------------------


def test_validator_wav_simple_ok(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(
        tmp_path / "clip.wav", duration_sec=0.2, sample_rate_hz=16000
    )

    def fake_run(cmd: list[str], **kw):
        # Probe cmd -> return fake valid audio stream JSON
        return FakeCompletedProcess(
            returncode=0,
            stdout=_make_ffprobe_json(duration=0.2, audio_count=1, audio_sample_rate=16000, audio_channels=1, audio_bits=16),
            stderr=b"",
        )

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        fake_run,
    )

    v = FFprobeMediaValidator()
    result: ValidatedMedia = v.validate(
        wav,
        job_id="TST" + "0" * 16,
        safety_cfg=_fake_safety_cfg(10.0),
        allowed_formats={"wav"},
        probe_bin_path="/fake/ffprobe",
        probe_timeout_s=5,
    )
    assert isinstance(result, ValidatedMedia)
    assert result.extension == "wav"
    assert result.probed_stream_count >= 1
    assert result.duration_guess_sec is not None
    assert float(result.duration_guess_sec) == pytest.approx(0.2)
    assert len(result.sha256) == 64


def test_validator_empty_file_raises(tmp_path, monkeypatch) -> None:
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValidationFailedError, match="EmptyMediaFile"):
        FFprobeMediaValidator().validate(
            empty,
            job_id="J" * 20,
            safety_cfg=_fake_safety_cfg(10.0),
            allowed_formats={"wav"},
            probe_bin_path="/fake/ffprobe",
        )


def test_validator_too_large_raises(tmp_path, monkeypatch) -> None:
    big = tmp_path / "big.wav"
    big.write_bytes(b"\x00" * 10)
    big_abs = big.resolve()

    class FakeStat:
        st_size = int(1.1 * 1024 * 1024 * 1024)
        st_mode = 0o100644

    orig = Path.stat

    def fake_stat(p, *args, **kwargs):
        try:
            resolved = orig(p)
            _ = resolved
        except Exception:
            pass
        if p == big_abs or (hasattr(p, "resolve") and Path(str(p)).resolve() == big_abs):
            return FakeStat()
        return orig(p, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(MediaTooLargeError):
        FFprobeMediaValidator().validate(
            big,
            job_id="J" * 20,
            safety_cfg=_fake_safety_cfg(1.0),
            allowed_formats={"wav"},
            probe_bin_path="/fake/ffprobe",
        )


def test_validator_unsupported_extension_raises(tmp_path) -> None:
    bad = tmp_path / "program.exe"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(UnsupportedMediaError):
        FFprobeMediaValidator().validate(
            bad,
            job_id="J" * 20,
            safety_cfg=_fake_safety_cfg(1.0),
            allowed_formats={"wav", "mp4"},
            probe_bin_path="/fake/ffprobe",
        )


def test_validator_ffprobe_exit1_raises(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "clip.wav", duration_sec=0.1)

    def fail(*a, **kw):
        return FakeCompletedProcess(returncode=1, stderr=b"not-a-valid-media err")

    monkeypatch.setattr("app.infrastructure.audio.ffmpeg_adapters.subprocess.run", fail)
    with pytest.raises(ValidationFailedError):
        FFprobeMediaValidator().validate(
            wav,
            job_id="J" * 20,
            safety_cfg=_fake_safety_cfg(1.0),
            allowed_formats={"wav"},
            probe_bin_path="/fake/ffprobe",
        )


def test_validator_no_audio_stream_raises(tmp_path, monkeypatch) -> None:
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "silent.wav", duration_sec=0.1)

    def probe_no_audio(cmd, **kw):
        # 0 streams audio
        return FakeCompletedProcess(
            returncode=0,
            stdout=_make_ffprobe_json(duration=0.1, audio_count=0),
            stderr=b"",
        )

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        probe_no_audio,
    )
    with pytest.raises(ValidationFailedError, match="No_audio_stream_found"):
        FFprobeMediaValidator().validate(
            wav,
            job_id="J" * 20,
            safety_cfg=_fake_safety_cfg(1.0),
            allowed_formats={"wav"},
            probe_bin_path="/fake/ffprobe",
        )


# ---------------------------------------------------------------------------
# FFmpegAudioExtractor
# ---------------------------------------------------------------------------


def _build_validated(wav_path: Path, duration: float = 0.2) -> ValidatedMedia:
    return ValidatedMedia(
        original_path=wav_path.resolve(),
        safe_name=wav_path.name,
        media_format=__import__("app.core.constants", fromlist=["MediaFormat"]).MediaFormat.WAV,
        extension="wav",
        size_bytes=int(wav_path.stat().st_size),
        sha256=_validate_sha256_hash("a" * 64),
        probed_info={"fake": True},
        duration_guess_sec=float(duration),
        probed_stream_count=1,
    )


def test_extractor_ok(tmp_path, monkeypatch) -> None:
    job_root = tmp_path / "temp" / "JOB0001"
    job_root.mkdir(parents=True, exist_ok=True)

    input_wav, *_ = create_synthetic_wav_pcm16(
        tmp_path / "in.wav", duration_sec=0.2, sample_rate_hz=16000
    )

    # Simulate: FFmpeg extract escribe un WAV output válido (el mismo helper)
    out_expected = job_root / "workspace" / "in_extracted.wav"

    cmd_seq: list[list[str]] = []

    def mock_subprocess_run(cmd: list[str], **kwargs):
        cmd_seq.append(list(cmd))
        first = Path(cmd[0]).name.lower() if len(cmd) > 0 else ""
        if first.startswith("ffmpeg") and "-i" in cmd:
            # Crear el WAV esperado para simular extracción OK
            create_synthetic_wav_pcm16(
                out_expected, duration_sec=0.2, sample_rate_hz=16000
            )
            return FakeCompletedProcess(returncode=0, stdout=b"", stderr=b"")
        if first.startswith("ffprobe"):
            return FakeCompletedProcess(
                returncode=0,
                stdout=_make_ffprobe_json(
                    duration=0.2, audio_count=1,
                    audio_sample_rate=16000, audio_channels=1, audio_bits=16,
                ),
                stderr=b"",
            )
        return FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        mock_subprocess_run,
    )

    extractor = FFmpegAudioExtractor()
    res = extractor.extract(
        validated=_build_validated(input_wav, 0.2),
        job_temp_root=job_root,
        ffmpeg_bin_path="/fake/ffmpeg",
        ffmpeg_cfg=_fake_ffmpeg_cfg(
            ffprobe_bin="/fake/ffprobe",
            target_sr=16000,
            target_ch=1,
            timeout=60,
        ),
    )
    assert isinstance(res, ExtractedAudio)
    assert res.wav_path == out_expected.resolve()
    assert int(res.sample_rate) == 16000
    assert int(res.channels) == 1
    assert int(res.bit_depth) == 16
    assert res.sanity_duration_pct_diff_ok is True


def test_extractor_rc1_raises(tmp_path, monkeypatch) -> None:
    job_root = tmp_path / "j"
    job_root.mkdir(parents=True, exist_ok=True)
    input_wav, *_ = create_synthetic_wav_pcm16(tmp_path / "in.wav", duration_sec=0.1)

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        lambda cmd, **kw: FakeCompletedProcess(returncode=1, stderr=b"ffmpeg error corrupted"),
    )
    with pytest.raises(AudioExtractionError):
        FFmpegAudioExtractor().extract(
            validated=_build_validated(input_wav, 0.1),
            job_temp_root=job_root,
            ffmpeg_bin_path="/fake/ffmpeg",
            ffmpeg_cfg=_fake_ffmpeg_cfg(ffprobe_bin="/fake/ffprobe"),
        )


def test_extractor_timeout(tmp_path, monkeypatch) -> None:
    job_root = tmp_path / "j"
    job_root.mkdir(parents=True, exist_ok=True)
    wav, *_ = create_synthetic_wav_pcm16(tmp_path / "in.wav", duration_sec=0.1)

    def fake_timeout(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, timeout=3600)

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        fake_timeout,
    )
    with pytest.raises(AudioExtractionError, match="timeout"):
        FFmpegAudioExtractor().extract(
            _build_validated(wav, 0.1),
            job_temp_root=job_root,
            ffmpeg_bin_path="/fake/ffmpeg",
            ffmpeg_cfg=_fake_ffmpeg_cfg(ffprobe_bin="/fake/ffprobe"),
        )


# ---------------------------------------------------------------------------
# FFmpegAudioPreprocessor
# ---------------------------------------------------------------------------


def _build_extracted(wav_path: Path, duration: float) -> ExtractedAudio:
    return ExtractedAudio(
        wav_path=wav_path.resolve(),
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        duration_sec=float(duration),
        size_bytes=int(wav_path.stat().st_size),
        sha256=_validate_sha256_hash("b" * 64),
        ffmpeg_duration_ms=int(duration * 1000),
        probed_duration_ms=int(duration * 1000),
        sanity_duration_pct_diff_ok=True,
    )


def _make_loudnorm_json(
    *,
    input_i: float = -23.0,
    input_tp: float = -6.0,
    input_lra: float = 7.0,
    target_offset: float = 0.15,
    output_i: float = -16.0,
    output_tp: float = -1.5,
    output_lra: float = 10.5,
) -> bytes:
    d = {
        "input_i": f"{input_i:.4f}",
        "input_tp": f"{input_tp:.4f}",
        "input_lra": f"{input_lra:.4f}",
        "input_thresh": "-35.1234",
        "target_offset": f"{target_offset:.4f}",
        "output_i": f"{output_i:.4f}",
        "output_tp": f"{output_tp:.4f}",
        "output_lra": f"{output_lra:.4f}",
        "output_thresh": "-36.1234",
        "normalization_type": "dynamic",
        "target_offset": f"{target_offset:.4f}",
    }
    return json.dumps(d, separators=(",", ":")).encode("utf-8")


def test_preprocessor_ok_applied_filters_deterministic(tmp_path, monkeypatch) -> None:
    job_root = tmp_path / "JOB0002" / "workspace"
    job_root.mkdir(parents=True, exist_ok=True)
    extracted_wav, *_ = create_synthetic_wav_pcm16(
        job_root / "in_extracted.wav", duration_sec=0.15
    )
    # Mismo stem que usa FFmpegAudioPreprocessor internamente: extracted.wav_path.stem
    stem = extracted_wav.stem
    wav_pre = job_root / f"{stem}_A_pre_dsp.wav"
    expected_pp = job_root / f"{stem}_B_final_preprocessed.wav"

    _calls: list[list[str]] = []

    def mock_subprocess_run(cmd, **kw):
        _calls.append(list(cmd))
        first = Path(cmd[0]).name.lower()
        cmd_s = " ".join(cmd)
        # (1) SubA: highpass+dc+resample → A_pre_dsp.wav. Tiene -vn -acodec pcm_s16le -af sin aformat ni linear.
        if (
            first.startswith("ffmpeg")
            and "-vn" in cmd
            and "-acodec" in cmd
            and "highpass" in cmd_s
            and "aresample=16000" in cmd_s
            and "aformat" not in cmd_s
            and "linear=true" not in cmd_s
        ):
            create_synthetic_wav_pcm16(wav_pre, duration_sec=0.15, sample_rate_hz=16000, channels=1)
            return FakeCompletedProcess(0, b"", b"", list(cmd))
        # (2) Analysis: loudnorm print_format=json a null (primer pass, sin aformat, sin linear)
        if (
            first.startswith("ffmpeg")
            and "-f" in cmd
            and "print_format=json" in cmd_s
            and "aformat" not in cmd_s
            and "linear=true" not in cmd_s
        ):
            return FakeCompletedProcess(0, b"", _make_loudnorm_json(input_i=-22.5, input_tp=-5.9, input_lra=6.8, target_offset=0.2, output_tp=-1.52), list(cmd))
        # (3) Apply: loudnorm linear=true + aformat sample_fmts=s16 → final
        if (
            first.startswith("ffmpeg")
            and "linear=true" in cmd_s
            and "aformat=sample_fmts=s16" in cmd_s
        ):
            create_synthetic_wav_pcm16(
                expected_pp, duration_sec=0.15, sample_rate_hz=16000, channels=1
            )
            return FakeCompletedProcess(0, b"", b"", list(cmd))
        # (4) Post-verify TP measurement (segundo pass loudnorm print_format=json a null)
        if (
            first.startswith("ffmpeg")
            and "-f" in cmd
            and "print_format=json" in cmd_s
            and "aformat" not in cmd_s
            and "measured_I" not in cmd_s
        ):
            return FakeCompletedProcess(0, b"", _make_loudnorm_json(output_tp=-1.55), list(cmd))
        # (5) ffprobe final / intermedio
        if first.startswith("ffprobe"):
            return FakeCompletedProcess(
                0,
                _make_ffprobe_json(
                    duration=0.15, audio_count=1,
                    audio_sample_rate=16000, audio_channels=1, audio_bits=16,
                ),
                b"", list(cmd),
            )
        return FakeCompletedProcess(0, b"", b"", list(cmd))

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        mock_subprocess_run,
    )
    p = FFmpegAudioPreprocessor()
    res = p.preprocess(
        extracted=_build_extracted(extracted_wav, 0.15),
        job_temp_root=job_root.parent,
        ffmpeg_bin_path="/fake/ffmpeg",
        ffmpeg_cfg=_fake_ffmpeg_cfg(ffprobe_bin="/fake/ffprobe"),
    )
    assert isinstance(res, PreprocessedAudio)
    assert res.wav_path == expected_pp.resolve()
    assert int(res.sample_rate) == 16000
    assert int(res.channels) == 1
    assert int(res.bit_depth) == 16
    # Applied filters debe incluir 2-pass explicitamente y no la etiqueta falsa "peak_norm_loudnorm" genérica.
    assert res.applied_filters == APPLIED_FILTERS_T03
    assert len(res.applied_filters) == 6
    assert any("2pass_analysis" in s for s in res.applied_filters)
    assert any("2pass_linear_apply" in s for s in res.applied_filters)
    assert any("post_verify_tp_le_-1.5dBTP" in s for s in res.applied_filters)


def test_preprocessor_fail_if_wrong_sr_after(tmp_path, monkeypatch) -> None:
    job_root = tmp_path / "JOB0003" / "workspace"
    job_root.mkdir(parents=True, exist_ok=True)
    extracted_wav, *_ = create_synthetic_wav_pcm16(job_root / "in.wav", duration_sec=0.1)
    stem = extracted_wav.stem
    wav_pre = job_root / f"{stem}_A_pre_dsp.wav"
    bad_pp = job_root / f"{stem}_B_final_preprocessed.wav"

    def mock_run(cmd, **kw):
        first = Path(cmd[0]).name.lower()
        cmd_s = " ".join(cmd)
        # SubA
        if (
            first.startswith("ffmpeg")
            and "-vn" in cmd
            and "highpass" in cmd_s
            and "aresample=16000" in cmd_s
            and "aformat" not in cmd_s
            and "linear=true" not in cmd_s
        ):
            create_synthetic_wav_pcm16(wav_pre, duration_sec=0.1, sample_rate_hz=8000)
            return FakeCompletedProcess(0, b"", b"", cmd)
        # Analysis loudnorm json
        if (
            first.startswith("ffmpeg")
            and "-f" in cmd
            and "print_format=json" in cmd_s
            and "aformat" not in cmd_s
            and "linear=true" not in cmd_s
        ):
            return FakeCompletedProcess(0, b"", _make_loudnorm_json(input_tp=-5.0, output_tp=-1.5), cmd)
        # Apply linear + aformat
        if first.startswith("ffmpeg") and "linear=true" in cmd_s and "aformat=sample_fmts=s16" in cmd_s:
            if not bad_pp.parent.exists():
                bad_pp.parent.mkdir(parents=True, exist_ok=True)
            create_synthetic_wav_pcm16(bad_pp, duration_sec=0.1, sample_rate_hz=8000)
            return FakeCompletedProcess(0, b"", b"", cmd)
        # Post-verify TP measurement
        if (
            first.startswith("ffmpeg")
            and "-f" in cmd
            and "print_format=json" in cmd_s
            and "measured_I" not in cmd_s
        ):
            return FakeCompletedProcess(0, b"", _make_loudnorm_json(output_tp=-1.6), cmd)
        # ffprobe final con sample_rate=8000 (debe fallar)
        if first.startswith("ffprobe"):
            return FakeCompletedProcess(
                0,
                _make_ffprobe_json(
                    duration=0.1, audio_count=1,
                    audio_sample_rate=8000, audio_channels=1, audio_bits=16,
                ),
                b"", cmd,
            )
        return FakeCompletedProcess(0, b"", b"", cmd)

    monkeypatch.setattr(
        "app.infrastructure.audio.ffmpeg_adapters.subprocess.run",
        mock_run,
    )
    with pytest.raises(AudioExtractionError, match="sample_rate"):
        FFmpegAudioPreprocessor().preprocess(
            _build_extracted(extracted_wav, 0.1),
            job_temp_root=job_root.parent,
            ffmpeg_bin_path="/fake/ffmpeg",
            ffmpeg_cfg=_fake_ffmpeg_cfg(ffprobe_bin="/fake/ffprobe"),
        )
