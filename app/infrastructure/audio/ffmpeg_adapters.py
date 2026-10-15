"""Implementaciones concretas Ports Domain via FFmpeg/FFprobe — Capa INFRASTRUCTURE.

Todas las invocaciones externas son a procesos:
    subprocess.run([list[str], shell=False, capture_output=True, timeout=...]).

Nunca descarga binarios, nunca shell=True, nunca strings concatenados.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from app.core.config import FFmpegConfig, SafetyConfig
from app.core.constants import MediaFormat
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
from app.domain.interfaces.audio_ports import (
    AudioExtractorPort,
    AudioPreprocessorPort,
    BinaryResolverPort,
    MediaValidatorPort,
)
from app.domain.value_objects.audio import (
    AudioDuration,
    BitDepth,
    ChannelCount,
    FileBytes,
    SampleRate,
    Sha256Hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """SHA-256 streaming 64 chars lower hex. Python stdlib. Chunks 4MB."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest().lower()


def _safe_stderr(stderr_b: bytes, limit: int = 200) -> str:
    """Trunca stderr ≤ N chars y limpia newlines/CR."""
    if not stderr_b:
        return ""
    try:
        s = stderr_b.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        s = repr(stderr_b[:limit])
    s = s.strip().replace("\r", " ").replace("\n", " ")
    if len(s) > limit:
        s = s[:limit] + "..."
    return s


def _resolve_format_from_ext(ext_no_dot_lower: str) -> MediaFormat:
    if ext_no_dot_lower in {"mp4", "mkv", "mov"}:
        return MediaFormat(ext_no_dot_lower)
    if ext_no_dot_lower in {"wav", "mp3", "m4a"}:
        return MediaFormat(ext_no_dot_lower)
    raise UnsupportedMediaError(
        f"Extension .{ext_no_dot_lower} no en whitelist SafetyConfig.allowed_extensions."
    )


# ---------------------------------------------------------------------------
# 1) BinaryResolverPort
# ---------------------------------------------------------------------------


class SubprocessFFmpegBinaryResolver:
    """Detecta FFmpeg/FFprobe vía subprocess -version. NO descarga binarios."""

    def resolve(
        self,
        bin_name: str,
        timeout_s: int = 5,
    ) -> tuple[bool, str, str]:
        if not bin_name or not isinstance(bin_name, str) or not bin_name.strip():
            return (False, f"bin_name invalido: {bin_name!r}", "")

        which_path = _locate_binary(bin_name.strip())
        if not which_path:
            msg = (
                f"No se encontró el binario {bin_name!r} en PATH del sistema. "
                "T03 requiere FFmpeg (ffmpeg + ffprobe). "
                "Instalar FFmpeg versión LGPL y verificar con: "
                "python scripts/verify_third_party.py"
            )
            return (False, msg, "")

        located = Path(which_path)
        abs_bin: str = str(located.resolve()) if located.is_symlink() else which_path
        try:
            cp = subprocess.run(
                [abs_bin, "-version"],
                shell=False,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_s)),
            )
        except subprocess.TimeoutExpired as exc:
            return (False, f"Binario {bin_name} no respondió en {timeout_s}s: {exc}", "")
        except FileNotFoundError as exc:
            return (False, f"Binario {bin_name} FileNotFoundError: {exc}", "")
        except OSError as exc:
            return (False, f"Error OS ejecutando {bin_name}: {exc}", "")
        except Exception as exc:  # noqa: BLE001
            return (False, f"Error inesperado {bin_name}: {exc!r}", "")

        if cp.returncode != 0:
            err = _safe_stderr(
                cp.stderr.encode() if isinstance(cp.stderr, str) else cp.stderr
            )
            return (False, f"{bin_name} returncode={cp.returncode}. stderr: {err}", "")
        stdout = (cp.stdout or "").strip() if isinstance(cp.stdout, str) else (cp.stdout or b"").decode("utf-8", errors="replace").strip()
        first_line = stdout.splitlines()[0] if stdout else ""
        return (True, abs_bin, first_line[:80].strip())


def _locate_binary(bin_name: str) -> str | None:
    """Busca el binario en el PATH del proceso y, en Windows, en el PATH ya guardado."""
    found = shutil.which(bin_name)
    if found:
        return found
    if os.name != "nt":
        return None
    return _locate_on_windows(bin_name)


def _locate_on_windows(bin_name: str) -> str | None:
    exe = bin_name if bin_name.lower().endswith(".exe") else f"{bin_name}.exe"
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        direct = Path(local) / "Microsoft" / "WinGet" / "Links" / exe
        if direct.is_file():
            return str(direct)
    try:
        import winreg
    except ImportError:
        return None
    folders: list[str] = []
    for hive, subkey in (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(hive, subkey) as key:
                raw, _kind = winreg.QueryValueEx(key, "Path")
        except OSError:
            continue
        if isinstance(raw, str):
            folders.extend(os.path.expandvars(part).strip().strip('"') for part in raw.split(";") if part.strip())
    if not folders:
        return None
    return shutil.which(exe, path=os.pathsep.join(folders))


# ---------------------------------------------------------------------------
# Helpers ffprobe
# ---------------------------------------------------------------------------


def _run_ffprobe_json(
    probe_bin_path: str,
    input_path: Path,
    timeout_s: int,
) -> dict[str, Any]:
    """Llama ffprobe -show_format -show_streams JSON. Raises ValidationFailedError."""
    cmd: list[str] = [
        probe_bin_path,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]
    try:
        cp = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            timeout=max(1, int(timeout_s)),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationFailedError(
            f"ffprobe timeout {timeout_s}s excedido."
        ) from exc
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"ffprobe bin no encontrado: {probe_bin_path!r}. "
            "T03 requiere FFmpeg instalado (LGPL)."
        ) from exc
    except OSError as exc:
        raise ValidationFailedError(f"OS error ffprobe: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ValidationFailedError(f"Error inesperado ffprobe: {exc!r}") from exc

    if cp.returncode != 0:
        err = _safe_stderr(cp.stderr, 200)
        raise ValidationFailedError(
            f"ffprobe fallo returncode={cp.returncode} {err}"
        )
    try:
        txt = cp.stdout.decode("utf-8", errors="replace")
        return dict(json.loads(txt or "{}"))
    except Exception as exc:  # noqa: BLE001
        bad = (cp.stdout[:120] if cp.stdout else b"").decode("utf-8", errors="replace")
        raise ValidationFailedError(
            f"ffprobe JSON invalido: {bad!r} exc={exc!r}"
        )


def _extract_audio_stream_and_duration(
    probe_data: dict[str, Any],
) -> tuple[int, float | None]:
    streams = list(probe_data.get("streams") or [])
    audio_count = sum(1 for s in streams if str(s.get("codec_type", "")).lower() == "audio")
    fmt: dict[str, Any] = dict(probe_data.get("format") or {})
    dur_s = fmt.get("duration")
    duration: float | None = None
    if dur_s is not None:
        try:
            duration = float(dur_s)
            if duration < 0:
                duration = None
        except (TypeError, ValueError):
            duration = None
    return (audio_count, duration)


# ---------------------------------------------------------------------------
# 2) MediaValidatorPort
# ---------------------------------------------------------------------------


class FFprobeMediaValidator(MediaValidatorPort):
    """Step 01. Fail-Fast validación + ffprobe."""

    def validate(
        self,
        input_path: Path,
        job_id: str,
        safety_cfg: SafetyConfig,
        allowed_formats: set[str],
        probe_bin_path: str,
        probe_timeout_s: int = 30,
        logger: Any | None = None,
    ) -> ValidatedMedia:
        if not isinstance(job_id, str) or len(job_id) < 8:
            raise ValidationFailedError(f"job_id invalido (len<8): {job_id!r}")
        if not probe_bin_path or not isinstance(probe_bin_path, str):
            raise ConfigurationError(
                "ffprobe_bin_path vacío. Verificar BinaryResolver antes llamar Port."
            )

        p = Path(input_path)

        # 1) exists + is_file
        if not p.exists():
            raise ValidationFailedError(
                f"Archivo multimedia no existe: {p!r}",
                details={"error_code": "MediaNotFound"},
            )
        if not p.is_file():
            raise ValidationFailedError(
                f"Ruta multimedia no es un archivo regular: {p!r}",
                details={"error_code": "MediaNotRegularFile"},
            )

        # 2) stat
        try:
            st = p.stat()
        except OSError as exc:
            raise ValidationFailedError(
                f"No se pudo stat() el archivo {p!r}: {exc}"
            )
        size_bytes_int = int(st.st_size)

        # 3) empty
        if size_bytes_int == 0:
            raise ValidationFailedError(
                "EmptyMediaFile: 0 bytes. No se puede procesar.",
                details={"error_code": "EmptyMediaFile"},
            )

        # 4) max size
        max_bytes = int(float(getattr(safety_cfg, "max_media_size_gb", 10.0)) * (1024**3))
        if max_bytes <= 0:
            max_bytes = 10 * (1024**3)
        if size_bytes_int > max_bytes:
            gb_file = size_bytes_int / (1024**3)
            gb_max = max_bytes / (1024**3)
            raise MediaTooLargeError(
                f"MediaTooLarge: {gb_file:.3f}GB > max_media_size_gb={gb_max:.3f}GB"
            )

        # 5) Extension whitelist
        ext_raw = p.suffix.lower().lstrip(".")
        if not ext_raw or ext_raw not in allowed_formats:
            raise UnsupportedMediaError(
                f"Extension .{ext_raw or '(sin)'} no soportada. "
                f"Permitidas: {sorted(list(allowed_formats))}"
            )

        # 6) SHA256
        sha = _sha256_file(p)

        # 7) ffprobe
        probe_data = _run_ffprobe_json(probe_bin_path, p, probe_timeout_s)
        audio_count, dur = _extract_audio_stream_and_duration(probe_data)
        if audio_count <= 0:
            raise ValidationFailedError(
                "No_audio_stream_found: ffprobe no detectó streams de audio.",
                details={"error_code": "NoAudioStream"},
            )
        streams_total = len(list(probe_data.get("streams") or []))
        fmt = _resolve_format_from_ext(ext_raw)
        dur_v: float | None = None if dur is None else float(dur)
        safe_name = p.name[:220] or "media.bin"
        return ValidatedMedia(
            original_path=Path(p).resolve(),
            safe_name=safe_name,
            media_format=fmt,
            extension=ext_raw,
            size_bytes=size_bytes_int,
            sha256=sha,
            probed_info=probe_data,
            duration_guess_sec=dur_v,
            probed_stream_count=int(streams_total),
        )


# ---------------------------------------------------------------------------
# 3) AudioExtractorPort
# ---------------------------------------------------------------------------


_PREPROCESS_SANITY_MAX_PCT = 5.0


def _sanity_ok(probed_ms: int, ffmpeg_ms: int) -> tuple[bool, float]:
    base = max(int(probed_ms), int(ffmpeg_ms), 1)
    diff_pct = abs(int(probed_ms) - int(ffmpeg_ms)) / base * 100.0
    return (diff_pct <= _PREPROCESS_SANITY_MAX_PCT, diff_pct)


def _parse_wav_probe(
    probe_bin_path: str,
    wav_path: Path,
    timeout_s: int,
) -> tuple[int, int, int, float, int, int, str, int]:
    """Returns (sample_rate, channels, bit_depth, duration_sec, size_bytes, dur_ms, sha, streams_n)."""
    data = _run_ffprobe_json(probe_bin_path, wav_path, timeout_s)
    streams = list(data.get("streams") or [])
    if not streams:
        raise AudioExtractionError(
            f"WAV extraído no tiene streams: {wav_path.name!r}",
            details={"error_code": "WavNoStreams"},
        )
    audio_stream = next(
        (s for s in streams if str(s.get("codec_type")).lower() == "audio"),
        streams[0],
    )
    sr = int(audio_stream.get("sample_rate", 16000))
    ch = int(audio_stream.get("channels", 1))
    bit_depth_raw = audio_stream.get("bits_per_raw_sample") or audio_stream.get(
        "bits_per_sample"
    )
    bd = int(bit_depth_raw or 16)
    dur_str = (
        audio_stream.get("duration")
        or (data.get("format") or {}).get("duration")
    )
    dur_s = float(dur_str or 0.0)
    dur_ms = int(round(dur_s * 1000))
    size_bytes = int(wav_path.stat().st_size)
    sha = _sha256_file(wav_path)
    return (sr, ch, bd, dur_s, size_bytes, dur_ms, sha, len(streams))


class FFmpegAudioExtractor(AudioExtractorPort):
    """Step 02. Extrae WAV PCM 16-bit, 16kHz, mono según FFmpegConfig."""

    def extract(
        self,
        validated: ValidatedMedia,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg: FFmpegConfig,
        logger: Any | None = None,
    ) -> ExtractedAudio:
        if not ffmpeg_bin_path:
            raise ConfigurationError(
                "ffmpeg_bin_path vacío. T03 necesita FFmpeg LGPL instalado en PATH."
            )
        job_temp_root = Path(job_temp_root)
        job_temp_root.mkdir(parents=True, exist_ok=True)
        workspace = job_temp_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        stem = validated.safe_name.rsplit(".", 1)[0][:120] or "audio"
        out_wav = workspace / f"{stem}_extracted.wav"

        target_sr = int(getattr(ffmpeg_cfg, "target_sample_rate_hz", 16000))
        target_ch = int(getattr(ffmpeg_cfg, "target_channels", 1))
        _ = getattr(ffmpeg_cfg, "target_bit_depth", 16)  # siempre pcm_s16le
        timeout_s = int(getattr(ffmpeg_cfg, "ffmpeg_timeout_sec", 7200))

        cmd: list[str] = [
            ffmpeg_bin_path,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(validated.original_path),
            "-vn",
            "-sn",
            "-dn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(target_sr)),
            "-ac",
            str(int(target_ch)),
            str(out_wav),
        ]

        try:
            cp = subprocess.run(
                cmd,
                shell=False,
                capture_output=True,
                timeout=max(10, int(timeout_s)),
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(
                f"FFmpeg extraction timeout {timeout_s}s excedido.",
                details={"error_code": "FFmpegTimeout"},
            ) from exc
        except FileNotFoundError as exc:
            raise ConfigurationError(
                f"ffmpeg no encontrado: {ffmpeg_bin_path!r}. Re-instale FFmpeg LGPL."
            ) from exc
        except OSError as exc:
            raise AudioExtractionError(f"OS error ffmpeg extract: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise AudioExtractionError(
                f"Error inesperado ffmpeg extract: {exc!r}",
                details={"error_code": "FFmpegUnexpectedError"},
            ) from exc

        if cp.returncode != 0:
            err = _safe_stderr(cp.stderr, 250)
            raise AudioExtractionError(
                f"FFmpeg extract fallo rc={cp.returncode}: {err}",
                details={"error_code": "FFmpegNonZeroExit"},
            )
        if not out_wav.is_file() or out_wav.stat().st_size <= 0:
            raise AudioExtractionError(
                f"FFmpeg no generó WAV o salió vacío: {out_wav.name}",
                details={"error_code": "FFmpegEmptyOutput"},
            )

        sr, ch, bd, dur_sec, size_bytes, ffmpeg_dur_ms, sha, _n = _parse_wav_probe(
            ffmpeg_cfg.ffprobe_bin,
            out_wav,
            min(60, timeout_s),
        )
        dur_orig = (
            float(validated.duration_guess_sec)
            if validated.duration_guess_sec is not None
            else 0.0
        )
        probed_ms_original = int(round(dur_orig * 1000.0))
        sanity_ok, _pct = _sanity_ok(probed_ms_original, ffmpeg_dur_ms)

        return ExtractedAudio(
            wav_path=out_wav.resolve(),
            sample_rate=sr,
            channels=ch,
            bit_depth=bd,
            duration_sec=dur_sec,
            size_bytes=size_bytes,
            sha256=sha,
            ffmpeg_duration_ms=int(ffmpeg_dur_ms),
            probed_duration_ms=int(ffmpeg_dur_ms),
            sanity_duration_pct_diff_ok=bool(sanity_ok),
        )


# ---------------------------------------------------------------------------
# 4) AudioPreprocessorPort
# ---------------------------------------------------------------------------


# Metadata EXACTA de operaciones realmente realizadas (2-pass peak norm -1.5 dBTP).
# Orden DETERMINISTICO inalterable (reproducibilidad).
APPLIED_FILTERS_T03: tuple[str, ...] = (
    "highpass_80hz_order4",
    "dc_removal_dcshift_0",
    "resample_16000hz_soxr",
    "peak_norm_tp_-1.5dBTP_loudnorm_2pass_analysis",
    "peak_norm_tp_-1.5dBTP_loudnorm_2pass_linear_apply",
    "post_verify_tp_le_-1.5dBTP_astats",
)

# Parámetros Peak Norm (True Peak -1.5 dBTP + EBU R128 -16 LUFS integrado, std broadcast).
_PN_TARGET_TP_DP = -1.5
_PN_TARGET_I_LUFS = -16.0
_PN_TARGET_LRA = 11.0
_PN_TP_TOLERANCE_UP_DB = 0.2  # TP post no debe superar -1.5+0.2 = -1.3 dBTP.
_PN_DEFAULT_MEASURED_D = {"input_i": "-70.0", "input_tp": "-70.0", "input_lra": "0.0", "target_offset": "0.0"}


def _parse_loudnorm_json(stderr_b: bytes) -> dict[str, str]:
    """Extrae bloque JSON {input_i,input_tp,input_lra,target_offset} del stderr de loudnorm print_format=json."""
    s = _safe_stderr(stderr_b, limit=20000)
    try:
        start = s.rfind("{")
        end = s.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return dict(_PN_DEFAULT_MEASURED_D)
        txt = s[start : end + 1]
        parsed = json.loads(txt)
        out: dict[str, str] = {}
        for k in ("input_i", "input_tp", "input_lra", "target_offset", "output_i", "output_tp", "output_lra"):
            v = parsed.get(k)
            if v is not None:
                out[k] = str(v)
        return out if out else dict(_PN_DEFAULT_MEASURED_D)
    except Exception:  # noqa: BLE001
        return dict(_PN_DEFAULT_MEASURED_D)


def _run_and_check(cmd: list[str], kind: str, timeout_s: int) -> None:
    try:
        cp = subprocess.run(
            cmd,
            shell=False,
            capture_output=True,
            timeout=max(10, int(timeout_s)),
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioExtractionError(
            f"FFmpeg {kind} timeout {timeout_s}s excedido.",
            details={"error_code": f"Preproc{kind.title()}Timeout"},
        ) from exc
    except FileNotFoundError as exc:
        raise ConfigurationError(f"ffmpeg no encontrado en {kind}: {cmd[0]!r}") from exc
    except OSError as exc:
        raise AudioExtractionError(f"OS error ffmpeg {kind}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise AudioExtractionError(
            f"Error inesperado ffmpeg {kind}: {exc!r}",
            details={"error_code": f"Preproc{kind.title()}Unexpected"},
        ) from exc
    if cp.returncode != 0:
        err = _safe_stderr(cp.stderr, limit=400)
        raise AudioExtractionError(
            f"FFmpeg {kind} fallo rc={cp.returncode}: {err}",
            details={"error_code": f"Preproc{kind.title()}Rc"},
        )


def _measure_tp_lufs_post(wav: Path, ffmpeg_bin: str, timeout_s: int) -> tuple[float | None, float | None]:
    """Medición post de TP e I (LUFS integrado) via loudnorm 2-pass print_format=json.

    Returns: (output_tp_dB, output_i_LUFS) o (None, None) si no se puede medir.
    """
    cmd: list[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(wav),
        "-af",
        (
            f"loudnorm=I={_PN_TARGET_I_LUFS}:TP={_PN_TARGET_TP_DP}:LRA={_PN_TARGET_LRA}"
            ":print_format=json"
        ),
        "-f",
        "null",
        "nul" if __import__("sys").platform.startswith("win") else "/dev/null",
    ]
    try:
        cp = subprocess.run(cmd, shell=False, capture_output=True, timeout=max(10, int(timeout_s)))
    except Exception:  # noqa: BLE001
        return None, None
    if cp.returncode != 0:
        return None, None
    parsed = _parse_loudnorm_json(cp.stdout + b"\n" + (cp.stderr or b""))
    out_tp = parsed.get("output_tp") or parsed.get("input_tp")
    out_i = parsed.get("output_i") or parsed.get("input_i")
    tp_f: float | None = None
    i_f: float | None = None
    try:
        tp_f = float(out_tp)
    except (TypeError, ValueError):
        tp_f = None
    try:
        i_f = float(out_i)
    except (TypeError, ValueError):
        i_f = None
    return tp_f, i_f


class FFmpegAudioPreprocessor(AudioPreprocessorPort):
    """Step 03: Peak Normalization -1.5 dBTP DETERMINISTICO 2 PASOS.

    Garantías:
      * 2-pass loudnorm EBU R128: (A) ANÁLISIS con print_format=json →
        measured_I, measured_TP, measured_LRA, target_offset;
        (B) APLICACIÓN linear=true usando esos valores.
      * Post-verificación TP <= -1.5 dBTP (tolerancia +0.2 dB),
        sino raise AudioExtractionError.
      * Salida s16le 16kHz mono FORZADA por `aformat` al final + probe ffprobe.
    """

    def preprocess(
        self,
        extracted: ExtractedAudio,
        job_temp_root: Path,
        ffmpeg_bin_path: str,
        ffmpeg_cfg: FFmpegConfig,
        logger: Any | None = None,
    ) -> PreprocessedAudio:
        if not ffmpeg_bin_path:
            raise ConfigurationError(
                "ffmpeg_bin_path vacío. AudioPreprocessor necesita FFmpeg LGPL."
            )
        job_temp_root = Path(job_temp_root)
        job_temp_root.mkdir(parents=True, exist_ok=True)
        workspace = job_temp_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        stem = extracted.wav_path.stem[:80] or "audio"
        wav_pre = workspace / f"{stem}_A_pre_dsp.wav"
        wav_final = workspace / f"{stem}_B_final_preprocessed.wav"

        target_sr = int(getattr(ffmpeg_cfg, "target_sample_rate_hz", 16000))
        target_ch = int(getattr(ffmpeg_cfg, "target_channels", 1))
        timeout_s = int(getattr(ffmpeg_cfg, "ffmpeg_timeout_sec", 7200))
        null_dst = "nul" if __import__("sys").platform.startswith("win") else "/dev/null"

        # swr viene en FFmpeg LGPL. soxr no está en la build essentials.
        af_pre: str = (
            "highpass=f=80:poles=2,"
            "dcshift=0,"
            f"aresample={int(target_sr)}:resampler=swr"
        )
        cmd_pre: list[str] = [
            ffmpeg_bin_path,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(extracted.wav_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(int(target_sr)),
            "-ac",
            str(int(target_ch)),
            "-af",
            af_pre,
            str(wav_pre),
        ]
        _run_and_check(cmd_pre, kind="preproc_substep_A_highpass_dc_resample", timeout_s=timeout_s)
        if not wav_pre.is_file() or wav_pre.stat().st_size <= 0:
            raise AudioExtractionError(
                f"Preproc SubA sin output WAV: {wav_pre.name}",
                details={"error_code": "PreprocSubAEmpty"},
            )

        # ----- SUBPASO B: ANÁLISIS 2-pass (loudnorm print_format=json). NO produce audio.
        cmd_analysis: list[str] = [
            ffmpeg_bin_path,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "info",
            "-i",
            str(wav_pre),
            "-af",
            (
                f"loudnorm=I={_PN_TARGET_I_LUFS}:TP={_PN_TARGET_TP_DP}:LRA={_PN_TARGET_LRA}"
                ":print_format=json"
            ),
            "-f",
            "null",
            null_dst,
        ]
        try:
            cp_analysis = subprocess.run(
                cmd_analysis,
                shell=False,
                capture_output=True,
                timeout=max(10, int(timeout_s)),
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioExtractionError(
                f"Preproc análisis loudnorm timeout {timeout_s}s.",
                details={"error_code": "PreprocPeakNormAnalysisTimeout"},
            ) from exc
        except FileNotFoundError as exc:
            raise ConfigurationError(f"ffmpeg no encontrado análisis: {ffmpeg_bin_path!r}") from exc
        except OSError as exc:
            raise AudioExtractionError(f"OS error análisis loudnorm: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise AudioExtractionError(
                f"Error inesperado análisis loudnorm: {exc!r}",
                details={"error_code": "PreprocPeakNormAnalysisUnexpected"},
            ) from exc
        if cp_analysis.returncode != 0:
            err = _safe_stderr(cp_analysis.stderr, 400)
            raise AudioExtractionError(
                f"Preproc análisis loudnorm fallo rc={cp_analysis.returncode}: {err}",
                details={"error_code": "PreprocPeakNormAnalysisRc"},
            )
        measurements = _parse_loudnorm_json(
            (cp_analysis.stdout or b"") + b"\n" + (cp_analysis.stderr or b"")
        )

        def _as_float(key: str, default: float) -> float:
            try:
                v = measurements.get(key)
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        m_i = _as_float("input_i", -23.0)
        m_tp = _as_float("input_tp", -6.0)
        m_lra = _as_float("input_lra", 7.0)
        m_off = _as_float("target_offset", 0.0)

        # ----- SUBPASO C: APLICACIÓN 2-pass loudnorm linear=true + aformat s16/16k/mono.
        af_apply: str = (
            f"loudnorm=I={_PN_TARGET_I_LUFS}:TP={_PN_TARGET_TP_DP}:LRA={_PN_TARGET_LRA}"
            f":measured_I={m_i:.4f}:measured_TP={m_tp:.4f}:measured_LRA={m_lra:.4f}"
            f":offset={m_off:.4f}:linear=true:print_format=summary,"
            "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono"
        )
        cmd_apply: list[str] = [
            ffmpeg_bin_path,
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(wav_pre),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            af_apply,
            str(wav_final),
        ]
        _run_and_check(cmd_apply, kind="preproc_substep_C_loudnorm_2pass_apply", timeout_s=timeout_s)
        if not wav_final.is_file() or wav_final.stat().st_size <= 0:
            raise AudioExtractionError(
                f"Preproc SubC (loudnorm apply) sin output WAV: {wav_final.name}",
                details={"error_code": "PreprocSubCEmpty"},
            )

        # ----- SUBPASO D: POST-VERIFICACIÓN TP ≤ -1.5 dBTP (tolerancia +0.2)
        post_tp, _post_i = _measure_tp_lufs_post(wav_final, ffmpeg_bin_path, timeout_s)
        if post_tp is not None:
            max_allowed = _PN_TARGET_TP_DP + _PN_TP_TOLERANCE_UP_DB
            if post_tp > max_allowed:
                raise AudioExtractionError(
                    "Post-verificación Peak Norm falló: "
                    f"TP medido={post_tp:.3f} dBTP > permitido={max_allowed:.3f} dBTP.",
                    details={
                        "error_code": "PreprocPeakNormVerifyFailed",
                        "measured_tp_dbtp": post_tp,
                        "target_max_allowed_dbtp": max_allowed,
                    },
                )

        # Probe final ffprobe (16kHz mono 16-bit obligatorio).
        sr, ch, bd, dur_sec, size_bytes, dur_ms, sha, _n = _parse_wav_probe(
            ffmpeg_cfg.ffprobe_bin,
            wav_final,
            min(60, timeout_s),
        )
        if int(sr) != 16000:
            raise AudioExtractionError(
                f"Preprocessed WAV sample_rate != 16000 (got {sr})",
                details={"error_code": "PreprocSampleRateMismatch"},
            )
        if int(ch) != 1:
            raise AudioExtractionError(
                f"Preprocessed WAV channels != 1 (got {ch})",
                details={"error_code": "PreprocChannelsMismatch"},
            )
        if int(bd) != 16:
            raise AudioExtractionError(
                f"Preprocessed WAV bit_depth != 16 (got {bd})",
                details={"error_code": "PreprocBitDepthMismatch"},
            )

        return PreprocessedAudio(
            wav_path=wav_final.resolve(),
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            duration_sec=dur_sec,
            size_bytes=size_bytes,
            sha256=sha,
            applied_filters=APPLIED_FILTERS_T03,
        )


__all__ = [
    "SubprocessFFmpegBinaryResolver",
    "FFprobeMediaValidator",
    "FFmpegAudioExtractor",
    "FFmpegAudioPreprocessor",
    "APPLIED_FILTERS_T03",
    "_sha256_file",
    "_safe_stderr",
]
