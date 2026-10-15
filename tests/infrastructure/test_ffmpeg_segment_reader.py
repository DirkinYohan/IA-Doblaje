"""Lector de audio por segmentos sobre FFmpeg REAL (sin dobles)."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.core.exceptions import AudioExtractionError
from app.infrastructure.audio.ffmpeg_segment_reader import FFmpegSegmentReader

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="FFmpeg no instalado")


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """MP4 de 12 s: 0-4 s silencio, 4-12 s tono de 440 Hz (estéreo, 44,1 kHz)."""
    target = tmp_path_factory.mktemp("seg") / "tone.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=160x120:d=12",
         "-f", "lavfi", "-i", "aevalsrc='if(gte(t,4),0.5*sin(2*PI*440*t),0)':s=44100:c=stereo:d=12",
         "-shortest", "-c:v", "libx264", "-c:a", "aac", str(target)],
        check=True, capture_output=True, timeout=120,
    )
    return target


def _reader(path: Path) -> FFmpegSegmentReader:
    return FFmpegSegmentReader(path, ffmpeg_bin=str(FFMPEG), ffprobe_bin=str(FFPROBE))


def test_mide_la_duracion(media: Path) -> None:
    reader = _reader(media)
    assert 11_800 <= reader.duration_ms <= 12_200
    assert reader.total_frames == int(reader.duration_ms * 16)


def test_lee_solo_el_tramo_pedido_en_16k_mono(media: Path) -> None:
    reader = _reader(media)
    audio = reader.read(5_000, 7_000)
    assert audio.dtype == np.float32
    assert abs(audio.size - 32_000) <= 160, audio.size
    assert float(np.sqrt(np.mean(audio**2))) > 0.1, "el tramo con tono debe tener energía"


def test_la_busqueda_es_precisa(media: Path) -> None:
    reader = _reader(media)
    silent = reader.read(500, 3_500)
    assert float(np.sqrt(np.mean(silent**2))) < 0.01, "0-4 s es silencio"


def test_recorta_a_la_duracion(media: Path) -> None:
    reader = _reader(media)
    assert reader.read(11_000, 99_000).size <= 16_000 + 400
    assert reader.read(50_000, 60_000).size == 0
    assert reader.read(3_000, 3_000).size == 0


def test_no_modifica_el_original(media: Path) -> None:
    before = hashlib.sha256(media.read_bytes()).hexdigest()
    reader = _reader(media)
    reader.read(0, 12_000)
    assert hashlib.sha256(media.read_bytes()).hexdigest() == before


def test_rechaza_medio_inexistente(tmp_path: Path) -> None:
    with pytest.raises(AudioExtractionError):
        FFmpegSegmentReader(tmp_path / "nada.mp4", ffmpeg_bin=str(FFMPEG), ffprobe_bin=str(FFPROBE))


def test_rechaza_medio_sin_audio(tmp_path: Path) -> None:
    mute = tmp_path / "mute.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=black:s=160x120:d=2", "-c:v", "libx264", str(mute)],
        check=True, capture_output=True, timeout=60,
    )
    with pytest.raises(AudioExtractionError):
        FFmpegSegmentReader(mute, ffmpeg_bin=str(FFMPEG), ffprobe_bin=str(FFPROBE))
