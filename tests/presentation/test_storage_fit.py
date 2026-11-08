from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from app.presentation.api import storage_fit


FFMPEG = shutil.which("ffmpeg")


def _video(path: Path, seconds: int) -> Path:
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )
    return path


def test_si_cabe_no_recodifica(tmp_path: Path) -> None:
    source = tmp_path / "small.mp4"
    source.write_bytes(b"x" * 100)
    assert storage_fit.fit(source, 1000, tmp_path / "out.mp4", limit=1000) == source


def test_demasiado_largo_avisa_sin_recodificar(tmp_path: Path) -> None:
    source = tmp_path / "big.mp4"
    source.write_bytes(b"x" * 2000)
    calls: list[object] = []
    with pytest.raises(storage_fit.TooLongForStorage, match="minutos"):
        storage_fit.fit(source, 3_600_000, tmp_path / "out.mp4", limit=1000, run=calls.append)
    assert calls == []


def test_tasa_de_bits_respeta_el_limite() -> None:
    limit = 50 * 1024 * 1024
    kbps = storage_fit.video_kbps(limit, 600_000)
    total_bits = (kbps + storage_fit.AUDIO_KBPS) * 1000 * 600
    assert total_bits / 8 <= limit
    assert storage_fit.max_minutes(limit) == 25


def test_admite_25_minutos_y_no_mas() -> None:
    limit = 50 * 1024 * 1024
    assert storage_fit.needs_fit(limit + 1, 25 * 60_000, limit=limit) is True
    assert storage_fit.needs_fit(limit + 1, 25 * 60_000 + 59_000, limit=limit) is True
    with pytest.raises(storage_fit.TooLongForStorage, match="25 minutos"):
        storage_fit.needs_fit(limit + 1, 26 * 60_000, limit=limit)
    assert storage_fit.audio_kbps(limit, 25 * 60_000) == 64
    assert storage_fit.audio_kbps(limit, 5 * 60_000) == 128
    total = (storage_fit.video_kbps(limit, 25 * 60_000) + 64) * 1000 * 25 * 60 / 8
    assert total <= limit


def test_un_video_largo_pequeno_no_se_toca() -> None:
    assert storage_fit.needs_fit(1000, 60 * 60_000, limit=2000) is False


@pytest.mark.skipif(FFMPEG is None, reason="sin ffmpeg")
def test_comprime_un_video_real_por_debajo_del_limite(tmp_path: Path) -> None:
    source = _video(tmp_path / "source.mp4", 6)
    limit = 1024 * 1024
    assert source.stat().st_size > limit
    out = storage_fit.fit(source, 6000, tmp_path / "fit.mp4", limit=limit)
    assert out == tmp_path / "fit.mp4"
    assert 0 < out.stat().st_size <= limit
    probe = subprocess.run(
        [FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,height",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "video" in probe and "audio" in probe and "720" in probe
