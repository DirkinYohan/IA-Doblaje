from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from app.presentation.api import cluster
from app.presentation.api.common import _remux_for_web

FFMPEG = shutil.which("ffmpeg")


def _top_level_boxes(path: Path) -> list[str]:
    boxes: list[str] = []
    with path.open("rb") as handle:
        while True:
            header = handle.read(8)
            if len(header) < 8:
                return boxes
            size, kind = struct.unpack(">I4s", header)
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
                handle.seek(size - 16, 1)
            else:
                handle.seek(size - 8, 1)
            boxes.append(kind.decode("latin-1"))


@pytest.mark.skipif(FFMPEG is None, reason="sin ffmpeg")
def test_remux_pone_el_indice_al_principio(tmp_path: Path) -> None:
    source = tmp_path / "movil.mov"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=duration=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(source)],
        check=True,
    )
    boxes = _top_level_boxes(source)
    assert boxes.index("mdat") < boxes.index("moov")

    web = _remux_for_web(source)
    assert web is not None and web.suffix == ".mp4"
    boxes = _top_level_boxes(web)
    assert boxes.index("moov") < boxes.index("mdat")


def test_remux_de_algo_que_no_es_video_devuelve_none(tmp_path: Path) -> None:
    broken = tmp_path / "roto.mp4"
    broken.write_bytes(b"no soy un video")
    assert _remux_for_web(broken) is None
    assert not (tmp_path / "roto.web.mp4").exists()


def test_ticket_se_valida_con_la_hora_de_llegada(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.presentation.api import db

    conn = db.connect(tmp_path / "app.sqlite")
    db.migrate(conn)
    clock = {"t": 1_000.0}
    monkeypatch.setattr(cluster, "now", lambda: clock["t"])

    cluster.issue_ticket(conn, "h1", "u1", 60)
    clock["t"] = 1_090.0  # la subida tardó 90 s en llegar entera
    assert cluster.consume_ticket(conn, "h2-no-existe", at=1_005.0) is None
    assert cluster.consume_ticket(conn, "h1", at=1_005.0) == "u1"
    assert cluster.consume_ticket(conn, "h1", at=1_005.0) is None  # de un solo uso

    cluster.issue_ticket(conn, "h3", "u1", 60)
    clock["t"] = 1_200.0
    assert cluster.consume_ticket(conn, "h3") is None  # sin hora de llegada: caducado

    cluster.issue_ticket(conn, "h4", "u1", 60)
    clock["t"] = 1_300.0
    cluster.issue_ticket(conn, "h5", "u1", 60)  # emitir otro no borra uno recién caducado
    assert cluster.consume_ticket(conn, "h4", at=1_210.0) == "u1"
    conn.close()


@pytest.fixture
def client(tmp_path: Path):
    import os

    from fastapi.testclient import TestClient

    from app.presentation.api import worker

    worker.reset()
    os.environ["IA_DB_PATH"] = str(tmp_path / "app.sqlite")
    os.environ["IA_MAIL_DIR"] = str(tmp_path / "mail")
    os.environ["IA_DISABLE_WORKER"] = "1"
    from app.presentation.api.main import create_app
    from tests.fixtures.accounts import sign_up

    with TestClient(create_app()) as test_client:
        response = sign_up(
            test_client,
            json={"name": "Larga", "email": "larga@example.com", "password": "password123", "confirm": "password123"},
        )
        assert response.status_code == 200, response.text
        yield test_client
    os.environ.pop("IA_DISABLE_WORKER", None)
    worker.reset()


@pytest.mark.skipif(FFMPEG is None, reason="sin ffmpeg")
def test_subida_larga_con_ticket_no_cierra_la_sesion(
    client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from fastapi.testclient import TestClient

    video = tmp_path / "largo.mov"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25:duration=2",
         "-f", "lavfi", "-i", "sine=duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(video)],
        check=True,
    )
    ticket = client.post("/api/ws-ticket").json()["ticket"]
    real = time.time
    # Cuando el servidor termina de recibir el archivo ya pasaron 2 minutos.
    monkeypatch.setattr(cluster, "now", lambda: real() + 120)
    anonymous = TestClient(client.app)
    upload = anonymous.post(
        f"/api/videos?ticket={ticket}",
        files={"file": ("largo.mov", video.read_bytes(), "video/quicktime")},
        data={"source_language": "en", "target_languages": "es"},
    )
    assert upload.status_code == 200, upload.text
    media = client.get(f"/api/videos/{upload.json()['id']}/media")
    assert media.status_code == 200
    assert media.headers["content-type"] == "video/mp4"
    stored = tmp_path / "stored.mp4"
    stored.write_bytes(media.content)
    boxes = _top_level_boxes(stored)
    assert boxes.index("moov") < boxes.index("mdat")
