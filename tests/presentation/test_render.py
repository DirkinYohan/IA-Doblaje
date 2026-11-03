"""Vídeo con subtítulos incrustados: FFmpeg real sobre un vídeo generado."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.infrastructure.subtitles.exporters import to_ass_burn
from app.presentation.api import db, render
from app.presentation.api.main import create_app

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="falta FFmpeg")

W, H = 640, 360


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str = ""


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3 s de vídeo de prueba (fondo gris uniforme) con audio."""
    target = tmp_path_factory.mktemp("render") / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s={W}x{H}:d=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(target),
        ],
        check=True, timeout=120,
    )
    return target


def _frame(video: Path, at: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(at), "-i", str(video),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True, timeout=60,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(H, W).astype(np.int16)


def _streams(video: Path) -> set[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return {s["codec_name"] for s in json.loads(out)["streams"]}


def test_los_subtitulos_quedan_dibujados_en_la_imagen(clip: Path, tmp_path: Path) -> None:
    width, height, duration = render.probe(clip)
    assert (width, height) == (W, H) and 2900 <= duration <= 3100
    ass = to_ass_burn([Cue(1, 0, 2000, "HOLA MUNDO")], language="es", width=W, height=H)
    progress: list[int] = []
    out = tmp_path / "es.mp4"
    render.burn(clip, ass, out, duration_ms=duration, on_progress=progress.append)

    assert out.is_file()
    assert {"h264", "aac"} <= _streams(out), "vídeo H.264 y el audio conservado"
    assert progress and progress == sorted(progress), "progreso creciente"

    def changed(a: np.ndarray, b: np.ndarray) -> float:
        """Fracción de píxeles claramente distintos (el texto ocupa poca área)."""
        return float((np.abs(a - b) > 30).mean())

    with_text, original = _frame(out, 1.0), _frame(clip, 1.0)
    bottom = changed(with_text[int(H * 0.7):], original[int(H * 0.7):])
    top = changed(with_text[: int(H * 0.5)], original[: int(H * 0.5)])
    assert bottom > 0.03, f"el subtítulo debe verse abajo ({bottom:.1%} de píxeles cambiados)"
    assert top < 0.001, f"la parte de arriba no cambia ({top:.1%} de píxeles cambiados)"
    # Pasado su intervalo, el subtítulo desaparece.
    after = changed(_frame(out, 2.6)[int(H * 0.7):], _frame(clip, 2.6)[int(H * 0.7):])
    assert after < 0.001


# --- API -------------------------------------------------------------------

@pytest.fixture
def api(tmp_path: Path, monkeypatch, clip: Path):
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    render.reset()
    client = TestClient(create_app())
    yield client
    render.reset()


def _user(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/register", json={
        "name": "Usuario Prueba", "email": email, "password": "password123", "confirm": "password123",
    })
    assert r.status_code == 200, r.text
    return client.get("/api/auth/me").json()["id"]


def _project_with_track(client: TestClient, user_id: str, clip: Path) -> str:
    project_id = str(uuid.uuid4())
    conn = client.app.state.conn
    db.run(conn, "INSERT INTO projects (id, user_id, title, source_language, target_languages, status,"
           " media_path, created_at, duration_ms) VALUES (?,?,?,?,?,?,?,?,?)",
           (project_id, user_id, "Mi_video.mp4", "en", "es", "COMPLETED", str(clip),
            "2026-01-01T00:00:00+00:00", 3000))
    cues = [{"index": 1, "start_ms": 0, "end_ms": 2000, "text": "HOLA", "speaker": "", "language": "es"}]
    db.run(conn, "INSERT INTO subtitles (project_id, language, cues_json, updated_at) VALUES (?,?,?,?)",
           (project_id, "es", json.dumps(cues), "v1"))
    return project_id


def _wait_done(client: TestClient, project_id: str) -> dict:
    for _ in range(240):
        status = client.get(f"/api/videos/{project_id}/render", params={"lang": "es"}).json()
        if status["state"] in {"done", "failed"}:
            return status
        time.sleep(0.25)
    raise AssertionError("el render no terminó")


def test_flujo_completo_por_la_api(api: TestClient, clip: Path) -> None:
    project_id = _project_with_track(api, _user(api, "r1@example.com"), clip)
    assert api.get(f"/api/videos/{project_id}/render", params={"lang": "es"}).json()["state"] == "idle"
    assert api.get(f"/api/videos/{project_id}/render/file", params={"lang": "es"}).status_code == 404

    started = api.post(f"/api/videos/{project_id}/render", json={"lang": "es"}).json()
    assert started["state"] in {"queued", "running", "done"}
    assert _wait_done(api, project_id)["state"] == "done"

    file = api.get(f"/api/videos/{project_id}/render/file", params={"lang": "es"})
    assert file.status_code == 200
    assert file.headers["content-type"] == "video/mp4"
    assert "Mi video (es).mp4" in file.headers["content-disposition"]
    assert len(file.content) > 1000

    # Sin cambios en la pista, no se vuelve a renderizar.
    assert api.post(f"/api/videos/{project_id}/render", json={"lang": "es"}).json()["state"] == "done"

    # Editar la pista deja el vídeo desactualizado, pero se puede descargar
    # (la pista puede cambiar mientras se renderiza) y volver a preparar.
    db.run(api.app.state.conn, "UPDATE subtitles SET updated_at='v2' WHERE project_id=?", (project_id,))
    assert api.get(f"/api/videos/{project_id}/render", params={"lang": "es"}).json()["state"] == "stale"
    assert api.get(f"/api/videos/{project_id}/render/file", params={"lang": "es"}).status_code == 200
    again = api.post(f"/api/videos/{project_id}/render", json={"lang": "es"}).json()
    assert again["state"] in {"queued", "running"}
    assert _wait_done(api, project_id)["state"] == "done"


def test_idioma_sin_pista_y_proyecto_ajeno(api: TestClient, clip: Path) -> None:
    project_id = _project_with_track(api, _user(api, "r2@example.com"), clip)
    assert api.post(f"/api/videos/{project_id}/render", json={"lang": "fr"}).status_code == 404
    assert api.post(f"/api/videos/{project_id}/render", json={}).status_code == 422
    api.post("/api/auth/logout")
    _user(api, "r3@example.com")
    assert api.post(f"/api/videos/{project_id}/render", json={"lang": "es"}).status_code == 404
    assert api.get(f"/api/videos/{project_id}/render/file", params={"lang": "es"}).status_code == 404
