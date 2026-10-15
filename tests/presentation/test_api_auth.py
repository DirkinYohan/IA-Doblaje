import os
from pathlib import Path

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    from app.core.config import clear_settings_cache
    from app.presentation.api.main import create_app

    clear_settings_cache()

    with TestClient(create_app()) as test_client:
        yield test_client


def test_registro_login_logout_y_reset(client: TestClient, tmp_path: Path) -> None:
    reg = client.post(
        "/api/auth/register",
        json={"name": "Ada", "email": "ada@example.com", "password": "secreto123", "confirm": "secreto123"},
    )
    assert reg.status_code == 200
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json()["email"] == "ada@example.com"
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    login = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "secreto123", "remember": True})
    assert login.status_code == 200
    forgot = client.post("/api/auth/forgot-password", json={"email": "ada@example.com"})
    assert forgot.status_code == 200
    mail = next(Path(os.environ["IA_MAIL_DIR"]).glob("*.txt"))
    token = [line.split("=", 1)[1] for line in mail.read_text(encoding="utf-8").splitlines() if line.startswith("token=")][0]
    reset = client.post("/api/auth/reset-password", json={"token": token, "password": "otraClave9", "confirm": "otraClave9"})
    assert reset.status_code == 200
    assert client.post("/api/auth/login", json={"email": "ada@example.com", "password": "secreto123"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "ada@example.com", "password": "otraClave9"}).status_code == 200


def test_ruta_protegida_y_export(client: TestClient) -> None:
    assert client.get("/api/videos").status_code == 401
    client.post(
        "/api/auth/register",
        json={"name": "Ada", "email": "ada@example.com", "password": "secreto123", "confirm": "secreto123"},
    )
    project = "proj-12345678"
    conn = client.app.state.conn  # type: ignore[attr-defined]
    from app.presentation.api import db

    db.run(
        conn,
        """
        INSERT INTO projects (id, user_id, title, source_language, target_languages, status, media_path, created_at)
        VALUES (?, ?, 'clip.wav', 'en', 'es', 'COMPLETED', 'data/input/x.wav', '2026-01-01T00:00:00+00:00')
        """,
        (project, client.get("/api/auth/me").json()["id"]),
    )
    saved = client.put(
        f"/api/subtitles/{project}",
        json={
            "language": "es",
            "cues": [
                {"index": 1, "start_ms": 0, "end_ms": 1200, "text": "Hola", "speaker": "SPEAKER_00", "language": "es", "state": "final"}
            ],
        },
    )
    assert saved.status_code == 200
    srt = client.get(f"/api/export/{project}/srt", params={"lang": "es"})
    vtt = client.get(f"/api/export/{project}/vtt", params={"lang": "es"})
    ass = client.get(f"/api/export/{project}/ass", params={"lang": "es"})
    assert srt.status_code == 200
    assert srt.text.startswith("1")
    assert "Hola" in srt.text
    assert vtt.text.startswith("WEBVTT")
    assert "Dialogue:" in ass.text
    packed = client.get(f"/api/export/{project}/zip")
    assert packed.status_code == 200
    assert packed.headers["content-type"] == "application/zip"
