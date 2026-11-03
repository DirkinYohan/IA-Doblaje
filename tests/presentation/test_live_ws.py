from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.presentation.api import db, live
from app.presentation.api.main import create_app
from tests.fixtures.accounts import sign_up

SR = 16_000
DURATION = 60_000
DIALOGUE = [(1_000, 2_500, "hello"), (14_000, 15_500, "middle"), (40_000, 42_000, "late")]


class _Seg:
    def __init__(self, start_ms: int, end_ms: int, text: str) -> None:
        self.start_ms, self.end_ms, self.text = start_ms, end_ms, text


class _SlowModels:
    def __init__(self, delay: float = 0.15) -> None:
        self.delay = delay
        self.transcribed = 0

    def transcribe(self, audio: Any, offset_ms: int, _language: str | None) -> list[Any]:
        time.sleep(self.delay)
        self.transcribed += 1
        end = offset_ms + int(audio.size * 1000 / SR)
        return [_Seg(s, e, t) for s, e, t in DIALOGUE if offset_ms <= s and e <= end]

    def translate(self, texts: list[str], _source: str, target: str) -> list[str]:
        time.sleep(self.delay)
        return [f"[{target}] {t}" for t in texts]


class _Reader:
    duration_ms = DURATION
    sample_rate = SR

    def read(self, start: int, end: int) -> Any:
        return np.full(int((end - start) * SR / 1000), 0.3, dtype=np.float32)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    models = _SlowModels()
    monkeypatch.setattr(live, "open_reader", lambda _project: _Reader())
    monkeypatch.setattr(live, "model_transcribe", models.transcribe)
    monkeypatch.setattr(live, "model_translate", models.translate)
    monkeypatch.setattr(live, "model_detect", lambda _audio: "en")
    monkeypatch.setattr(live, "warm", lambda **_kw: None)
    client = TestClient(create_app())
    yield client, models
    live.stop_all()


def _register(client: TestClient, email: str) -> str:
    response = sign_up(
        client,
        json={"name": "Usuario Prueba", "email": email, "password": "password123", "confirm": "password123"},
    )
    assert response.status_code == 200, response.text
    return client.get("/api/auth/me").json()["id"]


def _project(client: TestClient, user_id: str) -> str:
    project_id = str(uuid.uuid4())
    db.run(
        client.app.state.conn,
        "INSERT INTO projects (id, user_id, title, source_language, target_languages, status,"
        " media_path, created_at, duration_ms) VALUES (?,?,?,?,?,?,?,?,?)",
        (project_id, user_id, "t", "en", "es,fr", "READY", "x.mp4", "2026-01-01T00:00:00+00:00", DURATION),
    )
    return project_id


def _collect(socket, until, limit: int = 400) -> list[dict[str, Any]]:
    events = []
    for _ in range(limit):
        event = socket.receive_json()
        events.append(event)
        if until(event):
            break
    return events


def test_los_primeros_subtitulos_llegan_antes_de_terminar(setup) -> None:
    client, _models = setup
    project_id = _project(client, _register(client, "a@example.com"))
    with client.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        snapshot = socket.receive_json()
        assert snapshot["type"] == "snapshot" and snapshot["tracks"] == {}
        socket.send_json({"type": "position", "time_ms": 0, "language": "es"})
        events = _collect(socket, lambda e: e["type"] == "completed")
    kinds = [e["type"] for e in events]
    first_es = next(i for i, e in enumerate(events) if e["type"] == "cues" and e["language"] == "es" and e["cues"])
    assert first_es < kinds.index("completed"), "subtítulos parciales antes de terminar"
    first = events[first_es]["cues"][0]
    assert (first["start_ms"], first["end_ms"]) == (1_000, 2_500), "tiempos del audio original"
    assert first["id"] == "es:1000:2500"
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    tracks = client.get(f"/api/subtitles/{project_id}").json()["subtitles"]
    by_lang = {t["language"]: t for t in tracks}
    assert set(by_lang) == {"en", "es", "fr"}
    for track in by_lang.values():
        assert [c["start_ms"] for c in track["cues"]] == [1_000, 14_000, 40_000]
        assert track["covered_ms"] == DURATION


def test_reconectar_recupera_lo_perdido_sin_duplicar(setup) -> None:
    client, models = setup
    models.delay = 0.3
    project_id = _project(client, _register(client, "b@example.com"))
    with client.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        socket.receive_json()
        socket.send_json({"type": "position", "time_ms": 0, "language": "es"})
        _collect(socket, lambda e: e["type"] == "cues" and e["language"] == "es" and e["cues"])
    time.sleep(1.0)
    with client.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        snapshot = socket.receive_json()
        assert snapshot["type"] == "snapshot" and snapshot["tracks"]["es"]
        later = _collect(socket, lambda e: e["type"] == "completed")
    assert all(e["seq"] > snapshot["seq"] for e in later), "nada repetido de la instantánea"
    ids = [c["id"] for c in snapshot["tracks"]["es"]]
    ids += [c["id"] for e in later if e["type"] == "cues" and e["language"] == "es" for c in e["cues"]]
    assert len(ids) == len(set(ids)), "sin duplicados"
    assert sorted(ids) == sorted(["es:1000:2500", "es:14000:15500", "es:40000:42000"])


def test_volver_a_abrir_un_video_procesado_no_repite_trabajo(setup) -> None:
    client, models = setup
    project_id = _project(client, _register(client, "c@example.com"))
    with client.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        socket.receive_json()
        _collect(socket, lambda e: e["type"] == "completed")
    before = models.transcribed
    with client.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        snapshot = socket.receive_json()
        assert snapshot["complete"] is True and len(snapshot["tracks"]["fr"]) == 3
        assert socket.receive_json()["type"] == "completed"
    assert models.transcribed == before, "ni una pasada más de Whisper"


def test_un_usuario_no_ve_los_subtitulos_de_otro(setup) -> None:
    client, _models = setup
    project_id = _project(client, _register(client, "d@example.com"))
    client.post("/api/auth/logout")
    _register(client, "e@example.com")
    with client.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        message = socket.receive_json()
    assert message["type"] == "error" and message["code"] == "not_found"
