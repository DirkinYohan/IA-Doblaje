"""Tickets de WebSocket: autentican sin depender de que la cookie viaje."""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    from app.presentation.api import worker

    worker.reset()
    tmp = tmp_path_factory.mktemp("ws_ticket")
    os.environ["IA_DB_PATH"] = str(tmp / "app.sqlite")
    os.environ["IA_MAIL_DIR"] = str(tmp / "mail")
    os.environ["IA_DISABLE_WORKER"] = "1"
    from app.presentation.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    os.environ.pop("IA_DISABLE_WORKER", None)
    worker.reset()


@pytest.fixture(scope="module")
def job(client: TestClient) -> dict[str, str]:
    """Usuario con un job real en PENDING (creado al subir el medio)."""
    response = client.post(
        "/api/auth/register",
        json={"name": "Ticket", "email": "ticket@example.com", "password": "password123", "confirm": "password123"},
    )
    assert response.status_code == 200, response.text
    upload = client.post(
        "/api/videos",
        files={"file": ("a.wav", b"RIFF" + b"\0" * 2000, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    assert upload.status_code == 200, upload.text
    return {"job_id": upload.json()["job_id"], "token": client.cookies.get("ia_session") or ""}


def test_sin_sesion_no_hay_ticket(client: TestClient, job: dict[str, str]) -> None:
    anonymous = TestClient(client.app)
    assert anonymous.post("/api/ws-ticket").status_code == 401


def test_ticket_autentica_sin_cookie(client: TestClient, job: dict[str, str]) -> None:
    ticket = client.post("/api/ws-ticket").json()["ticket"]
    # Navegador en otro host: la cookie NO viaja al backend.
    anonymous = TestClient(client.app)
    with anonymous.websocket_connect(f"/api/jobs/{job['job_id']}/events?ticket={ticket}") as socket:
        first = socket.receive_json()
    assert first["type"] == "snapshot", first
    assert first["id"] == job["job_id"]


def test_ticket_es_de_un_solo_uso(client: TestClient, job: dict[str, str]) -> None:
    ticket = client.post("/api/ws-ticket").json()["ticket"]
    anonymous = TestClient(client.app)
    with anonymous.websocket_connect(f"/api/jobs/{job['job_id']}/events?ticket={ticket}") as socket:
        assert socket.receive_json()["type"] == "snapshot"
    with anonymous.websocket_connect(f"/api/jobs/{job['job_id']}/events?ticket={ticket}") as socket:
        message = socket.receive_json()
    assert message["type"] == "error" and message["code"] == "unauthorized"


def test_ticket_inventado_se_rechaza(client: TestClient, job: dict[str, str]) -> None:
    anonymous = TestClient(client.app)
    with anonymous.websocket_connect(f"/api/jobs/{job['job_id']}/events?ticket=falso") as socket:
        message = socket.receive_json()
    assert message == {"type": "error", "code": "unauthorized", "message": "No autenticado"}


def test_sin_ticket_ni_cookie_se_rechaza_con_codigo(client: TestClient, job: dict[str, str]) -> None:
    anonymous = TestClient(client.app)
    with anonymous.websocket_connect("/api/realtime") as socket:
        message = socket.receive_json()
    assert message["code"] == "unauthorized"


def test_la_cookie_sigue_funcionando(client: TestClient, job: dict[str, str]) -> None:
    with client.websocket_connect(f"/api/jobs/{job['job_id']}/events") as socket:
        assert socket.receive_json()["type"] == "snapshot"
