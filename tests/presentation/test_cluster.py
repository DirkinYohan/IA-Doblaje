from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.presentation.api import cluster, db, render, worker
from tests.fixtures.accounts import sign_up

USER = {"name": "Ana", "email": "ana@example.com", "password": "secreto123", "confirm": "secreto123"}


@pytest.fixture()
def shared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, TestClient]]:
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "shared.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    from app.presentation.api.main import create_app

    with TestClient(create_app()) as first, TestClient(create_app()) as second:
        yield first, second


def as_other_instance(monkeypatch: pytest.MonkeyPatch, name: str = "otro-proceso") -> None:
    monkeypatch.setattr(cluster, "INSTANCE_ID", name)


def test_un_ticket_emitido_en_un_proceso_vale_en_otro_y_una_sola_vez(shared) -> None:
    first, second = shared
    sign_up(first, json=USER)
    upload = first.post(
        "/api/videos",
        files={"file": ("a.wav", b"RIFF" + b"\0" * 2000, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    job_id = upload.json()["job_id"]
    ticket = first.post("/api/ws-ticket").json()["ticket"]
    anonymous = TestClient(second.app)
    with anonymous.websocket_connect(f"/api/jobs/{job_id}/events?ticket={ticket}") as socket:
        assert socket.receive_json()["type"] == "snapshot"
    with anonymous.websocket_connect(f"/api/jobs/{job_id}/events?ticket={ticket}") as socket:
        assert socket.receive_json()["type"] == "error"


def test_el_limite_de_intentos_se_comparte_entre_procesos(shared) -> None:
    first, second = shared
    body = {"email": "nadie@example.com", "password": "incorrecta1"}
    codes = [(first if i % 2 else second).post("/api/auth/login", json=body).status_code for i in range(11)]
    assert codes[:10] == [401] * 10
    assert codes[10] == 429, "el undécimo intento debe bloquearse aunque llegue a otro proceso"


def _queued_job(conn, job_id: str = "job-1") -> None:
    db.run(
        conn,
        "INSERT INTO jobs (id, project_id, user_id, status, progress, stage, error, created_at, updated_at) "
        "VALUES (?, 'p', 'u', 'QUEUED', 0, 'QUEUED', '', 't', 't')",
        (job_id,),
    )


def test_un_trabajo_solo_lo_reclama_un_proceso(shared, monkeypatch: pytest.MonkeyPatch) -> None:
    first, _second = shared
    conn = first.app.state.conn
    _queued_job(conn)
    assert worker.claim(conn, "job-1") is True
    as_other_instance(monkeypatch)
    assert worker.claim(conn, "job-1") is False


def test_si_un_proceso_muere_su_trabajo_vuelve_a_la_cola(shared, monkeypatch: pytest.MonkeyPatch) -> None:
    first, _second = shared
    conn = first.app.state.conn
    _queued_job(conn)
    assert worker.claim(conn, "job-1")
    db.run(conn, "UPDATE jobs SET status='TRANSCRIBING' WHERE id='job-1'")
    assert worker.requeue_stale(conn) == 0, "un proceso vivo conserva su trabajo"
    db.run(conn, "UPDATE jobs SET heartbeat_at=? WHERE id='job-1'", (cluster.now() - cluster.STALE_S - 1,))
    assert worker.requeue_stale(conn) == 1
    row = db.one(conn, "SELECT status, claimed_by FROM jobs WHERE id='job-1'")
    assert row["status"] == "QUEUED" and row["claimed_by"] == ""
    as_other_instance(monkeypatch)
    assert worker.claim(conn, "job-1") is True


def test_cancelar_desde_otro_proceso_detiene_el_trabajo(shared) -> None:
    import threading

    first, _second = shared
    conn = first.app.state.conn
    _queued_job(conn)
    assert worker.claim(conn, "job-1")
    stop, cancel = threading.Event(), threading.Event()
    beat = threading.Thread(target=worker._heartbeat, args=(conn, "job-1", stop, cancel), daemon=True)
    original = cluster.HEARTBEAT_S
    cluster.HEARTBEAT_S = 0.05
    try:
        beat.start()
        db.run(conn, "UPDATE jobs SET status='CANCELLED' WHERE id='job-1'")
        assert cancel.wait(2), "el latido debe ver la cancelación hecha en otro proceso"
    finally:
        stop.set()
        cluster.HEARTBEAT_S = original


def test_un_solo_dueno_por_proyecto_en_vivo(shared, monkeypatch: pytest.MonkeyPatch) -> None:
    first, _second = shared
    conn = first.app.state.conn
    assert cluster.acquire_lease(conn, "p1") is True
    assert cluster.acquire_lease(conn, "p1") is True, "renovar la propia concesión"
    as_other_instance(monkeypatch)
    assert cluster.acquire_lease(conn, "p1") is False
    db.run(conn, "UPDATE live_leases SET heartbeat_at=? WHERE project_id='p1'", (cluster.now() - cluster.STALE_S - 1,))
    assert cluster.acquire_lease(conn, "p1") is True, "si el dueño deja de latir, otro lo toma"
    monkeypatch.setattr(cluster, "INSTANCE_ID", "primer-proceso-ya-sin-concesion")
    assert cluster.renew_lease(conn, "p1") is None


def test_la_posicion_del_espectador_llega_al_dueno(shared) -> None:
    first, _second = shared
    conn = first.app.state.conn
    assert cluster.acquire_lease(conn, "p1")
    cluster.publish_position(conn, "p1", 65_000, "es")
    position, language, at = cluster.renew_lease(conn, "p1")
    assert (position, language) == (65_000, "es") and at > 0


def test_el_estado_del_render_sobrevive_a_un_reinicio(shared) -> None:
    first, second = shared
    conn = first.app.state.conn
    db.run(
        conn,
        "INSERT INTO renders (project_id, language, state, progress, stamp, owner, heartbeat_at) "
        "VALUES ('p1', 'es', 'running', 40, 'v1', 'x', ?)",
        (cluster.now(),),
    )
    out = Path(second.app.state.paths["output"])
    assert render.status(second.app.state.conn, out, "p1", "es", "v1")["state"] == "running"
    assert render.status(second.app.state.conn, out, "p1", "es", "v1")["progress"] == 40
    db.run(conn, "UPDATE renders SET heartbeat_at=? WHERE project_id='p1'", (cluster.now() - cluster.STALE_S - 1,))
    dead = render.status(second.app.state.conn, out, "p1", "es", "v1")
    assert dead["state"] == "failed" and "detuvo" in dead["error"]


def _project_with_track(client: TestClient) -> str:
    sign_up(client, json=USER)
    user_id = client.get("/api/auth/me").json()["id"]
    project_id = str(uuid.uuid4())
    conn = client.app.state.conn
    db.run(
        conn,
        "INSERT INTO projects (id, user_id, title, source_language, target_languages, status, media_path, "
        "created_at, duration_ms) VALUES (?,?,?,?,?,?,?,?,?)",
        (project_id, user_id, "v.mp4", "en", "es", "PROCESSING", "nada.mp4", "t", 60_000),
    )
    for language, text in [("en", "Hello"), ("es", "Hola")]:
        cues = [{"index": 1, "start_ms": 0, "end_ms": 2000, "text": text, "speaker": ""}]
        db.run(
            conn,
            "INSERT INTO subtitles (project_id, language, cues_json, updated_at, covered_ms, covered_json) "
            "VALUES (?,?,?,?,?,?)",
            (project_id, language, json.dumps(cues), "t", 15_000, json.dumps([[0, 15_000], [30_000, 45_000]])),
        )
    db.run(
        conn,
        "INSERT INTO live_leases (project_id, owner, heartbeat_at) VALUES (?, 'otro-proceso', ?)",
        (project_id, cluster.now()),
    )
    return project_id


def test_un_espectador_en_otro_proceso_recibe_los_subtitulos_desde_la_base(shared) -> None:
    first, _second = shared
    project_id = _project_with_track(first)
    conn = first.app.state.conn
    with first.websocket_connect(f"/api/videos/{project_id}/live") as socket:
        snapshot = socket.receive_json()
        assert snapshot["type"] == "snapshot"
        assert snapshot["tracks"]["es"][0]["text"] == "Hola"
        assert snapshot["covered"]["es"] == [[0, 15_000], [30_000, 45_000]], "cobertura exacta, con huecos"
        assert snapshot["complete"] is False

        socket.send_json({"type": "position", "time_ms": 31_000, "language": "es"})
        cues = [
            {"index": 1, "start_ms": 0, "end_ms": 2000, "text": "Hola", "speaker": ""},
            {"index": 2, "start_ms": 16_000, "end_ms": 18_000, "text": "Más", "speaker": ""},
        ]
        db.run(
            conn,
            "UPDATE subtitles SET cues_json=?, covered_json=? WHERE project_id=? AND language='es'",
            (json.dumps(cues), json.dumps([[0, 45_000]]), project_id),
        )
        update = socket.receive_json()
        assert update["type"] == "snapshot" and update["seq"] > snapshot["seq"]
        assert [c["text"] for c in update["tracks"]["es"]] == ["Hola", "Más"]
        position = db.one(conn, "SELECT position_ms, position_lang FROM live_leases WHERE project_id=?", (project_id,))
        assert (position["position_ms"], position["position_lang"]) == (31_000, "es")

        db.run(conn, "DELETE FROM live_leases WHERE project_id=?", (project_id,))
        with pytest.raises(WebSocketDisconnect) as closed:
            for _ in range(5):
                socket.receive_json()
        assert closed.value.code == 1012, "sin dueño, se pide reconectar para tomar el relevo"


def test_borrar_un_proyecto_con_el_video_abierto_no_falla(shared, tmp_path: Path) -> None:
    import time

    first, _second = shared
    sign_up(first, json=USER)
    user_id = first.get("/api/auth/me").json()["id"]
    media = tmp_path / "abierto.mp4"
    media.write_bytes(b"\0" * 1000)
    project_id = str(uuid.uuid4())
    db.run(
        first.app.state.conn,
        "INSERT INTO projects (id, user_id, title, source_language, target_languages, status, media_path, "
        "created_at, duration_ms) VALUES (?,?,?,?,?,?,?,?,?)",
        (project_id, user_id, "abierto.mp4", "en", "es", "PROCESSING", str(media), "t", 1000),
    )
    handle = media.open("rb")
    try:
        assert first.delete(f"/api/videos/{project_id}").status_code == 200
        assert first.get(f"/api/videos/{project_id}").status_code == 404
    finally:
        handle.close()
    for _ in range(40):
        if not media.exists():
            break
        time.sleep(0.1)
    assert not media.exists(), "el archivo se borra en cuanto el lector lo suelta"
