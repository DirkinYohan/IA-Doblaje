from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.presentation.api import db
from tests.fixtures.accounts import sign_up

PG_URL = os.environ.get("IA_TEST_POSTGRES_URL", "")


def test_transaccion_sqlite_deshace_todo_si_falla(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "t.sqlite")
    db.migrate(conn)
    db.run(conn, "INSERT INTO users (id, name, email, password_hash, created_at) VALUES ('u1','A','a@b.co','h','t')")
    with pytest.raises(RuntimeError), db.transaction(conn) as tx:
        tx.run("UPDATE users SET name='B' WHERE id='u1'")
        tx.run("DELETE FROM users WHERE id='u1'")
        raise RuntimeError("fallo a mitad")
    assert db.one(conn, "SELECT name FROM users WHERE id='u1'")["name"] == "A"
    with db.transaction(conn) as tx:
        tx.run("UPDATE users SET name='C' WHERE id='u1'")
        assert tx.one("SELECT name FROM users WHERE id='u1'")["name"] == "C"
    assert db.one(conn, "SELECT name FROM users WHERE id='u1'")["name"] == "C"
    conn.close()


pg =pytest.mark.skipif(not PG_URL, reason="IA_TEST_POSTGRES_URL no definido")


@pytest.fixture()
def pg_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    schema = f"t_{uuid.uuid4().hex[:10]}"
    admin = __import__("psycopg").connect(PG_URL, autocommit=True)
    admin.execute(f"CREATE SCHEMA {schema}")
    sep = "&" if "?" in PG_URL else "?"
    monkeypatch.setenv("IA_DATABASE_URL", f"{PG_URL}{sep}options=-csearch_path%3D{schema}")
    monkeypatch.setenv("IA_DB_BACKEND", "postgres")
    monkeypatch.setenv("IA_DB_POOL_SIZE", "4")
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    from app.presentation.api.main import create_app

    with TestClient(create_app()) as client:
        yield client, admin, schema
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()


@pg
def test_postgres_flujo_completo_con_pool(pg_client) -> None:
    client, _admin, _schema = pg_client
    assert isinstance(client.app.state.conn, db.PooledConnection)
    sign_up(client, json={"name": "Ana", "email": "ana@example.com", "password": "secreto123", "confirm": "secreto123"})
    upload = client.post(
        "/api/videos",
        files={"file": ("a.wav", b"RIFF" + b"\0" * 2000, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    assert upload.status_code == 200, upload.text
    project_id = upload.json()["id"]
    cues = [{"index": 1, "start_ms": 0, "end_ms": 1000, "text": "Hola", "speaker": ""}]
    assert client.put(f"/api/subtitles/{project_id}", json={"language": "es", "cues": cues}).json() == {"count": 1}
    assert "Hola" in client.get(f"/api/export/{project_id}/srt", params={"lang": "es"}).text
    assert client.delete(f"/api/videos/{project_id}").status_code == 200
    assert client.get(f"/api/videos/{project_id}").status_code == 404


@pg
def test_postgres_transaccion_deshace_todo_si_falla(pg_client) -> None:
    client, _admin, _schema = pg_client
    conn = client.app.state.conn
    db.run(conn, "INSERT INTO users (id, name, email, password_hash, created_at) VALUES ('u1','A','a@b.co','h','t')")
    with pytest.raises(RuntimeError), db.transaction(conn) as tx:
        tx.run("UPDATE users SET name='B' WHERE id='u1'")
        raise RuntimeError("fallo a mitad")
    assert db.one(conn, "SELECT name FROM users WHERE id='u1'")["name"] == "A"


@pg
def test_postgres_consultas_en_paralelo(pg_client) -> None:
    client, _admin, _schema = pg_client
    conn = client.app.state.conn
    errors: list[BaseException] = []

    def slow_query() -> None:
        try:
            db.one(conn, "SELECT pg_sleep(0.3)")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=slow_query) for _ in range(4)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    assert not errors
    assert elapsed < 0.9, f"las consultas no corren en paralelo ({elapsed:.2f} s para 4 × 0,3 s)"


@pg
def test_postgres_se_recupera_si_el_servidor_corta_las_conexiones(pg_client) -> None:
    client, admin, schema = pg_client
    conn = client.app.state.conn
    db.one(conn, "SELECT 1")
    admin.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE pid <> pg_backend_pid() AND datname = current_database()"
    )
    assert db.one(conn, "SELECT 1 AS ok")["ok"] == 1
    assert client.get("/api/auth/me").status_code == 401


@pg
def test_postgres_coordinacion_entre_procesos(pg_client, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.presentation.api import cluster, render, worker

    client, _admin, _schema = pg_client
    conn = client.app.state.conn
    cluster.issue_ticket(conn, "h1", "u1", 60)
    assert cluster.consume_ticket(conn, "h1") == "u1"
    assert cluster.consume_ticket(conn, "h1") is None
    limiter = cluster.DbRateLimiter(conn, "prueba", 2, 60)
    assert [limiter.allow("ip") for _ in range(3)] == [True, True, False]
    db.run(
        conn,
        "INSERT INTO jobs (id, project_id, user_id, status, progress, stage, error, created_at, updated_at) "
        "VALUES ('j1', 'p', 'u', 'QUEUED', 0, 'QUEUED', '', 't', 't')",
    )
    assert worker.claim(conn, "j1") is True
    assert cluster.acquire_lease(conn, "p1") is True
    assert cluster.renew_lease(conn, "p1") is not None
    assert render._claim(conn, "p1", "es", "v1") is not None
    monkeypatch.setattr(cluster, "INSTANCE_ID", "otro-proceso")
    assert worker.claim(conn, "j1") is False
    assert cluster.acquire_lease(conn, "p1") is False
    assert render._claim(conn, "p1", "es", "v1") is None
