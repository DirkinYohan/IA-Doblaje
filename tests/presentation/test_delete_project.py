from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.presentation.api import db
from app.presentation.api import media_store
from tests.fixtures.accounts import sign_up

PAYLOAD = b"RIFF" + b"\1" * 4000
TABLES = ("projects", "jobs", "subtitles", "renders", "live_leases")


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state: dict[str, object] = {"files": {}, "deleted": [], "fail": False}

    def upload_file(path: Path, key: str, _type: str) -> str:
        state["files"][key] = Path(path).read_bytes()  # type: ignore[index]
        return key

    def upload_bytes(data: bytes, key: str, _type: str) -> str:
        state["files"][key] = data  # type: ignore[index]
        return key

    def delete(key: str) -> None:
        if state["fail"]:
            raise RuntimeError("Storage caído")
        state["deleted"].append(key)  # type: ignore[union-attr]
        state["files"].pop(key, None)  # type: ignore[union-attr]

    monkeypatch.setattr(media_store, "configured", lambda: True)
    monkeypatch.setattr(media_store, "upload_file", upload_file)
    monkeypatch.setattr(media_store, "upload_bytes", upload_bytes)
    monkeypatch.setattr(media_store, "delete", delete)
    monkeypatch.setattr(media_store, "signed_url", lambda *_a, **_k: "https://storage.example/firmada")
    return state


@pytest.fixture
def client(tmp_path: Path, storage: dict[str, object]):
    from app.presentation.api import worker

    worker.reset()
    os.environ["IA_DB_PATH"] = str(tmp_path / "app.sqlite")
    os.environ["IA_MAIL_DIR"] = str(tmp_path / "mail")
    os.environ["IA_DISABLE_WORKER"] = "1"
    from app.presentation.api.main import create_app

    with TestClient(create_app()) as test_client:
        response = sign_up(
            test_client,
            json={"name": "Borra", "email": "borra@example.com", "password": "password123", "confirm": "password123"},
        )
        assert response.status_code == 200, response.text
        yield test_client
    os.environ.pop("IA_DISABLE_WORKER", None)
    worker.reset()


def _upload(client: TestClient) -> str:
    upload = client.post(
        "/api/videos",
        files={"file": ("borrar.wav", PAYLOAD, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    assert upload.status_code == 200, upload.text
    project_id = upload.json()["id"]
    for thread in threading.enumerate():
        if thread.name == f"store-{project_id[:8]}":
            thread.join(10)
    return project_id


def _counts(project_id: str) -> dict[str, int]:
    conn = db.connect(Path(os.environ["IA_DB_PATH"]))
    try:
        counts = {}
        for table in TABLES:
            column = "id" if table == "projects" else "project_id"
            row = db.one(conn, f"SELECT COUNT(*) AS n FROM {table} WHERE {column}=?", (project_id,))
            counts[table] = int(row["n"])
        return counts
    finally:
        conn.close()


def test_borrar_quita_todo_de_la_base_y_de_storage(client: TestClient, storage: dict[str, object]) -> None:
    from app.core.config import get_settings

    project_id = _upload(client)
    conn = db.connect(Path(os.environ["IA_DB_PATH"]))
    db.run(conn, "INSERT INTO subtitles (project_id, language, cues_json, updated_at, covered_ms, covered_json) "
                 "VALUES (?, 'es', '[]', '', 0, '[]')", (project_id,))
    db.run(conn, "INSERT INTO renders (project_id, language, state) VALUES (?, 'es', 'done')", (project_id,))
    db.run(conn, "INSERT INTO live_leases (project_id, owner, heartbeat_at) VALUES (?, 'x', 0)", (project_id,))
    conn.close()
    assert all(value >= 1 for value in _counts(project_id).values())
    stored_keys = sorted(storage["files"])  # type: ignore[arg-type]
    assert any(key.endswith("borrar.wav") for key in stored_keys)

    temp = Path(str(get_settings().paths.data_temp_dir))
    story = temp / "storyboards" / f"{project_id}-14.jpg"
    story.parent.mkdir(parents=True, exist_ok=True)
    story.write_bytes(b"jpg")

    deleted = client.delete(f"/api/videos/{project_id}")
    assert deleted.status_code == 200
    assert _counts(project_id) == dict.fromkeys(TABLES, 0)
    assert sorted(storage["deleted"]) == stored_keys  # type: ignore[arg-type]
    assert storage["files"] == {}
    assert media_store.find_pending(temp, project_id) is None
    assert not story.exists()
    assert client.get(f"/api/videos/{project_id}").status_code == 404


def test_si_storage_falla_se_borra_de_la_base_y_se_avisa(
    client: TestClient, storage: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    project_id = _upload(client)
    storage["fail"] = True
    with caplog.at_level("WARNING", logger="ia.projects"):
        assert client.delete(f"/api/videos/{project_id}").status_code == 200
    assert _counts(project_id)["projects"] == 0
    assert "no se pudo borrar" in caplog.text


def test_un_proyecto_ajeno_no_se_puede_borrar(client: TestClient, storage: dict[str, object]) -> None:
    project_id = _upload(client)
    other = TestClient(client.app)
    response = sign_up(
        other,
        json={"name": "Otra", "email": "otra@example.com", "password": "password123", "confirm": "password123"},
    )
    assert response.status_code == 200, response.text
    assert other.delete(f"/api/videos/{project_id}").status_code == 404
    assert _counts(project_id)["projects"] == 1


def test_detenido_no_vuelve_a_escribir_subtitulos() -> None:
    from app.presentation.api import live

    writes: list[str] = []
    callbacks = live.LiveCallbacks(
        read=lambda *a: None, transcribe=lambda *a: [], translate=lambda *a: [],
        detect=lambda *a: "en", emit=lambda e: writes.append("emit"), persist=lambda *a, **k: writes.append("persist"),
    )
    sub = live.LiveSubtitler(callbacks=callbacks, source_language="en", duration_ms=10_000, targets=["es"])
    sub.stop()
    sub._publish("en", [{"start_ms": 0, "end_ms": 1000, "text": "hola"}], 0, 1000)
    assert writes == []


def test_supabase_avisa_si_no_puede_borrar(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from app.infrastructure.storage.supabase_storage import StorageError, SupabaseStorage

    statuses = iter([200, 404, 400])

    class _Client:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *a: object) -> None:
            pass

        def delete(self, url: str, headers: dict[str, str]) -> httpx.Response:
            return httpx.Response(next(statuses), text="detalle")

    monkeypatch.setattr(httpx, "Client", _Client)
    store = SupabaseStorage(url="https://x.supabase.co", service_key="k")
    store.delete("a/b.mp4")
    store.delete("a/b.mp4")
    with pytest.raises(StorageError, match="400"):
        store.delete("a/b.mp4")
