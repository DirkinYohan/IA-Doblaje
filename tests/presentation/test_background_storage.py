from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.presentation.api import media_store
from app.presentation.api import media_upload
from tests.fixtures.accounts import sign_up

PAYLOAD = b"RIFF" + b"\1" * 4000


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Storage falso: la subida se bloquea hasta que la prueba la libere."""
    state: dict[str, object] = {"files": {}, "release": threading.Event(), "started": threading.Event()}

    def upload_file(path: Path, key: str, _type: str) -> str:
        state["started"].set()  # type: ignore[union-attr]
        state["release"].wait(10)  # type: ignore[union-attr]
        state["files"][key] = Path(path).read_bytes()  # type: ignore[index]
        return key

    def upload_bytes(data: bytes, key: str, _type: str) -> str:
        state["files"][key] = data  # type: ignore[index]
        return key

    monkeypatch.setattr(media_store, "configured", lambda: True)
    monkeypatch.setattr(media_store, "upload_file", upload_file)
    monkeypatch.setattr(media_store, "upload_bytes", upload_bytes)
    monkeypatch.setattr(media_store, "delete", lambda key: state["files"].pop(key, None))  # type: ignore[union-attr]
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
            json={"name": "Fondo", "email": "fondo@example.com", "password": "password123", "confirm": "password123"},
        )
        assert response.status_code == 200, response.text
        yield test_client
    os.environ.pop("IA_DISABLE_WORKER", None)
    worker.reset()


def test_la_subida_responde_sin_esperar_al_almacenamiento(client: TestClient, storage: dict[str, object]) -> None:
    upload = client.post(
        "/api/videos",
        files={"file": ("a.wav", PAYLOAD, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    assert upload.status_code == 200, upload.text
    project_id = upload.json()["id"]
    assert storage["started"].wait(5)  # type: ignore[union-attr]
    assert storage["files"] == {}

    local = client.get(f"/api/videos/{project_id}/media", follow_redirects=False)
    assert local.status_code == 200
    assert local.content == PAYLOAD

    partial = client.get(f"/api/videos/{project_id}/media", headers={"Range": "bytes=0-3"}, follow_redirects=False)
    assert partial.status_code == 206
    assert partial.content == b"RIFF"

    storage["release"].set()  # type: ignore[union-attr]
    for thread in threading.enumerate():
        if thread.name == f"store-{project_id[:8]}":
            thread.join(10)
    assert any(key.endswith(f"{project_id}/a.wav") for key in storage["files"])  # type: ignore[union-attr]
    assert media_store.staged(project_id) is None


def test_video_demasiado_largo_se_rechaza_al_momento(
    client: TestClient, storage: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.presentation.api import storage_fit

    def too_long(*_a: object, **_k: object) -> bool:
        raise storage_fit.TooLongForStorage("El vídeo es demasiado largo: admite unos 17 minutos como máximo.")

    monkeypatch.setattr(storage_fit, "needs_fit", too_long)
    upload = client.post(
        "/api/videos",
        files={"file": ("largo.wav", PAYLOAD, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    assert upload.status_code == 413
    assert "17 minutos" in upload.json()["detail"]
    assert not storage["started"].is_set()  # type: ignore[union-attr]


def test_reintenta_y_borra_si_el_proyecto_ya_no_existe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    thumb = tmp_path / "t.jpg"
    thumb.write_bytes(b"jpg")
    calls: list[str] = []
    deleted: list[str] = []

    def flaky(path: Path, key: str, _type: str) -> str:
        calls.append(key)
        if len(calls) == 1:
            raise RuntimeError("red caída")
        return key

    monkeypatch.setattr(media_store, "upload_file", flaky)
    monkeypatch.setattr(media_store, "upload_bytes", lambda data, key, _t: key)
    monkeypatch.setattr(media_store, "delete", deleted.append)
    media_store.stage("p1", source, thumb)
    ok = media_upload.store(
        project_id="p1", source=source, duration_ms=1000, media_key="u/p1/v.mp4", content_type="video/mp4",
        thumbnail=thumb, thumbnail_key="u/p1/thumbnail.jpg", still_exists=lambda: False,
    )
    assert ok is True
    assert calls == ["u/p1/v.mp4", "u/p1/v.mp4"]
    assert deleted == ["u/p1/v.mp4", "u/p1/thumbnail.jpg"]
    assert not source.exists() and not thumb.exists()
    assert media_store.staged("p1") is None
