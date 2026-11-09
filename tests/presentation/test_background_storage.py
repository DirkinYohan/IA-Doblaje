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


def test_espera_al_modo_en_vivo_antes_de_comprimir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.presentation.api import storage_fit

    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    order: list[str] = []
    answers = iter([False, False, True])

    def ready(project_id: str) -> bool:
        value = next(answers)
        order.append(f"listo={value}")
        return value

    monkeypatch.setattr(media_upload, "POLL_S", 0.01)
    monkeypatch.setattr(storage_fit, "needs_fit", lambda *_a, **_k: True)
    monkeypatch.setattr(storage_fit, "fit", lambda src, *_a, **_k: order.append("comprimir") or src)
    monkeypatch.setattr(media_store, "upload_file", lambda *a: order.append("subir"))
    assert media_upload.store(
        project_id="p2", source=source, duration_ms=1000, media_key="u/p2/v.mp4", content_type="video/mp4",
        thumbnail=None, thumbnail_key="u/p2/thumbnail.jpg", still_exists=lambda: True, ready=ready,
    )
    assert order == ["listo=False", "listo=False", "listo=True", "comprimir", "subir"]


def test_sin_compresion_no_espera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(media_store, "upload_file", lambda *a: None)

    def never(_project_id: str) -> bool:
        raise AssertionError("no debía esperar")

    assert media_upload.store(
        project_id="p3", source=source, duration_ms=1000, media_key="u/p3/v.mp4", content_type="video/mp4",
        thumbnail=None, thumbnail_key="u/p3/thumbnail.jpg", still_exists=lambda: True, ready=never,
    )


def test_margen_del_modo_en_vivo() -> None:
    from app.presentation.api import live

    sub = live.LiveSubtitler(
        callbacks=None, source_language="en", duration_ms=300_000, targets=["es"],  # type: ignore[arg-type]
    )
    assert sub.buffered_ms() == 0
    sub._ranges("en").add(0, 120_000)
    sub._ranges("es").add(0, 60_000)
    assert sub.buffered_ms() == 60_000
    sub.failed.add("es")
    assert sub.buffered_ms() == 120_000
    assert live.head_start_ready("proyecto-sin-runner") is True


def test_mientras_se_guarda_redirige_directo_al_backend(
    client: TestClient, storage: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IA_PUBLIC_API_URL", "https://api.example")
    upload = client.post(
        "/api/videos",
        files={"file": ("b.wav", PAYLOAD, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    project_id = upload.json()["id"]
    assert storage["started"].wait(5)  # type: ignore[union-attr]

    media = client.get(f"/api/videos/{project_id}/media", follow_redirects=False)
    assert media.status_code == 307
    link = media.headers["location"]
    assert link.startswith(f"https://api.example/api/videos/{project_id}/media/local?exp=")

    anonymous = TestClient(client.app)
    path = link.removeprefix("https://api.example")
    direct = anonymous.get(path, headers={"Range": "bytes=0-3"}, follow_redirects=False)
    assert direct.status_code == 206 and direct.content == b"RIFF"
    assert anonymous.get(path.replace("sig=", "sig=0"), follow_redirects=False).status_code == 403

    storage["release"].set()  # type: ignore[union-attr]
    for thread in threading.enumerate():
        if thread.name == f"store-{project_id[:8]}":
            thread.join(10)
    stored = client.get(f"/api/videos/{project_id}/media", follow_redirects=False)
    assert stored.status_code == 307
    assert stored.headers["location"] == "https://storage.example/firmada"
    # El enlace directo nunca salta a otro servidor: primero la copia local...
    again = anonymous.get(path, headers={"Range": "bytes=0-3"}, follow_redirects=False)
    assert again.status_code == 206 and again.content == b"RIFF"

    # ...y cuando ya no existe, el backend reenvía los trozos de Storage.
    from app.core.config import get_settings

    media_store.drop_pending(Path(str(get_settings().paths.data_temp_dir)), project_id)
    seen: dict[str, object] = {}

    class _Upstream:
        status_code = 206
        headers = {"content-type": "audio/wav", "content-range": "bytes 0-3/4004", "content-length": "4", "x-otro": "no"}

        def iter_raw(self, _size: int):
            yield b"RIFF"

    def fake_open(url: str, range_header: str | None) -> _Upstream:
        seen.update(url=url, range=range_header)
        return _Upstream()

    monkeypatch.setattr(media_upload, "open_remote", fake_open)
    monkeypatch.setattr(media_upload, "close_remote", lambda r: seen.update(closed=True))
    proxied = anonymous.get(path, headers={"Range": "bytes=0-3"}, follow_redirects=False)
    assert proxied.status_code == 206 and proxied.content == b"RIFF"
    assert proxied.headers["content-range"] == "bytes 0-3/4004"
    assert "x-otro" not in proxied.headers
    assert seen == {"url": "https://storage.example/firmada", "range": "bytes=0-3", "closed": True}


def test_enlace_caducado() -> None:
    import time

    expires = int(time.time()) + 10
    signature = media_upload._signature("p9", expires)
    assert media_upload.link_valid("p9", expires, signature)
    assert not media_upload.link_valid("p9", expires, signature, now=expires + 1)
    assert not media_upload.link_valid("otro", expires, signature)
