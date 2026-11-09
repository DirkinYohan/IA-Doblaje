from __future__ import annotations

import os
import uuid
from pathlib import Path

import httpx
import pytest

from app.infrastructure.storage.azure_blob_storage import AzureBlobStorage
from app.infrastructure.storage.supabase_storage import StorageError

# Cadena de conexión a Azurite (emulador oficial) o a una cuenta de pruebas.
AZURE_URL = os.environ.get("IA_TEST_AZURE_STORAGE", "")
azure = pytest.mark.skipif(not AZURE_URL, reason="IA_TEST_AZURE_STORAGE no definido")


@pytest.fixture
def store() -> AzureBlobStorage:
    blob = AzureBlobStorage(connection_string=AZURE_URL, container=f"prueba{uuid.uuid4().hex[:10]}")
    blob.ensure_bucket()
    blob.ensure_bucket()  # repetir no falla
    yield blob
    blob._client.delete_container()


@azure
def test_subir_leer_descargar_y_borrar(store: AzureBlobStorage, tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"RIFF" + bytes(range(256)) * 400)
    assert store.upload_file("u1/p1/v.mp4", video, "video/mp4") == "u1/p1/v.mp4"
    store.upload_bytes("u1/p1/thumbnail.jpg", b"jpg", "image/jpeg")
    assert store.exists("u1/p1/v.mp4") and store.exists("u1/p1/thumbnail.jpg")
    assert store.read_bytes("u1/p1/thumbnail.jpg") == b"jpg"
    copy = store.download_to("u1/p1/v.mp4", tmp_path / "copia.mp4")
    assert copy.read_bytes() == video.read_bytes()

    store.delete("u1/p1/v.mp4")
    store.delete("u1/p1/v.mp4")  # ya no estaba: no es un error
    assert not store.exists("u1/p1/v.mp4")
    with pytest.raises(StorageError):
        store.read_bytes("u1/p1/v.mp4")


@azure
def test_enlace_firmado_sirve_rangos_y_caduca(store: AzureBlobStorage, tmp_path: Path) -> None:
    data = b"RIFF" + b"x" * 5000
    store.upload_bytes("u1/p1/v.mp4", data, "video/mp4")
    url = store.signed_url("u1/p1/v.mp4", expires_in=600)
    partial = httpx.get(url, headers={"Range": "bytes=0-3"})
    assert partial.status_code == 206 and partial.content == b"RIFF"
    assert partial.headers["content-type"] == "video/mp4"
    unsigned = httpx.get(url.split("?", 1)[0])
    assert unsigned.status_code in {401, 403, 404}  # contenedor privado


def test_elige_azure_si_hay_cadena_de_conexion(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.presentation.api import media_store

    monkeypatch.delenv("IA_STORAGE_BACKEND", raising=False)
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "AccountName=a;AccountKey=a2V5;EndpointSuffix=core.windows.net")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "videos")
    assert media_store.backend() == "azure"
    assert media_store.bucket() == "videos"
    monkeypatch.setenv("IA_STORAGE_BACKEND", "local")
    assert media_store.backend() == "local"
    assert media_store.client() is None


def test_sin_limite_de_tamano_en_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.presentation.api import storage_fit

    monkeypatch.delenv("IA_STORAGE_MAX_MB", raising=False)
    monkeypatch.setenv("IA_STORAGE_BACKEND", "azure")
    assert storage_fit.limit_bytes() == 0
    assert storage_fit.needs_fit(5 * 1024**3, 3 * 3600 * 1000) is False  # 5 GB y 3 h: se guarda tal cual
    monkeypatch.setenv("IA_STORAGE_BACKEND", "supabase")
    assert storage_fit.limit_bytes() == 50 * 1024 * 1024
