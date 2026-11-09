"""Almacenamiento de vídeos en Azure Blob Storage.

Misma interfaz que SupabaseStorage. A diferencia del plan gratuito de Supabase,
no hay límite de 50 MB por archivo, así que los vídeos se guardan sin comprimir.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from app.core.env import env
from app.core.env import load_project_env
from app.infrastructure.storage.supabase_storage import StorageError

DEFAULT_CONTAINER = "media"


class AzureBlobStorage:
    def __init__(self, *, connection_string: str, container: str = DEFAULT_CONTAINER) -> None:
        from azure.storage.blob import BlobServiceClient

        self.container = container
        self._service = BlobServiceClient.from_connection_string(connection_string)
        self._client = self._service.get_container_client(container)
        parts = dict(
            item.split("=", 1) for item in connection_string.split(";") if "=" in item
        )
        self._account = parts.get("AccountName", "")
        self._key = parts.get("AccountKey", "")

    @classmethod
    def from_env(cls) -> AzureBlobStorage | None:
        connection = env("AZURE_STORAGE_CONNECTION_STRING")
        if not connection:
            return None
        return cls(
            connection_string=connection,
            container=env("AZURE_STORAGE_CONTAINER", DEFAULT_CONTAINER) or DEFAULT_CONTAINER,
        )

    @staticmethod
    def configured() -> bool:
        load_project_env()
        return bool(env("AZURE_STORAGE_CONNECTION_STRING"))

    def _blob(self, key: str) -> Any:
        return self._client.get_blob_client(key.lstrip("/"))

    def ensure_bucket(self, *, public: bool = False) -> None:
        from azure.core.exceptions import ResourceExistsError

        try:
            self._client.create_container()
        except ResourceExistsError:
            return
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"no se pudo crear el contenedor {self.container!r}: {exc}") from exc

    def upload_file(self, key: str, source: Path, content_type: str) -> str:
        from azure.storage.blob import ContentSettings

        if not source.is_file():
            raise StorageError(f"no existe el archivo a subir: {source}")
        try:
            with source.open("rb") as handle:
                self._blob(key).upload_blob(
                    handle,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type, cache_control="max-age=3600"),
                    max_concurrency=2,
                )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"no se pudo subir {key}: {exc}") from exc
        return key

    def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        from azure.storage.blob import ContentSettings

        try:
            self._blob(key).upload_blob(
                data, overwrite=True, content_settings=ContentSettings(content_type=content_type)
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"no se pudo subir {key}: {exc}") from exc
        return key

    def download_to(self, key: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as sink:
                self._blob(key).download_blob(max_concurrency=2).readinto(sink)
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            raise StorageError(f"no se pudo descargar {key}: {exc}") from exc
        return target

    def read_bytes(self, key: str) -> bytes:
        try:
            return self._blob(key).download_blob().readall()
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"no se pudo leer {key}: {exc}") from exc

    def delete(self, key: str) -> None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            self._blob(key).delete_blob()
        except ResourceNotFoundError:
            return  # ya no estaba, que es lo que se quería
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"no se pudo borrar {key}: {exc}") from exc

    def exists(self, key: str) -> bool:
        try:
            return bool(self._blob(key).exists())
        except Exception:  # noqa: BLE001
            return False

    def signed_url(self, key: str, *, expires_in: int = 3600, download: str | None = None) -> str:
        """URL temporal de solo lectura (SAS); admite peticiones por rangos.

        `download`: nombre de archivo; Azure responde como adjunto y el navegador lo
        descarga en vez de abrirlo (el atributo download no vale entre dominios).
        """
        from azure.storage.blob import BlobSasPermissions
        from azure.storage.blob import generate_blob_sas

        if not self._account or not self._key:
            raise StorageError("la cadena de conexión de Azure no tiene AccountName/AccountKey")
        now = datetime.now(UTC)
        token = generate_blob_sas(
            account_name=self._account,
            container_name=self.container,
            blob_name=key.lstrip("/"),
            account_key=self._key,
            permission=BlobSasPermissions(read=True),
            start=now - timedelta(minutes=5),
            expiry=now + timedelta(seconds=int(expires_in)),
            content_disposition=_attachment(download) if download else None,
        )
        return f"{self._blob(key).url}?{token}"


def _attachment(name: str) -> str:
    from app.presentation.api.media_store import attachment_header

    return attachment_header(name)


__all__ = ["DEFAULT_CONTAINER", "AzureBlobStorage"]
