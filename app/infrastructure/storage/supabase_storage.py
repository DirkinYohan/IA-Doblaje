from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.core.env import env, load_project_env

DEFAULT_BUCKET = "media"


class StorageError(RuntimeError):
    pass


class SupabaseStorage:
    def __init__(
        self,
        *,
        url: str,
        service_key: str,
        bucket: str = DEFAULT_BUCKET,
        timeout: float = 300.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.bucket = bucket
        self._key = service_key
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> "SupabaseStorage | None":
        url = env("SUPABASE_URL")
        key = env("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            return None
        bucket = env("SUPABASE_BUCKET", DEFAULT_BUCKET) or DEFAULT_BUCKET
        return cls(url=url, service_key=key, bucket=bucket)

    @staticmethod
    def configured() -> bool:
        load_project_env()
        return bool(env("SUPABASE_URL") and env("SUPABASE_SERVICE_ROLE_KEY"))

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "apikey": self._key,
        }
        if extra:
            headers.update(extra)
        return headers

    def _object_url(self, key: str) -> str:
        return f"{self.url}/storage/v1/object/{self.bucket}/{key.lstrip('/')}"

    def ensure_bucket(self, *, public: bool = False) -> None:
        import httpx

        payload = {
            "id": self.bucket,
            "name": self.bucket,
            "public": public,
            "file_size_limit": None,
            "allowed_mime_types": None,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self.url}/storage/v1/bucket",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json=payload,
                )
            if response.status_code in {400, 409}:
                return
            if response.status_code >= 400:
                raise StorageError(
                    f"no se pudo crear el bucket {self.bucket!r}: "
                    f"{response.status_code} {response.text[:200]}"
                )
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red creando el bucket: {exc}") from exc

    def upload_file(self, key: str, source: Path, content_type: str) -> str:
        import httpx

        if not source.is_file():
            raise StorageError(f"no existe el archivo a subir: {source}")
        try:
            with source.open("rb") as handle:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        self._object_url(key),
                        headers=self._headers(
                            {
                                "Content-Type": content_type,
                                "x-upsert": "true",
                                "cache-control": "max-age=3600",
                            }
                        ),
                        content=handle,
                    )
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red subiendo {key}: {exc}") from exc
        if response.status_code >= 400:
            raise StorageError(
                f"no se pudo subir {key}: {response.status_code} {response.text[:200]}"
            )
        return key

    def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        import httpx

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._object_url(key),
                    headers=self._headers({"Content-Type": content_type, "x-upsert": "true"}),
                    content=data,
                )
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red subiendo {key}: {exc}") from exc
        if response.status_code >= 400:
            raise StorageError(
                f"no se pudo subir {key}: {response.status_code} {response.text[:200]}"
            )
        return key

    def download_to(self, key: str, target: Path) -> Path:
        import httpx

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                with client.stream(
                    "GET", self._object_url(key), headers=self._headers()
                ) as response:
                    if response.status_code >= 400:
                        raise StorageError(
                            f"no se pudo descargar {key}: {response.status_code}"
                        )
                    with target.open("wb") as sink:
                        for chunk in response.iter_bytes(1024 * 1024):
                            sink.write(chunk)
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red descargando {key}: {exc}") from exc
        return target

    def read_bytes(self, key: str) -> bytes:
        import httpx

        try:
            with httpx.Client(timeout=self._timeout, follow_redirects=True) as client:
                response = client.get(self._object_url(key), headers=self._headers())
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red leyendo {key}: {exc}") from exc
        if response.status_code >= 400:
            raise StorageError(f"no se pudo leer {key}: {response.status_code}")
        return response.content

    def delete(self, key: str) -> None:
        import httpx

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.delete(self._object_url(key), headers=self._headers())
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red borrando {key}: {exc}") from exc
        # 404: ya no estaba, que es lo que se quería.
        if response.status_code >= 400 and response.status_code != 404:
            raise StorageError(f"no se pudo borrar {key}: {response.status_code} {response.text[:200]}")

    def exists(self, key: str) -> bool:
        import httpx

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.head(self._object_url(key), headers=self._headers())
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    def signed_url(self, key: str, *, expires_in: int = 3600) -> str:
        import httpx

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.url}/storage/v1/object/sign/{self.bucket}/{key.lstrip('/')}",
                    headers=self._headers({"Content-Type": "application/json"}),
                    json={"expiresIn": expires_in},
                )
        except httpx.HTTPError as exc:  # pragma: no cover - red
            raise StorageError(f"error de red firmando {key}: {exc}") from exc
        if response.status_code >= 400:
            raise StorageError(f"no se pudo firmar {key}: {response.status_code}")
        data: dict[str, Any] = response.json()
        signed = str(data.get("signedURL") or data.get("signedUrl") or "")
        if not signed:
            raise StorageError(f"respuesta sin URL firmada para {key}: {data}")
        if signed.startswith("http"):
            return signed
        return f"{self.url}/storage/v1{signed}"
