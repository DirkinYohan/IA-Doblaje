"""Medios: local en desarrollo, Supabase Storage en producción.

El resto de la aplicación sigue hablando de una ruta (``projects.media_path``).
Este módulo decide qué significa esa ruta:

* ``C:\\...\\data\\input\\...\\video.mp4`` → archivo local (desarrollo).
* ``storage://media/<usuario>/<proyecto>/video.mp4`` → objeto en Storage.

Así la lógica de negocio no cambia: pide "materializa este medio", y aquí se
resuelve descargándolo a un temporal si hace falta.

**Los temporales son efímeros**: se crean para que FFmpeg/Whisper puedan leer
el archivo y se borran en cuanto termina el trabajo.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from app.infrastructure.storage.supabase_storage import SupabaseStorage, StorageError

SCHEME = "storage://"


# ---------------------------------------------------------------------------
# Identificación del medio
# ---------------------------------------------------------------------------
def is_remote(media_path: Any) -> bool:
    """¿La ruta apunta a Storage en lugar de a disco?"""
    return str(media_path or "").startswith(SCHEME)


def make_uri(bucket: str, key: str) -> str:
    return f"{SCHEME}{bucket}/{key.lstrip('/')}"


def object_key(media_path: Any) -> str:
    """Clave del objeto dentro del bucket, sin el esquema ni el bucket."""
    raw = str(media_path or "")
    if not raw.startswith(SCHEME):
        return ""
    rest = raw[len(SCHEME):]
    # storage://<bucket>/<clave>
    if "/" not in rest:
        return ""
    _bucket, key = rest.split("/", 1)
    return key


def configured() -> bool:
    """¿Se guardan los medios en Storage?

    `IA_STORAGE_BACKEND=local` lo desactiva aunque haya credenciales: es lo que
    usan las pruebas para no escribir objetos reales en el bucket.
    """
    from app.core.env import env

    if env("IA_STORAGE_BACKEND").lower() == "local":
        return False
    return SupabaseStorage.configured()


def client() -> SupabaseStorage | None:
    return SupabaseStorage.from_env()


# ---------------------------------------------------------------------------
# Subida
# ---------------------------------------------------------------------------
def save_stream_to_temp(chunks: Any, target: Path, *, max_bytes: int) -> int:
    """Escribe los chunks recibidos en un archivo temporal.

    Nunca carga el archivo entero en memoria: se copia trozo a trozo.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("wb") as sink:
        for chunk in chunks:
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                sink.close()
                target.unlink(missing_ok=True)
                raise ValueError(f"supera el límite de {max_bytes} bytes")
            sink.write(chunk)
    return written


def upload_file(local_path: Path, key: str, content_type: str) -> str:
    """Sube un archivo local al bucket configurado."""
    store = client()
    if store is None:
        raise StorageError("Storage no está configurado")
    return store.upload_file(key, local_path, content_type)


def upload_bytes(data: bytes, key: str, content_type: str) -> str:
    store = client()
    if store is None:
        raise StorageError("Storage no está configurado")
    return store.upload_bytes(key, data, content_type)


def delete(key: str) -> None:
    store = client()
    if store is not None:
        store.delete(key)


# ---------------------------------------------------------------------------
# Lectura y temporales
# ---------------------------------------------------------------------------
def materialize(
    media_path: Any,
    temp_dir: Path,
    *,
    suffix: str = "",
    label: str = "media",
) -> Path:
    """Devuelve una ruta local utilizable por FFmpeg/Whisper.

    * Si el medio es local, devuelve la ruta tal cual (no copia nada).
    * Si está en Storage, lo descarga a ``temp_dir/materialized``.

    El llamador debe liberar el temporal con :func:`release`.
    """
    if not is_remote(media_path):
        path = Path(str(media_path))
        if not path.is_file():
            raise FileNotFoundError(f"el medio no existe: {path}")
        return path

    store = client()
    if store is None:
        raise StorageError(
            f"el proyecto apunta a Storage pero falta configuración: {media_path}"
        )
    key = object_key(media_path)
    if not key:
        raise StorageError(f"ruta de Storage inválida: {media_path}")

    name = f"{label}-{uuid.uuid4().hex}{suffix or Path(key).suffix}"
    target = temp_dir / "materialized" / name
    store.download_to(key, target)
    return target


def release(path: Path | None, temp_dir: Path) -> None:
    """Borra un temporal descargado. Nunca toca archivos fuera de temp_dir."""
    if path is None:
        return
    try:
        target = Path(path).resolve()
        root = (Path(temp_dir) / "materialized").resolve()
        if target.is_file() and str(target).startswith(str(root)):
            target.unlink(missing_ok=True)
    except OSError:
        pass


def purge_temp(temp_dir: Path, *, attempts: int = 1, interval: float = 0.0) -> bool:
    """Vacía los temporales de medios de una pasada.

    En Windows, un proceso FFmpeg recién terminado puede seguir reteniendo el
    archivo durante unos segundos, así que el borrado puede fallar justo al
    acabar el trabajo. Con ``attempts`` > 1 se reintenta hasta liberarlo.

    Devuelve True si al final no queda nada.
    """
    import time

    folder = Path(temp_dir) / "materialized"
    if not folder.is_dir():
        return True

    total = max(1, int(attempts))
    for attempt in range(total):
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            return True
        # Borrado archivo a archivo por si un directorio se resiste.
        for path in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            try:
                path.unlink(missing_ok=True) if path.is_file() else path.rmdir()
            except OSError:
                pass
        if not folder.exists():
            return True
        if attempt < total - 1 and interval > 0:
            time.sleep(interval)
    return not folder.exists()


# ---------------------------------------------------------------------------
# Servir al cliente
# ---------------------------------------------------------------------------
def signed_url(media_path: Any, *, expires_in: int = 3600) -> str | None:
    """URL temporal para un objeto privado, o None si es local."""
    if not is_remote(media_path):
        return None
    store = client()
    if store is None:
        return None
    key = object_key(media_path)
    if not key:
        return None
    try:
        return store.signed_url(key, expires_in=expires_in)
    except StorageError:
        return None
