from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from app.infrastructure.storage.supabase_storage import SupabaseStorage, StorageError

SCHEME = "storage://"


def is_remote(media_path: Any) -> bool:
    return str(media_path or "").startswith(SCHEME)


def make_uri(bucket: str, key: str) -> str:
    return f"{SCHEME}{bucket}/{key.lstrip('/')}"


def object_key(media_path: Any) -> str:
    raw = str(media_path or "")
    if not raw.startswith(SCHEME):
        return ""
    rest = raw[len(SCHEME):]
    if "/" not in rest:
        return ""
    _bucket, key = rest.split("/", 1)
    return key


def configured() -> bool:
    from app.core.env import env

    if env("IA_STORAGE_BACKEND").lower() == "local":
        return False
    return SupabaseStorage.configured()


def client() -> SupabaseStorage | None:
    return SupabaseStorage.from_env()


def save_stream_to_temp(chunks: Any, target: Path, *, max_bytes: int) -> int:
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


def materialize(
    media_path: Any,
    temp_dir: Path,
    *,
    suffix: str = "",
    label: str = "media",
) -> Path:
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

    parts = key.split("/")
    if len(parts) == 3:
        local = find_pending(temp_dir, parts[1])
        if local is not None:
            return local

    name = f"{label}-{uuid.uuid4().hex}{suffix or Path(key).suffix}"
    target = temp_dir / "materialized" / name
    store.download_to(key, target)
    return target


def release(path: Path | None, temp_dir: Path) -> None:
    if path is None:
        return
    try:
        target = Path(path).resolve()
        root = (Path(temp_dir) / "materialized").resolve()
        if target.is_file() and str(target).startswith(str(root)):
            target.unlink(missing_ok=True)
    except OSError:
        pass


def pending_path(temp_dir: Path, project_id: str, suffix: str = "") -> Path:
    return Path(temp_dir) / "pending" / f"{project_id}{suffix}"


_STAGED: dict[str, dict[str, Path | None]] = {}
_STAGED_LOCK = threading.Lock()


def stage(project_id: str, media: Path, thumbnail: Path | None = None) -> None:
    """Copia local disponible mientras el vídeo se guarda en Storage en segundo plano."""
    with _STAGED_LOCK:
        _STAGED[project_id] = {"media": media, "thumbnail": thumbnail}


def unstage(project_id: str) -> None:
    with _STAGED_LOCK:
        _STAGED.pop(project_id, None)


def staged(project_id: str, kind: str = "media") -> Path | None:
    with _STAGED_LOCK:
        entry = _STAGED.get(project_id)
    path = entry.get(kind) if entry else None
    return path if path is not None and path.is_file() else None


def find_pending(temp_dir: Path, project_id: str) -> Path | None:
    folder = Path(temp_dir) / "pending"
    if folder.is_dir():
        for candidate in folder.glob(f"{project_id}.*"):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return staged(project_id)


def drop_pending(temp_dir: Path, project_id: str) -> None:
    folder = Path(temp_dir) / "pending"
    if not folder.is_dir():
        return
    for candidate in folder.glob(f"{project_id}.*"):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def purge_pending(temp_dir: Path) -> None:
    folder = Path(temp_dir) / "pending"
    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)


def purge_temp(temp_dir: Path, *, attempts: int = 1, interval: float = 0.0) -> bool:
    import time

    folder = Path(temp_dir) / "materialized"
    if not folder.is_dir():
        return True

    total = max(1, int(attempts))
    for attempt in range(total):
        shutil.rmtree(folder, ignore_errors=True)
        if not folder.exists():
            return True
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


def signed_url(media_path: Any, *, expires_in: int = 3600) -> str | None:
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
