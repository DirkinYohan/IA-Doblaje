"""Guarda en Storage un vídeo recién subido sin hacer esperar al usuario.

La subida responde en cuanto el archivo llega al servidor. Comprimir (si supera
el límite del almacenamiento) y subirlo a Supabase puede tardar minutos, así
que se hace en un hilo; mientras tanto el vídeo se sirve desde la copia local.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.presentation.api import media_store
from app.presentation.api import storage_fit

log = logging.getLogger("ia.media_upload")

ATTEMPTS = 2
HEAD_START_WAIT_S = 120.0
POLL_S = 2.0


def store_in_background(
    *,
    project_id: str,
    source: Path,
    duration_ms: int,
    media_key: str,
    content_type: str,
    thumbnail: Path | None,
    thumbnail_key: str,
    still_exists: Callable[[], bool],
) -> threading.Thread:
    media_store.stage(project_id, source, thumbnail)
    worker = threading.Thread(
        target=store,
        kwargs={
            "project_id": project_id, "source": source, "duration_ms": duration_ms,
            "media_key": media_key, "content_type": content_type, "thumbnail": thumbnail,
            "thumbnail_key": thumbnail_key, "still_exists": still_exists,
        },
        name=f"store-{project_id[:8]}",
        daemon=True,
    )
    worker.start()
    return worker


def store(
    *,
    project_id: str,
    source: Path,
    duration_ms: int,
    media_key: str,
    content_type: str,
    thumbnail: Path | None,
    thumbnail_key: str,
    still_exists: Callable[[], bool],
    ready: Callable[[str], bool] | None = None,
    wait_s: float = HEAD_START_WAIT_S,
) -> bool:
    stored = False
    try:
        if _needs_compression(source, duration_ms):
            _wait_for_live(project_id, ready, wait_s)
        for attempt in range(1, ATTEMPTS + 1):
            try:
                _upload(source, duration_ms, media_key, content_type, project_id)
                if thumbnail is not None and thumbnail.is_file():
                    media_store.upload_bytes(thumbnail.read_bytes(), thumbnail_key, "image/jpeg")
                stored = True
                break
            except storage_fit.TooLongForStorage as exc:
                log.warning("storage[%s]: %s", project_id[:8], exc)
                break
            except Exception as exc:  # noqa: BLE001 - se reintenta y se informa
                log.warning("storage[%s]: intento %d/%d fallido: %s", project_id[:8], attempt, ATTEMPTS, exc)
        if stored and not _safe(still_exists):
            for key in (media_key, thumbnail_key):
                _delete(key)
        return stored
    finally:
        media_store.unstage(project_id)
        source.unlink(missing_ok=True)
        if thumbnail is not None:
            thumbnail.unlink(missing_ok=True)


def _needs_compression(source: Path, duration_ms: int) -> bool:
    try:
        return storage_fit.needs_fit(source.stat().st_size, duration_ms)
    except (OSError, storage_fit.TooLongForStorage):
        return False


def _wait_for_live(project_id: str, ready: Callable[[str], bool] | None, wait_s: float) -> None:
    """Comprimir compite por la CPU con la transcripción en vivo: se espera a que tenga margen."""
    if ready is None:
        from app.presentation.api import live

        ready = live.head_start_ready
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            if ready(project_id):
                return
        except Exception:  # noqa: BLE001 - ante la duda no se bloquea el guardado
            return
        time.sleep(POLL_S)


def _upload(source: Path, duration_ms: int, key: str, content_type: str, project_id: str) -> None:
    fitted = storage_fit.fit(source, duration_ms, source.with_name(f"{project_id}-fit.mp4"))
    try:
        media_store.upload_file(fitted, key, content_type)
    finally:
        if fitted != source:
            fitted.unlink(missing_ok=True)


def _safe(check: Callable[[], bool]) -> bool:
    try:
        return bool(check())
    except Exception:  # noqa: BLE001 - ante la duda se conserva el archivo
        return True


def _delete(key: Any) -> None:
    try:
        media_store.delete(key)
    except Exception:  # noqa: BLE001
        pass
