from __future__ import annotations

import datetime as _dt
import json
import queue as _queue_module
import threading
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import PipelineCancelled
from app.presentation.api import cluster, db, media_store
from app.presentation.api.progress import ProgressTracker, canonical_key


_QUEUE: Any = None
_CONN: Any = None
_CANCEL: dict[str, threading.Event] = {}
_STARTED = False
_LOCK = threading.Lock()
_GENERATION = 0

_SUBSCRIBERS: dict[str, list[Any]] = {}
_SUBSCRIBERS_LOCK = threading.Lock()

_SUBSCRIBER_QUEUE_MAX = 500

TRANSLATION_BATCH_CHARS = 4000

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
ORPHAN_STATES = {"PENDING", "QUEUED", "PROCESSING", "TRANSCRIBING", "TRANSLATING", "GENERATING_SUBTITLES"}
POLL_S = 2.0


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def subscribe(job_id: str) -> _queue_module.Queue:
    channel: _queue_module.Queue = _queue_module.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
    with _SUBSCRIBERS_LOCK:
        _SUBSCRIBERS.setdefault(job_id, []).append(channel)
    return channel


def unsubscribe(job_id: str, channel: _queue_module.Queue) -> None:
    with _SUBSCRIBERS_LOCK:
        listeners = _SUBSCRIBERS.get(job_id)
        if not listeners:
            return
        if channel in listeners:
            listeners.remove(channel)
        if not listeners:
            _SUBSCRIBERS.pop(job_id, None)


def publish(job_id: str, event: dict[str, Any]) -> None:
    with _SUBSCRIBERS_LOCK:
        listeners = list(_SUBSCRIBERS.get(job_id, ()))
    for channel in listeners:
        try:
            channel.put_nowait(event)
        except _queue_module.Full:
            try:
                channel.get_nowait()
                channel.put_nowait(event)
            except (_queue_module.Empty, _queue_module.Full):
                pass


def start(conn: Any) -> None:
    global _QUEUE, _STARTED, _CONN, _GENERATION
    import os

    if os.environ.get("IA_DISABLE_WORKER") == "1":
        return
    with _LOCK:
        if _STARTED and _CONN is conn:
            return
        _GENERATION += 1
        generation = _GENERATION
        import queue

        _QUEUE = queue.Queue()
        _CONN = conn
        _STARTED = True
        threading.Thread(
            target=_loop,
            args=(conn, generation),
            name=f"ia-jobs-{generation}",
            daemon=True,
        ).start()


def reset() -> None:
    global _QUEUE, _STARTED, _CONN, _GENERATION
    with _LOCK:
        _GENERATION += 1
        _QUEUE = None
        _STARTED = False
        _CONN = None
        _CANCEL.clear()
        with _SUBSCRIBERS_LOCK:
            _SUBSCRIBERS.clear()


def enqueue(conn: Any, job_id: str) -> None:
    start(conn)
    _CANCEL[job_id] = threading.Event()
    queued_entry = [{"stage": "QUEUED", "status": "QUEUED", "progress": 0, "at": _utcnow()}]
    db.run(
        conn,
        "UPDATE jobs SET status='QUEUED', progress=0, stage='QUEUED', error='', "
        "stage_history=?, updated_at=?, claimed_by='', heartbeat_at=0 WHERE id=?",
        (json.dumps(queued_entry, ensure_ascii=False), _utcnow(), job_id),
    )
    if _QUEUE is None:
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error='worker desactivado (IA_DISABLE_WORKER=1)', updated_at=? WHERE id=?",
            (_utcnow(), job_id),
        )
        return
    _QUEUE.put(job_id)


def cancel(conn: Any, job_id: str) -> None:
    event = _CANCEL.get(job_id)
    if event is not None:
        event.set()
    db.run(
        conn,
        "UPDATE jobs SET status='CANCELLED', stage='CANCELLED', error='', updated_at=? WHERE id=?",
        (_utcnow(), job_id),
    )


def reconcile_orphans(conn: Any) -> dict[str, int]:
    rows = db.many(
        conn,
        "SELECT id, status FROM jobs WHERE status IN ({}) AND (claimed_by='' OR heartbeat_at < ?)".format(
            ",".join("?" for _ in ORPHAN_STATES)
        ),
        (*sorted(ORPHAN_STATES), cluster.stale_before()),
    )
    requeued = 0
    for row in rows:
        db.run(
            conn,
            "UPDATE jobs SET status='PENDING', progress=0, stage='PENDING', "
            "error='reencolado tras reinicio del servidor', updated_at=? WHERE id=?",
            (_utcnow(), row["id"]),
        )
        requeued += 1
    if requeued:
        start(conn)
        for row in rows:
            enqueue(conn, row["id"])
    return {"requeued": requeued}


def requeue_stale(conn: Any) -> int:
    running = sorted(ORPHAN_STATES)
    rows = db.many(
        conn,
        "SELECT id FROM jobs WHERE claimed_by<>'' AND heartbeat_at < ? AND status IN ({})".format(
            ",".join("?" for _ in running)
        ),
        (cluster.stale_before(), *running),
    )
    for row in rows:
        db.run(
            conn,
            "UPDATE jobs SET status='QUEUED', stage='QUEUED', claimed_by='', heartbeat_at=0, "
            "error='reencolado: el proceso que lo atendía dejó de responder', updated_at=? "
            "WHERE id=? AND heartbeat_at < ?",
            (_utcnow(), row["id"], cluster.stale_before()),
        )
    return len(rows)


def claim(conn: Any, job_id: str) -> bool:
    with db.transaction(conn) as tx:
        row = tx.one(
            "UPDATE jobs SET claimed_by=?, heartbeat_at=? "
            "WHERE id=? AND status='QUEUED' AND claimed_by='' RETURNING id",
            (cluster.INSTANCE_ID, cluster.now(), job_id),
        )
    return row is not None


def _next_queued(conn: Any) -> str | None:
    row = db.one(
        conn,
        "SELECT id FROM jobs WHERE status='QUEUED' AND claimed_by='' ORDER BY created_at LIMIT 1",
    )
    return str(row["id"]) if row is not None else None


def _heartbeat(conn: Any, job_id: str, stop: threading.Event, cancel_event: threading.Event) -> None:
    while not stop.wait(cluster.HEARTBEAT_S):
        try:
            with db.transaction(conn) as tx:
                row = tx.one(
                    "UPDATE jobs SET heartbeat_at=? WHERE id=? AND claimed_by=? RETURNING status",
                    (cluster.now(), job_id, cluster.INSTANCE_ID),
                )
        except Exception:  # noqa: BLE001 - un fallo puntual de la base no corta el trabajo
            continue
        if row is None or row["status"] == "CANCELLED":
            cancel_event.set()


def _run_claimed(conn: Any, job_id: str) -> None:
    cancel_event = _CANCEL.setdefault(job_id, threading.Event())
    stop = threading.Event()
    beat = threading.Thread(
        target=_heartbeat, args=(conn, job_id, stop, cancel_event), name=f"beat-{job_id[:8]}", daemon=True
    )
    beat.start()
    try:
        _run_job(conn, job_id)
    finally:
        stop.set()
        _CANCEL.pop(job_id, None)


def _loop(conn: Any, generation: int) -> None:
    while True:
        current = _QUEUE
        if generation != _GENERATION or current is None:
            return
        from_queue = True
        try:
            job_id = current.get(timeout=POLL_S)
        except Exception:  # noqa: BLE001 - cola vacía: se mira la base por si otro proceso encoló
            from_queue = False
            job_id = None
        if generation != _GENERATION or _QUEUE is not current:
            if from_queue:
                try:
                    current.task_done()
                except Exception:  # noqa: BLE001
                    pass
            return
        try:
            requeue_stale(conn)
            job_id = job_id or _next_queued(conn)
        except Exception:  # noqa: BLE001 - la base puede fallar un momento
            job_id = job_id if from_queue else None
        if job_id is None:
            continue
        if not claim(conn, job_id):
            if from_queue:
                current.task_done()
            continue
        try:
            _purge_media_temp()
            _run_claimed(conn, job_id)
        except Exception as exc:  # noqa: BLE001
            try:
                db.run(
                    conn,
                    "UPDATE jobs SET status='FAILED', error=?, updated_at=? WHERE id=?",
                    (str(exc), _utcnow(), job_id),
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            _purge_media_temp()
            if from_queue:
                try:
                    current.task_done()
                except Exception:  # noqa: BLE001
                    pass


def _run_job(conn: Any, job_id: str) -> None:
    job = db.one(conn, "SELECT * FROM jobs WHERE id=?", (job_id,))
    if job is None or job["status"] == "CANCELLED":
        return
    project = db.one(conn, "SELECT * FROM projects WHERE id=?", (job["project_id"],))
    if project is None:
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error='proyecto inexistente', updated_at=? WHERE id=?",
            (_utcnow(), job_id),
        )
        return

    mode = "full"
    try:
        if "mode" in job.keys() and job["mode"]:
            mode = str(job["mode"])
    except (KeyError, TypeError):
        mode = "full"

    publish(job_id, {"type": "processing_started", "job_id": job_id, "mode": mode})

    try:
        if mode == "progressive":
            _run_job_progressive(conn, job_id, job, project)
            return

        _run_job_full(conn, job_id, job, project)
    finally:
        _purge_media_temp()
        try:
            media_store.drop_pending(
                Path(str(get_settings().paths.data_temp_dir)), str(project["id"])
            )
        except Exception:  # noqa: BLE001
            pass


def preload_models() -> None:
    try:
        threading.Thread(target=_preload_models_sync, name="preload-models", daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


def _preload_models_sync() -> None:
    try:
        from app.infrastructure.translation.m2m100_translation_adapter import (
            M2M100TranslatorAdapter,
        )

        M2M100TranslatorAdapter(settings_getter=get_settings).load()
    except Exception:  # noqa: BLE001
        pass


def _purge_media_temp() -> None:
    try:
        temp_root = Path(str(get_settings().paths.data_temp_dir))
    except Exception:  # noqa: BLE001
        return

    def _worker() -> None:
        try:
            media_store.purge_temp(temp_root, attempts=40, interval=0.5)
        except Exception:  # noqa: BLE001
            pass

    try:
        threading.Thread(target=_worker, name="purge-media", daemon=True).start()
    except Exception:  # noqa: BLE001
        pass


def _run_job_full(conn: Any, job_id: str, job: Any, project: Any) -> None:
    tracker = ProgressTracker()
    try:
        history: list[dict[str, Any]] = json.loads(str(job["stage_history"] or "[]"))
    except (json.JSONDecodeError, KeyError, TypeError):
        history = []
    if not isinstance(history, list):
        history = []

    def _persist(status: str, percent: int, stage: str) -> None:
        history.append({"stage": stage, "status": status, "progress": percent, "at": _utcnow()})
        db.run(
            conn,
            "UPDATE jobs SET status=?, progress=?, stage=?, stage_history=?, updated_at=? "
            "WHERE id=? AND status!='CANCELLED'",
            (status, percent, stage, json.dumps(history, ensure_ascii=False), _utcnow(), job_id),
        )
        publish(
            job_id,
            {
                "type": "processing_progress",
                "job_id": job_id,
                "status": status,
                "stage": stage,
                "progress": percent,
            },
        )

    def _report(key: str) -> None:
        update = tracker.announce(key)
        if update is not None:
            _persist(*update)

    def _progress(step: str, _elapsed: float, _phase: str = "done") -> None:
        _report(canonical_key(step))

    from app.application.pipeline.analyze_audio_pipeline import build_pipeline
    from app.core.config import get_settings

    settings = get_settings()
    targets = str(project["target_languages"] or "")
    settings.translation.target_languages = targets
    if targets:
        settings.translation.target_language = (
            targets.split(",")[0].strip() or settings.translation.target_language
        )

    _report("PREPARING")
    try:
        pipeline = build_pipeline(settings=settings)
    except Exception as exc:  # noqa: BLE001
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error=?, updated_at=? WHERE id=?",
            (f"configuración del motor inválida: {exc}", _utcnow(), job_id),
        )
        return

    pipeline.cancel_event = _CANCEL.get(job_id)
    pipeline.progress_callback = _progress

    temp_root = Path(str(settings.paths.data_temp_dir))
    remote = media_store.is_remote(project["media_path"])
    local_media: Path | None = None
    try:
        local_media = media_store.find_pending(temp_root, str(project["id"])) or (
            media_store.materialize(project["media_path"], temp_root, label=job_id)
        )
    except Exception as exc:  # noqa: BLE001
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error=?, updated_at=? WHERE id=?",
            (f"no se pudo obtener el vídeo: {exc}", _utcnow(), job_id),
        )
        return

    try:
        results = pipeline.run(str(local_media), job_id=job_id)
    except PipelineCancelled:
        _persist("CANCELLED", tracker.percent, "CANCELLED")
        return
    except Exception as exc:  # noqa: BLE001
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error=?, updated_at=? WHERE id=? AND status!='CANCELLED'",
            (str(exc), _utcnow(), job_id),
        )
        return
    finally:
        if remote:
            media_store.release(local_media, temp_root)

    status = str(results.get("status") or "SUCCESS")

    detected = str(results.get("detected_language") or "").strip().lower()
    if detected:
        db.run(
            conn,
            "UPDATE projects SET source_language=? WHERE id=?",
            (detected, project["id"]),
        )
    else:
        detected = str(project["source_language"] or "und").lower()

    db.run(conn, "UPDATE projects SET status=? WHERE id=?", (status, project["id"]))
    _store_cues(conn, project["id"], Path(settings.paths.data_output_dir) / job_id)

    errors: list[str] = []
    if status == "PARTIAL":
        errors.extend(str(e) for e in (results.get("translation_errors") or []))
        subtitle_error = str(results.get("subtitle_error") or "")
        if subtitle_error:
            errors.append(f"subtitulos:{subtitle_error}")
    elif status == "SUCCESS":
        errors.extend(str(e) for e in (results.get("translation_errors") or []))

    final = "COMPLETED" if status in {"SUCCESS", "PARTIAL"} else "FAILED"
    _, percent, stage = tracker.finish()
    if final == "COMPLETED":
        stage = "PARTIAL" if status == "PARTIAL" else "COMPLETED"
    else:
        percent = tracker.percent
        stage = "FAILED"
    _persist(final, percent, stage)
    db.run(
        conn,
        "UPDATE jobs SET error=? WHERE id=? AND status!='CANCELLED'",
        ("; ".join(errors), job_id),
    )


def _store_cues(conn: Any, project_id: str, output_dir: Path) -> None:
    if not output_dir.is_dir():
        return
    for path in output_dir.glob("subtitles_*.json"):
        language = path.stem.removeprefix("subtitles_")
        db.run(
            conn,
            """
            INSERT INTO subtitles (project_id, language, cues_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, language) DO UPDATE SET
                cues_json=excluded.cues_json, updated_at=excluded.updated_at
            """,
            (project_id, language, path.read_text(encoding="utf-8"), _utcnow()),
        )


def _run_job_progressive(conn: Any, job_id: str, job: Any, project: Any) -> None:
    from app.presentation.api import live

    _ = job
    runner = live.start_for_project(conn, project, job_id=job_id)
    cancel_event = _CANCEL.get(job_id)
    while runner is None:
        if cancel_event is not None and cancel_event.wait(POLL_S):
            return
        current = db.one(conn, "SELECT * FROM projects WHERE id=?", (project["id"],))
        if current is None:
            return
        if live.stored_snapshot(conn, current)["complete"]:
            db.run(conn, "UPDATE projects SET status='COMPLETED' WHERE id=?", (project["id"],))
            db.run(
                conn,
                "UPDATE jobs SET status='COMPLETED', stage='COMPLETED', progress=100, error='', updated_at=? "
                "WHERE id=? AND status!='CANCELLED'",
                (_utcnow(), job_id),
            )
            publish(job_id, {"type": "processing_completed", "job_id": job_id, "progress": 100,
                             "status": "COMPLETED"})
            return
        if cluster.lease_owner(conn, str(project["id"])) is None:
            runner = live.start_for_project(conn, current, job_id=job_id)
    while not runner.done.wait(0.5):
        if cancel_event is not None and cancel_event.is_set():
            runner.stop()
            runner.done.wait(10)
            return

    tracker = runner.tracker or live.JobTracker(conn, job_id)
    if runner.error or runner.subtitler is None:
        message = runner.error or "no se pudo iniciar el procesamiento"
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error=?, stage='FAILED', updated_at=? "
            "WHERE id=? AND status!='CANCELLED'",
            (message, _utcnow(), job_id),
        )
        db.run(conn, "UPDATE projects SET status='FAILED' WHERE id=?", (project["id"],))
        publish(job_id, {"type": "processing_failed", "job_id": job_id, "error": message})
        return

    subtitler = runner.subtitler
    tracker.stage("FINALIZING", 97, "GENERATING_SUBTITLES")
    errors = [f"{code}: traducción fallida" for code in sorted(subtitler.failed)]
    for code in subtitler.targets:
        if code != subtitler.source and code not in subtitler.failed and not subtitler.cues.get(code):
            if subtitler.cues.get(subtitler.source):
                errors.append(f"{code}: sin subtítulos traducidos")
    project_status = "COMPLETED" if not errors else "PARTIAL"
    db.run(conn, "UPDATE projects SET status=? WHERE id=?", (project_status, project["id"]))
    tracker.stage("COMPLETED" if not errors else "PARTIAL", 100, "COMPLETED")
    db.run(
        conn,
        "UPDATE jobs SET status='COMPLETED', progress=100, error=?, last_cue_ms=?, updated_at=? "
        "WHERE id=? AND status!='CANCELLED'",
        ("; ".join(errors), subtitler.duration_ms, _utcnow(), job_id),
    )
    log_metrics = {key: round(value, 1) for key, value in runner.metrics.items()}
    publish(
        job_id,
        {
            "type": "processing_completed",
            "job_id": job_id,
            "progress": 100,
            "status": project_status,
            "languages": sorted(subtitler.cues),
            "cues": len(subtitler.cues.get(subtitler.source, [])),
            "errors": errors,
            "metrics": log_metrics,
        },
    )


__all__ = [
    "cancel",
    "enqueue",
    "publish",
    "reconcile_orphans",
    "reset",
    "start",
    "subscribe",
    "unsubscribe",
]
