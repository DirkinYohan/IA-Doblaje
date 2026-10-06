"""Cola en proceso. El HTTP no ejecuta el pipeline.

Garantías:
* El progreso es monótono y cada etapa tiene nombre real (`progress.PIPELINE_STAGES`).
* El idioma detectado por T05 se persiste en `projects.source_language`.
* Un reinicio de la API no deja jobs abandonados: `reconcile_orphans()` los
  reencola o los marca como fallidos con causa explícita.
"""
from __future__ import annotations

import datetime as _dt
import json
import queue as _queue_module
import threading
from pathlib import Path
from typing import Any

from app.core.exceptions import PipelineCancelled
from app.presentation.api import db
from app.presentation.api.progress import ProgressTracker, canonical_key


_QUEUE: Any = None
_CONN: Any = None
_CANCEL: dict[str, threading.Event] = {}
_STARTED = False
_LOCK = threading.Lock()
# Generación del worker vigente: un hilo sólo atiende la cola de su generación.
_GENERATION = 0

# --- Bus de eventos por job (WebSocket) -----------------------------------
# Cada suscriptor recibe los eventos a medida que se producen. Es en memoria
# a propósito: el estado real está en SQLite, esto sólo evita el sondeo.
_SUBSCRIBERS: dict[str, list[Any]] = {}
_SUBSCRIBERS_LOCK = threading.Lock()

# Tamaño máximo de la cola de un suscriptor lento: se descartan eventos viejos
# antes que bloquear al worker o crecer sin límite.
_SUBSCRIBER_QUEUE_MAX = 500

# Traduce el audio en bloques de esta duración dentro de una ventana.
TRANSLATION_BATCH_CHARS = 4000

# Estados terminales: nunca se reencolan.
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
# Estados que implican que la cola en memoria los perdió al reiniciar.
ORPHAN_STATES = {"PENDING", "QUEUED", "PROCESSING", "TRANSCRIBING", "TRANSLATING", "GENERATING_SUBTITLES"}


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Bus de eventos
# ---------------------------------------------------------------------------


def subscribe(job_id: str) -> _queue_module.Queue:
    """Alta de un suscriptor. Devuelve la cola donde recibirá los eventos."""
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
    """Emite un evento a todos los suscriptores del job. Nunca bloquea."""
    with _SUBSCRIBERS_LOCK:
        listeners = list(_SUBSCRIBERS.get(job_id, ()))
    for channel in listeners:
        try:
            channel.put_nowait(event)
        except _queue_module.Full:
            # Suscriptor lento: se descarta el evento más antiguo y se sigue.
            try:
                channel.get_nowait()
                channel.put_nowait(event)
            except (_queue_module.Empty, _queue_module.Full):
                pass


def start(conn: Any) -> None:
    """Arranca el hilo de trabajos para esta conexión.

    Se usa un número de generación en lugar de un flag de parada compartido: un
    worker nuevo no puede heredar la orden de parada de uno anterior (esa
    carrera dejaba el trabajo sin procesar).
    """
    global _QUEUE, _STARTED, _CONN, _GENERATION
    import os

    if os.environ.get("IA_DISABLE_WORKER") == "1":
        return
    with _LOCK:
        if _STARTED and _CONN is conn:
            return
        # Un worker de otra conexión quedaría escribiendo en una base de datos
        # ajena: se le invalida subiendo la generación.
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
    """Detiene el worker y limpia el estado global (pruebas, cambios de BD)."""
    global _QUEUE, _STARTED, _CONN, _GENERATION
    with _LOCK:
        _GENERATION += 1  # invalida cualquier hilo en vuelo
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
        "stage_history=?, updated_at=? WHERE id=?",
        (json.dumps(queued_entry, ensure_ascii=False), _utcnow(), job_id),
    )
    if _QUEUE is None:
        # Worker desactivado explícitamente: no dejar el job mintiendo.
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
    """Reencola jobs que quedaron colgados por un reinicio del proceso.

    Debe llamarse una vez al arrancar la API, antes de aceptar peticiones.
    Un job en estado no terminal no tiene hilo que lo ejecute, así que se
    reencola con progreso 0 en lugar de quedar eternamente en PROCESSING.
    """
    rows = db.many(
        conn,
        "SELECT id, status FROM jobs WHERE status IN ({})".format(
            ",".join("?" for _ in ORPHAN_STATES)
        ),
        tuple(sorted(ORPHAN_STATES)),
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


def _loop(conn: Any, generation: int) -> None:
    """Bucle del worker. Termina cuando su generación deja de ser la vigente.

    Comprobar la generación (en lugar de un flag compartido) evita que un
    worker recién arrancado herede una orden de parada anterior.
    """
    while True:
        current = _QUEUE
        if generation != _GENERATION or current is None:
            return
        try:
            job_id = current.get(timeout=0.5)
        except Exception:  # noqa: BLE001 - cola vacía o cerrada
            continue
        if generation != _GENERATION or _QUEUE is not current:
            try:
                current.task_done()
            except Exception:  # noqa: BLE001
                pass
            return
        try:
            _run_job(conn, job_id)
        except Exception as exc:  # noqa: BLE001
            # Marcar el fallo nunca debe matar el hilo: si la conexión ya no
            # sirve, se ignora y se sigue atendiendo la cola.
            try:
                db.run(
                    conn,
                    "UPDATE jobs SET status='FAILED', error=?, updated_at=? WHERE id=?",
                    (str(exc), _utcnow(), job_id),
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
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

    # El bus es en memoria: se avisa de que se empieza.
    publish(job_id, {"type": "processing_started", "job_id": job_id, "mode": mode})

    if mode == "progressive":
        _run_job_progressive(conn, job_id, job, project)
        return

    _run_job_full(conn, job_id, job, project)


def _run_job_full(conn: Any, job_id: str, job: Any, project: Any) -> None:
    tracker = ProgressTracker()
    # Se parte del historial ya persistido (p. ej. la entrada QUEUED que escribió
    # `enqueue`) para no perder ninguna transición previa al arrancar el job.
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

    try:
        results = pipeline.run(project["media_path"], job_id=job_id)
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

    status = str(results.get("status") or "SUCCESS")

    # Persistir el idioma realmente detectado por T05.
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


# ---------------------------------------------------------------------------
# Procesamiento progresivo
# ---------------------------------------------------------------------------


def _run_job_progressive(conn: Any, job_id: str, job: Any, project: Any) -> None:
    """Produce subtítulos por ventanas y los publica en cuanto están listos.

    Se apoya en los MISMOS adapters que el pipeline completo (VAD, ASR,
    traducción); lo único distinto es el orden: se consume el audio por tramos
    y cada tramo se persiste y se emite inmediatamente.
    """
    import time

    from app.application.use_cases.progressive_transcription import (
        ProgressiveAudioWindowReader,
        ProgressiveTranscriber,
    )
    from app.core.config import get_settings

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

    def _fail(message: str) -> None:
        db.run(
            conn,
            "UPDATE jobs SET status='FAILED', error=?, stage='FAILED', updated_at=? "
            "WHERE id=? AND status!='CANCELLED'",
            (message, _utcnow(), job_id),
        )
        db.run(conn, "UPDATE projects SET status='FAILED' WHERE id=?", (project["id"],))
        publish(job_id, {"type": "processing_failed", "job_id": job_id, "error": message})

    settings = get_settings()
    targets_raw = str(project["target_languages"] or "")
    targets = [t.strip().lower() for t in targets_raw.split(",") if t.strip()]
    source = str(project["source_language"] or "und").lower()

    _persist(*tracker.set_progress("PREPARING", 1))
    db.run(conn, "UPDATE projects SET status='PROCESSING' WHERE id=?", (project["id"],))

    # --- 1) Extraer audio temporal (el vídeo original NO se toca) ----------
    # La preparación informa de su avance: en vídeos largos tarda minutos y sin
    # esto el trabajo parece colgado en 3 % (y el usuario lo cancela).
    def _prep_progress(percent: int, stage: str) -> None:
        _persist(*tracker.set_progress(stage, percent))

    try:
        prep = _prepare_audio_for_progressive(
            settings, project, job_id, progress=_prep_progress
        )
    except Exception as exc:  # noqa: BLE001
        _fail(f"no se pudo preparar el audio: {exc}")
        return

    wav_path, duration_ms, detected_language = prep
    if detected_language:
        source = detected_language
        db.run(
            conn,
            "UPDATE projects SET source_language=? WHERE id=?",
            (detected_language, project["id"]),
        )

    try:
        reader = ProgressiveAudioWindowReader(str(wav_path))
    except Exception as exc:  # noqa: BLE001
        _fail(f"audio preprocesado inutilizable: {exc}")
        return

    # --- 2) ASR + traducción ---------------------------------------------
    from app.core.profile_loader import load_asr_profile, resolve_compute_type
    from app.domain.value_objects.asr import AsrThresholds
    from app.infrastructure.audio.faster_whisper_asr_adapter import (
        FasterWhisperSmallASRAdapter,
    )

    profile_name = str(getattr(settings.processing.profile, "value", settings.processing.profile))
    spec = load_asr_profile(profile_name, settings.paths.configs_dir)
    requested = str(getattr(settings.processing.device, "value", settings.processing.device)).lower()
    device = "cpu"
    if requested in {"auto", "cuda", "mps"} and not bool(
        getattr(settings.processing, "force_cpu", False)
    ):
        try:
            import torch

            if requested in {"auto", "cuda"} and torch.cuda.is_available():
                device = "cuda"
            elif requested in {"auto", "mps"} and getattr(torch.backends, "mps", None):
                if torch.backends.mps.is_available():
                    device = "mps"
        except Exception:  # noqa: BLE001
            device = "cpu"
    compute_type, _note = resolve_compute_type(spec.compute_type, device)

    asr_adapter = FasterWhisperSmallASRAdapter(
        compute_type=compute_type,
        model_folder=spec.model_folder,
        device_resolver=lambda: device,
    )
    thresholds = AsrThresholds(
        beam_size=int(spec.beam_size),
        word_timestamps=bool(spec.word_timestamps),
        chunk_by_vad=bool(spec.chunk_by_vad),
    )

    translator_holder: dict[str, Any] = {}
    translator_lock = threading.Lock()

    def _translator() -> Any:
        """Adaptador de traducción (uno por trabajo, creado una sola vez)."""
        with translator_lock:
            if "adapter" not in translator_holder:
                from app.infrastructure.translation.m2m100_translation_adapter import (
                    M2M100TranslatorAdapter,
                )

                translator_holder["adapter"] = M2M100TranslatorAdapter(
                    settings_getter=lambda: settings,
                    max_length=int(settings.translation.max_length),
                    num_beams=int(settings.translation.num_beams),
                    do_sample=bool(settings.translation.do_sample),
                )
            return translator_holder["adapter"]

    def _translate(texts: list[str], target: str) -> list[str]:
        if not texts:
            return []
        from app.domain.value_objects.translation import TranslationLanguageCode

        adapter = _translator()
        # El modelo de traducción (M2M100 418M, ~1,8 GB) no es reentrante: se
        # serializa para que precarga y traducción no coincidan.
        with translator_lock:
            return adapter.translate_batch(
                texts,
                source_language=TranslationLanguageCode(source),
                target_language=TranslationLanguageCode(target),
            )

    def _warm_translator() -> None:
        """Precarga M2M100 en segundo plano mientras se extrae el audio.

        La carga del modelo es lo más caro del arranque (~15 s). Hacerlo en
        paralelo con la extracción evita que el usuario espere de más antes del
        primer subtítulo.
        """
        try:
            from app.domain.value_objects.translation import TranslationLanguageCode

            adapter = _translator()
            first = next((t for t in targets if t != source), source)
            with translator_lock:
                adapter.translate(
                    source_text="ok",
                    source_language=TranslationLanguageCode(source),
                    target_language=TranslationLanguageCode(first),
                )
        except Exception as exc:  # noqa: BLE001
            # La precarga es una optimización: si falla, la traducción real
            # informará del error con su propia causa.
            _ = exc

    # La traducción se precarga EN PARALELO con la extracción: es lo más lento
    # del arranque (~15 s) y así el primer subtítulo llega antes.
    threading.Thread(
        target=_warm_translator, name=f"warm-{job_id[:8]}", daemon=True
    ).start()

    # Estado acumulado de cues por idioma.
    cues_by_language: dict[str, list[dict[str, Any]]] = {source: []}
    for target in targets:
        cues_by_language.setdefault(target, [])
    next_index: dict[str, int] = {code: 1 for code in cues_by_language}
    # Errores de traducción acumulados: determinan si el trabajo es PARTIAL.
    translation_errors: list[str] = []

    def _persist_language(code: str) -> int:
        """Guarda la pista completa del idioma. Devuelve el total de cues."""
        cues = cues_by_language[code]
        db.run(
            conn,
            """
            INSERT INTO subtitles (project_id, language, cues_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id, language) DO UPDATE SET
                cues_json=excluded.cues_json, updated_at=excluded.updated_at
            """,
            (project["id"], code, json.dumps(cues, ensure_ascii=False), _utcnow()),
        )
        return len(cues)

    cancel_event = _CANCEL.get(job_id)
    transcriber = ProgressiveTranscriber(
        asr=asr_adapter,
        vad=object(),
        language=None if source in {"", "und"} else source,
        window_ms=30_000,
        overlap_ms=1_000,
        # La primera ventana es corta para que el primer subtítulo aparezca en
        # segundos, no tras medio minuto de audio.
        first_window_ms=8_000,
    )

    # Índice de inicio de la ventana actual por idioma: así el evento publica
    # EXACTAMENTE los cues de este lote y no los de una ventana anterior.
    window_start: dict[str, int] = {code: 0 for code in cues_by_language}

    def _on_cues(window: Any) -> None:
        """Cada ventana: traducir, persistir y emitir. Nunca al final."""
        if cancel_event is not None and cancel_event.is_set():
            return
        source_cues = list(window.cues)
        if not source_cues:
            return

        # Cada pista arranca esta ventana donde terminó la anterior.
        for code in cues_by_language:
            window_start[code] = len(cues_by_language[code])

        for cue in source_cues:
            cues_by_language.setdefault(source, []).append(
                {
                    "index": next_index[source],
                    "start_ms": int(cue.start_ms),
                    "end_ms": int(cue.end_ms),
                    "text": str(cue.text),
                    "speaker": str(getattr(cue, "speaker", "") or ""),
                    "language": source,
                    "state": "final",
                }
            )
            next_index[source] += 1

        # Traducción por lotes: una sola pasada de `generate` por ventana.
        texts = [str(cue.text) for cue in source_cues]
        for target in targets:
            if target == source:
                continue
            try:
                translated = _translate(texts, target)
            except Exception as exc:  # noqa: BLE001
                # Una traducción fallida NO puede pasar por éxito silencioso:
                # se registra la causa y el trabajo terminará en PARTIAL.
                message = f"{target}: {exc}"
                if message not in translation_errors:
                    translation_errors.append(message)
                publish(
                    job_id,
                    {
                        "type": "translation_failed",
                        "job_id": job_id,
                        "language": target,
                        "error": str(exc),
                    },
                )
                continue
            for cue, text in zip(source_cues, translated, strict=False):
                cues_by_language.setdefault(target, []).append(
                    {
                        "index": next_index[target],
                        "start_ms": int(cue.start_ms),
                        "end_ms": int(cue.end_ms),
                        "text": str(text),
                        "speaker": str(getattr(cue, "speaker", "") or ""),
                        "language": target,
                        "state": "final",
                    }
                )
                next_index[target] += 1

        totals = {code: _persist_language(code) for code in cues_by_language}
        covered = int(window.end_ms)
        if duration_ms > 0:
            # 10 %..95 % del progreso corresponde al audio ya transcrito.
            target = 10 + int(85 * min(1.0, covered / duration_ms))
        else:
            target = max(tracker.percent, 20)
        # `_persist` deja la etapa en el historial; `set_progress` calcula el %
        # garantizando que nunca retrocede.
        _persist(*tracker.set_progress("CUES_READY", target))
        db.run(
            conn,
            "UPDATE jobs SET last_cue_ms=?, updated_at=? WHERE id=? AND status!='CANCELLED'",
            (covered, _utcnow(), job_id),
        )
        db.run(
            conn,
            "UPDATE projects SET status='PARTIALLY_READY' WHERE id=?",
            (project["id"],),
        )

        # El evento lleva TODAS las traducciones del lote, no sólo el original:
        # el reproductor cambia de idioma sin volver a procesar el vídeo.
        by_language: dict[str, list[dict[str, Any]]] = {}
        for code, entries in cues_by_language.items():
            fresh = entries[window_start.get(code, 0):]
            if not fresh:
                continue
            by_language[code] = [
                {
                    "start_ms": int(cue["start_ms"]),
                    "end_ms": int(cue["end_ms"]),
                    "text": str(cue["text"]),
                }
                for cue in fresh
            ]
        publish(
            job_id,
            {
                "type": "subtitle_created",
                "job_id": job_id,
                "project_id": project["id"],
                "covered_ms": covered,
                "duration_ms": duration_ms,
                "progress": tracker.percent,
                "source_language": source,
                # Compatibilidad: cues del idioma original.
                "cues": by_language.get(source, []),
                # Todas las pistas de este lote.
                "cues_by_language": by_language,
                "totals": totals,
            },
        )

    try:
        transcriber.process(
            reader,
            on_cues=_on_cues,
            cancel_event=cancel_event,
        )
    except PipelineCancelled:
        _persist("CANCELLED", tracker.percent, "CANCELLED")
        return
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
        return

    if cancel_event is not None and cancel_event.is_set():
        _persist("CANCELLED", tracker.percent, "CANCELLED")
        return

    # --- 3) Cierre --------------------------------------------------------
    _persist(*tracker.set_progress("FINALIZING", 97))
    total_cues = len(cues_by_language.get(source, []))
    # Si alguna pista pedida quedó vacía, el trabajo NO es un éxito completo.
    empty_targets = [
        code for code in targets if code != source and not cues_by_language.get(code)
    ]
    for code in empty_targets:
        note = f"{code}: sin subtítulos traducidos"
        if note not in translation_errors:
            translation_errors.append(note)

    project_status = "COMPLETED" if not translation_errors else "PARTIAL"
    db.run(conn, "UPDATE projects SET status=? WHERE id=?", (project_status, project["id"]))
    status, percent, stage = tracker.finish()
    if translation_errors:
        stage = "PARTIAL"
    _persist(status, percent, stage)
    db.run(
        conn,
        "UPDATE jobs SET error=? WHERE id=? AND status!='CANCELLED'",
        ("; ".join(translation_errors), job_id),
    )
    publish(
        job_id,
        {
            "type": "processing_completed",
            "job_id": job_id,
            "progress": 100,
            "status": project_status,
            "languages": sorted(cues_by_language),
            "cues": total_cues,
            "errors": translation_errors,
        },
    )


def _prepare_audio_for_progressive(
    settings: Any, project: Any, job_id: str, progress: Any | None = None
) -> tuple[Path, int, str]:
    """Extrae el audio temporal para la IA SIN tocar el vídeo original.

    Devuelve ``(ruta_wav, duración_ms, idioma_detectado)``.

    `progress` recibe ``(percent, stage)`` en cada subpaso: en vídeos largos la
    preparación (extracción + VAD + idioma) tarda minutos y sin informar parece
    que el trabajo se ha quedado colgado.
    """
    from pathlib import Path as _Path

    def _report(percent: int, stage: str) -> None:
        if progress is not None:
            try:
                progress(percent, stage)
            except Exception:  # noqa: BLE001
                pass

    from app.application.use_cases.analyze_media_prep import AnalyzeMediaPrepUseCase
    from app.infrastructure.audio.ffmpeg_adapters import (
        FFmpegAudioExtractor,
        FFmpegAudioPreprocessor,
        FFprobeMediaValidator,
        SubprocessFFmpegBinaryResolver,
    )

    media_path = _Path(str(project["media_path"]))
    if not media_path.is_file():
        raise FileNotFoundError(f"el vídeo original no existe: {media_path!s}")

    _report(3, "EXTRACTING_AUDIO")
    media_prep = AnalyzeMediaPrepUseCase(
        binary_resolver=SubprocessFFmpegBinaryResolver(),
        validator=FFprobeMediaValidator(),
        extractor=FFmpegAudioExtractor(),
        preprocessor=FFmpegAudioPreprocessor(),
    )
    result = media_prep.execute(
        str(media_path),
        job_id=job_id,
        settings=settings,
        progress_cb=lambda step: _report(
            8 if "T02" in step else 6 if "T03" in step else 4, "EXTRACTING_AUDIO"
        ),
    )
    preprocessed = result.preprocessed_audio
    if preprocessed is None:
        raise RuntimeError("la extracción de audio no produjo WAV preprocesado")
    _report(10, "EXTRACTING_AUDIO")

    # Idioma: si el proyecto ya lo declara, NO se detecta.
    #
    # El detector (Whisper encoder, 950 MB) tarda ~10 s y el VAD completo otros
    # ~5 s. Son 15 s de silencio antes del primer subtítulo, inaceptables en un
    # flujo que debe mostrar subtítulos cuanto antes. Sólo se paga ese coste
    # cuando el idioma es realmente desconocido ("und").
    declared = str(project["source_language"] or "").strip().lower()
    detected = "" if declared in {"", "und"} else declared

    if not detected:
        _report(12, "ANALYZING_AUDIO")
        try:
            from app.application.use_cases.run_language_detection import (
                RunLanguageDetectionUseCase,
            )
            from app.application.use_cases.run_vad import RunVoiceActivityDetectionUseCase
            from app.infrastructure.audio.silero_vad_adapter import SileroVADAdapter
            from app.infrastructure.audio.whisper_lid_adapter import (
                WhisperEncoderLanguageDetectionAdapter,
            )

            vad = RunVoiceActivityDetectionUseCase(
                vad_detector=SileroVADAdapter(settings=settings)
            )
            vad_result = vad.execute(result, job_id=job_id)
            _report(15, "ANALYZING_AUDIO")
            lid_result = RunLanguageDetectionUseCase(
                detector=WhisperEncoderLanguageDetectionAdapter(settings=settings)
            ).execute(result, vad_result, job_id=job_id)
            code = str(getattr(lid_result, "language_code", "") or "")
            detected = "" if code in {"", "und"} else code.lower()
            _report(18, "ANALYZING_AUDIO")
        except Exception:  # noqa: BLE001
            detected = ""

    duration_ms = int(round(float(getattr(preprocessed, "duration_sec", 0.0)) * 1000))
    return _Path(str(preprocessed.wav_path)), duration_ms, detected


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
