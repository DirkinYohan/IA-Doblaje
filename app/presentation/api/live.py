"""Subtítulos progresivos generados desde el ARCHIVO, por delante del cabezal.

Un único proceso por proyecto (``LiveRunner``) genera TODOS los subtítulos del
vídeo: arranca en cuanto termina la subida, sigue aunque nadie lo esté viendo y
termina cuando el vídeo entero está transcrito y traducido a cada idioma. El
trabajo «Producir» y cada reproductor abierto se ENGANCHAN a ese mismo proceso:
nunca hay dos transcribiendo el mismo vídeo.

* Prioridad: el hueco más cercano a la posición de quien está mirando, en el
  idioma que está mirando. Sin espectadores, de principio a fin.
* Después de lo prioritario se completa el resto: el original entero y cada
  idioma de destino (una traducción que falla no bloquea las demás).
* Cada lote se persiste y se publica con un número de secuencia. Un cliente
  que se conecta (o reconecta) recibe primero una instantánea completa: no se
  pierde nada aunque los primeros eventos salieran antes de conectarse.
* Whisper y el traductor se cargan una vez por proceso y se comparten.

Contrato de tiempos: milisegundos enteros sobre la línea temporal del vídeo.
Identificador estable de un segmento: ``{idioma}:{inicio}:{fin}``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.application.use_cases.build_subtitles import CueLimits, build_cues
from app.application.use_cases.progressive_transcription import ProgressiveTranscriber

log = logging.getLogger("ia.live")

# Hasta dónde se adelanta el trabajo prioritario respecto al cabezal.
LOOKAHEAD_MS = 120_000
# Ventana de audio por pasada de Whisper. Su coste en CPU apenas depende de la
# longitud (rellena hasta 30 s), así que ventanas largas rinden más.
WINDOW_MS = 24_000
# Ventana junto al cabezal: alguien está esperando a ese subtítulo.
NEAR_WINDOW_MS = 12_000
# Un segmento que toca el borde de una ventana puede estar cortado: se repite
# desde su inicio en la siguiente pasada.
EDGE_MS = 400
# Al empezar en mitad de un hueco se retrocede un poco para no cortar palabras.
BACKOFF_MS = 1_500
# Tramo máximo que se traduce de una vez (varias frases: contexto suficiente).
TRANSLATE_CHUNK_MS = 30_000
# Pasadas fallidas seguidas antes de dar el procesamiento por fallido.
MAX_FAILURES = 3


def segment_id(language: str, start_ms: int, end_ms: int) -> str:
    return f"{language}:{int(start_ms)}:{int(end_ms)}"


class Ranges:
    """Conjunto de intervalos [inicio, fin) ya procesados, fusionados y ordenados."""

    def __init__(self) -> None:
        self.items: list[tuple[int, int]] = []

    def add(self, start: int, end: int) -> None:
        if end <= start:
            return
        merged: list[tuple[int, int]] = []
        placed = False
        for a, b in self.items:
            if b < start or a > end:
                if a > end and not placed:
                    merged.append((start, end))
                    placed = True
                merged.append((a, b))
            else:
                start, end = min(a, start), max(b, end)
        if not placed:
            merged.append((start, end))
        self.items = sorted(merged)

    def contains(self, t: int) -> bool:
        return any(a <= t < b for a, b in self.items)

    def end_before(self, t: int) -> int:
        """Fin del último intervalo que termina en o antes de `t` (0 si no hay)."""
        ends = [b for _a, b in self.items if b <= t]
        return max(ends, default=0)

    def first_gap(self, lo: int, hi: int) -> tuple[int, int] | None:
        """Primer hueco dentro de [lo, hi), o None si está todo cubierto."""
        cursor = lo
        for a, b in self.items:
            if b <= cursor:
                continue
            if a > cursor:
                return (cursor, min(a, hi)) if cursor < hi else None
            cursor = max(cursor, b)
            if cursor >= hi:
                return None
        return (cursor, hi) if cursor < hi else None

    def prefix_end(self) -> int:
        """Hasta dónde está cubierto de forma CONTINUA desde 0."""
        if self.items and self.items[0][0] <= 0:
            return self.items[0][1]
        return 0

    def total(self) -> int:
        return sum(b - a for a, b in self.items)

    def as_list(self) -> list[list[int]]:
        return [[a, b] for a, b in self.items]


@dataclass
class LiveCallbacks:
    """Dependencias inyectadas (modelos y salida): permiten probar sin modelos."""

    read: Callable[[int, int], Any]
    transcribe: Callable[[Any, int, str | None], list[Any]]
    translate: Callable[[list[str], str, str], list[str]]
    detect: Callable[[Any], str]
    emit: Callable[[dict[str, Any]], None]
    persist: Callable[[str, list[dict[str, Any]], int], None]


class LiveSubtitler:
    """Genera las pistas del vídeo priorizando lo que se está viendo."""

    def __init__(
        self,
        *,
        callbacks: LiveCallbacks,
        source_language: str,
        duration_ms: int,
        targets: list[str] | None = None,
        existing: dict[str, tuple[list[dict[str, Any]], int]] | None = None,
        sample_rate: int = 16_000,
    ) -> None:
        self.cb = callbacks
        self.source = source_language if source_language not in {"", "und"} else ""
        self.duration_ms = int(duration_ms)
        self.sample_rate = sample_rate
        # Idiomas a completar además del original (en el orden pedido).
        self.targets = [t for t in (targets or []) if t]
        self.cues: dict[str, list[dict[str, Any]]] = {}
        self.covered: dict[str, Ranges] = {}
        for language, (cues, covered_ms) in (existing or {}).items():
            self.cues[language] = sorted(
                ({**c, "id": c.get("id") or segment_id(language, c["start_ms"], c["end_ms"])} for c in cues),
                key=lambda c: (int(c["start_ms"]), int(c["end_ms"])),
            )
            ranges = Ranges()
            ranges.add(0, int(covered_ms))
            self.covered[language] = ranges
        self._cond = threading.Condition()
        # Protege `cues`/`covered` frente a las instantáneas pedidas desde otros hilos.
        self.data_lock = threading.RLock()
        self._position = 0
        self._language = ""
        self._stopped = False
        # Idiomas cuya traducción falló: no se reintentan en bucle.
        self.failed: set[str] = set()

    # --- Control (desde otros hilos) ----------------------------------------
    def set_position(self, time_ms: int, language: str | None = None) -> None:
        with self._cond:
            self._position = max(0, min(int(time_ms), self.duration_ms))
            if language:
                self._language = language
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stopped

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def run(self, *, until_complete: bool = True) -> None:
        failures = 0
        while not self._stopped:
            try:
                worked = self.step()
                failures = 0
            except Exception as exc:  # noqa: BLE001 - se informa; repetido, se abandona
                failures += 1
                log.warning("live: error en una pasada (%d/%d): %s", failures, MAX_FAILURES, exc)
                if failures >= MAX_FAILURES:
                    # Sin reintentos infinitos: p. ej. el vídeo ya no se puede
                    # leer. El job se marca FAILED con la causa.
                    raise
                self.cb.emit({"type": "error", "message": str(exc)})
                worked = False
            if not worked:
                if until_complete and self.is_complete():
                    return
                with self._cond:
                    if not self._stopped:
                        self._cond.wait(timeout=1.0)

    # --- Estado ---------------------------------------------------------------
    def languages(self) -> list[str]:
        return [self.source] + [t for t in self.targets if t != self.source] if self.source else []

    def is_complete(self) -> bool:
        if not self.source:
            return False
        if self._ranges(self.source).first_gap(0, self.duration_ms) is not None:
            return False
        for target in self.targets:
            if target == self.source or target in self.failed:
                continue
            if self._first_untranslated(target, 0, self.duration_ms) is not None:
                return False
        return True

    def progress(self) -> int:
        """Porcentaje del trabajo total (original + cada idioma de destino)."""
        languages = self.languages()
        if not languages or self.duration_ms <= 0:
            return 0
        done = sum(min(self.duration_ms, self._ranges(code).total()) for code in languages)
        return int(100 * done / (self.duration_ms * len(languages)))

    def snapshot(self) -> dict[str, Any]:
        with self.data_lock:
            return {
                "source_language": self.source,
                "duration_ms": self.duration_ms,
                "tracks": {code: [dict(c) for c in cues] for code, cues in self.cues.items()},
                "covered": {code: r.as_list() for code, r in self.covered.items()},
                "failed": sorted(self.failed),
                "complete": self.is_complete(),
            }

    # --- Una unidad de trabajo --------------------------------------------
    def step(self) -> bool:
        """Procesa el hueco más urgente. False si no queda nada pendiente."""
        with self._cond:
            position, language = self._position, self._language
        if not self.source:
            self._detect_source(position)
            if not self.source:
                return False
        language = language or next((t for t in self.targets if t != self.source), self.source)

        # 1) Lo que se está viendo: por delante del cabezal, en su idioma.
        horizon = min(self.duration_ms, position + LOOKAHEAD_MS)
        if self._work_between(position, horizon, language, near=True):
            return True
        # 2) El resto del vídeo: original y cada idioma, el que se ve primero.
        ordered = [language] + [t for t in self.targets if t != language]
        for start in (position, 0):
            if self._work_between(start, self.duration_ms, None, near=False):
                return True
            for target in ordered:
                if self._work_between(start, self.duration_ms, target, near=False, source=False):
                    return True
        return False

    def _work_between(
        self, lo: int, hi: int, language: str | None, *, near: bool, source: bool = True
    ) -> bool:
        source_gap = self._ranges(self.source).first_gap(lo, hi) if source else None
        target_gap = None
        if language and language != self.source and language not in self.failed:
            target_gap = self._first_untranslated(language, lo, hi)
        if target_gap and (source_gap is None or target_gap[0] <= source_gap[0]):
            try:
                self._translate(language or "", *target_gap)
            except Exception as exc:  # noqa: BLE001 - un fallo no se disfraza de éxito
                with self.data_lock:
                    self.failed.add(language or "")
                log.warning("live: traducción a %s fallida: %s", language, exc)
                self.cb.emit({"type": "translation_failed", "language": language, "error": str(exc)})
            return True
        if source_gap:
            self._transcribe(lo if near else source_gap[0], *source_gap, near=near)
            return True
        return False

    def _ranges(self, language: str) -> Ranges:
        return self.covered.setdefault(language, Ranges())

    def _detect_source(self, position: int) -> None:
        audio = self.cb.read(position, min(self.duration_ms, position + 30_000))
        code = self.cb.detect(audio) if getattr(audio, "size", 0) else ""
        if code:
            self.source = code
            self.cb.emit({"type": "source_language", "language": code})

    def _transcribe(self, position: int, gap_start: int, gap_end: int, *, near: bool) -> None:
        ranges = self._ranges(self.source)
        start = gap_start
        if start == position and not ranges.contains(position - 1):
            # Se empieza en mitad de un hueco: un poco antes, para no cortar
            # la palabra que suena justo en la posición del salto.
            start = max(ranges.end_before(position), position - BACKOFF_MS, 0)
        size = NEAR_WINDOW_MS if near and start <= position + 1_000 else WINDOW_MS
        end = min(gap_end, start + size, self.duration_ms)
        # El borde es "blando" si no lo marca lo ya procesado ni el final.
        soft_edge = end == start + size and end < self.duration_ms and end < gap_end

        audio = self.cb.read(start, end)
        rows: list[tuple[int, int, str, str]] = []
        if getattr(audio, "size", 0) and ProgressiveTranscriber._speech_ratio(audio) >= 0.001:
            for seg in self.cb.transcribe(audio, start, self.source):
                text = str(getattr(seg, "text", "") or "").strip()
                seg_start = max(int(getattr(seg, "start_ms", start)), start)
                seg_end = min(int(getattr(seg, "end_ms", end)), end)
                if text and seg_end > seg_start:
                    rows.append((seg_start, seg_end, text, ""))

        covered_until = end
        if soft_edge and len(rows) > 1 and rows[-1][1] >= end - EDGE_MS:
            # Probablemente cortado por la ventana: se repite desde su inicio.
            covered_until = max(start + 1, rows.pop()[0])

        cues = build_cues(rows, language=self.source, limits=CueLimits(), state="final")
        fresh = [
            {"start_ms": int(c.start_ms), "end_ms": int(c.end_ms), "text": str(c.text)}
            for c in cues
            if int(c.start_ms) < covered_until
        ]
        self._publish(self.source, fresh, start, covered_until)

    def _first_untranslated(self, language: str, lo: int, hi: int) -> tuple[int, int] | None:
        target = self._ranges(language)
        for a, b in self._ranges(self.source).items:
            if b <= lo or a >= hi:
                continue
            gap = target.first_gap(max(a, lo), min(b, hi))
            if gap:
                return gap
        return None

    def _translate(self, language: str, gap_start: int, gap_end: int) -> None:
        target = self._ranges(language)
        source_cues = self.cues.get(self.source, [])
        start = gap_start
        # Un cue del original que ya estaba sonando en el inicio del hueco
        # también hay que traducirlo, o ese instante quedaría sin texto.
        for cue in source_cues:
            cue_start, cue_end = int(cue["start_ms"]), int(cue["end_ms"])
            if cue_start < start < cue_end and not target.contains(cue_start):
                start = cue_start
                break
        end = min(gap_end, start + TRANSLATE_CHUNK_MS)
        batch = [c for c in source_cues if start <= int(c["start_ms"]) < end]
        texts = [str(c["text"]) for c in batch]
        translated = self.cb.translate(texts, self.source, language) if texts else []
        fresh = [
            {"start_ms": int(c["start_ms"]), "end_ms": int(c["end_ms"]), "text": str(text)}
            for c, text in zip(batch, translated, strict=False)
        ]
        self._publish(language, fresh, start, end)

    def _publish(self, language: str, fresh: list[dict[str, Any]], start: int, end: int) -> None:
        with self.data_lock:
            current = self.cues.setdefault(language, [])
            known = {(int(c["start_ms"]), int(c["end_ms"])) for c in current}
            fresh = [
                {**c, "id": segment_id(language, c["start_ms"], c["end_ms"]), "language": language}
                for c in fresh
                if (c["start_ms"], c["end_ms"]) not in known
            ]
            current.extend(fresh)
            current.sort(key=lambda c: (int(c["start_ms"]), int(c["end_ms"])))
            ranges = self._ranges(language)
            ranges.add(start, end)
            event = {
                "type": "cues",
                "language": language,
                "cues": fresh,
                "range": [start, end],
                "covered": ranges.as_list(),
            }
            persisted = list(current)
            prefix = ranges.prefix_end()
        self.cb.emit(event)
        self.cb.persist(language, persisted, prefix)


# --- Un proceso por proyecto -----------------------------------------------

Listener = Callable[[dict[str, Any]], None]


class LiveRunner:
    """Hilo que genera los subtítulos de UN proyecto y reparte sus eventos."""

    def __init__(self, project_id: str, build: Callable[["LiveRunner"], LiveSubtitler]) -> None:
        self.project_id = project_id
        self._build = build
        self._listeners: dict[int, Listener] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._next_listener = 0
        self.done = threading.Event()
        self.started_at = time.monotonic()
        self.metrics: dict[str, float] = {}
        self.job_id: str | None = None
        self.tracker: JobTracker | None = None
        self.on_progress: Callable[["LiveRunner"], None] | None = None
        # Se construye en su propio hilo (abrir el vídeo puede tardar): hasta
        # entonces la posición pedida se guarda y se aplica al empezar.
        self.subtitler: LiveSubtitler | None = None
        self.error = ""
        self._pending_position: tuple[int, str | None] = (0, None)

    def attach_job(self, conn: Any, job_id: str) -> None:
        self.job_id = job_id
        self.tracker = JobTracker(conn, job_id)
        self.on_progress = self.tracker.on_cues

    def set_position(self, time_ms: int, language: str | None = None) -> None:
        with self._lock:
            self._pending_position = (int(time_ms), language or self._pending_position[1])
            subtitler = self.subtitler
        if subtitler is not None:
            subtitler.set_position(int(time_ms), language)

    # Eventos ---------------------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        self._mark_first(event)
        with self._lock:
            self._seq += 1
            event = {**event, "seq": self._seq}
            listeners = list(self._listeners.values())
        for listener in listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 - un cliente roto no frena a los demás
                pass
        if (
            event.get("type") == "cues"
            and self.on_progress is not None
            and self.subtitler is not None
            and event.get("language") == self.subtitler.source
        ):
            self.on_progress(self)

    def attach(self, listener: Listener) -> tuple[dict[str, Any], Callable[[], None]]:
        """Suscribe y devuelve la instantánea actual (con su secuencia)."""
        with self._lock:
            key = self._next_listener
            self._next_listener += 1
            self._listeners[key] = listener
            seq = self._seq
            subtitler = self.subtitler
        state = (
            subtitler.snapshot()
            if subtitler is not None
            else {"tracks": {}, "covered": {}, "failed": [], "complete": False}
        )
        snapshot = {"type": "snapshot", "seq": seq, **state}

        def detach() -> None:
            with self._lock:
                self._listeners.pop(key, None)

        return snapshot, detach

    def viewers(self) -> int:
        with self._lock:
            return len(self._listeners)

    # Ciclo de vida -----------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._run, name=f"live-{self.project_id[:8]}", daemon=True).start()

    def _run(self) -> None:
        log.info("live[%s]: procesamiento iniciado", self.project_id[:8])
        subtitler: LiveSubtitler | None = None
        try:
            subtitler = self._build(self)
            with self._lock:
                self.subtitler = subtitler
                position, language = self._pending_position
            subtitler.set_position(position, language)
            if self._stop_requested:
                subtitler.stop()
            subtitler.run(until_complete=True)
        except Exception as exc:  # noqa: BLE001 - p. ej. el vídeo no se puede leer
            self.error = str(exc)
            log.warning("live[%s]: no se pudo procesar: %s", self.project_id[:8], exc)
            self.emit({"type": "error", "message": f"No se pudo leer el audio: {exc}"})
        finally:
            elapsed = time.monotonic() - self.started_at
            if subtitler is not None and not subtitler.stopped and not self.error:
                self.metrics["completed"] = elapsed
                log.info("live[%s]: completado en %.1fs", self.project_id[:8], elapsed)
                self.emit({"type": "completed", "failed": sorted(subtitler.failed)})
            with _RUNNERS_LOCK:
                if _RUNNERS.get(self.project_id) is self:
                    _RUNNERS.pop(self.project_id, None)
            self.done.set()

    _stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True
        if self.subtitler is not None:
            self.subtitler.stop()

    def _mark_first(self, event: dict[str, Any]) -> None:
        """Instrumentación: primera vez que ocurre cada etapa (sin contenido)."""
        if event.get("type") != "cues":
            return
        key = f"first_cues_{event.get('language')}"
        if event.get("cues") and key not in self.metrics:
            elapsed = time.monotonic() - self.started_at
            self.metrics[key] = elapsed
            log.info(
                "live[%s]: primer lote %s a los %.1fs (tramo %s ms)",
                self.project_id[:8], event.get("language"), elapsed, event.get("range"),
            )


_RUNNERS: dict[str, LiveRunner] = {}
_RUNNERS_LOCK = threading.Lock()


def get_runner(project_id: str) -> LiveRunner | None:
    with _RUNNERS_LOCK:
        return _RUNNERS.get(project_id)


def ensure_runner(
    project_id: str,
    build: Callable[[LiveRunner], LiveSubtitler],
    *,
    job: tuple[Any, str] | None = None,
) -> tuple[LiveRunner, bool]:
    """El proceso del proyecto: el que ya existe o uno nuevo. (runner, creado)."""
    with _RUNNERS_LOCK:
        current = _RUNNERS.get(project_id)
        if current is not None and not current.done.is_set():
            return current, False
        runner = LiveRunner(project_id, build)
        if job is not None:
            runner.attach_job(*job)
        _RUNNERS[project_id] = runner
    runner.start()
    return runner, True


def stop_all() -> None:
    """Pruebas y apagado: detiene todos los procesos en curso."""
    with _RUNNERS_LOCK:
        runners = list(_RUNNERS.values())
        _RUNNERS.clear()
    for runner in runners:
        runner.stop()


# --- Construcción con la base de datos y el vídeo reales ---------------------


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_reader(project: Any) -> Any:
    """Lector de audio por tramos del ORIGINAL, sin descargarlo entero.

    Usa la copia local si existe (recién subido o almacenamiento local); si el
    vídeo está en Storage, FFmpeg lee por rangos de la URL firmada, así que el
    primer tramo está disponible en segundos y no tras la descarga completa.
    """
    from app.core.config import get_settings
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver
    from app.infrastructure.audio.ffmpeg_segment_reader import FFmpegSegmentReader
    from app.presentation.api import media_store

    temp_dir = Path(str(get_settings().paths.data_temp_dir))
    pending = media_store.find_pending(temp_dir, str(project["id"]))
    if pending is not None:
        source: Any = pending
    elif media_store.is_remote(project["media_path"]):
        source = media_store.signed_url(project["media_path"], expires_in=6 * 3600)
        if not source:
            raise RuntimeError("Almacenamiento no disponible")
    else:
        source = Path(str(project["media_path"]))
    resolver = SubprocessFFmpegBinaryResolver()
    ok, ffmpeg_bin, _version = resolver.resolve("ffmpeg")
    if not ok:
        raise RuntimeError(ffmpeg_bin)
    keys = project.keys() if hasattr(project, "keys") else []
    known = int(project["duration_ms"] or 0) if "duration_ms" in keys else 0
    probe_bin = ""
    if known <= 0:
        probe_ok, probe_path, _probe_version = resolver.resolve("ffprobe")
        probe_bin = probe_path if probe_ok else ""
    return FFmpegSegmentReader(source, ffmpeg_bin=ffmpeg_bin, ffprobe_bin=probe_bin, duration_ms=known)


def load_existing(conn: Any, project_id: str) -> dict[str, tuple[list[dict[str, Any]], int]]:
    from app.presentation.api import db

    existing: dict[str, tuple[list[dict[str, Any]], int]] = {}
    for row in db.many(conn, "SELECT * FROM subtitles WHERE project_id=?", (project_id,)):
        try:
            cues = json.loads(row["cues_json"]) or []
        except (TypeError, json.JSONDecodeError):
            cues = []
        keys = row.keys() if hasattr(row, "keys") else []
        covered = int(row["covered_ms"] or 0) if "covered_ms" in keys else 0
        cues = [
            {"start_ms": int(c["start_ms"]), "end_ms": int(c["end_ms"]), "text": str(c.get("text", ""))}
            for c in cues
            if isinstance(c, dict)
        ]
        if covered <= 0 and cues:
            covered = max(c["end_ms"] for c in cues)
        existing[str(row["language"])] = (cues, covered)
    return existing


class JobTracker:
    """Refleja el proceso de un proyecto en su job: estado, etapas y progreso.

    Mismo contrato que el pipeline anterior: historial de etapas (PREPARING,
    EXTRACTING_AUDIO, una CUES_READY por lote del original…) con porcentaje que
    nunca retrocede, `last_cue_ms` y eventos `processing_progress` en el bus.
    """

    def __init__(self, conn: Any, job_id: str) -> None:
        self.conn = conn
        self.job_id = job_id
        self._lock = threading.Lock()
        self._percent = 0

    def stage(self, stage: str, percent: int, status: str = "PROCESSING", **extra: Any) -> None:
        from app.presentation.api import db, worker

        with self._lock:
            percent = max(self._percent, int(percent))
            self._percent = percent
            row = db.one(self.conn, "SELECT stage_history, status FROM jobs WHERE id=?", (self.job_id,))
            if row is None or str(row["status"]) in {"CANCELLED", "COMPLETED", "FAILED"}:
                return
            try:
                history = json.loads(str(row["stage_history"] or "[]"))
            except json.JSONDecodeError:
                history = []
            history.append({"stage": stage, "status": status, "progress": percent, "at": _utcnow()})
            db.run(
                self.conn,
                "UPDATE jobs SET status=?, progress=?, stage=?, stage_history=?, updated_at=? "
                "WHERE id=? AND status NOT IN ('CANCELLED','COMPLETED','FAILED')",
                (status, percent, stage, json.dumps(history, ensure_ascii=False), _utcnow(), self.job_id),
            )
            if "last_cue_ms" in extra:
                db.run(
                    self.conn,
                    "UPDATE jobs SET last_cue_ms=? WHERE id=? AND status!='CANCELLED'",
                    (int(extra["last_cue_ms"]), self.job_id),
                )
        worker.publish(
            self.job_id,
            {
                "type": "processing_progress",
                "job_id": self.job_id,
                "status": status,
                "stage": stage,
                "progress": percent,
                **({"covered_ms": int(extra["last_cue_ms"])} if "last_cue_ms" in extra else {}),
                **({"duration_ms": int(extra["duration_ms"])} if "duration_ms" in extra else {}),
            },
        )

    def on_cues(self, runner: "LiveRunner") -> None:
        subtitler = runner.subtitler
        if subtitler is None:
            return
        covered = subtitler._ranges(subtitler.source).prefix_end() if subtitler.source else 0
        self.stage(
            "CUES_READY",
            max(3, min(95, subtitler.progress())),
            "TRANSCRIBING",
            last_cue_ms=covered,
            duration_ms=subtitler.duration_ms,
        )


def start_for_project(conn: Any, project: Any, *, job_id: str | None = None) -> LiveRunner:
    """Arranca (o recupera) el proceso de un proyecto con los modelos reales."""
    from app.presentation.api import db

    project_id = str(project["id"])
    targets = [t.strip().lower() for t in str(project["target_languages"] or "").split(",") if t.strip()]

    def persist(language: str, cues: list[dict[str, Any]], covered_ms: int) -> None:
        rows = [
            {
                "id": cue.get("id") or segment_id(language, cue["start_ms"], cue["end_ms"]),
                "index": position,
                "start_ms": int(cue["start_ms"]),
                "end_ms": int(cue["end_ms"]),
                "text": str(cue["text"]),
                "speaker": "",
                "language": language,
                "state": "final",
            }
            for position, cue in enumerate(cues, start=1)
        ]
        db.run(
            conn,
            """
            INSERT INTO subtitles (project_id, language, cues_json, updated_at, covered_ms)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project_id, language) DO UPDATE SET
                cues_json=excluded.cues_json, updated_at=excluded.updated_at,
                covered_ms=excluded.covered_ms
            """,
            (project_id, language, json.dumps(rows, ensure_ascii=False), _utcnow(), int(covered_ms)),
        )

    def build(runner: LiveRunner) -> LiveSubtitler:
        tracker = runner.tracker
        if tracker is not None:
            tracker.stage("PREPARING", 1)
        reader = open_reader(project)
        runner.metrics["reader_ready"] = time.monotonic() - runner.started_at
        log.info("live[%s]: audio disponible a los %.1fs", project_id[:8], runner.metrics["reader_ready"])
        tracker = runner.tracker
        if tracker is not None:
            tracker.stage("EXTRACTING_AUDIO", 3)

        def emit(event: dict[str, Any]) -> None:
            if event.get("type") == "source_language":
                db.run(
                    conn,
                    "UPDATE projects SET source_language=? WHERE id=?",
                    (str(event["language"]), project_id),
                )
            runner.emit(event)

        return LiveSubtitler(
            callbacks=LiveCallbacks(
                read=reader.read,
                transcribe=model_transcribe,
                translate=model_translate,
                detect=model_detect,
                emit=emit,
                persist=persist,
            ),
            source_language=str(project["source_language"] or ""),
            duration_ms=int(reader.duration_ms),
            targets=targets,
            existing=load_existing(conn, project_id),
        )

    runner, created = ensure_runner(project_id, build, job=(conn, job_id) if job_id else None)
    if job_id and runner.job_id != job_id:
        runner.attach_job(conn, job_id)
    if created:
        log.info("live[%s]: proceso creado (job=%s)", project_id[:8], (job_id or "-")[:8])
        # Los modelos se cargan en paralelo con la primera lectura del audio.
        source = str(project["source_language"] or "")
        threading.Thread(
            target=warm, kwargs={"translation": any(t != source for t in targets)},
            name="live-warm", daemon=True,
        ).start()
    return runner


# --- Modelos compartidos por todo el proceso --------------------------------

_asr_lock = threading.Lock()
_translate_lock = threading.Lock()
_holder: dict[str, Any] = {}


def _asr() -> Any:
    with _asr_lock:
        if "asr" not in _holder:
            from app.infrastructure.audio.faster_whisper_asr_adapter import (
                FasterWhisperSmallASRAdapter,
            )

            adapter = FasterWhisperSmallASRAdapter(compute_type="int8")
            adapter._get_or_load_model()
            _holder["asr"] = adapter
        return _holder["asr"]


def _translator() -> Callable[[list[str], str, str], list[str]]:
    """Traductor por lotes: CTranslate2 si el modelo convertido existe.

    CTranslate2 (el motor de Whisper) carga M2M100 en ~1,5 s y traduce unas
    cuatro veces más rápido que PyTorch en CPU. Sin el modelo convertido se usa
    el adaptador PyTorch de siempre.
    """
    with _translate_lock:
        if "mt" not in _holder:
            from app.infrastructure.translation import m2m100_ct2_adapter

            if m2m100_ct2_adapter.is_available():
                engine = m2m100_ct2_adapter.M2M100CT2Translator()
                engine.load()
                _holder["mt"] = engine.translate_batch
            else:
                from app.core.config import get_settings
                from app.domain.value_objects.translation import TranslationLanguageCode
                from app.infrastructure.translation.m2m100_translation_adapter import (
                    M2M100TranslatorAdapter,
                )

                adapter = M2M100TranslatorAdapter(settings_getter=get_settings)
                adapter._get_engine()  # carga los pesos (caché de módulo compartida)

                def _pytorch(texts: list[str], source: str, target: str) -> list[str]:
                    return adapter.translate_batch(
                        texts,
                        source_language=TranslationLanguageCode(source),
                        target_language=TranslationLanguageCode(target),
                    )

                _holder["mt"] = _pytorch
        return _holder["mt"]


def model_transcribe(audio: Any, offset_ms: int, language: str | None) -> list[Any]:
    import numpy as np

    adapter = _asr()
    with _asr_lock:
        return adapter.transcribe_array(
            np.asarray(audio, dtype=np.float32), offset_ms=int(offset_ms), language=language
        )


def model_detect(audio: Any) -> str:
    import numpy as np

    adapter = _asr()
    with _asr_lock:
        model = adapter._get_or_load_model()
        _segments, info = model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=None,
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
        )
    return str(getattr(info, "language", "") or "").lower()


def model_translate(texts: list[str], source: str, target: str) -> list[str]:
    translate = _translator()
    with _translate_lock:
        return translate(texts, source, target)


def warm(*, translation: bool) -> None:
    """Carga los modelos antes de que se necesiten."""
    try:
        _asr()
        if translation:
            _translator()
    except Exception as exc:  # noqa: BLE001 - la primera petición real informará del fallo
        log.warning("live: no se pudieron precargar los modelos: %s", exc)


__all__ = [
    "LiveCallbacks",
    "LiveRunner",
    "LiveSubtitler",
    "Ranges",
    "ensure_runner",
    "get_runner",
    "model_detect",
    "model_transcribe",
    "model_translate",
    "segment_id",
    "start_for_project",
    "stop_all",
    "warm",
]
