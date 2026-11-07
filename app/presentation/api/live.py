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

LOOKAHEAD_MS = 120_000
WINDOW_MS = 24_000
NEAR_WINDOW_MS = 12_000
EDGE_MS = 400
BACKOFF_MS = 1_500
TRANSLATE_CHUNK_MS = 30_000
MAX_FAILURES = 3


def segment_id(language: str, start_ms: int, end_ms: int) -> str:
    return f"{language}:{int(start_ms)}:{int(end_ms)}"


class Ranges:
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
        ends = [b for _a, b in self.items if b <= t]
        return max(ends, default=0)

    def first_gap(self, lo: int, hi: int) -> tuple[int, int] | None:
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
        if self.items and self.items[0][0] <= 0:
            return self.items[0][1]
        return 0

    def total(self) -> int:
        return sum(b - a for a, b in self.items)

    def as_list(self) -> list[list[int]]:
        return [[a, b] for a, b in self.items]


@dataclass
class LiveCallbacks:
    read: Callable[[int, int], Any]
    transcribe: Callable[[Any, int, str | None], list[Any]]
    translate: Callable[[list[str], str, str], list[str]]
    detect: Callable[[Any], str]
    emit: Callable[[dict[str, Any]], None]
    persist: Callable[..., None]


class LiveSubtitler:
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
        self.data_lock = threading.RLock()
        self._position = 0
        self._language = ""
        self._stopped = False
        self.failed: set[str] = set()

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
                    raise
                self.cb.emit({"type": "error", "message": str(exc)})
                worked = False
            if not worked:
                if until_complete and self.is_complete():
                    return
                with self._cond:
                    if not self._stopped:
                        self._cond.wait(timeout=1.0)

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

    def step(self) -> bool:
        with self._cond:
            position, language = self._position, self._language
        if not self.source:
            self._detect_source(position)
            if not self.source:
                return False
        language = language or next((t for t in self.targets if t != self.source), self.source)

        horizon = min(self.duration_ms, position + LOOKAHEAD_MS)
        if self._work_between(position, horizon, language, near=True):
            return True
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
            start = max(ranges.end_before(position), position - BACKOFF_MS, 0)
        size = NEAR_WINDOW_MS if near and start <= position + 1_000 else WINDOW_MS
        end = min(gap_end, start + size, self.duration_ms)
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
        self.cb.persist(language, persisted, prefix, event["covered"])


Listener = Callable[[dict[str, Any]], None]


class LiveRunner:
    def __init__(
        self, project_id: str, build: Callable[["LiveRunner"], LiveSubtitler], *, conn: Any = None
    ) -> None:
        self.project_id = project_id
        self._build = build
        self._conn = conn
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

    def start(self) -> None:
        threading.Thread(target=self._run, name=f"live-{self.project_id[:8]}", daemon=True).start()

    def _keep_lease(self, stop: threading.Event) -> None:
        from app.presentation.api import cluster

        applied = 0.0
        while not stop.wait(1.0):
            try:
                state = cluster.renew_lease(self._conn, self.project_id)
            except Exception:  # noqa: BLE001 - un fallo puntual de la base no corta el proceso
                continue
            if state is None:
                log.info("live[%s]: otro proceso tomó el proyecto o se borró; se detiene", self.project_id[:8])
                self.stop()
                return
            position, language, at = state
            if at > applied:
                applied = at
                self.set_position(position, language or None)

    def _run(self) -> None:
        log.info("live[%s]: procesamiento iniciado", self.project_id[:8])
        subtitler: LiveSubtitler | None = None
        lease_stop = threading.Event()
        if self._conn is not None:
            threading.Thread(
                target=self._keep_lease, args=(lease_stop,), name=f"lease-{self.project_id[:8]}", daemon=True
            ).start()
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
            lease_stop.set()
            if self._conn is not None:
                from app.presentation.api import cluster

                try:
                    cluster.release_lease(self._conn, self.project_id)
                except Exception:  # noqa: BLE001 - la concesión caduca sola
                    pass
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
    conn: Any = None,
) -> tuple[LiveRunner | None, bool]:
    with _RUNNERS_LOCK:
        current = _RUNNERS.get(project_id)
        if current is not None and not current.done.is_set():
            return current, False
        if conn is not None:
            from app.presentation.api import cluster

            if not cluster.acquire_lease(conn, project_id):
                return None, False
        runner = LiveRunner(project_id, build, conn=conn)
        if job is not None:
            runner.attach_job(*job)
        _RUNNERS[project_id] = runner
    runner.start()
    return runner, True


def stop_all() -> None:
    with _RUNNERS_LOCK:
        runners = list(_RUNNERS.values())
        _RUNNERS.clear()
    for runner in runners:
        runner.stop()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def media_source(project: Any) -> Any:
    from app.core.config import get_settings
    from app.presentation.api import media_store

    temp_dir = Path(str(get_settings().paths.data_temp_dir))
    pending = media_store.find_pending(temp_dir, str(project["id"]))
    if pending is not None:
        return pending
    if media_store.is_remote(project["media_path"]):
        url = media_store.signed_url(project["media_path"], expires_in=6 * 3600)
        if not url:
            raise RuntimeError("Almacenamiento no disponible")
        return url
    return Path(str(project["media_path"]))


def open_reader(project: Any) -> Any:
    from app.infrastructure.audio.ffmpeg_adapters import SubprocessFFmpegBinaryResolver
    from app.infrastructure.audio.ffmpeg_segment_reader import FFmpegSegmentReader

    source = media_source(project)
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


def stored_snapshot(conn: Any, project: Any) -> dict[str, Any]:
    from app.presentation.api import db

    project_id = str(project["id"])
    duration = int(project["duration_ms"] or 0)
    source = str(project["source_language"] or "")
    targets = [t for t in str(project["target_languages"] or "").split(",") if t]
    tracks: dict[str, list[dict[str, Any]]] = {}
    covered: dict[str, list[list[int]]] = {}
    for row in db.many(conn, "SELECT * FROM subtitles WHERE project_id=?", (project_id,)):
        code = str(row["language"])
        try:
            raw = json.loads(row["cues_json"]) or []
        except (TypeError, json.JSONDecodeError):
            raw = []
        cues = [
            {"start_ms": int(c["start_ms"]), "end_ms": int(c["end_ms"]), "text": str(c.get("text", ""))}
            for c in raw
            if isinstance(c, dict)
        ]
        tracks[code] = [dict(c, id=segment_id(code, c["start_ms"], c["end_ms"])) for c in cues]
        keys = row.keys() if hasattr(row, "keys") else []
        try:
            ranges = json.loads(row["covered_json"]) if "covered_json" in keys else []
        except (TypeError, json.JSONDecodeError):
            ranges = []
        if not ranges:
            prefix = int(row["covered_ms"] or 0) if "covered_ms" in keys else 0
            prefix = prefix or max((c["end_ms"] for c in cues), default=0)
            ranges = [[0, prefix]] if prefix else []
        covered[code] = [[int(a), int(b)] for a, b in ranges]
    languages = [source] + [t for t in targets if t != source]

    def full(code: str) -> bool:
        spans = covered.get(code) or []
        return bool(spans) and spans[0][0] <= 0 and spans[0][1] >= duration

    complete = duration > 0 and source not in {"", "und"} and all(full(code) for code in languages)
    return {
        "source_language": source, "duration_ms": duration, "tracks": tracks,
        "covered": covered, "failed": [], "complete": complete,
    }


class JobTracker:
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


def start_for_project(conn: Any, project: Any, *, job_id: str | None = None) -> LiveRunner | None:
    from app.presentation.api import db

    project_id = str(project["id"])
    targets = [t.strip().lower() for t in str(project["target_languages"] or "").split(",") if t.strip()]

    def persist(
        language: str, cues: list[dict[str, Any]], covered_ms: int, ranges: list[list[int]] | None = None
    ) -> None:
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
            INSERT INTO subtitles (project_id, language, cues_json, updated_at, covered_ms, covered_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, language) DO UPDATE SET
                cues_json=excluded.cues_json, updated_at=excluded.updated_at,
                covered_ms=excluded.covered_ms, covered_json=excluded.covered_json
            """,
            (
                project_id, language, json.dumps(rows, ensure_ascii=False), _utcnow(), int(covered_ms),
                json.dumps(ranges if ranges is not None else [[0, int(covered_ms)]]),
            ),
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

    runner, created = ensure_runner(project_id, build, job=(conn, job_id) if job_id else None, conn=conn)
    if runner is None:
        log.info("live[%s]: lo procesa otro proceso", project_id[:8])
        return None
    if job_id and runner.job_id != job_id:
        runner.attach_job(conn, job_id)
    if created:
        log.info("live[%s]: proceso creado (job=%s)", project_id[:8], (job_id or "-")[:8])
        source = str(project["source_language"] or "")
        threading.Thread(
            target=warm, kwargs={"translation": any(t != source for t in targets)},
            name="live-warm", daemon=True,
        ).start()
    return runner


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
                adapter._get_engine()

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
