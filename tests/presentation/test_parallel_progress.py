from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from app.presentation.api import db
from app.presentation.api import live
from app.presentation.api import worker


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.connect(tmp_path / "app.sqlite")
    db.migrate(connection)
    yield connection
    connection.close()


def _job(conn, job_id: str, mode: str = "progressive") -> None:
    db.run(
        conn,
        "INSERT INTO jobs (id, project_id, user_id, status, progress, stage, error, created_at, updated_at, mode) "
        "VALUES (?, ?, 'u1', 'QUEUED', 0, '', '', ?, '', ?)",
        (job_id, f"p-{job_id}", job_id, mode),
    )
    db.run(conn, "UPDATE jobs SET status='PROCESSING' WHERE id=?", (job_id,))


def _subtitler() -> live.LiveSubtitler:
    callbacks = live.LiveCallbacks(
        read=lambda *a: None, transcribe=lambda *a: [], translate=lambda *a: [],
        detect=lambda *a: "en", emit=lambda e: None, persist=lambda *a, **k: None,
    )
    return live.LiveSubtitler(callbacks=callbacks, source_language="en", duration_ms=100_000, targets=["es"])


class _Runner:
    def __init__(self, subtitler: live.LiveSubtitler) -> None:
        self.subtitler = subtitler


def test_el_porcentaje_avanza_con_las_traducciones(conn) -> None:
    _job(conn, "j1")
    tracker = live.JobTracker(conn, "j1")
    sub = _subtitler()
    sub._ranges("en").add(0, 100_000)  # el idioma original ya está completo
    tracker.on_cues(_Runner(sub))
    first = db.one(conn, "SELECT progress FROM jobs WHERE id='j1'")["progress"]
    assert first == 50

    sub._ranges("es").add(0, 60_000)  # llegan traducciones
    tracker.on_cues(_Runner(sub))
    assert db.one(conn, "SELECT progress FROM jobs WHERE id='j1'")["progress"] == 80


def test_sin_cambios_no_se_escribe_en_la_base(conn) -> None:
    _job(conn, "j2")
    tracker = live.JobTracker(conn, "j2")
    sub = _subtitler()
    sub._ranges("en").add(0, 40_000)
    for _ in range(5):
        tracker.on_cues(_Runner(sub))
    history = json.loads(db.one(conn, "SELECT stage_history FROM jobs WHERE id='j2'")["stage_history"])
    assert len(history) == 1


def test_un_lote_traducido_tambien_avisa_del_progreso() -> None:
    calls: list[str] = []
    runner = live.LiveRunner("p1", build=lambda r: _subtitler())
    runner.subtitler = _subtitler()
    runner.on_progress = lambda r: calls.append("progreso")
    runner.emit({"type": "cues", "language": "es", "cues": []})
    runner.emit({"type": "cues", "language": "en", "cues": []})
    runner.emit({"type": "status", "language": "es"})
    assert calls == ["progreso", "progreso"]


def test_dos_videos_se_procesan_a_la_vez(conn, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []
    both = threading.Event()
    release = threading.Event()

    def fake_run(_conn, job_id: str) -> None:
        started.append(job_id)
        if len(started) == 2:
            both.set()
        release.wait(10)

    monkeypatch.setattr(worker, "_run_claimed", fake_run)
    monkeypatch.delenv("IA_DISABLE_WORKER", raising=False)
    for job_id in ("a", "b"):
        db.run(
            conn,
            "INSERT INTO jobs (id, project_id, user_id, status, progress, stage, error, created_at, updated_at, mode) "
            "VALUES (?, ?, 'u1', 'QUEUED', 0, '', '', ?, '', 'progressive')",
            (job_id, f"p-{job_id}", job_id),
        )
    worker.reset()
    worker.start(conn)
    try:
        worker.enqueue(conn, "a")
        worker.enqueue(conn, "b")
        assert both.wait(10), f"solo empezó {started}: el segundo esperaba al primero"
    finally:
        release.set()
        worker.reset()
