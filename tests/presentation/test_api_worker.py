"""Tests de la cola de trabajos: historial de etapas y recuperación tras reinicio."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from app.presentation.api import db, worker


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = db.connect(tmp_path / "jobs.sqlite")
    db.migrate(connection)
    yield connection
    connection.close()


def _seed(conn: sqlite3.Connection, job_id: str, status: str, project_id: str = "p1") -> None:
    db.run(
        conn,
        "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?,?,?,?,?)",
        ("u1", "U", "u@e.co", "x", "2026-01-01T00:00:00+00:00"),
    )
    db.run(
        conn,
        "INSERT OR IGNORE INTO projects (id, user_id, title, source_language, target_languages,"
        " status, media_path, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (project_id, "u1", "t", "en", "es", "PENDING", "x.mp4", "2026-01-01T00:00:00+00:00"),
    )
    db.run(
        conn,
        "INSERT INTO jobs (id, project_id, user_id, status, progress, stage, error,"
        " created_at, updated_at, stage_history) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (job_id, project_id, "u1", status, 0, status, "", "2026-01-01T00:00:00+00:00",
         "2026-01-01T00:00:00+00:00", "[]"),
    )


class TestMigrate:
    def test_crea_columnas_nuevas(self, conn: sqlite3.Connection) -> None:
        project_cols = {r["name"] for r in conn.execute("PRAGMA table_info(projects)")}
        job_cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        assert {"duration_ms", "thumbnail_path", "updated_at"} <= project_cols
        assert "stage_history" in job_cols

    def test_migracion_es_idempotente(self, conn: sqlite3.Connection) -> None:
        """Ejecutar migrate() de nuevo no debe fallar ni duplicar columnas."""
        db.migrate(conn)
        db.migrate(conn)
        names = [r["name"] for r in conn.execute("PRAGMA table_info(projects)")]
        assert names.count("duration_ms") == 1


class TestReconcileOrphans:
    @pytest.mark.parametrize(
        "status",
        ["PENDING", "QUEUED", "PROCESSING", "TRANSCRIBING", "TRANSLATING", "GENERATING_SUBTITLES"],
    )
    def test_reencola_estados_no_terminales(
        self, conn: sqlite3.Connection, status: str, monkeypatch
    ) -> None:
        _seed(conn, "job-1", status)
        # No se arranca el hilo real; se comprueba la reconciliación de estado.
        monkeypatch.setattr(worker, "enqueue", lambda _conn, job_id: None)
        result = worker.reconcile_orphans(conn)
        assert result["requeued"] == 1
        row = db.one(conn, "SELECT * FROM jobs WHERE id=?", ("job-1",))
        assert row is not None
        assert row["status"] == "PENDING"
        assert "reencolado" in row["error"]
        assert row["progress"] == 0

    @pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "CANCELLED", "PARTIAL"])
    def test_no_toca_los_terminales(
        self, conn: sqlite3.Connection, status: str, monkeypatch
    ) -> None:
        _seed(conn, "job-2", status)
        monkeypatch.setattr(worker, "enqueue", lambda _conn, job_id: None)
        result = worker.reconcile_orphans(conn)
        assert result["requeued"] == 0
        row = db.one(conn, "SELECT * FROM jobs WHERE id=?", ("job-2",))
        assert row is not None and row["status"] == status

    def test_un_job_colgado_no_queda_en_processing(self, conn: sqlite3.Connection, monkeypatch) -> None:
        _seed(conn, "job-3", "PROCESSING")
        monkeypatch.setattr(worker, "enqueue", lambda _conn, job_id: None)
        worker.reconcile_orphans(conn)
        orphans = db.many(
            conn,
            "SELECT id FROM jobs WHERE status IN ('PROCESSING','TRANSCRIBING','TRANSLATING')",
        )
        assert orphans == [], "ningún job puede quedarse procesando eternamente"


class TestStageHistoryIntegrity:
    def test_historial_se_puede_leer_como_json(self, conn: sqlite3.Connection) -> None:
        _seed(conn, "job-4", "QUEUED")
        history = [
            {"stage": "QUEUED", "status": "QUEUED", "progress": 0, "at": "2026-01-01T00:00:00+00:00"},
            {"stage": "VAD", "status": "PROCESSING", "progress": 16, "at": "2026-01-01T00:00:01+00:00"},
        ]
        db.run(
            conn,
            "UPDATE jobs SET stage_history=? WHERE id=?",
            (json.dumps(history), "job-4"),
        )
        row = db.one(conn, "SELECT stage_history FROM jobs WHERE id=?", ("job-4",))
        assert row is not None
        restored = json.loads(row["stage_history"])
        assert [entry["stage"] for entry in restored] == ["QUEUED", "VAD"]
        percents = [entry["progress"] for entry in restored]
        assert percents == sorted(percents)
