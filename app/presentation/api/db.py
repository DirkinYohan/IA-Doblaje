"""SQLite local para usuarios, proyectos, jobs y subtítulos."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            remember INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS reset_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source_language TEXT NOT NULL,
            target_languages TEXT NOT NULL,
            status TEXT NOT NULL,
            media_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            stage TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS subtitles (
            project_id TEXT NOT NULL,
            language TEXT NOT NULL,
            cues_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (project_id, language)
        );
        """
    )
    _add_missing_columns(
        conn,
        "projects",
        {
            "duration_ms": "INTEGER NOT NULL DEFAULT 0",
            "thumbnail_path": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
        },
    )
    _add_missing_columns(
        conn,
        "jobs",
        {
            # Historial completo de etapas recorridas (JSON), para que la UI
            # pueda mostrar todo lo ocurrido aunque el sondeo llegue tarde.
            "stage_history": "TEXT NOT NULL DEFAULT '[]'",
            # 'progressive' (por ventanas, publica mientras procesa) | 'full'.
            "mode": "TEXT NOT NULL DEFAULT 'full'",
            # Último milisegundo de audio ya cubierto por subtítulos.
            "last_cue_ms": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    conn.commit()


def _add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Migración aditiva idempotente: añade columnas que falten."""
    existing = {
        str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with _LOCK:
        row = conn.execute(sql, params).fetchone()
    return row


def many(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with _LOCK:
        return list(conn.execute(sql, params).fetchall())


def run(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> None:
    with _LOCK:
        conn.execute(sql, params)
        conn.commit()
