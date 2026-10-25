"""Persistencia de la API: SQLite local o PostgreSQL (Supabase).

El motor completo sigue escribiendo SQL estándar con `?` como marcador, igual
que antes. Esta capa decide contra qué motor habla y traduce lo mínimo
imprescindible, de modo que el resto del código no cambia:

* **Sin `IA_DATABASE_URL`** → SQLite en `data/app.sqlite` (comportamiento
  histórico, y lo que usan las pruebas).
* **Con `IA_DATABASE_URL`** apuntando a `postgresql://…` → PostgreSQL.

Lo único específico de cada motor es lo que hay aquí: la apertura de conexión,
el marcador de parámetros, el script de creación y la lectura de columnas.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT

# Forma común de una fila: `sqlite3.Row` admite índice y clave, `dict` también.
Row = Any

_LOCK = threading.RLock()

POSTGRES_SCHEMES = ("postgres://", "postgresql://")

_ENV_LOADED = False

_SCHEMA = """
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

# Columnas añadidas después de la creación inicial de las tablas.
_PROJECT_COLUMNS = {
    "duration_ms": "INTEGER NOT NULL DEFAULT 0",
    "thumbnail_path": "TEXT NOT NULL DEFAULT ''",
    "updated_at": "TEXT NOT NULL DEFAULT ''",
}
_JOB_COLUMNS = {
    # Historial completo de etapas recorridas (JSON), para que la UI
    # pueda mostrar todo lo ocurrido aunque el sondeo llegue tarde.
    "stage_history": "TEXT NOT NULL DEFAULT '[]'",
    # 'progressive' (por ventanas, publica mientras procesa) | 'full'.
    "mode": "TEXT NOT NULL DEFAULT 'full'",
    # Último milisegundo de audio ya cubierto por subtítulos.
    "last_cue_ms": "INTEGER NOT NULL DEFAULT 0",
}


def _load_env_once() -> None:
    """Vuelca `.env` en el entorno del proceso, una sola vez.

    `pydantic-settings` lee `.env` para su propio modelo, pero NO lo exporta:
    sin esto, una `IA_DATABASE_URL` escrita en `.env` nunca llegaría a este
    módulo y la aplicación seguiría usando SQLite sin avisar.

    `override=False`: lo que ya esté definido en el entorno real manda.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = Path(PROJECT_ROOT) / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dependencia opcional
        return
    load_dotenv(path, override=False)


def database_url() -> str:
    """URL de PostgreSQL configurada, o cadena vacía para usar SQLite."""
    _load_env_once()
    return (
        os.environ.get("IA_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    ).strip()


def backend_name() -> str:
    """`postgres` o `sqlite`, según la configuración activa.

    `IA_DB_BACKEND=sqlite` fuerza SQLite aunque haya una URL configurada: es la
    salida de emergencia (y lo que usan las pruebas para no tocar datos reales).
    """
    forced = os.environ.get("IA_DB_BACKEND", "").strip().lower()
    if forced in {"sqlite", "postgres"}:
        return forced
    return "postgres" if database_url().startswith(POSTGRES_SCHEMES) else "sqlite"


def _translate(sql: str) -> str:
    """Convierte `?` (estilo SQLite) en `%s` (estilo psycopg).

    El SQL del proyecto no contiene `?` dentro de literales, así que la
    sustitución directa es segura.
    """
    return sql.replace("?", "%s")


class Connection:
    """Misma interfaz para SQLite y PostgreSQL.

    El resto del código usa `execute`, `executescript` y `commit`, y lee filas
    por nombre de columna. Esta clase garantiza que ambas cosas funcionen igual
    sobre los dos motores.
    """

    def __init__(self, raw: Any, backend: str) -> None:
        self.raw = raw
        self.backend = backend

    # -- ejecución -----------------------------------------------------
    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        if self.backend == "postgres":
            cursor = self.raw.cursor()
            cursor.execute(_translate(sql), params)
            return cursor
        return self.raw.execute(sql, params)

    def executescript(self, script: str) -> Any:
        if self.backend == "postgres":
            cursor = self.raw.cursor()
            for statement in script.split(";"):
                if statement.strip():
                    cursor.execute(statement)
            return cursor
        return self.raw.executescript(script)

    # -- transacción y ciclo de vida ------------------------------------
    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()

    def __repr__(self) -> str:
        return f"<Connection backend={self.backend}>"


def connect(target: Path | str, *, timeout: float = 15.0) -> Connection:
    """Abre la conexión según la configuración.

    Con `IA_DATABASE_URL` definida se ignora `target` y se usa PostgreSQL;
    sin ella, SQLite en esa ruta.
    """
    url = database_url()
    if backend_name() == "postgres" and url.startswith(POSTGRES_SCHEMES):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "IA_DATABASE_URL apunta a PostgreSQL pero falta el driver. "
                'Instálalo con: pip install "psycopg[binary]"'
            ) from exc
        try:
            raw = psycopg.connect(url, row_factory=dict_row, connect_timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001
            # Un mensaje útil: el fallo más común es la contraseña sin escapar.
            raise RuntimeError(
                f"No se pudo conectar a PostgreSQL ({type(exc).__name__}: {exc}). "
                "Revisa la URL: los caracteres especiales de la contraseña deben ir "
                "percent-encoded (@ → %40, # → %23, / → %2F)."
            ) from exc
        return Connection(raw, "postgres")

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(path, check_same_thread=False)
    raw.row_factory = sqlite3.Row
    return Connection(raw, "sqlite")


def _existing_columns(conn: Connection, table: str) -> set[str]:
    """Columnas presentes, consultando el catálogo de cada motor."""
    if conn.backend == "postgres":
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ? AND table_schema = current_schema()",
            (table,),
        ).fetchall()
        return {str(row["column_name"]) for row in rows}
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(row["name"]) for row in rows}


def migrate(conn: Connection) -> None:
    conn.executescript(_SCHEMA)
    _add_missing_columns(conn, "projects", _PROJECT_COLUMNS)
    _add_missing_columns(conn, "jobs", _JOB_COLUMNS)
    conn.commit()


def _add_missing_columns(
    conn: Connection, table: str, columns: dict[str, str]
) -> None:
    """Migración aditiva idempotente: añade columnas que falten."""
    existing = _existing_columns(conn, table)
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def one(conn: Connection, sql: str, params: tuple[Any, ...] = ()) -> Row | None:
    with _LOCK:
        return conn.execute(sql, params).fetchone()


def many(conn: Connection, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
    with _LOCK:
        return list(conn.execute(sql, params).fetchall())


def run(conn: Connection, sql: str, params: tuple[Any, ...] = ()) -> None:
    with _LOCK:
        conn.execute(sql, params)
        conn.commit()
