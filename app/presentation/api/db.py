from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


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
CREATE TABLE IF NOT EXISTS email_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    email TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS ws_tickets (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);
CREATE TABLE IF NOT EXISTS rate_hits (
    bucket TEXT NOT NULL,
    at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS rate_hits_bucket ON rate_hits (bucket, at);
CREATE TABLE IF NOT EXISTS live_leases (
    project_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    heartbeat_at DOUBLE PRECISION NOT NULL,
    position_ms INTEGER NOT NULL DEFAULT 0,
    position_lang TEXT NOT NULL DEFAULT '',
    position_at DOUBLE PRECISION NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS renders (
    project_id TEXT NOT NULL,
    language TEXT NOT NULL,
    state TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    stamp TEXT NOT NULL DEFAULT '',
    owner TEXT NOT NULL DEFAULT '',
    heartbeat_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    storage_key TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (project_id, language)
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

_USER_COLUMNS = {
    "email_verified_at": "TEXT NOT NULL DEFAULT ''",
}
_PROJECT_COLUMNS = {
    "duration_ms": "INTEGER NOT NULL DEFAULT 0",
    "thumbnail_path": "TEXT NOT NULL DEFAULT ''",
    "updated_at": "TEXT NOT NULL DEFAULT ''",
}
_JOB_COLUMNS = {
    "claimed_by": "TEXT NOT NULL DEFAULT ''",
    "heartbeat_at": "DOUBLE PRECISION NOT NULL DEFAULT 0",
    "stage_history": "TEXT NOT NULL DEFAULT '[]'",
    "mode": "TEXT NOT NULL DEFAULT 'full'",
    "last_cue_ms": "INTEGER NOT NULL DEFAULT 0",
}
_SUBTITLE_COLUMNS = {
    "covered_ms": "INTEGER NOT NULL DEFAULT 0",
    "covered_json": "TEXT NOT NULL DEFAULT '[]'",
}


def _load_env_once() -> None:
    from app.core.env import load_project_env

    load_project_env()


def database_url() -> str:
    _load_env_once()
    return (
        os.environ.get("IA_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    ).strip()


def backend_name() -> str:
    forced = os.environ.get("IA_DB_BACKEND", "").strip().lower()
    if forced in {"sqlite", "postgres"}:
        return forced
    return "postgres" if database_url().startswith(POSTGRES_SCHEMES) else "sqlite"


def _translate(sql: str) -> str:
    return sql.replace("?", "%s")


class Connection:
    def __init__(self, raw: Any, backend: str, reconnect: Any = None) -> None:
        self.raw = raw
        self.backend = backend
        self._reconnect = reconnect
        self._reconnect_lock = threading.Lock()
        self.reconnects = 0

    def _broken(self) -> bool:
        if self.backend != "postgres":
            return False
        raw = self.raw
        return (
            raw is None
            or bool(getattr(raw, "closed", False))
            or bool(getattr(raw, "broken", False))
        )

    @staticmethod
    def _is_disconnect(exc: BaseException) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        señales = (
            "connection is closed",
            "server closed the connection",
            "consuming input failed",
            "terminating connection",
            "connection lost",
            "connection already closed",
            "ssl connection has been closed",
        )
        return isinstance(exc, OSError) or any(s in text for s in señales)

    def _reopen(self) -> None:
        if self._reconnect is None:
            raise RuntimeError(
                "la conexión se cerró y este objeto no puede reabrirla"
            )
        with self._reconnect_lock:
            if self._broken():
                self.raw = self._reconnect()
                self.reconnects += 1

    def _execute_pg(self, sql: str, params: tuple[Any, ...]) -> Any:
        cursor = self.raw.cursor()
        cursor.execute(_translate(sql), params)
        return cursor

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        if self.backend != "postgres":
            return self.raw.execute(sql, params)
        try:
            return self._execute_pg(sql, params)
        except Exception as exc:  # noqa: BLE001
            if not self._is_disconnect(exc):
                raise
            self._reopen()
            return self._execute_pg(sql, params)

    def executescript(self, script: str) -> Any:
        if self.backend != "postgres":
            return self.raw.executescript(script)
        statements = [s for s in script.split(";") if s.strip()]

        def _run() -> Any:
            cursor = self.raw.cursor()
            for statement in statements:
                cursor.execute(statement)
            return cursor

        try:
            return _run()
        except Exception as exc:  # noqa: BLE001
            if not self._is_disconnect(exc):
                raise
            self._reopen()
            return _run()

    def commit(self) -> None:
        if self.backend == "postgres" and self._broken():
            return
        self.raw.commit()

    def rollback(self) -> None:
        if self.backend == "postgres" and self._broken():
            return
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()

    def __repr__(self) -> str:
        return f"<Connection backend={self.backend}>"


class _Rows:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[Any]:
        return list(self._rows)


def _dead(raw: Any) -> bool:
    return bool(getattr(raw, "broken", False) or getattr(raw, "closed", False))


class PooledConnection:
    backend = "postgres"

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Rows:
        import psycopg

        for attempt in (1, 2):
            retry = False
            with self.pool.connection() as raw:
                try:
                    cursor = raw.execute(_translate(sql), params)
                    return _Rows(cursor.fetchall() if cursor.description else [])
                except psycopg.OperationalError:
                    # Conexión cortada por el servidor mientras estaba en reposo: el pool
                    # la descarta al devolverla y se repite una vez con otra.
                    if attempt == 1 and _dead(raw):
                        retry = True
                    else:
                        raise
            if retry:
                self.pool.check()  # descarta también las demás conexiones cortadas
                continue
        raise RuntimeError("inalcanzable")

    def executescript(self, script: str) -> _Rows:
        with self.pool.connection() as raw:
            for statement in (s for s in script.split(";") if s.strip()):
                raw.execute(statement)
        return _Rows([])

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        self.pool.close()

    def __repr__(self) -> str:
        return f"<PooledConnection max_size={self.pool.max_size}>"


def _configure_session(raw: Any) -> None:
    with raw.cursor() as cursor:
        cursor.execute("SET statement_timeout = '60s'")
        cursor.execute("SET lock_timeout = '15s'")
    raw.commit()


IDLE_CHECK_S = 60.0
_LAST_USED: Any = None


def _check_if_idle(raw: Any) -> None:
    """Comprueba la conexión solo si llevaba un rato sin usarse.

    Comprobarla antes de cada consulta costaba un viaje de ida y vuelta extra a
    la base de datos (decenas de ms si está en otra región).
    """
    import time

    from psycopg_pool import ConnectionPool

    last = _LAST_USED.get(raw)
    if last is None or time.monotonic() - last > IDLE_CHECK_S:
        ConnectionPool.check_connection(raw)


def _mark_used(raw: Any) -> None:
    import time

    _LAST_USED[raw] = time.monotonic()


def _open_pool(url: str, timeout: float) -> PooledConnection:
    import weakref

    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    global _LAST_USED
    if _LAST_USED is None:
        _LAST_USED = weakref.WeakKeyDictionary()
    size = max(1, int(os.environ.get("IA_DB_POOL_SIZE", "5")))
    pool = ConnectionPool(
        url,
        min_size=1,
        max_size=size,
        timeout=timeout,
        kwargs={
            # Cada consulta suelta ya se confirmaba sola: en autocommit se evita
            # el BEGIN y el COMMIT extra (dos viajes menos). db.transaction()
            # sigue abriendo una transacción de verdad.
            "autocommit": True,
            "row_factory": dict_row,
            "connect_timeout": int(timeout),
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
        configure=_configure_session,
        check=_check_if_idle,
        reset=_mark_used,
        open=False,
    )
    pool.open(wait=True, timeout=timeout)
    return PooledConnection(pool)


def connect(target: Path | str, *, timeout: float = 15.0) -> Connection | PooledConnection:
    url = database_url()
    if backend_name() == "postgres" and url.startswith(POSTGRES_SCHEMES):
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError(
                "IA_DATABASE_URL apunta a PostgreSQL pero falta el driver. "
                'Instálalo con: pip install "psycopg[binary]"'
            ) from exc
        try:
            return _open_pool(url, timeout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"No se pudo conectar a PostgreSQL ({type(exc).__name__}: {exc}). "
                "Revisa la URL: los caracteres especiales de la contraseña deben ir "
                "percent-encoded (@ → %40, # → %23, / → %2F)."
            ) from exc

    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(path, check_same_thread=False)
    raw.row_factory = sqlite3.Row
    return Connection(raw, "sqlite")


def _existing_columns(conn: Connection, table: str) -> set[str]:
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
    _add_missing_columns(conn, "users", _USER_COLUMNS)
    _add_missing_columns(conn, "projects", _PROJECT_COLUMNS)
    _add_missing_columns(conn, "jobs", _JOB_COLUMNS)
    _add_missing_columns(conn, "subtitles", _SUBTITLE_COLUMNS)
    conn.commit()
    if conn.backend == "postgres":
        protect_tables(conn)


def schema_tables() -> list[str]:
    import re

    return re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", _SCHEMA)


def protect_tables(conn: Connection) -> list[str]:
    """Activa RLS en las tablas del backend (Supabase).

    Supabase publica una API REST para cada tabla de `public`; sin RLS, quien
    tenga la clave anónima podría leerlas o modificarlas. El backend se conecta
    como dueño de las tablas, así que RLS sin políticas no le afecta y cierra
    esa puerta. Devuelve las tablas que se han protegido ahora.
    """
    import logging

    log = logging.getLogger("ia.db")
    ours = set(schema_tables())
    rows = conn.execute(
        "SELECT tablename, tableowner = current_user AS owned, rowsecurity "
        "FROM pg_tables WHERE schemaname = current_schema()"
    ).fetchall()
    protected: list[str] = []
    for row in rows:
        name = str(row["tablename"])
        if name not in ours or row["rowsecurity"]:
            continue
        if not row["owned"]:
            log.warning("RLS: la tabla %s no es del usuario de la base; no se puede proteger", name)
            continue
        conn.execute(f'ALTER TABLE "{name}" ENABLE ROW LEVEL SECURITY')
        protected.append(name)
    conn.commit()
    if protected:
        log.warning("RLS activado en: %s", ", ".join(sorted(protected)))
    return protected


def _add_missing_columns(
    conn: Connection, table: str, columns: dict[str, str]
) -> None:
    existing = _existing_columns(conn, table)
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def one(conn: Connection | PooledConnection, sql: str, params: tuple[Any, ...] = ()) -> Row | None:
    if isinstance(conn, PooledConnection):
        return conn.execute(sql, params).fetchone()
    with _LOCK:
        return conn.execute(sql, params).fetchone()


def many(conn: Connection | PooledConnection, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
    if isinstance(conn, PooledConnection):
        return conn.execute(sql, params).fetchall()
    with _LOCK:
        return list(conn.execute(sql, params).fetchall())


def run(conn: Connection | PooledConnection, sql: str, params: tuple[Any, ...] = ()) -> None:
    if isinstance(conn, PooledConnection):
        conn.execute(sql, params)
        return
    with _LOCK:
        conn.execute(sql, params)
        conn.commit()


class Transaction:
    def __init__(self, execute: Any) -> None:
        self._execute = execute

    def run(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._execute(sql, params)

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> Row | None:
        return self._execute(sql, params).fetchone()

    def many(self, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
        return list(self._execute(sql, params).fetchall())


@contextmanager
def transaction(conn: Connection | PooledConnection) -> Iterator[Transaction]:
    if isinstance(conn, PooledConnection):
        import contextlib

        import psycopg

        for attempt in (1, 2):
            stack = contextlib.ExitStack()
            raw: Any = None
            try:
                raw = stack.enter_context(conn.pool.connection())
                stack.enter_context(raw.transaction())
            except psycopg.OperationalError:
                broken = raw is None or _dead(raw)
                stack.close()
                if attempt == 1 and broken:
                    conn.pool.check()
                    continue  # la conexión estaba muerta antes de empezar: se reintenta
                raise
            with stack:
                yield Transaction(lambda sql, params: raw.execute(_translate(sql), params))
            return
    with _LOCK:
        try:
            yield Transaction(conn.execute)
        except BaseException:
            conn.rollback()
            raise
        conn.commit()
