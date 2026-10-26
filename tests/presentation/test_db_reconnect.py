"""La conexión de PostgreSQL se reabre sola cuando la sesión se cae.

El pooler de Supabase cierra sesiones inactivas. Con una única conexión
compartida, eso dejaba la API entera devolviendo 500 («the connection is
closed») hasta reiniciarla: login incluido.
"""
from __future__ import annotations

import threading

import pytest
from psycopg import OperationalError

from app.presentation.api import db


class _Cursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def execute(self, sql: str, params: tuple | None = None) -> None:
        if "FAIL" in sql:
            raise OperationalError("server closed the connection unexpectedly")

    def fetchone(self) -> dict:
        return self._rows[0]

    def fetchall(self) -> list[dict]:
        return list(self._rows)


class _Sesion:
    """Sesión falsa: puede nacer ya cerrada o romperse al consultar."""

    def __init__(self, *, cerrada: bool = False, rompe_al_usar: bool = False) -> None:
        self.closed = cerrada
        self.broken = cerrada
        self._rompe = rompe_al_usar
        self.commits = 0

    def cursor(self):
        if self.closed or self.broken:
            raise OperationalError("the connection is closed")
        return _Cursor([{"n": 1}])

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def test_reabre_cuando_la_sesion_ya_esta_cerrada() -> None:
    creadas: list[_Sesion] = []

    def reconectar() -> _Sesion:
        nueva = _Sesion()
        creadas.append(nueva)
        return nueva

    conn = db.Connection(_Sesion(cerrada=True), "postgres", reconnect=reconectar)
    assert db.one(conn, "SELECT 1 AS n")["n"] == 1
    assert conn.reconnects == 1
    assert len(creadas) == 1


def test_reabre_cuando_la_sesion_muere_al_consultar() -> None:
    cerrrada_a_medias = _Sesion()
    conn = db.Connection(cerrrada_a_medias, "postgres", reconnect=_Sesion)

    # Primera consulta correcta.
    assert db.one(conn, "SELECT 1 AS n")["n"] == 1

    # El servidor cierra la sesión entre medias.
    cerrrada_a_medias.closed = True
    assert db.one(conn, "SELECT 1 AS n")["n"] == 1
    assert conn.reconnects == 1


def test_reintenta_solo_una_vez() -> None:
    """Si la consulta falla por otro motivo, no se insiste ni se reconecta."""
    conn = db.Connection(_Sesion(), "postgres", reconnect=_Sesion)
    with pytest.raises(OperationalError):
        db.one(conn, "SELECT FAIL")
    assert conn.reconnects == 0


def test_commit_sobre_sesion_cerrada_no_lanza() -> None:
    conn = db.Connection(_Sesion(cerrada=True), "postgres", reconnect=_Sesion)
    conn.commit()  # no debe propagar «the connection is closed»


def test_sin_reconector_el_error_es_claro() -> None:
    conn = db.Connection(_Sesion(cerrada=True), "postgres")
    with pytest.raises(RuntimeError, match="no puede reabrirla"):
        db.one(conn, "SELECT 1 AS n")


def test_sqlite_no_intenta_reconectar() -> None:
    import sqlite3

    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.row_factory = sqlite3.Row
    conn = db.Connection(raw, "sqlite")
    try:
        assert db.one(conn, "SELECT 1 AS n")["n"] == 1
        assert conn.reconnects == 0
    finally:
        conn.close()


def test_reconexion_concurrente_no_abre_varias() -> None:
    """Diez hilos que fallan a la vez deben provocar UNA reconexión."""
    creadas: list[_Sesion] = []
    cerrada = _Sesion(cerrada=True)

    def reconectar() -> _Sesion:
        nueva = _Sesion()
        creadas.append(nueva)
        return nueva

    conn = db.Connection(cerrada, "postgres", reconnect=reconectar)
    hilos = [threading.Thread(target=lambda: db.one(conn, "SELECT 1 AS n")) for _ in range(10)]
    for hilo in hilos:
        hilo.start()
    for hilo in hilos:
        hilo.join()
    assert len(creadas) == 1
