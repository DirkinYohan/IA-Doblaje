from __future__ import annotations

import os
import time
import uuid
from typing import Any

from app.presentation.api import db

INSTANCE_ID = os.environ.get("IA_INSTANCE_ID", "").strip() or f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

HEARTBEAT_S = float(os.environ.get("IA_HEARTBEAT_S", "5"))
STALE_S = float(os.environ.get("IA_STALE_S", "30"))


TICKET_GRACE_S = 6 * 3600


def now() -> float:
    return time.time()


def stale_before() -> float:
    return now() - STALE_S


class DbRateLimiter:
    def __init__(self, conn: Any, name: str, limit: int, window_sec: int) -> None:
        self.conn = conn
        self.name = name
        self.limit = limit
        self.window_sec = window_sec

    def allow(self, key: str) -> bool:
        bucket = f"{self.name}:{key}"
        moment = now()
        with db.transaction(self.conn) as tx:
            tx.run("DELETE FROM rate_hits WHERE bucket=? AND at < ?", (bucket, moment - self.window_sec))
            row = tx.one("SELECT COUNT(*) AS n FROM rate_hits WHERE bucket=?", (bucket,))
            if int(row["n"]) >= self.limit:
                return False
            tx.run("INSERT INTO rate_hits (bucket, at) VALUES (?, ?)", (bucket, moment))
        return True


def issue_ticket(conn: Any, token_hash: str, user_id: str, ttl_s: float) -> None:
    moment = now()
    with db.transaction(conn) as tx:
        # Margen: una subida larga valida su ticket con la hora en que empezó.
        tx.run("DELETE FROM ws_tickets WHERE expires_at < ?", (moment - TICKET_GRACE_S,))
        tx.run(
            "INSERT INTO ws_tickets (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, moment + ttl_s),
        )


def consume_ticket(conn: Any, token_hash: str, *, at: float | None = None) -> str | None:
    """`at`: hora a la que llegó la petición (una subida larga no caduca a mitad)."""
    with db.transaction(conn) as tx:
        row = tx.one(
            "DELETE FROM ws_tickets WHERE token_hash=? RETURNING user_id, expires_at",
            (token_hash,),
        )
    moment = now() if at is None else min(float(at), now())
    if row is None or float(row["expires_at"]) < moment:
        return None
    return str(row["user_id"])


def acquire_lease(conn: Any, project_id: str) -> bool:
    moment = now()
    with db.transaction(conn) as tx:
        tx.run(
            """
            INSERT INTO live_leases (project_id, owner, heartbeat_at) VALUES (?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET owner=excluded.owner, heartbeat_at=excluded.heartbeat_at
            WHERE live_leases.owner = excluded.owner OR live_leases.heartbeat_at < ?
            """,
            (project_id, INSTANCE_ID, moment, moment - STALE_S),
        )
        row = tx.one("SELECT owner FROM live_leases WHERE project_id=?", (project_id,))
    return row is not None and row["owner"] == INSTANCE_ID


def renew_lease(conn: Any, project_id: str) -> tuple[int, str, float] | None:
    with db.transaction(conn) as tx:
        tx.run(
            "UPDATE live_leases SET heartbeat_at=? WHERE project_id=? AND owner=?",
            (now(), project_id, INSTANCE_ID),
        )
        row = tx.one(
            "SELECT position_ms, position_lang, position_at FROM live_leases WHERE project_id=? AND owner=?",
            (project_id, INSTANCE_ID),
        )
    if row is None:
        return None
    return int(row["position_ms"] or 0), str(row["position_lang"] or ""), float(row["position_at"] or 0)


def release_lease(conn: Any, project_id: str) -> None:
    db.run(conn, "DELETE FROM live_leases WHERE project_id=? AND owner=?", (project_id, INSTANCE_ID))


def lease_owner(conn: Any, project_id: str) -> str | None:
    row = db.one(
        conn,
        "SELECT owner FROM live_leases WHERE project_id=? AND heartbeat_at >= ?",
        (project_id, stale_before()),
    )
    return str(row["owner"]) if row is not None else None


def publish_position(conn: Any, project_id: str, time_ms: int, language: str | None) -> None:
    db.run(
        conn,
        "UPDATE live_leases SET position_ms=?, position_lang=?, position_at=? WHERE project_id=?",
        (int(time_ms), language or "", now(), project_id),
    )


__all__ = [
    "INSTANCE_ID",
    "DbRateLimiter",
    "acquire_lease",
    "consume_ticket",
    "issue_ticket",
    "lease_owner",
    "publish_position",
    "release_lease",
    "renew_lease",
]
