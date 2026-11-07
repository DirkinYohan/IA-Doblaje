from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Callable

from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi.responses import Response

from app.presentation.api import cluster, db
from app.presentation.api import mailer
from app.presentation.api.common import COOKIE
from app.presentation.api.common import WS_TICKET_TTL_S
from app.presentation.api.common import _now
from app.presentation.api.common import sqlite3_row
from app.presentation.api.security import cookie_secure
from app.presentation.api.security import new_token
from app.presentation.api.security import token_hash


@dataclass(frozen=True)
class ApiContext:
    app: Any
    conn: Any
    paths: dict[str, Path]
    settings: Any
    _warm_live_models: Callable[..., Any]
    client_key: Callable[..., Any]
    user_from_cookie: Callable[..., Any]
    issue_ws_ticket: Callable[..., Any]
    user_from_socket: Callable[..., Any]
    set_session: Callable[..., Any]
    public_user: Callable[..., Any]
    send_verification: Callable[..., Any]


def build_context(app: Any, conn: Any, paths: dict[str, Path], settings: Any) -> ApiContext:
    live_warm = {"started": os.environ.get("IA_LIVE_PRELOAD", "1") != "1"}

    def _warm_live_models() -> None:
        if live_warm["started"]:
            return
        live_warm["started"] = True

        from app.presentation.api import live

        threading.Thread(
            target=live.warm, kwargs={"translation": True}, name="live-preload", daemon=True
        ).start()

    def client_key(request: Request) -> str:
        return request.client.host if request.client else "local"

    def user_from_cookie(token: str | None) -> sqlite3_row:
        if not token:
            raise HTTPException(status_code=401, detail="No autenticado")
        row = db.one(
            conn,
            """
            SELECT users.* FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash=? AND sessions.expires_at > ?
            """,
            (token_hash(token), _now()),
        )
        if row is None:
            raise HTTPException(status_code=401, detail="Sesión inválida")
        return row

    def issue_ws_ticket(user_id: str) -> str:
        ticket = new_token()
        cluster.issue_ticket(conn, token_hash(ticket), user_id, WS_TICKET_TTL_S)
        return ticket

    def user_from_socket(socket: WebSocket) -> sqlite3_row:
        ticket = socket.query_params.get("ticket") or ""
        if ticket:
            user_id = cluster.consume_ticket(conn, token_hash(ticket))
            if user_id is None:
                raise HTTPException(status_code=401, detail="Ticket inválido o caducado")
            row = db.one(conn, "SELECT * FROM users WHERE id=?", (user_id,))
            if row is None:
                raise HTTPException(status_code=401, detail="Sesión inválida")
            return row
        return user_from_cookie(socket.cookies.get(COOKIE))

    def set_session(response: Response, user_id: str, remember: bool) -> None:
        token = new_token()
        hours = 24 * 14 if remember else 12
        expires = datetime.now(timezone.utc) + timedelta(hours=hours)
        db.run(
            conn,
            "INSERT INTO sessions (token_hash, user_id, expires_at, remember) VALUES (?, ?, ?, ?)",
            (token_hash(token), user_id, expires.isoformat(), int(remember)),
        )
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            secure=cookie_secure(),
            samesite="lax",
            max_age=hours * 3600,
            path="/",
        )

    def public_user(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "created_at": row["created_at"],
            "email_verified": bool(row["email_verified_at"]),
        }

    def send_verification(user_id: str, name: str, email: str, locale: str) -> None:
        token = new_token()
        expires = datetime.now(timezone.utc) + timedelta(hours=48)
        db.run(
            conn,
            "INSERT INTO email_tokens (token_hash, user_id, email, expires_at, used) VALUES (?, ?, ?, ?, 0)",
            (token_hash(token), user_id, email, expires.isoformat()),
        )
        mail = mailer.compose(
            "verify", to=email, name=name, token=token, expires=expires.isoformat(), locale=locale,
            link=f"{mailer.public_url()}/verify-email?token={token}",
        )
        mailer.send_in_background(app.state.mailer, mail)

    return ApiContext(
        app=app, conn=conn, paths=paths, settings=settings,
        _warm_live_models=_warm_live_models,
        client_key=client_key,
        user_from_cookie=user_from_cookie,
        issue_ws_ticket=issue_ws_ticket,
        user_from_socket=user_from_socket,
        set_session=set_session,
        public_user=public_user,
        send_verification=send_verification,
    )
