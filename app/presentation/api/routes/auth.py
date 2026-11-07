from __future__ import annotations

import uuid
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from fastapi import APIRouter
from fastapi import Cookie
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import Response

from app.presentation.api import db
from app.presentation.api import mailer
from app.presentation.api.common import COOKIE
from app.presentation.api.common import MAX_NAME
from app.presentation.api.common import WS_TICKET_TTL_S
from app.presentation.api.common import _now
from app.presentation.api.context import ApiContext
from app.presentation.api.schemas import ChangeEmailBody
from app.presentation.api.schemas import ChangePasswordBody
from app.presentation.api.schemas import EmailBody
from app.presentation.api.schemas import LoginBody
from app.presentation.api.schemas import RegisterBody
from app.presentation.api.schemas import ResetPasswordBody
from app.presentation.api.schemas import TokenBody
from app.presentation.api.schemas import UpdateMeBody
from app.presentation.api.schemas import is_valid_email
from app.presentation.api.security import hash_password
from app.presentation.api.security import new_token
from app.presentation.api.security import token_hash
from app.presentation.api.security import verify_password


def build(ctx: ApiContext) -> APIRouter:
    app = ctx.app
    conn = ctx.conn
    client_key = ctx.client_key
    user_from_cookie = ctx.user_from_cookie
    issue_ws_ticket = ctx.issue_ws_ticket
    set_session = ctx.set_session
    public_user = ctx.public_user
    send_verification = ctx.send_verification
    router = APIRouter()

    @router.post("/api/auth/register")
    def register(request: Request, body: RegisterBody) -> dict[str, Any]:
        if not app.state.register_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiados registros")
        name = body.name.strip()
        email = body.email.strip().lower()
        password = body.password
        if not 2 <= len(name) <= MAX_NAME or not is_valid_email(email) or len(password) < 8 or password != body.confirm:
            raise HTTPException(status_code=422, detail="Datos de registro inválidos")
        locale = mailer.locale_of(body.locale)
        existing = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if existing is not None and existing["email_verified_at"]:
            raise HTTPException(status_code=409, detail="El email ya está registrado")
        if existing is not None:
            send_verification(existing["id"], existing["name"], email, locale)
            return {"status": "verify", "email": email}
        user_id = str(uuid.uuid4())
        db.run(
            conn,
            "INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, email, hash_password(password), _now()),
        )
        send_verification(user_id, name, email, locale)
        return {"status": "verify", "email": email}

    @router.post("/api/auth/login")
    def login(request: Request, response: Response, body: LoginBody) -> dict[str, Any]:
        if not app.state.login_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiados intentos")
        email = body.email.strip().lower()
        password = body.password
        remember = body.remember
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if row is None or not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Credenciales incorrectas")
        if not row["email_verified_at"]:
            raise HTTPException(status_code=403, detail="Confirma tu correo para iniciar sesión")
        set_session(response, row["id"], remember)
        return public_user(row)

    @router.post("/api/auth/logout")
    def logout(response: Response, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        if ia_session:
            db.run(conn, "DELETE FROM sessions WHERE token_hash=?", (token_hash(ia_session),))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @router.post("/api/auth/forgot-password")
    def forgot(request: Request, body: EmailBody) -> dict[str, str]:
        if not app.state.forgot_limit.allow(client_key(request)):
            raise HTTPException(status_code=429, detail="Demasiadas solicitudes")
        email = body.email.strip().lower()
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,))
        if row is not None:
            token = new_token()
            expires = datetime.now(timezone.utc) + timedelta(hours=1)
            db.run(
                conn,
                "INSERT INTO reset_tokens (token_hash, user_id, expires_at, used) VALUES (?, ?, ?, 0)",
                (token_hash(token), row["id"], expires.isoformat()),
            )
            mail = mailer.compose(
                "reset", to=email, name=row["name"], token=token, expires=expires.isoformat(),
                locale=mailer.locale_of(body.locale),
                link=f"{mailer.public_url()}/reset-password?token={token}",
            )
            mailer.send_in_background(app.state.mailer, mail)
        return {"status": "sent"}

    @router.post("/api/auth/verify-email")
    def verify_email(body: TokenBody) -> dict[str, Any]:
        token = body.token
        row = db.one(
            conn,
            "SELECT * FROM email_tokens WHERE token_hash=? AND used=0 AND expires_at > ?",
            (token_hash(token), _now()),
        ) if token else None
        user = db.one(conn, "SELECT * FROM users WHERE id=?", (row["user_id"],)) if row else None
        if row is None or user is None:
            raise HTTPException(status_code=400, detail="Enlace inválido o vencido")
        email = row["email"]
        if email != user["email"]:
            taken = db.one(conn, "SELECT id FROM users WHERE email=? AND id<>?", (email, user["id"]))
            if taken is not None:
                raise HTTPException(status_code=409, detail="El email ya está registrado")
        with db.transaction(conn) as tx:
            tx.run(
                "UPDATE users SET email=?, email_verified_at=? WHERE id=?",
                (email, user["email_verified_at"] or _now(), user["id"]),
            )
            tx.run("UPDATE email_tokens SET used=1 WHERE user_id=?", (user["id"],))
        return {"ok": True, "email": email}

    @router.post("/api/auth/resend-verification")
    def resend_verification(request: Request, body: EmailBody) -> dict[str, str]:
        email = body.email.strip().lower()
        if not app.state.verify_limit.allow(client_key(request)) or not app.state.verify_limit.allow(f"mail:{email}"):
            raise HTTPException(status_code=429, detail="Demasiadas solicitudes")
        row = db.one(conn, "SELECT * FROM users WHERE email=?", (email,)) if email else None
        if row is not None and not row["email_verified_at"]:
            send_verification(row["id"], row["name"], row["email"], mailer.locale_of(body.locale))
        return {"status": "sent"}

    @router.post("/api/auth/reset-password")
    def reset_password(body: ResetPasswordBody) -> dict[str, bool]:
        token = body.token
        password = body.password
        if len(password) < 8 or password != body.confirm or not token:
            raise HTTPException(status_code=422, detail="Contraseña inválida")
        row = db.one(
            conn,
            "SELECT * FROM reset_tokens WHERE token_hash=? AND used=0 AND expires_at > ?",
            (token_hash(token), _now()),
        )
        if row is None:
            raise HTTPException(status_code=400, detail="Token inválido o vencido")
        new_hash = hash_password(password)
        with db.transaction(conn) as tx:
            tx.run("UPDATE users SET password_hash=? WHERE id=?", (new_hash, row["user_id"]))
            tx.run("UPDATE users SET email_verified_at=? WHERE id=? AND email_verified_at=''", (_now(), row["user_id"]))
            tx.run("UPDATE reset_tokens SET used=1 WHERE token_hash=?", (token_hash(token),))
            tx.run("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        return {"ok": True}

    @router.post("/api/ws-ticket")
    def ws_ticket(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        return {"ticket": issue_ws_ticket(row["id"]), "expires_in": WS_TICKET_TTL_S}

    @router.get("/api/auth/me")
    def me(ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        return public_user(row)

    @router.post("/api/auth/change-email")
    def change_email(body: ChangeEmailBody, ia_session: str | None = Cookie(default=None)) -> dict[str, str]:
        row = user_from_cookie(ia_session)
        email = body.email.strip().lower()
        password = body.password
        if not is_valid_email(email):
            raise HTTPException(status_code=422, detail="Correo inválido")
        if not verify_password(password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        taken = db.one(conn, "SELECT id FROM users WHERE email=? AND id<>?", (email, row["id"]))
        if taken is not None:
            raise HTTPException(status_code=409, detail="El email ya está registrado")
        if email == row["email"]:
            return {"email": email, "pending_email": ""}
        send_verification(row["id"], row["name"], email, mailer.locale_of(body.locale))
        return {"email": row["email"], "pending_email": email}

    @router.post("/api/auth/change-password")
    def change_password(body: ChangePasswordBody, ia_session: str | None = Cookie(default=None)) -> dict[str, bool]:
        row = user_from_cookie(ia_session)
        password = body.password
        if not verify_password(body.current, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        if len(password) < 8 or password != body.confirm:
            raise HTTPException(status_code=422, detail="Contraseña inválida")
        db.run(conn, "UPDATE users SET password_hash=? WHERE id=?", (hash_password(password), row["id"]))
        return {"ok": True}

    @router.patch("/api/users/me")
    def update_me(body: UpdateMeBody, ia_session: str | None = Cookie(default=None)) -> dict[str, Any]:
        row = user_from_cookie(ia_session)
        if body.name is None:
            raise HTTPException(status_code=422, detail="Nada que actualizar")
        name = body.name.strip()
        if len(name) < 2 or len(name) > MAX_NAME:
            raise HTTPException(status_code=422, detail=f"El nombre debe tener entre 2 y {MAX_NAME} caracteres")
        db.run(conn, "UPDATE users SET name=? WHERE id=?", (name, row["id"]))
        return public_user(db.one(conn, "SELECT * FROM users WHERE id=?", (row["id"],)))

    return router
