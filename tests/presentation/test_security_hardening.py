from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.presentation.api import db, security
from tests.fixtures.accounts import sign_up

USER = {"name": "Ana", "email": "ana@example.com", "password": "secreto123", "confirm": "secreto123"}


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    monkeypatch.delenv("APP_PUBLIC_URL", raising=False)
    monkeypatch.delenv("IA_COOKIE_SECURE", raising=False)
    from app.presentation.api.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _legacy_hash(password: str) -> str:
    import hashlib
    import secrets

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, security.LEGACY_ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def test_hash_nuevo_usa_600000_iteraciones() -> None:
    stored = security.hash_password("clave-segura")
    scheme, iterations, _salt, _digest = stored.split("$")
    assert scheme == "pbkdf2_sha256" and int(iterations) == 600_000
    assert security.verify_password("clave-segura", stored)
    assert not security.verify_password("otra", stored)
    assert not security.needs_rehash(stored)


def test_hash_antiguo_sigue_valiendo_y_pide_actualizarse() -> None:
    legacy = _legacy_hash("clave-vieja")
    assert security.verify_password("clave-vieja", legacy)
    assert not security.verify_password("otra", legacy)
    assert security.needs_rehash(legacy)
    assert security.needs_rehash("basura-sin-formato")
    assert not security.verify_password("x", "basura-sin-formato")


def test_al_iniciar_sesion_se_actualiza_el_hash_antiguo(client: TestClient) -> None:
    sign_up(client, json=USER)
    conn = client.app.state.conn
    db.run(conn, "UPDATE users SET password_hash=? WHERE email=?", (_legacy_hash("secreto123"), USER["email"]))
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"email": USER["email"], "password": "secreto123"}).status_code == 200
    stored = db.one(conn, "SELECT password_hash FROM users WHERE email=?", (USER["email"],))["password_hash"]
    assert stored.startswith("pbkdf2_sha256$600000$")
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"email": USER["email"], "password": "secreto123"}).status_code == 200


def test_el_login_tarda_lo_mismo_exista_o_no_la_cuenta(client: TestClient) -> None:
    sign_up(client, json=USER)
    client.post("/api/auth/logout")

    def timed(email: str) -> float:
        best = float("inf")
        for _ in range(3):
            started = time.perf_counter()
            response = client.post("/api/auth/login", json={"email": email, "password": "incorrecta1"})
            best = min(best, time.perf_counter() - started)
            assert response.status_code == 401
        return best

    existing, missing = timed(USER["email"]), timed("nadie@example.com")
    assert missing > existing * 0.6, f"existe {existing * 1000:.0f} ms, no existe {missing * 1000:.0f} ms"


def test_cookie_sin_secure_en_desarrollo(client: TestClient) -> None:
    response = sign_up(client, json=USER)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie


def test_cookie_secure_cuando_la_app_va_por_https(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_PUBLIC_URL", "https://ia-doblaje.example.com")
    response = sign_up(client, json=USER)
    assert "secure" in response.headers["set-cookie"].lower()
    logout = client.post("/api/auth/logout")
    assert "secure" in logout.headers["set-cookie"].lower()


def test_cookie_secure_forzable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_PUBLIC_URL", "https://x.example.com")
    monkeypatch.setenv("IA_COOKIE_SECURE", "0")
    assert security.cookie_secure() is False
    monkeypatch.delenv("APP_PUBLIC_URL")
    monkeypatch.setenv("IA_COOKIE_SECURE", "1")
    assert security.cookie_secure() is True
