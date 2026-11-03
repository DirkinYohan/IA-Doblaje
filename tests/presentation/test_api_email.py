"""Correo de confirmación de cuenta y de recuperación de contraseña."""
from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.presentation.api import mailer
from tests.fixtures.mailbox import read_mails, wait_for_mail

PUBLIC = "https://ia-doblaje.example.app"


@pytest.fixture()
def mail_dir(tmp_path: Path) -> Path:
    return tmp_path / "mail"


@pytest.fixture()
def client(tmp_path: Path, mail_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(mail_dir))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    monkeypatch.setenv("APP_PUBLIC_URL", PUBLIC + "/")
    monkeypatch.setenv("BREVO_API_KEY", "")
    from app.core.config import clear_settings_cache
    from app.presentation.api.main import create_app

    clear_settings_cache()
    with TestClient(create_app()) as test_client:
        yield test_client


def _register(client: TestClient, email: str = "ada@example.com", locale: str = "es") -> dict:
    response = client.post("/api/auth/register", json={
        "name": "Ada Lovelace", "email": email, "password": "secreto123", "confirm": "secreto123", "locale": locale,
    })
    assert response.status_code == 200, response.text
    return response.json()


def test_al_registrarse_llega_el_correo_de_confirmacion(client: TestClient, mail_dir: Path) -> None:
    user = _register(client)
    assert user["email_verified"] is False
    assert client.get("/api/auth/me").json()["email_verified"] is False

    mail = wait_for_mail(mail_dir, "verify", "ada@example.com")
    assert mail["subject"] == "Confirma tu correo en IA Doblaje"
    assert mail["link"] == f"{PUBLIC}/verify-email?token={mail['token']}"
    assert "Hola Ada" in mail["html"] and mail["link"] in mail["html"]

    # El enlace funciona sin sesión (puede abrirse en otro navegador).
    client.post("/api/auth/logout")
    ok = client.post("/api/auth/verify-email", json={"token": mail["token"]})
    assert ok.status_code == 200 and ok.json() == {"ok": True, "email": "ada@example.com"}
    assert client.post("/api/auth/verify-email", json={"token": mail["token"]}).status_code == 400, "un solo uso"

    login = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "secreto123"})
    assert login.json()["email_verified"] is True
    assert client.get("/api/auth/me").json()["email_verified"] is True
    assert client.post("/api/auth/resend-verification", json={}).json() == {"status": "verified"}


def test_enlaces_invalidos(client: TestClient) -> None:
    _register(client)
    for token in ["", "inventado", "x" * 64]:
        assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 400


def test_reenviar_y_limite(client: TestClient, mail_dir: Path) -> None:
    _register(client, locale="en")
    first = wait_for_mail(mail_dir, "verify")
    assert first["subject"] == "Confirm your email for IA Doblaje"
    assert client.post("/api/auth/resend-verification", json={"locale": "es"}).json() == {"status": "sent"}
    second = wait_for_mail(mail_dir, "verify", count=2)
    assert second["token"] != first["token"]
    # Cualquiera de los enlaces vigentes confirma; después ninguno vale.
    assert client.post("/api/auth/verify-email", json={"token": first["token"]}).status_code == 200
    assert client.post("/api/auth/verify-email", json={"token": second["token"]}).status_code == 400

    _register(client, "otro@example.com")
    for _ in range(3):
        assert client.post("/api/auth/resend-verification", json={}).status_code == 200
    assert client.post("/api/auth/resend-verification", json={}).status_code == 429
    client.post("/api/auth/logout")
    assert client.post("/api/auth/resend-verification", json={}).status_code == 401


def test_cambiar_de_correo_obliga_a_confirmarlo(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    old = wait_for_mail(mail_dir, "verify", "ada@example.com")
    assert client.post("/api/auth/verify-email", json={"token": old["token"]}).status_code == 200
    # Otro enlace pendiente para la dirección vieja.
    client.post("/api/auth/change-email", json={"email": "ada@nuevo.com", "password": "secreto123"})
    assert client.get("/api/auth/me").json()["email_verified"] is False
    new = wait_for_mail(mail_dir, "verify", "ada@nuevo.com")
    assert client.post("/api/auth/verify-email", json={"token": new["token"]}).status_code == 200
    assert client.get("/api/auth/me").json() | {"email": "ada@nuevo.com", "email_verified": True} == client.get("/api/auth/me").json()


def test_enlace_de_una_direccion_anterior_no_confirma_la_nueva(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    old = wait_for_mail(mail_dir, "verify", "ada@example.com")
    client.post("/api/auth/change-email", json={"email": "ada@nuevo.com", "password": "secreto123"})
    assert client.post("/api/auth/verify-email", json={"token": old["token"]}).status_code == 400
    assert client.get("/api/auth/me").json()["email_verified"] is False


def test_recuperar_contrasena_envia_el_enlace(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    client.post("/api/auth/logout")
    assert client.post("/api/auth/forgot-password", json={"email": "ADA@example.com ", "locale": "es"}).status_code == 200
    mail = wait_for_mail(mail_dir, "reset", "ada@example.com")
    assert mail["subject"] == "Recupera tu contraseña de IA Doblaje"
    assert mail["link"] == f"{PUBLIC}/reset-password?token={mail['token']}"
    reset = client.post("/api/auth/reset-password", json={"token": mail["token"], "password": "nueva12345", "confirm": "nueva12345"})
    assert reset.status_code == 200
    assert client.post("/api/auth/login", json={"email": "ada@example.com", "password": "nueva12345"}).status_code == 200


def test_correo_desconocido_misma_respuesta_y_sin_envio(client: TestClient, mail_dir: Path) -> None:
    response = client.post("/api/auth/forgot-password", json={"email": "nadie@example.com"})
    assert response.status_code == 200 and response.json() == {"status": "sent"}
    assert [m for m in read_mails(mail_dir) if m.get("kind") == "reset"] == []


# --- Brevo -----------------------------------------------------------------

class _FakeBrevo:
    def __init__(self, status: int = 201) -> None:
        self.requests: list[tuple[dict, dict]] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((dict(self.headers), body))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"messageId":"<1@brevo>"}' if status < 300 else b'{"message":"sender not valid"}')

            def log_message(self, *_args) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v3/smtp/email"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _mail() -> mailer.Mail:
    return mailer.compose("verify", to="ada@example.com", name="Ada <b>", link=f"{PUBLIC}/verify-email?token=a&b",
                          token="tok", expires="2030-01-01")


def test_brevo_recibe_el_correo_completo() -> None:
    fake = _FakeBrevo()
    try:
        mailer.BrevoMailer("xkeysib-test", "no-reply@ia.app", "IA Doblaje", url=fake.url).send(_mail())
    finally:
        fake.close()
    headers, body = fake.requests[0]
    assert headers["api-key"] == "xkeysib-test"
    assert body["sender"] == {"name": "IA Doblaje", "email": "no-reply@ia.app"}
    assert body["to"] == [{"email": "ada@example.com", "name": "Ada <b>"}]
    assert body["subject"] == "Confirma tu correo en IA Doblaje"
    assert "token=a&amp;b" in body["htmlContent"], "el enlace va escapado en el HTML"
    assert "Ada &lt;b&gt;" not in body["htmlContent"] and "<b>" not in body["htmlContent"]
    assert f"{PUBLIC}/verify-email?token=a&b" in body["textContent"]
    assert body["tags"] == ["verify"]


def test_brevo_error_se_informa() -> None:
    fake = _FakeBrevo(status=400)
    try:
        with pytest.raises(RuntimeError, match="Brevo respondió 400"):
            mailer.BrevoMailer("k", "no-reply@ia.app", "IA", url=fake.url).send(_mail())
    finally:
        fake.close()


def test_configuracion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BREVO_API_KEY", "")
    assert isinstance(mailer.from_env(tmp_path), mailer.FileMailer)
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    monkeypatch.delenv("MAIL_FROM_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="MAIL_FROM_EMAIL"):
        mailer.from_env(tmp_path)
    monkeypatch.setenv("MAIL_FROM_EMAIL", "no-reply@ia.app")
    configured = mailer.from_env(tmp_path)
    assert isinstance(configured, mailer.BrevoMailer) and configured.sender_name == "IA Doblaje"
    monkeypatch.delenv("APP_PUBLIC_URL", raising=False)
    assert mailer.public_url() == "http://localhost:3000"


def test_un_fallo_de_envio_no_rompe_el_registro(client: TestClient) -> None:
    class Broken:
        def send(self, _mail: mailer.Mail) -> None:
            raise RuntimeError("caído")

    client.app.state.mailer = Broken()
    assert _register(client)["email_verified"] is False


def test_las_pruebas_nunca_envian_correos_reales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aunque `.env` tenga la clave de Brevo, la suite escribe a disco."""
    from app.core.env import load_project_env
    from app.presentation.api.main import create_app

    load_project_env()
    import os

    assert os.environ.get("BREVO_API_KEY") == ""
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    assert isinstance(create_app().state.mailer, mailer.FileMailer)
