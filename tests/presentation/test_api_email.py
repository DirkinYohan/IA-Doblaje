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


def _login(client: TestClient, email: str = "ada@example.com", password: str = "secreto123"):
    return client.post("/api/auth/login", json={"email": email, "password": password})


def _confirm(client: TestClient, mail_dir: Path, email: str = "ada@example.com") -> None:
    mail = wait_for_mail(mail_dir, "verify", email)
    assert client.post("/api/auth/verify-email", json={"token": mail["token"]}).status_code == 200


def test_no_se_entra_hasta_confirmar_el_correo(client: TestClient, mail_dir: Path) -> None:
    assert _register(client) == {"status": "verify", "email": "ada@example.com"}
    assert client.get("/api/auth/me").status_code == 401
    assert "ia_session" not in client.cookies

    mail = wait_for_mail(mail_dir, "verify", "ada@example.com")
    assert mail["subject"] == "Confirma tu correo en IA Doblaje"
    assert mail["link"] == f"{PUBLIC}/verify-email?token={mail['token']}"
    assert "Hola Ada" in mail["html"] and mail["link"] in mail["html"]

    blocked = _login(client)
    assert blocked.status_code == 403 and "Confirma tu correo" in blocked.json()["detail"]
    assert client.get("/api/auth/me").status_code == 401
    assert _login(client, password="incorrecta1").status_code == 401

    ok = client.post("/api/auth/verify-email", json={"token": mail["token"]})
    assert ok.status_code == 200 and ok.json() == {"ok": True, "email": "ada@example.com"}
    assert client.get("/api/auth/me").status_code == 401, "confirmar no abre sesión: se entra con el login"
    assert client.post("/api/auth/verify-email", json={"token": mail["token"]}).status_code == 400, "un solo uso"

    login = _login(client)
    assert login.status_code == 200 and login.json()["email_verified"] is True
    assert client.get("/api/auth/me").json()["email_verified"] is True


def test_registrarse_otra_vez_sin_confirmar_reenvia_el_enlace(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    first = wait_for_mail(mail_dir, "verify")
    again = client.post("/api/auth/register", json={
        "name": "Otra Persona", "email": "ada@example.com", "password": "distinta123", "confirm": "distinta123",
    })
    assert again.status_code == 200
    second = wait_for_mail(mail_dir, "verify", count=2)
    assert second["token"] != first["token"]
    assert client.post("/api/auth/verify-email", json={"token": second["token"]}).status_code == 200
    assert _login(client, password="distinta123").status_code == 401
    assert _login(client).status_code == 200
    client.post("/api/auth/logout")
    assert client.post("/api/auth/register", json={
        "name": "Ada", "email": "ada@example.com", "password": "secreto123", "confirm": "secreto123",
    }).status_code == 409


def test_enlaces_invalidos(client: TestClient) -> None:
    _register(client)
    for token in ["", "inventado", "x" * 64]:
        assert client.post("/api/auth/verify-email", json={"token": token}).status_code == 400


def test_reenviar_sin_sesion_y_limite(client: TestClient, mail_dir: Path) -> None:
    _register(client, locale="en")
    first = wait_for_mail(mail_dir, "verify")
    assert first["subject"] == "Confirm your email for IA Doblaje"
    resent = client.post("/api/auth/resend-verification", json={"email": "ADA@example.com", "locale": "es"})
    assert resent.json() == {"status": "sent"}
    second = wait_for_mail(mail_dir, "verify", count=2)
    assert second["subject"] == "Confirma tu correo en IA Doblaje"
    assert client.post("/api/auth/verify-email", json={"token": first["token"]}).status_code == 200
    assert client.post("/api/auth/verify-email", json={"token": second["token"]}).status_code == 400

    before = len(read_mails(mail_dir))
    for email in ["ada@example.com", "nadie@example.com"]:
        assert client.post("/api/auth/resend-verification", json={"email": email}).json() == {"status": "sent"}
    assert len(read_mails(mail_dir)) == before

    for _ in range(2):
        assert client.post("/api/auth/resend-verification", json={"email": "otro@example.com"}).status_code == 200
    assert client.post("/api/auth/resend-verification", json={"email": "otro@example.com"}).status_code == 429


def test_cambiar_de_correo_se_aplica_al_confirmarlo(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    _confirm(client, mail_dir)
    assert _login(client).status_code == 200
    changed = client.post("/api/auth/change-email", json={"email": "ada@nuevo.com", "password": "secreto123"})
    assert changed.json() == {"email": "ada@example.com", "pending_email": "ada@nuevo.com"}
    assert client.get("/api/auth/me").json()["email"] == "ada@example.com"
    client.post("/api/auth/logout")
    assert _login(client).status_code == 200

    _confirm(client, mail_dir, "ada@nuevo.com")
    me = client.get("/api/auth/me").json()
    assert me["email"] == "ada@nuevo.com" and me["email_verified"] is True
    client.post("/api/auth/logout")
    assert _login(client).status_code == 401
    assert _login(client, "ada@nuevo.com").status_code == 200


def test_un_enlace_viejo_no_devuelve_la_cuenta_a_otra_direccion(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    _confirm(client, mail_dir)
    _login(client)
    client.post("/api/auth/change-email", json={"email": "uno@nuevo.com", "password": "secreto123"})
    client.post("/api/auth/change-email", json={"email": "dos@nuevo.com", "password": "secreto123"})
    one = wait_for_mail(mail_dir, "verify", "uno@nuevo.com")
    two = wait_for_mail(mail_dir, "verify", "dos@nuevo.com")
    assert client.post("/api/auth/verify-email", json={"token": two["token"]}).status_code == 200
    assert client.post("/api/auth/verify-email", json={"token": one["token"]}).status_code == 400
    assert client.get("/api/auth/me").json()["email"] == "dos@nuevo.com"


def test_no_se_puede_cambiar_a_un_correo_ocupado(client: TestClient, mail_dir: Path) -> None:
    _register(client, "otra@example.com")
    _confirm(client, mail_dir, "otra@example.com")
    _register(client)
    _confirm(client, mail_dir)
    _login(client)
    assert client.post(
        "/api/auth/change-email", json={"email": "otra@example.com", "password": "secreto123"}
    ).status_code == 409


def test_recuperar_contrasena_envia_el_enlace_y_confirma_el_correo(client: TestClient, mail_dir: Path) -> None:
    _register(client)
    assert client.post("/api/auth/forgot-password", json={"email": "ADA@example.com ", "locale": "es"}).status_code == 200
    mail = wait_for_mail(mail_dir, "reset", "ada@example.com")
    assert mail["subject"] == "Recupera tu contraseña de IA Doblaje"
    assert mail["link"] == f"{PUBLIC}/reset-password?token={mail['token']}"
    reset = client.post("/api/auth/reset-password", json={"token": mail["token"], "password": "nueva12345", "confirm": "nueva12345"})
    assert reset.status_code == 200
    login = _login(client, password="nueva12345")
    assert login.status_code == 200 and login.json()["email_verified"] is True


def test_correo_desconocido_misma_respuesta_y_sin_envio(client: TestClient, mail_dir: Path) -> None:
    response = client.post("/api/auth/forgot-password", json={"email": "nadie@example.com"})
    assert response.status_code == 200 and response.json() == {"status": "sent"}
    assert [m for m in read_mails(mail_dir) if m.get("kind") == "reset"] == []


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
    assert _register(client) == {"status": "verify", "email": "ada@example.com"}


def test_las_pruebas_nunca_envian_correos_reales(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.env import load_project_env
    from app.presentation.api.main import create_app

    load_project_env()
    import os

    assert os.environ.get("BREVO_API_KEY") == ""
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    assert isinstance(create_app().state.mailer, mailer.FileMailer)
