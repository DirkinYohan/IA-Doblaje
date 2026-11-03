from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.fixtures.mailbox import wait_for_mail


def sign_up(client: TestClient, json: dict[str, Any]) -> Any:
    registered = client.post("/api/auth/register", json=json)
    assert registered.status_code == 200, registered.text
    email = str(json["email"]).strip().lower()
    mail = wait_for_mail(client.app.state.paths["mail"], "verify", email)
    confirmed = client.post("/api/auth/verify-email", json={"token": mail["token"]})
    assert confirmed.status_code == 200, confirmed.text
    login = client.post("/api/auth/login", json={"email": email, "password": json["password"]})
    assert login.status_code == 200, login.text
    return login
