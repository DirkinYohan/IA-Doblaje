from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tests.fixtures.accounts import sign_up


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    from app.presentation.api.main import create_app

    with TestClient(create_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def _valid(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {"name": "Ana", "email": "ana@example.com", "password": "secreto123", "confirm": "secreto123"}
    body.update(changes)
    return body


@pytest.mark.parametrize(
    "path",
    ["/api/auth/login", "/api/auth/register", "/api/auth/forgot-password", "/api/auth/verify-email", "/api/auth/reset-password"],
)
@pytest.mark.parametrize("raw", ["{no es json", "[]", '"texto"', "42"])
def test_cuerpo_mal_formado_da_422_y_no_500(client: TestClient, path: str, raw: str) -> None:
    response = client.post(path, content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.json() == {"detail": "Datos inválidos"}


@pytest.mark.parametrize(
    "body",
    [
        _valid(name="a" * 10_000),
        _valid(name="a" * 81),
        _valid(email="a@"),
        _valid(email="sin-arroba.com"),
        _valid(email="dos@@dominio.com"),
        _valid(email="ana@dominio"),
        _valid(email="con espacio@dominio.com"),
        _valid(password="x" * 2000, confirm="x" * 2000),
        _valid(name=["lista"]),
        _valid(email=5),
    ],
)
def test_registro_rechaza_datos_invalidos(client: TestClient, body: dict[str, object]) -> None:
    assert client.post("/api/auth/register", json=body).status_code == 422


def test_registro_valido_sigue_funcionando(client: TestClient) -> None:
    response = client.post("/api/auth/register", json=_valid())
    assert response.status_code == 200
    assert response.json() == {"status": "verify", "email": "ana@example.com"}


def test_cambiar_correo_usa_la_misma_regla_que_el_registro(client: TestClient) -> None:
    sign_up(client, json=_valid())
    for email in ["a@", "ana@dominio", "x" * 250 + "@a.co"]:
        response = client.post("/api/auth/change-email", json={"email": email, "password": "secreto123"})
        assert response.status_code == 422, email


def test_subtitulos_mal_formados_dan_422(client: TestClient) -> None:
    sign_up(client, json=_valid())
    upload = client.post(
        "/api/videos",
        files={"file": ("a.wav", b"RIFF" + b"\0" * 2000, "audio/wav")},
        data={"source_language": "en", "target_languages": "es"},
    )
    project_id = upload.json()["id"]
    bad = client.put(f"/api/subtitles/{project_id}", json={"language": "es", "cues": [{"sin": "campos"}]})
    assert bad.status_code == 422
    not_list = client.put(f"/api/subtitles/{project_id}", json={"language": "es", "cues": "texto"})
    assert not_list.status_code == 422


def test_ninguna_ruta_http_bloquea_el_bucle_de_eventos(client: TestClient) -> None:
    async_routes = [
        route.path
        for route in client.app.routes
        if isinstance(route, APIRoute) and inspect.iscoroutinefunction(route.endpoint)
    ]
    assert async_routes == [], f"rutas HTTP async con trabajo bloqueante: {async_routes}"
