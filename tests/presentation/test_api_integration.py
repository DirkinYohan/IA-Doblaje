"""Tests de integración de la API: subida por streaming, jobs, exportación.

Ejercitan la app real (FastAPI + SQLite en tmp) contra ficheros de medios
reales generados con FFmpeg. Sin mocks del pipeline.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT
from app.presentation.api import db

pytestmark = pytest.mark.integration

FFMPEG = Path(
    os.environ.get("LOCALAPPDATA", "")
) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"


def _ffmpeg(args: list[str], target: Path) -> bool:
    binary = str(FFMPEG) if FFMPEG.is_file() else "ffmpeg"
    try:
        done = subprocess.run(
            [binary, "-hide_banner", "-loglevel", "error", "-y", *args, str(target)],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and target.is_file() and target.stat().st_size > 0


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Vídeo de 3 s con pista de audio real + WAV de silencio."""
    folder = tmp_path_factory.mktemp("media")
    video = folder / "clip.mp4"
    if not _ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
        ],
        video,
    ):
        pytest.skip("FFmpeg no disponible para generar medios de prueba")
    silent = folder / "silence.wav"
    _ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", "-c:a", "pcm_s16le"], silent)
    return {"video": video, "silent": silent}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory):
    """App real con base de datos, correo y almacenamiento aislados."""
    from app.presentation.api import worker

    worker.reset()
    tmp = tmp_path_factory.mktemp("api")
    os.environ["IA_DB_PATH"] = str(tmp / "app.sqlite")
    os.environ["IA_MAIL_DIR"] = str(tmp / "mail")
    # El worker se desactiva: los tests comprueban el contrato HTTP, no la IA.
    os.environ["IA_DISABLE_WORKER"] = "1"
    os.environ["CORS_ORIGINS"] = "http://localhost:3000"

    from app.presentation.api.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    os.environ.pop("IA_DISABLE_WORKER", None)
    worker.reset()


@pytest.fixture(autouse=True)
def _unlimited_rate_limits(client: TestClient):
    """Los límites por IP son 5-10/min y la suite hace muchos registros."""
    from app.presentation.api.security import RateLimiter

    app = client.app
    app.state.register_limit = RateLimiter(10_000, 1)
    app.state.login_limit = RateLimiter(10_000, 1)
    app.state.forgot_limit = RateLimiter(10_000, 1)
    yield


@pytest.fixture(scope="module")
def auth(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Prueba Integración",
            "email": "integracion@example.com",
            "password": "password123",
            "confirm": "password123",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _login(client: TestClient, email: str, password: str = "password123") -> None:
    """Vuelve a autenticar la sesión del cliente de pruebas."""
    response = client.post("/api/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text


class TestAuthRequirements:
    def test_registro_valida_datos(self, client: TestClient) -> None:
        for body in (
            {"name": "x", "email": "a@b.co", "password": "password123", "confirm": "password123"},
            {"name": "Valido", "email": "sin-arroba", "password": "password123", "confirm": "password123"},
            {"name": "Valido", "email": "v@b.co", "password": "corta", "confirm": "corta"},
            {"name": "Valido", "email": "v2@b.co", "password": "password123", "confirm": "otracosa1"},
        ):
            assert client.post("/api/auth/register", json=body).status_code == 422

    def test_email_duplicado(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            "/api/auth/register",
            json={
                "name": "Otro",
                "email": auth["email"],
                "password": "password123",
                "confirm": "password123",
            },
        )
        assert response.status_code == 409

    def test_rutas_protegidas_sin_sesion(self) -> None:
        from app.presentation.api.main import create_app

        app = create_app()
        with TestClient(app) as anon:
            assert anon.get("/api/videos").status_code == 401
            assert anon.get("/api/auth/me").status_code == 401
            assert anon.patch("/api/users/me", json={"name": "X Y"}).status_code == 401
            assert anon.post("/api/jobs", json={}).status_code == 401
            assert anon.get("/api/system/preflight").status_code == 401
            assert anon.get("/api/subtitles/inexistente").status_code == 401
            assert anon.get("/api/export/inexistente/srt", params={"lang": "es"}).status_code == 401

    def test_login_logout(self, client: TestClient, auth: dict[str, str]) -> None:
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401
        response = client.post(
            "/api/auth/login",
            json={"email": auth["email"], "password": "password123"},
        )
        assert response.status_code == 200
        assert client.get("/api/auth/me").status_code == 200
        bad = client.post("/api/auth/login", json={"email": auth["email"], "password": "mal"})
        assert bad.status_code == 401

    def test_recuperacion_y_restablecimiento(self, client: TestClient, auth: dict[str, str]) -> None:
        mail_dir = Path(os.environ["IA_MAIL_DIR"])
        response = client.post("/api/auth/forgot-password", json={"email": auth["email"]})
        assert response.status_code == 200
        files = list(mail_dir.glob("*.txt"))
        assert files, "no se escribió el correo de recuperación"
        token = ""
        for line in files[-1].read_text(encoding="utf-8").splitlines():
            if line.startswith("token="):
                token = line.split("=", 1)[1].strip()
        assert token, "el correo no contiene token"
        reset = client.post(
            "/api/auth/reset-password",
            json={"token": token, "password": "nuevaclave123", "confirm": "nuevaclave123"},
        )
        assert reset.status_code == 200
        # El token es de un solo uso.
        again = client.post(
            "/api/auth/reset-password",
            json={"token": token, "password": "otraclave123", "confirm": "otraclave123"},
        )
        assert again.status_code == 400
        assert client.get("/api/auth/me").status_code == 401, "el reset invalida sesiones"
        # Se restaura la contraseña original para el resto de la suite.
        login = client.post("/api/auth/login", json={"email": auth["email"], "password": "nuevaclave123"})
        assert login.status_code == 200
        changed = client.post(
            "/api/auth/change-password",
            json={"current": "nuevaclave123", "password": "password123", "confirm": "password123"},
        )
        assert changed.status_code == 200


class TestUserProfile:
    def test_patch_nombre(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.patch("/api/users/me", json={"name": "Nombre Nuevo"})
        assert response.status_code == 200, response.text
        assert response.json()["name"] == "Nombre Nuevo"
        assert client.get("/api/auth/me").json()["name"] == "Nombre Nuevo"

    def test_patch_nombre_invalido(self, client: TestClient, auth: dict[str, str]) -> None:
        assert client.patch("/api/users/me", json={"name": "x"}).status_code == 422
        assert client.patch("/api/users/me", json={"name": "  "}).status_code == 422
        assert client.patch("/api/users/me", json={"name": "a" * 81}).status_code == 422
        assert client.patch("/api/users/me", json={}).status_code == 422

    def test_cambio_de_correo(self, client: TestClient, auth: dict[str, str]) -> None:
        bad = client.post(
            "/api/auth/change-email",
            json={"email": "nuevo@example.com", "password": "incorrecta"},
        )
        assert bad.status_code == 401
        ok = client.post(
            "/api/auth/change-email",
            json={"email": "nuevo@example.com", "password": "password123"},
        )
        assert ok.status_code == 200
        assert client.get("/api/auth/me").json()["email"] == "nuevo@example.com"
        # Se restaura para los demás tests.
        assert client.post(
            "/api/auth/change-email",
            json={"email": auth["email"], "password": "password123"},
        ).status_code == 200

    def test_preflight_refleja_el_motor(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.get("/api/system/preflight")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["items"], list) and data["items"]
        components = {item["component"]: item for item in data["items"]}
        for name in ("torch", "FFmpeg", "FFprobe", "VAD model (silero_vad_v5.1.jit)"):
            assert name in components, f"falta el check {name}"


class TestStreamingUpload:
    def test_subida_por_streaming_guarda_el_archivo(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        _login(client, auth["email"])
        payload = media["video"].read_bytes()
        with media["video"].open("rb") as handle:
            response = client.post(
                "/api/videos",
                files={"file": ("clip.mp4", handle, "video/mp4")},
                data={"source_language": "und", "target_languages": "es,en"},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["title"] == "clip.mp4"
        assert "job_id" in body

        detail = client.get(f"/api/videos/{body['id']}").json()
        video = detail["video"]
        # Duración medida con ffprobe en el servidor, no en el cliente.
        assert video["duration_ms"] > 2000, video
        # Metadatos para la tarjeta, sin <video> oculto.
        assert "thumbnail_url" in video
        assert video["source_language"] in {"und", "es", "en"}
        assert video["target_languages"] == ["es", "en"]

        media_response = client.get(f"/api/videos/{body['id']}/media")
        assert media_response.status_code == 200
        assert len(media_response.content) == len(payload), "el archivo subido quedó corrupto"

    def test_miniatura_generada(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("thumb.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        video = client.get(f"/api/videos/{created['id']}").json()["video"]
        if not video["thumbnail_url"]:
            pytest.skip("FFmpeg no pudo generar la miniatura en este entorno")
        thumbnail = client.get(video["thumbnail_url"])
        assert thumbnail.status_code == 200
        assert thumbnail.headers["content-type"].startswith("image/")
        assert thumbnail.content[:2] == b"\xff\xd8", "no es un JPEG"

    def test_nombre_de_archivo_se_sanea(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        with media["video"].open("rb") as handle:
            response = client.post(
                "/api/videos",
                files={"file": ("../../evil name;rm.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            )
        assert response.status_code == 200
        title = response.json()["title"]
        assert "/" not in title and "\\" not in title and ".." not in title
        assert " " not in title and ";" not in title

    def test_rechaza_extension_no_permitida(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            "/api/videos",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
            data={"source_language": "en", "target_languages": "es"},
        )
        assert response.status_code == 422

    def test_rechaza_idioma_no_soportado(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        with media["video"].open("rb") as handle:
            response = client.post(
                "/api/videos",
                files={"file": ("lang.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "xx"},
            )
        assert response.status_code == 422

    def test_rechaza_archivo_vacio(self, client: TestClient, auth: dict[str, str]) -> None:
        response = client.post(
            "/api/videos",
            files={"file": ("vacio.mp4", b"", "video/mp4")},
            data={"source_language": "en", "target_languages": "es"},
        )
        assert response.status_code == 422


class TestJobStates:
    def test_el_upload_no_arranca_el_procesamiento(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        """Subir deja el trabajo listo; producir es una acción aparte."""
        _login(client, auth["email"])
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("ready.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        job = client.get(f"/api/jobs/{created['job_id']}").json()
        assert job["status"] == "PENDING", (
            f"subir no debe procesar; el job quedó en {job['status']}"
        )
        assert job["mode"] == "progressive"
        project = client.get(f"/api/videos/{created['id']}").json()["video"]
        assert project["status"] == "READY", f"el proyecto debería estar READY ({project['status']})"

    def test_producir_arranca_y_con_worker_apagado_falla_explicito(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        """«Producir» encola de verdad; si el worker está apagado, lo dice."""
        _login(client, auth["email"])
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("job.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        produce = client.post(f"/api/videos/{created['id']}/produce", json={})
        assert produce.status_code == 200
        assert produce.json()["started"] is True
        assert produce.json()["id"] == created["job_id"], "reutiliza el job preparado al subir"

        progress = client.get(f"/api/jobs/{created['job_id']}/progress").json()
        assert progress["status"] == "FAILED", (
            f"con el worker apagado debe fallar, no quedarse en {progress['status']}"
        )
        assert "worker" in progress["error"].lower()
        assert progress["progress"] == 0

    def test_producir_reutiliza_el_job_preparado_al_subir(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        """Sin trabajo en curso, Producir arranca el job ya preparado.

        No debe crear uno nuevo: con el worker apagado el primer intento falla
        al instante, así que se vuelve a dejar el trabajo en PENDING para
        comprobar de forma determinista que se reutiliza (no se duplica).
        """
        _login(client, auth["email"])
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("reuse.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        project_id = created["id"]
        conn = client.app.state.conn

        first = client.post(f"/api/videos/{project_id}/produce", json={}).json()
        assert first["reused"] is True
        assert first["id"] == created["job_id"]
        assert first["started"] is True

        before = len(client.get(f"/api/videos/{project_id}").json()["jobs"])
        # Se devuelve el trabajo a PENDING y se vuelve a producir.
        db.run(
            conn,
            "UPDATE jobs SET status='PENDING', progress=0, stage='PENDING', error='' WHERE id=?",
            (created["job_id"],),
        )
        second = client.post(f"/api/videos/{project_id}/produce", json={}).json()
        assert second["id"] == created["job_id"], "debe reutilizar el job pendiente"
        after = len(client.get(f"/api/videos/{project_id}").json()["jobs"])
        assert after == before, "Producir no debe acumular trabajos"

    def test_progreso_es_monotono_en_el_historial(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        _login(client, auth["email"])
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("mono.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        client.post(f"/api/videos/{created['id']}/produce", json={})
        progress = client.get(f"/api/jobs/{created['job_id']}/progress").json()
        assert 0 <= progress["progress"] <= 100
        assert progress["stage"], "toda etapa debe tener nombre"
        for entry in progress.get("stage_history") or []:
            assert 0 <= int(entry["progress"]) <= 100

    def test_cancelar_tiene_estado_propio(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("cancel.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        cancelled = client.post(f"/api/jobs/{created['job_id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "CANCELLED"
        assert client.get(f"/api/jobs/{created['job_id']}/progress").json()["status"] == "CANCELLED"

    def test_job_de_otro_usuario_no_es_visible(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        with media["video"].open("rb") as handle:
            created = client.post(
                "/api/videos",
                files={"file": ("privado.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            ).json()
        client.post("/api/auth/logout")
        other = client.post(
            "/api/auth/register",
            json={
                "name": "Intruso",
                "email": "intruso@example.com",
                "password": "password123",
                "confirm": "password123",
            },
        )
        assert other.status_code == 200, other.text
        assert client.get(f"/api/jobs/{created['job_id']}").status_code == 404
        assert client.get(f"/api/videos/{created['id']}").status_code == 404
        # Se recupera la sesión original.
        client.post("/api/auth/logout")
        _login(client, auth["email"])


class TestSubtitlesAndExport:
    def _project(self, client: TestClient, media: dict[str, Path]) -> str:
        with media["video"].open("rb") as handle:
            response = client.post(
                "/api/videos",
                files={"file": ("subs.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            )
        assert response.status_code == 200, response.text
        return response.json()["id"]

    def test_roundtrip_de_subtitulos(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        _login(client, auth["email"])
        project_id = self._project(client, media)
        cues = [
            {
                "index": 1,
                "start_ms": 0,
                "end_ms": 1500,
                "text": "Hola, ¿qué tal?",
                "speaker": "SPEAKER_00",
                "language": "es",
                "state": "final",
            },
            {
                "index": 2,
                "start_ms": 1600,
                "end_ms": 3200,
                "text": "Todo bien, gracias.",
                "speaker": "SPEAKER_01",
                "language": "es",
                "state": "final",
            },
        ]
        saved = client.put(f"/api/subtitles/{project_id}", json={"language": "es", "cues": cues})
        assert saved.status_code == 200
        assert saved.json()["count"] == 2
        stored = client.get(f"/api/subtitles/{project_id}", params={"lang": "es"}).json()
        assert stored["subtitles"][0]["cues"][0]["text"] == "Hola, ¿qué tal?"

    def test_exportaciones_reales(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        project_id = self._project(client, media)
        cues = [
            {
                "index": 1,
                "start_ms": 0,
                "end_ms": 1500,
                "text": "Hola, ¿qué tal?",
                "speaker": "SPEAKER_00",
                "language": "es",
                "state": "final",
            },
            {
                "index": 2,
                "start_ms": 1600,
                "end_ms": 3200,
                "text": "Todo bien, gracias.",
                "speaker": "SPEAKER_01",
                "language": "es",
                "state": "final",
            },
        ]
        client.put(f"/api/subtitles/{project_id}", json={"language": "es", "cues": cues})

        srt = client.get(f"/api/export/{project_id}/srt", params={"lang": "es"})
        assert srt.status_code == 200
        srt_text = srt.content.decode("utf-8")
        assert "1\n00:00:00,000 --> 00:00:01,500\nHola, ¿qué tal?" in srt_text
        assert "2\n00:00:01,600 --> 00:00:03,200\nTodo bien, gracias." in srt_text

        vtt = client.get(f"/api/export/{project_id}/vtt", params={"lang": "es"})
        assert vtt.status_code == 200
        vtt_text = vtt.content.decode("utf-8")
        assert vtt_text.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:01.500" in vtt_text

        ass = client.get(f"/api/export/{project_id}/ass", params={"lang": "es"})
        assert ass.status_code == 200
        ass_text = ass.content.decode("utf-8")
        for marker in ("[Script Info]", "[V4+ Styles]", "[Events]", "Dialogue:"):
            assert marker in ass_text, marker

        archive = client.get(f"/api/export/{project_id}/zip")
        assert archive.status_code == 200
        import io
        import zipfile

        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            names = set(bundle.namelist())
            assert {"es.srt", "es.vtt", "es.ass"} <= names
            assert bundle.read("es.srt").decode("utf-8").startswith("1\n")

    def test_export_sin_subtitulos_da_404(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        project_id = self._project(client, media)
        assert client.get(f"/api/export/{project_id}/srt", params={"lang": "ja"}).status_code == 404
        assert client.get(f"/api/export/{project_id}/zip").status_code == 404

    def test_rechaza_idioma_de_subtitulo_invalido(
        self, client: TestClient, auth: dict[str, str], media: dict[str, Path]
    ) -> None:
        project_id = self._project(client, media)
        response = client.put(f"/api/subtitles/{project_id}", json={"language": "xx", "cues": []})
        assert response.status_code == 422


class TestProjectManagement:
    def test_renombrar_y_eliminar(self, client: TestClient, auth: dict[str, str], media: dict[str, Path]) -> None:
        _login(client, auth["email"])
        with media["video"].open("rb") as handle:
            response = client.post(
                "/api/videos",
                files={"file": ("gestion.mp4", handle, "video/mp4")},
                data={"source_language": "en", "target_languages": "es"},
            )
        assert response.status_code == 200, response.text
        project_id = response.json()["id"]

        renamed = client.put(f"/api/videos/{project_id}", json={"title": "Mi documental"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Mi documental"
        assert client.put(f"/api/videos/{project_id}", json={"title": ""}).status_code == 422
        assert client.put(f"/api/videos/{project_id}", json={"title": "x" * 181}).status_code == 422

        listed = client.get("/api/videos").json()
        assert any(item["id"] == project_id for item in listed["videos"])
        entry = next(item for item in listed["videos"] if item["id"] == project_id)
        for field in ("id", "title", "duration_ms", "created_at", "status", "source_language", "target_languages", "thumbnail_url"):
            assert field in entry, f"la tarjeta necesita {field}"

        deleted = client.delete(f"/api/videos/{project_id}")
        assert deleted.status_code == 200
        assert client.get(f"/api/videos/{project_id}").status_code == 404
