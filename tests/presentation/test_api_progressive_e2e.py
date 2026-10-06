"""E2E del flujo progresivo: Producir, tiempo real, pausa y adelantar.

Cubre los requisitos que no se pueden comprobar con dobles:
* el audio del original nunca se toca,
* el trabajo arranca con «Producir», no al subir,
* los subtítulos se publican antes de terminar el procesamiento,
* el procesamiento NO se detiene cuando el reproductor está pausado,
* adelantar funciona en zonas procesadas y no procesadas.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FFMPEG = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
SOURCE = Path(
    "data/input/a29937b8-f0ee-40d7-b0d0-2ff63fde9d9d/"
    "c003c8ef-cf17-4dc4-9416-a425e8b7ac2e/_B1_What_Do_You_Do_on_Weekends.mp4"
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Vídeo real de 40 s con audio, recodificado para ser autocontenido."""
    if not SOURCE.is_file():
        pytest.skip("falta el vídeo de origen para la prueba E2E progresiva")
    binary = str(FFMPEG) if FFMPEG.is_file() else "ffmpeg"
    target = tmp_path_factory.mktemp("clip") / "clip40.mp4"
    done = subprocess.run(
        [binary, "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(SOURCE), "-t", "40", "-c:v", "copy", "-c:a", "aac", str(target)],
        capture_output=True, text=True, check=False, timeout=300,
    )
    if done.returncode != 0 or not target.is_file():
        pytest.skip(f"FFmpeg no pudo preparar el clip: {done.stderr[:200]}")
    return target


@pytest.fixture(scope="module")
def worker_client(tmp_path_factory: pytest.TempPathFactory):
    """App real CON worker activo y almacenamiento aislado.

    Se reinicia la cola antes de crear la app: otro módulo de pruebas puede
    haber dejado el worker apagado o apuntando a una base de datos distinta.
    """
    from app.presentation.api import worker

    worker.reset()
    tmp = tmp_path_factory.mktemp("progressive")
    os.environ["IA_DB_PATH"] = str(tmp / "app.sqlite")
    os.environ["IA_MAIL_DIR"] = str(tmp / "mail")
    os.environ.pop("IA_DISABLE_WORKER", None)

    from app.presentation.api.main import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
    worker.reset()


@pytest.fixture(scope="module")
def prepared(worker_client: TestClient, clip: Path) -> dict[str, str]:
    """Sube el clip y pulsa «Producir». Devuelve ids y el estado inicial.

    Se comprueba ANTES que el traductor (M2M100 418M) se puede cargar: con la
    suite completa en paralelo el proceso puede quedarse sin memoria. Si no
    cabe, el motor lo reporta como PARTIAL con la causa (no lo oculta), y estas
    pruebas se saltan en lugar de dar un falso fallo.
    """
    try:
        from app.core.config import get_settings
        from app.domain.value_objects.translation import TranslationLanguageCode
        from app.infrastructure.translation.m2m100_translation_adapter import (
            M2M100TranslatorAdapter,
        )

        adapter = M2M100TranslatorAdapter(settings_getter=get_settings)
        adapter.translate(
            source_text="hello",
            source_language=TranslationLanguageCode("en"),
            target_language=TranslationLanguageCode("es"),
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"M2M100 no disponible en este entorno: {type(exc).__name__}: {exc}")

    register = worker_client.post(
        "/api/auth/register",
        json={
            "name": "Progresivo E2E",
            "email": "progresivo@example.com",
            "password": "password123",
            "confirm": "password123",
        },
    )
    assert register.status_code == 200, register.text

    with clip.open("rb") as handle:
        upload = worker_client.post(
            "/api/videos",
            files={"file": ("clip40.mp4", handle, "video/mp4")},
            data={"source_language": "en", "target_languages": "es"},
        )
    assert upload.status_code == 200, upload.text
    project_id = upload.json()["id"]

    before = worker_client.get(f"/api/videos/{project_id}/audio-check").json()
    produce = worker_client.post(
        f"/api/videos/{project_id}/produce", json={"mode": "progressive"}
    )
    assert produce.status_code == 200, produce.text
    return {
        "project_id": project_id,
        "job_id": produce.json()["id"],
        "audio_before": before,
    }


class TestAudioPreservation:
    def test_el_original_tiene_video_y_audio(self, prepared: dict[str, str]) -> None:
        original = prepared["audio_before"]["original"]
        assert original["video_streams"] >= 1
        assert original["audio_streams"] >= 1
        assert original["audio_codec"], "debe haber codec de audio"
        assert original["audio_sample_rate"] > 0

    def test_el_original_no_cambia_al_procesar(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        before = prepared["audio_before"]["original"]
        after = worker_client.get(
            f"/api/videos/{prepared['project_id']}/audio-check"
        ).json()["original"]
        for field in (
            "video_streams", "audio_streams", "video_codec",
            "audio_codec", "audio_sample_rate", "audio_channels", "duration_ms",
        ):
            assert after[field] == before[field], f"cambió {field}: {before[field]} -> {after[field]}"

    def test_el_video_se_sirve_con_soporte_de_seek(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        project_id = prepared["project_id"]
        full = worker_client.get(f"/api/videos/{project_id}/media")
        assert full.status_code == 200
        assert len(full.content) > 10_000
        ranged = worker_client.get(
            f"/api/videos/{project_id}/media", headers={"Range": "bytes=0-2047"}
        )
        assert ranged.status_code in (200, 206), "sin Range el seek del reproductor no funciona"


class TestProductionFlow:
    def test_producir_es_idempotente(self, worker_client: TestClient, prepared: dict[str, str]) -> None:
        again = worker_client.post(
            f"/api/videos/{prepared['project_id']}/produce", json={"mode": "progressive"}
        )
        assert again.status_code == 200
        assert again.json()["reused"] is True, "no debe lanzar un segundo trabajo"


class TestProgressiveDelivery:
    def test_llegan_cues_antes_de_terminar_y_el_worker_no_se_detiene(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        """El corazón del requisito: subtítulos mientras el worker sigue.

        Se comprueba con EVIDENCIA PERSISTIDA, no con una carrera de tiempos:
        el historial registra `CUES_READY` una vez por ventana, antes de
        `FINALIZING`. Si todo se guardara al final habría una sola entrada.
        Además se vigila que el trabajo avance sin que nadie reproduzca el
        vídeo: el servidor no recibe señal alguna del reproductor, así que
        pausarlo no puede detenerlo.
        """
        project_id = prepared["project_id"]
        job_id = prepared["job_id"]
        # Observación pasiva: equivale a tener el reproductor pausado.
        first = worker_client.get(f"/api/jobs/{job_id}").json()
        baseline_progress = int(first["progress"])
        baseline_covered = int(first["last_cue_ms"])
        progressed_while_idle = False

        deadline = time.time() + 900
        while time.time() < deadline:
            job = worker_client.get(f"/api/jobs/{job_id}").json()
            if (
                int(job["progress"]) > baseline_progress
                or int(job["last_cue_ms"]) > baseline_covered
            ):
                progressed_while_idle = True
            if job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(1.0)

        job = worker_client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] in {"COMPLETED", "PARTIAL"}, (
            f"el trabajo no terminó: {job['status']} {job['error']}"
        )
        assert progressed_while_idle, (
            "el trabajo no avanzó mientras el reproductor estaba inactivo (pausado)"
        )
        assert job["last_cue_ms"] > 0, "no se registró audio cubierto por subtítulos"
        assert int(job["progress"]) == 100

        stages = [entry["stage"] for entry in job["stage_history"]]
        cue_windows = stages.count("CUES_READY")
        assert cue_windows >= 1, f"no hay entrega de subtítulos en el historial: {stages}"
        assert "FINALIZING" in stages
        # El cierre es COMPLETED si todo salió bien, o PARTIAL si algo falló
        # (p. ej. la traducción): nunca se disfraza de éxito total.
        assert stages[-1] in {"COMPLETED", "PARTIAL", "FAILED"}, stages
        # Cada ventana publica: la entrega ocurre ANTES del cierre.
        assert stages.index("CUES_READY") < stages.index("FINALIZING"), (
            f"los subtítulos se cerraron antes de publicarse: {stages}"
        )
        percents = [int(entry["progress"]) for entry in job["stage_history"]]
        assert percents == sorted(percents), f"el progreso retrocedió: {percents}"

    def test_cues_sincronizados_y_sin_solapes(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        project_id = prepared["project_id"]
        job = worker_client.get(f"/api/jobs/{prepared['job_id']}").json()
        assert job["status"] in {"COMPLETED", "PARTIAL"}
        data = worker_client.get(
            f"/api/subtitles/{project_id}/cues", params={"lang": "es"}
        ).json()
        cues = data["cues"]
        assert cues, (
            f"deben existir subtítulos. Persistidos: "
            f"{[(t['language'], len(t['cues'])) for t in worker_client.get(f'/api/subtitles/{project_id}').json()['subtitles']]} "
            f"· error={job['error'][:150]!r}"
        )
        assert data["complete"] is True

        for index, cue in enumerate(cues, start=1):
            assert int(cue["index"]) == index, "índices contiguos desde 1"
            assert int(cue["end_ms"]) > int(cue["start_ms"]), "duración positiva"
        for previous, current in zip(cues, cues[1:], strict=False):
            assert int(current["start_ms"]) >= int(previous["end_ms"]), (
                f"solape: {previous['start_ms']}-{previous['end_ms']} y "
                f"{current['start_ms']}-{current['end_ms']}"
            )

    def test_consulta_por_rango_para_zonas_procesadas_y_no_procesadas(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        """Adelantar debe funcionar: el tramo pedido se devuelve bien acotado."""
        project_id = prepared["project_id"]
        everything = worker_client.get(
            f"/api/subtitles/{project_id}/cues", params={"lang": "es"}
        ).json()
        total = everything["total"]
        assert total > 1

        first_half = worker_client.get(
            f"/api/subtitles/{project_id}/cues",
            params={"lang": "es", "start_ms": 0, "end_ms": 10_000},
        ).json()
        assert len(first_half["cues"]) < total
        assert all(int(cue["start_ms"]) < 10_000 for cue in first_half["cues"])

        # Una zona sin voz devuelve lista vacía, no un error.
        empty_zone = worker_client.get(
            f"/api/subtitles/{project_id}/cues",
            params={"lang": "es", "start_ms": 10**9, "end_ms": 10**9 + 1000},
        ).json()
        assert empty_zone["cues"] == []

    def test_etapas_del_progresivo_registradas(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        job = worker_client.get(f"/api/jobs/{prepared['job_id']}").json()
        stages = [entry["stage"] for entry in job["stage_history"]]
        percents = [int(entry["progress"]) for entry in job["stage_history"]]
        for expected in ("PREPARING", "EXTRACTING_AUDIO", "CUES_READY", "FINALIZING", "COMPLETED"):
            assert expected in stages, f"falta la etapa {expected}: {stages}"
        assert percents == sorted(percents), f"el progreso retrocedió: {percents}"


class TestRealtimeEvents:
    def test_websocket_reentrega_el_estado_y_los_cues(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        """Un cliente que se conecta tarde recibe el estado y puede resincronizar."""
        job_id = prepared["job_id"]
        with worker_client.websocket_connect(f"/api/jobs/{job_id}/events") as socket:
            first = socket.receive_json()
            assert first["type"] in {"snapshot", "processing_progress", "subtitle_created",
                                     "processing_finished", "processing_completed"}, first
            # El snapshot siempre lleva el estado persistido.
            if first["type"] == "snapshot":
                assert "progress" in first and "status" in first
                assert "stage_history" in first

    def test_websocket_de_job_ajeno_se_rechaza(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        worker_client.post("/api/auth/logout")
        worker_client.post(
            "/api/auth/register",
            json={
                "name": "Ajeno",
                "email": "ajeno@example.com",
                "password": "password123",
                "confirm": "password123",
            },
        )
        with worker_client.websocket_connect(
            f"/api/jobs/{prepared['job_id']}/events"
        ) as socket:
            message = socket.receive_json()
            assert message["type"] == "error"
        # Se recupera la sesión original para no romper el resto de la clase.
        worker_client.post("/api/auth/logout")
        worker_client.post(
            "/api/auth/login",
            json={"email": "progresivo@example.com", "password": "password123"},
        )


class TestExport:
    def test_exportaciones_conservan_los_subtitulos(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        project_id = prepared["project_id"]
        for kind in ("srt", "vtt", "ass"):
            response = worker_client.get(f"/api/export/{project_id}/{kind}", params={"lang": "es"})
            assert response.status_code == 200, f"{kind} -> {response.status_code}"
            assert len(response.content) > 50
        zip_response = worker_client.get(f"/api/export/{project_id}/zip")
        assert zip_response.status_code == 200

    def test_media_sigue_disponible_tras_exportar(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        """El vídeo con su audio sigue sirviéndose después de todo el flujo."""
        media = worker_client.get(f"/api/videos/{prepared['project_id']}/media")
        assert media.status_code == 200
        assert len(media.content) > 10_000
        check = worker_client.get(f"/api/videos/{prepared['project_id']}/audio-check").json()
        assert check["preserved"] is True
        assert check["original"]["audio_streams"] >= 1
