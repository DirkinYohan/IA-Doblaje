from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.fixtures.accounts import sign_up

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FFMPEG = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
SOURCE = Path(
    os.environ.get("IA_E2E_SOURCE")
    or "data/input/a29937b8-f0ee-40d7-b0d0-2ff63fde9d9d/"
    "3d12599a-5971-49e6-b426-c1a6ae1efa86/_B1_What_Do_You_Do_on_Weekends.mp4"
)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
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

    register = sign_up(
        worker_client,
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
        project_id = prepared["project_id"]
        job_id = prepared["job_id"]
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
        assert stages[-1] in {"COMPLETED", "PARTIAL", "FAILED"}, stages
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

    def test_pistas_con_contrato_de_sincronizacion(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        project_id = prepared["project_id"]
        tracks = worker_client.get(f"/api/subtitles/{project_id}").json()["subtitles"]
        by_language = {track["language"]: track for track in tracks}
        assert {"en", "es"} <= set(by_language), list(by_language)
        clip_ms = 40_000
        for language, track in by_language.items():
            assert track["project_id"] == project_id
            assert track["status"] == "complete"
            assert track["covered_ms"] >= clip_ms - 1_000, (language, track["covered_ms"])
            ids = [cue["id"] for cue in track["cues"]]
            assert len(ids) == len(set(ids)), "identificadores únicos por segmento"
            for cue in track["cues"]:
                assert cue["language"] == language
                assert cue["project_id"] == project_id
                assert isinstance(cue["start_ms"], int) and isinstance(cue["end_ms"], int)
                assert 0 <= cue["start_ms"] < cue["end_ms"] <= clip_ms + 1_000
        source_times = [(c["start_ms"], c["end_ms"]) for c in by_language["en"]["cues"]]
        target_times = [(c["start_ms"], c["end_ms"]) for c in by_language["es"]["cues"]]
        assert target_times == source_times

    def test_consulta_por_rango_para_zonas_procesadas_y_no_procesadas(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
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
        job_id = prepared["job_id"]
        with worker_client.websocket_connect(f"/api/jobs/{job_id}/events") as socket:
            first = socket.receive_json()
            assert first["type"] in {"snapshot", "processing_progress", "subtitle_created",
                                     "processing_finished", "processing_completed"}, first
            if first["type"] == "snapshot":
                assert "progress" in first and "status" in first
                assert "stage_history" in first

    def test_websocket_de_job_ajeno_se_rechaza(
        self, worker_client: TestClient, prepared: dict[str, str]
    ) -> None:
        worker_client.post("/api/auth/logout")
        sign_up(
            worker_client,
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
        media = worker_client.get(f"/api/videos/{prepared['project_id']}/media")
        assert media.status_code == 200
        assert len(media.content) > 10_000
        check = worker_client.get(f"/api/videos/{prepared['project_id']}/audio-check").json()
        assert check["preserved"] is True
        assert check["original"]["audio_streams"] >= 1
