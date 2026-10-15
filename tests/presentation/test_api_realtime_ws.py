"""Tests del WebSocket de tiempo real: PCM real -> transcripción -> cues.

El WebSocket de `TestClient` es síncrono y no todos los envíos producen
respuesta (una trama cuyo parcial sale vacío no emite nada), así que las
lecturas se hacen desde un hilo con timeout en vez de bloquear el test.
"""
from __future__ import annotations

import base64
import queue
import struct
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.presentation.api.main import create_app


def _pcm_b64(samples: list[float]) -> str:
    """Codifica floats [-1,1] como PCM s16le en base64, igual que el frontend."""
    raw = b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
    return base64.b64encode(raw).decode("ascii")


def _make_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("IA_DB_PATH", str(tmp_path / "app.sqlite"))
    monkeypatch.setenv("IA_MAIL_DIR", str(tmp_path / "mail"))
    monkeypatch.setenv("IA_DISABLE_WORKER", "1")
    return TestClient(create_app())


def _authenticate(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={
            "name": "Tiempo Real",
            "email": "realtime@example.com",
            "password": "password123",
            "confirm": "password123",
        },
    )
    assert response.status_code == 200, response.text


class _StubTranscriber:
    """Transcribe de verdad el flujo: texto distinto para parcial y final."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def __call__(self, samples: list[float], partial: bool) -> str:
        self.calls.append((len(samples), partial))
        return "Hola" if partial else "Hola, ¿cómo estás?"


class _Reader:
    """Lee mensajes del socket en un hilo y los encola, sin bloquear el test."""

    def __init__(self, socket: Any) -> None:
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._socket = socket
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.queue.put(self._socket.receive_json())
            except Exception:  # noqa: BLE001 - socket cerrado
                return

    def next(self, timeout: float = 5.0) -> dict[str, Any] | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def collect(self, timeout: float = 0.4) -> list[dict[str, Any]]:
        """Todo lo recibido hasta que pase `timeout` sin mensajes nuevos."""
        out: list[dict[str, Any]] = []
        while True:
            item = self.next(timeout)
            if item is None:
                return out
            out.append(item)

    def close(self) -> None:
        self._stop.set()


def test_websocket_requiere_sesion(tmp_path, monkeypatch) -> None:
    with _make_client(tmp_path, monkeypatch) as client:
        with client.websocket_connect("/api/realtime") as socket:
            message = socket.receive_json()
            assert message["type"] == "error"
            assert "autenticado" in message["message"].lower()


def test_websocket_rechaza_mensaje_desconocido(tmp_path, monkeypatch) -> None:
    with _make_client(tmp_path, monkeypatch) as client:
        _authenticate(client)
        with client.websocket_connect("/api/realtime") as socket:
            socket.send_json({"type": "ordenar_pizza"})
            message = socket.receive_json()
            assert message["type"] == "error"
            assert "no soportado" in message["message"]


@pytest.fixture
def stub_transcriber(monkeypatch):
    """Sustituye ASR y traducción reales por dobles deterministas.

    Se dobla también el traductor para que la prueba no cargue M2M100 (17 s) y
    pueda comprobar el contrato de traducción en vivo.
    """
    import app.presentation.api.main as main_module

    stub = _StubTranscriber()
    monkeypatch.setattr(main_module, "whisper_transcriber", lambda _adapter: stub)
    monkeypatch.setattr(
        main_module,
        "m2m_translator",
        lambda _adapter, _source, targets: (
            lambda text, _partial: {target: f"[{target}] {text}" for target in targets}
        ),
    )
    return stub


def test_websocket_config_y_pcm_real(tmp_path, monkeypatch, stub_transcriber) -> None:
    """Camino completo: config -> audio PCM -> parcial -> final -> seek.

    El WebSocket de `TestClient` procesa una trama por iteración del bucle de
    eventos, así que se envían tramas hasta reunir las evidencias esperadas,
    con un límite duro para que el test no pueda quedarse colgado.
    """
    with _make_client(tmp_path, monkeypatch) as client:
        _authenticate(client)
        with client.websocket_connect("/api/realtime") as socket:
            reader = _Reader(socket)
            try:
                socket.send_json({"type": "config", "source_language": "es", "targets": ["en"]})
                configured = reader.next()
                assert configured is not None and configured["type"] == "configured"
                assert configured["source_language"] == "es"
                assert configured["targets"] == ["en"]

                partials: list[dict[str, Any]] = []
                finals: list[dict[str, Any]] = []

                def harvest(timeout: float = 0.15) -> None:
                    for event in reader.collect(timeout):
                        if event["type"] == "partial":
                            partials.append(event)
                        elif event["type"] == "final":
                            finals.append(event)

                # 1. 20 tramas de 100 ms de "voz" a 16 kHz (como el AudioWorklet).
                voice = _pcm_b64([0.25] * 1_600)
                for index in range(20):
                    if partials:
                        break
                    socket.send_json({"type": "audio", "pcm_b64": voice, "time_ms": index * 100})
                    harvest()

                assert partials, "debe emitir al menos un subtítulo parcial"
                assert partials[0]["state"] == "provisional"
                assert partials[0]["text"] == "Hola"
                assert partials[0]["language"] == "es"
                # El contrato siempre incluye el mapa de traducciones.
                assert partials[0]["translations"] == {"en": "[en] Hola"}

                # 2. Silencio: cierra el segmento y emite el subtítulo DEFINITIVO.
                silence = _pcm_b64([0.0] * 1_600)
                for _ in range(12):
                    if finals:
                        break
                    socket.send_json({"type": "audio", "pcm_b64": silence})
                    harvest()

                assert finals, "debe emitir un subtítulo final"
                assert finals[0]["state"] == "final"
                assert finals[0]["text"] == "Hola, ¿cómo estás?"
                assert finals[0]["end_ms"] >= finals[0]["start_ms"]

                # 3. Seek: limpia el estado y reanuda el reloj.
                socket.send_json({"type": "seek", "time_ms": 5_000})
                cleared = None
                for _ in range(10):
                    candidate = reader.next(timeout=1.0)
                    assert candidate is not None, "el seek debe responder"
                    if candidate["type"] == "cleared":
                        cleared = candidate
                        break
                    harvest(0.05)
                assert cleared is not None and cleared["time_ms"] == 5_000

                socket.send_json({"type": "resume", "time_ms": 5_000})
                resumed = None
                for _ in range(10):
                    candidate = reader.next(timeout=1.0)
                    assert candidate is not None, "el resume debe responder"
                    if candidate["type"] == "resumed":
                        resumed = candidate
                        break
                    harvest(0.05)
                assert resumed is not None and resumed["time_ms"] == 5_000
            finally:
                reader.close()

        assert stub_transcriber.calls, "el transcriber debe haberse invocado"
        assert any(partial for _n, partial in stub_transcriber.calls)
        assert any(not partial for _n, partial in stub_transcriber.calls)


def test_websocket_pausa_limpia_el_buffer(tmp_path, monkeypatch, stub_transcriber) -> None:
    with _make_client(tmp_path, monkeypatch) as client:
        _authenticate(client)
        with client.websocket_connect("/api/realtime") as socket:
            reader = _Reader(socket)
            try:
                socket.send_json({"type": "config", "source_language": "en", "targets": []})
                assert (reader.next() or {}).get("type") == "configured"
                socket.send_json({"type": "pause", "time_ms": 1_234})
                message = reader.next()
                assert message is not None and message["type"] == "cleared"
                assert message["time_ms"] == 1_234
            finally:
                reader.close()


def test_websocket_pcm_invalido_no_rompe_la_sesion(tmp_path, monkeypatch, stub_transcriber) -> None:
    with _make_client(tmp_path, monkeypatch) as client:
        _authenticate(client)
        with client.websocket_connect("/api/realtime") as socket:
            reader = _Reader(socket)
            try:
                socket.send_json({"type": "config", "source_language": "en", "targets": []})
                assert (reader.next() or {}).get("type") == "configured"
                # Tramas degeneradas: vacía, base64 inválido y sin campo de audio.
                socket.send_json({"type": "audio", "pcm_b64": ""})
                socket.send_json({"type": "audio", "pcm_b64": "!!!no-base64!!!"})
                socket.send_json({"type": "audio"})
                # La sesión sigue viva: un mensaje válido vuelve a responder.
                socket.send_json({"type": "seek", "time_ms": 10})
                cleared = reader.next()
                assert cleared is not None and cleared["type"] == "cleared"
            finally:
                reader.close()
