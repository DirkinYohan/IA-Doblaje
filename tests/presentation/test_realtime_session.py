from app.presentation.api.realtime import RealtimeSession


def test_parcial_y_final_y_seek_vacia():
    calls: list[bool] = []

    def transcriber(samples: list[float], partial: bool) -> str:
        calls.append(partial)
        return "Hola" if partial else "Hola, ¿cómo estás?"

    session = RealtimeSession(transcriber, sample_rate=10)
    voice = [0.2] * 50
    events = []
    events.extend(session.handle({"type": "audio", "samples": voice}))
    events.extend(session.handle({"type": "audio", "samples": [0.0] * 10}))
    kinds = [event["type"] for event in events if event]
    assert "partial" in kinds
    assert "final" in kinds
    cleared = session.handle({"type": "seek", "time_ms": 5000})
    assert cleared[0]["type"] == "cleared"
def test_traduccion_fallida_no_copia_la_fuente():
    def transcriber(_samples: list[float], _partial: bool) -> str:
        return "Hello"

    def translator(_text: str, _partial: bool) -> dict[str, str]:
        raise RuntimeError("modelo caido")

    session = RealtimeSession(transcriber, sample_rate=10, translator=translator)
    session.targets = ["es"]
    events = session.handle({"type": "audio", "samples": [0.2] * 50})
    events.extend(session.handle({"type": "audio", "samples": [0.0] * 10}))
    finals = [event for event in events if event.get("type") == "final"]
    assert finals
    assert finals[-1]["text"] == "Hello"
    assert finals[-1]["translation_error"] == "modelo caido"
    # El mapa de traducciones está siempre presente (contrato estable) pero
    # vacío: un fallo NUNCA se sustituye por el texto original.
    assert finals[-1]["translations"] == {}
    assert "Hello" not in finals[-1]["translations"].values()
