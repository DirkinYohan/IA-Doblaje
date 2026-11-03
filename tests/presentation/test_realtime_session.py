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
    assert finals[-1]["translations"] == {}
    assert "Hello" not in finals[-1]["translations"].values()


def test_reloj_del_reproductor_y_salto_descarta_audio_antiguo():
    def transcriber(_samples: list[float], _partial: bool) -> str:
        return "texto"

    session = RealtimeSession(transcriber, sample_rate=10)
    session.handle({"type": "seek", "time_ms": 0, "epoch": 1})
    events = session.handle({"type": "audio", "samples": [0.2] * 5, "time_ms": 31_000, "epoch": 1})
    events += session.handle({"type": "audio", "samples": [0.2] * 5, "time_ms": 32_000, "epoch": 1})
    events += session.handle({"type": "audio", "samples": [0.0] * 5, "time_ms": 33_000, "epoch": 1})
    final = [event for event in events if event.get("type") == "final"][-1]
    assert final["epoch"] == 1
    assert 30_000 <= final["start_ms"] <= 31_000
    assert final["end_ms"] == 33_000

    cleared = session.handle({"type": "seek", "time_ms": 300_000, "epoch": 2})
    assert cleared == [{"type": "cleared", "time_ms": 300_000, "epoch": 2}]
    assert session.handle({"type": "audio", "samples": [0.2] * 50, "time_ms": 34_000, "epoch": 1}) == []
    events = session.handle({"type": "audio", "samples": [0.2] * 50, "time_ms": 305_000, "epoch": 2})
    events += session.handle({"type": "audio", "samples": [0.0] * 5, "time_ms": 305_500, "epoch": 2})
    emitted = [event for event in events if event.get("type") in {"partial", "final"}]
    assert emitted
    assert all(event["epoch"] == 2 for event in emitted)
    assert all(event["start_ms"] >= 300_000 for event in emitted)


def test_lote_acumulado_transcribe_una_vez_y_no_pierde_finales():
    calls: list[bool] = []

    def transcriber(_samples: list[float], partial: bool) -> str:
        calls.append(partial)
        return "frase"

    session = RealtimeSession(transcriber, sample_rate=10)
    voice = [{"type": "audio", "samples": [0.2] * 2, "time_ms": 1000 + 200 * i} for i in range(20)]
    events = session.handle_batch(voice)
    assert calls.count(True) == 1, "un solo parcial para todo el lote"
    assert [event["type"] for event in events] == ["partial"]

    calls.clear()
    batch = [{"type": "audio", "samples": [0.0] * 5, "time_ms": 5500}]
    batch += [{"type": "audio", "samples": [0.2] * 2, "time_ms": 6000 + 200 * i} for i in range(5)]
    events = session.handle_batch(batch)
    assert [event["type"] for event in events][0] == "final", "el fin de frase no se pierde"
    assert events[0]["end_ms"] == 5500


def test_frase_sin_pausa_se_cierra_al_limite():
    from app.presentation.api.realtime import MAX_UTTERANCE_S

    session = RealtimeSession(lambda _s, _p: "larga", sample_rate=10)
    total = int(10 * MAX_UTTERANCE_S)
    events = []
    for i in range(total // 2):
        events += session.handle({"type": "audio", "samples": [0.2] * 2, "time_ms": 200 * (i + 1)})
    assert any(event["type"] == "final" for event in events)


def test_lote_con_salto_descarta_el_audio_anterior_sin_transcribirlo():
    calls: list[int] = []

    def transcriber(samples: list[float], _partial: bool) -> str:
        calls.append(len(samples))
        return "nuevo"

    session = RealtimeSession(transcriber, sample_rate=10)
    batch = [{"type": "audio", "samples": [0.2] * 2, "time_ms": 1000 + 200 * i, "epoch": 0} for i in range(20)]
    batch.append({"type": "seek", "time_ms": 300_000, "epoch": 1})
    batch += [{"type": "audio", "samples": [0.2] * 2, "time_ms": 300_200 + 200 * i, "epoch": 1} for i in range(5)]
    events = session.handle_batch(batch)
    assert calls == [10], "sólo se transcribe el audio posterior al salto"
    assert all(event.get("epoch") == 1 for event in events)


def test_parciales_sin_traduccion_y_finales_traducidos():
    translated: list[str] = []

    def translator(text: str, _partial: bool) -> dict[str, str]:
        translated.append(text)
        return {"es": "hola"}

    session = RealtimeSession(lambda _s, p: "parcial" if p else "final", sample_rate=10, translator=translator)
    events = session.handle({"type": "audio", "samples": [0.2] * 50})
    partials = [event for event in events if event["type"] == "partial"]
    assert partials and all(event["translations"] == {} for event in partials)

    session.handle({"type": "config", "source_language": "en", "targets": ["es"]})
    events = session.handle({"type": "audio", "samples": [0.2] * 50})
    events += session.handle({"type": "audio", "samples": [0.0] * 10})
    assert [event["type"] for event in events] == ["final"]
    assert events[-1]["translations"] == {"es": "hola"}
    assert translated == ["final"]


def test_frase_que_ya_llega_tarde_se_descarta_sin_transcribir():
    from app.presentation.api.realtime import MAX_LAG_MS

    calls: list[int] = []

    def transcriber(samples: list[float], _partial: bool) -> str:
        calls.append(len(samples))
        return "frase"

    session = RealtimeSession(transcriber, sample_rate=10)
    session.handle({"type": "config", "source_language": "en", "targets": ["es"]})
    old = [{"type": "audio", "samples": [0.2] * 5, "time_ms": 1000 + 500 * i} for i in range(4)]
    old.append({"type": "audio", "samples": [0.0] * 5, "time_ms": 3000})
    now = 3000 + MAX_LAG_MS + 2000
    recent = [{"type": "audio", "samples": [0.2] * 5, "time_ms": now + 500 * i} for i in range(4)]
    recent.append({"type": "audio", "samples": [0.0] * 5, "time_ms": now + 2500})
    events = session.handle_batch(old + recent)
    finals = [event for event in events if event["type"] == "final"]
    assert session.dropped == 1, "la frase vieja se descarta"
    assert len(finals) == 1 and finals[0]["start_ms"] >= now - 500
    assert len(calls) == 1, "sólo se transcribe la frase actual"
