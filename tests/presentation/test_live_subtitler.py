from __future__ import annotations

from typing import Any

import numpy as np

from app.presentation.api.live import (
    BACKOFF_MS,
    LOOKAHEAD_MS,
    LiveCallbacks,
    LiveSubtitler,
    Ranges,
)

SR = 16_000
DURATION = 600_000
DIALOGUE = [
    (1_000, 2_800, "hello there"),
    (4_000, 5_600, "welcome"),
    (7_200, 9_900, "second line"),
    (12_500, 13_900, "third"),
    (299_200, 301_500, "five minutes"),
    (302_000, 304_000, "right after"),
    (30_200, 31_800, "thirty"),
]


class _Seg:
    def __init__(self, start_ms: int, end_ms: int, text: str) -> None:
        self.start_ms, self.end_ms, self.text = start_ms, end_ms, text


class Fakes:
    def __init__(self, *, fail_translation: bool = False, detect: str = "en") -> None:
        self.events: list[dict[str, Any]] = []
        self.persisted: dict[str, tuple[list[dict[str, Any]], int]] = {}
        self.transcribed: list[tuple[int, int]] = []
        self.translated: list[tuple[str, list[str]]] = []
        self.fail_translation = fail_translation
        self.detected = detect

    def read(self, start: int, end: int) -> Any:
        return np.full(int((end - start) * SR / 1000), 0.3, dtype=np.float32)

    def transcribe(self, audio: Any, offset_ms: int, _language: str | None) -> list[Any]:
        end = offset_ms + int(audio.size * 1000 / SR)
        self.transcribed.append((offset_ms, end))
        out = []
        for start, stop, text in sorted(DIALOGUE):
            if stop <= offset_ms or start >= end:
                continue
            clipped = start < offset_ms or stop > end
            out.append(_Seg(max(start, offset_ms), min(stop, end), text + (" (cortado)" if clipped else "")))
        return out

    def translate(self, texts: list[str], _source: str, target: str) -> list[str]:
        if self.fail_translation is True or target in (self.fail_translation or ()):
            raise RuntimeError("modelo caído")
        self.translated.append((target, list(texts)))
        return [f"[{target}] {text}" for text in texts]

    def detect(self, _audio: Any) -> str:
        return self.detected

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def persist(self, language: str, cues: list[dict[str, Any]], covered: int) -> None:
        self.persisted[language] = ([dict(c) for c in cues], covered)

    def callbacks(self) -> LiveCallbacks:
        return LiveCallbacks(
            read=self.read, transcribe=self.transcribe, translate=self.translate,
            detect=self.detect, emit=self.emit, persist=self.persist,
        )


def _subtitler(fakes: Fakes, source: str = "en", existing=None, targets=None) -> LiveSubtitler:
    return LiveSubtitler(
        callbacks=fakes.callbacks(), source_language=source, duration_ms=DURATION,
        existing=existing, targets=targets,
    )


def _texts(cues: list[dict[str, Any]]) -> list[str]:
    return [c["text"] for c in cues]


def _run_until(sub: LiveSubtitler, predicate, limit: int = 200) -> None:
    for _ in range(limit):
        if predicate():
            return
        if not sub.step():
            break
    assert predicate(), "no se alcanzó el estado esperado"


def test_ranges_fusiona_y_encuentra_huecos() -> None:
    r = Ranges()
    r.add(0, 1000)
    r.add(2000, 3000)
    r.add(900, 2100)
    assert r.items == [(0, 3000)]
    r.add(5000, 6000)
    assert r.first_gap(0, 10_000) == (3000, 5000)
    assert r.first_gap(5500, 10_000) == (6000, 10_000)
    assert r.first_gap(0, 3000) is None
    assert r.prefix_end() == 3000
    assert r.contains(5500) and not r.contains(4000)


def test_el_primer_subtitulo_sale_de_la_primera_pasada_desde_el_cabezal() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(0, "en")
    assert sub.step()
    first = fakes.events[0]
    assert first["type"] == "cues" and first["language"] == "en"
    assert len(fakes.transcribed) == 1
    assert (first["cues"][0]["start_ms"], first["cues"][0]["end_ms"]) == (1_000, 2_800)


def test_tiempos_coinciden_con_el_audio_original_y_sin_cortes_en_los_bordes() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(0, "en")
    _run_until(sub, lambda: sub._ranges("en").first_gap(0, 40_000) is None)
    cues = sub.cues["en"]
    expected = sorted((s, e, t) for s, e, t in DIALOGUE if e <= 40_000)
    assert [(c["start_ms"], c["end_ms"], c["text"]) for c in cues] == expected
    assert not any("cortado" in c["text"] for c in cues), "un segmento cortado se repite entero"


def test_saltar_al_minuto_5_procesa_alli_y_no_sigue_por_el_principio() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(30_000, "en")
    sub.step()
    fakes.transcribed.clear()
    sub.set_position(300_000, "en")
    sub.step()
    start, _end = fakes.transcribed[0]
    assert start == 300_000 - BACKOFF_MS, "empieza junto al nuevo cabezal"
    assert "five minutes" in _texts(sub.cues["en"])


def test_retroceder_reutiliza_lo_ya_generado() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(0, "en")
    _run_until(sub, lambda: sub._ranges("en").first_gap(0, 20_000) is None)
    sub.set_position(300_000, "en")
    sub.step()
    fakes.transcribed.clear()
    sub.set_position(5_000, "en")
    sub.step()
    first_start = fakes.transcribed[0][0]
    assert first_start >= 20_000, "lo ya transcrito no se repite"


def test_traduce_solo_el_idioma_que_se_ve_y_conserva_los_tiempos() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(0, "es")
    _run_until(sub, lambda: "es" in sub.cues and len(sub.cues["es"]) >= 2)
    es = sub.cues["es"]
    en = sub.cues["en"]
    assert {t for t, _ in fakes.translated} == {"es"}
    by_time = {(c["start_ms"], c["end_ms"]): c["text"] for c in en}
    for cue in es:
        assert by_time[(cue["start_ms"], cue["end_ms"])] in cue["text"]
    assert fakes.transcribed[-1][0] < 20_000


def test_cambiar_de_idioma_tras_saltar_traduce_alli() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(300_000, "es")
    _run_until(sub, lambda: any(c["text"] == "[es] five minutes" for c in sub.cues.get("es", [])))
    sub.set_position(300_500, "fr")
    _run_until(sub, lambda: any(c["text"] == "[fr] five minutes" for c in sub.cues.get("fr", [])))
    assert not any(c["start_ms"] < 290_000 for c in sub.cues["fr"])


def test_primero_el_horizonte_del_cabezal_y_despues_el_resto() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(0, "en")
    while sub._ranges("en").prefix_end() < LOOKAHEAD_MS:
        assert sub.step()
        assert all(start < LOOKAHEAD_MS for start, _ in fakes.transcribed), "nada lejano antes de tiempo"
    _run_until(sub, sub.is_complete, limit=500)
    assert sub._ranges("en").prefix_end() == DURATION, "sin espectadores también termina el vídeo"


def test_proyecto_ya_generado_no_repite_trabajo() -> None:
    fakes = Fakes()
    existing = {"en": ([{"start_ms": 1_000, "end_ms": 2_800, "text": "hello there"}], DURATION)}
    sub = _subtitler(fakes, existing=existing)
    sub.set_position(0, "en")
    assert not sub.step()
    assert fakes.transcribed == []


def test_persiste_la_pista_y_su_cobertura() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes)
    sub.set_position(0, "es")
    _run_until(sub, lambda: "es" in fakes.persisted)
    cues, covered = fakes.persisted["en"]
    assert covered > 0 and cues[0]["start_ms"] == 1_000


def test_traduccion_fallida_se_informa_una_vez_sin_bucle() -> None:
    fakes = Fakes(fail_translation=True)
    sub = _subtitler(fakes)
    sub.set_position(0, "es")
    for _ in range(30):
        sub.step()
    failures = [e for e in fakes.events if e["type"] == "translation_failed"]
    assert len(failures) == 1 and failures[0]["language"] == "es"
    assert "es" not in sub.cues or not sub.cues["es"], "nunca se presenta como traducido"


def test_idioma_de_origen_desconocido_se_detecta() -> None:
    fakes = Fakes(detect="en")
    sub = _subtitler(fakes, source="und")
    sub.set_position(0, None)
    sub.step()
    assert sub.source == "en"
    assert {"type": "source_language", "language": "en"} in fakes.events


def test_silencio_avanza_sin_llamar_al_asr() -> None:
    fakes = Fakes()
    fakes.read = lambda s, e: np.zeros(int((e - s) * SR / 1000), dtype=np.float32)  # type: ignore[method-assign]
    sub = LiveSubtitler(callbacks=fakes.callbacks(), source_language="en", duration_ms=20_000)
    sub.set_position(0, "en")
    _run_until(sub, lambda: sub._ranges("en").prefix_end() == 20_000)
    assert fakes.transcribed == []


def test_los_adaptadores_reales_exponen_lo_que_usa_el_motor() -> None:
    from app.infrastructure.audio.faster_whisper_asr_adapter import FasterWhisperSmallASRAdapter
    from app.infrastructure.translation.m2m100_translation_adapter import M2M100TranslatorAdapter

    for name in ("_get_engine", "translate_batch"):
        assert callable(getattr(M2M100TranslatorAdapter, name, None)), name
    for name in ("_get_or_load_model", "transcribe_array"):
        assert callable(getattr(FasterWhisperSmallASRAdapter, name, None)), name


def test_completa_todos_los_idiomas_y_un_fallo_no_bloquea_los_demas() -> None:
    fakes = Fakes(fail_translation=("fr",))
    sub = _subtitler(fakes, targets=["es", "fr", "de"])
    sub.set_position(0, "es")
    _run_until(sub, sub.is_complete, limit=1000)
    en = {(c["start_ms"], c["end_ms"]) for c in sub.cues["en"]}
    for lang in ("es", "de"):
        assert {(c["start_ms"], c["end_ms"]) for c in sub.cues[lang]} == en, lang
        assert sub._ranges(lang).prefix_end() == DURATION
    assert sub.failed == {"fr"} and not sub.cues.get("fr")
    assert sub.progress() < 100, "el idioma fallido no cuenta como hecho"


def test_el_idioma_visto_se_traduce_antes_que_los_demas() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes, targets=["es", "fr"])
    sub.set_position(0, "fr")
    _run_until(sub, lambda: bool(sub.cues.get("fr")))
    assert {t for t, _ in fakes.translated} == {"fr"}


def test_ids_estables_e_idempotentes() -> None:
    fakes = Fakes()
    sub = _subtitler(fakes, targets=["es"])
    sub.set_position(0, "es")
    _run_until(sub, lambda: bool(sub.cues.get("es")))
    cue = sub.cues["es"][0]
    assert cue["id"] == f"es:{cue['start_ms']}:{cue['end_ms']}"
    before = len(sub.cues["es"])
    sub._publish("es", [{k: cue[k] for k in ("start_ms", "end_ms", "text")}], 0, 1)
    assert len(sub.cues["es"]) == before, "volver a publicar no duplica"


def test_runner_unico_por_proyecto_instantanea_y_secuencia() -> None:
    import threading

    from app.presentation.api import live

    built = []
    gate = threading.Event()

    def build(runner):
        built.append(runner)
        fakes = Fakes()
        orig = fakes.transcribe

        def slow(audio, offset, lang):
            gate.wait(5)
            return orig(audio, offset, lang)

        fakes.transcribe = slow  # type: ignore[method-assign]
        fakes.emit = runner.emit  # type: ignore[method-assign]
        return LiveSubtitler(callbacks=fakes.callbacks(), source_language="en",
                             duration_ms=20_000, targets=["es"])

    try:
        runner, created = live.ensure_runner("p-unico", build)
        again, created_again = live.ensure_runner("p-unico", build)
        assert created and not created_again and again is runner, "sin procesos duplicados"
        runner.set_position(0, "es")
        received: list[dict] = []
        snapshot, detach = runner.attach(received.append)
        assert snapshot["type"] == "snapshot" and snapshot["seq"] == 0
        gate.set()
        assert runner.done.wait(10), "termina al completar el vídeo"
        assert len(built) == 1
        seqs = [e["seq"] for e in received]
        assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), "secuencia ordenada y sin repetir"
        assert received[-1]["type"] == "completed"
        late, _ = runner.attach(lambda _e: None)
        assert late["complete"] and late["tracks"]["es"] and late["seq"] == seqs[-1]
        detach()
        assert live.get_runner("p-unico") is None, "terminado, deja de estar registrado"
    finally:
        gate.set()
        live.stop_all()


def test_si_el_video_no_se_puede_leer_falla_sin_reintentar_para_siempre() -> None:
    from app.presentation.api import live

    calls = []

    def build(runner):
        fakes = Fakes()

        def broken(_start, _end):
            calls.append(1)
            raise RuntimeError("Error opening input file")

        fakes.read = broken  # type: ignore[method-assign]
        fakes.emit = runner.emit  # type: ignore[method-assign]
        return LiveSubtitler(callbacks=fakes.callbacks(), source_language="en", duration_ms=20_000)

    try:
        runner, _ = live.ensure_runner("p-roto", build)
        assert runner.done.wait(10), "termina en vez de quedarse en bucle"
        assert "Error opening input file" in runner.error
        assert len(calls) == live.MAX_FAILURES
    finally:
        live.stop_all()
