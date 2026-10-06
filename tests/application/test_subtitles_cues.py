from app.application.use_cases.build_subtitles import active_cue, build_cues
from app.infrastructure.subtitles.exporters import to_ass, to_srt, to_vtt


def test_vacio_no_genera_cue():
    assert build_cues([(0, 1000, "   ", "SPEAKER_00")], language="es") == []


def test_corta_en_dos_lineas_y_no_solapa():
    text = " ".join(["palabra"] * 30)
    cues = build_cues([(0, 8000, text, "SPEAKER_00")], language="es")
    assert cues
    assert all(len(cue.lines) <= 2 for cue in cues)
    assert all(len(line) <= 42 for cue in cues for line in cue.lines)
    for prev, nxt in zip(cues, cues[1:]):
        assert nxt.start_ms >= prev.end_ms


def test_activo_por_tiempo():
    cues = build_cues([(0, 2000, "Hola mundo.", "SPEAKER_00")], language="es")
    assert active_cue(cues, 100).text.startswith("Hola")
    assert active_cue(cues, 9000) is None


def test_export_utf8_y_cabecera_vtt():
    cues = build_cues([(0, 1500, "¿Qué tal?", "SPEAKER_00")], language="es")
    srt = to_srt(cues)
    vtt = to_vtt(cues)
    ass = to_ass(cues, language="es")
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "¿Qué tal?" in srt
    assert vtt.startswith("WEBVTT")
    assert "[Script Info]" in ass
    assert "¿Qué tal?" in ass
