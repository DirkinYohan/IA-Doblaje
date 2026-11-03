"""Los subtítulos exportados deben poder subirse a YouTube tal cual.

YouTube admite SubRip (.srt) y WebVTT (.vtt) —no ASS—. Se validan los archivos
con un lector estricto de cada formato, como haría el importador:

* SRT: bloques «número / HH:MM:SS,mmm --> HH:MM:SS,mmm / texto / línea en
  blanco», numeración consecutiva desde 1, sin bloques vacíos.
* WebVTT: cabecera «WEBVTT», tiempos «HH:MM:SS.mmm --> HH:MM:SS.mmm», sin
  «-->» ni «<»/«&» sin escapar en el texto.
* En ambos: UTF-8, tiempos crecientes y fin posterior al inicio.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.infrastructure.subtitles.exporters import to_ass_burn, to_srt, to_vtt

SRT_TIME = re.compile(r"^(\d{2}):([0-5]\d):([0-5]\d),(\d{3}) --> (\d{2}):([0-5]\d):([0-5]\d),(\d{3})$")
VTT_TIME = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3}) --> (\d{2,}):([0-5]\d):([0-5]\d)\.(\d{3})$")


@dataclass
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str = ""


def _ms(groups: tuple[str, ...]) -> int:
    h, m, s, ms = (int(g) for g in groups)
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def parse_srt_strict(body: str) -> list[tuple[int, int, str]]:
    assert body.encode("utf-8").decode("utf-8") == body
    assert not body.startswith("﻿"), "sin BOM"
    blocks = body.strip("\n").split("\n\n")
    out = []
    for expected, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        assert len(lines) >= 3, f"bloque incompleto: {block!r}"
        assert lines[0] == str(expected), f"numeración no consecutiva: {lines[0]} != {expected}"
        match = SRT_TIME.match(lines[1])
        assert match, f"tiempo inválido: {lines[1]!r}"
        start, end = _ms(match.groups()[:4]), _ms(match.groups()[4:])
        text = "\n".join(lines[2:])
        assert text.strip(), "bloque sin texto"
        out.append((start, end, text))
    return out


def parse_vtt_strict(body: str) -> list[tuple[int, int, str]]:
    assert body.startswith("WEBVTT\n"), "cabecera WEBVTT"
    blocks = body.strip("\n").split("\n\n")[1:]
    out = []
    for block in blocks:
        lines = block.split("\n")
        match = VTT_TIME.match(lines[0])
        assert match, f"tiempo inválido: {lines[0]!r}"
        text = "\n".join(lines[1:])
        assert text.strip(), "cue sin texto"
        assert "-->" not in text, "«-->» no puede ir en el texto"
        assert "<" not in text and re.search(r"&(?!amp;|lt;|gt;)", text) is None, "texto sin escapar"
        out.append((_ms(match.groups()[:4]), _ms(match.groups()[4:]), text))
    return out


def _check_times(entries: list[tuple[int, int, str]]) -> None:
    for start, end, _text in entries:
        assert end > start, "el fin debe ser posterior al inicio"
    starts = [start for start, _end, _text in entries]
    assert starts == sorted(starts), "tiempos crecientes"


# Una pista "real" tras editar: desordenada, con índices repetidos, un cue
# vacío, uno sin duración, líneas en blanco, caracteres especiales y > 1 h.
MESSY = [
    Cue(7, 4_000, 5_500, "Segunda frase"),
    Cue(7, 1_000, 2_500, "Primera\n\nfrase con línea en blanco"),
    Cue(3, 6_000, 6_000, "sin duración"),
    Cue(9, 7_000, 8_000, "   "),
    Cue(1, 9_000, 10_250, "Tom & Jerry <3 --> fin"),
    Cue(2, 3_725_000, 3_727_500, "Pasada la hora: ¿qué tal? ñandú 🎬"),
]


def test_srt_valido_para_youtube() -> None:
    entries = parse_srt_strict(to_srt(MESSY))
    _check_times(entries)
    assert [text.split("\n")[0] for _s, _e, text in entries] == [
        "Primera", "Segunda frase", "Tom & Jerry <3 --> fin", "Pasada la hora: ¿qué tal? ñandú 🎬",
    ]
    assert entries[-1][:2] == (3_725_000, 3_727_500), "horas bien formadas"


def test_vtt_valido_para_youtube() -> None:
    entries = parse_vtt_strict(to_vtt(MESSY))
    _check_times(entries)
    assert len(entries) == 4
    assert entries[2][2] == "Tom &amp; Jerry &lt;3 → fin"


def test_pista_vacia_no_rompe_los_archivos() -> None:
    assert to_srt([]) == ""
    assert to_vtt([]) == "WEBVTT\n"


def test_ass_para_incrustar_se_adapta_a_la_resolucion() -> None:
    ass = to_ass_burn(MESSY, language="es", width=1280, height=720)
    assert "PlayResX: 1280" in ass and "PlayResY: 720" in ass
    style = next(line for line in ass.splitlines() if line.startswith("Style: Default"))
    fields = style.split(",")
    assert fields[2] == str(round(720 * 0.052)), "tamaño proporcional al alto"
    assert fields[15] == "3", "caja de fondo (BorderStyle 3)"
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogues) == 4
    assert "Primera\\Nfrase" in dialogues[0]
    assert "{" not in "".join(dialogues), "el texto no abre bloques de estilo"
