"""Exportación SRT, VTT y ASS/SSA. UTF-8. Sin lógica de segmentación.

Los archivos deben poder subirse tal cual a YouTube (que admite SRT y WebVTT,
no ASS). Por eso, antes de escribir cualquier formato, la pista se normaliza:

* orden por tiempo de inicio;
* fuera los cues vacíos o sin duración (un bloque SRT sin texto rompe el
  archivo: la línea en blanco cierra el bloque antes de tiempo);
* sin líneas en blanco DENTRO de un cue, por el mismo motivo;
* numeración SRT consecutiva desde 1, sea cual sea el índice guardado.

WebVTT además escapa `&`, `<` y `>` y no admite `-->` dentro del texto.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_BLANK_LINES = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class _Clean:
    start_ms: int
    end_ms: int
    text: str


def _hms(ms: int, *, separator: str, centis: bool = False) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if centis:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{millis // 10:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _clean(cues: Iterable[object]) -> list[_Clean]:
    """Pista lista para escribir: ordenada, sin vacíos ni líneas en blanco."""
    out: list[_Clean] = []
    for cue in cues:
        text = str(getattr(cue, "text", "") or "").replace("\r\n", "\n").replace("\r", "\n")
        text = _BLANK_LINES.sub("\n", text).strip()
        start, end = int(getattr(cue, "start_ms", 0)), int(getattr(cue, "end_ms", 0))
        if text and end > start:
            out.append(_Clean(max(0, start), end, text))
    out.sort(key=lambda c: (c.start_ms, c.end_ms))
    return out


def to_srt(cues: Iterable[object]) -> str:
    blocks = [
        f"{number}\n{_hms(c.start_ms, separator=',')} --> {_hms(c.end_ms, separator=',')}\n{c.text}\n"
        for number, c in enumerate(_clean(cues), start=1)
    ]
    return "\n".join(blocks)


def _vtt_text(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # «-->» (ya escapado como «--&gt;») no puede aparecer en el texto de un cue.
    return escaped.replace("--&gt;", "→")


def to_vtt(cues: Iterable[object]) -> str:
    lines = ["WEBVTT", ""]
    for c in _clean(cues):
        lines.append(f"{_hms(c.start_ms, separator='.')} --> {_hms(c.end_ms, separator='.')}")
        lines.append(_vtt_text(c.text))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _ass_text(text: str) -> str:
    # Las llaves abren bloques de estilo en ASS: el texto no puede crearlos.
    return text.replace("{", "(").replace("}", ")").replace("\n", r"\N")


def _ass_document(
    cues: Iterable[object],
    *,
    language: str,
    width: int,
    height: int,
    style: str,
) -> str:
    header = "\n".join(
        (
            "[Script Info]",
            "ScriptType: v4.00+",
            f"Language: {language}",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            style,
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        )
    )
    rows = [header]
    for c in _clean(cues):
        rows.append(
            "Dialogue: 0,"
            f"{_hms(c.start_ms, separator='.', centis=True)},"
            f"{_hms(c.end_ms, separator='.', centis=True)},"
            f"Default,,0,0,0,,{_ass_text(c.text)}"
        )
    return "\n".join(rows) + "\n"


def to_ass(cues: Iterable[object], *, language: str = "und") -> str:
    style = (
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,40,40,30,1"
    )
    return _ass_document(cues, language=language, width=1920, height=1080, style=style)


def to_ass_burn(cues: Iterable[object], *, language: str, width: int, height: int) -> str:
    """ASS para INCRUSTAR en el vídeo (libass), a la medida de su resolución.

    Igual que el reproductor: texto blanco sobre una caja oscura translúcida
    (BorderStyle 3), en la parte inferior y proporcional al alto del vídeo.
    """
    width = max(2, int(width))
    height = max(2, int(height))
    font = max(16, round(height * 0.052))
    pad = max(4, round(font * 0.32))
    margin_v = max(12, round(height * 0.07))
    margin_h = max(16, round(width * 0.08))
    # BorderStyle 3: la "caja" usa OutlineColour; &H70 ≈ 56 % de opacidad.
    style = (
        f"Style: Default,Arial,{font},&H00FFFFFF,&H000000FF,&H70000000,&H70000000,"
        f"0,0,0,0,100,100,0,0,3,{pad},0,2,{margin_h},{margin_h},{margin_v},1"
    )
    return _ass_document(cues, language=language, width=width, height=height, style=style)
