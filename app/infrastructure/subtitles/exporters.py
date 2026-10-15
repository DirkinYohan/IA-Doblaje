"""Exportación SRT, VTT y ASS/SSA. UTF-8. Sin lógica de segmentación."""
from __future__ import annotations

from typing import Iterable


def _hms(ms: int, *, separator: str, centis: bool = False) -> str:
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    if centis:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}.{millis // 10:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def to_srt(cues: Iterable[object]) -> str:
    blocks: list[str] = []
    for cue in cues:
        blocks.append(
            "\n".join(
                (
                    str(int(cue.index)),
                    f"{_hms(cue.start_ms, separator=',')} --> {_hms(cue.end_ms, separator=',')}",
                    str(cue.text).strip(),
                    "",
                )
            )
        )
    return "\n".join(blocks).strip() + ("\n" if blocks else "")


def to_vtt(cues: Iterable[object]) -> str:
    lines = ["WEBVTT", ""]
    for cue in cues:
        lines.append(f"{_hms(cue.start_ms, separator='.')} --> {_hms(cue.end_ms, separator='.')}")
        voice = str(getattr(cue, "speaker", "") or "")
        text = str(cue.text).strip()
        if voice:
            lines.append(f"<v {voice}>{text}")
        else:
            lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def to_ass(cues: Iterable[object], *, language: str = "und") -> str:
    header = "\n".join(
        (
            "[Script Info]",
            "ScriptType: v4.00+",
            f"Language: {language}",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
            "0,0,0,0,100,100,0,0,1,2,0,2,40,40,30,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        )
    )
    rows = [header]
    for cue in cues:
        text = str(cue.text).replace("\n", r"\N")
        name = str(getattr(cue, "speaker", "") or "")
        rows.append(
            "Dialogue: 0,"
            f"{_hms(cue.start_ms, separator='.', centis=True)},"
            f"{_hms(cue.end_ms, separator='.', centis=True)},"
            f"Default,{name},0,0,0,,{text}"
        )
    return "\n".join(rows) + "\n"
