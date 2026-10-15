"""Cues de subtítulo a partir de segmentos ASR ya alineados.

ASR Segment y Subtitle Cue son tipos distintos. Este módulo solo produce cues.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal


CueState = Literal["provisional", "final"]


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    speaker: str
    language: str
    state: CueState = "final"

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(line for line in self.text.split("\n") if line.strip())


@dataclass(frozen=True, slots=True)
class CueLimits:
    max_lines: int = 2
    max_chars_per_line: int = 42
    max_cps: float = 17.0
    min_duration_ms: int = 800
    max_duration_ms: int = 7000
    min_gap_ms: int = 80


def build_cues_from_alignment(
    alignment: object,
    *,
    language: str,
    text_by_index: dict[int, str] | None = None,
    limits: CueLimits | None = None,
    state: CueState = "final",
) -> list[SubtitleCue]:
    segments = list(getattr(alignment, "dialogue_segments", ()) or ())
    raw: list[tuple[int, int, str, str]] = []
    for seg in segments:
        idx = int(getattr(seg, "segment_index", len(raw)))
        text = (text_by_index or {}).get(idx, str(getattr(seg, "text", "") or ""))
        raw.append(
            (
                int(getattr(seg, "start_ms", 0)),
                int(getattr(seg, "end_ms", 0)),
                text,
                str(getattr(seg, "speaker_label", "") or ""),
            )
        )
    return build_cues(raw, language=language, limits=limits, state=state)


def build_cues(
    segments: Iterable[tuple[int, int, str, str]],
    *,
    language: str,
    limits: CueLimits | None = None,
    state: CueState = "final",
) -> list[SubtitleCue]:
    limits = limits or CueLimits()
    prepared: list[tuple[int, int, str, str]] = []
    for start_ms, end_ms, text, speaker in segments:
        clean = " ".join(str(text or "").split())
        if not clean:
            continue
        if end_ms <= start_ms:
            end_ms = start_ms + limits.min_duration_ms
        prepared.append((start_ms, end_ms, clean, speaker))
    prepared.sort(key=lambda item: (item[0], item[1]))

    cues: list[SubtitleCue] = []
    for start_ms, end_ms, text, speaker in prepared:
        blocks = _wrap_blocks(text, limits.max_chars_per_line, limits.max_lines)
        span = max(limits.min_duration_ms, end_ms - start_ms)
        weights = [max(1, sum(len(line) for line in block)) for block in blocks]
        total_w = sum(weights)
        cursor = start_ms
        for block, weight in zip(blocks, weights, strict=True):
            piece = max(limits.min_duration_ms, int(span * weight / total_w))
            piece_end = min(cursor + piece, cursor + limits.max_duration_ms)
            cues.append(
                SubtitleCue(
                    index=len(cues) + 1,
                    start_ms=cursor,
                    end_ms=max(piece_end, cursor + 1),
                    text="\n".join(block),
                    speaker=speaker,
                    language=language,
                    state=state,
                )
            )
            cursor = cues[-1].end_ms
    return _resolve_timing(cues, limits)


def active_cue(cues: list[SubtitleCue], time_ms: int) -> SubtitleCue | None:
    for cue in cues:
        if cue.start_ms <= time_ms < cue.end_ms:
            return cue
    return None


def _wrap_blocks(text: str, max_chars: int, max_lines: int) -> list[list[str]]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        token = word if len(word) <= max_chars else word[:max_chars]
        candidate = token if not current else f"{current} {token}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = token
    if current:
        lines.append(current)
    if not lines:
        return []
    blocks: list[list[str]] = []
    bucket: list[str] = []
    for line in lines:
        bucket.append(line)
        if len(bucket) == max_lines:
            blocks.append(_prefer_punctuation(bucket))
            bucket = []
    if bucket:
        blocks.append(bucket)
    return blocks


def _prefer_punctuation(lines: list[str]) -> list[str]:
    return lines


def _resolve_timing(cues: list[SubtitleCue], limits: CueLimits) -> list[SubtitleCue]:
    if not cues:
        return []
    fixed: list[SubtitleCue] = []
    for idx, cue in enumerate(cues):
        start = cue.start_ms
        end = cue.end_ms
        if fixed and start < fixed[-1].end_ms + limits.min_gap_ms:
            start = fixed[-1].end_ms + limits.min_gap_ms
        if end < start + limits.min_duration_ms:
            end = start + limits.min_duration_ms
        nxt = cues[idx + 1].start_ms if idx + 1 < len(cues) else end + limits.min_gap_ms
        if end > nxt - limits.min_gap_ms:
            end = max(start + 1, nxt - limits.min_gap_ms)
        chars = len(cue.text.replace("\n", ""))
        duration_s = max((end - start) / 1000.0, 0.001)
        if chars / duration_s > limits.max_cps:
            needed = int(chars / limits.max_cps * 1000)
            room = max(start + 1, nxt - limits.min_gap_ms)
            end = min(max(end, start + needed), room)
        fixed.append(
            SubtitleCue(
                index=len(fixed) + 1,
                start_ms=start,
                end_ms=max(end, start + 1),
                text=cue.text,
                speaker=cue.speaker,
                language=cue.language,
                state=cue.state,
            )
        )
    return fixed


def write_subtitle_exports(directory: str | Path, cues: list[SubtitleCue], *, language: str) -> dict[str, str]:
    from app.infrastructure.subtitles.exporters import to_ass
    from app.infrastructure.subtitles.exporters import to_srt
    from app.infrastructure.subtitles.exporters import to_vtt

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        f"subtitles_{language}.srt": to_srt(cues),
        f"subtitles_{language}.vtt": to_vtt(cues),
        f"subtitles_{language}.ass": to_ass(cues, language=language),
        f"subtitles_{language}.json": json.dumps([cue_to_dict(c) for c in cues], ensure_ascii=False),
    }
    for name, body in written.items():
        (out / name).write_text(body, encoding="utf-8")
    return written


def cue_to_dict(cue: SubtitleCue) -> dict[str, object]:
    return {
        "index": cue.index,
        "start_ms": cue.start_ms,
        "end_ms": cue.end_ms,
        "text": cue.text,
        "speaker": cue.speaker,
        "language": cue.language,
        "state": cue.state,
    }


def cue_from_dict(data: dict[str, object]) -> SubtitleCue:
    return SubtitleCue(
        index=int(data["index"]),
        start_ms=int(data["start_ms"]),
        end_ms=int(data["end_ms"]),
        text=str(data.get("text") or ""),
        speaker=str(data.get("speaker") or ""),
        language=str(data.get("language") or "und"),
        state="final" if str(data.get("state") or "final") != "provisional" else "provisional",
    )


def zip_exports(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in files.items():
            archive.writestr(name, body.encode("utf-8"))
    return buffer.getvalue()
