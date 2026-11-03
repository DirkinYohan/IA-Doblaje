from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class StageSpec:
    key: str
    status: str
    percent: int
    label_es: str
    label_en: str


PIPELINE_STAGES: Final[tuple[StageSpec, ...]] = (
    StageSpec("QUEUED", "QUEUED", 0, "En cola", "Queued"),
    StageSpec("PREPARING", "PROCESSING", 3, "Preparando", "Preparing"),
    StageSpec("MEDIA_PREP", "PROCESSING", 8, "Audio", "Audio"),
    StageSpec("VAD", "PROCESSING", 16, "Detección de voz (VAD)", "Voice activity (VAD)"),
    StageSpec("LID", "PROCESSING", 22, "Detección de idioma", "Language detection"),
    StageSpec("ASR", "TRANSCRIBING", 32, "Transcripción", "Transcribing"),
    StageSpec("TIMESTAMPS", "TRANSCRIBING", 52, "Timestamps", "Timestamps"),
    StageSpec("DIARIZATION", "TRANSCRIBING", 58, "Diarización", "Speaker diarization"),
    StageSpec("ALIGNMENT", "PROCESSING", 66, "Alineación de hablantes", "Speaker alignment"),
    StageSpec("QUALITY", "PROCESSING", 72, "Control de calidad", "Quality control"),
    StageSpec("VALIDATION", "PROCESSING", 76, "Validación", "Validation"),
    StageSpec("METRICS", "PROCESSING", 79, "Métricas", "Metrics"),
    StageSpec("OUTPUT", "PROCESSING", 82, "JSON estructurado", "Structured JSON"),
    StageSpec("TRANSLATING", "TRANSLATING", 88, "Traducción", "Translating"),
    StageSpec("GENERATING_SUBTITLES", "GENERATING_SUBTITLES", 95, "Generando subtítulos", "Generating subtitles"),
    StageSpec("COMPLETED", "COMPLETED", 100, "Finalizado", "Finished"),
)

PROGRESSIVE_STAGES: Final[tuple[StageSpec, ...]] = (
    StageSpec("PREPARING", "PROCESSING", 1, "Preparando", "Preparing"),
    StageSpec("EXTRACTING_AUDIO", "PROCESSING", 5, "Extrayendo audio", "Extracting audio"),
    StageSpec("ANALYZING_AUDIO", "PROCESSING", 14, "Analizando audio", "Analyzing audio"),
    StageSpec("PROCESSING_WINDOW", "TRANSCRIBING", 16, "Procesando por segmentos", "Processing segments"),
    StageSpec("CUES_READY", "TRANSCRIBING", 20, "Subtítulos disponibles", "Subtitles available"),
    StageSpec("FINALIZING", "GENERATING_SUBTITLES", 97, "Cerrando subtítulos", "Finalizing subtitles"),
)

_BY_KEY: Final[dict[str, StageSpec]] = {
    spec.key: spec for spec in (*PIPELINE_STAGES, *PROGRESSIVE_STAGES)
}
_FALLBACK: Final[StageSpec] = StageSpec("PROCESSING", "PROCESSING", 50, "Procesando", "Processing")

STEP_TO_STAGE: Final[dict[str, str]] = {
    "T01_T02_T03": "MEDIA_PREP",
    "T04_VAD": "VAD",
    "T05_LID": "LID",
    "T06_ASR": "ASR",
    "T07_TIMESTAMPS": "TIMESTAMPS",
    "T08_DIARIZATION": "DIARIZATION",
    "T09_ALIGNMENT": "ALIGNMENT",
    "T10_QUALITY": "QUALITY",
    "T11_VALIDATION": "VALIDATION",
    "T12_METRICS": "METRICS",
    "T13_OUTPUT": "OUTPUT",
    "GENERATING_SUBTITLES": "GENERATING_SUBTITLES",
    "T14_CLEANUP": "COMPLETED",
}


STEP_PREFIX_TO_STAGE: Final[tuple[tuple[str, str], ...]] = (
    ("T01", "MEDIA_PREP"),
    ("T02", "MEDIA_PREP"),
    ("T03", "MEDIA_PREP"),
    ("T15", "TRANSLATING"),
    ("T14", "COMPLETED"),
)


def canonical_key(step: str) -> str:
    raw = str(step or "").strip()
    if not raw:
        return _FALLBACK.key
    if raw in STEP_TO_STAGE:
        return STEP_TO_STAGE[raw]
    if raw in _BY_KEY:
        return raw
    for prefix, key in STEP_PREFIX_TO_STAGE:
        if raw.startswith(prefix):
            return key
    return _FALLBACK.key


def stage_spec(key: str) -> StageSpec:
    return _BY_KEY.get(str(key or "").strip(), _FALLBACK)


def stage_percent(key: str) -> int:
    return stage_spec(key).percent


def stage_status(key: str) -> str:
    return stage_spec(key).status


class ProgressTracker:
    __slots__ = ("_last_percent", "_last_key")

    def __init__(self) -> None:
        self._last_percent = -1
        self._last_key = ""

    @property
    def percent(self) -> int:
        return max(0, self._last_percent)

    @property
    def key(self) -> str:
        return self._last_key

    def announce(self, key: str) -> tuple[str, int, str] | None:
        spec = stage_spec(key)
        percent = max(self._last_percent, spec.percent)
        if spec.key == self._last_key and percent == self._last_percent:
            return None
        self._last_key = spec.key
        self._last_percent = percent
        return (spec.status, percent, spec.key)

    def set_progress(self, key: str, percent: int) -> tuple[str, int, str]:
        spec = stage_spec(key)
        bounded = max(0, min(100, int(percent)))
        self._last_percent = max(self._last_percent, bounded)
        self._last_key = spec.key
        return (spec.status, self._last_percent, spec.key)

    def finish(self) -> tuple[str, int, str]:
        spec = _BY_KEY["COMPLETED"]
        self._last_key = spec.key
        self._last_percent = max(self._last_percent, spec.percent)
        return (spec.status, self._last_percent, spec.key)


__all__ = [
    "PIPELINE_STAGES",
    "PROGRESSIVE_STAGES",
    "ProgressTracker",
    "StageSpec",
    "canonical_key",
    "stage_percent",
    "stage_spec",
    "stage_status",
]
