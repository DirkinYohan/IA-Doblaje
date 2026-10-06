"""Procesamiento progresivo por ventanas sobre el audio ya extraído.

Reutiliza los adapters existentes (VAD, ASR, traducción) sin duplicar la
arquitectura: este módulo sólo decide **en qué orden** se consume el audio y
**cuándo** se publica cada subtítulo.

Contrato:
* El audio se lee por ventanas de ``window_ms`` con solape de contexto.
* En cada ventana se ejecuta VAD → ASR y se construyen cues.
* Los cues se publican INMEDIATAMENTE vía ``on_cues`` (nunca al final).
* ``on_progress`` informa del avance en milisegundos de audio cubiertos.

El audio original del vídeo no se toca: se trabaja sobre el WAV preprocesado.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app.application.use_cases.build_subtitles import CueLimits, build_cues
from app.core.exceptions import ASRProcessingError, AudioExtractionError

SAMPLE_RATE = 16000

# Ventana de trabajo y solape (el solape evita cortar palabras a la mitad).
DEFAULT_WINDOW_MS = 30_000
DEFAULT_OVERLAP_MS = 1_000
# Ventanas más cortas al final del audio no aportan nada.
MIN_WINDOW_MS = 500
# Separación mínima entre cues consecutivos (invariante de la línea temporal).
MIN_GAP_MS = 1


@dataclass
class WindowResult:
    """Resultado de una ventana: cues ya listos para persistir."""

    index: int
    start_ms: int
    end_ms: int
    cues: list[Any] = field(default_factory=list)
    language: str = "und"


class ProgressiveAudioWindowReader:
    """Lee tramos de un WAV 16 kHz mono sin cargarlo entero en memoria."""

    def __init__(self, wav_path: str | Path, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.path = Path(wav_path)
        if not self.path.is_file():
            raise AudioExtractionError(f"WAV preprocesado no encontrado: {self.path!s}")
        self.sample_rate = int(sample_rate)
        self.total_frames = self._count_frames()
        self.duration_ms = int(round(self.total_frames * 1000 / self.sample_rate))

    def _count_frames(self) -> int:
        import soundfile as sf

        with sf.SoundFile(str(self.path)) as handle:
            if int(handle.samplerate) != self.sample_rate:
                raise AudioExtractionError(
                    f"Se esperaba {self.sample_rate} Hz y el WAV tiene {handle.samplerate} Hz"
                )
            return int(len(handle))

    def read(self, start_ms: int, end_ms: int) -> Any:
        """Devuelve float32 mono [-1,1] del tramo pedido (recortado a los límites)."""
        import numpy as np
        import soundfile as sf

        start_frame = max(0, int(start_ms * self.sample_rate / 1000))
        stop_frame = min(self.total_frames, int(end_ms * self.sample_rate / 1000))
        if stop_frame <= start_frame:
            return np.zeros(0, dtype=np.float32)
        with sf.SoundFile(str(self.path)) as handle:
            handle.seek(start_frame)
            data = handle.read(stop_frame - start_frame, dtype="int16", always_2d=True)
        if data.size == 0:
            return np.zeros(0, dtype=np.float32)
        mono = data.astype(np.float32).mean(axis=1) / 32768.0
        return np.clip(mono, -1.0, 1.0)


def _normalize_text(text: str) -> str:
    """Clave de comparación para detectar el mismo diálogo entre ventanas."""
    return " ".join(str(text or "").split()).casefold()


def iter_windows(    total_ms: int,
    *,
    window_ms: int = DEFAULT_WINDOW_MS,
    overlap_ms: int = DEFAULT_OVERLAP_MS,
    first_window_ms: int | None = None,
) -> list[tuple[int, int]]:
    """Divide la duración en ventanas ``(start_ms, end_ms)`` no vacías.

    ``first_window_ms`` acorta SÓLO la primera ventana. Sirve para que el primer
    subtítulo llegue cuanto antes (el usuario no debe esperar a que se transcriba
    medio minuto para ver algo); las siguientes mantienen ``window_ms``, que da
    mejor calidad al ASR.
    """
    if total_ms <= 0:
        return []
    if window_ms <= overlap_ms:
        raise ValueError("window_ms debe ser mayor que overlap_ms")
    windows: list[tuple[int, int]] = []
    start = 0
    first = True
    while start < total_ms:
        size = window_ms
        if first and first_window_ms is not None:
            size = max(MIN_WINDOW_MS, min(int(first_window_ms), window_ms))
        end = min(total_ms, start + size)
        if end - start >= MIN_WINDOW_MS:
            windows.append((start, end))
        if end >= total_ms:
            break
        # El solape se descuenta siempre del tamaño nominal, no del primero.
        start = end - overlap_ms if first else start + (window_ms - overlap_ms)
        first = False
    return windows


class ProgressiveTranscriber:
    """Ejecuta VAD + ASR + cues ventana a ventana."""

    def __init__(
        self,
        *,
        asr: Any,
        vad: Any | None = None,
        translator: Any | None = None,
        language: str | None = None,
        limits: CueLimits | None = None,
        window_ms: int = DEFAULT_WINDOW_MS,
        overlap_ms: int = DEFAULT_OVERLAP_MS,
        first_window_ms: int | None = None,
    ) -> None:
        self.asr = asr
        self.vad = vad
        self.translator = translator
        self.language = language
        self.limits = limits or CueLimits()
        self.window_ms = int(window_ms)
        self.overlap_ms = int(overlap_ms)
        # La primera ventana puede ser más corta para publicar antes el primer
        # subtítulo; el resto conservan `window_ms`.
        self.first_window_ms = first_window_ms

    # ------------------------------------------------------------------
    def process(
        self,
        reader: ProgressiveAudioWindowReader,
        *,
        on_cues: Callable[[WindowResult], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_event: Any | None = None,
    ) -> list[WindowResult]:
        """Procesa el audio completo. Publica cues en cuanto están listos."""
        results: list[WindowResult] = []
        if reader.total_frames == 0:
            return results

        windows = iter_windows(
            reader.duration_ms,
            window_ms=self.window_ms,
            overlap_ms=self.overlap_ms,
            first_window_ms=self.first_window_ms,
        )
        cue_index = 0
        previous_end_ms = 0
        emitted_until_ms = 0
        # Texto normalizado -> fin en ms, para descartar repeticiones del solape.
        recent_texts: dict[str, int] = {}
        # Cues ya emitidos: garantizan que la línea temporal no se solape.
        emitted: list[Any] = []

        for index, (start_ms, end_ms) in enumerate(windows):
            if cancel_event is not None and getattr(cancel_event, "is_set", lambda: False)():
                break

            audio = reader.read(start_ms, end_ms)
            if audio.size == 0:
                if on_progress is not None:
                    on_progress(end_ms, reader.duration_ms)
                continue

            segments = self._transcribe_window(audio, offset_ms=start_ms, start_index=cue_index)
            raw: list[tuple[int, int, str, str]] = [
                (
                    int(getattr(seg, "start_ms", start_ms)),
                    int(getattr(seg, "end_ms", end_ms)),
                    str(getattr(seg, "text", "") or ""),
                    str(getattr(seg, "speaker", "") or ""),
                )
                for seg in segments
            ]
            # El solape de ventana hace que Whisper vuelva a oír el tramo final
            # de la ventana anterior y suele devolver texto/timestamps algo
            # distintos, así que no basta comparar por texto exacto.
            # Invariante: un cue nunca puede EMPEZAR antes de donde terminó el
            # último cue emitido (deja un hueco mínimo de separación).
            floor_ms = 0 if not emitted else emitted[-1].end_ms + MIN_GAP_MS

            fresh: list[tuple[int, int, str, str]] = []
            for item in raw:
                start, end = item[0], item[1]
                if end <= max(emitted_until_ms, floor_ms):
                    continue
                start = max(start, floor_ms)
                if end <= start:
                    continue
                key = _normalize_text(item[2])
                if key and key in recent_texts and start <= emitted_until_ms + self.overlap_ms:
                    continue
                fresh.append((start, end, item[2], item[3]))

            cues = build_cues(
                fresh,
                language=self.language or "und",
                limits=self.limits,
                state="final",
            )
            if cues:
                cue_index += len(cues)
                window_end = max(cue.end_ms for cue in cues)
                previous_end_ms = max(previous_end_ms, window_end)
                emitted_until_ms = max(emitted_until_ms, window_end)
                for cue in cues:
                    key = _normalize_text(cue.text)
                    if key:
                        recent_texts[key] = max(recent_texts.get(key, 0), cue.end_ms)
                # Poda: se recuerda al menos una ventana completa hacia atrás,
                # que es el alcance máximo del solape entre ventanas.
                cutoff = emitted_until_ms - (self.window_ms + 2 * self.overlap_ms)
                recent_texts = {
                    key: end for key, end in recent_texts.items() if end >= cutoff
                }
                emitted.extend(cues)

            result = WindowResult(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                cues=list(cues),
                language=self.language or "und",
            )
            results.append(result)
            if on_cues is not None:
                on_cues(result)
            if on_progress is not None:
                on_progress(end_ms, reader.duration_ms)

        return results

    # ------------------------------------------------------------------
    def _transcribe_window(self, audio: Any, *, offset_ms: int, start_index: int) -> list[Any]:
        import numpy as np

        # VAD opcional sobre la ventana: reduce alucinaciones en silencio.
        if self.vad is not None:
            speech = self._speech_ratio(audio)
            if speech < 0.001:
                return []

        return self.asr.transcribe_array(
            np.asarray(audio, dtype=np.float32),
            offset_ms=int(offset_ms),
            language=self.language,
            start_index=int(start_index),
        )

    @staticmethod
    def _speech_ratio(audio: Any) -> float:
        """Proporción de muestras por encima del umbral de ruido (RMS por bloque)."""
        import numpy as np

        block = 1600  # 100 ms
        total = len(audio) // block
        if total == 0:
            return 0.0
        trimmed = audio[: total * block].reshape(total, block)
        rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
        return float(np.mean(rms >= 0.01))


__all__ = [
    "DEFAULT_OVERLAP_MS",
    "DEFAULT_WINDOW_MS",
    "ProgressiveAudioWindowReader",
    "ProgressiveTranscriber",
    "WindowResult",
    "iter_windows",
]
