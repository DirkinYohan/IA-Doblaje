"""Sesión de subtitulado en vivo. El transcriber se inyecta; el VAD es real (energía)."""
from __future__ import annotations

import base64
from typing import Any, Callable


Transcriber = Callable[[list[float], bool], str]


Translator = Callable[[str, bool], dict[str, str]]


class RealtimeSession:
    def __init__(
        self,
        transcriber: Transcriber,
        sample_rate: int = 16000,
        translator: Translator | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.translator = translator
        self.sample_rate = sample_rate
        self.source_language = ""
        self.targets: list[str] = []
        self._samples: list[float] = []
        self._voice = False
        self._silence_samples = 0
        self._start_ms = 0
        self._clock_ms = 0
        self._partial_marks = 0

    def handle(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        kind = str(message.get("type") or "")
        if kind == "config":
            self.source_language = str(message.get("source_language") or "")
            self.targets = [str(item) for item in (message.get("targets") or []) if str(item)]
            return [{
                "type": "configured",
                "source_language": self.source_language,
                "targets": self.targets,
            }]
        if kind in {"seek", "pause"}:
            self._samples.clear()
            self._voice = False
            self._silence_samples = 0
            self._clock_ms = int(message.get("time_ms") or 0)
            return [{"type": "cleared", "time_ms": self._clock_ms}]
        if kind == "resume":
            self._clock_ms = int(message.get("time_ms") or self._clock_ms)
            return [{"type": "resumed", "time_ms": self._clock_ms}]
        if kind != "audio":
            return [{"type": "error", "message": f"mensaje no soportado: {kind}"}]
        samples = _decode_pcm(message)
        if not samples:
            return []
        events: list[dict[str, Any]] = []
        frame = max(1, int(self.sample_rate * 0.2))
        self._samples.extend(samples)
        self._clock_ms += int(len(samples) * 1000 / self.sample_rate)
        rms = _rms(samples)
        if rms >= 0.02:
            if not self._voice:
                self._voice = True
                self._start_ms = max(0, self._clock_ms - int(len(samples) * 1000 / self.sample_rate))
                self._partial_marks = 0
            self._silence_samples = 0
        elif self._voice:
            self._silence_samples += len(samples)
        if self._voice and len(self._samples) >= frame * 4 and len(self._samples) // frame > self._partial_marks:
            self._partial_marks = len(self._samples) // frame
            events.append(self._emit(partial=True))
        if self._voice and self._silence_samples >= int(self.sample_rate * 0.4):
            events.append(self._emit(partial=False))
            self._samples.clear()
            self._voice = False
            self._silence_samples = 0
        return [event for event in events if event]

    def _emit(self, *, partial: bool) -> dict[str, Any]:
        try:
            text = self.transcriber(list(self._samples), partial).strip()
        except Exception as exc:  # noqa: BLE001
            return {"type": "error", "message": str(exc)}
        if not text:
            return {}
        event: dict[str, Any] = {
            "type": "partial" if partial else "final",
            "text": text,
            "start_ms": self._start_ms,
            "end_ms": self._clock_ms,
            "state": "provisional" if partial else "final",
            "language": self.source_language,
            # Contrato estable: siempre hay un mapa de traducciones, aunque
            # esté vacío. El cliente no tiene que comprobar su existencia.
            "translations": {},
        }
        if self.translator is not None and self.targets:
            try:
                translated = self.translator(text, partial)
            except Exception as exc:  # noqa: BLE001
                event["translation_error"] = str(exc)
                partial_map = getattr(exc, "translations", None)
                if isinstance(partial_map, dict) and partial_map:
                    event["translations"] = {
                        str(key): str(value) for key, value in partial_map.items()
                    }
            else:
                if translated:
                    event["translations"] = translated
        return event


def _decode_pcm(message: dict[str, Any]) -> list[float]:
    raw = message.get("pcm_b64")
    if isinstance(raw, str) and raw:
        data = base64.b64decode(raw)
        out: list[float] = []
        for i in range(0, len(data) - 1, 2):
            value = int.from_bytes(data[i : i + 2], "little", signed=True)
            out.append(value / 32768.0)
        return out
    samples = message.get("samples")
    if isinstance(samples, list):
        return [float(x) for x in samples]
    return []


def _rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    acc = sum(s * s for s in samples) / len(samples)
    return acc ** 0.5


def whisper_transcriber(adapter: Any) -> Transcriber:
    """Adapta Faster-Whisper a una ventana PCM. Falla explícito si el modelo no carga."""

    def _run(samples: list[float], _partial: bool) -> str:
        import numpy as np

        audio = np.asarray(samples, dtype=np.float32)
        model = adapter._get_or_load_model()
        segments, _info = model.transcribe(
            audio,
            beam_size=1,
            vad_filter=False,
            word_timestamps=False,
            without_timestamps=False,
            task="transcribe",
            condition_on_previous_text=False,
        )
        return " ".join(str(getattr(seg, "text", "") or "").strip() for seg in segments).strip()

    return _run


def m2m_translator(adapter: Any, source: str, targets: list[str]) -> Translator:
    """Traduce la hipótesis en vivo. Un fallo no se sustituye por el texto fuente."""

    def _run(text: str, _partial: bool) -> dict[str, str]:
        from app.domain.value_objects.translation import TranslationLanguageCode

        output: dict[str, str] = {}
        errors: list[str] = []
        for target in targets:
            if target == source:
                continue
            try:
                output[target] = adapter.translate(
                    source_text=text,
                    source_language=TranslationLanguageCode(source),
                    target_language=TranslationLanguageCode(target),
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{target}:{exc}")
        if errors:
            error = RuntimeError("; ".join(errors))
            if output:
                error.translations = output  # type: ignore[attr-defined]
            raise error
        return output

    return _run
