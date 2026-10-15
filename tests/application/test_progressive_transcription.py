"""Tests del procesamiento progresivo por ventanas (con audio real)."""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from app.application.use_cases.progressive_transcription import (
    MIN_WINDOW_MS,
    ProgressiveAudioWindowReader,
    ProgressiveTranscriber,
    iter_windows,
)


def _write_wav(path: Path, seconds: float, *, sr: int = 16000, value: float = 0.3) -> Path:
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    audio = (value * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    return path


class TestIterWindows:
    def test_cubre_toda_la_duracion(self) -> None:
        windows = iter_windows(100_000, window_ms=30_000, overlap_ms=1_000)
        assert windows[0][0] == 0
        assert windows[-1][1] == 100_000
        # Cada ventana empieza antes de que acabe la anterior (solape).
        for previous, current in zip(windows, windows[1:], strict=False):
            assert current[0] < previous[1], "debe haber solape para no cortar palabras"

    def test_avanza_sin_huecos(self) -> None:
        windows = iter_windows(200_000, window_ms=20_000, overlap_ms=2_000)
        for previous, current in zip(windows, windows[1:], strict=False):
            # El salto es exactamente el paso (ventana - solape).
            assert current[0] - previous[0] == 18_000

    def test_duracion_cero_o_negativa(self) -> None:
        assert iter_windows(0) == []
        assert iter_windows(-5) == []

    def test_ventana_mas_corta_que_el_minimo_se_descarta(self) -> None:
        windows = iter_windows(MIN_WINDOW_MS - 1, window_ms=30_000, overlap_ms=1_000)
        assert windows == []

    def test_configuracion_invalida(self) -> None:
        with pytest.raises(ValueError):
            iter_windows(10_000, window_ms=1_000, overlap_ms=1_000)

    def test_primera_ventana_corta_publica_antes(self) -> None:
        """La primera ventana puede acortarse para mostrar subtítulos antes."""
        windows = iter_windows(150_000, window_ms=30_000, overlap_ms=1_000, first_window_ms=8_000)
        assert windows[0] == (0, 8_000), f"la primera debe ser corta: {windows[0]}"
        # Las siguientes mantienen el tamaño nominal y arrancan en el solape.
        assert windows[1] == (7_000, 37_000), f"la segunda arranca en el solape: {windows[1]}"
        assert windows[-1][1] == 150_000

    def test_primera_ventana_cubre_todo_sin_huecos(self) -> None:
        windows = iter_windows(150_000, window_ms=30_000, overlap_ms=1_000, first_window_ms=8_000)
        assert windows[0][0] == 0
        for (_prev_start, prev_end), (next_start, _next_end) in zip(
            windows, windows[1:], strict=False
        ):
            assert next_start < prev_end, f"hueco entre ventanas: {prev_end} -> {next_start}"

    def test_primera_ventana_no_supera_la_nominal(self) -> None:
        windows = iter_windows(60_000, window_ms=30_000, overlap_ms=1_000, first_window_ms=90_000)
        assert windows[0] == (0, 30_000), "nunca debe superar window_ms"

    def test_audio_mas_corto_que_la_primera_ventana(self) -> None:
        windows = iter_windows(3_000, window_ms=30_000, overlap_ms=1_000, first_window_ms=8_000)
        assert windows == [(0, 3_000)]

    def test_cobertura_completa_con_primera_ventana_corta(self) -> None:
        """El audio entero queda cubierto: ningún tramo se pierde."""
        total = 150_000
        windows = iter_windows(total, window_ms=30_000, overlap_ms=1_000, first_window_ms=8_000)
        assert windows[0][0] == 0 and windows[-1][1] == total
        for (_s1, e1), (s2, _e2) in zip(windows, windows[1:], strict=False):
            assert s2 <= e1, "sin huecos"


class TestAudioWindowReader:
    def test_lee_tramos_sin_cargar_todo(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "a.wav", 6.0)
        reader = ProgressiveAudioWindowReader(wav)
        assert reader.duration_ms == 6000

        # Tramo central: valor absoluto medio razonable (tono, no silencio).
        chunk = reader.read(2000, 3000)
        assert chunk.dtype == np.float32
        assert 15_000 < chunk.size < 17_000
        assert 0.1 < float(np.mean(np.abs(chunk))) < 0.5

    def test_recorta_a_los_limites(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "b.wav", 2.0)
        reader = ProgressiveAudioWindowReader(wav)
        assert reader.read(-500, 500).size > 0
        assert reader.read(1900, 5000).size > 0
        assert reader.read(3000, 4000).size == 0, "fuera del audio devuelve vacío"

    def test_rechaza_wav_inexistente(self, tmp_path: Path) -> None:
        from app.core.exceptions import AudioExtractionError

        with pytest.raises(AudioExtractionError):
            ProgressiveAudioWindowReader(tmp_path / "no-existe.wav")


class _FakeSegment:
    def __init__(self, start_ms: int, end_ms: int, text: str) -> None:
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.text = text


class _FakeASR:
    """Devuelve un segmento por ventana, con timestamps absolutos."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def transcribe_array(self, audio, *, offset_ms, language, start_index):
        self.calls.append((offset_ms, int(audio.size)))
        index = len(self.calls)
        return [_FakeSegment(offset_ms, offset_ms + 1500, f"texto ventana {index}")]


class TestProgressiveTranscriber:
    def test_publica_cues_ventana_a_ventana(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "c.wav", 10.0)
        reader = ProgressiveAudioWindowReader(wav)
        asr = _FakeASR()
        published: list[int] = []
        progress: list[tuple[int, int]] = []

        transcriber = ProgressiveTranscriber(
            asr=asr, window_ms=4000, overlap_ms=500, language="es"
        )
        results = transcriber.process(
            reader,
            on_cues=lambda window: published.append(len(window.cues)),
            on_progress=lambda done, total: progress.append((done, total)),
        )

        # Varias ventanas => varias publicaciones (nunca todo al final).
        assert len(results) >= 3
        assert len(published) == len(results)
        assert all(count >= 1 for count in published), "cada ventana publica sus cues"

        # El progreso es monótono y termina cubriendo todo el audio.
        done_values = [item[0] for item in progress]
        assert done_values == sorted(done_values)
        assert done_values[-1] == reader.duration_ms
        assert all(total == reader.duration_ms for _done, total in progress)

    def test_timestamps_absolutos_y_crecientes(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "d.wav", 12.0)
        reader = ProgressiveAudioWindowReader(wav)
        transcriber = ProgressiveTranscriber(asr=_FakeASR(), window_ms=4000, overlap_ms=500)
        results = transcriber.process(reader)

        starts = [cue.start_ms for window in results for cue in window.cues]
        assert starts == sorted(starts)
        assert starts[0] < 1000
        assert starts[-1] > 5000, "los timestamps cubren el audio, no se reinician por ventana"

    def test_cancelacion_detiene_el_procesamiento(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "e.wav", 60.0)
        reader = ProgressiveAudioWindowReader(wav)
        asr = _FakeASR()

        class CancelAfterTwo:
            def __init__(self) -> None:
                self.count = 0

            def is_set(self) -> bool:
                self.count += 1
                return self.count > 4

        results = ProgressiveTranscriber(asr=asr, window_ms=5000, overlap_ms=500).process(
            reader, cancel_event=CancelAfterTwo()
        )
        assert len(results) < 12, "se detiene antes de recorrer todo el audio"

    def test_silencio_no_produce_cues(self, tmp_path: Path) -> None:
        wav = _write_wav(tmp_path / "f.wav", 5.0, value=0.0001)
        reader = ProgressiveAudioWindowReader(wav)

        class _NeverCalled:
            def transcribe_array(self, *args, **kwargs):
                raise AssertionError("no debe llamarse al ASR sobre silencio")

        transcriber = ProgressiveTranscriber(asr=_NeverCalled(), vad=object(), window_ms=4000)
        results = transcriber.process(reader)
        assert all(not window.cues for window in results)

    def test_audio_vacio(self, tmp_path: Path) -> None:
        path = tmp_path / "vacio.wav"
        with wave.open(str(path), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
        reader = ProgressiveAudioWindowReader(path)
        assert ProgressiveTranscriber(asr=_FakeASR()).process(reader) == []

    def test_no_duplica_cues_por_el_solape(self, tmp_path: Path) -> None:
        """El solape sólo repite el tramo inmediatamente anterior."""
        wav = _write_wav(tmp_path / "g.wav", 12.0)
        reader = ProgressiveAudioWindowReader(wav)

        class OverlappingASR:
            """Simula a Whisper: reemite el final de la ventana anterior.

            Con paso 3000 ms y solape 1000 ms, "repetido" (final de la ventana
            previa) sólo debe aparecer una vez por frontera, no en cada ventana.
            """

            def transcribe_array(self, audio, *, offset_ms, language, start_index):
                items = []
                if offset_ms > 0:
                    # Final de la ventana anterior, dentro de la zona de solape.
                    items.append(
                        _FakeSegment(offset_ms - 800, offset_ms + 400, "repetido")
                    )
                items.append(_FakeSegment(offset_ms + 600, offset_ms + 1500, f"nuevo {offset_ms}"))
                return items

        results = ProgressiveTranscriber(
            asr=OverlappingASR(), window_ms=4000, overlap_ms=1000
        ).process(reader)
        cues = [cue for window in results for cue in window.cues]
        texts = [cue.text for cue in cues]

        assert texts.count("repetido") == 1, f"el solape duplicó cues: {texts}"
        assert len([t for t in texts if t.startswith("nuevo")]) == len(results)
        assert all(cue.end_ms > cue.start_ms for cue in cues)
        # INVARIANTE: la línea temporal nunca se solapa consigo misma.
        for previous, current in zip(cues, cues[1:], strict=False):
            assert current.start_ms >= previous.end_ms, (
                f"solape: {previous.start_ms}-{previous.end_ms} y "
                f"{current.start_ms}-{current.end_ms}"
            )

    def test_sin_solapes_aunque_el_asr_se_superponga(self, tmp_path: Path) -> None:
        """El ASR puede devolver segmentos que se pisan; la salida no debe."""
        wav = _write_wav(tmp_path / "h.wav", 20.0)
        reader = ProgressiveAudioWindowReader(wav)

        class SloppyASR:
            """Devuelve segmentos largos que invaden la ventana siguiente."""

            def transcribe_array(self, audio, *, offset_ms, language, start_index):
                return [
                    _FakeSegment(offset_ms, offset_ms + 3500, f"a {offset_ms}"),
                    _FakeSegment(offset_ms + 2000, offset_ms + 5000, f"b {offset_ms}"),
                ]

        cues = [
            cue
            for window in ProgressiveTranscriber(
                asr=SloppyASR(), window_ms=5000, overlap_ms=1000
            ).process(reader)
            for cue in window.cues
        ]
        assert cues, "debe producir cues"
        for previous, current in zip(cues, cues[1:], strict=False):
            assert current.start_ms >= previous.end_ms, (
                f"solape: {previous.start_ms}-{previous.end_ms} y "
                f"{current.start_ms}-{current.end_ms}"
            )
        assert [c.start_ms for c in cues] == sorted(c.start_ms for c in cues)
        assert all(c.end_ms > c.start_ms for c in cues)
