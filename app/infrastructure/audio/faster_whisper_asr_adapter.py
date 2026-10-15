from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from app.core.exceptions import (
    ASRProcessingError,
    ConfigurationError,
    ModelLoadError,
)

# =============================================================================
# IMPORT FASTER-WHISPER — NO en tiempo de import. Lazy.
# =============================================================================
_faster_whisper_imported_ok = False
_WhisperModel_cls: Any = None
_ctranslate2_import_err: str | None = None
try:  # pragma: no cover - runtime
    from faster_whisper import WhisperModel as _FW  # noqa: WPS433

    _WhisperModel_cls = _FW
    _faster_whisper_imported_ok = True
except Exception as exc:  # noqa: BLE001 - import guard runtime
    _ctranslate2_import_err = (
        "Faster-Whisper/CTranslate2 extra [asr] no instalado. "
        "Instalar manualmente: pip install ia-doblaje-engine[asr] "
        "(regla T06: NO pip install automático)."
        f" Detalle interno: {type(exc).__name__}: {exc}"
    )


# =============================================================================
# CONSTANTES FINALES (NO acepta configuración usuario) — D#2 + D#3
# =============================================================================
MODEL_FOLDERNAME = "whisper_small_ct2_int8_fp16_local_v1"

MODEL_LABEL = "faster-whisper:small:int8-fp16:ct2-local-v1"

REQUIRED_FILES_INSIDE_FOLDER = (
    "model.bin",
    "config.json",
)

REQUIRED_FILES_INSIDE_FOLDER_OR = (
    # CTranslate2 folder con tokenizer.json (modern) o vocabulary.txt (legacy)
    "tokenizer.json",
    "vocabulary.txt",
)

FOLDER_MIN_SIZE_BYTES = 50_000_000  # 50 MB approx min
FOLDER_MAX_SIZE_BYTES = 5_000_000_000  # 5 GB max safety


def _as_mono_f32(np_mod: Any, audio: Any, sr_read: int) -> Any:
    if int(sr_read) != 16000:
        raise ASRProcessingError(f"WAV sample_rate={sr_read} != 16000")
    audio_f32 = audio.astype(np_mod.float32, copy=False) / 32768.0
    audio_f32 = np_mod.nan_to_num(audio_f32, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)
    if getattr(audio_f32, "ndim", 1) == 2:
        if audio_f32.shape[1] == 0:
            raise ASRProcessingError("WAV 2D sin canales")
        audio_f32 = audio_f32.mean(axis=1, dtype=np_mod.float32)
    return audio_f32


def _read_wav_full(sf_mod: Any, np_mod: Any, wav_path: Any) -> Any:
    try:
        audio, sr_read = sf_mod.read(str(wav_path), dtype="int16", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise ASRProcessingError(
            f"No se pudo leer WAV {wav_path}: {type(exc).__name__}: {exc}"
        ) from exc
    return _as_mono_f32(np_mod, audio, int(sr_read))


def _read_wav_window(
    sf_mod: Any,
    np_mod: Any,
    wav_path: Any,
    start_ms: int,
    end_ms: int,
    pad_ms: int = 120,
) -> tuple[Any, int]:
    """Lee solo la ventana de voz. Devuelve (float32, offset_ms de la ventana)."""
    try:
        with sf_mod.SoundFile(str(wav_path)) as handle:
            sr = int(handle.samplerate)
            if sr != 16000:
                raise ASRProcessingError(f"WAV sample_rate={sr} != 16000")
            origin = max(0, int(start_ms) - int(pad_ms))
            stop = int(end_ms) + int(pad_ms)
            start_frame = max(0, int(origin * sr / 1000))
            stop_frame = min(len(handle), max(start_frame + 1, int(stop * sr / 1000)))
            handle.seek(start_frame)
            audio = handle.read(stop_frame - start_frame, dtype="int16", always_2d=False)
    except ASRProcessingError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ASRProcessingError(
            f"No se pudo leer ventana WAV {wav_path}: {type(exc).__name__}: {exc}"
        ) from exc
    offset_ms = int(round(start_frame * 1000 / sr))
    return _as_mono_f32(np_mod, audio, sr), offset_ms


def _segment_from_raw(raw_seg: Any, index: int, offset_ms: int) -> Any:
    from app.domain.entities.asr import ASRSegment
    from app.domain.entities.asr import ASRWord

    text = getattr(raw_seg, "text", None)
    if not isinstance(text, str):
        text = ""
    text = text.strip("\x00")

    def _ms(value: Any) -> int | None:
        if isinstance(value, (int, float)) and float("-inf") < float(value) < float("inf"):
            scaled = int(round(float(value) * 1000.0)) + int(offset_ms)
            return scaled if scaled >= 0 else 0
        return None

    start_ms = _ms(getattr(raw_seg, "start", None))
    end_ms = _ms(getattr(raw_seg, "end", None))
    if start_ms is not None and end_ms is not None and end_ms < start_ms:
        end_ms = start_ms
    avg_lp = getattr(raw_seg, "avg_logprob", None)
    avg_logprob_val: float | None = None
    confidence: float | None = None
    if isinstance(avg_lp, (int, float)) and float("-inf") < float(avg_lp) < float("inf"):
        avg_logprob_val = float(avg_lp)
        confidence = float(min(max((avg_logprob_val + 6.0) / 6.0, 0.0), 1.0))
    no_speech_p = getattr(raw_seg, "no_speech_prob", None)
    no_speech_val: float | None = None
    if isinstance(no_speech_p, (int, float)):
        ns = float(no_speech_p)
        if 0.0 <= ns <= 1.0:
            no_speech_val = ns
    words: list[Any] = []
    for raw_word in getattr(raw_seg, "words", None) or ():
        wtext = str(getattr(raw_word, "word", "") or "").strip("\x00").strip()
        wstart = _ms(getattr(raw_word, "start", None))
        wend = _ms(getattr(raw_word, "end", None))
        if not wtext or wstart is None or wend is None:
            continue
        if wend < wstart:
            wend = wstart
        prob = getattr(raw_word, "probability", None)
        wconf = float(prob) if isinstance(prob, (int, float)) and 0.0 <= float(prob) <= 1.0 else None
        words.append(ASRWord(text=wtext, start_ms=wstart, end_ms=wend, confidence=wconf))
    return ASRSegment(
        segment_index=index,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        avg_logprob=avg_logprob_val,
        no_speech_prob=no_speech_val,
        confidence=confidence,
        words=tuple(words),
    )


# =============================================================================
# LOADER — Port ASRModelLoaderPort
# =============================================================================
class FasterWhisperLocalModelLoader:
    """Loader 100% local. Valida carpeta y contenido. SIN descargas."""

    __slots__ = ("_settings_getter", "_device_getter", "_folder_name")

    def __init__(
        self,
        settings_getter: Any | None = None,
        device_getter: Any | None = None,
        folder_name: str | None = None,
    ) -> None:
        self._settings_getter = settings_getter
        self._device_getter = device_getter
        self._folder_name = folder_name or MODEL_FOLDERNAME

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _get_models_dir(self) -> Path:
        if self._settings_getter is not None:
            settings = self._settings_getter()
            models_dir = settings.paths.models_cache_dir
        else:
            from app.core.config import get_settings

            settings = get_settings()
            models_dir = settings.paths.models_cache_dir
        resolved = Path(models_dir).expanduser().resolve()
        if not resolved.is_absolute():
            raise ModelLoadError(f"models_cache_dir={resolved} no es ruta absoluta")
        return resolved

    def _resolve_expected_folder_absolute(self) -> Path:
        models_dir = self._get_models_dir()
        raw_path = models_dir / self._folder_name
        resolved = raw_path.expanduser().resolve()
        models_norm = models_dir.resolve().as_posix()
        res_norm = resolved.as_posix()
        if not (res_norm == models_norm or res_norm.startswith(models_norm.rstrip("/") + "/")):
            raise ModelLoadError(
                f"Path traversal detectado: {raw_path!r} se resuelve FUERA de models_cache_dir. "
                "Regla T06 seguridad."
            )
        return resolved

    # ---------------------------------------------------------------------
    # Validaciones estructura carpeta
    # ---------------------------------------------------------------------
    def validate_folder_structure(self, folder_path: Path) -> None:
        # 0) Seguridad Path Traversal: ruta resuelta debe estar DENTRO de models_cache_dir
        models_dir = self._get_models_dir()
        resolved_folder = Path(folder_path).expanduser().resolve()
        models_norm = models_dir.as_posix()
        folder_norm = resolved_folder.as_posix()
        if not (folder_norm == models_norm or folder_norm.startswith(models_norm.rstrip("/") + "/")):
            raise ModelLoadError(
                f"Path traversal detectado: {folder_path!r} se resuelve FUERA de models_cache_dir "
                f"({models_dir!r}). Regla T06 seguridad."
            )
        if not resolved_folder.exists():
            raise ModelLoadError(
                f"{MODEL_FOLDERNAME} no encontrado en models/. "
                "Colocar manualmente. NO descargar automáticamente (regla T06)."
            )
        if not resolved_folder.is_dir():
            raise ModelLoadError(
                f"{MODEL_FOLDERNAME} debe ser una CARPETA CTranslate2, "
                f"no un archivo. Encontrado: {folder_path} (file)."
            )
        for fname in REQUIRED_FILES_INSIDE_FOLDER:
            fp = resolved_folder / fname
            if not fp.is_file():
                raise ModelLoadError(
                    f"Carpeta ASR {MODEL_FOLDERNAME} incompleta: falta archivo requerido {fname!r}."
                )
        any_vocab = False
        for fname in REQUIRED_FILES_INSIDE_FOLDER_OR:
            if (resolved_folder / fname).is_file():
                any_vocab = True
                break
        if not any_vocab:
            raise ModelLoadError(
                f"Carpeta ASR {MODEL_FOLDERNAME} incompleta: falta uno de "
                f"{REQUIRED_FILES_INSIDE_FOLDER_OR}."
            )
        total_bytes = 0
        for root, _dirs, files in os.walk(str(resolved_folder)):
            for f in files:
                try:
                    sz = os.path.getsize(os.path.join(root, f))
                    total_bytes += sz
                except OSError:
                    pass
        if total_bytes < FOLDER_MIN_SIZE_BYTES:
            raise ModelLoadError(
                f"Carpeta ASR {MODEL_FOLDERNAME} demasiado pequeña: "
                f"{total_bytes} bytes < {FOLDER_MIN_SIZE_BYTES} mínimos. Modelo corrupto o incompleto."
            )
        if total_bytes > FOLDER_MAX_SIZE_BYTES:
            raise ModelLoadError(
                f"Carpeta ASR {MODEL_FOLDERNAME} excede límite de seguridad: "
                f"{total_bytes} bytes > {FOLDER_MAX_SIZE_BYTES}."
            )

    # ---------------------------------------------------------------------
    # Load API (implementa ASRModelLoaderPort.load)
    # ---------------------------------------------------------------------
    def load(
        self,
        model_folder: Path,
        *,
        device: str,
        compute_type: str | None = None,
    ) -> Any:
        global _WhisperModel_cls, _faster_whisper_imported_ok, _ctranslate2_import_err
        # 1) Validar carpeta ANTES de cualquier cosa (pure Path checks, no dependencia)
        self.validate_folder_structure(Path(model_folder))
        # 2) Chequear dependencia sólo cuando realmente se va a instanciar WhisperModel
        if not _faster_whisper_imported_ok:
            raise ConfigurationError(_ctranslate2_import_err or "")
        if compute_type is None:
            compute_type = "int8_float16"
        local_abs_path_str = str(Path(model_folder).expanduser().resolve())
        try:
            # WhisperModel de Faster-Whisper recibe ruta local absoluta del folder.
            # No pasar string identificador HuggingFace. local_files_only implícito
            # porque la ruta SÍ es un path absoluto existente = carpeta local.
            model = _WhisperModel_cls(
                local_abs_path_str,
                device=device,
                compute_type=compute_type,
            )
        except (ModelLoadError, ConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001 - safety wrap
            raise ModelLoadError(
                f"FasterWhisperLocalModelLoader error cargando modelo desde carpeta "
                f"{model_folder!r}. Detalle: {type(exc).__name__}: {exc}"
            ) from exc
        return model


# =============================================================================
# ADAPTER PRINCIPAL PERF = Faster-Whisper Small int8_float16 + CT2 local
# =============================================================================
class FasterWhisperSmallASRAdapter:
    """Adapter ASR Step 06 PERF. Implementa ASRPort Protocol."""

    __slots__ = (
        "_loader",
        "_cached_model",
        "_model_loaded_flag",
        "_device_resolver",
        "_compute_type",
        "__weakref__",
    )

    def __init__(
        self,
        loader: FasterWhisperLocalModelLoader | None = None,
        *,
        device_resolver: Any | None = None,
        compute_type: str = "int8_float16",
        model_folder: str | None = None,
    ) -> None:
        self._loader = loader or FasterWhisperLocalModelLoader(folder_name=model_folder)
        self._cached_model: Any = None
        self._model_loaded_flag: bool = False
        self._device_resolver = device_resolver
        self._compute_type = compute_type

    # ---------------------------------------------------------------------
    # Lazy model cache
    # ---------------------------------------------------------------------
    @property
    def model_is_loaded(self) -> bool:
        return self._model_loaded_flag

    def _resolve_device(self) -> str:
        if self._device_resolver is not None:
            return str(self._device_resolver())
        try:
            from app.core.device import resolve_runtime_device_for_profile
            from app.core.constants import QualityProfile

            return resolve_runtime_device_for_profile(QualityProfile.PERFORMANCE)
        except Exception:  # noqa: BLE001 - safety fallback CPU
            return "cpu"

    def _get_or_load_model(self) -> Any:
        if self._model_loaded_flag and self._cached_model is not None:
            return self._cached_model
        folder = self._loader._resolve_expected_folder_absolute()
        device = self._resolve_device()
        model = self._loader.load(folder, device=device, compute_type=self._compute_type)
        self._cached_model = model
        self._model_loaded_flag = True
        return model

    def was_model_loaded_in_this_adapter(self) -> bool:
        return self._model_loaded_flag

    # ---------------------------------------------------------------------
    # ASRPort.transcribe
    # ---------------------------------------------------------------------
    def transcribe_array(
        self,
        audio_f32: Any,
        *,
        offset_ms: int = 0,
        language: str | None = None,
        thresholds: Any | None = None,
        start_index: int = 0,
    ) -> list[Any]:
        """Transcribe un array float32 [-1,1] ya en memoria.

        Primitiva del procesamiento progresivo: el llamante trocea el audio y
        sólo se paga la carga del modelo una vez (cacheada en la instancia).
        Devuelve segmentos de dominio con timestamps ABSOLUTOS.
        """
        from app.domain.value_objects.asr import AsrThresholds

        if audio_f32 is None or getattr(audio_f32, "size", 0) == 0:
            return []
        eff_thr = thresholds if isinstance(thresholds, AsrThresholds) else AsrThresholds()
        model = self._get_or_load_model()
        beam_size = max(1, int(getattr(eff_thr, "beam_size", 1) or 1))
        want_words = bool(getattr(eff_thr, "word_timestamps", False))
        try:
            segments_iter, _info = model.transcribe(
                audio_f32,
                language=language,
                beam_size=beam_size,
                vad_filter=False,
                word_timestamps=want_words,
                without_timestamps=False,
                task="transcribe",
                condition_on_previous_text=True,
            )
            raw_list = list(segments_iter)
        except (ModelLoadError, ConfigurationError, ASRProcessingError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ASRProcessingError(
                f"FasterWhisperSmallASRAdapter transcribe_array error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return [
            _segment_from_raw(raw, int(start_index) + position, int(offset_ms))
            for position, raw in enumerate(raw_list)
        ]

    def transcribe(
        self,
        preprocessed_audio: Any,
        *,
        vad_result: Any,
        lid_result: Any,
        thresholds: Any | None = None,
        job_id: Any | None = None,
    ) -> Any:
        # Late imports para no romper imports de archivos Domain
        from app.domain.entities.asr import ASRResult, ASRSegment
        from app.domain.value_objects.asr import AsrThresholds
        import numpy as _np
        import soundfile as _sf

        # 1) None checks (redundante con UseCase pero defensive)
        if preprocessed_audio is None:
            raise ASRProcessingError("preprocessed_audio None en adapter ASR")
        if lid_result is None:
            raise ASRProcessingError("lid_result None en adapter ASR")
        if vad_result is None:
            raise ASRProcessingError("vad_result None en adapter ASR")
        eff_thr: AsrThresholds = thresholds if isinstance(thresholds, AsrThresholds) else AsrThresholds()

        # 2) D#9 Silent fallback ya se gestionó en UseCase. Pero doble check:
        if (
            int(getattr(vad_result, "num_intervals", -1)) == 0
            and getattr(eff_thr, "allow_empty_transcript_when_no_voice", True)
        ):
            raise ASRProcessingError(
                "Adapter ASR recibió num_intervals==0 pero UseCase debió construir silent fallback. "
                "Chain of control corrupto."
            )

        wav_path = getattr(preprocessed_audio, "wav_path", None)
        if wav_path is None:
            raise ASRProcessingError("PreprocessedAudio no tiene wav_path")

        # 4) Idioma: T05 language_code (si != "und") → explicit. "und" → None (auto-detect decoder)
        t05_lang_code: str = str(getattr(lid_result, "language_code", "und"))
        decoder_language: str | None = None if t05_lang_code == "und" else t05_lang_code

        # 5) Cargar modelo LAZY
        model = self._get_or_load_model()

        beam_size = int(getattr(eff_thr, "beam_size", 1) or 1)
        if beam_size < 1:
            beam_size = 1
        want_words = bool(getattr(eff_thr, "word_timestamps", False))
        chunk_by_vad = bool(getattr(eff_thr, "chunk_by_vad", False))
        intervals = list(getattr(vad_result, "voice_intervals", ()) or ())
        use_chunks = chunk_by_vad and len(intervals) > 0
        strategy = "only_voice_concat" if use_chunks else "full_audio"

        pieces: list[tuple[list[Any], Any, int]] = []
        try:
            if use_chunks:
                for interval in intervals:
                    audio_f32, offset_ms = _read_wav_window(
                        _sf, _np, wav_path, int(interval.start_ms), int(interval.end_ms)
                    )
                    if audio_f32.size == 0:
                        continue
                    segments_iter, info = model.transcribe(
                        audio_f32,
                        language=decoder_language,
                        beam_size=beam_size,
                        vad_filter=False,
                        word_timestamps=want_words,
                        without_timestamps=False,
                        task="transcribe",
                        condition_on_previous_text=True,
                    )
                    pieces.append((list(segments_iter), info, offset_ms))
            else:
                audio_f32 = _read_wav_full(_sf, _np, wav_path)
                segments_iter, info = model.transcribe(
                    audio_f32,
                    language=decoder_language,
                    beam_size=beam_size,
                    vad_filter=False,
                    word_timestamps=want_words,
                    without_timestamps=False,
                    task="transcribe",
                    condition_on_previous_text=True,
                )
                pieces.append((list(segments_iter), info, 0))
        except (ModelLoadError, ConfigurationError, ASRProcessingError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise ASRProcessingError(
                f"FasterWhisperSmallASRAdapter transcribe error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        domain_segments: list[ASRSegment] = []
        running_avg_logprob: float = 0.0
        cnt_with_logprob = 0
        info = pieces[-1][1] if pieces else None
        for raw_list, _info, offset_ms in pieces:
            for raw_seg in raw_list:
                built = _segment_from_raw(raw_seg, len(domain_segments), offset_ms)
                domain_segments.append(built)
                if built.avg_logprob is not None:
                    running_avg_logprob += float(built.avg_logprob)
                    cnt_with_logprob += 1
        num_segments = len(domain_segments)

        # 8) Transcript completo: concatenar texto de segmentos
        transcript_text = "".join(seg.text for seg in domain_segments).strip()

        # 9) Confianza promedio normalizada [0,1]
        #    Mapeo monotónico determinista:
        #      avg_logprob habitual (-6 .. 0) → 0 .. 1
        #      confidence = clip( (avg + 6) / 6 , 0, 1)
        if num_segments > 0 and cnt_with_logprob > 0:
            mean_avg_lp = running_avg_logprob / cnt_with_logprob
            confidence_raw = (mean_avg_lp + 6.0) / 6.0
            confidence = float(min(max(confidence_raw, 0.0), 1.0))
        else:
            confidence = 0.0

        # 10) Language usado realmente por decoder: viene en info.language
        transcript_language_code: str
        inferred_lang = getattr(info, "language", None) if info is not None else None
        if isinstance(inferred_lang, str) and len(inferred_lang) in (2, 3):
            transcript_language_code = inferred_lang.lower()
        elif isinstance(decoder_language, str):
            transcript_language_code = decoder_language.lower()
        else:
            transcript_language_code = "und"

        analysis_metadata: dict[str, Any] = {
            "faster_whisper_language_detected": inferred_lang if isinstance(inferred_lang, str) else None,
            "faster_whisper_language_used_input": decoder_language,
            "beam_size": beam_size,
            "word_timestamps": want_words,
            "chunk_by_vad": use_chunks,
            "num_raw_segments_fw": num_segments,
            "segments_with_avg_logprob": cnt_with_logprob,
        }

        return ASRResult.build_chain(
            preprocessed_audio=preprocessed_audio,
            vad_result=vad_result,
            lid_result=lid_result,
            transcript_text=transcript_text,
            transcript_language_code=transcript_language_code,
            confidence=confidence,
            segments=tuple(domain_segments),
            num_segments=num_segments,
            strategy=strategy,
            model_label=MODEL_LABEL,
            thresholds_used=eff_thr,
            job_id=job_id,
            analysis_metadata=analysis_metadata,
        )
