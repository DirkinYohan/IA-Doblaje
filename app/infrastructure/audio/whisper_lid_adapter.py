"""Infrastructure Adapter Step 05 — Whisper Encoder Small + LID Head TorchScript JIT.

INFRASTRUCTURE ONLY. Aquí sí se importa torch/numpy/soundfile.
Offline 100% local: sin integraciones cloud, sin librerías de fetch remoto.
Modelo cargado LAZY únicamente desde archivo local:
    AppSettings.paths.models_cache_dir / MODEL_FILENAME

Si falta archivo/dependencias → ModelLoadError con instrucción explícita (regla T05).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final, Optional

# ---------------------------------------------------------------------------
# Mapa oficial 99 idiomas Whisper multilingual: index → (ISO 639-1, English name).
# Hardcodeado LITERAL (no archivo externo, no descarga). Mismo orden que OpenAI Whisper.
# ---------------------------------------------------------------------------
WHISPER_MULTILINGUAL_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"),       # 0
    ("zh", "Chinese"),       # 1
    ("de", "German"),        # 2
    ("es", "Spanish"),       # 3
    ("ru", "Russian"),       # 4
    ("ko", "Korean"),        # 5
    ("fr", "French"),        # 6
    ("ja", "Japanese"),      # 7
    ("pt", "Portuguese"),    # 8
    ("tr", "Turkish"),       # 9
    ("pl", "Polish"),        # 10
    ("ca", "Catalan"),       # 11
    ("nl", "Dutch"),         # 12
    ("ar", "Arabic"),        # 13
    ("sv", "Swedish"),       # 14
    ("it", "Italian"),       # 15
    ("id", "Indonesian"),    # 16
    ("hi", "Hindi"),         # 17
    ("fi", "Finnish"),       # 18
    ("vi", "Vietnamese"),    # 19
    ("he", "Hebrew"),        # 20
    ("uk", "Ukrainian"),     # 21
    ("el", "Greek"),         # 22
    ("ms", "Malay"),         # 23
    ("cs", "Czech"),         # 24
    ("ro", "Romanian"),      # 25
    ("da", "Danish"),        # 26
    ("hu", "Hungarian"),     # 27
    ("ta", "Tamil"),         # 28
    ("no", "Norwegian"),     # 29
    ("th", "Thai"),          # 30
    ("ur", "Urdu"),          # 31
    ("hr", "Croatian"),      # 32
    ("bg", "Bulgarian"),     # 33
    ("la", "Latin"),         # 34
    ("mi", "Maori"),         # 35
    ("ml", "Malayalam"),     # 36
    ("cy", "Welsh"),         # 37
    ("sk", "Slovak"),        # 38
    ("te", "Telugu"),        # 39
    ("fa", "Persian"),       # 40
    ("lv", "Latvian"),       # 41
    ("bn", "Bengali"),       # 42
    ("sr", "Serbian"),       # 43
    ("az", "Azerbaijani"),   # 44
    ("sl", "Slovenian"),     # 45
    ("kn", "Kannada"),       # 46
    ("et", "Estonian"),      # 47
    ("mk", "Macedonian"),    # 48
    ("br", "Breton"),        # 49
    ("eu", "Basque"),        # 50
    ("is", "Icelandic"),     # 51
    ("hy", "Armenian"),      # 52
    ("ne", "Nepali"),        # 53
    ("mn", "Mongolian"),     # 54
    ("bs", "Bosnian"),       # 55
    ("kk", "Kazakh"),        # 56
    ("sq", "Albanian"),      # 57
    ("sw", "Swahili"),       # 58
    ("gl", "Galician"),      # 59
    ("mr", "Marathi"),       # 60
    ("pa", "Punjabi"),       # 61
    ("si", "Sinhala"),       # 62
    ("km", "Khmer"),         # 63
    ("sn", "Shona"),         # 64
    ("yo", "Yoruba"),        # 65
    ("so", "Somali"),        # 66
    ("af", "Afrikaans"),     # 67
    ("oc", "Occitan"),       # 68
    ("ka", "Georgian"),      # 69
    ("be", "Belarusian"),    # 70
    ("tg", "Tajik"),         # 71
    ("sd", "Sindhi"),        # 72
    ("gu", "Gujarati"),      # 73
    ("am", "Amharic"),       # 74
    ("yi", "Yiddish"),       # 75
    ("lo", "Lao"),           # 76
    ("uz", "Uzbek"),         # 77
    ("fo", "Faroese"),       # 78
    ("ht", "Haitian"),       # 79
    ("ps", "Pashto"),        # 80
    ("tk", "Turkmen"),       # 81
    ("nn", "Norwegian Nynorsk"),  # 82
    ("mt", "Maltese"),       # 83
    ("sa", "Sanskrit"),      # 84
    ("lb", "Luxembourgish"), # 85
    ("my", "Burmese"),       # 86
    ("bo", "Tibetan"),       # 87
    ("tl", "Tagalog"),       # 88
    ("mg", "Malagasy"),      # 89
    ("as", "Assamese"),      # 90
    ("tt", "Tatar"),         # 91
    ("haw", "Hawaiian"),     # 92  (ISO 639-3)
    ("ln", "Lingala"),       # 93
    ("ha", "Hausa"),         # 94
    ("ba", "Bashkir"),       # 95
    ("jw", "Javanese"),      # 96
    ("su", "Sundanese"),     # 97
    ("und", "Undetermined"), # 98 (sentinel; idx 98 se ignora normalmente en favor del argmax real)
)

NUM_LID_CLASSES: Final[int] = len(WHISPER_MULTILINGUAL_LANGUAGES)
assert NUM_LID_CLASSES == 99

# ISO name reverse lookup index by code
_CODE_TO_NAME: dict[str, str] = {c: n for c, n in WHISPER_MULTILINGUAL_LANGUAGES}


def language_name_for_code(code: str) -> str:
    """Return English name for ISO code. Defaults to code title if unknown."""
    c = str(code).strip().lower()
    if c == "und":
        return "Undetermined"
    if c in _CODE_TO_NAME:
        return _CODE_TO_NAME[c]
    return c.title()


# ---------------------------------------------------------------------------
# Constante única D#3. Nombre físico archivo JIT en models/ — NO configurable.
# ---------------------------------------------------------------------------
MODEL_FILENAME: Final[str] = "whisper_encoder_small_multilingual_lid_v1.jit"

# Minimum reasonable file size for a Whisper small encoder JIT (~250MB typical). We set low bound 100 MB as sanity guard.
_MIN_MODEL_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB


# ===========================================================================
# 1) MODEL LOADER (mismo archivo, alta cohesión igual que T04)
# ===========================================================================


class WhisperLidModelLoader:
    """Carga el JIT localmente. NO descarga. weights_only=True anti pickle-code.

    Implementa la firma LidModelLoaderPort Protocol.
    """

    def __init__(self, device_override: Optional[str] = None) -> None:
        self._device_override = device_override

    # ------------------------------------------------------------------
    def load(self, model_path: Path, device: str) -> Any:
        """(1) existe archivo, (2) size razonable, (3) torch import ok, (4) torch.jit.load local.

        Cualquier fallo → ModelLoadError con mensaje OFFICIAL.
        """
        # (1) Existencia + tipo archivo
        mp = Path(model_path)
        if not mp.exists():
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                "Whisper LID model no encontrado en models/. "
                "Colocar manualmente. NO descargar automáticamente (regla T05). "
                f"Esperado: models/{MODEL_FILENAME}"
            )
        if not mp.is_file():
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                f"Whisper LID model path no es archivo regular: {mp!s}"
            )

        # (2) Size sanity (mínimo 100MB para evitar stub/corruptos)
        try:
            sz = mp.stat().st_size
        except OSError as exc:
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                f"No se puede stat modelo LID: {mp!s} ({exc!r})"
            ) from exc

        if sz <= 0:
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                f"Whisper LID model corrupto size=0: {mp!s}. Colocar manualmente. NO descargar automáticamente (regla T05)."
            )
        if sz < _MIN_MODEL_BYTES:
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                f"Whisper LID model demasiado pequeño ({sz} bytes < {_MIN_MODEL_BYTES}). "
                "Posiblemente corrupto o incompleto. Colocar manualmente. NO descargar (T05)."
            )

        # (3) Dependencia torch disponible (ya está en dependencies core T01; double check)
        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - torch dep core
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                "PyTorch no instalado. Verificar dependencias IA-Doblaje Engine core. "
                "NO descargar modelo automáticamente (regla T05)."
            ) from exc

        # (4) Carga JIT local. weights_only=True anti pickle code injection (mismo T04 Silero)
        try:
            import torch

            dev = self._device_override if self._device_override is not None else str(device)
            # map_location asegura que funciona tanto CPU como CUDA si GPU presente
            from app.infrastructure.audio.jit_loader import load_jit_confined

            model = load_jit_confined(torch, mp, dev)
            model.eval()
            try:
                import torch as _t

                for p in model.parameters(False):
                    p.requires_grad_(False)
            except Exception:  # noqa: BLE001
                # Algunos JIT exportados no tienen parameters(); no es fatal.
                pass
            return model
        except Exception as exc:
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                "Whisper LID model torch.jit.load falló. "
                "Verificar archivo JIT correcto y no corrupto. Colocar manualmente. "
                f"NO descargar automáticamente (regla T05). Detalle: {exc!r}"
            ) from exc


# ===========================================================================
# 2) ADAPTER (LanguageDetectorPort implementación INFRAESTRUCTURA)
# ===========================================================================


class WhisperEncoderLanguageDetectionAdapter:
    """Language Detection via Whisper Encoder Small + LID Head TorchScript JIT.

    Modelo lazy-loaded; no se carga durante __init__ ni import.
    D#7 strategy: ONLY VOICE INTERVALS de VadResult → concatenar samples.
    """

    def __init__(
        self,
        *,
        settings: Any | None = None,
        model_loader: Any | None = None,
        device_override: str | None = None,
    ) -> None:
        # Resolve settings para obtener paths.models_cache_dir
        if settings is None:
            from app.core.config import get_app_settings as _get_s

            settings = _get_s()
        self._settings = settings
        self._paths = settings.paths
        # Lazy cached modelo singleton (cargado la primera vez que _get_model se llama)
        self._cached_model: Any = None
        self._cached_model_path: Path | None = None
        # Loader inyectable DI para tests mock (LidModelLoaderPort Protocol)
        self._loader: Any = model_loader if model_loader is not None else WhisperLidModelLoader(
            device_override=device_override
        )
        self._device: str = device_override or self._resolve_default_device()

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_default_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _get_model_path(self) -> Path:
        """ÚNICO PATH: models_cache_dir / MODEL_FILENAME (D#3). Sin user input."""
        return Path(self._paths.models_cache_dir) / MODEL_FILENAME

    def _get_model(self) -> Any:
        """Carga lazy singleton. Thread-safe enough (atomic assign instance attr post-load)."""
        mp = self._get_model_path()
        if self._cached_model is None or self._cached_model_path != mp:
            self._cached_model = self._loader.load(mp, self._device)
            self._cached_model_path = mp
        return self._cached_model

    # ------------------------------------------------------------------
    # Public API: LanguageDetectorPort interface
    # ------------------------------------------------------------------

    def detect(
        self,
        preprocessed: Any,
        vad: Any,
        *,
        thresholds: Any | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> Any:
        from app.core.exceptions import LanguageDetectionError as _LDE
        from app.domain.entities.lid import (
            LID_STRATEGY_SILENT_FALLBACK,
            LID_STRATEGY_VOICE_ONLY,
            LanguageDetectionResult,
            MODEL_LABEL_WHISPER_LID_V1_LITERAL,
        )
        from app.domain.value_objects.lid import (
            LanguageAlternative,
            LanguageDetectionThresholds,
        )

        log = logger
        t0 = time.perf_counter()
        thr: LanguageDetectionThresholds = (
            thresholds if thresholds is not None else LanguageDetectionThresholds()
        )
        source_sha = str(preprocessed.sha256).lower()
        vad_sha = str(vad.source_preprocessed_sha256).lower()

        total_duration_ms = int(round(float(preprocessed.duration_sec) * 1000.0))
        num_intervals = int(vad.num_intervals)

        # 0 intervals → silent fallback handled upstream por UseCase generalmente.
        # Aquí double check por seguridad.
        if num_intervals == 0:
            return LanguageDetectionResult(
                source_preprocessed_sha256=source_sha,
                vad_reference_sha256=vad_sha,
                language_code="und",
                language_name="Undetermined",
                confidence=0.0,
                alternatives=tuple(),
                total_duration_ms=total_duration_ms,
                analyzed_duration_ms=0,
                analyzed_strategy=LID_STRATEGY_SILENT_FALLBACK,
                num_speech_intervals_considered=0,
                model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,
                sample_rate=int(preprocessed.sample_rate),
                channels=int(preprocessed.channels),
                bit_depth=int(preprocessed.bit_depth),
                thresholds=thr,
                analysis_metadata={"fallback_reason": "adapter_zero_intervals"},
            )

        # ------------------------------------------------------------------
        # (1) LEER WAV y extraer SOLO VOICE INTERVALS (D#7 strategy ONLY VOICE)
        # ------------------------------------------------------------------
        audio_f32, analyzed_duration_ms, samples_considered_count = (
            self._extract_only_voice_samples(preprocessed, vad)
        )

        # (2) Cargar modelo lazy (si falla → ModelLoadError). Se llama ANTES de tocar datos.
        model = self._get_model()

        # (3) Inferencia encoder-only → probs 99 clases.
        try:
            import numpy as _np
            import torch as _t

            sr_expected = int(preprocessed.sample_rate)
            if audio_f32.size <= 0:
                raise _LDE("Adapter LID: audio_f32 tras concatenar intervals está vacío.")

            # Máximo 1 hora anti OOM.
            max_samples = sr_expected * 3600
            if audio_f32.size > max_samples:
                audio_f32 = audio_f32[:max_samples]

            t_inf_start = time.perf_counter()
            with _t.inference_mode():
                x = _t.from_numpy(audio_f32).to(device=self._device, dtype=_t.float32)
                # add batch dim → shape [1, N]
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                # Llamada JIT. El modelo exportado recibe (audio: [B, N], sr: int) y devuelve logits [B, 99]
                out = model(x, sr_expected)
                # Normaliza a probs softmax sobre dim clases (-1)
                if isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out
                if hasattr(logits, "detach"):
                    logits = logits.detach().to("cpu")
                logits_np = _np.asarray(logits, dtype=_np.float32)
                if logits_np.ndim == 2:
                    logits_np = logits_np[0]
                if logits_np.shape[0] != NUM_LID_CLASSES:
                    raise _LDE(
                        f"Adapter LID: modelo JIT salida shape {logits_np.shape} no coincide con NUM_LID_CLASSES={NUM_LID_CLASSES}. "
                        "Verificar modelo correcto whisper_encoder_small_multilingual_lid_v1.jit. NO intentar convertir a ASR."
                    )
                # Subtract max for numerical stability
                logits_stable = logits_np - float(_np.max(logits_np))
                exp = _np.exp(logits_stable)
                probs = exp / float(_np.sum(exp))
                # Reemplazar NaN/Inf por 0 para no romper
                probs = _np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                s = float(_np.sum(probs))
                if s <= 0:
                    raise _LDE("Adapter LID: softmax suma 0 (numéricamente inestable).")
                probs = probs / s
            t_inf_ms = int(round((time.perf_counter() - t_inf_start) * 1000.0))
        except (_LDE,):
            raise
        except Exception as exc:
            raise _LDE(
                f"Adapter LID: inferencia Whisper LID falló inesperadamente: {exc!r}"
            ) from exc

        # (4) Top-K índices ordenados desc
        try:
            top_k = int(min(int(thr.top_k), NUM_LID_CLASSES))
            indices_desc = _np.argsort(-probs, kind="stable")[:top_k]  # type: ignore[arg-type]
            alternatives_list: list[LanguageAlternative] = []
            rank_counter = 1
            primary_idx = int(indices_desc[0])
            primary_code, primary_name = WHISPER_MULTILINGUAL_LANGUAGES[primary_idx]
            primary_conf = float(probs[primary_idx])
            # Iteramos todos los top_k. Primary no va a alternatives (va en campo principal).
            for idx in indices_desc:
                i = int(idx)
                conf = float(probs[i])
                if i == primary_idx:
                    continue
                if len(alternatives_list) >= (top_k - 1):
                    # Top-3 alternatives excluyendo primary = 2 alternativas max si top_k=3.
                    # Permitimos top_k=3 y alternatives size hasta 2 (por lo total top3 incluye primary).
                    # Si top_k=3 y queremos 3 alternatives sin prim => usamos top_k=4. No; PIPELINE.md Top-3 incluye el primary.
                    break
                code, name = WHISPER_MULTILINGUAL_LANGUAGES[i]
                alternatives_list.append(
                    LanguageAlternative(
                        language_code=code,
                        language_name=name,
                        confidence=float(conf),
                        rank=int(rank_counter),
                    )
                )
                rank_counter += 1
        except Exception as exc:
            raise _LDE(f"Adapter LID: post procesamiento Top-K falló: {exc!r}") from exc

        # (5) Override por usuario si primary conf < threshold.default_language_override
        final_code: str = primary_code
        final_name: str = primary_name
        final_conf: float = primary_conf
        override_used: bool = False
        if (
            thr.default_language_override is not None
            and float(primary_conf) < float(thr.min_confidence)
        ):
            ovr_code = str(thr.default_language_override).strip().lower()
            final_code = ovr_code
            final_name = language_name_for_code(ovr_code)
            final_conf = 0.0  # Override no aporta confianza estadística.
            override_used = True
            alternatives_list = []  # No conservar alternatives cuando hay override.

        # (6) Construir LanguageDetectionResult frozen
        elapsed = time.perf_counter() - t0
        try:
            result = LanguageDetectionResult(
                source_preprocessed_sha256=source_sha,
                vad_reference_sha256=vad_sha,
                language_code=final_code,
                language_name=final_name,
                confidence=float(final_conf),
                alternatives=tuple(alternatives_list),
                total_duration_ms=int(total_duration_ms),
                analyzed_duration_ms=int(analyzed_duration_ms),
                analyzed_strategy=LID_STRATEGY_VOICE_ONLY,  # type: ignore[arg-type]
                num_speech_intervals_considered=int(samples_considered_count),
                model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,  # type: ignore[arg-type]
                sample_rate=int(preprocessed.sample_rate),
                channels=int(preprocessed.channels),
                bit_depth=int(preprocessed.bit_depth),
                thresholds=thr,
                analysis_metadata={
                    "inference_ms": int(t_inf_ms),
                    "total_elapsed_ms": int(round(elapsed * 1000.0)),
                    "device": str(self._device),
                    "override_applied": bool(override_used),
                    "num_classes": int(NUM_LID_CLASSES),
                    "topk_indices": ",".join(str(int(i)) for i in indices_desc[:top_k]),  # type: ignore[has-type]
                },
            )
        except Exception as exc:
            raise _LDE(
                f"Adapter LID: construcción LanguageDetectionResult inválido: {exc!r}"
            ) from exc

        if log is not None:
            try:
                log.info(
                    "whisper_lid_adapter_inference_done",
                    extra={
                        "elapsed_ms": int(round(elapsed * 1000.0)),
                        "lang": str(result.language_code),
                        "conf": round(float(result.confidence), 4),
                        "alts": int(len(result.alternatives)),
                        "strategy": str(result.analyzed_strategy),
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        return result

    # ------------------------------------------------------------------
    # _extract_only_voice_samples (D#7 only-voice intervals)
    # ------------------------------------------------------------------

    def _extract_only_voice_samples(
        self,
        preprocessed: Any,
        vad: Any,
    ) -> tuple[Any, int, int]:
        """Leer WAV PCM 16-bit int16. Convertir a float32 [-1,1].

        Retorna:
            audio_concat_f32: np.ndarray shape (M,) float32 [-1,1] solo intervalos voz concatenados
            analyzed_duration_ms: int (ms total concatenados)
            num_intervals_used: int len voice_intervals procesados
        """
        import numpy as _np
        import soundfile as _sf

        wav_path = Path(preprocessed.wav_path)
        sr_want = int(preprocessed.sample_rate)
        audio_int16, sr_read = _sf.read(str(wav_path), dtype="int16", always_2d=False)
        if int(sr_read) != sr_want:
            from app.core.exceptions import LanguageDetectionError as _LDE

            raise _LDE(
                f"Adapter LID sample rate mismatch: WAV={sr_read}, Preprocessed={sr_want}"
            )
        # Shape: si mono int16 (N,) ; si stereo (N,2) siempre mono T03 por lo tanto (N,) ok
        if audio_int16.ndim == 2:
            audio_int16 = audio_int16[:, 0]
        total_samples = audio_int16.shape[0]
        # Convertir a float32 [-1, 1]
        audio_f32 = audio_int16.astype(_np.float32) / 32768.0
        audio_f32 = _np.clip(audio_f32, -1.0, 1.0)

        intervals = list(vad.voice_intervals)
        if len(intervals) == 0:
            return (_np.zeros(0, dtype=_np.float32), 0, 0)

        chunks: list[Any] = []
        analyzed_ms = 0
        used_intervals = 0
        for interval in intervals:
            s_ms = int(interval.start_ms)
            e_ms = int(interval.end_ms)
            s_sample = int(s_ms * sr_want // 1000)
            e_sample = int(e_ms * sr_want // 1000)
            s_sample = max(0, min(s_sample, total_samples))
            e_sample = max(s_sample, min(e_sample, total_samples))
            if e_sample <= s_sample:
                continue
            segment = audio_f32[s_sample:e_sample]
            if segment.size == 0:
                continue
            chunks.append(segment)
            analyzed_ms += int((e_sample - s_sample) * 1000 // sr_want)
            used_intervals += 1

        if not chunks:
            return (_np.zeros(0, dtype=_np.float32), 0, 0)

        concat = _np.concatenate(chunks, axis=0).astype(_np.float32, copy=False)
        return (concat, int(analyzed_ms), int(used_intervals))


# ===========================================================================
# Exports
# ===========================================================================

__all__ = [
    "MODEL_FILENAME",
    "WHISPER_MULTILINGUAL_LANGUAGES",
    "NUM_LID_CLASSES",
    "language_name_for_code",
    "WhisperLidModelLoader",
    "WhisperEncoderLanguageDetectionAdapter",
]
