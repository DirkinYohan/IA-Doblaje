from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Final, Optional

WHISPER_MULTILINGUAL_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("zh", "Chinese"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ru", "Russian"),
    ("ko", "Korean"),
    ("fr", "French"),
    ("ja", "Japanese"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("ca", "Catalan"),
    ("nl", "Dutch"),
    ("ar", "Arabic"),
    ("sv", "Swedish"),
    ("it", "Italian"),
    ("id", "Indonesian"),
    ("hi", "Hindi"),
    ("fi", "Finnish"),
    ("vi", "Vietnamese"),
    ("he", "Hebrew"),
    ("uk", "Ukrainian"),
    ("el", "Greek"),
    ("ms", "Malay"),
    ("cs", "Czech"),
    ("ro", "Romanian"),
    ("da", "Danish"),
    ("hu", "Hungarian"),
    ("ta", "Tamil"),
    ("no", "Norwegian"),
    ("th", "Thai"),
    ("ur", "Urdu"),
    ("hr", "Croatian"),
    ("bg", "Bulgarian"),
    ("la", "Latin"),
    ("mi", "Maori"),
    ("ml", "Malayalam"),
    ("cy", "Welsh"),
    ("sk", "Slovak"),
    ("te", "Telugu"),
    ("fa", "Persian"),
    ("lv", "Latvian"),
    ("bn", "Bengali"),
    ("sr", "Serbian"),
    ("az", "Azerbaijani"),
    ("sl", "Slovenian"),
    ("kn", "Kannada"),
    ("et", "Estonian"),
    ("mk", "Macedonian"),
    ("br", "Breton"),
    ("eu", "Basque"),
    ("is", "Icelandic"),
    ("hy", "Armenian"),
    ("ne", "Nepali"),
    ("mn", "Mongolian"),
    ("bs", "Bosnian"),
    ("kk", "Kazakh"),
    ("sq", "Albanian"),
    ("sw", "Swahili"),
    ("gl", "Galician"),
    ("mr", "Marathi"),
    ("pa", "Punjabi"),
    ("si", "Sinhala"),
    ("km", "Khmer"),
    ("sn", "Shona"),
    ("yo", "Yoruba"),
    ("so", "Somali"),
    ("af", "Afrikaans"),
    ("oc", "Occitan"),
    ("ka", "Georgian"),
    ("be", "Belarusian"),
    ("tg", "Tajik"),
    ("sd", "Sindhi"),
    ("gu", "Gujarati"),
    ("am", "Amharic"),
    ("yi", "Yiddish"),
    ("lo", "Lao"),
    ("uz", "Uzbek"),
    ("fo", "Faroese"),
    ("ht", "Haitian"),
    ("ps", "Pashto"),
    ("tk", "Turkmen"),
    ("nn", "Norwegian Nynorsk"),
    ("mt", "Maltese"),
    ("sa", "Sanskrit"),
    ("lb", "Luxembourgish"),
    ("my", "Burmese"),
    ("bo", "Tibetan"),
    ("tl", "Tagalog"),
    ("mg", "Malagasy"),
    ("as", "Assamese"),
    ("tt", "Tatar"),
    ("haw", "Hawaiian"),
    ("ln", "Lingala"),
    ("ha", "Hausa"),
    ("ba", "Bashkir"),
    ("jw", "Javanese"),
    ("su", "Sundanese"),
    ("und", "Undetermined"),
)

NUM_LID_CLASSES: Final[int] = len(WHISPER_MULTILINGUAL_LANGUAGES)
assert NUM_LID_CLASSES == 99

_CODE_TO_NAME: dict[str, str] = {c: n for c, n in WHISPER_MULTILINGUAL_LANGUAGES}


def language_name_for_code(code: str) -> str:
    c = str(code).strip().lower()
    if c == "und":
        return "Undetermined"
    if c in _CODE_TO_NAME:
        return _CODE_TO_NAME[c]
    return c.title()


MODEL_FILENAME: Final[str] = "whisper_encoder_small_multilingual_lid_v1.jit"

_MIN_MODEL_BYTES: Final[int] = 100 * 1024 * 1024


class WhisperLidModelLoader:
    def __init__(self, device_override: Optional[str] = None) -> None:
        self._device_override = device_override

    def load(self, model_path: Path, device: str) -> Any:
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

        try:
            import torch  # noqa: F401
        except ImportError as exc:  # pragma: no cover - torch dep core
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                "PyTorch no instalado. Verificar dependencias IA-Doblaje Engine core. "
                "NO descargar modelo automáticamente (regla T05)."
            ) from exc

        try:
            import torch

            dev = self._device_override if self._device_override is not None else str(device)
            from app.infrastructure.audio.jit_loader import load_jit_confined

            model = load_jit_confined(torch, mp, dev)
            model.eval()
            try:
                import torch as _t

                for p in model.parameters(False):
                    p.requires_grad_(False)
            except Exception:  # noqa: BLE001
                pass
            return model
        except Exception as exc:
            from app.core.exceptions import ModelLoadError as _MLE

            raise _MLE(
                "Whisper LID model torch.jit.load falló. "
                "Verificar archivo JIT correcto y no corrupto. Colocar manualmente. "
                f"NO descargar automáticamente (regla T05). Detalle: {exc!r}"
            ) from exc


class WhisperEncoderLanguageDetectionAdapter:
    def __init__(
        self,
        *,
        settings: Any | None = None,
        model_loader: Any | None = None,
        device_override: str | None = None,
    ) -> None:
        if settings is None:
            from app.core.config import get_app_settings as _get_s

            settings = _get_s()
        self._settings = settings
        self._paths = settings.paths
        self._cached_model: Any = None
        self._cached_model_path: Path | None = None
        self._loader: Any = model_loader if model_loader is not None else WhisperLidModelLoader(
            device_override=device_override
        )
        self._device: str = device_override or self._resolve_default_device()


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
        return Path(self._paths.models_cache_dir) / MODEL_FILENAME

    def _get_model(self) -> Any:
        mp = self._get_model_path()
        if self._cached_model is None or self._cached_model_path != mp:
            self._cached_model = self._loader.load(mp, self._device)
            self._cached_model_path = mp
        return self._cached_model


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

        audio_f32, analyzed_duration_ms, samples_considered_count = (
            self._extract_only_voice_samples(preprocessed, vad)
        )

        model = self._get_model()

        try:
            import numpy as _np
            import torch as _t

            sr_expected = int(preprocessed.sample_rate)
            if audio_f32.size <= 0:
                raise _LDE("Adapter LID: audio_f32 tras concatenar intervals está vacío.")

            max_samples = sr_expected * 3600
            if audio_f32.size > max_samples:
                audio_f32 = audio_f32[:max_samples]

            t_inf_start = time.perf_counter()
            with _t.inference_mode():
                x = _t.from_numpy(audio_f32).to(device=self._device, dtype=_t.float32)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                out = model(x, sr_expected)
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
                logits_stable = logits_np - float(_np.max(logits_np))
                exp = _np.exp(logits_stable)
                probs = exp / float(_np.sum(exp))
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

        try:
            top_k = int(min(int(thr.top_k), NUM_LID_CLASSES))
            indices_desc = _np.argsort(-probs, kind="stable")[:top_k]  # type: ignore[arg-type]
            alternatives_list: list[LanguageAlternative] = []
            rank_counter = 1
            primary_idx = int(indices_desc[0])
            primary_code, primary_name = WHISPER_MULTILINGUAL_LANGUAGES[primary_idx]
            primary_conf = float(probs[primary_idx])
            for idx in indices_desc:
                i = int(idx)
                conf = float(probs[i])
                if i == primary_idx:
                    continue
                if len(alternatives_list) >= (top_k - 1):
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
            final_conf = 0.0
            override_used = True
            alternatives_list = []

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


    def _extract_only_voice_samples(
        self,
        preprocessed: Any,
        vad: Any,
    ) -> tuple[Any, int, int]:
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
        if audio_int16.ndim == 2:
            audio_int16 = audio_int16[:, 0]
        total_samples = audio_int16.shape[0]
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


__all__ = [
    "MODEL_FILENAME",
    "WHISPER_MULTILINGUAL_LANGUAGES",
    "NUM_LID_CLASSES",
    "language_name_for_code",
    "WhisperLidModelLoader",
    "WhisperEncoderLanguageDetectionAdapter",
]
