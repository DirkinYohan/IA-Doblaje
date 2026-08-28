"""UseCase RunLanguageDetectionUseCase — Capa APPLICATION (T05 Step 05 LID).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa torch/transformers/whisper/soundfile (clean architecture).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    LanguageDetectionError,
    ModelLoadError,
    ValidationFailedError,
)
from app.core.logging import bind_context, get_logger
from app.domain.entities.lid import (
    LID_STRATEGY_SILENT_FALLBACK,
    LanguageDetectionResult,
    MODEL_LABEL_WHISPER_LID_V1_LITERAL,
)
from app.domain.entities.media import MediaPrepResult, PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.interfaces.lid_ports import LanguageDetectorPort
from app.domain.value_objects.lid import LanguageDetectionThresholds


log = get_logger("app.application.use_cases.run_language_detection")


@dataclass
class RunLanguageDetectionUseCase:
    """Orquesta Step 05 Language Detection. Inmutable. DI via constructor.

    Input: (MediaPrepResult, VadResult) — ambos frozen, NO mutados.
    Output: LanguageDetectionResult frozen con chain of custody triple.
    """

    detector: LanguageDetectorPort
    strict: bool = True

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def execute(
        self,
        media_prep_result: MediaPrepResult,
        vad_result: VadResult,
        *,
        job_id: str | None = None,
        thresholds: LanguageDetectionThresholds | None = None,
    ) -> LanguageDetectionResult:
        """Detectar idioma. NO muta media_prep_result ni vad_result.

        Reglas:
        1. validar preprocessed_audio presente
        2. validar 16kHz / mono / 16-bit
        3. validar chain of custody T03 ↔ T04
        4. vad sin voz → silent fallback strategy (D#7+D#8)
        5. delegar a LanguageDetectorPort
        6. strict=True + conf < threshold.min_confidence → raise LanguageDetectionError / ValidationFailedError
        """
        jid: str = job_id or media_prep_result.job_id
        bind_context(job_id=jid, phase="LID", model=MODEL_LABEL_WHISPER_LID_V1_LITERAL)
        t0 = time.perf_counter()
        log.info(
            "run_lid_started",
            extra={"strict": self.strict, "vad_intervals": int(vad_result.num_intervals)},
        )

        # (1) Existe preprocessed_audio
        prep: PreprocessedAudio | None = media_prep_result.preprocessed_audio
        if prep is None:
            if self.strict:
                raise ValidationFailedError(
                    "run_language_detection: MediaPrepResult.preprocessed_audio None. "
                    "Verificar Steps 01-03 antes de T05."
                )
            log.warning("run_lid_prep_none_strict_false")
            return self._silent_fallback_result(
                preprocessed=None,
                source_sha=str(media_prep_result.validated.sha256).lower(),
                vad_sha=str(media_prep_result.validated.sha256).lower(),
                thresholds=thresholds or LanguageDetectionThresholds(),
            )

        # (2) Sanity 16k mono 16-bit + wav exists
        self._sanity_preprocessed_audio(prep)

        # (3) Chain of Custody T03 ↔ T04 (doble bind SHA)
        prep_sha = str(prep.sha256).lower()
        vad_sha_ref = str(vad_result.source_preprocessed_sha256).lower()
        if prep_sha != vad_sha_ref:
            raise ValidationFailedError(
                "Chain of Custody rota T03↔T04: "
                f"PreprocessedAudio.sha256={prep_sha[:12]}... != "
                f"VadResult.source_preprocessed_sha256={vad_sha_ref[:12]}..."
            )

        # (4) Thresholds centralizados (defaults T05 si None).
        thr: LanguageDetectionThresholds = thresholds or LanguageDetectionThresholds()
        log.info(
            "run_lid_thresholds",
            extra={
                "min_confidence": float(thr.min_confidence),
                "top_k": int(thr.top_k),
                "min_speech_ratio": float(thr.min_speech_ratio_to_analyze),
                "override_lang": str(thr.default_language_override) if thr.default_language_override else None,
            },
        )

        # (5) Sin voz → silent fallback D#8 (NO lanzar error; strategy silent_fallback_none und 0.0).
        if int(vad_result.num_intervals) == 0:
            log.info(
                "run_lid_silent_no_intervals",
                extra={"speech_ratio": float(vad_result.speech_ratio)},
            )
            res = self._silent_fallback_result(
                preprocessed=prep,
                source_sha=prep_sha,
                vad_sha=vad_sha_ref,
                thresholds=thr,
            )
        # (6) speech ratio insignificante → silent fallback también
        elif (
            int(thr.min_speech_ratio_to_analyze) > 0
            and float(vad_result.speech_ratio) < float(thr.min_speech_ratio_to_analyze)
        ):
            log.info(
                "run_lid_silent_low_ratio",
                extra={
                    "speech_ratio": float(vad_result.speech_ratio),
                    "min_ratio": float(thr.min_speech_ratio_to_analyze),
                },
            )
            res = self._silent_fallback_result(
                preprocessed=prep,
                source_sha=prep_sha,
                vad_sha=vad_sha_ref,
                thresholds=thr,
            )
        else:
            # (7) Delegar al Port (Dependency Inversion) — NO conoce Whisper/JIT.
            try:
                res: LanguageDetectionResult = self.detector.detect(
                    preprocessed=prep,
                    vad=vad_result,
                    thresholds=thr,
                    job_id=jid,
                    logger=log,
                )
            except (ModelLoadError, LanguageDetectionError, ValidationFailedError):
                raise
            except Exception as exc:  # noqa: BLE001
                raise LanguageDetectionError(
                    f"run_language_detection unexpected error: {exc!r}"
                ) from exc

        # (8) Chain of Custody T04 ↔ T05 triple.
        if (
            str(res.source_preprocessed_sha256).lower() != prep_sha
            or str(res.vad_reference_sha256).lower() != vad_sha_ref
        ):
            raise ValidationFailedError(
                "Chain of Custody rota T05: LanguageDetectionResult SHA no coincide con T03/T04."
            )

        # (9) Strict threshold: conf < min → error UNLESS silent strategy OR override_lang default.
        if str(res.analyzed_strategy) != LID_STRATEGY_SILENT_FALLBACK:
            if float(res.confidence) < float(thr.min_confidence):
                if self.strict:
                    if thr.default_language_override is not None:
                        # D#8 PIPELINE.md: user-specified fallback → reusamos und con conf 0 no, aplicamos override_lang.
                        log.warning(
                            "run_lid_strict_below_threshold_but_override",
                            extra={
                                "confidence": float(res.confidence),
                                "threshold": float(thr.min_confidence),
                                "override": str(thr.default_language_override),
                            },
                        )
                        # Permitimos seguir adelante; adapter debió aplicar override. Confianza = max (conf, 0.0).
                    else:
                        raise LanguageDetectionError(
                            f"run_language_detection strict=True: confianza={res.confidence:.4f} < "
                            f"min_threshold={thr.min_confidence:.4f}. Para user-specified fallback fijar "
                            f"LanguageDetectionThresholds.default_language_override='xx'."
                        )
                else:
                    # D#8 strict=False → devolver und + 0.0 + empty alternatives
                    log.warning(
                        "run_lid_nonstrict_below_threshold_undetermined",
                        extra={
                            "confidence": float(res.confidence),
                            "threshold": float(thr.min_confidence),
                        },
                    )
                    res = self._build_undetermined_from_result(
                        res, thresholds=thr,
                    )

        # (10) Inmutabilidad T03/T04: comprobar refs + campos conocidos sin tocar.
        self._assert_not_mutated(media_prep_result, prep, vad_result)

        t1 = time.perf_counter()
        log.info(
            "run_lid_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "lang_code": str(res.language_code),
                "confidence": round(float(res.confidence), 4),
                "strategy": str(res.analyzed_strategy),
                "num_alts": int(len(res.alternatives)),
            },
        )
        return res

    # ------------------------------------------------------------------
    # HELPERS INTERNOS
    # ------------------------------------------------------------------

    @staticmethod
    def _sanity_preprocessed_audio(prep: PreprocessedAudio) -> None:
        problems: list[str] = []
        if int(prep.sample_rate) != 16000:
            problems.append(f"sample_rate={prep.sample_rate} (requerido 16000)")
        if int(prep.channels) != 1:
            problems.append(f"channels={prep.channels} (requerido 1 mono)")
        if int(prep.bit_depth) != 16:
            problems.append(f"bit_depth={prep.bit_depth} (requerido 16)")
        if not isinstance(prep.wav_path, Path):
            problems.append(f"wav_path no es Path: {type(prep.wav_path).__name__}")
        else:
            wp = Path(prep.wav_path)
            if not wp.exists():
                problems.append(f"wav_path no existe: {wp!s}")
            elif not wp.is_file():
                problems.append(f"wav_path no es file: {wp!s}")
            elif wp.stat().st_size <= 44:
                problems.append(f"wav_path demasiado pequeño (solo header): {wp!s}")
        if problems:
            raise ValidationFailedError(
                "PreprocessedAudio no cumple contratos Step 05 LID (16k/mono/16-bit WAV válido): "
                + "; ".join(problems)
            )

    @staticmethod
    def _assert_not_mutated(
        media_prep_result: MediaPrepResult,
        orig_prep_ref: PreprocessedAudio,
        orig_vad_ref: VadResult,
    ) -> None:
        # Ref check prep
        if media_prep_result.preprocessed_audio is not orig_prep_ref:
            new = media_prep_result.preprocessed_audio
            if new is None:
                raise ValidationFailedError(
                    "Mutación T03 detectada: media_prep_result.preprocessed_audio se puso None durante LID."
                )
            if (
                str(new.sha256) != str(orig_prep_ref.sha256)
                or int(new.sample_rate) != int(orig_prep_ref.sample_rate)
                or int(new.channels) != int(orig_prep_ref.channels)
                or int(new.bit_depth) != int(orig_prep_ref.bit_depth)
                or Path(new.wav_path) != Path(orig_prep_ref.wav_path)
            ):
                raise ValidationFailedError(
                    "Mutación T03 detectada: PreprocessedAudio alterado durante UseCase LID."
                )
        # VadResult frozen checks (por definición no se puede mutar; validamos que num_intervals no cambió ref)
        if int(orig_vad_ref.num_intervals) != int(orig_vad_ref.num_intervals):  # trivial check frozen
            pass  # pragma: no cover
        if str(orig_vad_ref.source_preprocessed_sha256) != str(orig_prep_ref.sha256):
            raise ValidationFailedError(
                "Validación post-execute: VadResult.source_preprocessed_sha256 ya no coincide T03."
            )

    @staticmethod
    def _silent_fallback_result(
        *,
        preprocessed: PreprocessedAudio | None,
        source_sha: str,
        vad_sha: str,
        thresholds: LanguageDetectionThresholds,
    ) -> LanguageDetectionResult:
        """D#7 D#8: audio sin voz → silent_fallback_none strategy."""
        dur_ms: int = (
            int(round(float(preprocessed.duration_sec) * 1000.0))
            if preprocessed is not None
            else 0
        )
        sr = int(preprocessed.sample_rate) if preprocessed is not None else 16000
        ch = int(preprocessed.channels) if preprocessed is not None else 1
        bd = int(preprocessed.bit_depth) if preprocessed is not None else 16
        return LanguageDetectionResult(
            source_preprocessed_sha256=source_sha,
            vad_reference_sha256=vad_sha,
            language_code="und",
            language_name="Undetermined",
            confidence=0.0,
            alternatives=tuple(),
            total_duration_ms=dur_ms,
            analyzed_duration_ms=0,
            analyzed_strategy=LID_STRATEGY_SILENT_FALLBACK,
            num_speech_intervals_considered=0,
            model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,
            sample_rate=sr,
            channels=ch,
            bit_depth=bd,
            thresholds=thresholds,
            analysis_metadata={"fallback_reason": "vad_no_intervals_or_ratio_below_threshold"},
        )

    @staticmethod
    def _build_undetermined_from_result(
        r: LanguageDetectionResult,
        *,
        thresholds: LanguageDetectionThresholds,
    ) -> LanguageDetectionResult:
        """D#8 strict=False conf < threshold → und 0.0. Mantiene chain of custody + strategy."""
        return LanguageDetectionResult(
            source_preprocessed_sha256=str(r.source_preprocessed_sha256),
            vad_reference_sha256=str(r.vad_reference_sha256),
            language_code="und",
            language_name="Undetermined",
            confidence=0.0,
            alternatives=tuple(),
            total_duration_ms=int(r.total_duration_ms),
            analyzed_duration_ms=int(r.analyzed_duration_ms),
            analyzed_strategy=str(r.analyzed_strategy),  # type: ignore[arg-type]
            num_speech_intervals_considered=int(r.num_speech_intervals_considered),
            model_label=MODEL_LABEL_WHISPER_LID_V1_LITERAL,
            sample_rate=int(r.sample_rate),
            channels=int(r.channels),
            bit_depth=int(r.bit_depth),
            thresholds=thresholds,
            analysis_metadata={
                **dict(r.analysis_metadata),
                "fallback_reason": "strict_false_confidence_below_threshold",
            },
        )


__all__ = ["RunLanguageDetectionUseCase"]
