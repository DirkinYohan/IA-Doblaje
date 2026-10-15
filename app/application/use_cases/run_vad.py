"""UseCase RunVoiceActivityDetection — Capa APPLICATION (T04 Step 04 VAD).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa torch/silero/soundfile/subprocess (clean architecture).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    ConfigurationError,
    ModelLoadError,
    VADProcessingError,
    ValidationFailedError,
)
from app.core.logging import bind_context, get_logger
from app.domain.entities.media import MediaPrepResult, PreprocessedAudio
from app.domain.entities.vad import MODEL_LABEL_SILERO_V5_1_LITERAL, VadResult
from app.domain.interfaces.vad_ports import VoiceActivityDetectorPort
from app.domain.value_objects.vad import (
    SilenceSegment,
    VadThresholds,
    VoiceInterval,
)


log = get_logger("app.application.use_cases.run_vad")


@dataclass
class RunVoiceActivityDetectionUseCase:
    """Orquesta Step 04 VAD. Inmutable. DI via constructor."""

    vad_detector: VoiceActivityDetectorPort

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def execute(
        self,
        media_prep_result: MediaPrepResult,
        *,
        job_id: str | None = None,
        thresholds: VadThresholds | None = None,
        strict: bool = True,
    ) -> VadResult:
        """Ejecutar VAD sobre media_prep_result.preprocessed_audio (T03).

        NO muta media_prep_result (T03 protegido). Return VadResult chain-of-custody.
        """
        # (0) context logging
        jid: str = job_id or media_prep_result.job_id
        bind_context(job_id=jid, phase="VAD", model=MODEL_LABEL_SILERO_V5_1_LITERAL)

        t0 = time.perf_counter()
        log.info(
            "run_vad_started",
            extra={"profile": str(media_prep_result.profile), "strict": strict},
        )

        # (1) Sanity exist preprocessed_audio
        prep = media_prep_result.preprocessed_audio
        if prep is None:
            if strict:
                raise VADProcessingError(
                    "run_vad: preprocessed_audio es None. Verificar Step 03 (T03) antes de ejecutar Step 04."
                )
            log.warning("run_vad_empty_preprocessed_strict_false", extra={"strict": strict})
            return self._empty_result(
                preprocessed=None,
                source_sha=media_prep_result.validated.sha256,
                thresholds=thresholds or VadThresholds(),
            )

        # (2) Sanity formato 16k mono 16-bit
        self._sanity_preprocessed_audio(prep, strict=strict)

        # (3) Thresholds centralizados (defaults T04 si None).
        thr: VadThresholds = thresholds or VadThresholds()
        log.info(
            "run_vad_thresholds_used",
            extra={
                "speech_threshold": float(thr.speech_threshold),
                "min_speech_ms": int(thr.min_speech_duration_ms),
                "min_silence_ms": int(thr.min_silence_between_ms),
                "merge_proximal_ms": int(thr.merge_proximal_ms),
            },
        )

        # (4) Delegar al Port (Dependency Inversion)
        try:
            vad_result: VadResult = self.vad_detector.detect(
                preprocessed=prep,
                thresholds=thr,
                job_id=jid,
                logger=log,
            )
        except (ModelLoadError, VADProcessingError, ValidationFailedError, ConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise VADProcessingError(
                f"run_vad unexpected error: {exc!r}"
            ) from exc

        # (5) Chain of Custody SHA-256 CHECK
        if str(vad_result.source_preprocessed_sha256).lower() != str(prep.sha256).lower():
            raise ValidationFailedError(
                "Chain of Custody rota: VadResult.source_preprocessed_sha256 != "
                f"PreprocessedAudio.sha256. ({vad_result.source_preprocessed_sha256} != {prep.sha256})"
            )

        # (6) Inmutabilidad T03: Asegurar que no se modificó MediaPrepResult (deep copy igual)
        # (Sólo un check defensivo; Python no lo garantiza pero detectamos asignaciones attrs conocidos.)
        self._assert_not_mutated(media_prep_result, prep)

        t1 = time.perf_counter()
        log.info(
            "run_vad_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "num_intervals": int(vad_result.num_intervals),
                "num_silences": int(vad_result.num_silences),
                "speech_ratio": round(float(vad_result.speech_ratio), 4),
                "total_speech_ms": int(vad_result.total_speech_ms),
            },
        )
        return vad_result

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _sanity_preprocessed_audio(prep: PreprocessedAudio, strict: bool) -> None:
        problems: list[str] = []
        if int(prep.sample_rate) != 16000:
            problems.append(f"sample_rate={prep.sample_rate} (necesario 16000)")
        if int(prep.channels) != 1:
            problems.append(f"channels={prep.channels} (necesario 1)")
        if int(prep.bit_depth) != 16:
            problems.append(f"bit_depth={prep.bit_depth} (necesario 16)")
        if not isinstance(prep.wav_path, Path):
            problems.append(f"wav_path no es Path: {type(prep.wav_path).__name__}")
        else:
            wp = Path(prep.wav_path)
            if not wp.exists():
                problems.append(f"wav_path no existe: {wp!s}")
            elif not wp.is_file():
                problems.append(f"wav_path no es archivo regular: {wp!s}")
            elif wp.stat().st_size <= 0:
                problems.append(f"wav_path size=0: {wp!s}")
        if problems:
            msg = "PreprocessedAudio (T03) no cumple contratos Step 04: " + "; ".join(problems)
            if strict:
                raise VADProcessingError(msg)
            log.warning("run_vad_sanity_issues_strict_false", extra={"problems": problems})

    @staticmethod
    def _assert_not_mutated(
        media_prep_result: MediaPrepResult,
        orig_prep_ref: PreprocessedAudio,
    ) -> None:
        if media_prep_result.preprocessed_audio is not orig_prep_ref:
            # El adapter reasignó el mismo valor por alguna razón? No permitimos que altere nada.
            new = media_prep_result.preprocessed_audio
            if new is None:
                raise ValidationFailedError(
                    "Mutación detectada T03: media_prep_result.preprocessed_audio se puso None durante VAD."
                )
            if (
                str(new.sha256) != str(orig_prep_ref.sha256)
                or int(new.sample_rate) != int(orig_prep_ref.sample_rate)
                or int(new.channels) != int(orig_prep_ref.channels)
                or int(new.bit_depth) != int(orig_prep_ref.bit_depth)
                or Path(new.wav_path) != Path(orig_prep_ref.wav_path)
            ):
                raise ValidationFailedError(
                    "Mutación detectada T03: PreprocessedAudio alterado dentro del UseCase VAD."
                )

    @staticmethod
    def _empty_result(
        *,
        preprocessed: PreprocessedAudio | None,
        source_sha: str,
        thresholds: VadThresholds,
    ) -> VadResult:
        dur_ms: int = int(round(float(preprocessed.duration_sec) * 1000.0)) if preprocessed is not None else 0
        sr = int(preprocessed.sample_rate) if preprocessed is not None else 16000
        ch = int(preprocessed.channels) if preprocessed is not None else 1
        return VadResult(
            source_preprocessed_sha256=source_sha,
            voice_intervals=tuple(),
            silence_segments=(
                tuple()
                if dur_ms <= 0
                else (
                    SilenceSegment(
                        start_ms=0, end_ms=dur_ms, speech_index_before=-1, speech_index_after=-1
                    ),
                )
            ),
            speech_ratio=0.0,
            total_speech_ms=0,
            total_silence_ms=dur_ms,
            num_intervals=0,
            num_silences=0 if dur_ms <= 0 else 1,
            thresholds=thresholds,
            model_label=MODEL_LABEL_SILERO_V5_1_LITERAL,
            sample_rate=sr,
            channels=ch,
            duration_ms=dur_ms,
        )


__all__ = ["RunVoiceActivityDetectionUseCase"]
