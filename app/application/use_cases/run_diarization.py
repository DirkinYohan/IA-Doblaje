"""UseCase RunSpeakerDiarizationUseCase — Capa APPLICATION (T08 Step 08 Diarization).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa torch/pyannote/speechbrain/numpy (clean architecture).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import (
    DiarizationError,
    ModelLoadError,
    ValidationFailedError,
)
from app.core.logging import bind_context, get_logger
from app.domain.entities.diarization import (
    DIARIZATION_STRATEGY_SILENT_FALLBACK,
    MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
    DiarizationResult,
)
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.interfaces.diarization_ports import SpeakerDiarizerPort
from app.domain.value_objects.diarization import DiarizationThresholds


log = get_logger("app.application.use_cases.run_diarization")


@dataclass(frozen=True, slots=True)
class RunSpeakerDiarizationUseCase:
    """Orquesta Step 08 Speaker Diarization. Inmutable. DI via constructor.

    Input: (PreprocessedAudio, VadResult) — ambos frozen, NO mutados.
    Output: DiarizationResult frozen con chain of custody T03+T04.
    """

    diarizer: SpeakerDiarizerPort

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        *,
        thresholds: DiarizationThresholds | None = None,
        job_id: str | None = None,
    ) -> DiarizationResult:
        # 1) None inputs
        if preprocessed_audio is None:
            raise ValidationFailedError("RunSpeakerDiarizationUseCase: preprocessed_audio es None")
        if vad_result is None:
            raise ValidationFailedError("RunSpeakerDiarizationUseCase: vad_result es None")

        bind_context(job_id=job_id, phase="DIARIZATION", model=MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL)
        t0 = time.perf_counter()

        # 2) Thresholds defaults
        effective_thresholds: DiarizationThresholds
        if thresholds is None:
            effective_thresholds = DiarizationThresholds()
        else:
            if not isinstance(thresholds, DiarizationThresholds):
                raise ValidationFailedError(
                    "RunSpeakerDiarizationUseCase: thresholds no es DiarizationThresholds"
                )
            effective_thresholds = thresholds

        # 3) Chain of Custody T03 ↔ T04
        prep_sha = str(preprocessed_audio.sha256).lower()
        vad_sha = str(vad_result.source_preprocessed_sha256).lower()
        if prep_sha != vad_sha:
            raise ValidationFailedError(
                "Chain of Custody rota T03↔T04: "
                f"PreprocessedAudio.sha256={prep_sha[:12]}... != "
                f"VadResult.source_preprocessed_sha256={vad_sha[:12]}..."
            )

        log.info(
            "run_diarization_started",
            extra={
                "vad_intervals": int(vad_result.num_intervals),
                "total_duration_ms": int(round(float(preprocessed_audio.duration_sec) * 1000.0)),
            },
        )

        # 4) Silent fallback (sin cargar modelo ni invocar al adapter)
        if int(vad_result.num_intervals) == 0:
            if not effective_thresholds.allow_silent_fallback:
                raise ValidationFailedError(
                    "RunSpeakerDiarizationUseCase: VAD sin voz y allow_silent_fallback=False"
                )
            return self._build_silent_fallback(
                preprocessed_audio=preprocessed_audio,
                vad_result=vad_result,
                job_id=job_id,
            )

        # 5) Delegar al Port (Dependency Inversion) — solo cuando hay voz.
        try:
            result: DiarizationResult = self.diarizer.diarize(
                preprocessed_audio=preprocessed_audio,
                vad_result=vad_result,
                thresholds=effective_thresholds,
                job_id=job_id,
                logger=log,
            )
        except (ModelLoadError, DiarizationError, ValidationFailedError):
            raise
        except Exception as exc:  # noqa: BLE001
            raise DiarizationError(
                f"run_diarization unexpected error: {exc!r}"
            ) from exc

        # 6) Post-condiciones
        if not isinstance(result, DiarizationResult):
            raise ValidationFailedError(
                "RunSpeakerDiarizationUseCase: adapter devolvió tipo no DiarizationResult"
            )
        if str(result.source_preprocessed_sha256).lower() != prep_sha:
            raise ValidationFailedError(
                "RunSpeakerDiarizationUseCase: DiarizationResult.source_preprocessed_sha256 "
                "no coincide con T03."
            )
        if str(result.vad_reference_sha256).lower() != vad_sha:
            raise ValidationFailedError(
                "RunSpeakerDiarizationUseCase: DiarizationResult.vad_reference_sha256 "
                "no coincide con T04."
            )
        if effective_thresholds.strict_validity and int(result.num_turns) > int(
            effective_thresholds.max_turns_safety
        ):
            raise ValidationFailedError(
                f"RunSpeakerDiarizationUseCase: num_turns={result.num_turns} > "
                f"max_turns_safety={effective_thresholds.max_turns_safety}"
            )

        t1 = time.perf_counter()
        log.info(
            "run_diarization_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "num_speakers": int(result.num_speakers),
                "num_turns": int(result.num_turns),
                "strategy": str(result.strategy),
            },
        )
        return result

    # ------------------------------------------------------------------
    # HELPERS INTERNOS
    # ------------------------------------------------------------------

    @staticmethod
    def _build_silent_fallback(
        *,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        job_id: str | None,
    ) -> DiarizationResult:
        """VAD sin voz → DiarizationResult vacío. NO carga modelo."""
        total_ms = int(round(float(preprocessed_audio.duration_sec) * 1000.0))
        return DiarizationResult(
            source_preprocessed_sha256=str(preprocessed_audio.sha256),
            vad_reference_sha256=str(vad_result.source_preprocessed_sha256),
            total_duration_ms=total_ms,
            speaker_turns=(),
            num_speakers=0,
            num_turns=0,
            strategy=DIARIZATION_STRATEGY_SILENT_FALLBACK,
            model_label=MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
            job_id=job_id,
            analysis_metadata={"fallback_reason": "vad_no_intervals"},
        )


__all__ = ["RunSpeakerDiarizationUseCase"]
