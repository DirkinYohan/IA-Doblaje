"""UseCase RunSpeakerAlignmentUseCase — Capa APPLICATION (T09 Step 09 Alignment).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa torch/pyannote/numpy/speechbrain (clean architecture).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import AlignmentError, ValidationFailedError
from app.core.logging import bind_context, get_logger
from app.domain.entities.alignment import (
    ALIGNMENT_STRATEGY_SILENT_FALLBACK,
    ALIGNMENT_STRATEGY_WOMV,
    MODEL_LABEL_ALIGNMENT_V1_LITERAL,
    AlignmentResult,
)
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.value_objects.alignment import (
    AlignmentThresholds,
    DialogueSegment,
)


log = get_logger("app.application.use_cases.align_speakers")


@dataclass(frozen=True, slots=True)
class RunSpeakerAlignmentUseCase:
    """Orquesta Step 09 ASR ↔ Speaker Alignment. Inmutable.

    Input: (TimestampGenerationResult, ASRResult, DiarizationResult) — frozen, NO mutados.
    Output: AlignmentResult frozen con chain of custody 6-SHA.
    """

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        timestamps: TimestampGenerationResult,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        *,
        thresholds: AlignmentThresholds | None = None,
        job_id: str | None = None,
    ) -> AlignmentResult:
        # 1) None inputs
        if timestamps is None:
            raise ValidationFailedError("RunSpeakerAlignmentUseCase: timestamps es None")
        if asr_result is None:
            raise ValidationFailedError("RunSpeakerAlignmentUseCase: asr_result es None")
        if diarization is None:
            raise ValidationFailedError("RunSpeakerAlignmentUseCase: diarization es None")

        bind_context(job_id=job_id, phase="ALIGNMENT", model=MODEL_LABEL_ALIGNMENT_V1_LITERAL)
        t0 = time.perf_counter()

        # 2) Thresholds defaults
        effective_thresholds: AlignmentThresholds
        if thresholds is None:
            effective_thresholds = AlignmentThresholds()
        else:
            if not isinstance(thresholds, AlignmentThresholds):
                raise ValidationFailedError(
                    "RunSpeakerAlignmentUseCase: thresholds no es AlignmentThresholds"
                )
            effective_thresholds = thresholds

        # 3) Chain of Custody T03→T07→T08
        self._validate_chain(timestamps, diarization)

        # 4) Safety: max segments
        if int(timestamps.num_segments_ts) > int(effective_thresholds.max_segments_safety):
            raise ValidationFailedError(
                f"T09 safety: num_segments={timestamps.num_segments_ts} > "
                f"max_segments_safety={effective_thresholds.max_segments_safety}"
            )

        log.info(
            "align_speakers_started",
            extra={
                "num_ts_segments": int(timestamps.num_segments_ts),
                "num_speaker_turns": int(diarization.num_turns),
            },
        )

        # 5) Diarización vacía → silent fallback
        if int(diarization.num_speakers) == 0 or len(diarization.speaker_turns) == 0:
            if not effective_thresholds.allow_silent_fallback:
                raise ValidationFailedError(
                    "RunSpeakerAlignmentUseCase: diarización vacía y allow_silent_fallback=False"
                )
            return self._build_silent_fallback(
                timestamps=timestamps,
                diarization=diarization,
                job_id=job_id,
            )

        # 6) Alineación segmento a segmento
        dialogue_segments: list[DialogueSegment] = []
        num_unassigned = 0
        asr_segments = list(asr_result.segments)
        asr_by_index = {int(s.segment_index): s for s in asr_segments}

        for ts_seg in timestamps.segments:
            idx = int(ts_seg.segment_index)
            # Texto desde ASR T06 (no desde TimestampedSegment)
            asr_seg = asr_by_index.get(idx)
            if asr_seg is None:
                raise ValidationFailedError(
                    f"T09: ASRResult no contiene segment_index={idx} referenciado por T07."
                )
            text = str(asr_seg.text)

            speaker = self._majority_vote_speaker(
                ts_start=int(ts_seg.start_ms),
                ts_end=int(ts_seg.end_ms),
                turns=diarization.speaker_turns,
                min_overlap_ms=int(effective_thresholds.min_overlap_ms),
            )

            if speaker is None:
                # Caso B: segmento sin overlap → no alineado
                num_unassigned += 1
                continue

            label, alignment_conf = speaker
            # segment_index debe ser CONTIGUO 0-based dentro del resultado
            # alineado (contrato AlignmentResult), no el índice original T07.
            dialogue_segments.append(
                DialogueSegment(
                    segment_index=len(dialogue_segments),
                    start_ms=int(ts_seg.start_ms),
                    end_ms=int(ts_seg.end_ms),
                    duration_ms=int(ts_seg.duration_ms),
                    text=text,
                    speaker_label=label,
                    alignment_confidence=alignment_conf,
                    ts_confidence=float(ts_seg.ts_confidence),
                )
            )

        # 7) Post-condiciones
        total_segments = int(timestamps.num_segments_ts)
        if len(dialogue_segments) + num_unassigned != total_segments:
            raise ValidationFailedError(
                "RunSpeakerAlignmentUseCase: alineados + no_alineados != num_segments T07."
            )

        result = AlignmentResult(
            strategy=ALIGNMENT_STRATEGY_WOMV,
            source_preprocessed_sha256=str(timestamps.source_preprocessed_sha256),
            vad_reference_sha256=str(timestamps.vad_reference_sha256),
            lid_reference_sha256=str(timestamps.lid_reference_sha256),
            asr_reference_sha256=str(timestamps.asr_reference_sha256),
            ts_reference_sha256=str(timestamps.source_preprocessed_sha256),
            diarization_reference_sha256=str(diarization.source_preprocessed_sha256),
            total_duration_ms=int(timestamps.total_duration_ms),
            transcript_language_code=str(timestamps.transcript_language_code),
            job_id=job_id,
            dialogue_segments=tuple(dialogue_segments),
            num_segments=total_segments,
            num_segments_aligned=len(dialogue_segments),
            num_unassigned=num_unassigned,
        )

        # 8) strict validity
        if effective_thresholds.strict_validity and not result.strict_ok:
            raise ValidationFailedError(
                f"T09 strict_validity=True con strict_ok=False; "
                f"errors={list(result.validation_errors)}"
            )

        t1 = time.perf_counter()
        log.info(
            "align_speakers_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "num_segments": int(result.num_segments),
                "num_aligned": int(result.num_segments_aligned),
                "num_unassigned": int(result.num_unassigned),
                "strategy": str(result.strategy),
            },
        )
        return result

    # ------------------------------------------------------------------
    # HELPERS INTERNOS
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_chain(
        timestamps: TimestampGenerationResult,
        diarization: DiarizationResult,
    ) -> None:
        ts_source = str(timestamps.source_preprocessed_sha256).lower()
        diar_source = str(diarization.source_preprocessed_sha256).lower()
        if ts_source != diar_source:
            raise ValidationFailedError(
                "Chain of Custody rota T07↔T08: TimestampGenerationResult.sha != "
                "DiarizationResult.source_preprocessed_sha256."
            )
        # T07 internamente ya valida sus 4 primeras referencias; aquí confirmamos
        # coherencia de las que T09 re-expone.
        for name in ("vad_reference_sha256", "lid_reference_sha256", "asr_reference_sha256"):
            if str(getattr(timestamps, name)).lower() != ts_source:
                raise ValidationFailedError(
                    f"Chain of Custody rota T07: TimestampGenerationResult.{name} != source."
                )

    @staticmethod
    def _majority_vote_speaker(
        *,
        ts_start: int,
        ts_end: int,
        turns: tuple,
        min_overlap_ms: int,
    ) -> tuple[str, float] | None:
        """Weighted Overlap Majority Vote por speaker.

        Retorna (speaker_label, alignment_confidence) del ganador, o None si no hay
        candidato con overlap > 0.
        """
        # Agrupar score ponderado por speaker (suma de weighted_score de sus turns)
        weighted_by_speaker: dict[str, float] = {}
        best_confidence_by_speaker: dict[str, float] = {}
        earliest_start_by_speaker: dict[str, int] = {}

        for turn in turns:
            overlap = max(
                0,
                min(ts_end, int(turn.end_ms)) - max(ts_start, int(turn.start_ms)),
            )
            if overlap < min_overlap_ms:
                continue
            label = str(turn.speaker_label)
            conf = float(turn.confidence)
            weighted_score = float(overlap) * conf

            weighted_by_speaker[label] = weighted_by_speaker.get(label, 0.0) + weighted_score
            best_confidence_by_speaker[label] = max(
                best_confidence_by_speaker.get(label, 0.0), conf
            )
            earliest_start_by_speaker[label] = min(
                earliest_start_by_speaker.get(label, 10**18), int(turn.start_ms)
            )

        if not weighted_by_speaker:
            return None

        # Determinista: mayor weighted_score, luego mayor confidence, luego menor start_ms
        winner = max(
            weighted_by_speaker,
            key=lambda label: (
                weighted_by_speaker[label],
                best_confidence_by_speaker[label],
                -earliest_start_by_speaker[label],
            ),
        )

        # alignment_confidence = proporción del segmento cubierta por el speaker ganador
        total_duration = ts_end - ts_start
        if total_duration <= 0:
            return None
        coverage = min(1.0, max(0.0, weighted_by_speaker[winner] / float(total_duration)))
        return (winner, coverage)

    @staticmethod
    def _build_silent_fallback(
        *,
        timestamps: TimestampGenerationResult,
        diarization: DiarizationResult,
        job_id: str | None,
    ) -> AlignmentResult:
        """Diarización vacía → alignment_silent_fallback. Sin speaker artificial."""
        return AlignmentResult(
            strategy=ALIGNMENT_STRATEGY_SILENT_FALLBACK,
            source_preprocessed_sha256=str(timestamps.source_preprocessed_sha256),
            vad_reference_sha256=str(timestamps.vad_reference_sha256),
            lid_reference_sha256=str(timestamps.lid_reference_sha256),
            asr_reference_sha256=str(timestamps.asr_reference_sha256),
            ts_reference_sha256=str(timestamps.source_preprocessed_sha256),
            diarization_reference_sha256=str(diarization.source_preprocessed_sha256),
            total_duration_ms=int(timestamps.total_duration_ms),
            transcript_language_code=str(timestamps.transcript_language_code),
            job_id=job_id,
            dialogue_segments=(),
            num_segments=int(timestamps.num_segments_ts),
            num_segments_aligned=0,
            num_unassigned=int(timestamps.num_segments_ts),
            validation_errors=(),
            strict_ok=True,
        )


__all__ = ["RunSpeakerAlignmentUseCase"]
