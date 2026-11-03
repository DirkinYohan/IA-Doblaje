from __future__ import annotations

from typing import Any

from app.core.exceptions import ValidationFailedError
from app.domain.entities.asr import ASRResult, ASRSegment
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.timestamps import (
    TimestampGenerationResult,
    TimestampedSegment,
)
from app.domain.entities.vad import VadResult, VoiceInterval
from app.domain.interfaces.timestamp_ports import TimestampNormalizerPort
from app.domain.value_objects.timestamps import (
    AsrTimestampThresholds,
)


class BasicTimestampNormalizerAdapter(TimestampNormalizerPort):
    def normalize(
        self,
        preprocessed_audio: PreprocessedAudio,
        *,
        vad_result: VadResult,
        lid_result: LanguageDetectionResult,
        asr_result: ASRResult,
        thresholds: AsrTimestampThresholds | None = None,
        job_id: str | None = None,
    ) -> TimestampGenerationResult:
        eff: AsrTimestampThresholds = (
            thresholds if thresholds is not None else AsrTimestampThresholds()
        )
        total_duration_ms = int(preprocessed_audio.duration_sec * 1000)
        validation_errors: list[str] = []

        raw_segments: list[tuple[int, int, bool, str]] = []
        asr_segments_list: list[ASRSegment] = list(asr_result.segments)
        vad_intervals_list: list[VoiceInterval] = list(vad_result.voice_intervals)

        num_vad_interpolated = 0
        num_fw_none_startms_fixed = 0
        strategy_raw: bool = True

        for seg in asr_segments_list:
            start = seg.start_ms
            end = seg.end_ms
            was_vad = False

            if start is None or end is None:
                strategy_raw = False
                if not eff.allow_vad_interpolate_fallback:
                    raise ValidationFailedError(
                        f"T07 D#3 ASRSegment idx={seg.segment_index} start/end is None "
                        "pero allow_vad_interpolate_fallback=False"
                    )
                matched_start, matched_end = self._match_vad_interval(
                    seg_index=int(seg.segment_index),
                    num_asr_total=len(asr_segments_list),
                    vad_intervals=vad_intervals_list,
                    segment_text_len=max(1, len(str(seg.text).strip()) or 1),
                    total_duration_ms=total_duration_ms,
                )
                start = matched_start
                end = matched_end
                if start is None or end is None:
                    raise ValidationFailedError(
                        f"T07 D#3: sin VAD match válido para ASRSegment idx={seg.segment_index} "
                        f"(start/end=None y VAD sin match). NO se inventan timestamps arbitrarios."
                    )
                was_vad = True
                num_vad_interpolated += 1
                num_fw_none_startms_fixed += 1
            source_kind = "vad_interpolate" if was_vad else "fw_raw"
            raw_segments.append((int(start), int(end), was_vad, source_kind))

        indexed = list(enumerate(raw_segments))
        sorted_pairs = sorted(indexed, key=lambda p: (p[1][0], p[1][1], p[0]))
        ordered_raw: list[tuple[int, int, bool, str]] = [p[1] for p in sorted_pairs]

        num_out_of_audio_dropped = 0
        filtered_raw: list[tuple[int, int, bool, str]] = []
        for t in ordered_raw:
            if t[0] >= total_duration_ms:
                num_out_of_audio_dropped += 1
                continue
            filtered_raw.append(t)
        ordered_raw = filtered_raw

        num_overlaps_resolved = 0
        overlap_flags = [False] * len(ordered_raw)
        for i in range(len(ordered_raw) - 1):
            s_i, e_i, v_i, k_i = ordered_raw[i]
            s_next, _e_next, _v_next, _k_next = ordered_raw[i + 1]
            if e_i > s_next:
                new_end = s_next
                if new_end <= s_i:
                    validation_errors.append(
                        f"Overlap idx={i}/{i+1} resolución produjo end<=start"
                    )
                ordered_raw[i] = (s_i, new_end, v_i, k_i)
                overlap_flags[i] = True
                num_overlaps_resolved += 1

        num_end_clamped = 0
        end_clamp_flags = [False] * len(ordered_raw)
        for i in range(len(ordered_raw)):
            s_i, e_i, v_i, k_i = ordered_raw[i]
            if eff.allow_end_clamp_to_total and e_i > total_duration_ms:
                new_e = total_duration_ms
                if new_e <= s_i:
                    validation_errors.append(
                        f"Clamp idx={i} end>total_duration y end<=start post clamp"
                    )
                ordered_raw[i] = (s_i, new_e, v_i, k_i)
                end_clamp_flags[i] = True
                num_end_clamped += 1

        durations: list[int] = []
        for i in range(len(ordered_raw)):
            s_i, e_i, _v, _k = ordered_raw[i]
            if e_i <= s_i:
                msg = (
                    f"segment idx={i} start={s_i} end={e_i} inválido (end<=start) "
                    f"post-normalización. D#3: sin timestamp seguro, abortar."
                )
                validation_errors.append(msg)
            durations.append(e_i - s_i if e_i > s_i else 0)

        for i in range(len(ordered_raw)):
            if durations[i] < int(eff.min_segment_duration_ms):
                validation_errors.append(
                    f"segment idx={i} duration={durations[i]} < "
                    f"min_segment_duration_ms={eff.min_segment_duration_ms}"
                )

        final_segments: list[TimestampedSegment] = []
        for i in range(len(ordered_raw)):
            s_i, e_i, was_vad, kind = ordered_raw[i]
            if e_i <= s_i and eff.strict_validity:
                raise ValidationFailedError(
                    f"T07 strict: idx={i} end<=start post-normalización; "
                    f"errors={list(validation_errors)}"
                )
            ts_seg = TimestampedSegment(
                segment_index=i,
                start_ms=s_i,
                end_ms=e_i,
                duration_ms=max(1, e_i - s_i) if e_i > s_i else 1,
                source=kind,  # type: ignore[arg-type]
                overlap_normalized=bool(overlap_flags[i]),
                gap_filled=False,
                end_clamped_to_total_duration=bool(end_clamp_flags[i]),
                vad_interpolated=bool(was_vad),
                ts_confidence=1.0 if not was_vad else 0.7,
            )
            final_segments.append(ts_seg)

        if strategy_raw:
            strategy = "segment_ts_from_fw_raw"
        else:
            strategy = "segment_ts_from_vad_interpolate"

        strict_ok = True
        if validation_errors:
            strict_ok = False
            if eff.strict_validity:
                raise ValidationFailedError(
                    f"T07 strict_validity con validation_errors={list(validation_errors)}"
                )

        return TimestampGenerationResult.build_chain(
            preprocessed_audio=preprocessed_audio,
            vad_result=vad_result,
            language_detection_result=lid_result,
            asr_result=asr_result,
            strategy=strategy,  # type: ignore[arg-type]
            transcript_language_code=asr_result.transcript_language_code,
            segments=tuple(final_segments),
            words=(),
            job_id=job_id,
            num_overlaps_resolved=num_overlaps_resolved,
            num_gaps_filled=0,
            num_end_clamped=num_end_clamped,
            num_vad_interpolated=num_vad_interpolated,
            num_fw_none_startms_fixed=num_fw_none_startms_fixed,
            validation_errors=tuple(validation_errors),
            strict_ok=strict_ok,
            transcript_text="",
            confidence=float(asr_result.confidence) if asr_result.confidence is not None else 0.0,
        )

    @staticmethod
    def _match_vad_interval(
        *,
        seg_index: int,
        num_asr_total: int,
        vad_intervals: list[VoiceInterval],
        segment_text_len: int,
        total_duration_ms: int,
    ) -> tuple[int | None, int | None]:
        if not vad_intervals:
            return None, None
        M = len(vad_intervals)
        N = num_asr_total
        if M == N:
            vi = vad_intervals[seg_index]
            return int(vi.start_ms), int(vi.end_ms)
        if 0 <= seg_index < M:
            vi = vad_intervals[seg_index]
            return int(vi.start_ms), int(vi.end_ms)
        if M > 0:
            vi = vad_intervals[-1]
            return int(vi.start_ms), int(vi.end_ms)
        return None, None
