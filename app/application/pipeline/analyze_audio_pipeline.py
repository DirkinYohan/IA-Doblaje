"""Orquestador AnalyzeAudioPipeline — Integración end-to-end T01→T14.

CAPA DE INTEGRACIÓN. Coordina los 14 pasos ya implementados (T01–T14) sin
contener lógica de IA propia. Cada Step sigue siendo responsable de su lógica.

NO modifica resultados upstream. T14 se ejecuta en `finally`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.application.use_cases.aggregate_metrics import RunMetricsAggregationUseCase
from app.application.use_cases.align_speakers import RunSpeakerAlignmentUseCase
from app.application.use_cases.analyze_media_prep import AnalyzeMediaPrepUseCase
from app.application.use_cases.analyze_quality import RunQualityAnalysisUseCase
from app.application.use_cases.cleanup_temp import RunTempCleanupUseCase
from app.application.use_cases.generate_timestamps import GenerateTimestampsUseCase
from app.application.use_cases.run_asr import RunASRUseCase
from app.application.use_cases.run_diarization import RunSpeakerDiarizationUseCase
from app.application.use_cases.run_language_detection import RunLanguageDetectionUseCase
from app.application.use_cases.run_vad import RunVoiceActivityDetectionUseCase
from app.application.use_cases.validate_results import RunValidationUseCase
from app.application.use_cases.write_structured_json import RunStructuredJsonOutputUseCase
from app.core.config import AppSettings, get_settings
from app.core.constants import JSON_SCHEMA_VERSION
from app.core.exceptions import EngineBaseError
from app.core.exceptions import PipelineCancelled
from app.core.logging import bind_context, get_logger
from app.core.paths import PathManager, generate_job_id
from app.domain.entities.alignment import AlignmentResult
from app.domain.entities.asr import ASRResult
from app.domain.entities.diarization import DiarizationResult
from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import MediaPrepResult
from app.domain.entities.metrics import MetricsResult
from app.domain.entities.output import StructuredJsonOutputResult
from app.domain.entities.quality import QualityResult
from app.domain.entities.timestamps import TimestampGenerationResult
from app.domain.entities.vad import VadResult
from app.domain.entities.validation import ValidationResult
from app.domain.value_objects.metrics import RuntimeMetricsSnapshot


log = get_logger("app.application.pipeline.analyze_audio_pipeline")


@dataclass
class AnalyzeAudioPipeline:
    """Orquestador T01→T14. Solo coordina (wiring + secuencia)."""

    media_prep: AnalyzeMediaPrepUseCase
    vad: RunVoiceActivityDetectionUseCase
    language: RunLanguageDetectionUseCase
    asr: RunASRUseCase
    timestamps: GenerateTimestampsUseCase
    diarization: RunSpeakerDiarizationUseCase
    alignment: RunSpeakerAlignmentUseCase
    quality: RunQualityAnalysisUseCase
    validation: RunValidationUseCase
    metrics: RunMetricsAggregationUseCase
    output: RunStructuredJsonOutputUseCase
    cleanup: RunTempCleanupUseCase
    settings: AppSettings
    path_manager: PathManager
    step_times: dict[str, float] = field(default_factory=dict)
    translation: Any | None = None  # RunTranslationUseCase (T15) — opcional
    cancel_event: Any | None = None
    progress_callback: Any | None = None
    asr_device: str = "cpu"

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str | Path,
        *,
        job_id: str | None = None,
        force_save: bool = False,
    ) -> dict[str, Any]:
        """Ejecuta T01→T14. Retorna un resumen (no muta resultados upstream)."""
        jid: str = job_id or generate_job_id()
        bind_context(job_id=jid, phase="PIPELINE")

        # Configuración real
        general = self.settings.general
        safety = self.settings.safety
        output_cfg = self.settings.output
        metrics_cfg = self.settings.metrics

        cleanup_enabled = bool(safety.enable_auto_temp_cleanup)
        keep_temp_files = bool(general.dev_keep_temp_files)

        results: dict[str, Any] = {}
        temp_root = str(self.path_manager.data_temp_dir)

        # T14 se ejecuta SIEMPRE (finally), respetando flags
        try:
            # T01–T03
            results["media_prep"] = self._run_step(
                "T01_T02_T03",
                lambda: self.media_prep.execute(
                    input_path,
                    job_id=jid,
                    settings=self.settings,
                    progress_cb=lambda step: self._notify(step, 0.0, "start"),
                ),
            )

            media_prep: MediaPrepResult = results["media_prep"]
            prep = media_prep.preprocessed_audio

            # T04 VAD
            results["vad"] = self._run_step(
                "T04_VAD",
                lambda: self.vad.execute(media_prep, job_id=jid),
            )

            # T05 LID
            results["language"] = self._run_step(
                "T05_LID",
                lambda: self.language.execute(media_prep, results["vad"], job_id=jid),
            )
            # Idioma realmente detectado (para persistirlo en el proyecto).
            results["detected_language"] = _detected_language_code(results["language"])

            # T06 ASR
            results["asr"] = self._run_step(
                "T06_ASR",
                lambda: self.asr.run(
                    prep, results["vad"], results["language"], job_id=jid
                ),
            )

            # T07 Timestamps
            results["timestamps"] = self._run_step(
                "T07_TIMESTAMPS",
                lambda: self.timestamps.run(
                    prep, results["vad"], results["language"], results["asr"], job_id=jid
                ),
            )

            # T08 Diarization
            results["diarization"] = self._run_step(
                "T08_DIARIZATION",
                lambda: self.diarization.run(prep, results["vad"], job_id=jid),
            )

            # T09 Alignment
            results["alignment"] = self._run_step(
                "T09_ALIGNMENT",
                lambda: self.alignment.run(
                    results["timestamps"], results["asr"], results["diarization"], job_id=jid
                ),
            )

            # T10 Quality
            results["quality"] = self._run_step(
                "T10_QUALITY",
                lambda: self.quality.run(
                    results["alignment"],
                    results["vad"],
                    results["asr"],
                    results["diarization"],
                    job_id=jid,
                ),
            )

            # T11 Validation
            results["validation"] = self._run_step(
                "T11_VALIDATION",
                lambda: self.validation.run(
                    results["quality"],
                    results["alignment"],
                    results["vad"],
                    results["asr"],
                    results["diarization"],
                    job_id=jid,
                    force_save=force_save,
                ),
            )

            # Runtime metrics snapshot (mediciones reales, stdlib)
            snapshot = self._build_runtime_snapshot(
                media_prep, results["asr"], results["vad"], results["diarization"],
            )

            # T12 Metrics
            results["metrics"] = self._run_step(
                "T12_METRICS",
                lambda: self.metrics.run(
                    snapshot,
                    results["language"],
                    results["asr"],
                    results["vad"],
                    results["diarization"],
                    results["alignment"],
                    results["quality"],
                    results["validation"],
                    job_id=jid,
                    metrics_enabled=bool(metrics_cfg.metrics_enabled),
                ),
            )

            # T13 Structured JSON Output
            output_dir = str(self.path_manager.data_output_dir / jid)
            results["output"] = self._run_step(
                "T13_OUTPUT",
                lambda: self.output.run(
                    results["asr"],
                    results["timestamps"],
                    results["diarization"],
                    results["alignment"],
                    results["quality"],
                    results["validation"],
                    results["metrics"],
                    output_directory=output_dir,
                    schema_version=JSON_SCHEMA_VERSION,
                    job_id=jid,
                    indent=int(output_cfg.output_json_indent),
                    ensure_ascii=bool(output_cfg.output_json_ensure_ascii),
                    force_save=force_save,
                ),
            )

            # T15 + subtítulos. Un fallo aquí es PARTIAL, nunca SUCCESS.
            results["translation_errors"] = []
            results["translations"] = []
            if self.translation is not None:
                glossary, glossary_error = _load_glossary(
                    str(getattr(self.settings.translation, "glossary_path", "") or "")
                )
                if glossary_error:
                    results["translation_errors"].append(glossary_error)
                cache = None
                if not glossary_error:
                    from app.infrastructure.translation.translation_cache import JsonTranslationCache

                    cache = JsonTranslationCache(
                        Path(self.settings.paths.data_output_dir).parent / "cache" / "translation.json"
                    )
                targets = [] if glossary_error else list(self.settings.translation.targets)
                for target in targets:
                    try:
                        translated = self._run_step(
                            f"T15_TRANSLATION_{target}",
                            lambda target=target: self.translation.run(
                                segments_path=str(
                                    self.path_manager.data_output_dir / jid / "segments.json"
                                ),
                                transcript_path=str(
                                    self.path_manager.data_output_dir / jid / "transcript.json"
                                ),
                                output_directory=output_dir,
                                target_language=target,
                                schema_version=JSON_SCHEMA_VERSION,
                                job_id=jid,
                                indent=int(output_cfg.output_json_indent),
                                ensure_ascii=bool(output_cfg.output_json_ensure_ascii),
                                force_save=force_save,
                                glossary=glossary,
                                cache=cache,
                            ),
                        )
                        results["translations"].append(translated)
                        if getattr(getattr(translated, "validation", None), "ok", True) is False:
                            results["translation_errors"].append(
                                f"{target}:validation_failed"
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.error("t15_translation_failed", extra={"error": str(exc), "target": target})
                        results["translation_errors"].append(f"{target}:{exc}")
                results["translation"] = (
                    results["translations"][-1] if results["translations"] else None
                )
            else:
                results["translation"] = None

            try:
                if self.progress_callback is not None:
                    self._notify("GENERATING_SUBTITLES", 0.0, "start")
                from app.application.use_cases.build_subtitles import (
                    build_cues_from_alignment,
                    write_subtitle_exports,
                )

                cues = build_cues_from_alignment(
                    results["alignment"],
                    language=str(getattr(results["alignment"], "transcript_language_code", "und")),
                )
                results["subtitles"] = cues
                if cues:
                    write_subtitle_exports(output_dir, cues, language=cues[0].language)
                for translated in results["translations"]:
                    tcues = build_cues_from_alignment(
                        results["alignment"],
                        language=str(translated.target_language),
                        text_by_index={
                            int(seg.segment_index): str(seg.adapted_text)
                            for seg in translated.segments
                        },
                    )
                    if tcues:
                        write_subtitle_exports(
                            output_dir, tcues, language=str(translated.target_language)
                        )
            except Exception as exc:  # noqa: BLE001
                log.error("subtitle_export_failed", extra={"error": str(exc)})
                results["subtitle_error"] = str(exc)

            if results["translation_errors"] or results.get("subtitle_error"):
                results["status"] = "PARTIAL"
            else:
                results["status"] = "SUCCESS"
            return results

        finally:
            # T14 Temp Cleanup (SIEMPRE)
            try:
                cleanup_result = self.cleanup.run(
                    temp_root,
                    jid,
                    cleanup_enabled=cleanup_enabled,
                    keep_temp_files=keep_temp_files,
                    force_save=force_save,
                )
                results["cleanup"] = cleanup_result
            except EngineBaseError as exc:
                log.error("t14_cleanup_security_error", extra={"error": str(exc)})
                results["cleanup"] = None

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _notify(self, step_name: str, elapsed: float, phase: str) -> None:
        """Notifica progreso. Admite callbacks de 2 (compat) o 3 argumentos."""
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(step_name, elapsed, phase)
        except TypeError:
            self.progress_callback(step_name, elapsed)

    def _run_step(self, step_name: str, fn: Any) -> Any:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise PipelineCancelled("Procesamiento cancelado por el usuario.")
        t0 = time.perf_counter()
        # El trabajo pesado ocurre dentro de fn(): anunciar el INICIO permite
        # que la UI muestre la etapa real mientras se ejecuta.
        self._notify(step_name, 0.0, "start")
        try:
            result = fn()
            self.step_times[step_name] = round(time.perf_counter() - t0, 6)
            log.info("pipeline_step_ok", extra={"step": step_name, "elapsed_sec": self.step_times[step_name]})
            self._notify(step_name, self.step_times[step_name], "done")
            return result
        except Exception as exc:  # noqa: BLE001
            self.step_times[step_name] = round(time.perf_counter() - t0, 6)
            log.error("pipeline_step_failed", extra={"step": step_name, "error": str(exc)})
            self._notify(step_name, self.step_times[step_name], "error")
            raise

    def _build_runtime_snapshot(
        self,
        media_prep: MediaPrepResult,
        asr_result: ASRResult,
        vad_result: VadResult,
        diarization: DiarizationResult,
    ) -> RuntimeMetricsSnapshot:
        """Construye RuntimeMetricsSnapshot con mediciones reales (stdlib)."""
        # wall time total = suma de step_times (aproximación real)
        wall = sum(self.step_times.values())
        # audio duration desde preprocessed
        audio_sec = 0.0
        prep = media_prep.preprocessed_audio
        if prep is not None:
            audio_sec = float(prep.duration_sec)

        # Mapear step_times internos a los 14 nombres oficiales de PipelineStep
        _STEP_NAME_MAP = {
            "T01_T02_T03": "MEDIA_VALIDATION",
            "T04_VAD": "VOICE_ACTIVITY_DETECTION",
            "T05_LID": "LANGUAGE_DETECTION",
            "T06_ASR": "SPEECH_TO_TEXT",
            "T07_TIMESTAMPS": "TIMESTAMPS_GENERATION",
            "T08_DIARIZATION": "SPEAKER_DIARIZATION",
            "T09_ALIGNMENT": "ASR_SPEAKER_ALIGNMENT",
            "T10_QUALITY": "QUALITY_ANALYSIS",
            "T11_VALIDATION": "VALIDATION",
            "T12_METRICS": "METRICS_AGGREGATION",
            "T13_OUTPUT": "STRUCTURED_JSON_OUTPUT",
            "T14_CLEANUP": "TEMP_CLEANUP",
        }
        step_timings = {
            _STEP_NAME_MAP.get(k, k): v for k, v in self.step_times.items()
        }

        from app.infrastructure.metrics.runtime_probe import probe_runtime

        probed = probe_runtime(self.asr_device)
        return RuntimeMetricsSnapshot(
            processing_wall_time_sec=max(0.0, wall),
            step_timings_sec=step_timings,
            peak_cpu_rss_mb=float(probed["rss_mb"]),
            peak_gpu_vram_used_mb=float(probed["vram_used_mb"]),
            peak_gpu_vram_total_mb=float(probed["vram_total_mb"]),
            cpu_utilization_avg_percent=float(probed["cpu_percent"]),
            gpu_utilization_avg_percent=float(probed["gpu_percent"]),
            profile_requested=str(getattr(self.settings.processing, "profile", "")),
            profile_applied=str(getattr(self.settings.processing, "profile", "")),
            profile_downgrade_applied=False,
            profile_downgrade_reason="",
            device_requested=str(getattr(self.settings.processing, "device", "")),
            device_used=str(probed["device"]),
        )


def build_pipeline(
    settings: AppSettings | None = None,
) -> AnalyzeAudioPipeline:
    """Wiring: construye adapters reales (Infrastructure) y use cases (Application)."""
    settings = settings or get_settings()

    from app.infrastructure.audio.ffmpeg_adapters import (
        FFmpegAudioExtractor,
        FFmpegAudioPreprocessor,
        FFprobeMediaValidator,
        SubprocessFFmpegBinaryResolver,
    )
    from app.infrastructure.audio.faster_whisper_asr_adapter import FasterWhisperSmallASRAdapter
    from app.infrastructure.audio.silero_vad_adapter import SileroVADAdapter
    from app.infrastructure.audio.whisper_lid_adapter import (
        WhisperEncoderLanguageDetectionAdapter,
    )
    from app.infrastructure.diarization.wespeaker_diarization_adapter import (
        WespeakerSpeakerDiarizationAdapter,
    )
    from app.infrastructure.storage.json_output_writer import JsonOutputWriter
    from app.infrastructure.storage.temp_cleanup_adapter import TempCleanupAdapter

    pm = PathManager.from_settings(settings)

    media_prep = AnalyzeMediaPrepUseCase(
        binary_resolver=SubprocessFFmpegBinaryResolver(),
        validator=FFprobeMediaValidator(),
        extractor=FFmpegAudioExtractor(),
        preprocessor=FFmpegAudioPreprocessor(),
    )
    vad = RunVoiceActivityDetectionUseCase(vad_detector=SileroVADAdapter(settings=settings))
    language = RunLanguageDetectionUseCase(
        detector=WhisperEncoderLanguageDetectionAdapter(settings=settings)
    )
    from app.core.profile_loader import load_asr_profile
    from app.core.profile_loader import resolve_compute_type
    from app.domain.value_objects.asr import AsrThresholds

    profile_name = str(getattr(settings.processing.profile, "value", settings.processing.profile))
    spec = load_asr_profile(profile_name, settings.paths.configs_dir)
    device_name = _resolve_asr_device(settings)
    compute_type, compute_note = resolve_compute_type(spec.compute_type, device_name)
    if compute_note:
        log.info("compute_type_adjusted", extra={"note": compute_note, "device": device_name})
    asr = RunASRUseCase(
        asr_port=FasterWhisperSmallASRAdapter(
            compute_type=compute_type,
            model_folder=spec.model_folder,
            device_resolver=lambda: device_name,
        ),
        default_thresholds=AsrThresholds(
            beam_size=int(spec.beam_size),
            word_timestamps=bool(spec.word_timestamps),
            chunk_by_vad=bool(spec.chunk_by_vad),
        ),
    )
    timestamps = GenerateTimestampsUseCase(
        timestamp_normalizer=_build_timestamp_normalizer()
    )
    diarization = RunSpeakerDiarizationUseCase(
        diarizer=WespeakerSpeakerDiarizationAdapter(settings_getter=lambda: settings)
    )
    alignment = RunSpeakerAlignmentUseCase()
    quality = RunQualityAnalysisUseCase(strict=True)
    validation = RunValidationUseCase(strict_quality=True)
    metrics = RunMetricsAggregationUseCase()
    output = RunStructuredJsonOutputUseCase(writer=JsonOutputWriter())
    cleanup = RunTempCleanupUseCase(cleanup=TempCleanupAdapter())

    # T15 Translation Engine (opcional, según configuración)
    translation = None
    if getattr(settings.translation, "enabled", True):
        from app.application.use_cases.run_translation import RunTranslationUseCase
        from app.infrastructure.translation.m2m100_translation_adapter import (
            M2M100TranslatorAdapter,
        )
        from app.infrastructure.translation.translation_adaptation import (
            TranslationAdaptationRule,
        )
        from app.domain.value_objects.translation import TranslationThresholds

        translation = RunTranslationUseCase(
            translator=M2M100TranslatorAdapter(
                settings_getter=lambda: settings,
                max_length=int(settings.translation.max_length),
                num_beams=int(settings.translation.num_beams),
                do_sample=bool(settings.translation.do_sample),
            ),
            writer=JsonOutputWriter(),
            adaptation_rule=TranslationAdaptationRule(),
            thresholds=TranslationThresholds(
                max_chars_per_second=float(
                    settings.translation.max_chars_per_second
                ),
                max_adaptation_ratio=float(
                    settings.translation.max_adaptation_ratio
                ),
            ),
        )

    return AnalyzeAudioPipeline(
        media_prep=media_prep,
        vad=vad,
        language=language,
        asr=asr,
        timestamps=timestamps,
        diarization=diarization,
        alignment=alignment,
        quality=quality,
        validation=validation,
        metrics=metrics,
        output=output,
        cleanup=cleanup,
        settings=settings,
        path_manager=pm,
        translation=translation,
        asr_device=device_name,
    )


def _resolve_asr_device(settings: AppSettings) -> str:
    requested = str(getattr(settings.processing.device, "value", settings.processing.device)).lower()
    if bool(getattr(settings.processing, "force_cpu", False)) or requested == "cpu":
        return "cpu"
    try:
        import torch

        if requested in {"auto", "cuda"} and torch.cuda.is_available():
            return "cuda"
        if requested in {"auto", "mps"} and getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available():
                return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def _load_glossary(path: str) -> tuple[dict[str, dict[str, str]] | None, str | None]:
    raw_path = path.strip()
    if not raw_path:
        return None, None
    file_path = Path(raw_path)
    if not file_path.is_file():
        return None, f"glossary:no existe {file_path}"
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, f"glossary:{exc}"
    if not isinstance(loaded, dict):
        return None, "glossary:se esperaba un objeto JSON"
    glossary: dict[str, dict[str, str]] = {}
    for language, mapping in loaded.items():
        if not isinstance(mapping, dict):
            return None, f"glossary:{language} no es un objeto"
        glossary[str(language)] = {str(term): str(value) for term, value in mapping.items()}
    return glossary, None


def _detected_language_code(lid: Any) -> str:
    """Extrae el código ISO del idioma detectado por T05. '' si no es fiable."""
    raw = getattr(lid, "language_code", None)
    if raw is None and isinstance(lid, dict):
        raw = lid.get("language_code")
    value = str(getattr(raw, "value", raw) or "").strip().lower()
    if not value or value == "und":
        return ""
    return value


def _build_timestamp_normalizer() -> Any:
    from app.infrastructure.audio.basic_timestamp_normalizer_adapter import (
        BasicTimestampNormalizerAdapter,
    )

    return BasicTimestampNormalizerAdapter()


__all__ = ["AnalyzeAudioPipeline", "build_pipeline"]
