"""UseCase RunTranslationUseCase — Capa APPLICATION (T15 Step 15 Translation).

Dependency Injection via constructor. Solo depende de Domain Ports/Entities.
NO importa transformers/torch (clean architecture).

Flujo oficial T15:
    TRADUCCIÓN SEMÁNTICA CONTEXTUAL
        ↓
    TRADUCCIÓN NATURAL (vía TranslatorPort, 1:1 por segmento)
        ↓
    ADAPTACIÓN PARA DOBLAJE
        ↓
    VALIDACIÓN
        ↓
    translation.json (vía JsonOutputWriterPort)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    TranslationError,
    UnsupportedLanguageError,
    ValidationFailedError,
)
from app.core.logging import bind_context, get_logger
from app.domain.entities.translation import (
    MODEL_LABEL_TRANSLATION_V1_LITERAL,
    TranslationResult,
)
from app.domain.interfaces.output_ports import JsonOutputWriterPort
from app.domain.interfaces.translation_ports import TranslatorPort
from app.domain.value_objects.translation import (
    TRANSLATION_LANGUAGE_CODES,
    TranslationLanguageCode,
    TranslationSegment,
    TranslationSegmentValidation,
    TranslationThresholds,
    TranslationValidation,
)

log = get_logger("app.application.use_cases.run_translation")

TRANSLATION_OUTPUT_FILENAME = "translation.json"


@dataclass(frozen=True, slots=True)
class RunTranslationUseCase:
    """Orquesta Step 15. Inmutable. DI via constructor.

    Entrada: ruta a segments.json + transcript.json (output del job persistido por T13).
    Salida: TranslationResult + escribe translation.json en data/output/<job_id>/.
    """

    translator: TranslatorPort
    writer: JsonOutputWriterPort
    adaptation_rule: Any = None  # TranslationAdaptationRule (infra puro, inyectado)
    thresholds: TranslationThresholds | None = None

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        segments_path: str,
        transcript_path: str,
        output_directory: str,
        target_language: str,
        source_language: str | None = None,
        schema_version: str = "0.1.0",
        job_id: str | None = None,
        indent: int = 2,
        ensure_ascii: bool = False,
        force_save: bool = False,
    ) -> TranslationResult:
        bind_context(job_id=job_id, phase="TRANSLATION", model=MODEL_LABEL_TRANSLATION_V1_LITERAL)
        t0 = time.perf_counter()

        # 1) Validar target_language (contrato 8 idiomas)
        if str(target_language).strip().lower() not in TRANSLATION_LANGUAGE_CODES:
            raise UnsupportedLanguageError(
                f"Idioma destino no soportado T15: {target_language!r}. "
                f"Oficiales: {sorted(TRANSLATION_LANGUAGE_CODES)}."
            )
        target = str(target_language).strip().lower()

        # 2) Cargar segments.json (entrada principal)
        segs_path = Path(segments_path)
        if not segs_path.is_file():
            raise ValidationFailedError(f"T15: segments.json no existe: {segs_path!s}")
        try:
            segs_data = json.loads(segs_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise ValidationFailedError(f"T15: segments.json inválido: {exc!r}") from exc
        segments = list(segs_data.get("segments", []))

        # 3) source_language: preferir el provisto; si no, leer transcript.json
        source: str | None = None
        if source_language is not None and str(source_language).strip():
            source = str(source_language).strip().lower()
        else:
            tr_path = Path(transcript_path)
            if tr_path.is_file():
                try:
                    tr_data = json.loads(tr_path.read_text(encoding="utf-8"))
                    src = tr_data.get("transcript_language_code")
                    if src and str(src).strip():
                        source = str(src).strip().lower()
                except Exception:  # noqa: BLE001
                    source = None

        # 4) source inválido → error contractual
        if source is None or source == "und":
            raise TranslationError(
                "T15: source_language es 'und' (indeterminado). "
                "No se puede traducir sin idioma origen. Error contractual determinista."
            )
        if source not in TRANSLATION_LANGUAGE_CODES:
            raise UnsupportedLanguageError(
                f"T15: source_language {source!r} no está entre los 8 idiomas oficiales."
            )

        # 5) source == target → passthrough (no ejecutar M2M100)
        translation_performed: bool = source != target

        thr = self.thresholds if self.thresholds is not None else TranslationThresholds()

        # 6) Traducir + adaptar cada segmento (1:1, orden conservado)
        out_segments: list[TranslationSegment] = []
        global_warnings: list[str] = []
        global_errors: list[str] = []
        for i, seg in enumerate(segments):
            idx = int(seg.get("segment_index", i))
            source_text = str(seg.get("text", "") or "")
            speaker = str(seg.get("speaker_label", "SPEAKER_00"))
            start_ms = int(seg.get("start_ms", 0))
            end_ms = int(seg.get("end_ms", 0))
            duration_ms = int(seg.get("duration_ms", end_ms - start_ms))

            seg_warnings: list[str] = []
            seg_errors: list[str] = []

            # Traducción (contexto NO se inyecta al modelo; se pasa a adaptación/validación)
            if translation_performed:
                try:
                    translated = self.translator.translate(
                        source_text=source_text,
                        source_language=TranslationLanguageCode(source),
                        target_language=TranslationLanguageCode(target),
                        context_prev=(
                            str(segments[i - 1].get("text", "")) if i > 0 else None
                        ),
                        context_next=(
                            str(segments[i + 1].get("text", ""))
                            if i + 1 < len(segments)
                            else None
                        ),
                        speaker_label=speaker,
                        duration_ms=duration_ms,
                    )
                except (TranslationError, UnsupportedLanguageError):
                    raise
                except Exception as exc:  # noqa: BLE001
                    raise TranslationError(
                        f"T15 traducción segmento {idx} falló: {exc!r}"
                    ) from exc
            else:
                translated = source_text

            # Adaptación para doblaje
            if self.adaptation_rule is not None:
                adapt = self.adaptation_rule.adapt(
                    translated_text=translated,
                    source_text=source_text,
                    duration_ms=duration_ms,
                    target_language=target,
                    thresholds=thr,
                )
                adapted = adapt.adapted_text
                seg_warnings.extend(adapt.warnings)
            else:
                adapted = translated

            # Validación por segmento
            if source_text.strip() and not translated.strip():
                seg_errors.append("translated_text vacío con source_text no vacío")
            if source_text.strip() and not adapted.strip():
                seg_errors.append("adapted_text vacío con source_text no vacío")
            if end_ms <= start_ms:
                seg_errors.append(f"start_ms={start_ms} >= end_ms={end_ms} (inválido)")

            # Si hay errores de traducción vacía, usar el texto fuente como
            # fallback seguro (mantiene 1:1 y trazabilidad) pero con error.
            effective_translated = translated if translated.strip() else source_text
            effective_adapted = adapted if adapted.strip() else source_text

            out_segments.append(
                TranslationSegment(
                    segment_index=idx,
                    speaker_label=speaker,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    source_text=source_text,
                    translated_text=effective_translated,
                    adapted_text=effective_adapted,
                    validation=TranslationSegmentValidation(
                        ok=not seg_errors,
                        errors=tuple(seg_errors),
                        warnings=tuple(seg_warnings),
                    ),
                )
            )
            global_errors.extend(seg_errors)
            global_warnings.extend(seg_warnings)

        # 7) Validación global 1:1
        if len(out_segments) != len(segments):
            global_errors.append(
                f"1:1 violado: input={len(segments)} output={len(out_segments)}"
            )

        # 8) Construir TranslationResult (valida 1:1 / continuidad)
        validation = TranslationValidation(
            ok=not global_errors,
            errors=tuple(global_errors),
            warnings=tuple(global_warnings),
            num_segments_input=len(segments),
            num_segments_output=len(out_segments),
            translation_performed=translation_performed,
            deterministic=True,
        )
        try:
            result = TranslationResult(
                job_id=job_id,
                source_language=source,
                target_language=target,
                segments=tuple(out_segments),
                validation=validation,
            )
        except Exception as exc:  # noqa: BLE001
            raise ValidationFailedError(f"T15: TranslationResult inválido: {exc!r}") from exc

        # 9) Escribir translation.json (persistencia vía port; T14 NO lo borra)
        doc = self._build_document(result, schema_version=schema_version)
        self.writer.write_json(
            output_directory=output_directory,
            filename=TRANSLATION_OUTPUT_FILENAME,
            document=doc,
            indent=int(indent),
            ensure_ascii=bool(ensure_ascii),
        )

        log.info(
            "run_translation_finished_ok",
            extra={
                "elapsed_sec": round(time.perf_counter() - t0, 4),
                "source": source,
                "target": target,
                "num_segments": len(out_segments),
                "performed": translation_performed,
                "errors": len(global_errors),
                "warnings": len(global_warnings),
            },
        )
        return result

    # ------------------------------------------------------------------
    # Documento translation.json
    # ------------------------------------------------------------------
    @staticmethod
    def _build_document(result: TranslationResult, *, schema_version: str) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "job_id": result.job_id,
            "source_language": str(result.source_language),
            "target_language": str(result.target_language),
            "segments": [
                {
                    "segment_index": int(s.segment_index),
                    "speaker_label": str(s.speaker_label),
                    "start_ms": int(s.start_ms),
                    "end_ms": int(s.end_ms),
                    "duration_ms": int(s.duration_ms),
                    "source_text": s.source_text,
                    "translated_text": s.translated_text,
                    "adapted_text": s.adapted_text,
                    "validation": {
                        "ok": bool(s.validation.ok),
                        "errors": list(s.validation.errors),
                        "warnings": list(s.validation.warnings),
                    },
                }
                for s in result.segments
            ],
            "validation": {
                "ok": bool(result.validation.ok),
                "errors": list(result.validation.errors),
                "warnings": list(result.validation.warnings),
                "num_segments_input": int(result.validation.num_segments_input),
                "num_segments_output": int(result.validation.num_segments_output),
                "translation_performed": bool(result.validation.translation_performed),
                "deterministic": bool(result.validation.deterministic),
            },
        }


__all__ = ["RunTranslationUseCase", "TRANSLATION_OUTPUT_FILENAME"]
