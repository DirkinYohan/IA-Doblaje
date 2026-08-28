"""Entidad LanguageDetectionResult Step 05 LID — Capa DOMAIN.

Frozen + extra=forbid. Chain of custody triple:
  PreprocessedAudio.sha256 == VadResult.source_preprocessed_sha256 == source_preprocessed_sha256

Sin dependencias infraestructura (no torch/transformers).
"""
from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.domain.value_objects.audio import (
    BitDepth,
    ChannelCount,
    SampleRate,
    Sha256Hash,
)
from app.domain.value_objects.lid import (
    LanguageAlternative,
    LanguageCode,
    LanguageConfidence,
    LanguageDetectionThresholds,
    LidDurationMs,
    validate_alternatives_sorted_desc,
)


MODEL_LABEL_WHISPER_LID_V1_LITERAL: Literal[
    "whisper-encoder-small-lid:v1"
] = "whisper-encoder-small-lid:v1"

LID_STRATEGY_VOICE_ONLY: Literal["only_voice_intervals"] = "only_voice_intervals"
LID_STRATEGY_FULL_AUDIO: Literal["full_audio"] = "full_audio"
LID_STRATEGY_SILENT_FALLBACK: Literal["silent_fallback_none"] = "silent_fallback_none"

LID_STRATEGIES_TYPE = (
    Literal["only_voice_intervals"]
    | Literal["full_audio"]
    | Literal["silent_fallback_none"]
)


class LanguageDetectionResult(BaseModel):
    """Resultado Step 05 Language Detection. Inmutable. Chain of Custody triple."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------
    # Chain of Custody
    # ------------------------------------------------------------------

    source_preprocessed_sha256: Sha256Hash = Field(
        ...,
        description="Enlace T03 → T04 → T05. Debe coincidir con PreprocessedAudio.sha256 y VadResult.source_preprocessed_sha256.",
    )
    vad_reference_sha256: Sha256Hash = Field(
        ...,
        description="Copia directa de VadResult.source_preprocessed_sha256 para auditoría doble binding.",
    )

    # ------------------------------------------------------------------
    # Predicción primaria
    # ------------------------------------------------------------------

    language_code: LanguageCode = Field(
        ..., description="ISO 639-1 lowercase primary. 'und' para silent fallback / baja confianza undetermined."
    )
    language_name: str = Field(
        ..., min_length=1, max_length=64, description="Nombre idioma EN INGLÉS."
    )
    confidence: LanguageConfidence = Field(
        ..., description="Confianza predicción primaria [0,1]. Silent fallback = 0.0."
    )

    # ------------------------------------------------------------------
    # Alternatives Top-3 (PIPELINE.md requerido)
    # ------------------------------------------------------------------

    alternatives: tuple[LanguageAlternative, ...] = Field(
        default_factory=tuple,
        description="Top-K alternativas, excluyendo primario. Orden DESC confidence. Máx 3. Silent fallback = ().",
    )

    # ------------------------------------------------------------------
    # Duraciones y estrategia
    # ------------------------------------------------------------------

    total_duration_ms: LidDurationMs = Field(
        0, description="Duración total audio PreprocessedAudio T03 (ms)."
    )
    analyzed_duration_ms: LidDurationMs = Field(
        0, description="Duración analizada realmente (solo voice intervals normalmente). <= total_duration_ms."
    )
    analyzed_strategy: LID_STRATEGIES_TYPE = Field(
        LID_STRATEGY_VOICE_ONLY,
        description="'only_voice_intervals' (normal D#7), 'full_audio' (fallback), 'silent_fallback_none' (sin voz).",
    )
    num_speech_intervals_considered: int = Field(
        0, ge=0, description="Número de VadResult.voice_intervals concatenados para el encoder."
    )

    # ------------------------------------------------------------------
    # Modelo + Metadata T03
    # ------------------------------------------------------------------

    model_label: Literal["whisper-encoder-small-lid:v1"] = Field(
        MODEL_LABEL_WHISPER_LID_V1_LITERAL,
        description="Etiqueta fija del modelo usado. T05 = Whisper Small + LID Head TorchScript.",
    )
    sample_rate: SampleRate = Field(16000, description="Sample rate PreprocessedAudio. Siempre 16k.")
    channels: ChannelCount = Field(1, description="Canales PreprocessedAudio. Siempre 1 mono.")
    bit_depth: BitDepth = Field(16, description="Bit depth PreprocessedAudio. Siempre 16-bit PCM.")

    # ------------------------------------------------------------------
    # Configuración aplicada + metadata auditoría
    # ------------------------------------------------------------------

    thresholds: LanguageDetectionThresholds = Field(
        default_factory=LanguageDetectionThresholds,
        description="Thresholds aplicados a esta predicción (standalone VO; NO en ProcessingConfig).",
    )
    analysis_metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Metadata agregada auditoría. Sin tensores / logits raw. Sin audio. Solo str/int/float/bool/nested dicts de esos tipos.",
    )

    # ==================================================================
    # VALIDADORES GLOBALES
    # ==================================================================

    @model_validator(mode="after")
    def _validate_all(self) -> "LanguageDetectionResult":
        # (A) sha 64hex lower consistency
        src = str(self.source_preprocessed_sha256).strip().lower()
        vad = str(self.vad_reference_sha256).strip().lower()
        if src != vad:
            raise ValueError(
                "Chain of Custody rota: source_preprocessed_sha256 != vad_reference_sha256. "
                f"{src} vs {vad}"
            )
        object.__setattr__(self, "source_preprocessed_sha256", src)
        object.__setattr__(self, "vad_reference_sha256", vad)

        # (B) analyzed <= total
        if int(self.analyzed_duration_ms) > int(self.total_duration_ms):
            raise ValueError(
                f"analyzed_duration_ms {self.analyzed_duration_ms} > total_duration_ms {self.total_duration_ms}"
            )

        # (C) alternatives sort DESC + ranks únicos + <top_k (de thresholds)
        validated_alts = validate_alternatives_sorted_desc(
            tuple(self.alternatives), int(self.thresholds.top_k)
        )
        object.__setattr__(self, "alternatives", validated_alts)

        # (D) strategy silent => code=und, conf=0, alternatives=empty, intervals_considered=0, analyzed=0
        if str(self.analyzed_strategy) == LID_STRATEGY_SILENT_FALLBACK:
            if str(self.language_code) != "und":
                raise ValueError(
                    f"silent_fallback_none debe tener language_code='und', recibido={self.language_code!r}"
                )
            if str(self.language_name).lower() != "undetermined":
                raise ValueError(
                    f"silent_fallback_none language_name debe ser 'Undetermined', recibido={self.language_name!r}"
                )
            if abs(float(self.confidence) - 0.0) > 1e-12:
                raise ValueError(
                    f"silent_fallback_none confidence debe ser 0.0, recibido={self.confidence}"
                )
            if len(self.alternatives) != 0:
                raise ValueError(
                    "silent_fallback_none alternatives debe ser tupla vacía."
                )
            if int(self.num_speech_intervals_considered) != 0:
                raise ValueError(
                    "silent_fallback_none num_speech_intervals_considered debe ser 0."
                )

        # (E) strategy voice-only => intervals >=1
        if (
            str(self.analyzed_strategy) == LID_STRATEGY_VOICE_ONLY
            and int(self.num_speech_intervals_considered) <= 0
            and int(self.analyzed_duration_ms) > 0
        ):
            raise ValueError(
                "only_voice_intervals requiere num_speech_intervals_considered >= 1 si analyzed>0."
            )

        # (F) analysis_metadata sin bytes/tensores prohibidos (solo chequeo superficial)
        for k, v in self.analysis_metadata.items():
            if isinstance(v, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"analysis_metadata[{k!r}] contiene bytes/prohibido. Sin raw audio/logits."
                )

        # (G) language_name strip
        object.__setattr__(self, "language_name", str(self.language_name).strip())

        return self


__all__ = [
    "LanguageDetectionResult",
    "MODEL_LABEL_WHISPER_LID_V1_LITERAL",
    "LID_STRATEGY_VOICE_ONLY",
    "LID_STRATEGY_FULL_AUDIO",
    "LID_STRATEGY_SILENT_FALLBACK",
    "LID_STRATEGIES_TYPE",
]
