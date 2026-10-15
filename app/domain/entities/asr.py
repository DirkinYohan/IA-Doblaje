from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.domain.entities.lid import LanguageDetectionResult
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.value_objects.asr import (
    ASRSegmentIndex,
    ASRStrategyName,
    ASRText,
    AsrThresholds,
    TranscriptConfidence,
)
from app.domain.value_objects.audio import (
    BitDepth,
    ChannelCount as _ChannelsVO,
    SampleRate as _SampleRateVO,
    Sha256Hash,
)
from app.domain.value_objects.lid import LidDurationMs


class ASRWord(BaseModel):
    """Palabra con tiempo. Opcional: solo si el decoder devolvió word timestamps."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    text: str = Field(..., max_length=500)
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("end_ms")
    @classmethod
    def _word_end_after_start(cls, v: int, info: ValidationInfo) -> int:
        start = info.data.get("start_ms")
        if isinstance(start, int) and v < start:
            raise ValueError(f"word end_ms={v} < start_ms={start}")
        return v


class ASRSegment(BaseModel):
    """Segmento ASR raw de decoder Step 06 (NO son timestamps oficiales T07)."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    segment_index: ASRSegmentIndex = Field(
        ..., description="Índice 0-based del segmento dentro de la transcripción."
    )
    text: ASRText = Field(
        ..., description="Texto decodificado del segmento (ASR raw)."
    )
    start_ms: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Metadata cruda start_ms devuelta por decoder (sólo referencia). "
            "TIMESTAMPS OFICIALES PERTENECEN EXCLUSIVAMENTE A T07 — Step 07 Timestamp Generation. "
            "No usar este campo para Steps 07+."
        ),
    )
    end_ms: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Metadata cruda end_ms devuelta por decoder (sólo referencia). "
            "TIMESTAMPS OFICIALES = T07 Timestamp Generation. No usar en T07+."
        ),
    )
    avg_logprob: float | None = Field(
        default=None,
        description=(
            "avg_logprob crudo decoder Faster-Whisper. Rango típico (-∞, 0]. "
            "Se transforma deterministicamente a TranscriptConfidence en UseCase."
        ),
    )
    no_speech_prob: float | None = Field(
        default=None,
        description="no_speech_prob raw Faster-Whisper rango [0,1] (probabilidad de no habla).",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confianza normalizada [0,1] del segmento. None si el decoder no la dio.",
    )
    words: tuple[ASRWord, ...] = Field(
        default_factory=tuple,
        description="Palabras con timestamp. Vacío si word timestamps no se pidieron.",
    )

    @field_validator("end_ms")
    @classmethod
    def _end_after_start(cls, v: int | None, info: ValidationInfo) -> int | None:
        if v is None:
            return v
        start = info.data.get("start_ms")
        if start is not None and v < start:
            raise ValueError(f"end_ms={v} < start_ms={start}")
        return v


class ASRResult(BaseModel):
    """Entity frozen Step 06 ASR. Chain of custody 4-sha triple binding."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    # --- CHAIN OF CUSTODY 4-SHA ------------------------------------------
    source_preprocessed_sha256: Sha256Hash = Field(
        ...,
        description="Sha256Hash PreprocessedAudio T03. == VadResult.source_preprocessed_sha256 == LanguageDetectionResult.source_preprocessed_sha256.",
    )
    vad_reference_sha256: Sha256Hash = Field(
        ..., description="== VadResult.source_preprocessed_sha256 (chain binding T04)."
    )
    lid_reference_sha256: Sha256Hash = Field(
        ..., description="== LanguageDetectionResult.source_preprocessed_sha256 (chain binding T05)."
    )

    # --- AUDIO METADATA PASSTHROUGH T03 -----------------------------------
    sample_rate: _SampleRateVO
    channels: _ChannelsVO
    bit_depth: BitDepth
    total_duration_ms: LidDurationMs

    # --- CORE ASR T06 ----------------------------------------------------
    transcript_text: ASRText
    transcript_language_code: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description=(
            "ISO 639-1 lowercase ó ISO 639-3 si multilingual Whisper lo requiere. "
            "Idioma REAL usado en el decoder (puede diferir de T05 si hubo auto-detect)."
        ),
        pattern=r"^[a-z]{2,3}$",
    )
    confidence: TranscriptConfidence
    segments: tuple[ASRSegment, ...] = Field(
        default_factory=tuple,
        description="Segmentos ASR raw (sin timestamps oficiales T07). Max len = 100.000.",
    )
    num_segments: ASRSegmentIndex

    # --- METADATA INFERENCIA ---------------------------------------------
    strategy: ASRStrategyName
    model_label: str = Field(
        ...,
        min_length=3,
        max_length=120,
        description="Identificador humano del adapter+modelo. Ej: 'faster-whisper:small:int8-fp16:ct2-local-v1'.",
    )
    thresholds_used: AsrThresholds

    # --- CUSTODY / JOB ---------------------------------------------------
    job_id: str | None = Field(default=None, min_length=8, max_length=128)
    analysis_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Sólo metadata auditoría. NO arrays/tensores/audio/logits completos.",
    )

    # =====================================================================
    # VALIDADORES
    # =====================================================================
    @field_validator("segments")
    @classmethod
    def _segments_ordered_and_limited(
        cls, v: tuple[ASRSegment, ...]
    ) -> tuple[ASRSegment, ...]:
        if len(v) > 100_000:
            raise ValueError(f"Segments exceden límite 100.000: {len(v)}")
        prev_end: int | None = None
        for idx, seg in enumerate(v):
            if seg.segment_index != idx:
                raise ValueError(
                    f"Segment index discontinuo. Esperado {idx}, encontrado {seg.segment_index}"
                )
            if (
                prev_end is not None
                and seg.start_ms is not None
                and seg.start_ms < prev_end
            ):
                # Solapamiento moderado permitido, pero >15% duración warning
                # (no impedir, Whisper decoder produce solapes naturales).
                pass
            if seg.end_ms is not None:
                prev_end = seg.end_ms
        return v

    @field_validator("analysis_metadata")
    @classmethod
    def _metadata_no_bytes(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("analysis_metadata debe ser dict")
        for k, val in v.items():
            if isinstance(val, (bytes, bytearray, memoryview)):
                raise ValueError(f"analysis_metadata[{k!r}] contiene bytes/tensores prohibidos")
            if isinstance(val, (list, tuple)) and len(val) > 10_000:
                raise ValueError(f"analysis_metadata[{k!r}] excede 10.000 elementos")
        return v

    @model_validator(mode="after")
    def _chain_of_custody_and_consistency(self) -> "ASRResult":
        # 1) Chain 4-sha
        if (
            self.source_preprocessed_sha256 != self.vad_reference_sha256
            or self.source_preprocessed_sha256 != self.lid_reference_sha256
        ):
            raise ValueError(
                "Chain of custody SHA256 roto: source_preprocessed_sha256 "
                "== vad_reference_sha256 == lid_reference_sha256."
            )
        # 2) num_segments consistency
        if self.num_segments != len(self.segments):
            raise ValueError(
                f"num_segments={self.num_segments} != len(segments)={len(self.segments)}"
            )
        # 3) silent fallback consistency
        if self.strategy == "asr_silent_fallback":
            if self.transcript_text != "":
                raise ValueError("asr_silent_fallback requiere transcript_text=''")
            if len(self.segments) != 0:
                raise ValueError("asr_silent_fallback requiere segments=()")
            if self.num_segments != 0:
                raise ValueError("asr_silent_fallback requiere num_segments=0")
            if self.confidence != 0.0:
                raise ValueError("asr_silent_fallback requiere confidence=0.0")
        # 4) confidence when segments present
        if len(self.segments) > 0 and self.confidence < 0.0:
            raise ValueError("confidence < 0 con segments presentes")
        # 5) NO validar end_ms vs total_duration_ms: start_ms/end_ms son metadata cruda del decoder T06
        #    (timestamps oficiales pertenecen a T07). Permitir que FW devuelva offsets que excedan duration.
        # 6) transcript_text vacío pero segments no vacíos: permitido (caso límite Whisper <|nospeech|>)
        return self

    # =====================================================================
    # BUILDER HELPER (chain of custody desde entidades anteriores)
    # =====================================================================
    @classmethod
    def build_chain(
        cls,
        *,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        lid_result: LanguageDetectionResult,
        **asr_fields: Any,
    ) -> "ASRResult":
        return cls(
            source_preprocessed_sha256=preprocessed_audio.sha256,
            vad_reference_sha256=vad_result.source_preprocessed_sha256,
            lid_reference_sha256=lid_result.source_preprocessed_sha256,
            sample_rate=preprocessed_audio.sample_rate,
            channels=preprocessed_audio.channels,
            bit_depth=preprocessed_audio.bit_depth,
            total_duration_ms=int(preprocessed_audio.duration_sec * 1000),
            **asr_fields,
        )
