"""Entidad DiarizationResult Step 08 — Capa DOMAIN.

Frozen + extra=forbid. Chain of custody via:
  PreprocessedAudio.sha256 == VadResult.source_preprocessed_sha256
  == DiarizationResult.source_preprocessed_sha256

Sin dependencias infraestructura (no torch/pyannote/speechbrain/numpy).
"""
from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.domain.value_objects.audio import Sha256Hash
from app.domain.value_objects.diarization import (
    DiarizationMs,
    SpeakerTurn,
)

DIARIZATION_STRATEGY_PYANNOTE: Literal["pyannote_speaker_diarization_v1"] = (
    "pyannote_speaker_diarization_v1"
)
DIARIZATION_STRATEGY_SILENT_FALLBACK: Literal["diarization_silent_fallback"] = (
    "diarization_silent_fallback"
)

DIARIZATION_STRATEGIES_TYPE = (
    Literal["pyannote_speaker_diarization_v1"]
    | Literal["diarization_silent_fallback"]
)

MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL: Literal[
    "pyannote:speaker-diarization:3.1:local-v1"
] = "pyannote:speaker-diarization:3.1:local-v1"


class DiarizationResult(BaseModel):
    """Resultado Step 08 Speaker Diarization. Inmutable. Chain of Custody.

    No almacena: bytes, tensores, audio, transcripciones, objetos pyannote/torch.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    # ---- Chain of Custody T03 → T04 → T08 -------------------------------
    source_preprocessed_sha256: Sha256Hash = Field(
        ..., description="Enlace T03 → T08. Debe coincidir con PreprocessedAudio.sha256."
    )
    vad_reference_sha256: Sha256Hash = Field(
        ..., description="Enlace T04 → T08. Debe coincidir con VadResult.source_preprocessed_sha256."
    )

    # ---- Metadata T03 passthrough ---------------------------------------
    total_duration_ms: DiarizationMs = Field(
        ..., description="Duración total del audio (ms) = PreprocessedAudio.duration_sec*1000."
    )

    # ---- Outputs --------------------------------------------------------
    speaker_turns: tuple[SpeakerTurn, ...] = Field(
        default_factory=tuple, description="Turnos ASC, sin overlaps, dentro de [0, total]."
    )
    num_speakers: int = Field(0, ge=0, description="Número de speakers distintos.")
    num_turns: int = Field(0, ge=0, description="Número total de turnos.")

    # ---- Estrategia / modelo -------------------------------------------
    strategy: DIARIZATION_STRATEGIES_TYPE = Field(
        DIARIZATION_STRATEGY_PYANNOTE,
        description="'pyannote_speaker_diarization_v1' o 'diarization_silent_fallback'.",
    )
    model_label: Literal["pyannote:speaker-diarization:3.1:local-v1"] = Field(
        MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
        description="Etiqueta fija del modelo aplicado (pipeline local v3.1).",
    )

    # ---- Job / auditoría ------------------------------------------------
    job_id: str | None = Field(default=None, min_length=8, max_length=128)
    analysis_metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Solo metadata auditoría. Sin bytes/tensores/audio/objetos pyannote.",
    )

    # =====================================================================
    # VALIDADORES
    # =====================================================================

    @field_validator("source_preprocessed_sha256", "vad_reference_sha256")
    @classmethod
    def _sha_64hex_lower(cls, v: object) -> str:
        s = str(v or "").strip().lower()
        if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
            raise ValueError(f"SHA inválido 64-hex lower: {s[:20]!r}...")
        return s

    @field_validator("analysis_metadata")
    @classmethod
    def _metadata_no_bytes(cls, v: object) -> dict[str, object]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("analysis_metadata debe ser dict")
        for k, val in v.items():
            if isinstance(val, (bytes, bytearray, memoryview)):
                raise ValueError(
                    f"analysis_metadata[{k!r}] contiene bytes/tensores prohibidos."
                )
        return v

    @model_validator(mode="after")
    def _validate_all(self) -> "DiarizationResult":
        # (A) Chain of custody: source == vad
        if self.source_preprocessed_sha256 != self.vad_reference_sha256:
            raise ValueError(
                "Chain of Custody rota T08: source_preprocessed_sha256 != vad_reference_sha256."
            )

        # (B) num consistency
        if self.num_turns != len(self.speaker_turns):
            raise ValueError(
                f"num_turns={self.num_turns} != len(speaker_turns)={len(self.speaker_turns)}"
            )

        # (C) num_speakers consistency: labels únicos
        labels = {str(t.speaker_label) for t in self.speaker_turns}
        if self.num_speakers != len(labels):
            raise ValueError(
                f"num_speakers={self.num_speakers} != labels únicos={len(labels)} "
                f"({sorted(labels)})"
            )

        # (D) Orden ASC + sin overlaps + dentro de [0, total]
        prev_end: int = -1
        for i, t in enumerate(self.speaker_turns):
            if int(t.start_ms) < 0:
                raise ValueError(f"speaker_turns[{i}].start_ms < 0")
            if int(t.end_ms) > int(self.total_duration_ms):
                raise ValueError(
                    f"speaker_turns[{i}].end_ms={t.end_ms} > total_duration_ms={self.total_duration_ms}"
                )
            if int(t.start_ms) < prev_end:
                raise ValueError(
                    f"speaker_turns overlap/unsorted en idx={i}: start={t.start_ms} < prev_end={prev_end}"
                )
            prev_end = int(t.end_ms)

        # (E) Silent fallback consistency
        if self.strategy == DIARIZATION_STRATEGY_SILENT_FALLBACK:
            if len(self.speaker_turns) != 0:
                raise ValueError(
                    "diarization_silent_fallback requiere speaker_turns=()"
                )
            if self.num_speakers != 0:
                raise ValueError("diarization_silent_fallback requiere num_speakers=0")
            if self.num_turns != 0:
                raise ValueError("diarization_silent_fallback requiere num_turns=0")

        return self


__all__ = [
    "DiarizationResult",
    "DIARIZATION_STRATEGY_PYANNOTE",
    "DIARIZATION_STRATEGY_SILENT_FALLBACK",
    "DIARIZATION_STRATEGIES_TYPE",
    "MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL",
]
