"""Infrastructure Adapter Step 08 — Pyannote Speaker Diarization.

INFRASTRUCTURE ONLY. Aquí sí se importa pyannote (lazy, runtime).
Offline 100% local:
  - entry point = models/pyannote/config.yaml (ruta local)
  - sin Hugging Face remoto / sin descargas / sin URLs / sin tokens de autenticación
  - sin fallback remoto

Si falta modelo → ModelLoadError (sin descargar).
Si falla inferencia → DiarizationError.

El adapter NO implementa clustering, ni embeddings, ni segmentación:
delega todo al pipeline oficial pyannote.audio 3.3.2.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import (
    ConfigurationError,
    DiarizationError,
    ModelLoadError,
)
from app.domain.entities.diarization import (
    DIARIZATION_STRATEGY_PYANNOTE,
    MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
    DiarizationResult,
)
from app.domain.entities.media import PreprocessedAudio
from app.domain.entities.vad import VadResult
from app.domain.interfaces.diarization_ports import SpeakerDiarizerPort
from app.domain.value_objects.diarization import (
    DiarizationThresholds,
    SpeakerTurn,
)


# ---------------------------------------------------------------------------
# Constantes fijas (no configurables por usuario)
# ---------------------------------------------------------------------------
MODEL_FOLDERNAME = "pyannote"
PIPELINE_CONFIG_FILENAME = "config.yaml"

_MODEL_MIN_FILE_BYTES = 1024  # 1 KB mínimo por checkpoint/config (seguridad)
_CONFIG_MAX_BYTES = 1_000_000  # 1 MB máximo para config.yaml

_FORBIDDEN_REMOTE_MARKERS = (
    "://",
    "http:",
    "https:",
    "hf:",
    "huggingface",
)


# ---------------------------------------------------------------------------
# Loader local (inyectable para tests)
# ---------------------------------------------------------------------------


class PyannoteDiarizationModelLoader:
    """Valida y carga el pipeline local. 100% offline. SIN descargas."""

    __slots__ = ("_settings_getter",)

    def __init__(self, settings_getter: Any | None = None) -> None:
        self._settings_getter = settings_getter

    # -- helpers de paths ------------------------------------------------
    def _get_models_dir(self) -> Path:
        if self._settings_getter is not None:
            settings = self._settings_getter()
            models_dir = settings.paths.models_cache_dir
        else:
            from app.core.config import get_settings

            models_dir = get_settings().paths.models_cache_dir
        resolved = Path(models_dir).expanduser().resolve()
        return resolved

    def _pipeline_root(self) -> Path:
        return self._get_models_dir() / MODEL_FOLDERNAME

    def _config_path(self) -> Path:
        return self._pipeline_root() / PIPELINE_CONFIG_FILENAME

    @staticmethod
    def _ensure_contained(path: Path, root: Path, what: str) -> Path:
        """Anti path-traversal: la ruta resuelta debe estar dentro de root."""
        resolved = Path(path).expanduser().resolve()
        root_norm = root.resolve()
        try:
            resolved.relative_to(root_norm)
        except ValueError:
            raise ModelLoadError(
                f"Path traversal detectado: {what}={path!r} se resuelve FUERA de "
                f"models_cache_dir ({root_norm!s}). Regla T08 seguridad."
            ) from None
        return resolved

    @staticmethod
    def _has_remote_marker(value: str) -> bool:
        low = value.lower()
        return any(m in low for m in _FORBIDDEN_REMOTE_MARKERS)

    # -- validaciones de estructura --------------------------------------
    def validate(self) -> Path:
        """Valida la estructura de models/pyannote y devuelve la ruta del config."""
        root = self._pipeline_root()
        if not root.exists():
            raise ModelLoadError(
                f"{root!s} no existe. Colocar manualmente el pipeline pyannote "
                f"en models_cache_dir (regla T08: NO descargar automáticamente)."
            )
        if not root.is_dir():
            raise ModelLoadError(f"{root!s} debe ser un directorio, no un archivo.")

        config_path = self._config_path()
        if not config_path.is_file():
            raise ModelLoadError(
                f"Falta {PIPELINE_CONFIG_FILENAME!r} en {root!s}. "
                f"Es el ENTRY POINT local del pipeline (regla T08)."
            )
        if config_path.stat().st_size > _CONFIG_MAX_BYTES:
            raise ModelLoadError(
                f"config.yaml excede tamaño máximo {_CONFIG_MAX_BYTES} bytes."
            )

        self._validate_config_references(config_path, root)
        return config_path

    def _validate_config_references(self, config_path: Path, root: Path) -> None:
        """Valida que config.yaml solo contenga referencias locales válidas."""
        try:
            import yaml  # runtime, infra only
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError("PyYAML no disponible para leer config.yaml.") from exc

        try:
            raw = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ModelLoadError(f"No se pudo leer config.yaml: {exc!r}") from exc

        try:
            config = yaml.safe_load(raw)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"config.yaml no es YAML válido: {exc!r}") from exc

        if not isinstance(config, dict):
            raise ModelLoadError("config.yaml debe ser un mapping (dict) en la raíz.")

        # pipeline.name debe estar presente y ser local
        pipeline_block = config.get("pipeline")
        if not isinstance(pipeline_block, dict):
            raise ModelLoadError("config.yaml debe contener la clave 'pipeline' (dict).")
        name = pipeline_block.get("name")
        if not isinstance(name, str) or "SpeakerDiarization" not in name:
            raise ModelLoadError(
                "config.yaml 'pipeline.name' debe referenciar la clase local "
                "SpeakerDiarization (sin IDs remotos)."
            )

        params = pipeline_block.get("params", {})
        if not isinstance(params, dict):
            raise ModelLoadError("config.yaml 'pipeline.params' debe ser dict.")

        # Referencias a segmentation (checkpoint) y embedding (source)
        self._validate_local_reference(params, "segmentation", root, expect_dir=False)
        self._validate_local_reference(params, "embedding", root, expect_dir=True)

        # No permitir use_auth_token ni valores remotos en todo el config
        self._reject_remote_strings(config)

    def _validate_local_reference(
        self, params: dict[str, Any], key: str, root: Path, *, expect_dir: bool
    ) -> None:
        if key not in params:
            # No obligatorio en el config si se pasa por constructor (pero offline
            # requiere que, si está, sea local). Permitir ausencia para flexibilidad.
            return
        value = params[key]
        if isinstance(value, (dict, list)):
            # pyannote acepta Mapping; para T08 exigimos ruta local simple (str/Path)
            raise ModelLoadError(
                f"config.yaml 'pipeline.params.{key}' debe ser una ruta local (str), "
                f"no un mapping/lista (regla T08 offline)."
            )
        s = str(value)
        if self._has_remote_marker(s):
            raise ModelLoadError(
                f"config.yaml 'pipeline.params.{key}' contiene referencia remota "
                f"prohibida: {s!r}. Solo rutas filesystem locales (regla T08)."
            )
        p = Path(s)
        if not p.is_absolute():
            p = root / p
        resolved = self._ensure_contained(p, root, f"pipeline.params.{key}")
        if expect_dir:
            if not resolved.is_dir():
                raise ModelLoadError(
                    f"config.yaml 'pipeline.params.{key}'={s!r} no existe como directorio local."
                )
        else:
            if not resolved.is_file():
                raise ModelLoadError(
                    f"config.yaml 'pipeline.params.{key}'={s!r} no existe como archivo local."
                )
            if resolved.stat().st_size < _MODEL_MIN_FILE_BYTES:
                raise ModelLoadError(
                    f"Checkpoint {key}={s!r} demasiado pequeño "
                    f"(< {_MODEL_MIN_FILE_BYTES} bytes), posible corrupto."
                )

    def _reject_remote_strings(self, config: dict[str, Any]) -> None:
        """Recorre el config y rechaza cualquier string con marcador remoto/token."""

        def _walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("use_auth_token", "token", "auth_token"):
                        raise ModelLoadError(
                            f"config.yaml contiene clave prohibida {k!r} en {path} (regla T08 offline)."
                        )
                    _walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    _walk(v, f"{path}[{i}]")
            elif isinstance(node, str):
                if self._has_remote_marker(node):
                    raise ModelLoadError(
                        f"config.yaml contiene referencia remota prohibida en {path}: "
                        f"{node!r}. Solo rutas locales (regla T08)."
                    )

        _walk(config, "config")

    def load(self, config_path: Path) -> Any:
        """Carga el pipeline desde config local. Sin auth, sin cache remoto."""
        try:
            from pyannote.audio import Pipeline as _Pipeline  # runtime, infra only
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"pyannote.audio no disponible: {exc!r}. "
                "Requiere el extra [diarization] instalado."
            ) from exc
        try:
            return _Pipeline.from_pretrained(str(config_path))
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"No se pudo cargar el pipeline local desde {config_path!s}: {exc!r}"
            ) from exc


# ---------------------------------------------------------------------------
# Procesamiento puro de turnos (determinista, testable sin pyannote)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RawTurn:
    start_ms: int
    end_ms: int
    label: str
    confidence: float = 1.0


def _remap_label(index: int) -> str:
    return f"SPEAKER_{index:02d}"


def _normalize_labels(turns: list[tuple[int, int, str]]) -> list[_RawTurn]:
    """Valida labels; los inválidos se remapean por primera aparición cronológica."""
    mapping: dict[str, str] = {}
    next_index = 0
    out: list[_RawTurn] = []
    for start_ms, end_ms, label in turns:
        key = str(label).strip().upper()
        valid = (
            len(key) == 10
            and key.startswith("SPEAKER_")
            and key[8:].isdigit()
            and 0 <= int(key[8:]) <= 99
        )
        if not valid:
            if key not in mapping:
                mapping[key] = _remap_label(next_index)
                next_index += 1
            key = mapping[key]
        out.append(_RawTurn(start_ms=start_ms, end_ms=end_ms, label=key))
    return out


def _seconds_to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000.0))


def _vad_allowed_intervals(
    voice_intervals: list[tuple[int, int]],
    total_duration_ms: int,
    tolerance_ms: int,
) -> list[tuple[int, int]]:
    """Expande (con tolerancia) y fusiona los intervalos VAD."""
    expanded: list[tuple[int, int]] = []
    for s, e in voice_intervals:
        ns = max(0, int(s) - int(tolerance_ms))
        ne = min(int(total_duration_ms), int(e) + int(tolerance_ms))
        if ne > ns:
            expanded.append((ns, ne))
    if not expanded:
        return []
    expanded.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[int, int]] = [expanded[0]]
    for s, e in expanded[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _intersect_with_vad(
    turns: list[_RawTurn],
    allowed: list[tuple[int, int]],
) -> list[_RawTurn]:
    """Cada turno se recorta a la unión de intervalos VAD. Fuera → descartado."""
    if not allowed:
        return []
    out: list[_RawTurn] = []
    for t in turns:
        pieces: list[tuple[int, int]] = []
        for s, e in allowed:
            is_ = max(int(t.start_ms), s)
            ie = min(int(t.end_ms), e)
            if ie > is_:
                pieces.append((is_, ie))
        # Un turno puede intersectar varios intervalos VAD; los unimos solo si
        # quedan contiguos (mismo label); si no, se fragmentan en turnos separados.
        for s, e in pieces:
            out.append(
                _RawTurn(start_ms=s, end_ms=e, label=t.label, confidence=t.confidence)
            )
    return out


def _apply_domain_rules(
    turns: list[_RawTurn],
    *,
    total_duration_ms: int,
    vad_voice_intervals: list[tuple[int, int]],
    thresholds: DiarizationThresholds,
) -> list[_RawTurn]:
    """Pipeline determinista: clamp → sort → overlaps → VAD → >0 → min → dedupe."""
    # 2) clamp (start >= 0, end <= total)
    clamped: list[_RawTurn] = []
    for t in turns:
        s = max(0, int(t.start_ms))
        e = min(int(total_duration_ms), int(t.end_ms))
        if e > s:
            clamped.append(_RawTurn(s, e, t.label, t.confidence))

    # 3) sort ASC
    clamped.sort(key=lambda x: (x.start_ms, x.end_ms))

    # 4) resolver overlaps (corte limpio: recortar solapado; contenido → descartar)
    resolved: list[_RawTurn] = []
    for t in clamped:
        if not resolved:
            resolved.append(t)
            continue
        last = resolved[-1]
        if t.start_ms < last.end_ms:
            # solapado: recortar el turno posterior al fin del anterior
            new_start = last.end_ms
            if new_start >= t.end_ms:
                # completamente contenido o sin espacio → descartar
                continue
            t = _RawTurn(new_start, t.end_ms, t.label, t.confidence)
        resolved.append(t)

    # 5) intersección con VAD (regla T04 autoridad)
    vad_allowed: list[tuple[int, int]] = _vad_allowed_intervals(
        vad_voice_intervals,
        total_duration_ms,
        int(thresholds.max_overlap_tolerance_ms),
    )
    resolved = _intersect_with_vad(resolved, vad_allowed)

    # 6) eliminar duración <= 0 (ya garantizado por clamp/intersect, safety)
    resolved = [t for t in resolved if t.end_ms > t.start_ms]

    # 7) duración mínima: merge de adyacentes mismo label con gap <= min_gap, luego descartar cortos
    resolved = _merge_proximal_same_label(resolved, int(thresholds.min_gap_ms))
    resolved = [
        t for t in resolved if (t.end_ms - t.start_ms) >= int(thresholds.min_speaker_duration_ms)
    ]

    # 8) dedupe (mismo label + mismo rango)
    seen: set[tuple[str, int, int]] = set()
    deduped: list[_RawTurn] = []
    for t in resolved:
        key = (t.label, t.start_ms, t.end_ms)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    deduped.sort(key=lambda x: (x.start_ms, x.end_ms))
    return deduped


def _merge_proximal_same_label(turns: list[_RawTurn], min_gap_ms: int) -> list[_RawTurn]:
    """Fusiona turnos adyacentes del mismo label si el gap <= min_gap_ms."""
    if not turns:
        return []
    turns = sorted(turns, key=lambda x: (x.start_ms, x.end_ms))
    out: list[_RawTurn] = [turns[0]]
    for t in turns[1:]:
        last = out[-1]
        if t.label == last.label and (t.start_ms - last.end_ms) <= int(min_gap_ms):
            out[-1] = _RawTurn(
                min(last.start_ms, t.start_ms),
                max(last.end_ms, t.end_ms),
                last.label,
                last.confidence,
            )
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PyannoteSpeakerDiarizationAdapter(SpeakerDiarizerPort):
    """Adapter pyannote.audio 3.3.2. Lazy singleton. Offline estricto."""

    def __init__(
        self,
        *,
        model_loader: Any | None = None,
        settings_getter: Any | None = None,
    ) -> None:
        self._loader: Any = model_loader or PyannoteDiarizationModelLoader(
            settings_getter=settings_getter
        )
        self._pipeline: Any = None
        self._pipeline_loaded = False

    # ------------------------------------------------------------------
    # Carga lazy (singleton)
    # ------------------------------------------------------------------
    def _get_pipeline(self) -> Any:
        if not self._pipeline_loaded:
            config_path = self._loader.validate()
            self._pipeline = self._loader.load(config_path)
            self._pipeline_loaded = True
        return self._pipeline

    # ------------------------------------------------------------------
    # Public API (SpeakerDiarizerPort)
    # ------------------------------------------------------------------
    def diarize(
        self,
        preprocessed_audio: PreprocessedAudio,
        vad_result: VadResult,
        *,
        thresholds: DiarizationThresholds | None = None,
        job_id: str | None = None,
        logger: Any | None = None,
    ) -> DiarizationResult:
        thr: DiarizationThresholds = thresholds or DiarizationThresholds()
        total_ms = int(round(float(preprocessed_audio.duration_sec) * 1000.0))
        t0 = time.perf_counter()

        pipeline = self._get_pipeline()

        # min/max speakers desde thresholds
        min_speakers = 1
        max_speakers = int(thr.max_num_speakers)

        try:
            annotation = pipeline.apply(
                str(preprocessed_audio.wav_path),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except Exception as exc:  # noqa: BLE001
            raise DiarizationError(
                f"pyannote apply falló sobre {preprocessed_audio.wav_path.name!r}: {exc!r}"
            ) from exc

        turns = self._annotation_to_turns(annotation)

        # Aplicar reglas de dominio (overlap, VAD, duración mínima)
        vad_voice_intervals: list[tuple[int, int]] = [
            (int(vi.start_ms), int(vi.end_ms)) for vi in vad_result.voice_intervals
        ]
        processed = _apply_domain_rules(
            turns,
            total_duration_ms=total_ms,
            vad_voice_intervals=vad_voice_intervals,
            thresholds=thr,
        )

        speaker_turns: tuple[SpeakerTurn, ...] = tuple(
            SpeakerTurn(
                speaker_label=t.label,
                start_ms=t.start_ms,
                end_ms=t.end_ms,
                duration_ms=t.end_ms - t.start_ms,
                confidence=t.confidence,
            )
            for t in processed
        )
        num_speakers = len({t.speaker_label for t in speaker_turns})

        if logger is not None:
            try:
                logger.info(
                    "diarization_inference_done",
                    extra={
                        "elapsed_sec": round(time.perf_counter() - t0, 4),
                        "num_speakers": num_speakers,
                        "num_turns": len(speaker_turns),
                        "job_id": job_id,
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        return DiarizationResult(
            source_preprocessed_sha256=str(preprocessed_audio.sha256),
            vad_reference_sha256=str(vad_result.source_preprocessed_sha256),
            total_duration_ms=total_ms,
            speaker_turns=speaker_turns,
            num_speakers=num_speakers,
            num_turns=len(speaker_turns),
            strategy=DIARIZATION_STRATEGY_PYANNOTE,
            model_label=MODEL_LABEL_PYANNOTE_DIARIZATION_V1_LITERAL,
            job_id=job_id,
            analysis_metadata={"source": "pyannote.audio.local.v3.3.2"},
        )

    # ------------------------------------------------------------------
    # Conversión Annotation → turns
    # ------------------------------------------------------------------
    @staticmethod
    def _annotation_to_turns(annotation: Any) -> list[_RawTurn]:
        raw: list[tuple[int, int, str]] = []
        for segment, _track, label in annotation.itertracks(yield_label=True):
            raw.append(
                (
                    _seconds_to_ms(float(segment.start)),
                    _seconds_to_ms(float(segment.end)),
                    str(label),
                )
            )
        return _normalize_labels(raw)


__all__ = [
    "PyannoteSpeakerDiarizationAdapter",
    "PyannoteDiarizationModelLoader",
    "MODEL_FOLDERNAME",
    "PIPELINE_CONFIG_FILENAME",
]
