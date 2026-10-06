"""Carga el YAML del perfil y lo convierte en parámetros reales de ASR.

El único modelo ASR local del proyecto es Faster-Whisper small. El YAML
elige carpeta, compute type, beam, word timestamps y chunking. Si la carpeta
no existe, el adapter falla al cargar: no hay sustitución silenciosa de modelo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import ConfigurationError

LOCAL_SMALL_FOLDER = "whisper_small_ct2_int8_fp16_local_v1"


@dataclass(frozen=True, slots=True)
class AsrProfileSpec:
    profile: str
    adapter: str
    model_name: str
    model_folder: str
    compute_type: str
    beam_size: int
    word_timestamps: bool
    chunk_by_vad: bool
    compute_note: str = ""


def load_asr_profile(profile: str, configs_dir: Path) -> AsrProfileSpec:
    name = str(profile).strip().lower()
    path = Path(configs_dir) / f"{name}.yaml"
    if not path.is_file():
        raise ConfigurationError(f"Perfil YAML no encontrado: {path}")
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(f"PyYAML no disponible: {exc!r}") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(f"YAML de perfil inválido {path}: {exc!r}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"YAML de perfil {path} no es un mapping.")
    asr = data.get("asr") or {}
    if not isinstance(asr, dict):
        raise ConfigurationError(f"Clave asr inválida en {path}")
    adapter = str(asr.get("adapter") or "faster-whisper").strip()
    if adapter != "faster-whisper":
        raise ConfigurationError(
            f"Perfil {name} pide adapter={adapter!r}. "
            "El motor local solo tiene el adapter faster-whisper."
        )
    model_name = str(asr.get("model_name") or "small").strip()
    folder = str(asr.get("model_folder") or LOCAL_SMALL_FOLDER).strip()
    compute = str(asr.get("compute_type") or "int8").strip()
    beam = int(asr.get("beam_size") or 1)
    if beam < 1:
        raise ConfigurationError(f"beam_size inválido en {path}: {beam}")
    return AsrProfileSpec(
        profile=name,
        adapter=adapter,
        model_name=model_name,
        model_folder=folder,
        compute_type=compute,
        beam_size=beam,
        word_timestamps=bool(asr.get("word_timestamps", True)),
        chunk_by_vad=bool(asr.get("chunk_by_vad", True)),
    )


def resolve_compute_type(requested: str, device: str) -> tuple[str, str]:
    """float16 en CPU no es estable en CTranslate2. El cambio queda explícito."""
    req = requested.strip()
    if device == "cpu" and req in {"float16", "int8_float16"}:
        return "int8", f"compute_type {req} no aplica en CPU; se usa int8"
    return req, ""
