"""Preflight / precheck del pipeline — diagnóstico de preparación real.

SOLO LECTURA. Verifica dependencias, binarios, modelos y directorios.
NO descarga nada. NO inicia el pipeline.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from app.core.config import AppSettings, get_settings


@dataclass(frozen=True)
class PreflightItem:
    component: str
    ok: bool
    detail: str = ""


def run_preflight(settings: AppSettings | None = None) -> list[PreflightItem]:
    """Devuelve la lista de checks de preparación (solo lectura)."""
    settings = settings or get_settings()
    items: list[PreflightItem] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        items.append(PreflightItem(component=name, ok=ok, detail=detail))

    # --- Python / PyTorch / libs ---
    add("Python", True)
    for mod in ("torch", "torchaudio", "faster_whisper", "pyannote.audio"):
        try:
            spec = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            spec = None
        add(mod, spec is not None, "" if spec is not None else "NO INSTALADO")

    # --- FFmpeg / FFprobe ---
    # Se usa el mismo resolutor que el pipeline real: `shutil.which` no ve el
    # PATH de WinGet ni el PATH persistido del usuario en Windows.
    from app.infrastructure.audio.ffmpeg_adapters import _locate_binary

    for name, binary in (("FFmpeg", "ffmpeg"), ("FFprobe", "ffprobe")):
        located = ""
        try:
            located = _locate_binary(binary) or ""
        except Exception:  # noqa: BLE001
            located = ""
        add(name, bool(located), located or "NO ENCONTRADO")

    # --- Modelos ---
    models_dir = Path(settings.paths.models_cache_dir)
    model_checks: dict[str, tuple[Path, int]] = {
        "VAD model (silero_vad_v5.1.jit)": (models_dir / "silero_vad_v5.1.jit", 1024),
        "LID model (whisper_encoder_small_multilingual_lid_v1.jit)": (
            models_dir / "whisper_encoder_small_multilingual_lid_v1.jit", 100 * 1024 * 1024,
        ),
        "ASR model (whisper_small_ct2_int8_fp16_local_v1/model.bin)": (
            models_dir / "whisper_small_ct2_int8_fp16_local_v1" / "model.bin", 1024,
        ),
        "Diarization model (pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin)": (
            models_dir / "pyannote" / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin", 1024,
        ),
    }
    if bool(getattr(settings.translation, "enabled", False)):
        model_checks["Translation model (m2m100_418M/pytorch_model.bin)"] = (
            models_dir / "m2m100_418M" / "pytorch_model.bin", 1024,
        )
    for name, (path, min_bytes) in model_checks.items():
        if not path.is_file():
            add(name, False, f"NO EXISTE: {path}")
        elif path.stat().st_size < min_bytes:
            add(name, False, f"ARCHIVO INCOMPLETO ({path.stat().st_size} bytes): {path}")
        else:
            add(name, True)

    # --- Directorios ---
    for name, path in (
        ("Input directory", settings.paths.data_input_dir),
        ("Output directory", settings.paths.data_output_dir),
        ("Temp directory", settings.paths.data_temp_dir),
        ("Models directory", settings.paths.models_cache_dir),
    ):
        exists = Path(path).exists()
        add(name, exists, str(path) if not exists else "")

    return items


def preflight_all_ok(items: list[PreflightItem]) -> bool:
    return all(item.ok for item in items)


__all__ = ["PreflightItem", "run_preflight", "preflight_all_ok"]
