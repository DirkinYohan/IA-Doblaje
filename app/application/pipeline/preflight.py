"""Preflight / precheck del pipeline — diagnóstico de preparación real.

SOLO LECTURA. Verifica dependencias, binarios, modelos y directorios.
NO descarga nada. NO inicia el pipeline.
"""
from __future__ import annotations

import importlib.util
import shutil
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
    for mod in ("torch", "torchaudio", "faster_whisper", "pyannote.audio", "speechbrain"):
        spec = importlib.util.find_spec(mod)
        add(mod, spec is not None, "" if spec is not None else "NO INSTALADO")

    # --- FFmpeg / FFprobe ---
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    add("FFmpeg", ffmpeg is not None, ffmpeg or "NO ENCONTRADO EN PATH")
    add("FFprobe", ffprobe is not None, ffprobe or "NO ENCONTRADO EN PATH")

    # --- Modelos ---
    models_dir = Path(settings.paths.models_cache_dir)
    model_checks = {
        "VAD model (silero_vad_v5.1.jit)": models_dir / "silero_vad_v5.1.jit",
        "LID model (whisper_encoder_small_multilingual_lid_v1.jit)": models_dir / "whisper_encoder_small_multilingual_lid_v1.jit",
        "ASR model (whisper_small_ct2_int8_fp16_local_v1/)": models_dir / "whisper_small_ct2_int8_fp16_local_v1",
        "Diarization model (pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin)": models_dir / "pyannote" / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin",
    }
    for name, path in model_checks.items():
        add(name, path.exists(), str(path) if not path.exists() else "")

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
