"""Medición real de proceso. Sin inventar utilización de GPU si no hay API."""
from __future__ import annotations


def probe_runtime(preferred_device: str = "cpu") -> dict[str, float | str]:
    rss_mb = 0.0
    cpu = 0.0
    try:
        import psutil

        proc = psutil.Process()
        rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
        cpu = float(psutil.cpu_percent(interval=None))
        if cpu < 0:
            cpu = 0.0
        if cpu > 100:
            cpu = 100.0
    except Exception:
        rss_mb = 0.0
        cpu = 0.0

    device = "cpu"
    vram_used = 0.0
    vram_total = 0.0
    gpu_util = 0.0
    try:
        import torch

        pref = str(preferred_device or "cpu").lower()
        if pref != "cpu" and torch.cuda.is_available():
            device = "cuda"
            vram_used = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            vram_total = float(torch.cuda.get_device_properties(0).total_memory) / (
                1024.0 * 1024.0
            )
        elif pref == "mps" and getattr(torch.backends, "mps", None) is not None:
            if torch.backends.mps.is_available():
                device = "mps"
    except Exception:
        device = "cpu" if str(preferred_device).lower() == "cpu" else str(preferred_device)

    return {
        "rss_mb": max(0.0, rss_mb),
        "cpu_percent": cpu,
        "vram_used_mb": max(0.0, vram_used),
        "vram_total_mb": max(0.0, vram_total),
        "gpu_percent": gpu_util,
        "device": device,
    }
