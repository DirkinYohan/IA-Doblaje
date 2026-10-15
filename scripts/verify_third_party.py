"""Verificacion standalone de entorno de terceros (T02.6 implementacion).

Uso: python scripts/verify_third_party.py

Salida: Rich table CHECK | STATUS | VALUE | MESSAGE
Estados: PASS | WARNING | ERROR
Exit codes:
  0 = todo correcto
  1 = warnings (no critico, continuar con cuidado)
  2 = errores criticos (no se puede operar el motor)

REGLAS T02:
  - Ausencia de PyTorch / CUDA en Fase 1 NO es ERROR critico (WARNING).
  - No importar modelos IA, no descargar pesos, no aceptar HF gates.
  - No leakear tokens. HF_TOKEN: si existe -> PASS (sin imprimir valor).
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Hacemos ejecutable standalone (no necesita el paquete instalado via pip install -e)
# pero si esta disponible, mejor. Si no, aun asi corre.
try:  # pragma: no cover - import standalone
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:  # pragma: no cover - rich en el proyecto
    print("ERROR: Rich no esta disponible. Instalar dependencias de proyecto primero.")
    print("    pip install -e \".[dev]\"")
    sys.exit(2)


# =============================================================================
# Estructuras internas
# =============================================================================

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_ERROR = "ERROR"

STATUS_COLORS = {
    STATUS_PASS: "green",
    STATUS_WARNING: "yellow",
    STATUS_ERROR: "red",
}


@dataclass(slots=True)
class CheckResult:
    check: str
    status: str
    value: str
    message: str

    @property
    def rank(self) -> int:
        if self.status == STATUS_ERROR:
            return 2
        if self.status == STATUS_WARNING:
            return 1
        return 0


# =============================================================================
# Checkers individuales
# =============================================================================


def _check_python() -> CheckResult:
    py_min = (3, 11)
    py_max = (3, 13)  # <3.13
    ver = sys.version_info[:3]
    ver_str = f"{ver[0]}.{ver[1]}.{ver[2]}"
    if py_min <= ver[:2] < py_max:
        return CheckResult("Python", STATUS_PASS, ver_str, "Version compatible.")
    return CheckResult(
        "Python", STATUS_ERROR, ver_str,
        f"Requerido Python >= {py_min[0]}.{py_min[1]} y < {py_max[0]}.{py_max[1]}."
    )


def _find_pyproject_root() -> Path:
    """Encuentra el proyecto root (donde esta pyproject.toml)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here


def _check_torch() -> CheckResult:
    try:
        import torch
    except Exception as exc:
        return CheckResult(
            "PyTorch", STATUS_WARNING, "not installed",
            "PyTorch no disponible (no critico en Fase 1 T02). Instalar via pip install torch.",
        )
    try:
        ver = str(getattr(torch, "__version__", "unknown"))
        cuda_ok = bool(getattr(torch.cuda, "is_available", lambda: False)())
        val = f"{ver} (cuda={cuda_ok})"
        return CheckResult("PyTorch", STATUS_PASS, val, "Libreria instalada.")
    except Exception as exc:
        return CheckResult(
            "PyTorch", STATUS_WARNING, "broken",
            f"PyTorch instalado pero error al inspeccionar: {exc}",
        )


def _check_cuda() -> CheckResult:
    try:
        import torch
    except Exception:
        return CheckResult(
            "CUDA", STATUS_WARNING, "unknown",
            "PyTorch no instalado, no se puede detectar CUDA (no critico T02).",
        )
    try:
        avail = bool(torch.cuda.is_available())
        if not avail:
            return CheckResult(
                "CUDA", STATUS_WARNING, "no",
                "CUDA no disponible. Solo CPU (no critico T02).",
            )
        version = None
        try:
            version = str(torch.cuda.version())
        except Exception:
            version = "?"
        count = int(torch.cuda.device_count() or 0)
        return CheckResult(
            "CUDA", STATUS_PASS, f"v{version} devices={count}",
            "CUDA disponible via PyTorch.",
        )
    except Exception as exc:
        return CheckResult(
            "CUDA", STATUS_WARNING, "error",
            f"Fallo deteccion CUDA: {exc}",
        )


def _check_cudnn() -> CheckResult:
    try:
        import torch
    except Exception:
        return CheckResult(
            "cuDNN", STATUS_WARNING, "unknown",
            "PyTorch no instalado, no se puede detectar cuDNN (no critico T02).",
        )
    try:
        back = getattr(torch, "backends", None)
        cudnn = getattr(back, "cudnn", None) if back else None
        if cudnn is None:
            return CheckResult("cuDNN", STATUS_WARNING, "n/a", "PyTorch sin backends.cudnn.")
        avail = bool(cudnn.is_available())
        if not avail:
            return CheckResult(
                "cuDNN", STATUS_WARNING, "no",
                "cuDNN no disponible. CUDA puede degradar rendimiento (no critico T02).",
            )
        try:
            ver_int = int(cudnn.version())
        except Exception:
            ver_int = 0
        ver_str = f"{ver_int // 1000}.{(ver_int // 100) % 10}.{ver_int % 100}" if ver_int else str(ver_int)
        return CheckResult("cuDNN", STATUS_PASS, ver_str, "cuDNN disponible via PyTorch.")
    except Exception as exc:
        return CheckResult("cuDNN", STATUS_WARNING, "error", f"Fallo deteccion cuDNN: {exc}")


def _check_bin(name: str, friendly: str) -> CheckResult:
    path = shutil.which(name)
    if not path:
        return CheckResult(
            friendly, STATUS_ERROR, "not found",
            f"Binario '{name}' no encontrado en PATH. Es obligatorio para extraer audio.",
        )
    try:
        import subprocess
        out = subprocess.run(
            [name, "-version"] if name == "ffprobe" else [name, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = (out.stdout or "") + (out.stderr or "")
        first = text.splitlines()[0][:80] if text.splitlines() else name
        return CheckResult(friendly, STATUS_PASS, path[:80], f"OK: {first}")
    except Exception as exc:
        return CheckResult(friendly, STATUS_WARNING, path[:80], f"Encontrado pero -version fallo: {exc}")


def _check_hf_lib() -> CheckResult:
    try:
        import huggingface_hub
        ver = str(getattr(huggingface_hub, "__version__", "unknown"))
        return CheckResult("HuggingFace Hub", STATUS_PASS, ver, "Libreria instalada.")
    except Exception as exc:
        return CheckResult(
            "HuggingFace Hub", STATUS_WARNING, "not installed",
            f"Libreria no disponible (no critico T02): {exc}",
        )


def _check_hf_token() -> CheckResult:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    token = str(token).strip()
    if not token:
        # Buscar en ~/.cache/huggingface/token
        home = Path.home()
        candidate = home / ".cache" / "huggingface" / "token"
        try:
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8").strip()
                if content:
                    token = content
        except OSError:
            pass
    if not token:
        return CheckResult(
            "HF_TOKEN", STATUS_WARNING, "not set",
            "Token HF no encontrado. Modelos gated (pyannote) no accesibles sin token.",
        )
    # Presente: NO imprimir valor. Longitud se puede mostrar.
    val = f"set (len={len(token)})"
    return CheckResult("HF_TOKEN", STATUS_PASS, val, "Token detectado (no se imprime por seguridad).")


def _check_vram() -> CheckResult:
    try:
        import torch
    except Exception:
        return CheckResult(
            "VRAM", STATUS_WARNING, "unknown",
            "PyTorch no instalado: no se puede medir VRAM (no critico T02).",
        )
    try:
        if not torch.cuda.is_available():
            return CheckResult("VRAM", STATUS_WARNING, "N/A (CPU only)", "CUDA no disponible.")
        count = int(torch.cuda.device_count() or 0)
        total_mb = 0
        avail_mb = 0
        min_avail_gb = None
        for i in range(count):
            try:
                total_bytes = int(getattr(torch.cuda.get_device_properties(i), "total_memory", 0))
                info = torch.cuda.mem_get_info(i)
                avail_bytes = int(info[0])
            except Exception:
                total_bytes, avail_bytes = 0, 0
            total_mb += int(total_bytes / 1024**2)
            avail_mb += int(avail_bytes / 1024**2)
            g = avail_bytes / 1024**3
            min_avail_gb = g if min_avail_gb is None else min(min_avail_gb, g)
        val = f"{count} GPUs, total={total_mb}MB avail={avail_mb}MB"
        if min_avail_gb is None or min_avail_gb < 3.0:
            msg = (
                f"VRAM disponible por GPU < 3 GB. "
                f"Solo perfil performance o CPU. No critico T02."
            )
            return CheckResult("VRAM", STATUS_WARNING, val, msg)
        return CheckResult("VRAM", STATUS_PASS, val, "VRAM suficiente para al menos performance.")
    except Exception as exc:
        return CheckResult("VRAM", STATUS_WARNING, "error", f"Fallo medicion VRAM: {exc}")


def _check_disk() -> CheckResult:
    root = _find_pyproject_root()
    try:
        usage = shutil.disk_usage(root)
        total_gb = usage.total / 1024**3
        free_gb = usage.free / 1024**3
        val = f"free={free_gb:.1f}GB / total={total_gb:.1f}GB"
        if free_gb < 15.0:
            return CheckResult(
                "Disk Space", STATUS_WARNING, val,
                "< 15 GB libres. Modelos Whisper/pyannote + cache pesan ~10 GB.",
            )
        return CheckResult("Disk Space", STATUS_PASS, val, "Espacio en disco suficiente.")
    except Exception as exc:
        return CheckResult("Disk Space", STATUS_ERROR, "error", f"No se puede medir disco: {exc}")


def _check_cpu() -> CheckResult:
    logical = os.cpu_count() or 1
    physical: int | None = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    val = f"{platform.processor() or 'CPU'} logical={logical}"
    if physical:
        val += f" phys={physical}"
    if logical < 4:
        return CheckResult(
            "CPU", STATUS_WARNING, val,
            "< 4 cores logicos. ASR Whisper puede ser lento (no critico).",
        )
    return CheckResult("CPU", STATUS_PASS, val, "CPU disponible.")


def _check_ram() -> CheckResult:
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / 1024**3
        avail_gb = mem.available / 1024**3
        val = f"avail={avail_gb:.1f}GB / total={total_gb:.1f}GB"
        if avail_gb < 6.0:
            return CheckResult(
                "RAM", STATUS_WARNING, val,
                "< 6 GB disponibles. Puede haber OOM al cargar modelos grandes.",
            )
        return CheckResult("RAM", STATUS_PASS, val, "RAM suficiente.")
    except Exception:
        total = None
        try:
            if sys.platform.startswith("win"):
                import ctypes
                class MEMSTAT(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                    ]
                stat = MEMSTAT()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
                total = stat.ullTotalPhys / 1024**3
                avail = stat.ullAvailPhys / 1024**3
                val = f"avail={avail:.1f}GB / total={total:.1f}GB"
                if avail < 6.0:
                    return CheckResult("RAM", STATUS_WARNING, val, "< 6 GB disponibles.")
                return CheckResult("RAM", STATUS_PASS, val, "RAM suficiente.")
        except Exception:
            pass
        return CheckResult(
            "RAM", STATUS_WARNING, f"psutil={total}",
            "psutil no disponible, no se puede medir RAM exacta (no critico).",
        )


# =============================================================================
# Ejecucion principal
# =============================================================================

ALL_CHECKS_ORDERED = [
    _check_python,
    _check_torch,
    _check_cuda,
    _check_cudnn,
    _check_bin.__wrapped__ if False else lambda n="ffmpeg", f="FFmpeg": _check_bin(n, f),
    lambda: _check_bin("ffprobe", "FFprobe"),
    _check_hf_lib,
    _check_hf_token,
    _check_vram,
    _check_disk,
    _check_cpu,
    _check_ram,
]


def run_all_checks() -> tuple[list[CheckResult], int]:
    results: list[CheckResult] = []
    for fn in ALL_CHECKS_ORDERED:
        try:
            results.append(fn())
        except Exception as exc:  # pragma: no cover - defensivo
            name = getattr(fn, "__name__", "check")
            results.append(CheckResult(name, STATUS_ERROR, "exception", str(exc)))
    max_rank = max((r.rank for r in results), default=0)
    exit_code = 0 if max_rank == 0 else (1 if max_rank == 1 else 2)
    return results, exit_code


def render(results: list[CheckResult], *, exit_code: int) -> None:
    console = Console()
    table = Table(
        title="IA Doblaje Engine — Verificacion de Terceros (T02)",
        box=box.ROUNDED,
        show_lines=False,
    )
    table.add_column("CHECK", style="bold", overflow="fold", min_width=18)
    table.add_column("STATUS", style="bold", min_width=9)
    table.add_column("VALUE", overflow="fold", min_width=24)
    table.add_column("MESSAGE", overflow="fold")

    for r in results:
        color = STATUS_COLORS.get(r.status, "white")
        status_cell = f"[{color}]{r.status}[/{color}]"
        table.add_row(r.check, status_cell, r.value, r.message)

    console.print(table)

    counts: dict[str, int] = {STATUS_PASS: 0, STATUS_WARNING: 0, STATUS_ERROR: 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    summary = (
        f"[bold]Resumen:[/bold] PASS={counts[STATUS_PASS]}, "
        f"[yellow]WARNING={counts[STATUS_WARNING]}[/yellow], "
        f"[red]ERROR={counts[STATUS_ERROR]}[/red]  |  "
    )
    if exit_code == 0:
        summary += "[green]TODO OK: exit code 0[/green]"
    elif exit_code == 1:
        summary += "[yellow]WARNINGS: continuar con cuidado. exit code 1[/yellow]"
    else:
        summary += "[red]ERRORES CRITICOS: corregir antes de ejecutar pipeline. exit code 2[/red]"
    console.print(summary)


def main() -> int:
    results, exit_code = run_all_checks()
    render(results, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
