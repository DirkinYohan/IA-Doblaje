"""Deteccion de hardware + politica de VRAM (T02.4 implementacion completa).

Clases publicas:
    GPUInfo
    DeviceInfo
    DeviceDetector  (detecta CPU/CUDA/MPS con soft imports de torch)

Funciones publicas:
    validate_profile_vram_capability()  -> POLITICA CRITICA PUNTO 2
        NO SILENT DOWNGRADE POR DEFECTO.
        Solo downgrade si GPU_AUTO_DOWNGRADE_ENABLED=True explicitamente.

Soft imports:
    - torch: NO a nivel modulo. Importa try/except dentro de metodos.
    - Si torch no esta instalado: DeviceDetector retorna CPU-only info con
      torch_available=False, NO lanza excepcion.

Dependency Injection (tests):
    DeviceDetector(torch_module=fake_torch) permite mockear CUDA/MPS completamente
    sin tener PyTorch instalado ni GPU real.
"""

from __future__ import annotations

import os
import platform as _platform
import sys
from dataclasses import dataclass, field
from typing import Any

from app.core.constants import DeviceType, QualityProfile
from app.core.exceptions import ProfileDowngradeRequired


# =============================================================================
# Data structures
# =============================================================================


@dataclass(slots=True, frozen=True)
class GPUInfo:
    """Info de una GPU detectada. VRAM siempre en MB enteros (0 = desconocido)."""

    name: str
    total_vram_mb: int
    available_vram_mb: int
    used_vram_mb: int = 0
    cuda_device_index: int | None = None
    backend: str = "unknown"  # "cuda" | "mps" | "unknown"

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024.0

    @property
    def available_vram_gb(self) -> float:
        return self.available_vram_mb / 1024.0

    @property
    def used_vram_gb(self) -> float:
        return self.used_vram_mb / 1024.0


@dataclass(slots=True)
class DeviceInfo:
    """Info completa del sistema de inferencia."""

    device_type: DeviceType
    device_name: str
    cpu_cores_logical: int
    cpu_cores_physical: int | None
    ram_total_mb: int
    ram_available_mb: int
    platform_system: str
    platform_release: str
    python_version: str
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    mps_available: bool
    primary_gpu: GPUInfo | None = None
    all_gpus: list[GPUInfo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ram_total_gb(self) -> float:
        return self.ram_total_mb / 1024.0

    @property
    def ram_available_gb(self) -> float:
        return self.ram_available_mb / 1024.0


# =============================================================================
# DeviceDetector
# =============================================================================


class DeviceDetector:
    """Detecta CPU, RAM, CUDA, MPS. Soft import de torch.

    Ejemplo test sin GPU:
        fake = types.SimpleNamespace()
        fake.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda i: "Fake RTX 3060",
            get_device_properties=lambda i: types.SimpleNamespace(total_memory=12*1024**3),
            mem_get_info=lambda i: (8*1024**3, 12*1024**3),
        )
        fake.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 8900))
        fake.__version__ = "2.5.0+fake"
        dd = DeviceDetector(torch_module=fake)
    """

    def __init__(
        self,
        *,
        torch_module: Any = None,
        psutil_module: Any = None,
        allow_mps: bool = True,
        force_cpu: bool = False,
    ) -> None:
        self._torch_injected = torch_module
        self._psutil_injected = psutil_module
        self._allow_mps = allow_mps
        self._force_cpu = force_cpu

    # --- helpers internos: importadores soft ---

    def _import_torch(self) -> tuple[Any, str | None]:
        """Retorna (module_or_None, version_or_None). Nunca lanza."""
        if self._torch_injected is not None:
            v = getattr(self._torch_injected, "__version__", None)
            return self._torch_injected, v
        try:
            import torch  # noqa: F401
            return torch, getattr(torch, "__version__", None)
        except Exception:  # pragma: no cover - import real
            return None, None

    def _import_psutil(self) -> Any:
        if self._psutil_injected is not None:
            return self._psutil_injected
        try:
            import psutil
            return psutil
        except Exception:  # pragma: no cover - import real
            return None

    # --- CPU / RAM via psutil ---

    def _cpu_info(self) -> tuple[int, int | None]:
        ps = self._import_psutil()
        logical = os.cpu_count() or 1
        physical: int | None = None
        if ps is not None:
            try:
                logical = ps.cpu_count(logical=True) or logical
                physical = ps.cpu_count(logical=False)
            except Exception:
                pass
        return logical, physical

    def _ram_info(self) -> tuple[int, int]:
        ps = self._import_psutil()
        total_mb = 0
        avail_mb = 0
        if ps is not None:
            try:
                mem = ps.virtual_memory()
                total_mb = int(getattr(mem, "total", 0) / (1024 * 1024))
                avail_mb = int(getattr(mem, "available", 0) / (1024 * 1024))
            except Exception:
                pass
        return total_mb, avail_mb

    # --- CUDA via torch ---

    def _probe_cuda(self, torch: Any) -> tuple[bool, str | None, list[GPUInfo]]:
        ok = False
        cuda_version: str | None = None
        gpus: list[GPUInfo] = []
        cuda_mod = getattr(torch, "cuda", None)
        if cuda_mod is None:
            return ok, cuda_version, gpus
        try:
            ok = bool(cuda_mod.is_available())
        except Exception:
            ok = False
        if not ok:
            return ok, None, gpus
        # cuda version
        try:
            ver_fn = getattr(cuda_mod, "version", None)
            if callable(ver_fn):
                cuda_version = str(ver_fn())
        except Exception:
            cuda_version = None
        try:
            backends = getattr(torch, "backends", None)
            cudnn = getattr(backends, "cudnn", None)
            if cudnn is not None:
                v = getattr(cudnn, "version", None)
                if callable(v):
                    _ = v()
        except Exception:  # noqa: S110
            pass
        # devices
        try:
            count = int(cuda_mod.device_count() or 0)
        except Exception:
            count = 0
        for i in range(count):
            try:
                name = str(cuda_mod.get_device_name(i))
            except Exception:
                name = f"CUDA Device {i}"
            props = None
            total_bytes = 0
            try:
                props = cuda_mod.get_device_properties(i)
                total_bytes = int(getattr(props, "total_memory", 0) or 0)
            except Exception:
                total_bytes = 0
            avail_bytes = 0
            used_bytes = 0
            try:
                meminfo = cuda_mod.mem_get_info(i)
                if meminfo and len(meminfo) >= 2:
                    avail_bytes = int(meminfo[0])
                    total_meminfo = int(meminfo[1])
                    if total_bytes == 0:
                        total_bytes = total_meminfo
                    used_bytes = max(0, total_bytes - avail_bytes)
            except Exception:
                avail_bytes = max(0, total_bytes - used_bytes)
            total_mb = int(total_bytes / (1024 * 1024))
            avail_mb = int(avail_bytes / (1024 * 1024))
            used_mb = int(used_bytes / (1024 * 1024))
            gpus.append(GPUInfo(
                name=name,
                total_vram_mb=total_mb,
                available_vram_mb=avail_mb,
                used_vram_mb=used_mb,
                cuda_device_index=i,
                backend="cuda",
            ))
        return ok, cuda_version, gpus

    # --- MPS (Apple Silicon) via torch ---

    def _probe_mps(self, torch: Any) -> bool:
        if not self._allow_mps:
            return False
        try:
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None)
            if mps is None:
                return False
            fn = getattr(mps, "is_available", None)
            if callable(fn):
                return bool(fn())
        except Exception:
            pass
        return False

    # --- metodo principal ---

    def detect(self, preferred: DeviceType = DeviceType.AUTO) -> DeviceInfo:
        notes: list[str] = []
        logical, physical = self._cpu_info()
        ram_t, ram_a = self._ram_info()
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        plat_sys = _platform.system() or "Unknown"
        plat_rel = _platform.release() or "Unknown"

        torch_mod, torch_ver = self._import_torch()
        torch_ok = torch_mod is not None

        cuda_ok = False
        cuda_ver: str | None = None
        gpus: list[GPUInfo] = []
        mps_ok = False
        if torch_ok:
            cuda_ok, cuda_ver, gpus = self._probe_cuda(torch_mod)
            mps_ok = self._probe_mps(torch_mod)
        else:
            notes.append("PyTorch no disponible: deteccion CUDA/MPS omitida.")
            if not self._force_cpu:
                notes.append("Usando CPU unico dispositivo.")

        primary_gpu: GPUInfo | None = gpus[0] if gpus else None

        # Resolver device final
        if self._force_cpu:
            dt = DeviceType.CPU
        elif preferred == DeviceType.AUTO:
            if cuda_ok and primary_gpu is not None:
                dt = DeviceType.CUDA
            elif mps_ok:
                dt = DeviceType.MPS
            else:
                dt = DeviceType.CPU
        elif preferred == DeviceType.CUDA:
            if cuda_ok and primary_gpu is not None:
                dt = DeviceType.CUDA
            else:
                dt = DeviceType.CPU
                notes.append("DEVICE=cuda solicitado pero CUDA no disponible -> CPU fallback.")
        elif preferred == DeviceType.MPS:
            if mps_ok:
                dt = DeviceType.MPS
            else:
                dt = DeviceType.CPU
                notes.append("DEVICE=mps solicitado pero MPS no disponible -> CPU fallback.")
        else:
            dt = DeviceType.CPU

        device_name = "CPU"
        if dt == DeviceType.CUDA and primary_gpu:
            device_name = f"CUDA:{primary_gpu.cuda_device_index} {primary_gpu.name}"
        elif dt == DeviceType.MPS:
            device_name = "MPS (Apple Silicon)"

        return DeviceInfo(
            device_type=dt,
            device_name=device_name,
            cpu_cores_logical=logical,
            cpu_cores_physical=physical,
            ram_total_mb=ram_t,
            ram_available_mb=ram_a,
            platform_system=plat_sys,
            platform_release=plat_rel,
            python_version=py_version,
            torch_available=torch_ok,
            torch_version=torch_ver,
            cuda_available=cuda_ok,
            cuda_version=cuda_ver,
            mps_available=mps_ok,
            primary_gpu=primary_gpu,
            all_gpus=list(gpus),
            notes=notes,
        )


# =============================================================================
# Politica CRITICA PUNTO 2 — Validacion VRAM / NO SILENT DOWNGRADE
# =============================================================================


_BYTES_TO_GB_DIV = 1024.0


@dataclass(slots=True)
class VramValidationResult:
    """Retorno de validate_profile_vram_capability."""

    profile_requested: QualityProfile
    profile_applied: QualityProfile
    required_vram_gb: float
    available_vram_gb: float
    device_name: str
    device_type: DeviceType
    downgrade_applied: bool
    auto_downgrade_enabled: bool
    alternatives: list[tuple[str, float | None]]

    @property
    def alternatives_plain(self) -> list[str]:
        return [a[0] for a in self.alternatives]


def _build_alternatives(
    requested: QualityProfile,
    profile_vram_cfg: Any,
    device_can_do_accel: bool,
) -> list[tuple[str, float | None]]:
    """Alternativas validas en orden de preferencia."""
    order: list[QualityProfile] = [
        QualityProfile.QUALITY,
        QualityProfile.BALANCED,
        QualityProfile.PERFORMANCE,
    ]
    result: list[tuple[str, float | None]] = []
    for p in order:
        if p == requested:
            continue
        vram = profile_vram_cfg.required_for(p) if hasattr(profile_vram_cfg, "required_for") else None
        result.append((p.value, vram))
    # Siempre CPU sin VRAM req como ultimo recurso
    result.append(("cpu", None))
    return result


def validate_profile_vram_capability(
    *,
    requested_profile: QualityProfile | str,
    device_info: DeviceInfo,
    profile_vram_cfg: Any,
    vram_headroom_mb: int = 0,
    gpu_auto_downgrade_enabled: bool = False,
    logger: Any = None,
) -> VramValidationResult:
    """POLITICA CRITICA PUNTO 2 — NO SILENT DOWNGRADE POR DEFECTO.

    Flujo:
      1. Si device_type es CPU -> siempre OK (no hay VRAM).
      2. Calcular VRAM real usable = available_vram - headroom.
      3. Si VRAM requerido <= VRAM usable -> OK, no downgrade.
      4. Si VRAM insuficiente:
         a. gpu_auto_downgrade_enabled=False -> RAISE ProfileDowngradeRequired 412
         b. gpu_auto_downgrade_enabled=True  -> pick primer perfil que cabe
                                              -> WARNING en log
                                              -> downgrade_applied=True
    """
    req_p = QualityProfile(requested_profile) if isinstance(requested_profile, str) else requested_profile
    required = float(profile_vram_cfg.required_for(req_p)) if hasattr(profile_vram_cfg, "required_for") else 0.0
    alternatives = _build_alternatives(req_p, profile_vram_cfg, device_info.device_type.is_accelerator)

    # Caso simple: CPU no requiere VRAM
    if device_info.device_type == DeviceType.CPU or device_info.primary_gpu is None:
        return VramValidationResult(
            profile_requested=req_p,
            profile_applied=req_p,
            required_vram_gb=required,
            available_vram_gb=0.0,
            device_name=device_info.device_name,
            device_type=device_info.device_type,
            downgrade_applied=False,
            auto_downgrade_enabled=gpu_auto_downgrade_enabled,
            alternatives=alternatives,
        )

    gpu = device_info.primary_gpu
    headroom_gb = max(0.0, float(vram_headroom_mb) / 1024.0)
    available_after_headroom = max(0.0, float(gpu.available_vram_gb) - headroom_gb)
    device_name = gpu.name

    # Caso 1: cabe sin downgrade
    if required <= available_after_headroom + 1e-6:
        return VramValidationResult(
            profile_requested=req_p,
            profile_applied=req_p,
            required_vram_gb=required,
            available_vram_gb=available_after_headroom,
            device_name=device_name,
            device_type=device_info.device_type,
            downgrade_applied=False,
            auto_downgrade_enabled=gpu_auto_downgrade_enabled,
            alternatives=alternatives,
        )

    # Caso 2: NO CABE. Aplicar politica PUNTO 2
    alternatives_plain = [a[0] for a in alternatives]

    if not gpu_auto_downgrade_enabled:
        # POLITICA CRITICA: raise ProfileDowngradeRequired 412
        raise ProfileDowngradeRequired(
            requested_profile=req_p.value,
            required_vram_gb=required,
            available_vram_gb=available_after_headroom,
            alternatives=alternatives_plain,
            message=(
                f"Perfil '{req_p.value}' requiere {required:.1f} GB VRAM "
                f"(headroom {vram_headroom_mb} MB). GPU '{device_name}' "
                f"disponible: {available_after_headroom:.1f} GB. "
                f"Habilita GPU_AUTO_DOWNGRADE_ENABLED=true para downgrade automatico."
            ),
        )

    # gpu_auto_downgrade_enabled=True -> buscar perfil que sí quepa (menor o igual precision, NUNCA upgrade)
    # Perfiles ordenados por VRAM REQUERIDA ASCENDENTE: PERFORMANCE(3G) < BALANCED(5G) < QUALITY(8G)
    asc_order: list[QualityProfile] = [
        QualityProfile.PERFORMANCE,
        QualityProfile.BALANCED,
        QualityProfile.QUALITY,
    ]
    _rank: dict[QualityProfile, int] = {p: i for i, p in enumerate(asc_order)}
    # Solo perfiles con precision <= solicitada (rank[perfil] <= rank[req_p])
    # Ej: req=QUALITY(rank=2) -> [PERF, BAL, QUAL] todos permitidos (downgrade a cualquiera)
    # Ej: req=BALANCED(rank=1) -> [PERF, BAL] solo (nunca upgrade a QUALITY)
    # Ej: req=PERFORMANCE(rank=0) -> [PERF] solo
    allowed_asc: list[QualityProfile] = [p for p in asc_order if _rank[p] <= _rank[req_p]]

    picked: QualityProfile = allowed_asc[0]  # default el mas pobre PERO que siempre cabe (PERFORMANCE si permitido)
    # Iterar DESCENDENTE (mayor calidad primero) para coger el MEJOR perfil que SI cabe
    for cand in reversed(allowed_asc):
        cand_req = float(profile_vram_cfg.required_for(cand)) if hasattr(profile_vram_cfg, "required_for") else 0.0
        if cand_req <= available_after_headroom + 1e-6:
            picked = cand
            break
    # Requerido del perfil realmente aplicado
    applied_required = float(profile_vram_cfg.required_for(picked)) if hasattr(profile_vram_cfg, "required_for") else 0.0

    if logger is not None:
        warn = getattr(logger, "warning", None)
        if callable(warn):
            warn(
                "profile_downgrade_applied",
                profile_requested=req_p.value,
                profile_applied=picked.value,
                required_vram_gb=required,
                applied_required_vram_gb=applied_required,
                available_vram_gb=available_after_headroom,
                device_name=device_name,
                vram_headroom_mb=vram_headroom_mb,
                auto_downgrade_enabled=True,
            )

    return VramValidationResult(
        profile_requested=req_p,
        profile_applied=picked,
        required_vram_gb=required,
        available_vram_gb=available_after_headroom,
        device_name=device_name,
        device_type=device_info.device_type,
        downgrade_applied=(picked != req_p),
        auto_downgrade_enabled=True,
        alternatives=alternatives,
    )


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "GPUInfo",
    "DeviceInfo",
    "DeviceDetector",
    "VramValidationResult",
    "validate_profile_vram_capability",
]
