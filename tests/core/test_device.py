"""Tests unitarios T02 para app.core.device.

NO requiere GPU NVIDIA / CUDA / PyTorch instalado.
Todos los tests usan:
  - DeviceDetector(torch_module=fake_module) para mockear CUDA/MPS
  - DeviceDetector(psutil_module=fake_psutil) para mock RAM/CPU
  - Monkeypatch sys.modules["torch"] = None para simular torch no instalado.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from app.core.constants import DeviceType, QualityProfile
from app.core.device import (
    GPUInfo,
    DeviceDetector,
    DeviceInfo,
    VramValidationResult,
    validate_profile_vram_capability,
)
from app.core.exceptions import ProfileDowngradeRequired
from app.core.config import ProfileVramConfig


# ---------------------------------------------------------------------------
# Helpers: fake torch module + fake psutil
# ---------------------------------------------------------------------------


def make_fake_torch(
    *,
    cuda_ok: bool = True,
    device_count: int = 1,
    device_name: str = "Fake RTX 3060",
    total_vram_bytes: int = 12 * 1024 ** 3,  # 12 GB
    available_vram_bytes: int = 8 * 1024 ** 3,  # 8 GB libres
    cuda_version: str = "12.1",
    cudnn_version_int: int = 8900,
    mps_ok: bool = False,
    torch_version: str = "2.5.0+cpu",
) -> Any:
    """Fabrica un modulo torch falso para tests via DI."""
    fake_torch = types.SimpleNamespace()
    fake_torch.__version__ = torch_version
    fake_cuda = types.SimpleNamespace()
    fake_cuda.is_available = lambda: cuda_ok
    fake_cuda.device_count = lambda: device_count if cuda_ok else 0
    fake_cuda.get_device_name = lambda i: device_name
    fake_cuda.get_device_properties = lambda i: types.SimpleNamespace(
        total_memory=total_vram_bytes
    )
    fake_cuda.mem_get_info = lambda i: (available_vram_bytes, total_vram_bytes)
    fake_cuda.version = lambda: cuda_version if cuda_ok else None
    fake_torch.cuda = fake_cuda
    fake_backends = types.SimpleNamespace()
    fake_cudnn = types.SimpleNamespace()
    fake_cudnn.is_available = lambda: cuda_ok
    fake_cudnn.version = lambda: cudnn_version_int
    fake_backends.cudnn = fake_cudnn
    fake_mps = types.SimpleNamespace()
    fake_mps.is_available = lambda: mps_ok
    fake_backends.mps = fake_mps
    fake_torch.backends = fake_backends
    return fake_torch


def make_fake_psutil(*, total_ram_bytes: int, avail_ram_bytes: int, cpu_logical: int = 8, cpu_physical: int = 4) -> Any:
    fake = types.SimpleNamespace()
    vm = types.SimpleNamespace(total=total_ram_bytes, available=avail_ram_bytes)
    fake.virtual_memory = lambda: vm
    fake.cpu_count = lambda logical=True: cpu_logical if logical else cpu_physical
    return fake


def _profile_vram_cfg() -> ProfileVramConfig:
    return ProfileVramConfig(quality_required_gb=8.0, balanced_required_gb=5.0, performance_required_gb=3.0)


# ---------------------------------------------------------------------------
# Tests: CPU-only / torch ausente
# ---------------------------------------------------------------------------


def test_device_detector_torch_missing_via_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simular que torch no está instalado (no importar)."""
    # Forzamos que DeviceDetector._import_torch falle sin sys.modules hack:
    # pasamos torch_module explícitamente = None (no significa None import,
    # significa que el DI no se usa; luego con monkeypatch sys.modules["torch"]
    # lo rompemos indirectamente)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # Inyectamos tambien un sys.modules["torch"] = sentinel que al intentar
    # getattr __version__ falle pero sea truthey? No; mejor: el Detector solo
    # intenta import torch via _import_torch. Con psutil inyectado, solo importa
    # si no tiene _torch_injected. Pasamos psutil y no torch para que vaya por
    # el camino try/except ImportError.
    dd = DeviceDetector(psutil_module=make_fake_psutil(total_ram_bytes=16 * 1024**3, avail_ram_bytes=8 * 1024**3))
    info = dd.detect(DeviceType.CPU)
    assert info.device_type is DeviceType.CPU
    # CPU info siempre presente
    assert info.cpu_cores_logical > 0
    # Como no tenemos GPU -> primary_gpu None
    assert info.primary_gpu is None
    assert len(info.all_gpus) == 0


def test_device_detector_cpu_only_detected() -> None:
    """Pasar force_cpu=True y CUDA fake para garantizar que devuelve CPU."""
    fake_torch = make_fake_torch(cuda_ok=True, device_name="RTX 4090", available_vram_bytes=20 * 1024**3)
    dd = DeviceDetector(torch_module=fake_torch, force_cpu=True, psutil_module=make_fake_psutil(total_ram_bytes=16 * 1024**3, avail_ram_bytes=8 * 1024**3))
    info = dd.detect(preferred=DeviceType.CUDA)
    assert info.device_type is DeviceType.CPU
    assert info.device_name == "CPU"
    # Aun asi CUDA se detecta en flags de capacidad
    assert info.cuda_available is True
    assert len(info.all_gpus) == 1


# ---------------------------------------------------------------------------
# Tests: CUDA disponible via mock (DI)
# ---------------------------------------------------------------------------


def test_cuda_available_mock_fake_gpu() -> None:
    fake_torch = make_fake_torch(
        cuda_ok=True,
        device_count=1,
        device_name="Mock RTX 6000",
        total_vram_bytes=24 * 1024**3,
        available_vram_bytes=22 * 1024**3,
        cuda_version="12.4",
    )
    dd = DeviceDetector(torch_module=fake_torch, psutil_module=make_fake_psutil(total_ram_bytes=32 * 1024**3, avail_ram_bytes=20 * 1024**3))
    info = dd.detect(DeviceType.AUTO)
    assert info.device_type is DeviceType.CUDA
    assert info.cuda_available is True
    assert info.cuda_version == "12.4"
    assert info.torch_available is True
    gpu = info.primary_gpu
    assert gpu is not None
    assert gpu.name == "Mock RTX 6000"
    assert gpu.backend == "cuda"
    assert gpu.total_vram_gb == pytest.approx(24.0, abs=0.01)
    assert gpu.available_vram_gb == pytest.approx(22.0, abs=0.01)
    assert info.device_name.startswith("CUDA:")


def test_mps_available_mock() -> None:
    fake_torch = make_fake_torch(cuda_ok=False, mps_ok=True, torch_version="2.5.0")
    dd = DeviceDetector(torch_module=fake_torch, allow_mps=True)
    info = dd.detect(preferred=DeviceType.AUTO)
    assert info.device_type is DeviceType.MPS
    assert info.mps_available is True
    assert "MPS" in info.device_name


def test_device_cuda_requested_but_unavailable_fallback_cpu() -> None:
    fake_torch = make_fake_torch(cuda_ok=False)
    dd = DeviceDetector(torch_module=fake_torch, allow_mps=False)
    info = dd.detect(preferred=DeviceType.CUDA)
    assert info.device_type is DeviceType.CPU
    assert any("CUDA no disponible" in n for n in info.notes)


# ---------------------------------------------------------------------------
# Tests: validate_profile_vram_capability
# ---------------------------------------------------------------------------


def _make_device_info_with_gpu(*, available_vram_gb: float, total_vram_gb: float = 12.0, name: str = "Test GPU") -> DeviceInfo:
    avail_mb = int(available_vram_gb * 1024)
    total_mb = int(total_vram_gb * 1024)
    used_mb = total_mb - avail_mb
    gpu = GPUInfo(name=name, total_vram_mb=total_mb, available_vram_mb=avail_mb, used_vram_mb=used_mb, backend="cuda", cuda_device_index=0)
    return DeviceInfo(
        device_type=DeviceType.CUDA,
        device_name=f"CUDA:0 {name}",
        cpu_cores_logical=8, cpu_cores_physical=4,
        ram_total_mb=16000, ram_available_mb=8000,
        platform_system="Linux", platform_release="5.10",
        python_version="3.12.10",
        torch_available=True, torch_version="2.5.0",
        cuda_available=True, cuda_version="12.1", mps_available=False,
        primary_gpu=gpu, all_gpus=[gpu], notes=[],
    )


# ----- VRAM suficiente (sin downgrade) -----


def test_vram_sufficient_quality_10gb() -> None:
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=10.0)
    r: VramValidationResult = validate_profile_vram_capability(
        requested_profile=QualityProfile.QUALITY,
        device_info=dev,
        profile_vram_cfg=cfg,
        vram_headroom_mb=0,
        gpu_auto_downgrade_enabled=False,
    )
    assert r.profile_requested is QualityProfile.QUALITY
    assert r.profile_applied is QualityProfile.QUALITY
    assert r.downgrade_applied is False
    assert r.required_vram_gb == 8.0


def test_vram_sufficient_balanced_6gb_headroom() -> None:
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=6.0)  # 6 - 0.8 headroom = 5.2 usable >= 5 BALANCED
    r = validate_profile_vram_capability(
        requested_profile=QualityProfile.BALANCED,
        device_info=dev,
        profile_vram_cfg=cfg,
        vram_headroom_mb=800,
        gpu_auto_downgrade_enabled=False,
    )
    assert r.profile_applied is QualityProfile.BALANCED
    assert r.downgrade_applied is False


def test_vram_sufficient_performance_4gb() -> None:
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=4.0)
    r = validate_profile_vram_capability(
        requested_profile=QualityProfile.PERFORMANCE,
        device_info=dev,
        profile_vram_cfg=cfg,
        gpu_auto_downgrade_enabled=False,
    )
    assert r.profile_applied is QualityProfile.PERFORMANCE
    assert r.downgrade_applied is False


# ----- VRAM insuficiente con auto_downgrade_enabled=False -> raise ProfileDowngradeRequired -----


def test_quality_6gb_requires_raise_profile_downgrade_required() -> None:
    """Caso exacto del spec: QUALITY requiere 8GB, hay 6GB -> raise 412."""
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=6.0)
    with pytest.raises(ProfileDowngradeRequired) as excinfo:
        validate_profile_vram_capability(
            requested_profile=QualityProfile.QUALITY,
            device_info=dev,
            profile_vram_cfg=cfg,
            vram_headroom_mb=0,
            gpu_auto_downgrade_enabled=False,
        )
    err = excinfo.value
    assert err.status_code == 412
    assert err.requested_profile == "quality"
    assert err.required_vram_gb == 8.0
    assert 5.9 < err.available_vram_gb < 6.1
    # Alternativas deben incluir balanced, performance, cpu
    alts = set(a.lower() for a in err.alternatives)
    assert "balanced" in alts
    assert "performance" in alts
    assert "cpu" in alts


def test_balanced_3gb_requires_downgrade_raises() -> None:
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=3.0)  # BALANCED pide 5 GB
    with pytest.raises(ProfileDowngradeRequired) as excinfo:
        validate_profile_vram_capability(
            requested_profile=QualityProfile.BALANCED,
            device_info=dev,
            profile_vram_cfg=cfg,
            gpu_auto_downgrade_enabled=False,
        )
    assert excinfo.value.status_code == 412
    assert excinfo.value.requested_profile == "balanced"


# ----- Alternativas correctas por perfil -----


def test_alternatives_for_quality_contains_only_lower_profiles_and_cpu() -> None:
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=1.0)  # insuficiente incluso performance
    with pytest.raises(ProfileDowngradeRequired) as excinfo:
        validate_profile_vram_capability(
            requested_profile=QualityProfile.QUALITY,
            device_info=dev, profile_vram_cfg=cfg,
            gpu_auto_downgrade_enabled=False,
        )
    err = excinfo.value
    # Alternativas NO deben incluir quality (perfil solicitado)
    assert "quality" not in [a.lower() for a in err.alternatives]
    # Deben incluir balanced, performance, cpu EXACTAMENTE + orden
    assert len(err.alternatives) == 3


# ----- auto downgrade disabled ya se probó arriba; ahora auto_downgrade enabled -----


def test_auto_downgrade_enabled_quality_6gb_picks_balanced_or_performance() -> None:
    """QUALITY + 6GB disponible + AUTO-DOWNGRADE=true: debe escoger BALANCED (5GB cabe)."""
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=6.0)
    warnings_collector: list[dict[str, Any]] = []

    class FakeLog:
        @staticmethod
        def warning(event: str, **kw: Any) -> None:  # noqa: D401 - test fake
            warnings_collector.append({"event": event, **kw})

    r = validate_profile_vram_capability(
        requested_profile=QualityProfile.QUALITY,
        device_info=dev,
        profile_vram_cfg=cfg,
        gpu_auto_downgrade_enabled=True,
        logger=FakeLog,
    )
    # 6 GB available: BALANCED req=5 <= 6 -> debe escoger BALANCED
    assert r.profile_applied is QualityProfile.BALANCED
    assert r.profile_requested is QualityProfile.QUALITY
    assert r.downgrade_applied is True
    assert r.auto_downgrade_enabled is True
    # Logger warning invocado con trazabilidad
    assert any("downgrade_applied" in str(w) for w in warnings_collector)
    # Flags para metrics: original y applied presentes
    assert any(w.get("profile_requested") == "quality" for w in warnings_collector)
    assert any(w.get("profile_applied") == "balanced" for w in warnings_collector)


def test_auto_downgrade_enabled_even_performance_doesnt_fit() -> None:
    """Si ni performance cabe -> escoge performance igual con flag aplicado."""
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=1.0)  # ni 3GB performance
    r = validate_profile_vram_capability(
        requested_profile=QualityProfile.QUALITY,
        device_info=dev,
        profile_vram_cfg=cfg,
        gpu_auto_downgrade_enabled=True,
    )
    # Aun asi escoge el menor (performance) con downgrade_applied=True
    assert r.profile_applied is QualityProfile.PERFORMANCE
    assert r.downgrade_applied is True
    # alternatives incluye CPU como ultimo recurso
    assert r.alternatives_plain[-1] == "cpu"


def test_cpu_device_always_passes_no_downgrade() -> None:
    cfg = _profile_vram_cfg()
    dev = DeviceInfo(
        device_type=DeviceType.CPU, device_name="CPU",
        cpu_cores_logical=4, cpu_cores_physical=2,
        ram_total_mb=8000, ram_available_mb=4000,
        platform_system="Linux", platform_release="5.0",
        python_version="3.12.10",
        torch_available=False, torch_version=None,
        cuda_available=False, cuda_version=None, mps_available=False,
        primary_gpu=None, all_gpus=[], notes=[],
    )
    # QUALITY pide 8GB pero en CPU no importa VRAM
    r = validate_profile_vram_capability(
        requested_profile=QualityProfile.QUALITY,
        device_info=dev,
        profile_vram_cfg=cfg,
        gpu_auto_downgrade_enabled=False,
    )
    assert r.profile_applied is QualityProfile.QUALITY
    assert r.downgrade_applied is False


# ---------------------------------------------------------------------------
# Tests: GPUInfo properties math
# ---------------------------------------------------------------------------


def test_gpu_info_gb_properties() -> None:
    g = GPUInfo(name="X", total_vram_mb=8192, available_vram_mb=6144, used_vram_mb=2048)
    assert g.total_vram_gb == 8.0
    assert g.available_vram_gb == 6.0
    assert g.used_vram_gb == 2.0


# ---------------------------------------------------------------------------
# Tests: VramValidationResult alternatives_plain
# ---------------------------------------------------------------------------


def test_alternatives_plain_helper() -> None:
    cfg = _profile_vram_cfg()
    dev = _make_device_info_with_gpu(available_vram_gb=10.0)
    r = validate_profile_vram_capability(
        requested_profile=QualityProfile.PERFORMANCE,
        device_info=dev,
        profile_vram_cfg=cfg,
    )
    # alternatives includes quality/balanced (excluye performance) + cpu
    assert "quality" in r.alternatives_plain
    assert "balanced" in r.alternatives_plain
    assert "cpu" in r.alternatives_plain
