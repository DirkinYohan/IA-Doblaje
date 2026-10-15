"""Tests unitarios T02 para app.core.config (AppSettings + get_settings).

Sin dependencias de GPU / CUDA / PyTorch / HF_TOKEN real.
Todos los tests usan monkeypatch para simular env vars.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import PROJECT_ROOT  # noqa: E402 - import after path setup

from app.core.config import (
    AppSettings,
    GPUConfig,
    PathsConfig,
    ProfileVramConfig,
    SafetyConfig,
    clear_settings_cache,
    get_settings,
)
from app.core.constants import DeviceType, QualityProfile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autouse: limpia cache + resetea variables planas antes de cada test."""
    clear_settings_cache()
    # Quitar todas las flat vars que pudieran existir en el entorno real
    flat_keys = [
        "PROFILE", "DEVICE", "FORCE_CPU",
        "GPU_AUTO_DOWNGRADE_ENABLED", "VRAM_HEADROOM_MB",
        "MAX_MEDIA_SIZE_GB", "HF_TOKEN",
        "DATA_INPUT_DIR", "DATA_OUTPUT_DIR", "DATA_TEMP_DIR", "MODELS_CACHE_DIR",
        "APP_LOG_LEVEL", "APP_LOG_FORMAT", "APP_ENV",
    ]
    for k in flat_keys:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Tests: defaults
# ---------------------------------------------------------------------------


def test_defaults_app() -> None:
    s = AppSettings()
    assert s.processing.profile is QualityProfile.BALANCED
    assert s.processing.device is DeviceType.AUTO
    # Regla CRITICA: GPU_AUTO_DOWNGRADE_ENABLED = False por defecto
    assert s.gpu.gpu_auto_downgrade_enabled is False
    assert isinstance(s.gpu.vram_headroom_mb, int)
    assert s.gpu.vram_headroom_mb > 0
    assert s.safety.max_media_size_gb > 0.1


def test_defaults_profile_vram() -> None:
    s = AppSettings()
    pv = s.profile_vram
    assert pv.quality_required_gb == 8.0
    assert pv.balanced_required_gb == 5.0
    assert pv.performance_required_gb == 3.0
    assert pv.required_for(QualityProfile.QUALITY) == 8.0
    assert pv.required_for(QualityProfile.BALANCED) == 5.0
    assert pv.required_for(QualityProfile.PERFORMANCE) == 3.0
    # Strings tambien funcionan
    assert pv.required_for("quality") == 8.0


# ---------------------------------------------------------------------------
# Tests: env loading flat -> nested (PROFILE, DEVICE, etc.)
# ---------------------------------------------------------------------------


def test_env_flat_profile_and_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE", "performance")
    monkeypatch.setenv("DEVICE", "cuda")
    clear_settings_cache()
    s = AppSettings()
    assert s.processing.profile is QualityProfile.PERFORMANCE
    assert s.processing.device is DeviceType.CUDA


def test_env_flat_gpu_downgrade_and_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_AUTO_DOWNGRADE_ENABLED", "true")
    monkeypatch.setenv("VRAM_HEADROOM_MB", "400")
    clear_settings_cache()
    s = AppSettings()
    assert s.gpu.gpu_auto_downgrade_enabled is True
    assert s.gpu.vram_headroom_mb == 400


# ---------------------------------------------------------------------------
# Tests: PROFILE valido / invalido
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ok_value", ["quality", "balanced", "performance", "Quality"])
def test_profile_valid(ok_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE", ok_value)
    clear_settings_cache()
    s = AppSettings()
    assert isinstance(s.processing.profile, QualityProfile)


@pytest.mark.parametrize("bad_value", ["insane", "ultra", "QUALITEE"])
def test_profile_invalid(bad_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE", bad_value)
    clear_settings_cache()
    with pytest.raises(ValidationError):
        AppSettings()


# ---------------------------------------------------------------------------
# Tests: DEVICE valido / invalido
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ok_value", ["auto", "cpu", "cuda", "mps", "AUTO"])
def test_device_valid(ok_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", ok_value)
    clear_settings_cache()
    s = AppSettings()
    assert isinstance(s.processing.device, DeviceType)


@pytest.mark.parametrize("bad_value", ["tpu", "vulkan", "cuda2"])
def test_device_invalid(bad_value: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", bad_value)
    clear_settings_cache()
    with pytest.raises(ValidationError):
        AppSettings()


# ---------------------------------------------------------------------------
# Tests: GPU_AUTO_DOWNGRADE=false por defecto (doble comprobacion)
# ---------------------------------------------------------------------------


def test_gpu_auto_downgrade_default_false_no_env() -> None:
    # Asegurarse que no existe env var
    import os
    os.environ.pop("GPU_AUTO_DOWNGRADE_ENABLED", None)
    clear_settings_cache()
    gpu = GPUConfig()
    assert gpu.gpu_auto_downgrade_enabled is False


# ---------------------------------------------------------------------------
# Tests: path relativo -> absoluto desde PROJECT_ROOT
# ---------------------------------------------------------------------------


def test_paths_relative_resolved_to_project_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_INPUT_DIR", "data/in_alt")
    monkeypatch.setenv("DATA_OUTPUT_DIR", "out_alt")
    clear_settings_cache()
    s = AppSettings()
    for attr in ("data_input_dir", "data_output_dir", "data_temp_dir", "models_cache_dir"):
        p = getattr(s.paths, attr)
        assert p.is_absolute()
        # El path resuelto debe estar dentro de PROJECT_ROOT
        assert p.is_relative_to(PROJECT_ROOT) or str(PROJECT_ROOT) in str(p), \
            f"Path {p} no esta dentro de PROJECT_ROOT {PROJECT_ROOT}"


def test_paths_config_resolver_validator() -> None:
    # Un path absoluto NO debe modificarse (mantiene C:\\x o /abs)
    abs_val = Path.cwd().resolve()
    pc = PathsConfig(data_input_dir=str(abs_val))
    assert pc.data_input_dir == abs_val


# ---------------------------------------------------------------------------
# Tests: MAX_MEDIA_SIZE_GB invalido (<=0.1)
# ---------------------------------------------------------------------------


def test_max_media_size_gb_invalid_low(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MEDIA_SIZE_GB", "0.01")
    clear_settings_cache()
    with pytest.raises(ValidationError):
        AppSettings()


def test_max_media_size_gb_invalid_non_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_MEDIA_SIZE_GB", "diez")
    clear_settings_cache()
    with pytest.raises(ValidationError):
        AppSettings()


def test_safety_config_default_valid() -> None:
    s = SafetyConfig()
    assert isinstance(s.allowed_extensions_set, set)
    assert "mp4" in s.allowed_extensions_set


# ---------------------------------------------------------------------------
# Tests: HF_TOKEN como SecretStr (nunca imprimir el valor)
# ---------------------------------------------------------------------------


def test_hf_token_is_secretstr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_XYZ_SUPER_SECRETO_12345678")
    clear_settings_cache()
    s = AppSettings()
    assert isinstance(s.hf.hf_token, SecretStr)
    # str() del token NO debe mostrar el valor real
    shown = str(s.hf.hf_token)
    assert "hf_XYZ_SUPER_SECRETO" not in shown
    # repr() del token tampoco
    assert "hf_XYZ_SUPER_SECRETO" not in repr(s.hf.hf_token)


# Sentinel usado por test de dump_safe
_SENTINEL_EXPECTED_ANY = "***HIDDEN_SECRET_STR***"


def test_hf_token_dump_safe_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_VERY_SECRET_999")
    clear_settings_cache()
    s = AppSettings()
    dumped = s.dump_safe()
    hf = dumped.get("hf") or {}
    val = hf.get("hf_token", "")
    assert "hf_VERY_SECRET_999" not in str(val)
    assert "HIDDEN_SECRET_STR" in str(val) or str(val) == _SENTINEL_EXPECTED_ANY


def test_hf_token_empty_is_allowed() -> None:
    # Token vacio debe ser permitido y dump_safe debe devolver ""
    s = AppSettings()
    dumped = s.dump_safe()
    hf = dumped.get("hf") or {}
    val = hf.get("hf_token", "??")
    assert val == ""


# ---------------------------------------------------------------------------
# Tests: get_settings() singleton + clear_settings_cache()
# ---------------------------------------------------------------------------


def test_get_settings_singleton_same(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_clear_settings_cache_refreshes_values(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    monkeypatch.setenv("PROFILE", "quality")
    s1 = get_settings()
    assert s1.processing.profile is QualityProfile.QUALITY
    # Cambiar env sin clear -> sigue siendo cacheado (aunque get_settings lee AppSettings via cached func)
    monkeypatch.setenv("PROFILE", "performance")
    s2 = get_settings()
    # s2 debe ser igual que s1 porque esta cacheado (mismo objeto)
    assert s2.processing.profile is QualityProfile.QUALITY
    # Ahora limpiar cache, debe cambiar
    clear_settings_cache()
    s3 = get_settings()
    assert s3.processing.profile is QualityProfile.PERFORMANCE


# ---------------------------------------------------------------------------
# Tests: ProfileVramConfig modelo directo
# ---------------------------------------------------------------------------


def test_profilevramconfig_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ProfileVramConfig(quality_required_gb=-1.0)
