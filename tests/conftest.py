"""Fixtures globales de pytest.

T01: Configuracion basica y paths para tests.
No hay fixtures de modelos IA, audio sintetico, etc. esos se agregan en T04+.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# -----------------------------------------------------------------------------
# PATHS GLOBALES
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TESTS_DIR: Path = PROJECT_ROOT / "tests"
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"
INTEGRATION_FIXTURES_DIR: Path = TESTS_DIR / "integration" / "fixtures_clips"
DATA_INPUT_DIR: Path = PROJECT_ROOT / "data" / "input"
DATA_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "output"
DATA_TEMP_DIR: Path = PROJECT_ROOT / "data" / "temporary"
MODELS_DIR: Path = PROJECT_ROOT / "models"


# -----------------------------------------------------------------------------
# PYTEST HOOKS
# -----------------------------------------------------------------------------
def pytest_configure(config: pytest.Config) -> None:
    """Registra markers y paths."""
    # Asegurar que el project root este en sys.path para imports editable
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Crear directorios de trabajo si no existen
    for directory in (DATA_INPUT_DIR, DATA_OUTPUT_DIR, DATA_TEMP_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # Guardar paths para uso en tests
    config.stash["project_root"] = PROJECT_ROOT
    config.stash["data_dirs"] = {
        "input": DATA_INPUT_DIR,
        "output": DATA_OUTPUT_DIR,
        "temp": DATA_TEMP_DIR,
    }


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Ordena tests: unit first, integration last."""
    _ORDER = {
        "tests/domain": 0,
        "tests/application": 1,
        "tests/infrastructure": 2,
        "tests/integration": 99,
    }

    def _sort_key(item: pytest.Item) -> tuple[int, str]:
        path_str = str(Path(item.fspath).relative_to(PROJECT_ROOT)).replace("\\", "/")
        for prefix, order in _ORDER.items():
            if path_str.startswith(prefix):
                return (order, path_str)
        return (50, path_str)

    items.sort(key=_sort_key)


# -----------------------------------------------------------------------------
# FIXTURES: PATHS
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return TESTS_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def data_input_dir() -> Path:
    return DATA_INPUT_DIR


@pytest.fixture(scope="session")
def data_output_dir() -> Path:
    return DATA_OUTPUT_DIR


@pytest.fixture()
def isolated_temp_dir(tmp_path: Path) -> Path:
    """Directorio temporal aislado por test.

    Preferible sobre DATA_TEMP_DIR para evitar colisiones.
    """
    d = tmp_path / f"job_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def monkeypatch_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Helper para limpiar variables de entorno sensibles en tests."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    # Asegurar no se usan GPUs reales en unit tests
    monkeypatch.setenv("FORCE_CPU", "true")
    return monkeypatch
