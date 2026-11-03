from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
TESTS_DIR: Path = PROJECT_ROOT / "tests"
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"
INTEGRATION_FIXTURES_DIR: Path = TESTS_DIR / "integration" / "fixtures_clips"
DATA_INPUT_DIR: Path = PROJECT_ROOT / "data" / "input"
DATA_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "output"
DATA_TEMP_DIR: Path = PROJECT_ROOT / "data" / "temporary"
MODELS_DIR: Path = PROJECT_ROOT / "models"


def pytest_configure(config: pytest.Config) -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    os.environ["IA_DB_BACKEND"] = "sqlite"
    os.environ.pop("IA_DATABASE_URL", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ["IA_STORAGE_BACKEND"] = "local"
    os.environ["BREVO_API_KEY"] = ""
    os.environ["IA_PRELOAD_MODELS"] = "0"
    os.environ["IA_LIVE_PRELOAD"] = "0"

    sandbox = PROJECT_ROOT / "tests" / ".sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox, ignore_errors=True)
    os.environ["DATA_INPUT_DIR"] = str(sandbox / "input")
    os.environ["DATA_OUTPUT_DIR"] = str(sandbox / "output")
    os.environ["DATA_TEMP_DIR"] = str(sandbox / "temporary")

    for directory in (DATA_INPUT_DIR, DATA_OUTPUT_DIR, DATA_TEMP_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    config.stash["project_root"] = PROJECT_ROOT
    config.stash["data_dirs"] = {
        "input": DATA_INPUT_DIR,
        "output": DATA_OUTPUT_DIR,
        "temp": DATA_TEMP_DIR,
    }


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
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
    d = tmp_path / f"job_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def monkeypatch_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FORCE_CPU", "true")
    return monkeypatch
