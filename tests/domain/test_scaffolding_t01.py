from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest


_PACKAGE_PATHS: list[tuple[str, str]] = [
    ("app", "app"),
    ("app.core", "app/core"),
    ("app.core.constants", "app/core/constants.py"),
    ("app.core.exceptions", "app/core/exceptions.py"),
    ("app.domain", "app/domain"),
    ("app.domain.entities", "app/domain/entities"),
    ("app.domain.value_objects", "app/domain/value_objects"),
    ("app.domain.interfaces", "app/domain/interfaces"),
    ("app.application", "app/application"),
    ("app.application.use_cases", "app/application/use_cases"),
    ("app.application.services", "app/application/services"),
    ("app.application.services.quality", "app/application/services/quality"),
    ("app.application.pipeline", "app/application/pipeline"),
    ("app.infrastructure", "app/infrastructure"),
    ("app.infrastructure.audio", "app/infrastructure/audio"),
    ("app.infrastructure.asr", "app/infrastructure/asr"),
    ("app.infrastructure.diarization", "app/infrastructure/diarization"),
    ("app.infrastructure.language", "app/infrastructure/language"),
    ("app.infrastructure.storage", "app/infrastructure/storage"),
    ("app.presentation", "app/presentation"),
    ("app.presentation.cli", "app/presentation/cli"),
    ("app.presentation.cli.main_group", "app/presentation/cli/main_group.py"),
]


@pytest.mark.parametrize(("module_name", "_"), _PACKAGE_PATHS, ids=[m for m, _ in _PACKAGE_PATHS])
def test_module_import_without_circular_dependency(module_name: str, _: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Fallo al importar {module_name}: {type(exc).__name__}: {exc}")


def test_app_metadata_exports() -> None:
    import app

    assert isinstance(app.__version__, str) and app.__version__ != ""
    assert app.__version__.count(".") == 2
    assert isinstance(app.__title__, str) and len(app.__title__) > 3
    assert "Fase 1" in app.__phase__
    assert app.__status__ == "in-development"


class TestCoreConstants:
    @staticmethod
    def test_pipeline_steps_15_exact() -> None:
        from app.core.constants import PipelineStep

        values = sorted(step.value for step in PipelineStep)
        assert values == list(range(1, 16)), (
            f"PipelineStep debe tener 15 pasos 1..15. Valores: {values}"
        )

    @staticmethod
    def test_pipeline_step_display_names_unique() -> None:
        from app.core.constants import PipelineStep

        names = [step.display_name for step in PipelineStep]
        assert len(names) == len(set(names)), "Display names duplicados en PipelineStep"

    @staticmethod
    def test_quality_profiles_three_levels() -> None:
        from app.core.constants import QualityProfile

        profile_names = {p.value for p in QualityProfile}
        assert profile_names == {"quality", "balanced", "performance"}

    @staticmethod
    def test_supported_media_formats_whitelist() -> None:
        from app.core.constants import (
            SUPPORTED_MEDIA_EXTENSIONS,
            SUPPORTED_MEDIA_FORMATS,
            MediaFormat,
        )

        assert MediaFormat.MP4 in SUPPORTED_MEDIA_FORMATS
        assert MediaFormat.WAV in SUPPORTED_MEDIA_FORMATS
        assert ".mp4" in SUPPORTED_MEDIA_EXTENSIONS
        assert ".wav" in SUPPORTED_MEDIA_EXTENSIONS
        assert ".exe" not in SUPPORTED_MEDIA_EXTENSIONS

    @staticmethod
    def test_device_type_enum() -> None:
        from app.core.constants import DeviceType

        assert {d.value for d in DeviceType} == {"auto", "cpu", "cuda", "mps"}
        assert DeviceType.CUDA.is_accelerator
        assert DeviceType.CPU.is_accelerator is False

    @staticmethod
    def test_speaker_label_regex_pattern() -> None:
        import re
        from app.core.constants import SPEAKER_LABEL_REGEX

        regex = re.compile(SPEAKER_LABEL_REGEX)
        assert regex.match("SPEAKER_00") is not None
        assert regex.match("SPEAKER_99") is not None
        assert regex.match("SPEAKER_0") is None
        assert regex.match("Luke") is None
        assert regex.match("speaker_00") is None
        assert regex.match("SPEAKER_100") is None


class TestExceptionsHierarchy:
    @staticmethod
    def test_profile_downgrade_required_exception_payload() -> None:
        from app.core.exceptions import ProfileDowngradeRequired

        exc = ProfileDowngradeRequired(
            requested_profile="quality",
            required_vram_gb=8.0,
            available_vram_gb=3.5,
            alternatives=["balanced", "performance", "cpu-only"],
        )
        assert exc.code == "PROFILE_DOWNGRADE_REQUIRED"
        assert exc.details["requested_profile"] == "quality"
        assert exc.details["alternative_profiles"] == ["balanced", "performance", "cpu-only"]
        assert "8.0 GB VRAM" in str(exc) or "8.0" in str(exc)

    @staticmethod
    def test_all_custom_exceptions_inherit_enginebase() -> None:
        import inspect
        from app.core import exceptions as exc_module
        from app.core.exceptions import EngineBaseError

        classes = [
            cls for _, cls in inspect.getmembers(exc_module, inspect.isclass)
            if issubclass(cls, Exception) and cls is not EngineBaseError and
            cls.__module__ == exc_module.__name__
        ]
        for cls in classes:
            assert issubclass(cls, EngineBaseError), (
                f"{cls.__name__} debe heredar de EngineBaseError"
            )


_EXPECTED_DIRS: list[str] = [
    "app/core",
    "app/domain/entities",
    "app/domain/value_objects",
    "app/domain/interfaces",
    "app/application/use_cases",
    "app/application/services/quality",
    "app/application/pipeline",
    "app/infrastructure/audio",
    "app/infrastructure/asr",
    "app/infrastructure/diarization",
    "app/infrastructure/language",
    "app/infrastructure/storage",
    "app/presentation/cli",
    "tests/domain",
    "tests/application/quality",
    "tests/infrastructure/audio",
    "tests/integration",
    "scripts",
    "data/input",
    "data/output",
    "data/temporary",
    "configs",
    "docs",
    "models",
]


@pytest.mark.parametrize("subdir", _EXPECTED_DIRS)
def test_expected_directories_exist(project_root: Path, subdir: str) -> None:
    target = project_root / subdir
    assert target.exists(), f"Falta directorio obligatorio: {subdir}"
    assert target.is_dir(), f"No es un directorio: {subdir}"


_EXPECTED_ROOT_FILES: list[str] = [
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    "LICENSE",
    "README.md",
    "app/__init__.py",
    "app/__main__.py",
    "app/presentation/cli/main_group.py",
    "app/core/constants.py",
    "app/core/exceptions.py",
    "tests/conftest.py",
]


@pytest.mark.parametrize("relpath", _EXPECTED_ROOT_FILES)
def test_expected_root_files_exist(project_root: Path, relpath: str) -> None:
    target = project_root / relpath
    assert target.exists(), f"Falta archivo obligatorio: {relpath}"
    assert target.is_file(), f"No es un archivo: {relpath}"
