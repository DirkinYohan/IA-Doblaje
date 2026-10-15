"""Tests minimos CLI (T01).

Verifica que Typer este integrado y los comandos basicos responden.
No ejecuta pipeline real (eso es T11).
"""

from __future__ import annotations

from typer.testing import CliRunner

import pytest


@pytest.fixture(scope="module")
def cli_runner() -> CliRunner:
    return CliRunner()


class TestCLIMinimum:
    @staticmethod
    def test_cli_main_group_can_be_imported() -> None:
        from app.presentation.cli.main_group import app
        assert app is not None
        assert callable(app)

    @staticmethod
    def test_cli_no_args_shows_help_and_banner(cli_runner: CliRunner) -> None:
        from app.presentation.cli.main_group import app

        result = cli_runner.invoke(app, [])
        # No args = ayuda. Typer con no_args_is_help=True retorna exit 0
        assert result.exit_code in (0, 2)  # 0 ayuda, 2 depende version Typer
        # Debe mencionar comandos existentes
        text = result.stdout
        assert any(needle in text for needle in ("analyze", "diagnose", "validate", "Usage"))

    @staticmethod
    def test_cli_version_flag(cli_runner: CliRunner) -> None:
        from app import __version__
        from app.presentation.cli.main_group import app

        result = cli_runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    @staticmethod
    def test_cli_help_flag(cli_runner: CliRunner) -> None:
        from app.presentation.cli.main_group import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "IA Profesional" in result.stdout or "Doblaje" in result.stdout

    @staticmethod
    def test_cli_diagnose_placeholder_runs(cli_runner: CliRunner) -> None:
        from app.presentation.cli.main_group import app

        result = cli_runner.invoke(app, ["diagnose"])
        # Placeholder T01 debe correr sin error fatal aun si PyTorch falta
        assert result.exit_code == 0, (
            f"diagnose fallo con exit={result.exit_code}\n"
            f"stdout: {result.stdout}\n"
            f"exc: {result.exception!r}"
        )

    @staticmethod
    def test_cli_analyze_placeholder_with_input(cli_runner: CliRunner, isolated_temp_dir) -> None:
        from app.presentation.cli.main_group import app

        fake_file = isolated_temp_dir / "fake_clip.mp4"
        fake_file.touch()
        result = cli_runner.invoke(app, [
            "analyze",
            str(fake_file),
            "--profile", "balanced",
            "--device", "cpu",
        ])
        # La CLI ya NO es un placeholder: ejecuta el flujo real y, sin
        # FFmpeg/modelos locales, el preflight falla con exit code 3.
        assert result.exit_code in (3, 5), (
            f"analyze real fallo inesperado. stdout:\n{result.stdout}\n"
            f"exc: {result.exception!r}"
        )
        # No debe considerarse éxito una ejecución que no pudo procesar el video.
        assert result.exit_code != 0
