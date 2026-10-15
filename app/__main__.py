"""Entrypoint: python -m app.

En Fase 1-T01 solo muestra un mensaje de bienvenida y delega a Typer CLI
(implementacion minima: version, help, diagnose stub).
"""

from __future__ import annotations

import sys

from app.presentation.cli.main_group import app


def main() -> int:
    app(prog_name="ia-doblaje")
    return 0


if __name__ == "__main__":
    sys.exit(main())
