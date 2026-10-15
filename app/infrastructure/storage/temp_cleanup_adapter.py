"""Infrastructure Adapter Step 14 — TempCleanupAdapter (filesystem + stdlib).

INFRASTRUCTURE ONLY. Elimina recursivamente <temp_root>/<job_id> validando
path containment con safe_resolve_path. Sin IA, sin red, sin subprocess.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.exceptions import InvalidMediaPathError, StorageError
from app.core.paths import safe_resolve_path
from app.domain.interfaces.cleanup_ports import TempCleanupPort


class TempCleanupAdapter(TempCleanupPort):
    """Implementación concreta del TempCleanupPort.

    Elimina únicamente <temp_root>/<job_id>, con containment estricto.
    """

    def remove_tree(self, *, temp_root: str, job_id: str) -> tuple[str, ...]:
        root = Path(temp_root).expanduser()
        if not root.is_absolute():
            raise InvalidMediaPathError(
                "temp_root debe ser una ruta absoluta.",
                details={"temp_root": temp_root},
            )

        # Path containment: construir <temp_root>/<job_id> y validar que esté
        # DENTRO de temp_root. safe_resolve_path rechaza traversal/escape.
        try:
            target = safe_resolve_path(
                str(root / job_id),
                allowed_root=str(root),
                create_parent=False,
                must_exist=False,
            )
        except InvalidMediaPathError:
            raise
        except OSError as exc:
            raise StorageError(
                f"No se pudo resolver la ruta del job temp dir: {exc!r}"
            ) from exc

        # Garantía extra: el target no debe ser el propio root (borrar raíz prohibido)
        if target.resolve() == root.resolve():
            raise InvalidMediaPathError(
                "No se permite eliminar el directorio temporal raíz.",
                details={"temp_root": temp_root},
            )

        if not target.exists():
            # Idempotente: no hay nada que eliminar
            return ()

        # Recolectar rutas relativas (determinista) antes de eliminar
        removed: list[str] = []
        try:
            for p in target.rglob("*"):
                removed.append(p.relative_to(root).as_posix())
            removed.append(target.relative_to(root).as_posix())
        except OSError as exc:
            raise StorageError(f"Error al inspeccionar el job temp dir: {exc!r}") from exc

        # Orden lexicográfico determinista de TODAS las rutas
        removed.sort()

        try:
            shutil.rmtree(target)
        except (PermissionError, OSError) as exc:
            raise StorageError(f"Error de filesystem al eliminar: {exc!r}") from exc

        return tuple(removed)


__all__ = ["TempCleanupAdapter"]
