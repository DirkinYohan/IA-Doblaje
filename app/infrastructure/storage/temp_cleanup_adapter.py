from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.exceptions import InvalidMediaPathError, StorageError
from app.core.paths import safe_resolve_path
from app.domain.interfaces.cleanup_ports import TempCleanupPort


class TempCleanupAdapter(TempCleanupPort):
    def remove_tree(self, *, temp_root: str, job_id: str) -> tuple[str, ...]:
        root = Path(temp_root).expanduser()
        if not root.is_absolute():
            raise InvalidMediaPathError(
                "temp_root debe ser una ruta absoluta.",
                details={"temp_root": temp_root},
            )

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

        if target.resolve() == root.resolve():
            raise InvalidMediaPathError(
                "No se permite eliminar el directorio temporal raíz.",
                details={"temp_root": temp_root},
            )

        if not target.exists():
            return ()

        removed: list[str] = []
        try:
            for p in target.rglob("*"):
                removed.append(p.relative_to(root).as_posix())
            removed.append(target.relative_to(root).as_posix())
        except OSError as exc:
            raise StorageError(f"Error al inspeccionar el job temp dir: {exc!r}") from exc

        removed.sort()

        try:
            shutil.rmtree(target)
        except (PermissionError, OSError) as exc:
            raise StorageError(f"Error de filesystem al eliminar: {exc!r}") from exc

        return tuple(removed)


__all__ = ["TempCleanupAdapter"]
