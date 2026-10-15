"""Ports abstractos Temp Cleanup Step 14 — Capa DOMAIN.

Clean Architecture. Domain NO usa filesystem. Solo ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class TempCleanupPort(ABC):
    """Puerto abstracto de limpieza temporal. Application depende solo de esto.

    Implementación concreta (Infrastructure): filesystem + stdlib.
    """

    @abstractmethod
    def remove_tree(self, *, temp_root: str, job_id: str) -> tuple[str, ...]:
        """Elimina recursivamente <temp_root>/<job_id>.

        Devuelve la lista ordenada de rutas (strings, relativas a temp_root)
        que fueron eliminadas. Si no existe nada, devuelve ().

        Raises:
            InvalidMediaPathError si la ruta escapa del temp_root.
            StorageError si hay un error de filesystem estructural.
        """
        raise NotImplementedError  # pragma: no cover


__all__ = ["TempCleanupPort"]
