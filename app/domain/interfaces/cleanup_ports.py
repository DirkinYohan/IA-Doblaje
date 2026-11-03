from __future__ import annotations

from abc import ABC, abstractmethod


class TempCleanupPort(ABC):
    @abstractmethod
    def remove_tree(self, *, temp_root: str, job_id: str) -> tuple[str, ...]:
        raise NotImplementedError  # pragma: no cover


__all__ = ["TempCleanupPort"]
