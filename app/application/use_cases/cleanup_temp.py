from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.exceptions import InvalidMediaPathError, StorageError, ValidationFailedError
from app.core.logging import bind_context, get_logger
from app.domain.entities.cleanup import TempCleanupResult
from app.domain.interfaces.cleanup_ports import TempCleanupPort


log = get_logger("app.application.use_cases.cleanup_temp")


@dataclass(frozen=True, slots=True)
class RunTempCleanupUseCase:
    cleanup: TempCleanupPort


    def run(
        self,
        temp_root: str,
        job_id: str | None,
        *,
        cleanup_enabled: bool,
        keep_temp_files: bool,
        force_save: bool,
    ) -> TempCleanupResult:
        bind_context(job_id=job_id, phase="CLEANUP")
        t0 = time.perf_counter()

        if not temp_root or not str(temp_root).strip():
            raise ValidationFailedError("RunTempCleanupUseCase: temp_root vacío")

        if not cleanup_enabled or keep_temp_files:
            log.info(
                "cleanup_skipped",
                extra={"cleanup_enabled": cleanup_enabled, "keep_temp_files": keep_temp_files},
            )
            return TempCleanupResult(
                job_id=job_id,
                cleaned=False,
                removed_paths=(),
                errors=(),
                success=True,
            )

        if job_id is None or not str(job_id).strip():
            raise ValidationFailedError(
                "RunTempCleanupUseCase: job_id vacío con cleanup habilitado."
            )

        try:
            removed = self.cleanup.remove_tree(temp_root=temp_root, job_id=str(job_id))
        except (InvalidMediaPathError, StorageError):
            raise
        except (PermissionError, OSError) as exc:
            log.warning("cleanup_filesystem_error", extra={"error": str(exc)})
            return TempCleanupResult(
                job_id=job_id,
                cleaned=False,
                removed_paths=(),
                errors=(str(exc),),
                success=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("cleanup_unexpected_error", extra={"error": str(exc)})
            return TempCleanupResult(
                job_id=job_id,
                cleaned=False,
                removed_paths=(),
                errors=(str(exc),),
                success=False,
            )

        cleaned = True
        result = TempCleanupResult(
            job_id=job_id,
            cleaned=cleaned,
            removed_paths=tuple(removed),
            errors=(),
            success=True,
        )

        t1 = time.perf_counter()
        log.info(
            "cleanup_finished_ok",
            extra={
                "elapsed_sec": round(t1 - t0, 4),
                "removed": len(removed),
                "force_save": force_save,
            },
        )
        return result


__all__ = ["RunTempCleanupUseCase"]
