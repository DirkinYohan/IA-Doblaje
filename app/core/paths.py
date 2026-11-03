from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

from app.core.exceptions import InvalidMediaPathError


_UNSAFE_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_UNSAFE_JOB_CHARS = re.compile(r"[^a-zA-Z0-9._\-]")
_WINDOWS_RESERVED_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})


def safe_filename(raw: str, fallback: str = "file") -> str:
    if not raw:
        return fallback
    name = str(raw).strip()
    name = _UNSAFE_FS_CHARS.sub("_", name)
    name = name.replace(" ", "_")
    name = name.rstrip(" .")
    if name.upper().split(".")[0] in _WINDOWS_RESERVED_NAMES:
        name = f"_{name}"
    if not name:
        name = fallback
    if len(name) > 200:
        name, ext = os.path.splitext(name)
        name = name[: 200 - len(ext)] + ext
    return name


def safe_job_name(raw: str, fallback: str = "unnamed_job") -> str:
    if not raw:
        return fallback
    s = _UNSAFE_JOB_CHARS.sub("_", str(raw).strip())
    s = s.strip("._-")
    if not s:
        return fallback
    if len(s) > 64:
        s = s[:64]
    return s


def generate_job_id(prefix: str = "job") -> str:
    p = safe_job_name(prefix, fallback="job")
    rnd = os.urandom(2).hex()
    return f"{p}-{uuid.uuid4().hex[:12]}-{rnd}"


def safe_resolve_path(
    user_input: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
    *,
    create_parent: bool = False,
    must_exist: bool = False,
    allow_outside: bool = False,
) -> Path:
    if user_input is None or not str(user_input):
        raise InvalidMediaPathError(
            "Ruta vacia no permitida.",
            details={"allowed_root": str(allowed_root)},
        )

    root = Path(allowed_root).expanduser().resolve()
    if not root.exists():
        if create_parent:
            root.mkdir(parents=True, exist_ok=True)
            root = Path(allowed_root).expanduser().resolve()
        elif must_exist:
            raise InvalidMediaPathError(
                f"Directorio base no existe: {root}",
                details={"allowed_root": str(root)},
            )

    target = Path(user_input).expanduser()
    if not target.is_absolute():
        target = root / target

    try:
        resolved = target.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise InvalidMediaPathError(
            f"No se puede resolver la ruta: {user_input}",
            details={"allowed_root": str(root), "error": str(exc)},
            cause=exc,
        ) from exc

    if allow_outside:
        if create_parent:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    try:
        rel = resolved.relative_to(root)
    except ValueError:
        raise InvalidMediaPathError(
            "La ruta proporcionada escapa del directorio permitido "
            "(posible ataque path-traversal).",
            details={
                "allowed_root": str(root),
                "user_input": str(user_input),
                "resolved_path": str(resolved),
            },
        )

    r_drive, _ = os.path.splitdrive(str(root))
    t_drive, _ = os.path.splitdrive(str(resolved))
    if r_drive and t_drive and str(r_drive).lower() != str(t_drive).lower():
        raise InvalidMediaPathError(
            "La ruta pertenece a otra unidad de disco.",
            details={
                "allowed_drive": r_drive,
                "target_drive": t_drive,
                "resolved": str(resolved),
            },
        )

    if create_parent and not resolved.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


class PathManager:
    def __init__(
        self,
        project_root: Path | str,
        *,
        data_input_dir: Path | str | None = None,
        data_output_dir: Path | str | None = None,
        data_temp_dir: Path | str | None = None,
        models_cache_dir: Path | str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.data_input_dir = Path(data_input_dir).resolve() if data_input_dir else (self.project_root / "data" / "input").resolve()
        self.data_output_dir = Path(data_output_dir).resolve() if data_output_dir else (self.project_root / "data" / "output").resolve()
        self.data_temp_dir = Path(data_temp_dir).resolve() if data_temp_dir else (self.project_root / "data" / "temporary").resolve()
        self.models_cache_dir = Path(models_cache_dir).resolve() if models_cache_dir else (self.project_root / "models").resolve()
        self._job_temp_dirs: dict[str, Path] = {}

    @classmethod
    def from_settings(cls, settings: object | None = None) -> "PathManager":
        if settings is None:
            try:
                from app.core.config import get_settings
                settings = get_settings()
            except Exception:
                return cls(Path(__file__).resolve().parents[2])
        paths_cfg = getattr(settings, "paths", None)
        return cls(
            project_root=getattr(settings, "project_root", None) or Path(__file__).resolve().parents[2],
            data_input_dir=getattr(paths_cfg, "data_input_dir", None),
            data_output_dir=getattr(paths_cfg, "data_output_dir", None),
            data_temp_dir=getattr(paths_cfg, "data_temp_dir", None),
            models_cache_dir=getattr(paths_cfg, "models_cache_dir", None),
        )

    def ensure_dirs(self, *, extra: Iterable[Path | str] | None = None) -> list[Path]:
        dirs = [
            self.data_input_dir,
            self.data_output_dir,
            self.data_temp_dir,
            self.models_cache_dir,
        ]
        if extra:
            for p in extra:
                dirs.append(Path(p).resolve())
        created: list[Path] = []
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            created.append(d.resolve())
        return created


    def resolve_input_path(self, user_input: str | os.PathLike[str], *, must_exist: bool = True) -> Path:
        return safe_resolve_path(
            user_input,
            self.data_input_dir,
            must_exist=must_exist,
            create_parent=False,
        )

    def resolve_output_path(
        self,
        relative_output: str | os.PathLike[str],
        *,
        create_parent: bool = True,
        must_exist: bool = False,
    ) -> Path:
        return safe_resolve_path(
            relative_output,
            self.data_output_dir,
            must_exist=must_exist,
            create_parent=create_parent,
        )


    def create_job_temp_dir(
        self,
        job_id: str | None = None,
        *,
        prefix: str = "job",
        use_os_temp: bool = False,
    ) -> tuple[str, Path]:
        jid = safe_job_name(job_id) if job_id else generate_job_id(prefix=prefix)
        if use_os_temp:
            base = Path(tempfile.mkdtemp(prefix=f"{jid}_"))
        else:
            self.data_temp_dir.mkdir(parents=True, exist_ok=True)
            base = self.data_temp_dir / jid
            if base.exists():
                base = self.data_temp_dir / f"{jid}_{os.urandom(2).hex()}"
            base.mkdir(parents=True, exist_ok=False)

        for sub in ("workspace", "cache", "logs"):
            (base / sub).mkdir(parents=True, exist_ok=True)

        self._job_temp_dirs[jid] = base.resolve()
        return jid, base.resolve()

    def get_job_temp_dir(self, job_id: str) -> Path | None:
        return self._job_temp_dirs.get(safe_job_name(job_id))

    def job_subdir(self, job_id: str, sub: str) -> Path:
        base = self.get_job_temp_dir(job_id)
        if base is None:
            raise KeyError(f"Job no registrado en PathManager: {job_id}")
        sub_safe = safe_job_name(sub, fallback="misc")
        d = base / sub_safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def job_workspace(self, job_id: str) -> Path:
        return self.job_subdir(job_id, "workspace")

    def job_cache(self, job_id: str) -> Path:
        return self.job_subdir(job_id, "cache")

    def job_logs(self, job_id: str) -> Path:
        return self.job_subdir(job_id, "logs")

    def cleanup_job(self, job_id: str, *, ignore_errors: bool = True) -> bool:
        jid = safe_job_name(job_id)
        d = self._job_temp_dirs.pop(jid, None)
        if d is None:
            d = self.data_temp_dir / jid
            if not d.exists():
                return False
        try:
            shutil.rmtree(d, ignore_errors=ignore_errors)
            return True
        except OSError:
            if not ignore_errors:
                raise
            return False

    def output_filename(
        self,
        base_name: str,
        *,
        suffix: str = "json",
        job_id: str | None = None,
    ) -> str:
        stem = safe_filename(Path(str(base_name)).stem, fallback="output")
        suf = suffix.strip().lstrip(".")
        if not suf:
            suf = "bin"
        if job_id:
            return f"{stem}_{safe_job_name(job_id)}.{suf}"
        return f"{stem}.{suf}"


__all__ = [
    "safe_resolve_path",
    "safe_filename",
    "safe_job_name",
    "generate_job_id",
    "PathManager",
]
