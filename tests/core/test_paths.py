from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app.core.exceptions import InvalidMediaPathError
from app.core.paths import (
    PathManager,
    generate_job_id,
    safe_filename,
    safe_job_name,
    safe_resolve_path,
)


@pytest.mark.parametrize(
    ("raw", "expected_contains"),
    [
        ("file:name?with|bad*chars<very>", "file"),
        ("CON", "_CON"),
        ("  with spaces .. ", "with_spaces"),
        ("", "file"),
        ("a" * 300, "a"),
    ],
)
def test_safe_filename_sanitizes(raw: str, expected_contains: str) -> None:
    s = safe_filename(raw)
    assert expected_contains in s.lower() or s.lower().startswith(expected_contains.lower())
    for bad in '<>:"/\\|?*':
        assert bad not in s
    assert not s.endswith((".", " "))


def test_safe_job_name() -> None:
    assert safe_job_name("Job 123 #special") == "Job_123__special"
    assert safe_job_name("") == "unnamed_job"
    assert safe_job_name("a" * 100) == "a" * 64


def test_generate_job_id_unique_and_safe() -> None:
    ids = {generate_job_id() for _ in range(200)}
    assert len(ids) == 200
    for jid in ids:
        assert jid.startswith("job-")
        assert all(c not in jid for c in " \\/:?*<>|")
        parts = jid.split("-")
        assert len(parts) >= 3


@pytest.fixture
def temp_allowed_root(tmp_path: Path) -> Path:
    root = tmp_path / "allowed_root"
    root.mkdir(parents=True, exist_ok=True)
    (root / "inside.txt").write_text("hello")
    sub = root / "subdir"
    sub.mkdir(exist_ok=True)
    (sub / "nested.mp4").write_text("video")
    return root


def test_safe_resolve_normal_paths_ok(temp_allowed_root: Path) -> None:
    p = safe_resolve_path("inside.txt", temp_allowed_root, must_exist=True)
    assert p.name == "inside.txt"
    p2 = safe_resolve_path("subdir/../inside.txt", temp_allowed_root, must_exist=True)
    assert p2 == p
    p3 = safe_resolve_path(str(temp_allowed_root / "subdir" / "nested.mp4"), temp_allowed_root, must_exist=True)
    assert p3.name == "nested.mp4"


def test_traversal_up_rejected(temp_allowed_root: Path) -> None:
    with pytest.raises(InvalidMediaPathError):
        safe_resolve_path("../../../etc/passwd", temp_allowed_root)
    with pytest.raises(InvalidMediaPathError):
        safe_resolve_path("..\\..\\Windows\\System32", temp_allowed_root)
    with pytest.raises(InvalidMediaPathError):
        safe_resolve_path("subdir/../../../../etc/hosts", temp_allowed_root)


def test_traversal_drive_absolute_rejected(temp_allowed_root: Path) -> None:
    external = Path(temp_allowed_root.parent) / "external_file.txt"
    external.write_text("leak")
    with pytest.raises(InvalidMediaPathError):
        safe_resolve_path(str(external), temp_allowed_root)


def test_traversal_empty_path_rejected(temp_allowed_root: Path) -> None:
    with pytest.raises(InvalidMediaPathError):
        safe_resolve_path("", temp_allowed_root)
    with pytest.raises(InvalidMediaPathError):
        safe_resolve_path(None, temp_allowed_root)  # type: ignore[arg-type]


def test_safe_resolve_create_parent_true(tmp_path: Path) -> None:
    root = tmp_path / "r"
    result = safe_resolve_path(
        "a/b/c/out.json",
        root,
        create_parent=True,
        must_exist=False,
    )
    assert result.parent.exists()
    assert result.name == "out.json"


def test_allow_outside_flag_bypasses_security(tmp_path: Path) -> None:
    inside = tmp_path / "not_allowed"
    inside.write_text("x")
    root = tmp_path / "allowed"
    root.mkdir(exist_ok=True)
    result = safe_resolve_path(str(inside), root, allow_outside=True)
    assert result.name == "not_allowed"


@pytest.fixture
def pm(tmp_path: Path) -> PathManager:
    return PathManager(
        project_root=tmp_path,
        data_input_dir=tmp_path / "in",
        data_output_dir=tmp_path / "out",
        data_temp_dir=tmp_path / "tmp",
        models_cache_dir=tmp_path / "models",
    )


def test_ensure_dirs_creates_all_four(pm: PathManager) -> None:
    before = [pm.data_input_dir.exists(), pm.data_output_dir.exists(), pm.data_temp_dir.exists(), pm.models_cache_dir.exists()]
    assert not any(before)
    dirs = pm.ensure_dirs()
    assert len(dirs) == 4
    for d in (pm.data_input_dir, pm.data_output_dir, pm.data_temp_dir, pm.models_cache_dir):
        assert d.exists() and d.is_dir()


def test_create_job_temp_dir_structure_and_cleanup(pm: PathManager) -> None:
    pm.ensure_dirs()
    jid, jroot = pm.create_job_temp_dir()
    assert isinstance(jid, str) and len(jid) > 0
    assert jroot.exists()
    for sub in ("workspace", "cache", "logs"):
        assert (jroot / sub).is_dir()
    assert pm.get_job_temp_dir(jid) is not None
    ws = pm.job_workspace(jid)
    assert ws.exists() and ws.name == "workspace"
    ok = pm.cleanup_job(jid)
    assert ok is True
    assert not jroot.exists()
    assert pm.cleanup_job(jid) is False


def test_jobs_isolated_different_ids(pm: PathManager) -> None:
    pm.ensure_dirs()
    j1, r1 = pm.create_job_temp_dir("job-one")
    j2, r2 = pm.create_job_temp_dir("job-two")
    assert r1 != r2
    assert not r1.is_relative_to(r2) and not r2.is_relative_to(r1)
    (pm.job_workspace(j1) / "secret.txt").write_text("data")
    assert not (pm.job_workspace(j2) / "secret.txt").exists()
    pm.cleanup_job(j1)
    assert not r1.exists()
    assert r2.exists()
    pm.cleanup_job(j2)


def test_collision_job_id_suffix(pm: PathManager) -> None:
    pm.ensure_dirs()
    fixed = (pm.data_temp_dir / "same_id")
    fixed.mkdir(parents=True, exist_ok=True)
    jid, jroot = pm.create_job_temp_dir("same_id")
    assert jroot != fixed
    assert str(jroot.name).startswith("same_id_")


def test_resolve_input_rejects_traversal(pm: PathManager) -> None:
    pm.ensure_dirs()
    (pm.data_input_dir / "video.mp4").write_text("ok")
    p = pm.resolve_input_path("video.mp4")
    assert p.name == "video.mp4"
    with pytest.raises(InvalidMediaPathError):
        pm.resolve_input_path("../../Windows/explorer.exe")


def test_resolve_output_path_within_output(pm: PathManager) -> None:
    pm.ensure_dirs()
    p = pm.resolve_output_path("jobs/001/analysis.json")
    assert p.parent.exists()
    assert p.name == "analysis.json"
    assert str(p).startswith(str(pm.data_output_dir))


def test_output_filename_helper(pm: PathManager) -> None:
    s = pm.output_filename("clip 1.mp4", suffix="json", job_id="job_XYZ-123")
    assert s.lower().endswith(".json")
    assert "job_xyz_123" in s.lower() or "job_XYZ_123" in s or "job_XYZ-123" in s
    assert "clip_1" in s or "clip 1" in s
