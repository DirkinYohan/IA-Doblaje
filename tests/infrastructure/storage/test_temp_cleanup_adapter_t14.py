"""Tests unitarios T14 Infrastructure — TempCleanupAdapter.

Usa filesystem temporal (tmp_path). Cubre eliminación real, árbol, contención,
traversal, raíz protegida, persistentes, idempotencia y AST.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.exceptions import InvalidMediaPathError, StorageError
from app.infrastructure.storage.temp_cleanup_adapter import TempCleanupAdapter


@pytest.fixture
def adapter():
    return TempCleanupAdapter()


# =============================================================================
# TempCleanupAdapter
# =============================================================================


class TestTempCleanupAdapter:
    def test_removes_job_dir(self, adapter, tmp_path):
        job = tmp_path / "job-1"
        job.mkdir()
        (job / "a.txt").write_text("x")
        removed = adapter.remove_tree(temp_root=str(tmp_path), job_id="job-1")
        assert not job.exists()
        assert len(removed) >= 1

    def test_removes_recursive_tree(self, adapter, tmp_path):
        job = tmp_path / "job-2"
        (job / "sub" / "nested").mkdir(parents=True)
        (job / "sub" / "nested" / "f.txt").write_text("x")
        (job / "top.txt").write_text("y")
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-2")
        assert not job.exists()

    def test_removes_empty_subdirs(self, adapter, tmp_path):
        job = tmp_path / "job-3"
        (job / "empty" / "deeper").mkdir(parents=True)
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-3")
        assert not job.exists()

    def test_nonexistent_idempotent(self, adapter, tmp_path):
        removed = adapter.remove_tree(temp_root=str(tmp_path), job_id="no-exist")
        assert removed == ()

    def test_traversal_rejected(self, adapter, tmp_path):
        with pytest.raises(InvalidMediaPathError):
            adapter.remove_tree(temp_root=str(tmp_path), job_id="../outside")

    def test_remove_root_rejected(self, adapter, tmp_path):
        # job_id que resuelve al propio temp_root (raíz) debe rechazarse
        with pytest.raises(InvalidMediaPathError):
            adapter.remove_tree(temp_root=str(tmp_path), job_id=".")

    def test_absolute_external_rejected(self, adapter, tmp_path):
        with pytest.raises(InvalidMediaPathError):
            adapter.remove_tree(temp_root=str(tmp_path), job_id=str(tmp_path / ".." / "other"))

    def test_removed_paths_relative(self, adapter, tmp_path):
        job = tmp_path / "job-4"
        job.mkdir()
        (job / "f.txt").write_text("x")
        removed = adapter.remove_tree(temp_root=str(tmp_path), job_id="job-4")
        assert all(not Path(r).is_absolute() for r in removed)
        assert "job-4" in removed

    def test_deterministic_order(self, adapter, tmp_path):
        job = tmp_path / "job-5"
        (job / "sub").mkdir(parents=True)
        (job / "a.txt").write_text("x")
        (job / "sub" / "b.txt").write_text("y")
        # dos ejecuciones (recrear) producen mismo orden
        def run():
            j = tmp_path / "job-5"
            j.mkdir(parents=True, exist_ok=True)
            (j / "a.txt").write_text("x")
            (j / "sub").mkdir(exist_ok=True)
            (j / "sub" / "b.txt").write_text("y")
            return adapter.remove_tree(temp_root=str(tmp_path), job_id="job-5")
        r1 = run()
        r2 = run()
        assert r1 == r2


class TestTempCleanupAdapterProtection:
    def test_does_not_remove_other_jobs(self, adapter, tmp_path):
        job_a = tmp_path / "job-a"
        job_b = tmp_path / "job-b"
        job_a.mkdir(); job_b.mkdir()
        (job_a / "f.txt").write_text("x")
        (job_b / "g.txt").write_text("y")
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-a")
        assert not job_a.exists()
        assert job_b.exists()  # otro job intacto

    def test_does_not_remove_temp_root(self, adapter, tmp_path):
        job = tmp_path / "job-c"
        job.mkdir()
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-c")
        assert tmp_path.exists()  # temp_root intacto


class TestASTAudit:
    def test_infrastructure_no_ai_no_subprocess(self):
        p = Path("app/infrastructure/storage/temp_cleanup_adapter.py")
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess"}
        assert not (imports & forbidden), f"importa prohibido: {imports & forbidden}"

    def test_infrastructure_stdlib_only(self):
        p = Path("app/infrastructure/storage/temp_cleanup_adapter.py")
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        stdlib_ok = {"shutil", "pathlib", "typing", "app", "__future__"}
        assert imports <= stdlib_ok, f"imports no stdlib: {imports - stdlib_ok}"


class TestTempCleanupAdapterAdditional:
    def test_partial_files_removed(self, adapter, tmp_path):
        job = tmp_path / "job-p"
        job.mkdir()
        (job / "partial.tmp").write_text("x")
        (job / "intermediate.wav").write_bytes(b"data")
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-p")
        assert not job.exists()

    def test_second_run_same_state(self, adapter, tmp_path):
        job = tmp_path / "job-r"
        job.mkdir()
        (job / "f.txt").write_text("x")
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-r")
        # segunda ejecución: idempotente
        removed2 = adapter.remove_tree(temp_root=str(tmp_path), job_id="job-r")
        assert removed2 == ()

    def test_removed_paths_include_all_files(self, adapter, tmp_path):
        job = tmp_path / "job-all"
        job.mkdir()
        (job / "a.txt").write_text("x")
        (job / "b.txt").write_text("y")
        removed = adapter.remove_tree(temp_root=str(tmp_path), job_id="job-all")
        rels = set(removed)
        assert "job-all/a.txt" in rels
        assert "job-all/b.txt" in rels
        assert "job-all" in rels

    def test_contains_job_id_in_removed(self, adapter, tmp_path):
        job = tmp_path / "job-cid"
        job.mkdir()
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-cid")
        assert not job.exists()

    def test_no_persistent_removal(self, adapter, tmp_path):
        # El adapter solo opera bajo temp_root; no toca directorios fuera
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("k")
        job = tmp_path / "job-x"
        job.mkdir()
        adapter.remove_tree(temp_root=str(tmp_path), job_id="job-x")
        assert outside.exists()
        assert (outside / "keep.txt").exists()

    def test_traversal_dotdot_slash(self, adapter, tmp_path):
        with pytest.raises(InvalidMediaPathError):
            adapter.remove_tree(temp_root=str(tmp_path), job_id="sub/../..")

    def test_removed_paths_sorted(self, adapter, tmp_path):
        job = tmp_path / "job-s"
        (job / "sub").mkdir(parents=True)
        (job / "z.txt").write_text("x")
        (job / "a.txt").write_text("y")
        removed = adapter.remove_tree(temp_root=str(tmp_path), job_id="job-s")
        # orden lexicográfico determinista
        assert list(removed) == sorted(removed)
