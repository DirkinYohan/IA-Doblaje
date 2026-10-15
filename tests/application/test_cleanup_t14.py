"""Tests unitarios T14 Application RunTempCleanupUseCase.

No requiere filesystem (Fake port). Cubre políticas cleanup_enabled/keep_temp_files,
force_save, idempotencia, errores de seguridad y determinismo.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.core.exceptions import InvalidMediaPathError, StorageError, ValidationFailedError
from app.domain.entities.cleanup import TempCleanupResult
from app.application.use_cases.cleanup_temp import RunTempCleanupUseCase


class _FakeCleanup:
    def __init__(self, removed=(), error=None):
        self.removed = list(removed)
        self.error = error
        self.calls = 0
        self.last_args = None

    def remove_tree(self, *, temp_root, job_id):
        self.calls += 1
        self.last_args = (temp_root, job_id)
        if self.error is not None:
            raise self.error
        return tuple(self.removed)


def _run(cleanup, temp_root="/tmp", job_id="job-12345678", cleanup_enabled=True, keep_temp_files=False, force_save=False):
    return RunTempCleanupUseCase(cleanup=cleanup).run(
        temp_root, job_id, cleanup_enabled=cleanup_enabled, keep_temp_files=keep_temp_files, force_save=force_save
    )


# =============================================================================
# Políticas
# =============================================================================


class TestPolicies:
    def test_cleanup_disabled(self):
        c = _FakeCleanup()
        r = _run(c, cleanup_enabled=False)
        assert r.cleaned is False
        assert r.success is True
        assert r.removed_paths == ()
        assert c.calls == 0

    def test_keep_temp_files_true(self):
        c = _FakeCleanup()
        r = _run(c, keep_temp_files=True)
        assert r.cleaned is False
        assert r.success is True
        assert c.calls == 0

    def test_cleanup_enabled_removes(self):
        c = _FakeCleanup(removed=("job-12345678",))
        r = _run(c)
        assert r.cleaned is True
        assert r.success is True
        assert r.removed_paths == ("job-12345678",)
        assert c.calls == 1

    def test_cleanup_enabled_nonexistent_idempotent(self):
        c = _FakeCleanup(removed=())
        r = _run(c)
        assert r.cleaned is True
        assert r.success is True
        assert r.removed_paths == ()
        assert c.calls == 1


class TestForceSave:
    def test_force_save_true(self):
        c1 = _FakeCleanup(removed=("job-12345678",))
        r1 = _run(c1, force_save=True)
        assert r1.cleaned is True

    def test_force_save_false(self):
        c2 = _FakeCleanup(removed=("job-12345678",))
        r2 = _run(c2, force_save=False)
        assert r2.cleaned is True

    def test_force_save_equivalence(self):
        # Mismo comportamiento con force_save True y False
        c1 = _FakeCleanup(removed=("job-12345678",))
        c2 = _FakeCleanup(removed=("job-12345678",))
        r1 = _run(c1, force_save=True)
        r2 = _run(c2, force_save=False)
        assert r1.cleaned == r2.cleaned
        assert r1.success == r2.success
        assert r1.removed_paths == r2.removed_paths


class TestErrors:
    def test_security_error_propagates(self):
        c = _FakeCleanup(error=InvalidMediaPathError("traversal"))
        with pytest.raises(InvalidMediaPathError):
            _run(c)

    def test_storage_error_propagates(self):
        c = _FakeCleanup(error=StorageError("fs error"))
        with pytest.raises(StorageError):
            _run(c)

    def test_operational_error_returns_failure(self):
        c = _FakeCleanup(error=PermissionError("denied"))
        r = _run(c)
        assert r.success is False
        assert r.errors != ()

    def test_operational_oserror_returns_failure(self):
        c = _FakeCleanup(error=OSError("io"))
        r = _run(c)
        assert r.success is False


class TestInputValidation:
    def test_empty_temp_root_raises(self):
        with pytest.raises(ValidationFailedError):
            _run(_FakeCleanup(), temp_root="")

    def test_none_job_id_with_cleanup_raises(self):
        with pytest.raises(ValidationFailedError):
            _run(_FakeCleanup(), job_id=None)

    def test_none_job_id_cleanup_disabled_ok(self):
        c = _FakeCleanup()
        r = _run(c, job_id=None, cleanup_enabled=False)
        assert r.success is True
        assert r.cleaned is False


class TestNoMutation:
    def test_inputs_not_mutated(self):
        c = _FakeCleanup(removed=("job-12345678",))
        r = _run(c)
        assert isinstance(r, TempCleanupResult)
        # el port no muta los argumentos pasados
        assert c.last_args == ("/tmp", "job-12345678")


class TestDeterminism:
    def test_same_input_same_output(self):
        c1 = _FakeCleanup(removed=("job-12345678",))
        c2 = _FakeCleanup(removed=("job-12345678",))
        r1 = _run(c1)
        r2 = _run(c2)
        assert r1.model_dump() == r2.model_dump()


class TestPortIntegration:
    def test_job_id_passed_to_port(self):
        c = _FakeCleanup(removed=("job-abc",))
        _run(c, job_id="job-abc")
        assert c.last_args[1] == "job-abc"

    def test_temp_root_passed_to_port(self):
        c = _FakeCleanup(removed=("job-abc",))
        _run(c, temp_root="/data/temp")
        assert c.last_args[0] == "/data/temp"


class TestASTAudit:
    def test_application_no_filesystem_no_ai(self):
        p = Path("app/application/use_cases/cleanup_temp.py")
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess", "pathlib", "shutil", "os"}
        assert not (imports & forbidden), f"importa prohibido: {imports & forbidden}"

    def test_domain_no_ai(self):
        for p in ["app/domain/entities/cleanup.py", "app/domain/interfaces/cleanup_ports.py"]:
            tree = ast.parse(Path(p).read_text(encoding="utf-8"))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imports.add(a.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
            forbidden = {"torch", "pyannote", "numpy", "speechbrain", "huggingface_hub", "requests", "urllib", "httpx", "subprocess"}
            assert not (imports & forbidden), f"{p} importa prohibido: {imports & forbidden}"


# =============================================================================
# Cobertura adicional
# =============================================================================


class TestPoliciesAdditional:
    def test_cleanup_disabled_no_port_call(self):
        c = _FakeCleanup()
        _run(c, cleanup_enabled=False)
        assert c.calls == 0

    def test_keep_temp_files_no_port_call(self):
        c = _FakeCleanup()
        _run(c, keep_temp_files=True)
        assert c.calls == 0

    def test_cleanup_disabled_with_keep_true(self):
        c = _FakeCleanup()
        r = _run(c, cleanup_enabled=False, keep_temp_files=True)
        assert r.cleaned is False
        assert r.success is True
        assert c.calls == 0

    def test_cleanup_enabled_keep_false_calls_port(self):
        c = _FakeCleanup(removed=("job-12345678",))
        r = _run(c, cleanup_enabled=True, keep_temp_files=False)
        assert c.calls == 1
        assert r.cleaned is True

    def test_result_success_true_no_errors(self):
        c = _FakeCleanup(removed=("x",))
        r = _run(c)
        assert r.success is True
        assert r.errors == ()

    def test_result_removed_paths_deterministic_order(self):
        c = _FakeCleanup(removed=("a", "b", "c"))
        r = _run(c)
        assert r.removed_paths == ("a", "b", "c")


class TestErrorsAdditional:
    def test_unexpected_error_returns_failure(self):
        c = _FakeCleanup(error=RuntimeError("boom"))
        r = _run(c)
        assert r.success is False
        assert r.errors != ()

    def test_security_error_not_converted_to_warning(self):
        c = _FakeCleanup(error=InvalidMediaPathError("traversal"))
        with pytest.raises(InvalidMediaPathError):
            _run(c)

    def test_operational_error_deterministic(self):
        c1 = _FakeCleanup(error=OSError("io"))
        c2 = _FakeCleanup(error=OSError("io"))
        r1 = _run(c1)
        r2 = _run(c2)
        assert r1.errors == r2.errors


class TestJobIdValidation:
    def test_whitespace_job_id_raises(self):
        with pytest.raises(ValidationFailedError):
            _run(_FakeCleanup(), job_id="   ")

    def test_valid_job_id(self):
        c = _FakeCleanup(removed=("job-abc",))
        r = _run(c, job_id="job-abc")
        assert r.job_id == "job-abc"
