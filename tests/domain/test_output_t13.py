"""Tests unitarios T13 — Domain Value Objects y Entity Output.

Solo DOMAIN, sin filesystem/IA. Pydantic + patrón DUT.
"""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.domain.entities.output import StructuredJsonOutputResult
from app.domain.value_objects.output import OutputFileRecord


_SHA = "a" * 64
_SHA_B = "b" * 64


def _record(name="analysis.json", sha=_SHA):
    return OutputFileRecord(filename=name, sha256=sha)


def _result(**kw):
    base = dict(
        output_directory="/out",
        written_files=(_record("analysis.json"), _record("transcript.json")),
        file_sha256={"analysis.json": _SHA, "transcript.json": _SHA},
        files_written_count=2,
        output_success=True,
        schema_version="0.1.0",
    )
    base.update(kw)
    return StructuredJsonOutputResult(**base)


# =============================================================================
# OutputFileRecord
# =============================================================================


class TestOutputFileRecord:
    def test_valid(self):
        r = _record()
        assert r.filename == "analysis.json"
        assert r.sha256 == _SHA

    def test_filename_must_end_json(self):
        with pytest.raises(ValidationError):
            OutputFileRecord(filename="analysis.txt", sha256=_SHA)

    def test_sha_invalid(self):
        with pytest.raises(ValidationError):
            OutputFileRecord(filename="analysis.json", sha256="bad")

    def test_sha_64hex_lower_normalized(self):
        r = OutputFileRecord(filename="analysis.json", sha256=_SHA.upper())
        assert r.sha256 == _SHA

    def test_frozen(self):
        r = _record()
        with pytest.raises(ValidationError):
            r.sha256 = _SHA_B  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            OutputFileRecord(filename="analysis.json", sha256=_SHA, x=1)  # type: ignore[call-arg]


# =============================================================================
# StructuredJsonOutputResult
# =============================================================================


class TestStructuredJsonOutputResult:
    def test_valid(self):
        r = _result()
        assert r.step.name == "STRUCTURED_JSON_OUTPUT"
        assert r.output_success is True
        assert r.files_written_count == 2

    def test_count_mismatch_raises(self):
        with pytest.raises(ValidationError):
            _result(files_written_count=1)

    def test_file_sha_missing_raises(self):
        with pytest.raises(ValidationError):
            _result(file_sha256={"analysis.json": _SHA})  # falta transcript

    def test_file_sha_extra_raises(self):
        with pytest.raises(ValidationError):
            _result(file_sha256={"analysis.json": _SHA, "transcript.json": _SHA, "metadata.json": _SHA})

    def test_output_success_empty_raises(self):
        with pytest.raises(ValidationError):
            _result(written_files=(), file_sha256={}, files_written_count=0, output_success=True)

    def test_output_success_false_ok(self):
        r = _result(written_files=(), file_sha256={}, files_written_count=0, output_success=False)
        assert r.output_success is False

    def test_frozen(self):
        r = _result()
        with pytest.raises(ValidationError):
            r.output_directory = "/x"  # type: ignore[misc]

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            _result(x=1)  # type: ignore[call-arg]

    def test_determinism(self):
        a = _result()
        b = copy.deepcopy(a)
        assert a == b
        assert a.model_dump() == b.model_dump()


class TestStructuredJsonOutputResultAdditional:
    def test_model_label(self):
        r = _result()
        assert r.model_label == "t13:structured-json-output:v1"

    def test_job_id_optional(self):
        r = _result(job_id="job-12345678")
        assert r.job_id == "job-12345678"

    def test_job_id_min_length(self):
        with pytest.raises(ValidationError):
            _result(job_id="short")

    def test_output_directory_required(self):
        with pytest.raises(ValidationError):
            StructuredJsonOutputResult(
                output_directory="", written_files=(), file_sha256={},
                files_written_count=0, output_success=False, schema_version="0.1.0",
            )

    def test_schema_version_required(self):
        with pytest.raises(ValidationError):
            StructuredJsonOutputResult(
                output_directory="/out", written_files=(), file_sha256={},
                files_written_count=0, output_success=False, schema_version="",
            )

    def test_files_written_count_bool_rejected(self):
        with pytest.raises(ValidationError):
            _result(files_written_count=True)  # type: ignore[arg-type]
