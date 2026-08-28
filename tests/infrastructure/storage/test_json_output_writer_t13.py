"""Tests unitarios T13 Infrastructure — JsonOutputWriter.

Usa filesystem temporal (tmp_path). allow_nan=False, SHA256, OutputWriteError.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from app.core.exceptions import OutputWriteError
from app.infrastructure.storage.json_output_writer import JsonOutputWriter


@pytest.fixture
def writer():
    return JsonOutputWriter()


# =============================================================================
# JsonOutputWriter
# =============================================================================


class TestJsonOutputWriter:
    def test_writes_file_and_returns_sha(self, writer, tmp_path):
        doc = {"a": 1, "b": "x"}
        sha = writer.write_json(
            output_directory=str(tmp_path), filename="test.json",
            document=doc, indent=2, ensure_ascii=False,
        )
        written = (tmp_path / "test.json").read_bytes()
        assert hashlib.sha256(written).hexdigest() == sha
        assert len(sha) == 64

    def test_sha_over_exact_bytes(self, writer, tmp_path):
        doc = {"a": 1}
        sha = writer.write_json(output_directory=str(tmp_path), filename="x.json", document=doc, indent=2, ensure_ascii=False)
        raw = json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False).encode("utf-8")
        assert sha == hashlib.sha256(raw).hexdigest()

    def test_allow_nan_false_rejects_nan(self, writer, tmp_path):
        with pytest.raises(OutputWriteError):
            writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": float("nan")}, indent=2, ensure_ascii=False)

    def test_allow_nan_false_rejects_inf(self, writer, tmp_path):
        with pytest.raises(OutputWriteError):
            writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": float("inf")}, indent=2, ensure_ascii=False)

    def test_allow_nan_false_rejects_neg_inf(self, writer, tmp_path):
        with pytest.raises(OutputWriteError):
            writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": float("-inf")}, indent=2, ensure_ascii=False)

    def test_creates_output_directory(self, writer, tmp_path):
        out = tmp_path / "nested" / "dir"
        sha = writer.write_json(output_directory=str(out), filename="x.json", document={"a": 1}, indent=2, ensure_ascii=False)
        assert (out / "x.json").exists()
        assert len(sha) == 64

    def test_unicode_preserved(self, writer, tmp_path):
        doc = {"texto": "áéíóú"}
        writer.write_json(output_directory=str(tmp_path), filename="x.json", document=doc, indent=2, ensure_ascii=False)
        content = (tmp_path / "x.json").read_text(encoding="utf-8")
        assert "áéíóú" in content

    def test_ensure_ascii_true(self, writer, tmp_path):
        doc = {"texto": "áéíóú"}
        writer.write_json(output_directory=str(tmp_path), filename="x.json", document=doc, indent=2, ensure_ascii=True)
        content = (tmp_path / "x.json").read_text(encoding="utf-8")
        assert "\\u00e1" in content

    def test_none_serializes_as_null(self, writer, tmp_path):
        sha = writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": None}, indent=2, ensure_ascii=False)
        content = (tmp_path / "x.json").read_text(encoding="utf-8")
        assert "null" in content
        assert len(sha) == 64

    def test_bool_preserved(self, writer, tmp_path):
        writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": True}, indent=2, ensure_ascii=False)
        content = (tmp_path / "x.json").read_text(encoding="utf-8")
        assert "true" in content

    def test_deterministic_bytes(self, writer, tmp_path):
        doc = {"a": 1, "b": [1, 2, 3]}
        sha1 = writer.write_json(output_directory=str(tmp_path), filename="x.json", document=doc, indent=2, ensure_ascii=False)
        sha2 = writer.write_json(output_directory=str(tmp_path), filename="x.json", document=doc, indent=2, ensure_ascii=False)
        assert sha1 == sha2

    def test_write_error_raises_output_write_error(self, writer, tmp_path):
        # directorio que no se puede crear (archivo existente en el path)
        blocker = tmp_path / "blocker"
        blocker.write_text("no dir")
        with pytest.raises(OutputWriteError):
            writer.write_json(output_directory=str(blocker / "sub"), filename="x.json", document={"a": 1}, indent=2, ensure_ascii=False)


class TestJsonOutputWriterAdditional:
    def test_empty_document(self, writer, tmp_path):
        sha = writer.write_json(output_directory=str(tmp_path), filename="x.json", document={}, indent=2, ensure_ascii=False)
        assert len(sha) == 64
        assert (tmp_path / "x.json").exists()

    def test_nested_document(self, writer, tmp_path):
        doc = {"a": {"b": [1, 2, {"c": None}]}}
        sha = writer.write_json(output_directory=str(tmp_path), filename="x.json", document=doc, indent=2, ensure_ascii=False)
        assert len(sha) == 64

    def test_indent_respected(self, writer, tmp_path):
        writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"a": 1}, indent=4, ensure_ascii=False)
        content = (tmp_path / "x.json").read_text(encoding="utf-8")
        assert "\n    " in content

    def test_float_preserved(self, writer, tmp_path):
        sha = writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": 0.5}, indent=2, ensure_ascii=False)
        assert len(sha) == 64
        assert "0.5" in (tmp_path / "x.json").read_text(encoding="utf-8")

    def test_int_serialized(self, writer, tmp_path):
        writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"v": 12345}, indent=2, ensure_ascii=False)
        assert "12345" in (tmp_path / "x.json").read_text(encoding="utf-8")

    def test_sha_length_exact_64(self, writer, tmp_path):
        sha = writer.write_json(output_directory=str(tmp_path), filename="x.json", document={"a": 1}, indent=2, ensure_ascii=False)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


class TestASTAudit:
    def test_infrastructure_no_ai(self):
        p = Path("app/infrastructure/storage/json_output_writer.py")
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

    def test_infrastructure_uses_only_stdlib(self):
        p = Path("app/infrastructure/storage/json_output_writer.py")
        tree = ast.parse(p.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        # Solo stdlib + app.*
        stdlib_ok = {"hashlib", "json", "pathlib", "typing", "app", "__future__"}
        assert imports <= stdlib_ok, f"imports no stdlib: {imports - stdlib_ok}"
