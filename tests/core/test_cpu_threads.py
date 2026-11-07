from __future__ import annotations

from pathlib import Path

import pytest

from app.core import cpu


def test_cgroup_v2_redondea_hacia_arriba(tmp_path: Path) -> None:
    v2 = tmp_path / "cpu.max"
    v2.write_text("200000 100000\n")
    assert cpu.cgroup_cpus(v2, tmp_path / "x", tmp_path / "y") == 2
    v2.write_text("150000 100000\n")
    assert cpu.cgroup_cpus(v2, tmp_path / "x", tmp_path / "y") == 2


def test_cgroup_sin_limite(tmp_path: Path) -> None:
    v2 = tmp_path / "cpu.max"
    v2.write_text("max 100000\n")
    assert cpu.cgroup_cpus(v2, tmp_path / "x", tmp_path / "y") is None
    assert cpu.cgroup_cpus(tmp_path / "no", tmp_path / "x", tmp_path / "y") is None


def test_cgroup_v1(tmp_path: Path) -> None:
    quota, period = tmp_path / "q", tmp_path / "p"
    quota.write_text("400000")
    period.write_text("100000")
    assert cpu.cgroup_cpus(tmp_path / "no", quota, period) == 4
    quota.write_text("-1")
    assert cpu.cgroup_cpus(tmp_path / "no", quota, period) is None


def test_variable_de_entorno_manda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IA_CPU_THREADS", "3")
    assert cpu.thread_count() == 3
