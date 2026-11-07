"""Hilos de cómputo según las CPU que el proceso puede usar de verdad.

En un contenedor, os.cpu_count() devuelve las CPU del anfitrión aunque el
contenedor solo tenga permitidas 2. Si Whisper o CTranslate2 lanzan un hilo por
cada CPU visible, el sistema los frena y todo va el doble de lento.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP_V2 = Path("/sys/fs/cgroup/cpu.max")
CGROUP_V1_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
CGROUP_V1_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def cgroup_cpus(
    v2: Path = CGROUP_V2, v1_quota: Path = CGROUP_V1_QUOTA, v1_period: Path = CGROUP_V1_PERIOD
) -> int | None:
    try:
        if v2.is_file():
            quota, period = v2.read_text().split()[:2]
            if quota == "max":
                return None
            return max(1, math.ceil(int(quota) / int(period)))
        if v1_quota.is_file() and v1_period.is_file():
            quota_us = int(v1_quota.read_text().strip())
            if quota_us <= 0:
                return None
            return max(1, math.ceil(quota_us / int(v1_period.read_text().strip())))
    except (OSError, ValueError):
        return None
    return None


def thread_count() -> int:
    """0 deja el valor por defecto de la librería (sin límite detectado)."""
    forced = os.environ.get("IA_CPU_THREADS", "").strip()
    if forced.isdigit():
        return int(forced)
    return cgroup_cpus() or 0
