from __future__ import annotations

import time
from pathlib import Path


def read_mails(folder: Path) -> list[dict[str, str]]:
    mails = []
    for path in sorted(Path(folder).glob("*.txt"), key=lambda item: item.stat().st_mtime_ns):
        fields = dict(
            line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if "=" in line
        )
        fields["html"] = path.with_suffix(".html").read_text(encoding="utf-8") if path.with_suffix(".html").is_file() else ""
        mails.append(fields)
    return mails


def wait_for_mail(folder: Path, kind: str, to: str | None = None, *, count: int = 1, timeout: float = 5) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    while True:
        found = [m for m in read_mails(folder) if m.get("kind") == kind and (to is None or m.get("to") == to)]
        if len(found) >= count:
            return found[-1]
        if time.monotonic() > deadline:
            raise AssertionError(f"no llegó el correo {kind} para {to or 'nadie'} ({len(found)}/{count})")
        time.sleep(0.02)
