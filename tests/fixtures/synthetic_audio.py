from __future__ import annotations


def get_fixture_clip_path(duration_label: str) -> str:
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "integration" / "fixtures_clips"
    return str(p / f"clip_{duration_label}.wav")
