"""python -m app.presentation.api"""
from __future__ import annotations

import uvicorn

from app.presentation.api.main import create_app


def main() -> None:
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
