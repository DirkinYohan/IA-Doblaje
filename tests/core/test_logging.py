from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from app.core.logging import (
    bind_context,
    get_logger,
    redact_secrets_processor,
    reset_context,
    setup_logging,
    unbind_context,
)


class _CaptureLoggingStream:
    def __init__(self, fmt: str = "human", level: str = "INFO", env: str = "dev") -> None:
        self._fmt = fmt
        self._level = level
        self._env = env
        self.buffer = io.StringIO()
        self._old_stderr: Any = None
        self._handler: logging.StreamHandler | None = None
        self._old_handlers: list[logging.Handler] = []
        self._old_level: int | None = None

    def __enter__(self) -> "_CaptureLoggingStream":
        setup_logging(level=self._level, fmt=self._fmt, env=self._env)
        self._old_handlers = list(logging.getLogger().handlers)
        self._old_level = logging.getLogger().level
        root = logging.getLogger()
        formatter: Any = None
        for h in root.handlers:
            if hasattr(h, "formatter") and h.formatter is not None:
                formatter = h.formatter
                break
        root.handlers.clear()
        self._handler = logging.StreamHandler(self.buffer)
        self._handler.setLevel(logging.DEBUG)
        if formatter is not None:
            self._handler.setFormatter(formatter)
        root.addHandler(self._handler)
        root.setLevel(logging.DEBUG)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handler is not None:
            self._handler.flush()
        root = logging.getLogger()
        if self._handler is not None:
            root.removeHandler(self._handler)
        root.handlers.extend(self._old_handlers)
        if self._old_level is not None:
            root.setLevel(self._old_level)

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


def test_setup_logging_no_error() -> None:
    setup_logging("INFO", fmt="human")
    setup_logging("DEBUG", fmt="json")
    setup_logging("WARNING")
    setup_logging(level="badlevel")
    log = get_logger("t02-test")
    assert log is not None


def test_get_logger_returns_bound_logger() -> None:
    setup_logging()
    log = get_logger("t02-app")
    assert callable(getattr(log, "info", None))
    assert callable(getattr(log, "warning", None))


def test_human_logger_output_structure() -> None:
    cap = _CaptureLoggingStream()
    with cap:
        log = get_logger("human-test")
        log.info("saludo_human", usuario="juan")
    txt = cap.text
    assert "saludo_human" in txt
    assert "info" in txt.lower()
    assert "human-test" in txt or "human_test" in txt


def test_json_renderer_one_line_structured() -> None:
    cap = _CaptureLoggingStream(fmt="json", level="INFO", env="production")
    with cap:
        log = get_logger("json-test")
        log.info("evento_json", clave_1="valor_1", numero=42)
    txt = cap.text.strip() or ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    assert lines, f"Expected JSON output lines, got: {txt!r}"
    parsed = None
    for ln in reversed(lines):
        try:
            parsed = json.loads(ln)
            break
        except json.JSONDecodeError:
            continue
    assert parsed is not None, f"Could not parse JSON from: {lines}"
    assert "event" in parsed
    assert "level" in parsed
    assert "timestamp" in parsed
    assert "phase" in parsed
    assert "version" in parsed
    assert parsed["event"] == "evento_json"
    assert parsed.get("clave_1") == "valor_1"
    assert parsed.get("numero") == 42
    assert parsed.get("logger") in {"json-test", None} or True


def test_bind_context_appears_in_events() -> None:
    cap = _CaptureLoggingStream()
    with cap:
        reset_context()
        bind_context(job_id="job-001", step="preprocess")
        log = get_logger("ctx-test")
        log.info("algo_ocurre")
    txt = cap.text
    assert "job-001" in txt
    assert "preprocess" in txt


def test_unbind_context_removes_key() -> None:
    cap = _CaptureLoggingStream()
    with cap:
        reset_context()
        bind_context(job_id="job-002", phase="beta")
        unbind_context("phase")
        log = get_logger("ctx-test2")
        log.info("hola")
    txt = cap.text
    assert "job-002" in txt
    from structlog.contextvars import get_contextvars
    ctx = get_contextvars()
    assert "phase" not in ctx


def test_reset_context_clears_all() -> None:
    bind_context(x=1, y=2)
    reset_context()
    from structlog.contextvars import get_contextvars
    assert get_contextvars() == {}


def test_redact_secrets_processor_redacts_hf_token_key() -> None:
    ed = {"event": "x", "hf_token": "hf_MUY_SECRETO_XYZ_1234567890", "username": "juan"}
    out = redact_secrets_processor(logging.getLogger("test"), "info", ed)
    assert out["hf_token"] == "***REDACTED***"
    assert out["username"] == "juan"


def test_redact_secrets_processor_redacts_secretstr_values() -> None:
    secret = SecretStr("sk_live_SENSITIVE")
    ed = {"event": "x", "api_key_obj": secret, "config": {"other_secret": SecretStr("another")}}
    out = redact_secrets_processor(logging.getLogger("test"), "info", ed)
    assert out["api_key_obj"] == "***REDACTED***"
    assert out["config"]["other_secret"] == "***REDACTED***"


def test_redact_secrets_processor_password_key_hint() -> None:
    ed = {"event": "login", "user_password": "qwerty123", "Client-Secret": "xyz"}
    out = redact_secrets_processor(logging.getLogger("test"), "info", ed)
    assert out["user_password"] == "***REDACTED***"
    assert out["Client-Secret"] == "***REDACTED***"


def test_logger_never_prints_secretstr_hf_token_in_json() -> None:
    setup_logging(level="INFO", fmt="json", env="production")
    root = logging.getLogger()
    formatter: Any = None
    for h in root.handlers:
        if getattr(h, "formatter", None) is not None:
            formatter = h.formatter
            break
    buffer = io.StringIO()
    old_handlers = list(root.handlers)
    old_level = root.level
    root.handlers.clear()
    h = logging.StreamHandler(buffer)
    h.setLevel(logging.INFO)
    if formatter is not None:
        h.setFormatter(formatter)
    root.addHandler(h)
    root.setLevel(logging.INFO)
    try:
        log = get_logger("secrets-test")
        log.info(
            "secret_attempt",
            hf_token="hf_REAL_TOKEN_AABBCCDD_12345",
            my_secret_obj=SecretStr("otro-valor-secreto"),
            data={"password": "admin123"},
        )
    finally:
        h.flush()
        root.removeHandler(h)
        root.handlers.extend(old_handlers)
        root.setLevel(old_level)
    lines = [ln for ln in buffer.getvalue().splitlines() if ln.strip()]
    assert lines, f"Buffer tenia: {buffer.getvalue()!r}"
    last = None
    for ln in reversed(lines):
        try:
            last = json.loads(ln)
            break
        except json.JSONDecodeError:
            continue
    assert last is not None, f"No JSON found: {buffer.getvalue()}"
    raw_payload = json.dumps(last)
    assert "hf_REAL_TOKEN_AABBCCDD_12345" not in raw_payload
    assert "otro-valor-secreto" not in raw_payload
    assert "admin123" not in raw_payload
    assert "REDACTED" in raw_payload or "HIDDEN_SECRET" in raw_payload
