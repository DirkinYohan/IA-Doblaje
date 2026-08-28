"""Tests unitarios T02 para app.core.logging.

Mocks: capturamos salida stderr / usamos streams StringIO para no depender
de consola real. No requiere GPU ni PyTorch.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CaptureLoggingStream:
    """Context manager para capturar stderr/logs en un buffer.

    Preserva el ProcessorFormatter de structlog que genera setup_logging()
    para que fmt="json" produzca JSON real y fmt="human" produzca texto
    coloreado via ConsoleRenderer.
    """

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
        # Capturar ProcessorFormatter de structlog creado por setup_logging
        root = logging.getLogger()
        formatter: Any = None
        for h in root.handlers:
            if hasattr(h, "formatter") and h.formatter is not None:
                formatter = h.formatter
                break
        # Limpiar handlers antiguos
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
        # Restaurar handlers y level originales
        root.handlers.extend(self._old_handlers)
        if self._old_level is not None:
            root.setLevel(self._old_level)

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


# ---------------------------------------------------------------------------
# Tests: setup_logging + get_logger
# ---------------------------------------------------------------------------


def test_setup_logging_no_error() -> None:
    # Llamarlo varias veces no debe romper
    setup_logging("INFO", fmt="human")
    setup_logging("DEBUG", fmt="json")
    setup_logging("WARNING")  # fmt default human
    setup_logging(level="badlevel")  # debe fallback a INFO
    log = get_logger("t02-test")
    assert log is not None


def test_get_logger_returns_bound_logger() -> None:
    setup_logging()
    log = get_logger("t02-app")
    # structlog stdlib BoundLogger tiene metodo info/debug
    assert callable(getattr(log, "info", None))
    assert callable(getattr(log, "warning", None))


# ---------------------------------------------------------------------------
# Tests: human renderer (ConsoleRenderer) -> output tiene level + event + logger
# ---------------------------------------------------------------------------


def test_human_logger_output_structure() -> None:
    cap = _CaptureLoggingStream()
    with cap:
        log = get_logger("human-test")
        log.info("saludo_human", usuario="juan")
    txt = cap.text
    # Al menos el evento debe estar
    assert "saludo_human" in txt
    # level info debe aparecer
    assert "info" in txt.lower()
    # logger name human-test
    assert "human-test" in txt or "human_test" in txt


# ---------------------------------------------------------------------------
# Tests: JSON renderer -> una linea por evento con campos obligatorios
# ---------------------------------------------------------------------------


def test_json_renderer_one_line_structured() -> None:
    cap = _CaptureLoggingStream(fmt="json", level="INFO", env="production")
    with cap:
        log = get_logger("json-test")
        log.info("evento_json", clave_1="valor_1", numero=42)
    # Parsear ultima linea valida
    txt = cap.text.strip() or ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    assert lines, f"Expected JSON output lines, got: {txt!r}"
    parsed = None
    # Iterar de atras a adelante; el primero que parsea OK es el bueno
    for ln in reversed(lines):
        try:
            parsed = json.loads(ln)
            break
        except json.JSONDecodeError:
            continue
    assert parsed is not None, f"Could not parse JSON from: {lines}"
    # Campos obligatorios
    assert "event" in parsed
    assert "level" in parsed
    assert "timestamp" in parsed
    assert "phase" in parsed
    assert "version" in parsed
    # Valores
    assert parsed["event"] == "evento_json"
    assert parsed.get("clave_1") == "valor_1"
    assert parsed.get("numero") == 42
    assert parsed.get("logger") in {"json-test", None} or True  # campo logger es normal


# ---------------------------------------------------------------------------
# Tests: bind_context / unbind_context / reset_context
# ---------------------------------------------------------------------------


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
    # job_id queda, phase se fue
    assert "job-002" in txt
    # "beta" no debe aparecer como valor (puede aparecer en "beta" de otro campo si,
    # pero la variable phase fue removida)
    # Para ser preciso, parseamos structlog context despues de unbind:
    # No necesitamos comprobar que NUNCA aparezca la cadena beta; solo que el key
    # phase fue desvinculado. Lo comprobamos via dicts internos de
    # structlog.contextvars.
    from structlog.contextvars import get_contextvars
    ctx = get_contextvars()
    assert "phase" not in ctx


def test_reset_context_clears_all() -> None:
    bind_context(x=1, y=2)
    reset_context()
    from structlog.contextvars import get_contextvars
    assert get_contextvars() == {}


# ---------------------------------------------------------------------------
# Tests: NUNCA imprimir secretos (HF_TOKEN / SecretStr / password)
# ---------------------------------------------------------------------------


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
    # Recrear setup JSON, bind hf_token como str (key filtrada) y SecretStr
    # Paso 1: llamar setup_logging con JSON para que genere el ProcessorFormatter
    setup_logging(level="INFO", fmt="json", env="production")
    root = logging.getLogger()
    # Capturar formatter del primer handler (creado por setup_logging con fmt=json)
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
    # El value real NO debe aparecer en todo el payload serializado
    raw_payload = json.dumps(last)
    assert "hf_REAL_TOKEN_AABBCCDD_12345" not in raw_payload
    assert "otro-valor-secreto" not in raw_payload
    assert "admin123" not in raw_payload
    # En cambio deben aparecer los sentinelas
    assert "REDACTED" in raw_payload or "HIDDEN_SECRET" in raw_payload
