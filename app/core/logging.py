"""Logging estructurado mediante structlog (T02.3 implementacion completa).

Funcionalidad PUBLICA:
    setup_logging(level, fmt, env)
    get_logger(name)
    bind_context(**kwargs)
    unbind_context(*keys)
    reset_context()

REGLAS CRITICAS:
  - NUNCA imprimir HF_TOKEN / SecretStr / credenciales / tokens / passwords.
  - Modo dev  = ConsoleRenderer colorizado + legible.
  - Modo prod = JSONRenderer NDJSON una linea por evento.
  - Bridge bidireccional: logging stdlib Python <-> structlog.
  - Context global via structlog.contextvars (thread/greenlet safe).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Mapping

import structlog
from pydantic import SecretStr

from app.core.constants import ENGINE_VERSION


# =============================================================================
# Filter: NUNCA loggear secretos / tokens / passwords / credenciales
# =============================================================================

_SECRET_KEY_HINTS: tuple[str, ...] = (
    "token",
    "password",
    "passwd",
    "pwd",
    "secret",
    "credential",
    "apikey",
    "api_key",
    "private_key",
    "hf_token",
    "hftoken",
    "access_token",
    "refresh_token",
    "client_secret",
)

_SENTINEL = "***REDACTED***"


def _is_secret_key(key: str) -> bool:
    k = str(key).lower().replace("_", "").replace("-", "")
    for hint in _SECRET_KEY_HINTS:
        if hint.lower().replace("_", "") in k:
            return True
    return False


def _redact_value(value: Any) -> Any:
    if isinstance(value, SecretStr):
        return _SENTINEL
    if isinstance(value, (bytes, bytearray)):
        return _SENTINEL
    if isinstance(value, Mapping):
        # Recursivo: para cada key del dict anidado, CHEQUEAR si key es secret key tambien!
        out: dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(str(k)):
                out[k] = _SENTINEL
            else:
                out[k] = _redact_value(v)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        t = type(value)
        return t(_redact_value(v) for v in value)
    return value


def redact_secrets_processor(
    logger: logging.Logger, method_name: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """Processor structlog: elimina secretos de keys sospechosas + SecretStr.

    Safe: No elimina keys no relacionadas. Preserva valores normales.
    Compatible con dumps anidados (dict/list/tuple).
    """
    if not event_dict:
        return event_dict

    redacted: dict[str, Any] = {}
    for k, v in event_dict.items():
        if _is_secret_key(str(k)):
            redacted[k] = _SENTINEL
        else:
            redacted[k] = _redact_value(v)
    return redacted  # type: ignore[return-value]


# =============================================================================
# Configuracion de processors / renderers
# =============================================================================

def _shared_processors(env: str) -> list[structlog.types.Processor]:
    """Processors comunes a DEV y PROD."""
    procs: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        redact_secrets_processor,
    ]
    # Siempre inyectamos phase=ia_doblaje_engine + version como contexto base
    procs.insert(
        1,
        lambda l, m, ed: {  # noqa: E731
            **ed,
            "phase": env,
            "version": ENGINE_VERSION,
        },
    )
    return procs


def _dev_processors() -> list[structlog.types.Processor]:
    return [
        *_shared_processors(env="development"),
        structlog.dev.set_exc_info,
        structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ),
    ]


def _prod_processors() -> list[structlog.types.Processor]:
    return [
        *_shared_processors(env="production"),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(serializer=_json_dumps_safe),
    ]


def _json_dumps_safe(obj: Any, **kw: Any) -> str:
    """Wrapper json.dumps con fallback por si structlog recibe no serializable."""
    import json

    default_kw: dict[str, Any] = {
        "ensure_ascii": False,
        "separators": (",", ":"),
        "default": lambda o: (
            str(o)
            if not isinstance(o, (bytes, bytearray))
            else _SENTINEL
        ),
    }
    default_kw.update(kw)
    return json.dumps(obj, **default_kw)


# =============================================================================
# API Publica
# =============================================================================

_INITIALIZED: bool = False
_INITIALIZED_KWARGS: dict[str, Any] = {}


def setup_logging(
    level: str = "INFO",
    fmt: str = "human",
    env: str = "dev",
) -> None:
    """Inicializa logging estructurado.

    Args:
        level:  DEBUG | INFO | WARNING | ERROR | CRITICAL (case insensitive).
        fmt:    "human" (dev ConsoleRenderer color) | "json" (prod NDJSON).
        env:    "dev" | "staging" | "production" (etiqueta phase inyectada).
    """
    global _INITIALIZED, _INITIALIZED_KWARGS

    norm_level = str(level).upper()
    norm_fmt = str(fmt).lower().strip()
    norm_env = str(env).lower().strip()

    if norm_fmt not in {"human", "json"}:
        norm_fmt = "human"
    if norm_env not in {"dev", "staging", "production"}:
        norm_env = "dev"
    try:
        lvl = int(getattr(logging, norm_level, logging.INFO))
    except (TypeError, AttributeError):
        lvl = logging.INFO

    # --- structlog configure ---
    processors = _dev_processors() if norm_fmt == "human" else _prod_processors()

    # Ultimo paso para stdlib bridge: si processor final no es renderer, error.
    # Para stdlib handler, necesitamos separar renderer para ProcessingFormatter.
    renderer = processors[-1]
    non_render_processors = processors[:-1]

    structlog.configure(
        processors=non_render_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # --- stdlib logging root handler ---
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=non_render_processors,
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.setLevel(lvl)
    root.addHandler(handler)
    root.setLevel(lvl)

    # Silenciar ruidos de terceros a WARNING+
    for noisy in ("httpx", "urllib3", "httpcore", "huggingface_hub", "filelock", "torch"):
        logging.getLogger(noisy).setLevel(max(lvl, logging.WARNING))

    _INITIALIZED = True
    _INITIALIZED_KWARGS = {"level": norm_level, "fmt": norm_fmt, "env": norm_env}


def get_logger(name: str = "ia-doblaje") -> structlog.stdlib.BoundLogger:
    """Retorna BoundLogger de structlog. Si no se inicializo, setup default."""
    global _INITIALIZED
    if not _INITIALIZED:
        setup_logging()
        _INITIALIZED = True
    return structlog.stdlib.get_logger(name)


def bind_context(**kwargs: Any) -> structlog.stdlib.BoundLogger:
    """Agrega variables de contexto GLOBALES (thread/greenlet safe)."""
    # Filtrar secretos en contexto para evitar leak en logs futuros.
    safe: dict[str, Any] = {}
    for k, v in kwargs.items():
        if _is_secret_key(str(k)):
            safe[k] = _SENTINEL
        else:
            safe[k] = _redact_value(v)
    structlog.contextvars.bind_contextvars(**safe)
    return get_logger()


def unbind_context(*keys: str) -> None:
    """Remueve keys del contexto global."""
    structlog.contextvars.unbind_contextvars(*keys)


def reset_context() -> None:
    """Limpia TODO el contexto global (usar entre jobs distintos)."""
    structlog.contextvars.clear_contextvars()


# =============================================================================
# Export
# =============================================================================

__all__ = [
    "setup_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "reset_context",
    "redact_secrets_processor",
]
