from __future__ import annotations

import logging
import sys
from typing import Any, Mapping

import structlog
from pydantic import SecretStr

from app.core.constants import ENGINE_VERSION


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
    if not event_dict:
        return event_dict

    redacted: dict[str, Any] = {}
    for k, v in event_dict.items():
        if _is_secret_key(str(k)):
            redacted[k] = _SENTINEL
        else:
            redacted[k] = _redact_value(v)
    return redacted  # type: ignore[return-value]


def _shared_processors(env: str) -> list[structlog.types.Processor]:
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


_INITIALIZED: bool = False
_INITIALIZED_KWARGS: dict[str, Any] = {}


def setup_logging(
    level: str = "INFO",
    fmt: str = "human",
    env: str = "dev",
) -> None:
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

    processors = _dev_processors() if norm_fmt == "human" else _prod_processors()

    renderer = processors[-1]
    non_render_processors = processors[:-1]

    structlog.configure(
        processors=non_render_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

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

    for noisy in ("httpx", "urllib3", "httpcore", "huggingface_hub", "filelock", "torch"):
        logging.getLogger(noisy).setLevel(max(lvl, logging.WARNING))

    _INITIALIZED = True
    _INITIALIZED_KWARGS = {"level": norm_level, "fmt": norm_fmt, "env": norm_env}


def get_logger(name: str = "ia-doblaje") -> structlog.stdlib.BoundLogger:
    global _INITIALIZED
    if not _INITIALIZED:
        setup_logging()
        _INITIALIZED = True
    return structlog.stdlib.get_logger(name)


def bind_context(**kwargs: Any) -> structlog.stdlib.BoundLogger:
    safe: dict[str, Any] = {}
    for k, v in kwargs.items():
        if _is_secret_key(str(k)):
            safe[k] = _SENTINEL
        else:
            safe[k] = _redact_value(v)
    structlog.contextvars.bind_contextvars(**safe)
    return get_logger()


def unbind_context(*keys: str) -> None:
    structlog.contextvars.unbind_contextvars(*keys)


def reset_context() -> None:
    structlog.contextvars.clear_contextvars()


__all__ = [
    "setup_logging",
    "get_logger",
    "bind_context",
    "unbind_context",
    "reset_context",
    "redact_secrets_processor",
]
