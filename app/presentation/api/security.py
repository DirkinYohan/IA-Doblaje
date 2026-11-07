from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict
from functools import lru_cache


PBKDF2_ITERATIONS = 600_000
LEGACY_ITERATIONS = 200_000
_SCHEME = "pbkdf2_sha256"


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def _parse(stored: str) -> tuple[int, bytes, bytes]:
    parts = stored.split("$")
    if len(parts) == 4 and parts[0] == _SCHEME:
        return int(parts[1]), bytes.fromhex(parts[2]), bytes.fromhex(parts[3])
    if len(parts) == 2:
        return LEGACY_ITERATIONS, bytes.fromhex(parts[0]), bytes.fromhex(parts[1])
    raise ValueError("formato de hash desconocido")


def verify_password(password: str, stored: str) -> bool:
    try:
        iterations, salt, expected = _parse(stored)
    except ValueError:
        return False
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(got, expected)


def needs_rehash(stored: str) -> bool:
    try:
        iterations, _salt, _digest = _parse(stored)
    except ValueError:
        return True
    return iterations < PBKDF2_ITERATIONS


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


def burn_password_check(password: str) -> None:
    verify_password(password, _dummy_hash())


def cookie_secure() -> bool:
    forced = os.environ.get("IA_COOKIE_SECURE", "").strip().lower()
    if forced in {"1", "true", "yes"}:
        return True
    if forced in {"0", "false", "no"}:
        return False
    return os.environ.get("APP_PUBLIC_URL", "").strip().lower().startswith("https://")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(32)


class RateLimiter:
    def __init__(self, limit: int, window_sec: int) -> None:
        self.limit = limit
        self.window_sec = window_sec
        self._hits: dict[str, list[float]] = defaultdict(list)

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = [t for t in self._hits[key] if now - t < self.window_sec]
        if len(bucket) >= self.limit:
            self._hits[key] = bucket
            return False
        bucket.append(now)
        self._hits[key] = bucket
        return True
