from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MAX_EMAIL = 254
MAX_PASSWORD = 1024
MAX_TOKEN = 512
MAX_TEXT = 200
MAX_CUES = 50_000

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_email(email: str) -> bool:
    return len(email) <= MAX_EMAIL and bool(_EMAIL.match(email))


class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


class RegisterBody(_Body):
    name: str = Field("", max_length=MAX_TEXT)
    email: str = Field("", max_length=MAX_EMAIL)
    password: str = Field("", max_length=MAX_PASSWORD)
    confirm: str = Field("", max_length=MAX_PASSWORD)
    locale: str = Field("", max_length=16)


class LoginBody(_Body):
    email: str = Field("", max_length=MAX_EMAIL)
    password: str = Field("", max_length=MAX_PASSWORD)
    remember: bool = False


class EmailBody(_Body):
    email: str = Field("", max_length=MAX_EMAIL)
    locale: str = Field("", max_length=16)


class TokenBody(_Body):
    token: str = Field("", max_length=MAX_TOKEN)


class ResetPasswordBody(_Body):
    token: str = Field("", max_length=MAX_TOKEN)
    password: str = Field("", max_length=MAX_PASSWORD)
    confirm: str = Field("", max_length=MAX_PASSWORD)


class ChangeEmailBody(_Body):
    email: str = Field("", max_length=MAX_EMAIL)
    password: str = Field("", max_length=MAX_PASSWORD)
    locale: str = Field("", max_length=16)


class ChangePasswordBody(_Body):
    current: str = Field("", max_length=MAX_PASSWORD)
    password: str = Field("", max_length=MAX_PASSWORD)
    confirm: str = Field("", max_length=MAX_PASSWORD)


class UpdateMeBody(_Body):
    name: str | None = Field(None, max_length=MAX_TEXT)


class RenameBody(_Body):
    title: str = Field("", max_length=MAX_TEXT)


class CreateJobBody(_Body):
    project_id: str = Field("", max_length=MAX_TOKEN)
    mode: str = Field("progressive", max_length=32)


class ProduceBody(_Body):
    mode: str = Field("progressive", max_length=32)


class SubtitlesBody(_Body):
    language: str = Field("", max_length=16)
    cues: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_CUES)


class RenderBody(_Body):
    lang: str = Field("", max_length=16)
