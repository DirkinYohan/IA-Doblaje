from __future__ import annotations

import html
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger("ia.mail")

BREVO_URL = "https://api.brevo.com/v3/smtp/email"
DEFAULT_PUBLIC_URL = "http://localhost:3000"


@dataclass(frozen=True)
class Mail:
    to: str
    name: str
    subject: str
    html: str
    text: str
    kind: str
    link: str
    token: str
    expires: str


class Mailer(Protocol):
    def send(self, mail: Mail) -> None: ...


class FileMailer:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def send(self, mail: Mail) -> None:
        from app.presentation.api.security import token_hash

        self.folder.mkdir(parents=True, exist_ok=True)
        stem = token_hash(mail.token)
        (self.folder / f"{stem}.txt").write_text(
            f"kind={mail.kind}\nto={mail.to}\nsubject={mail.subject}\nlink={mail.link}\n"
            f"token={mail.token}\nexpires={mail.expires}\n",
            encoding="utf-8",
        )
        (self.folder / f"{stem}.html").write_text(mail.html, encoding="utf-8")
        log.info("correo (%s) para %s guardado en %s", mail.kind, mail.to, self.folder)


class BrevoMailer:
    def __init__(self, api_key: str, sender_email: str, sender_name: str, *, url: str = BREVO_URL) -> None:
        self.api_key = api_key
        self.sender_email = sender_email
        self.sender_name = sender_name
        self.url = url

    def send(self, mail: Mail) -> None:
        import httpx

        payload = {
            "sender": {"name": self.sender_name, "email": self.sender_email},
            "to": [{"email": mail.to, "name": mail.name or mail.to}],
            "subject": mail.subject,
            "htmlContent": mail.html,
            "textContent": mail.text,
            "tags": [mail.kind],
        }
        response = httpx.post(
            self.url,
            json=payload,
            headers={"api-key": self.api_key, "accept": "application/json"},
            timeout=20,
        )
        if response.status_code >= 300:
            raise RuntimeError(f"Brevo respondió {response.status_code}: {response.text[:300]}")
        log.info("correo (%s) enviado a %s por Brevo", mail.kind, mail.to)


def from_env(mail_dir: Path) -> Mailer:
    key = os.environ.get("BREVO_API_KEY", "").strip()
    if not key:
        return FileMailer(mail_dir)
    sender = os.environ.get("MAIL_FROM_EMAIL", "").strip()
    if not sender:
        raise RuntimeError("Falta MAIL_FROM_EMAIL: el remitente verificado en Brevo")
    return BrevoMailer(key, sender, os.environ.get("MAIL_FROM_NAME", "").strip() or "IA Doblaje")


def public_url() -> str:
    return (os.environ.get("APP_PUBLIC_URL", "").strip() or DEFAULT_PUBLIC_URL).rstrip("/")


def send_in_background(mailer: Mailer, mail: Mail) -> threading.Thread:
    def run() -> None:
        try:
            mailer.send(mail)
        except Exception as exc:  # noqa: BLE001 - un fallo de envío no tumba la API
            log.warning("no se pudo enviar el correo (%s) a %s: %s", mail.kind, mail.to, exc)

    thread = threading.Thread(target=run, name=f"mail-{mail.kind}", daemon=True)
    thread.start()
    return thread


_TEXTS = {
    "verify": {
        "es": {
            "subject": "Confirma tu correo en IA Doblaje",
            "title": "Confirma tu correo",
            "body": "Hola {name}, gracias por crear tu cuenta. Confirma que este correo es tuyo para terminar.",
            "button": "Confirmar correo",
            "expires": "El enlace caduca en 48 horas.",
            "ignore": "Si no creaste una cuenta en IA Doblaje, ignora este mensaje.",
        },
        "en": {
            "subject": "Confirm your email for IA Doblaje",
            "title": "Confirm your email",
            "body": "Hi {name}, thanks for signing up. Confirm this email address is yours to finish.",
            "button": "Confirm email",
            "expires": "The link expires in 48 hours.",
            "ignore": "If you didn't create an IA Doblaje account, ignore this message.",
        },
    },
    "reset": {
        "es": {
            "subject": "Recupera tu contraseña de IA Doblaje",
            "title": "Restablece tu contraseña",
            "body": "Hola {name}, recibimos una solicitud para cambiar la contraseña de tu cuenta.",
            "button": "Crear una contraseña nueva",
            "expires": "El enlace caduca en 1 hora y sólo sirve una vez.",
            "ignore": "Si no lo pediste tú, ignora este mensaje: tu contraseña no cambia.",
        },
        "en": {
            "subject": "Reset your IA Doblaje password",
            "title": "Reset your password",
            "body": "Hi {name}, we received a request to change your account password.",
            "button": "Choose a new password",
            "expires": "The link expires in 1 hour and works only once.",
            "ignore": "If you didn't ask for this, ignore this message: your password stays the same.",
        },
    },
}

_FALLBACK = {"es": "Si el botón no funciona, copia este enlace en tu navegador:", "en": "If the button doesn't work, paste this link into your browser:"}


def locale_of(value: object) -> str:
    return "en" if str(value or "").lower().startswith("en") else "es"


def compose(kind: str, *, to: str, name: str, link: str, token: str, expires: str, locale: str = "es") -> Mail:
    lang = locale_of(locale)
    copy = _TEXTS[kind][lang]
    first = (name or "").split(" ")[0] or to
    body = copy["body"].format(name=first)
    safe_link = html.escape(link, quote=True)
    page = f"""<!doctype html>
<html lang="{lang}">
<body style="margin:0;background:#f4f4f7;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1d1d26">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f7;padding:32px 16px">
    <tr><td align="center">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:520px;background:#ffffff;border-radius:16px;padding:36px 32px">
        <tr><td style="font-size:14px;font-weight:700;color:#5b4fe0;letter-spacing:.02em">IA Doblaje</td></tr>
        <tr><td style="padding-top:20px;font-size:22px;font-weight:700;line-height:1.3">{html.escape(copy["title"])}</td></tr>
        <tr><td style="padding-top:12px;font-size:15px;line-height:1.6;color:#4a4a57">{html.escape(body)}</td></tr>
        <tr><td style="padding-top:28px">
          <a href="{safe_link}" style="display:inline-block;background:#5b4fe0;color:#ffffff;text-decoration:none;font-weight:600;font-size:15px;padding:12px 22px;border-radius:10px">{html.escape(copy["button"])}</a>
        </td></tr>
        <tr><td style="padding-top:24px;font-size:13px;line-height:1.6;color:#6b6b78">{html.escape(copy["expires"])}</td></tr>
        <tr><td style="padding-top:16px;font-size:12px;line-height:1.6;color:#8a8a96">{html.escape(_FALLBACK[lang])}<br>
          <a href="{safe_link}" style="color:#5b4fe0;word-break:break-all">{safe_link}</a></td></tr>
        <tr><td style="padding-top:24px;border-top:1px solid #ececf1;margin-top:24px;font-size:12px;line-height:1.6;color:#8a8a96">{html.escape(copy["ignore"])}</td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
    text = "\n\n".join([copy["title"], body, f"{copy['button']}: {link}", copy["expires"], copy["ignore"]])
    return Mail(
        to=to, name=name, subject=copy["subject"], html=page, text=text,
        kind=kind, link=link, token=token, expires=expires,
    )


__all__ = [
    "BrevoMailer", "FileMailer", "Mail", "Mailer", "compose", "from_env", "locale_of",
    "public_url", "send_in_background",
]
