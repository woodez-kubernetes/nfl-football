"""Email the HTML report.

Credentials come from the environment and are never written to disk, logged, or
echoed — ``SmtpConfig.__repr__`` redacts the password deliberately.

Gmail requires an **App Password** (a 16-character token generated at
https://myaccount.google.com/apppasswords with 2FA enabled); a normal account
password will be rejected. Configure with:

    export SMTP_HOST=smtp.gmail.com
    export SMTP_PORT=587
    export SMTP_USER=you@gmail.com
    export SMTP_PASSWORD='xxxx xxxx xxxx xxxx'   # App Password
    export SMTP_FROM=you@gmail.com               # optional, defaults to SMTP_USER

Then ``--email``. Without ``--email`` nothing is ever sent.
"""

from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

from nfl_football import config


class MailError(RuntimeError):
    """Raised when the report cannot be sent."""


@dataclass
class SmtpConfig:
    host: str = ""
    port: int = 587
    user: str = ""
    password: str = ""
    sender: str = ""
    use_starttls: bool = True
    use_ssl: bool = False

    @classmethod
    def from_env(cls, env: dict | None = None) -> "SmtpConfig":
        env = env if env is not None else os.environ
        port = int(env.get("SMTP_PORT", "587") or 587)
        return cls(
            host=env.get("SMTP_HOST", ""),
            port=port,
            user=env.get("SMTP_USER", ""),
            password=env.get("SMTP_PASSWORD", ""),
            sender=env.get("SMTP_FROM") or env.get("SMTP_USER", ""),
            # Port 465 is implicit TLS; 587 is STARTTLS.
            use_ssl=env.get("SMTP_SSL", "").lower() in {"1", "true", "yes"} or port == 465,
            use_starttls=env.get("SMTP_STARTTLS", "true").lower() not in {"0", "false", "no"},
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.host and self.user and self.password)

    def missing(self) -> list[str]:
        names = {"SMTP_HOST": self.host, "SMTP_USER": self.user,
                 "SMTP_PASSWORD": self.password}
        return [key for key, value in names.items() if not value]

    def __repr__(self) -> str:  # never leak the credential
        shown = "***set***" if self.password else "***unset***"
        return (f"SmtpConfig(host={self.host!r}, port={self.port}, "
                f"user={self.user!r}, password={shown}, sender={self.sender!r})")


#: Gmail clips message bodies larger than roughly this and hides the remainder
#: behind a "View entire message" link, which breaks the layout mid-report.
GMAIL_CLIP_BYTES = 102_000


def build_message(
    *,
    html_body: str,
    text_body: str,
    subject: str,
    to: str,
    sender: str,
    attachments: list[Path] | None = None,
) -> EmailMessage:
    """A multipart/alternative message: plain text for clients that want it,
    the email-safe styled report as HTML, plus any attachments."""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = to
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain="nfl-thesis.local")
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    for path in attachments or []:
        path = Path(path)
        if not path.exists():
            continue
        message.add_attachment(
            path.read_bytes(),
            maintype="text",
            subtype="html",
            filename=path.name,
        )
    return message


def clipping_warning(html_body: str) -> str | None:
    """Warn when the HTML body is large enough for Gmail to clip it."""
    size = len(html_body.encode("utf-8"))
    if size <= GMAIL_CLIP_BYTES:
        return None
    return (
        f"email body is {size:,} bytes; Gmail clips above ~{GMAIL_CLIP_BYTES:,} "
        "and will hide the end behind 'View entire message'"
    )


def send(message: EmailMessage, smtp: SmtpConfig) -> None:
    if not smtp.is_configured:
        raise MailError(
            "SMTP is not configured — missing " + ", ".join(smtp.missing())
            + ". See nfl_football/mailer.py for setup."
        )
    try:
        if smtp.use_ssl:
            server = smtplib.SMTP_SSL(smtp.host, smtp.port, timeout=30)
        else:
            server = smtplib.SMTP(smtp.host, smtp.port, timeout=30)
        with server:
            server.ehlo()
            if smtp.use_starttls and not smtp.use_ssl:
                server.starttls()
                server.ehlo()
            server.login(smtp.user, smtp.password)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        raise MailError(
            "SMTP authentication failed. For Gmail this usually means an App "
            f"Password is required rather than the account password ({exc.smtp_code})."
        ) from exc
    except (smtplib.SMTPException, OSError) as exc:
        raise MailError(f"could not send report: {exc}") from exc


def subject_for(season: int, week: int, upsets: int, total: int) -> str:
    games = "game" if total == 1 else "games"
    return f"NFL {season} Week {week} — {total} {games}, {upsets} against the market"


def save_eml(message: EmailMessage, path: str | Path) -> Path:
    """Write the message to disk instead of sending it.

    Used by ``--email-dry-run`` so the exact bytes can be inspected, and the
    setup verified, without a live SMTP connection or any credential.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(message))
    return path


DEFAULT_RECIPIENT = config.EMAIL_TO
