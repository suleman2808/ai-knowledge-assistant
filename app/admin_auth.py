"""Password protection for the admin dashboard.

The dashboard shows names, phone numbers, complaints and transcripts.
That is the material this project has been careful about everywhere else,
so it does not get served to whoever finds the URL.

Deliberately a single shared password rather than user accounts. The
dashboard has one audience — whoever runs the clinic — and a login
system with registration, password reset and roles would be more code
than the thing it protects. When a second kind of user appears, this is
the wrong design and should be replaced rather than extended.

What it does do properly:

- **Constant-time comparison**, so the password cannot be recovered by
  timing how long a wrong guess takes.
- **A signed cookie**, not the password in a cookie. The signature is an
  HMAC over the expiry, keyed on the password, so a cookie cannot be
  forged without it and cannot be replayed after it expires.
- **HttpOnly**, so a script on the page cannot read the session.
- **Disabled by default.** With no password configured the dashboard
  refuses to serve rather than opening to everyone, because the failure
  everyone regrets is the one where protection was never switched on.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)

COOKIE_NAME = "admin_session"
# How long a login lasts. Long enough to work through a dashboard, short
# enough that a borrowed laptop stops being a problem by morning.
SESSION_SECONDS = 12 * 60 * 60


def is_enabled() -> bool:
    """Whether an admin password has been configured at all."""
    return bool(settings.admin_password.strip())


def _secret() -> bytes:
    """Signing key, derived from the password.

    Derived rather than used directly so the cookie never contains
    anything from which the password could be reconstructed, even if the
    hash function were later found to leak.
    """
    return hashlib.sha256(
        b"riverbend-admin-session|" + settings.admin_password.encode()
    ).digest()


def check_password(attempt: str) -> bool:
    """Verify a password attempt in constant time."""
    if not is_enabled():
        return False
    return hmac.compare_digest(attempt.encode(), settings.admin_password.encode())


def issue_session() -> str:
    """Create a signed session token."""
    expires = str(int(time.time()) + SESSION_SECONDS)
    signature = hmac.new(_secret(), expires.encode(), hashlib.sha256).hexdigest()
    token = f"{expires}.{signature}"
    return base64.urlsafe_b64encode(token.encode()).decode()


def is_valid_session(cookie: str | None) -> bool:
    """Whether a cookie is a session this server issued and still honours."""
    if not cookie or not is_enabled():
        return False

    try:
        token = base64.urlsafe_b64decode(cookie.encode()).decode()
        expires, signature = token.split(".", 1)
        expected = hmac.new(_secret(), expires.encode(), hashlib.sha256).hexdigest()
    except Exception:
        # Malformed cookies are simply not sessions. They arrive from old
        # deployments, other sites on the same host, and bots.
        return False

    if not hmac.compare_digest(signature, expected):
        return False
    return int(expires) > time.time()
