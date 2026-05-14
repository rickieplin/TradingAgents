"""Read and refresh ChatGPT-subscription OAuth tokens from `~/.codex/auth.json`.

The Codex CLI maintains a long-lived OAuth session under ``~/.codex/auth.json``
after the user runs ``codex login``. This module exposes that session so
TradingAgents can talk to the ChatGPT backend directly — without going
through ``codex exec`` — for full ``tool_calls`` and structured-output
support.

The endpoint we hit (``https://chatgpt.com/backend-api/codex/responses``)
is the OpenAI Responses-API variant the Codex CLI itself uses. It accepts
the standard tool-spec / Pydantic-schema flow exactly like the Platform
API, so the rest of TradingAgents doesn't need to know whether the deep
slot is on a paid API key or a ChatGPT subscription.

Auth-file shape (OAuth login):

    {
      "OPENAI_API_KEY": null,                  # legacy api-key login slot
      "tokens": {
        "id_token":      "<JWT>",
        "access_token":  "<JWT>",
        "refresh_token": "<opaque>",
        "account_id":    "<chatgpt account id>"
      },
      "last_refresh": "<ISO 8601 timestamp>"
    }

The same file is consulted by ``codex`` itself; we rewrite it after a
refresh so both clients see the rotated tokens.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# OpenAI's public OAuth client ID used by the Codex CLI. Refreshing
# against a different client ID rejects the user's existing refresh
# token, so this must stay in lockstep with the CLI's value.
_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"

# JWT exp claims are issued in seconds and Codex rotates well before
# expiry, but we refresh proactively at 5 minutes to handle clock skew
# and avoid mid-request 401s.
_REFRESH_BUFFER_SECONDS = 300

# Cross-thread guard so two concurrent agents don't race on the same
# refresh. The Codex CLI uses a file lock for the same reason but we
# stay in-process here — the TradingAgents graph runs in one process.
_refresh_lock = threading.Lock()


def _auth_path() -> Path:
    """Resolve the Codex auth file location.

    Honours ``CODEX_HOME`` so users with a non-default install (or our
    own tests using a tmp dir) work without code changes; falls back to
    the documented ``~/.codex/auth.json`` default.
    """
    home = os.environ.get("CODEX_HOME")
    if home:
        return Path(home) / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def _decode_jwt_payload(token: str) -> dict:
    """Decode a JWT's payload segment without verifying the signature.

    Signature verification requires OpenAI's JWKS and is not necessary
    here — the token is consumed by the same OpenAI backend that signed
    it. We only inspect the payload to read ``exp`` for refresh timing.
    """
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("malformed JWT: expected at least 2 segments")
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(payload_b64))


class CodexAuthError(RuntimeError):
    """Raised when Codex credentials are absent, malformed, or unrefreshable."""


class CodexCredentials:
    """In-memory wrapper around ``~/.codex/auth.json`` with refresh support.

    Construct via :func:`load`. The object caches the parsed file and
    refreshes the access token on demand (proactively, ~5 min before
    expiry) by calling OpenAI's OAuth token endpoint with the stored
    refresh token. Rotated tokens are written back so the Codex CLI
    sees the same fresh credentials on its next run.
    """

    def __init__(self, path: Path, data: dict):
        self._path = path
        self._data = data

    @property
    def account_id(self) -> str:
        account = self._tokens.get("account_id")
        if not account:
            raise CodexAuthError(
                "auth.json has no account_id — did `codex login` complete "
                "with an OAuth (ChatGPT) session? Plain API-key logins do "
                "not populate this field."
            )
        return account

    @property
    def _tokens(self) -> dict:
        tokens = self._data.get("tokens")
        if not isinstance(tokens, dict):
            raise CodexAuthError(
                "auth.json is missing the OAuth `tokens` block. Run "
                "`codex login` (and pick the ChatGPT-subscription option, "
                "not --with-api-key) to populate it."
            )
        return tokens

    def access_token(self) -> str:
        """Return a valid access token, refreshing in-place if it's near expiry."""
        token = self._tokens.get("access_token")
        if not token:
            raise CodexAuthError("auth.json has no access_token")

        if self._is_expired(token):
            with _refresh_lock:
                # Re-check under the lock so concurrent callers don't
                # both burn a refresh round-trip.
                token = self._tokens.get("access_token")
                if token and not self._is_expired(token):
                    return token
                self._refresh()
                token = self._tokens["access_token"]

        return token

    @staticmethod
    def _is_expired(token: str) -> bool:
        """``True`` if ``exp`` is within the refresh buffer (or missing)."""
        try:
            payload = _decode_jwt_payload(token)
        except (ValueError, json.JSONDecodeError):
            # Treat unparseable tokens as expired so the refresh path
            # surfaces the underlying problem rather than 401-looping.
            return True
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)):
            return True
        return time.time() >= (exp - _REFRESH_BUFFER_SECONDS)

    def _refresh(self) -> None:
        refresh_token = self._tokens.get("refresh_token")
        if not refresh_token:
            raise CodexAuthError(
                "auth.json has no refresh_token — re-run `codex login` "
                "to renew the OAuth session."
            )

        logger.info("Refreshing Codex/ChatGPT access token")
        try:
            resp = httpx.post(
                _TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": _OAUTH_CLIENT_ID,
                    "refresh_token": refresh_token,
                    "scope": "openid profile email",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            raise CodexAuthError(f"token refresh request failed: {exc}") from exc

        if resp.status_code != 200:
            raise CodexAuthError(
                f"token refresh returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        body = resp.json()
        new_access = body.get("access_token")
        if not new_access:
            raise CodexAuthError(
                f"token refresh response missing access_token: {body!r}"
            )

        self._tokens["access_token"] = new_access
        # OpenAI rotates refresh tokens on every refresh; persist the
        # new one so subsequent refreshes (this process or codex CLI)
        # keep working.
        if body.get("refresh_token"):
            self._tokens["refresh_token"] = body["refresh_token"]
        if body.get("id_token"):
            self._tokens["id_token"] = body["id_token"]
        self._data["last_refresh"] = datetime.now(timezone.utc).isoformat()

        self._persist()

    def _persist(self) -> None:
        """Write the in-memory state back to ``auth.json`` atomically.

        Codex itself reads this file on every invocation; an interrupted
        write would lock the user out, so we go via a same-directory
        temp file + ``os.replace`` for atomic rename.
        """
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp_path.write_text(json.dumps(self._data, indent=2))
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            # Best-effort: even if we can't persist, the in-memory token
            # is still valid for this process. Log so the user sees the
            # next codex invocation will re-refresh unnecessarily.
            logger.warning(
                "Could not persist refreshed Codex tokens to %s: %s",
                self._path, exc,
            )
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def load(path: Optional[Path] = None) -> CodexCredentials:
    """Read ``auth.json`` and return a :class:`CodexCredentials` handle.

    Raises :class:`CodexAuthError` with a user-actionable message when
    the file is missing or doesn't have an OAuth session — those are
    both fixed by running ``codex login``, so the CLI surfaces this
    error verbatim to the user.
    """
    auth_path = path or _auth_path()
    try:
        raw = auth_path.read_text()
    except FileNotFoundError as exc:
        raise CodexAuthError(
            f"Codex auth file not found at {auth_path}. Run `codex login` "
            f"first so TradingAgents can use your ChatGPT subscription."
        ) from exc
    except OSError as exc:
        raise CodexAuthError(f"Could not read {auth_path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodexAuthError(f"{auth_path} is not valid JSON: {exc}") from exc

    return CodexCredentials(auth_path, data)
