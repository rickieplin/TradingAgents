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

Concurrency model (matters because the Codex CLI shares this file):

- One :class:`CodexCredentials` instance per auth-file path, cached via
  :func:`load`. Multiple LangChain clients (e.g. the quick and deep
  slots) share the same instance and the same in-process refresh lock,
  so they cannot race each other into burning a refresh token twice.
- Each refresh-and-persist is also guarded by an ``fcntl`` exclusive
  file lock on a sibling lockfile, so a concurrent ``codex`` CLI run
  cannot race against us at the kernel level. Refresh tokens are
  single-use; without this lock two processes would both succeed on
  the first ``/oauth/token`` call and the loser would lock the user out.
- The tmp file used for the atomic ``os.replace`` is created with mode
  ``0o600`` via ``os.open(O_CREAT|O_EXCL)`` so credentials are never
  briefly visible under the umask-default mode.

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
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

try:  # POSIX only — Windows has no fcntl; we gracefully degrade there.
    import fcntl  # type: ignore[import-not-found]
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover — Windows path
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False


# OpenAI's public OAuth client ID used by the Codex CLI. Refreshing
# against a different client ID rejects the user's existing refresh
# token, so this must stay in lockstep with the CLI's value.
_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_TOKEN_URL = "https://auth.openai.com/oauth/token"

# JWT exp claims are issued in seconds and Codex rotates well before
# expiry, but we refresh proactively at 5 minutes to handle clock skew
# and avoid mid-request 401s.
_REFRESH_BUFFER_SECONDS = 300

# Defence against a malformed-token refresh loop: if `_refresh()` succeeds
# (HTTP 200) but the returned access token still parses as expired or
# unparseable, we'd otherwise re-refresh on every request. Cap consecutive
# bad refreshes and surface a hard error instead.
_MAX_CONSECUTIVE_BAD_REFRESHES = 3


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


@contextlib.contextmanager
def _flock_exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive POSIX file lock for the duration of the ``with`` block.

    The lock is taken on a sibling ``.lock`` file rather than on
    ``auth.json`` itself — ``fcntl`` advisory locks are per-fd, and we
    don't want our atomic ``os.replace`` to invalidate the lock fd.
    On Windows (no ``fcntl``) this is a no-op; concurrent access there
    is rare enough that we accept the gap rather than ship a flaky
    Windows-specific implementation.
    """
    if not _HAVE_FCNTL:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # O_CREAT so concurrent first-time users don't race to create the
    # lockfile. O_RDWR so flock works on macOS (LOCK_EX requires write
    # access in some BSD-derived kernels).
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class CodexCredentials:
    """In-memory wrapper around ``~/.codex/auth.json`` with refresh support.

    Construct via :func:`load`. The object caches the parsed file and
    refreshes the access token on demand (proactively, ~5 min before
    expiry) by calling OpenAI's OAuth token endpoint with the stored
    refresh token. Rotated tokens are written back so the Codex CLI
    sees the same fresh credentials on its next run.

    Instances are deduplicated by auth-file path via :func:`load`; do
    not instantiate directly when you have a real auth file, or two
    instances will hold separate in-process locks and the refresh-token
    rotation between them is racy.
    """

    def __init__(self, path: Path, data: dict):
        self._path = path
        self._lock_path = path.with_suffix(path.suffix + ".lock")
        self._data = data
        self._lock = threading.Lock()
        self._bad_refresh_streak = 0

    @property
    def path(self) -> Path:
        return self._path

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

        if not self._is_expired(token):
            return token

        with self._lock:
            # Re-check under the in-process lock so concurrent threads
            # don't both run a refresh round-trip.
            token = self._tokens.get("access_token")
            if token and not self._is_expired(token):
                return token

            # Cross-process guard: another `codex` process may have
            # already rotated the file. Re-read from disk under the
            # file lock before deciding to refresh.
            with _flock_exclusive(self._lock_path):
                self._reload_if_disk_newer()
                token = self._tokens.get("access_token")
                if token and not self._is_expired(token):
                    return token
                self._refresh_locked()
                return self._tokens["access_token"]

    def force_refresh(self) -> None:
        """Run a refresh unconditionally — used by the CLI preflight check.

        Returning without raising proves that the stored refresh token
        is still good, catching a revoked / rotated session before a
        long-running graph execution starts.
        """
        with self._lock, _flock_exclusive(self._lock_path):
            self._reload_if_disk_newer()
            self._refresh_locked()

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

    def _reload_if_disk_newer(self) -> None:
        """Pick up tokens rotated by a concurrent `codex` CLI invocation.

        Without this, our in-memory ``_data`` would still hold the old
        refresh_token after the CLI rotated the file — and our refresh
        call would then send that already-consumed token, getting back
        ``invalid_grant``.
        """
        try:
            raw = self._path.read_text()
        except (FileNotFoundError, OSError):
            return  # use what we already have; the next write will recover
        try:
            self._data = json.loads(raw)
        except json.JSONDecodeError:
            return

    def _refresh_locked(self) -> None:
        """Perform the actual refresh round-trip; caller must hold both locks."""
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

        # Sanity-check the returned token. If the OAuth endpoint hands
        # back something we can't parse or that already looks expired,
        # accepting it would put us in an unbounded refresh loop on the
        # next request. Cap consecutive failures and surface a hard
        # error so the user notices.
        if self._is_expired(new_access):
            self._bad_refresh_streak += 1
            if self._bad_refresh_streak >= _MAX_CONSECUTIVE_BAD_REFRESHES:
                raise CodexAuthError(
                    "Refresh succeeded but the returned access token is "
                    "unparseable or already expired (consecutive failures: "
                    f"{self._bad_refresh_streak}). Refusing to loop; "
                    "re-run `codex login` to reset the session."
                )
            # Soft-fail: store what we got but raise so the caller can
            # decide whether to retry. Returning silently here would
            # mask a server-side regression.
            raise CodexAuthError(
                "Refresh response carried an apparently-expired token "
                "(it may parse differently than expected). Caller may retry."
            )
        self._bad_refresh_streak = 0

        self._tokens["access_token"] = new_access
        # OpenAI rotates refresh tokens on every refresh; persist the
        # new one so subsequent refreshes (this process or codex CLI)
        # keep working.
        if body.get("refresh_token"):
            self._tokens["refresh_token"] = body["refresh_token"]
        if body.get("id_token"):
            self._tokens["id_token"] = body["id_token"]
        self._data["last_refresh"] = datetime.now(timezone.utc).isoformat()

        self._persist_locked()

    def _persist_locked(self) -> None:
        """Write the in-memory state back to ``auth.json`` atomically.

        Creates the tmp file via ``os.open(O_CREAT|O_EXCL, 0o600)`` so
        credentials are never briefly visible under the umask-default
        mode, then ``os.replace`` for an atomic rename. Caller must
        hold the file lock.
        """
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = json.dumps(self._data, indent=2).encode()

        # Remove any leftover tmp (e.g. from a previous crashed run);
        # O_EXCL would otherwise fail and we'd leak that crashed-run
        # state forward.
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()

        try:
            fd = os.open(
                str(tmp_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            # Best-effort cleanup of the tmp file on failure.
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            logger.warning(
                "Could not persist refreshed Codex tokens to %s: %s",
                self._path, exc,
            )


# Process-wide singleton cache so the quick and deep LLM slots share
# one CodexCredentials per auth file. Without this they'd each hold a
# separate `threading.Lock` and the rotation across them would race.
_credentials_cache: dict[Path, CodexCredentials] = {}
_credentials_cache_lock = threading.Lock()


def load(path: Optional[Path] = None) -> CodexCredentials:
    """Read ``auth.json`` and return a :class:`CodexCredentials` handle.

    Multiple calls for the same path return the same cached instance,
    so all LangChain clients in the process share one refresh lock and
    one in-memory token snapshot. Tests that need a fresh instance
    should pass a unique path or call :func:`_reset_cache_for_tests`.

    Raises :class:`CodexAuthError` with a user-actionable message when
    the file is missing or doesn't have an OAuth session — those are
    both fixed by running ``codex login``, so the CLI surfaces this
    error verbatim to the user.
    """
    auth_path = (path or _auth_path()).resolve()

    with _credentials_cache_lock:
        cached = _credentials_cache.get(auth_path)
        if cached is not None:
            return cached

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

        creds = CodexCredentials(auth_path, data)
        _credentials_cache[auth_path] = creds
        return creds


def _reset_cache_for_tests() -> None:
    """Clear the singleton cache. Test-only; not part of the public API."""
    with _credentials_cache_lock:
        _credentials_cache.clear()
