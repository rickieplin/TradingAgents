"""Read and refresh Claude Code OAuth tokens from the user's local session.

The Claude Code CLI stores a long-lived OAuth session after the user
signs in to their Claude Pro/Max subscription. We reuse that session so
TradingAgents can route agent calls through the user's subscription
quota instead of an Anthropic Platform API key — same model quality,
same tool-calling and structured-output behaviour, no per-token billing.

Storage locations:

- macOS: the keychain item ``Claude Code-credentials`` under the user's
  generic-password account. Read via ``security find-generic-password
  -a "$USER" -w -s "Claude Code-credentials"``.
- Other OSes (and macOS users who exported the file): the JSON file
  ``~/.claude/.credentials.json``. Honours ``CLAUDE_CONFIG_DIR`` and
  the test-only override ``CLAUDE_CREDENTIALS_PATH``.

The on-disk and keychain payloads share the same envelope::

    {"claudeAiOauth": {
        "accessToken":      "<opaque token>",
        "refreshToken":     "<opaque token>",
        "expiresAt":        1737900000000,   # ms since epoch, wall clock
        "scopes":           [...],
        "subscriptionType": "pro" | "max" | ...
    }}

Important: ``expiresAt`` is a millisecond wall-clock timestamp — it is
*not* a JWT exp claim. We compare it directly against ``time.time()``
(after converting to seconds) plus a 5-minute refresh buffer; do NOT
copy the Codex JWT-decode path here, the access tokens issued by
Claude's OAuth endpoint don't carry the same shape.

OAuth refresh:

- Endpoint: ``https://platform.claude.com/v1/oauth/token``
- Method: POST, ``Content-Type: application/json``
- Body keys: ``grant_type``, ``refresh_token``, ``client_id``, ``scope``
- Response keys: ``access_token``, ``refresh_token``, ``expires_in``,
  ``scope``.
- Refresh tokens are rotated on every refresh — must persist the new
  one back, otherwise the next refresh round-trip will fail with
  ``invalid_grant`` and the user will be silently logged out.

Concurrency model (matches codex_auth.py):

- One :class:`ClaudeCredentials` per resolved source, cached via
  :func:`load`. Multiple LangChain clients share the same instance and
  the same in-process refresh lock so a simultaneous expiry can't burn
  the refresh token twice.
- Each refresh-and-persist is guarded by an ``fcntl`` exclusive file
  lock on a sibling ``.lock`` file so a concurrent ``claude`` CLI run
  cannot race against us at the kernel level. The lock is taken even
  in keychain-fallback mode (it costs nothing and keeps the code path
  uniform with file-backed sessions).
- Persistence to file uses ``os.open(O_CREAT|O_EXCL, 0o600)`` followed
  by an atomic ``os.replace`` so credentials are never briefly visible
  under the umask-default mode.

Keychain fallback (read-only):

If only the macOS keychain has credentials, we read them but cannot
write rotated tokens back — the v1 implementation does not invoke
``security add-generic-password``. The rotation stays in memory for
the lifetime of this process; we emit a warning telling the user to
run ``claude`` once afterwards so the CLI persists the latest tokens
into the keychain. This trade-off is intentional: writing to the
keychain requires an interactive password prompt under certain access
control settings, which we cannot reliably handle from a background
analysis run.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import logging
import os
import subprocess
import sys
import threading
import time
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


# Claude Code's public OAuth client ID. Must match the value baked
# into the CLI or the refresh endpoint will reject the user's existing
# refresh token. Sourced from the upstream Claude Code reference impl.
_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_OAUTH_SCOPE = (
    "user:profile user:inference user:sessions:claude_code "
    "user:mcp_servers user:file_upload"
)

# Refresh proactively 5 minutes before expiry to absorb clock skew and
# avoid mid-request 401s; matches the codex behaviour.
_REFRESH_BUFFER_SECONDS = 300

# Defence against a malformed-token refresh loop: if `_refresh()`
# succeeds (HTTP 200) but the returned access token *still* looks
# expired (e.g. server-side clock skew, bogus `expires_in`), we'd
# otherwise re-refresh on every request. Cap consecutive bad refreshes
# and surface a hard error instead.
_MAX_CONSECUTIVE_BAD_REFRESHES = 3

# macOS keychain service name. Set by the upstream Claude Code CLI; do
# not change without coordinating, or we'll silently miss the user's
# real session.
_KEYCHAIN_SERVICE = "Claude Code-credentials"

# Envelope key the file/keychain payload uses. Centralised so we can't
# misspell it in one of the read/write paths and silently lose tokens.
_ENVELOPE_KEY = "claudeAiOauth"


def _credentials_path() -> Path:
    """Resolve the on-disk credentials file location.

    Honours ``CLAUDE_CREDENTIALS_PATH`` (full file path, mainly used by
    tests) first, then ``CLAUDE_CONFIG_DIR`` (treat as the parent dir
    containing ``.credentials.json``), then the documented default
    ``~/.claude/.credentials.json``.
    """
    override = os.environ.get("CLAUDE_CREDENTIALS_PATH")
    if override:
        return Path(override)
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / ".credentials.json"
    return Path.home() / ".claude" / ".credentials.json"


class ClaudeAuthError(RuntimeError):
    """Raised when Claude credentials are absent, malformed, or unrefreshable."""


@contextlib.contextmanager
def _flock_exclusive(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive POSIX file lock for the duration of the ``with`` block.

    Same rationale as ``codex_auth._flock_exclusive`` — the lock is
    taken on a sibling ``.lock`` file because ``fcntl`` advisory locks
    are per-fd, and our atomic ``os.replace`` would otherwise invalidate
    the lock fd. On Windows (no ``fcntl``) this degrades to a no-op.
    """
    if not _HAVE_FCNTL:
        yield
        return

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _read_keychain() -> Optional[dict]:
    """Read the Claude Code OAuth payload from the macOS keychain.

    Returns the parsed envelope dict, or ``None`` if the keychain item
    is absent / unreadable / not on darwin. We treat any failure as
    "fall through to other sources" rather than raising — the on-disk
    file is a perfectly valid alternative source.
    """
    if sys.platform != "darwin":
        return None
    try:
        user = getpass.getuser()
    except KeyError:
        return None
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-a", user, "-w", "-s", _KEYCHAIN_SERVICE,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Claude Code keychain entry is not valid JSON; ignoring")
        return None


class ClaudeCredentials:
    """In-memory wrapper around a Claude Code OAuth session with refresh support.

    Construct via :func:`load`. The object caches the parsed envelope
    and refreshes the access token on demand (proactively, ~5 min
    before expiry) by calling Anthropic's OAuth token endpoint with the
    stored refresh token. Rotated tokens are written back to disk so
    the Claude Code CLI sees the same fresh credentials on its next run.

    Instances are deduplicated by source path via :func:`load`; do not
    instantiate directly when you have a real session, or two instances
    will hold separate in-process locks and the refresh-token rotation
    between them is racy.

    ``keychain_fallback=True`` indicates the envelope was sourced from
    the macOS keychain (no file backing). We still allow refresh but
    we cannot persist rotated tokens — the new refresh_token stays in
    memory only and we emit a warning so the user knows to run
    ``claude`` once to persist them into the keychain.
    """

    def __init__(
        self,
        path: Optional[Path],
        data: dict,
        *,
        keychain_fallback: bool = False,
    ):
        self._path = path
        self._lock_path = (
            path.with_suffix(path.suffix + ".lock") if path else None
        )
        self._data = data
        self._lock = threading.Lock()
        self._bad_refresh_streak = 0
        self._keychain_fallback = keychain_fallback

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def subscription_type(self) -> Optional[str]:
        return self._oauth.get("subscriptionType")

    @property
    def _oauth(self) -> dict:
        oauth = self._data.get(_ENVELOPE_KEY)
        if not isinstance(oauth, dict):
            raise ClaudeAuthError(
                f"Claude credentials envelope is missing the {_ENVELOPE_KEY!r} "
                "block. Run `claude` once and sign in with your Pro/Max "
                "subscription to populate it."
            )
        return oauth

    def access_token(self) -> str:
        """Return a valid access token, refreshing in-place if it's near expiry."""
        token = self._oauth.get("accessToken")
        if not token:
            raise ClaudeAuthError("Claude credentials have no accessToken")

        if not self._is_expired():
            return token

        with self._lock:
            # Re-check under the in-process lock so concurrent threads
            # don't both run a refresh round-trip.
            token = self._oauth.get("accessToken")
            if token and not self._is_expired():
                return token

            # Cross-process guard: another `claude` process may have
            # already rotated the file. Re-read from disk under the
            # file lock before deciding to refresh.
            with self._maybe_flock():
                self._reload_if_disk_newer()
                token = self._oauth.get("accessToken")
                if token and not self._is_expired():
                    return token
                self._refresh_locked()
                return self._oauth["accessToken"]

    def force_refresh(self) -> None:
        """Run a refresh unconditionally — used by the CLI preflight check.

        Returning without raising proves that the stored refresh token
        is still good, catching a revoked / rotated session before a
        long-running graph execution starts.
        """
        with self._lock, self._maybe_flock():
            self._reload_if_disk_newer()
            self._refresh_locked()

    @contextlib.contextmanager
    def _maybe_flock(self) -> Iterator[None]:
        """Take the file lock when we have a path; no-op for keychain-only sessions."""
        if self._lock_path is None:
            yield
            return
        with _flock_exclusive(self._lock_path):
            yield

    def _is_expired(self) -> bool:
        """``True`` if ``expiresAt`` is within the refresh buffer (or missing).

        ``expiresAt`` is a wall-clock timestamp in **milliseconds** — we
        divide by 1000 before comparing with ``time.time()`` (seconds).
        A missing / non-numeric value forces a refresh so the next round
        trip can surface the underlying issue.
        """
        expires_at = self._oauth.get("expiresAt")
        if not isinstance(expires_at, (int, float)):
            return True
        return time.time() >= ((expires_at / 1000.0) - _REFRESH_BUFFER_SECONDS)

    def _reload_if_disk_newer(self) -> None:
        """Pick up tokens rotated by a concurrent ``claude`` CLI invocation.

        Without this, our in-memory ``_data`` would still hold the old
        refresh_token after the CLI rotated the file — and our refresh
        call would then send that already-consumed token, getting back
        ``invalid_grant``.
        """
        if self._path is None:
            return
        try:
            raw = self._path.read_text()
        except (FileNotFoundError, OSError):
            return  # use what we already have; the next write will recover
        try:
            new_data = json.loads(raw)
        except json.JSONDecodeError:
            return
        # Reject partial writes / empty envelopes — keep current in-memory state.
        oauth = new_data.get("claudeAiOauth") if isinstance(new_data, dict) else None
        if not isinstance(oauth, dict) or not oauth.get("accessToken"):
            logger.warning(
                "Ignoring on-disk credentials at %s — missing claudeAiOauth "
                "envelope or accessToken (likely a concurrent partial write).",
                self._path,
            )
            return
        self._data = new_data

    def _refresh_locked(self) -> None:
        """Perform the actual refresh round-trip; caller must hold both locks."""
        refresh_token = self._oauth.get("refreshToken")
        if not refresh_token:
            raise ClaudeAuthError(
                "Claude credentials have no refreshToken — re-run `claude` "
                "and sign in to renew the OAuth session."
            )

        logger.info("Refreshing Claude Code OAuth access token")
        try:
            resp = httpx.post(
                _TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": _OAUTH_CLIENT_ID,
                    "scope": _OAUTH_SCOPE,
                },
                headers={"Content-Type": "application/json"},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            raise ClaudeAuthError(f"token refresh request failed: {exc}") from exc

        if resp.status_code != 200:
            raise ClaudeAuthError(
                f"token refresh returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )

        body = resp.json()
        new_access = body.get("access_token")
        if not new_access:
            raise ClaudeAuthError(
                f"token refresh response missing access_token: {body!r}"
            )

        # Compute the new expiry; ``expires_in`` is seconds, but our
        # stored ``expiresAt`` is milliseconds.
        expires_in = body.get("expires_in")
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            new_expires_at_ms = int((time.time() + float(expires_in)) * 1000)
        else:
            # No expires_in -> treat the new token as instantly expired
            # so the bad-refresh guard kicks in below.
            new_expires_at_ms = 0

        # Tentatively apply the new values so ``_is_expired`` can judge
        # them, then roll back if they fail the sanity check.
        previous_oauth = dict(self._oauth)
        self._oauth["accessToken"] = new_access
        self._oauth["expiresAt"] = new_expires_at_ms
        if body.get("refresh_token"):
            self._oauth["refreshToken"] = body["refresh_token"]
        if body.get("scope"):
            # Scopes are stored as a list in the envelope; the response
            # ships them as a space-separated string.
            self._oauth["scopes"] = body["scope"].split()

        if self._is_expired():
            # Refresh succeeded but the new token already looks stale —
            # don't accept it, or every subsequent call would loop back
            # through refresh and burn through refresh tokens.
            self._data[_ENVELOPE_KEY] = previous_oauth
            self._bad_refresh_streak += 1
            if self._bad_refresh_streak >= _MAX_CONSECUTIVE_BAD_REFRESHES:
                raise ClaudeAuthError(
                    "Refresh succeeded but the returned access token is "
                    "already expired (consecutive failures: "
                    f"{self._bad_refresh_streak}). Refusing to loop; "
                    "re-run `claude` to reset the session."
                )
            raise ClaudeAuthError(
                "Refresh response carried an apparently-expired token. "
                "Caller may retry."
            )
        self._bad_refresh_streak = 0

        self._persist_locked()

    def _persist_locked(self) -> None:
        """Write the in-memory state back atomically.

        Creates the tmp file via ``os.open(O_CREAT|O_EXCL, 0o600)`` so
        credentials are never briefly visible under the umask-default
        mode, then ``os.replace`` for an atomic rename. Caller must
        hold the file lock.

        For keychain-only sessions we log a one-time warning instead;
        writing back to the keychain is not implemented in this version.
        """
        if self._path is None:
            if not getattr(self, "_warned_no_persist", False):
                logger.warning(
                    "Claude Code OAuth token rotated, but we have no file "
                    "to persist it to (keychain-only session). Run "
                    "`claude` once to refresh the on-disk/keychain copy."
                )
                self._warned_no_persist = True
            return

        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = json.dumps(self._data, indent=2).encode()

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
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()
            logger.warning(
                "Could not persist refreshed Claude tokens to %s: %s",
                self._path, exc,
            )


# Process-wide singleton cache so the quick and deep LLM slots share
# one ClaudeCredentials per source. Without this they'd each hold a
# separate ``threading.Lock`` and the rotation across them would race.
_credentials_cache: dict[str, ClaudeCredentials] = {}
_credentials_cache_lock = threading.Lock()


def _cache_key(path: Optional[Path], keychain_fallback: bool) -> str:
    if path is not None:
        return f"file:{path}"
    if keychain_fallback:
        return "keychain:Claude Code-credentials"
    return "unknown"


def load(path: Optional[Path] = None) -> ClaudeCredentials:
    """Read the Claude Code OAuth session and return a :class:`ClaudeCredentials` handle.

    Resolution order:
    1. ``path`` if explicitly provided (used by tests).
    2. The on-disk file resolved by :func:`_credentials_path`.
    3. The macOS keychain (read-only fallback).

    Multiple calls for the same source return the same cached instance,
    so all LangChain clients in the process share one refresh lock and
    one in-memory token snapshot.

    Raises :class:`ClaudeAuthError` with a user-actionable message when
    no source has credentials.
    """
    requested_path = (path or _credentials_path()).resolve() if path or _credentials_path() else None

    # Try the on-disk file first — if it exists we prefer file form so
    # refreshes are persisted.
    if requested_path is not None and requested_path.exists():
        with _credentials_cache_lock:
            key = _cache_key(requested_path, keychain_fallback=False)
            cached = _credentials_cache.get(key)
            if cached is not None:
                return cached

            try:
                raw = requested_path.read_text()
            except OSError as exc:
                raise ClaudeAuthError(
                    f"Could not read {requested_path}: {exc}"
                ) from exc

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ClaudeAuthError(
                    f"{requested_path} is not valid JSON: {exc}"
                ) from exc

            creds = ClaudeCredentials(requested_path, data)
            _credentials_cache[key] = creds
            return creds

    # Fall through to keychain (macOS only). We do NOT consult the
    # keychain when the caller explicitly provided a path (tests rely
    # on this isolation).
    if path is None:
        kc_data = _read_keychain()
        if kc_data is not None:
            with _credentials_cache_lock:
                key = _cache_key(None, keychain_fallback=True)
                cached = _credentials_cache.get(key)
                if cached is not None:
                    return cached
                creds = ClaudeCredentials(None, kc_data, keychain_fallback=True)
                _credentials_cache[key] = creds
                return creds

    # Nothing worked — emit a helpful, actionable error.
    raise ClaudeAuthError(
        f"Claude credentials not found at {requested_path}. Run `claude` "
        "and sign in with your Pro/Max subscription, or set "
        "CLAUDE_CREDENTIALS_PATH to point at an exported credentials "
        "file."
    )


def _reset_cache_for_tests() -> None:
    """Clear the singleton cache. Test-only; not part of the public API."""
    with _credentials_cache_lock:
        _credentials_cache.clear()
