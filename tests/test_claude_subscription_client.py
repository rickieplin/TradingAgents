"""Tests for the Claude Pro/Max subscription provider (`claude_subscription`).

The provider reads OAuth credentials from the Claude Code session (macOS
keychain or ``~/.claude/.credentials.json``) and uses them as a bearer
token against the standard Anthropic API. These tests stub the file
backing and the HTTP/keychain layers so we never actually hit the
network or the user's real session.

Coverage:
- ``claude_auth.load`` parses the OAuth envelope and surfaces helpful
  errors on missing source / malformed JSON / missing block.
- ``expiresAt`` is interpreted as **milliseconds** (the critical
  deviation from the Codex JWT-exp path).
- Token refresh fires only when ``expiresAt`` is within the buffer,
  rotates both access and refresh tokens, persists with mode ``0o600``,
  and round-trips the ``{"claudeAiOauth": {...}}`` envelope shape.
- ``force_refresh`` validates the session eagerly (CLI preflight).
- ``load()`` returns the same instance for the same path (singleton).
- The factory builds a ``ChatClaudeSubscription`` pinned to
  ``api.anthropic.com``, with ``anthropic-beta: oauth-2025-04-20`` in
  ``default_headers`` and the SDK clients' ``auth_token`` populated /
  ``api_key`` nulled.
- ``_generate`` rotates the token on the SDK clients before delegating.
- Provider registration: factory, validator, and api-key-env mapping
  all recognise the provider; ``ensure_api_key`` doesn't prompt.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.claude_auth import (
    ClaudeAuthError,
    ClaudeCredentials,
    _reset_cache_for_tests,
    load as load_claude_credentials,
)
from tradingagents.llm_clients.claude_subscription_client import (
    CLAUDE_SUBSCRIPTION_BASE_URL,
    ChatClaudeSubscription,
    ClaudeSubscriptionClient,
)
from tradingagents.llm_clients.validators import validate_model


# ---- fixtures --------------------------------------------------------------


def _write_credentials(
    path: Path,
    *,
    exp_in_seconds: int = 3600,
    access_token: str = "at-original",
    refresh_token: str = "rt-original",
) -> Path:
    """Write a `.credentials.json` whose access token expires in `exp_in_seconds`."""
    expires_at_ms = int((time.time() + exp_in_seconds) * 1000)
    data = {
        "claudeAiOauth": {
            "accessToken": access_token,
            "refreshToken": refresh_token,
            "expiresAt": expires_at_ms,
            "scopes": ["user:inference"],
            "subscriptionType": "pro",
        }
    }
    path.write_text(json.dumps(data))
    return path


@pytest.fixture(autouse=True)
def _reset_credentials_cache():
    """Clear the load() singleton cache between tests for isolation."""
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.fixture
def credentials_path(tmp_path, monkeypatch):
    """Point CLAUDE_CREDENTIALS_PATH at a temp file."""
    cred = tmp_path / ".credentials.json"
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(cred))
    return cred


# ---- credentials file loading ---------------------------------------------


def test_load_reads_oauth_envelope(credentials_path):
    _write_credentials(credentials_path)
    creds = load_claude_credentials()
    assert creds.access_token() == "at-original"
    assert creds.subscription_type == "pro"


def test_load_missing_file_surfaces_actionable_error(credentials_path):
    # No keychain on the CI host (or even on macOS in this isolated
    # tmp env), so this raises with a "run claude" hint.
    with patch(
        "tradingagents.llm_clients.claude_auth._read_keychain",
        return_value=None,
    ):
        with pytest.raises(ClaudeAuthError, match="Claude credentials not found"):
            load_claude_credentials()


def test_load_rejects_malformed_json(credentials_path):
    credentials_path.write_text("{not json")
    with pytest.raises(ClaudeAuthError, match="not valid JSON"):
        load_claude_credentials()


def test_load_rejects_missing_envelope_block(credentials_path):
    # If the file exists but has no `claudeAiOauth` wrapper (e.g. user
    # exported a different file), surface a helpful error rather than
    # silently failing on a later access_token() call.
    credentials_path.write_text(json.dumps({"some_other_key": "x"}))
    creds = load_claude_credentials()
    with pytest.raises(ClaudeAuthError, match="claudeAiOauth"):
        creds.access_token()


def test_load_returns_singleton_for_same_path(credentials_path):
    # Per-source singleton: quick + deep LLM slots must share refresh
    # state so a simultaneous expiry can't burn the refresh token twice.
    _write_credentials(credentials_path)
    a = load_claude_credentials()
    b = load_claude_credentials()
    assert a is b


def test_load_prefers_file_over_keychain(credentials_path):
    # When both the file and keychain have a session, file wins (so
    # rotated tokens persist).
    _write_credentials(credentials_path, access_token="file-token")
    with patch(
        "tradingagents.llm_clients.claude_auth._read_keychain",
        return_value={
            "claudeAiOauth": {
                "accessToken": "keychain-token",
                "refreshToken": "rt-kc",
                "expiresAt": int((time.time() + 3600) * 1000),
                "scopes": [],
                "subscriptionType": "max",
            }
        },
    ):
        creds = load_claude_credentials()
    assert creds.access_token() == "file-token"


def test_load_falls_back_to_keychain_when_file_absent(tmp_path, monkeypatch):
    # If the file doesn't exist but the keychain has a session, use it
    # in read-only fallback mode.
    nonexistent = tmp_path / "missing.json"
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(nonexistent))
    kc_payload = {
        "claudeAiOauth": {
            "accessToken": "keychain-only",
            "refreshToken": "rt-kc",
            "expiresAt": int((time.time() + 3600) * 1000),
            "scopes": [],
            "subscriptionType": "max",
        }
    }
    with patch(
        "tradingagents.llm_clients.claude_auth._read_keychain",
        return_value=kc_payload,
    ):
        creds = load_claude_credentials()
    assert creds.access_token() == "keychain-only"
    assert creds.path is None  # signals keychain-only mode


# ---- expiry semantics ------------------------------------------------------


def test_is_expired_treats_expires_at_as_milliseconds(credentials_path):
    # If we (incorrectly) interpreted expiresAt as seconds, a token
    # expiring 3600 *seconds* from now would parse as a far-future
    # timestamp and be considered fresh. Pin the millisecond contract.
    _write_credentials(credentials_path, exp_in_seconds=3600)
    creds = load_claude_credentials()
    assert not creds._is_expired()

    # Past expiry (1 second ago) -> definitely expired.
    creds._oauth["expiresAt"] = int((time.time() - 1) * 1000)
    assert creds._is_expired()

    # Within the 5-minute refresh buffer -> still considered expired.
    creds._oauth["expiresAt"] = int((time.time() + 60) * 1000)
    assert creds._is_expired()


def test_is_expired_missing_expires_at_forces_refresh(credentials_path):
    _write_credentials(credentials_path)
    creds = load_claude_credentials()
    creds._oauth.pop("expiresAt", None)
    assert creds._is_expired()


# ---- refresh behavior ------------------------------------------------------


def test_access_token_returns_cached_when_fresh(credentials_path):
    _write_credentials(credentials_path, exp_in_seconds=3600)
    creds = load_claude_credentials()
    with patch("tradingagents.llm_clients.claude_auth.httpx.post") as mock_post:
        token = creds.access_token()
    mock_post.assert_not_called()
    assert token == "at-original"


def _make_fake_refresh_response(
    new_access: str = "at-rotated",
    new_refresh: str = "rt-rotated",
    expires_in: int = 3600,
    status_code: int = 200,
    response_text: str = "",
):
    return type("R", (), {
        "status_code": status_code,
        "json": staticmethod(lambda: {
            "access_token": new_access,
            "refresh_token": new_refresh,
            "expires_in": expires_in,
            "scope": "user:inference user:profile",
        }),
        "text": response_text,
    })()


def test_access_token_refreshes_when_expired(credentials_path):
    _write_credentials(credentials_path, exp_in_seconds=60)  # within buffer
    creds = load_claude_credentials()

    fake = _make_fake_refresh_response()
    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=fake,
    ) as mock_post:
        result = creds.access_token()

    args, kwargs = mock_post.call_args
    assert "platform.claude.com/v1/oauth/token" in args[0]
    # Refresh body must be JSON, with the rotated refresh token.
    assert kwargs["json"]["grant_type"] == "refresh_token"
    assert kwargs["json"]["refresh_token"] == "rt-original"
    assert kwargs["json"]["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    assert kwargs["headers"]["Content-Type"] == "application/json"

    assert result == "at-rotated"

    # Rotated refresh token must persist back to disk in the envelope
    # shape so the next refresh — or the `claude` CLI itself — doesn't
    # try the now-invalidated old one.
    persisted = json.loads(credentials_path.read_text())
    assert persisted["claudeAiOauth"]["refreshToken"] == "rt-rotated"
    assert persisted["claudeAiOauth"]["accessToken"] == "at-rotated"
    # expiresAt was rewritten in milliseconds, not seconds.
    expires_at = persisted["claudeAiOauth"]["expiresAt"]
    assert expires_at > int(time.time() * 1000)


def test_envelope_wrapper_round_trips(credentials_path):
    # Confirm the {"claudeAiOauth": {...}} wrapper survives a refresh.
    _write_credentials(credentials_path, exp_in_seconds=60)
    creds = load_claude_credentials()
    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=_make_fake_refresh_response(),
    ):
        creds.access_token()

    persisted = json.loads(credentials_path.read_text())
    assert "claudeAiOauth" in persisted
    assert set(persisted["claudeAiOauth"].keys()) >= {
        "accessToken", "refreshToken", "expiresAt", "scopes", "subscriptionType",
    }


def test_refresh_failure_raises_actionable_error(credentials_path):
    _write_credentials(credentials_path, exp_in_seconds=60)
    creds = load_claude_credentials()

    fake = _make_fake_refresh_response(
        status_code=400,
        response_text='{"error":"invalid_grant"}',
    )
    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=fake,
    ):
        with pytest.raises(ClaudeAuthError, match="HTTP 400"):
            creds.access_token()


def test_refresh_loop_guard_raises_after_repeated_bad_tokens(credentials_path):
    # If the OAuth endpoint hands back a token that already looks
    # expired, we'd otherwise loop on every request. Cap consecutive
    # failures and raise a hard error.
    _write_credentials(credentials_path, exp_in_seconds=60)
    creds = load_claude_credentials()

    # `expires_in=0` -> the new token is instantly expired by our math.
    fake = _make_fake_refresh_response(expires_in=0)
    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=fake,
    ):
        for _ in range(3):
            with pytest.raises(ClaudeAuthError):
                creds.access_token()
        with pytest.raises(ClaudeAuthError, match="Refusing to loop"):
            creds.access_token()


def test_persisted_file_is_mode_0o600(credentials_path):
    # Defence in depth: the tmp file is created with O_CREAT|O_EXCL
    # and mode 0o600, then atomically renamed.
    _write_credentials(credentials_path, exp_in_seconds=60)
    creds = load_claude_credentials()

    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=_make_fake_refresh_response(),
    ):
        creds.access_token()

    mode = stat.S_IMODE(credentials_path.stat().st_mode)
    assert mode == 0o600, f"expected mode 0o600 after refresh, got 0o{mode:o}"


def test_force_refresh_validates_session_eagerly(credentials_path):
    # CLI preflight uses this to detect a revoked refresh token *before*
    # a 30-minute analysis run starts. A bare token-fetch is not enough
    # — we need an actual round-trip against the OAuth endpoint.
    _write_credentials(credentials_path, exp_in_seconds=3600)  # still fresh!
    creds = load_claude_credentials()

    fake = _make_fake_refresh_response(
        status_code=401,
        response_text='{"error":"invalid_grant"}',
    )
    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=fake,
    ):
        with pytest.raises(ClaudeAuthError, match="HTTP 401"):
            creds.force_refresh()


def test_file_lock_is_taken_during_refresh(credentials_path):
    # The file lock prevents two processes from racing into a refresh
    # and burning the single-use refresh_token. Patching _flock_exclusive
    # and asserting it was entered proves the call site holds it.
    _write_credentials(credentials_path, exp_in_seconds=60)
    creds = load_claude_credentials()

    flock_calls: list = []

    @__import__("contextlib").contextmanager
    def fake_flock(lock_path):
        flock_calls.append(lock_path)
        yield

    with patch(
        "tradingagents.llm_clients.claude_auth._flock_exclusive",
        side_effect=fake_flock,
    ):
        with patch(
            "tradingagents.llm_clients.claude_auth.httpx.post",
            return_value=_make_fake_refresh_response(),
        ):
            creds.access_token()

    assert flock_calls, "refresh must hold the file lock"


# ---- LangChain integration -------------------------------------------------


def test_factory_returns_claude_subscription_client(credentials_path):
    _write_credentials(credentials_path)
    client = create_llm_client(
        provider="claude_subscription", model="claude-sonnet-4-6",
    )
    assert isinstance(client, ClaudeSubscriptionClient)


def test_get_llm_wires_subscription_backend(credentials_path):
    _write_credentials(credentials_path)
    client = create_llm_client(
        provider="claude_subscription", model="claude-sonnet-4-6",
    )
    llm = client.get_llm()

    assert isinstance(llm, ChatClaudeSubscription)
    # The SDK's auth_headers property is the only source of auth on the
    # outgoing request. It must emit a Bearer header and NOT X-Api-Key.
    auth_headers = llm._client.auth_headers
    assert auth_headers == {"Authorization": "Bearer at-original"}, (
        f"Bearer-only contract violated: {auth_headers}"
    )
    assert llm._async_client.auth_headers == {"Authorization": "Bearer at-original"}

    # default_headers carry the OAuth beta marker the API requires.
    assert llm._client.default_headers.get("anthropic-beta") == "oauth-2025-04-20"


def test_get_llm_nulls_api_key_on_sdk_clients(credentials_path):
    _write_credentials(credentials_path)
    client = create_llm_client(
        provider="claude_subscription", model="claude-sonnet-4-6",
    )
    llm = client.get_llm()

    # Critical: api_key must be None on both SDK clients so the SDK's
    # auth_headers picks the Bearer branch instead of X-Api-Key.
    assert llm._client.api_key is None
    assert llm._async_client.api_key is None
    assert llm._client.auth_token == "at-original"
    assert llm._async_client.auth_token == "at-original"


def test_get_llm_pins_base_url_ignoring_caller_override(credentials_path):
    # A stale TRADINGAGENTS_LLM_BACKEND_URL would otherwise leak the
    # OAuth bearer to the wrong host. The provider must pin
    # api.anthropic.com unconditionally.
    _write_credentials(credentials_path)
    client = create_llm_client(
        provider="claude_subscription",
        model="claude-sonnet-4-6",
        base_url="https://api.openai.com/v1",
    )
    llm = client.get_llm()
    # ChatAnthropic stores the URL as either anthropic_api_url (str) or
    # via the SDK client's base_url. Check both surfaces.
    api_url = getattr(llm, "anthropic_api_url", None)
    if api_url is not None and hasattr(api_url, "get_secret_value"):
        api_url = api_url.get_secret_value()
    assert str(api_url).rstrip("/") == CLAUDE_SUBSCRIPTION_BASE_URL.rstrip("/")


def test_construction_fails_fast_on_missing_auth(credentials_path):
    # Loading credentials at __init__ time means broken auth surfaces
    # at config time, not 30 minutes into a graph run.
    with patch(
        "tradingagents.llm_clients.claude_auth._read_keychain",
        return_value=None,
    ):
        with pytest.raises(ClaudeAuthError):
            create_llm_client(
                provider="claude_subscription",
                model="claude-sonnet-4-6",
            )


def test_generate_rotates_token_on_sdk_clients(credentials_path):
    # Pin the contract that _generate refreshes the bearer token on the
    # SDK clients before delegating to super(). A future refactor that
    # moves this rotation elsewhere must update this test.
    _write_credentials(credentials_path, exp_in_seconds=3600)  # fresh now
    client = create_llm_client(
        provider="claude_subscription", model="claude-sonnet-4-6",
    )
    llm = client.get_llm()

    # Stale the token so the next request triggers refresh. Write to
    # disk too — `_reload_if_disk_newer` would otherwise reset us to
    # the fresh on-disk copy under the cross-process refresh lock.
    stale_expires_at_ms = int((time.time() - 1) * 1000)
    llm.claude_credentials._oauth["accessToken"] = "at-stale"
    llm.claude_credentials._oauth["expiresAt"] = stale_expires_at_ms
    credentials_path.write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "at-stale",
            "refreshToken": "rt-original",
            "expiresAt": stale_expires_at_ms,
            "scopes": [],
            "subscriptionType": "pro",
        }
    }))

    new_access = "at-rotated-from-generate"
    fake = _make_fake_refresh_response(new_access=new_access)

    # Stub super()._generate so we don't actually hit the network — we
    # only care that the rotation ran first. ChatAnthropic is the
    # closest ancestor that defines _generate; patch it there so super()
    # resolves through MRO to our stub.
    from langchain_anthropic import ChatAnthropic
    with patch(
        "tradingagents.llm_clients.claude_auth.httpx.post",
        return_value=fake,
    ):
        with patch.object(
            ChatAnthropic,
            "_generate",
            return_value=MagicMock(),
        ):
            llm._generate(messages=[], stop=None, run_manager=None)

    assert llm._client.auth_token == new_access
    assert llm._async_client.auth_token == new_access
    assert llm._client.api_key is None
    assert llm._async_client.api_key is None


# ---- registration smoke tests ---------------------------------------------


def test_claude_subscription_validator_accepts_any_model():
    # Like ollama/openrouter/codex, the validator must not gate
    # user-typed model IDs — Anthropic rotates supported models faster
    # than this catalog is updated.
    assert validate_model("claude_subscription", "claude-sonnet-4-6") is True
    assert validate_model("claude_subscription", "some-future-model") is True


def test_claude_subscription_api_key_env_is_none():
    # Credentials come from the Claude Code session, not an env var —
    # so ensure_api_key must NOT prompt the user.
    assert get_api_key_env("claude_subscription") is None


# ---- graph-layer wiring ---------------------------------------------------


def test_get_provider_kwargs_forwards_effort_for_claude_subscription():
    # Regression: the graph's _get_provider_kwargs originally only
    # matched ``anthropic`` — so picking ``claude_subscription`` in the
    # CLI silently dropped the user's effort selection on the floor.
    # ClaudeSubscriptionClient accepts ``effort`` via _PASSTHROUGH_KWARGS;
    # the graph must hand it through for both providers.
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    stub = type("Stub", (), {})()
    stub.config = {
        "llm_provider": "claude_subscription",
        "anthropic_effort": "high",
    }
    kwargs = TradingAgentsGraph._get_provider_kwargs(stub)
    assert kwargs == {"effort": "high"}

    # Sibling sanity: same path for native anthropic.
    stub.config = {"llm_provider": "anthropic", "anthropic_effort": "medium"}
    assert TradingAgentsGraph._get_provider_kwargs(stub) == {"effort": "medium"}

    # And explicit ``None``/missing must not inject the key.
    stub.config = {"llm_provider": "claude_subscription", "anthropic_effort": None}
    assert TradingAgentsGraph._get_provider_kwargs(stub) == {}


# ---- credential reload safety --------------------------------------------


def test_reload_if_disk_newer_ignores_partial_write(tmp_path, caplog):
    # If the real ``claude`` CLI is mid-write when our process polls the
    # file, json.loads can succeed on a truncated-but-syntactically-valid
    # payload (e.g. ``{}``). We must NOT replace our healthy in-memory
    # state with that empty envelope — otherwise the very next
    # ``_oauth`` lookup raises "missing claudeAiOauth block" instead of
    # using the refresh_token we already hold.
    cred_path = tmp_path / ".credentials.json"
    cred_path.write_text(json.dumps({}))  # valid JSON, no envelope

    healthy = {
        "claudeAiOauth": {
            "accessToken": "at-in-memory",
            "refreshToken": "rt-in-memory",
            "expiresAt": int((time.time() + 3600) * 1000),
            "scopes": ["user:inference"],
            "subscriptionType": "pro",
        }
    }
    creds = ClaudeCredentials(path=cred_path, data=dict(healthy))

    with caplog.at_level("WARNING", logger="tradingagents.llm_clients.claude_auth"):
        creds._reload_if_disk_newer()

    # In-memory state intact; access_token() still works.
    assert creds.access_token() == "at-in-memory"
    assert any(
        "missing claudeAiOauth" in rec.message for rec in caplog.records
    ), "expected a warning about the partial write being ignored"


def test_reload_if_disk_newer_ignores_envelope_without_access_token(tmp_path):
    # Same defence for the ``{"claudeAiOauth": {}}`` shape — envelope
    # present but the access token isn't written yet.
    cred_path = tmp_path / ".credentials.json"
    cred_path.write_text(json.dumps({"claudeAiOauth": {}}))

    healthy = {
        "claudeAiOauth": {
            "accessToken": "at-in-memory",
            "refreshToken": "rt-in-memory",
            "expiresAt": int((time.time() + 3600) * 1000),
            "scopes": ["user:inference"],
            "subscriptionType": "pro",
        }
    }
    creds = ClaudeCredentials(path=cred_path, data=dict(healthy))
    creds._reload_if_disk_newer()
    assert creds.access_token() == "at-in-memory"


def test_reload_if_disk_newer_accepts_valid_rotation(tmp_path):
    # Positive path: a fully-formed rotation on disk DOES replace the
    # in-memory copy so the next refresh uses the latest refresh_token.
    cred_path = tmp_path / ".credentials.json"
    rotated = {
        "claudeAiOauth": {
            "accessToken": "at-rotated",
            "refreshToken": "rt-rotated",
            "expiresAt": int((time.time() + 3600) * 1000),
            "scopes": ["user:inference"],
            "subscriptionType": "pro",
        }
    }
    cred_path.write_text(json.dumps(rotated))

    stale = {
        "claudeAiOauth": {
            "accessToken": "at-stale",
            "refreshToken": "rt-stale",
            "expiresAt": int((time.time() + 3600) * 1000),
            "scopes": ["user:inference"],
            "subscriptionType": "pro",
        }
    }
    creds = ClaudeCredentials(path=cred_path, data=dict(stale))
    creds._reload_if_disk_newer()
    assert creds.access_token() == "at-rotated"
