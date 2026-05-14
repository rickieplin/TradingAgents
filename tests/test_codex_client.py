"""Tests for the ChatGPT-subscription provider (`codex`).

The provider reads OAuth credentials from `~/.codex/auth.json` (populated
by `codex login`) and uses them as a bearer token against the same
Responses-API backend the Codex CLI itself talks to. These tests stub
the auth file and the HTTP layer so we never actually hit the network.

Coverage:
- `codex_auth.load` parses the OAuth shape and surfaces helpful errors
  on missing files / malformed JSON / non-OAuth (api-key-only) sessions.
- Token refresh fires only when the JWT exp is within the buffer, rotates
  both access and refresh tokens, and persists with 0o600 mode.
- Refresh-loop guard kicks in when the OAuth endpoint hands back a
  malformed/expired token.
- `load()` returns the same instance for the same path (singleton) so
  quick + deep LLM slots share refresh state instead of racing.
- The factory builds a `ChatCodexSubscription` pinned to the WHAM URL
  with the `ChatGPT-Account-Id` header, `use_responses_api=True`, and
  ignores any caller-provided base_url (security: a stale BACKEND_URL
  would otherwise leak ChatGPT-account auth to the wrong host).
- Token rotation in `_get_request_payload` writes `root_client.api_key`
  (the field the OpenAI SDK actually reads when assembling the
  Authorization header).
- The WHAM payload adapter lifts system messages into `instructions`,
  forces `stream=true`, and disables `store`.
- Provider registration: factory, validator, and api-key-env mapping all
  recognise the provider; ensure_api_key doesn't prompt for it.
"""

from __future__ import annotations

import base64
import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.codex_auth import (
    CodexAuthError,
    CodexCredentials,
    _reset_cache_for_tests,
    load as load_codex_credentials,
)
from tradingagents.llm_clients.codex_client import (
    CODEX_BASE_URL,
    ChatCodexSubscription,
    CodexClient,
    _adapt_payload_for_wham,
)
from tradingagents.llm_clients.validators import validate_model


# ---- fixtures --------------------------------------------------------------


def _make_jwt(payload: dict) -> str:
    """Build a fake JWT (no signature verification needed; we only decode the payload)."""
    header_b64 = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"{header_b64}.{payload_b64}.sig"


def _write_auth(path: Path, *, exp_in: int = 3600, with_api_key: bool = False) -> Path:
    """Write an auth.json with an OAuth session whose access token expires in `exp_in` seconds."""
    access = _make_jwt({"exp": int(time.time()) + exp_in})
    data = {
        "OPENAI_API_KEY": "sk-legacy" if with_api_key else None,
        "tokens": {
            "id_token": _make_jwt({"email": "test@example.com"}),
            "access_token": access,
            "refresh_token": "rt-original",
            "account_id": "acct-123",
        },
        "last_refresh": "2025-01-01T00:00:00Z",
    }
    auth_path = path / "auth.json"
    auth_path.write_text(json.dumps(data))
    return auth_path


@pytest.fixture(autouse=True)
def _reset_credentials_cache():
    """Clear the load() singleton cache between tests so each test sees a fresh instance."""
    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    """Point CODEX_HOME at a temp dir so `load()` reads our fake auth.json."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


# ---- auth file loading -----------------------------------------------------


def test_load_reads_oauth_session(codex_home):
    _write_auth(codex_home)
    creds = load_codex_credentials()
    assert creds.account_id == "acct-123"
    assert creds.access_token().count(".") == 2


def test_load_missing_file_surfaces_actionable_error(codex_home):
    with pytest.raises(CodexAuthError, match="codex login"):
        load_codex_credentials()


def test_load_rejects_malformed_json(codex_home):
    (codex_home / "auth.json").write_text("{not json")
    with pytest.raises(CodexAuthError, match="not valid JSON"):
        load_codex_credentials()


def test_load_rejects_api_key_only_session(codex_home):
    auth_path = codex_home / "auth.json"
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-test", "tokens": None}))
    creds = load_codex_credentials()
    with pytest.raises(CodexAuthError, match="OAuth"):
        creds.access_token()


def test_load_returns_singleton_for_same_path(codex_home):
    # Per-path singleton: without it, quick and deep LLM slots would
    # each create their own CodexCredentials with their own threading
    # lock, and a simultaneous expiry would let both refresh in
    # parallel — burning the one-time refresh_token.
    _write_auth(codex_home)
    a = load_codex_credentials()
    b = load_codex_credentials()
    assert a is b


# ---- refresh behavior ------------------------------------------------------


def test_access_token_returns_cached_when_fresh(codex_home):
    _write_auth(codex_home, exp_in=3600)
    creds = load_codex_credentials()
    with patch("tradingagents.llm_clients.codex_auth.httpx.post") as mock_post:
        token = creds.access_token()
    mock_post.assert_not_called()
    assert token


def test_access_token_refreshes_when_expired(codex_home):
    _write_auth(codex_home, exp_in=60)  # within refresh buffer
    creds = load_codex_credentials()

    new_access = _make_jwt({"exp": int(time.time()) + 3600})
    fake_response = type("R", (), {
        "status_code": 200,
        "json": staticmethod(lambda: {
            "access_token": new_access,
            "refresh_token": "rt-rotated",
            "expires_in": 3600,
        }),
        "text": "",
    })()

    with patch("tradingagents.llm_clients.codex_auth.httpx.post", return_value=fake_response) as mock_post:
        result = creds.access_token()

    args, kwargs = mock_post.call_args
    assert "auth.openai.com/oauth/token" in args[0]
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "rt-original"
    assert kwargs["data"]["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"

    assert result == new_access

    # Rotated refresh token must be persisted so the next refresh — or
    # the codex CLI itself — doesn't try the now-invalidated old one.
    persisted = json.loads((codex_home / "auth.json").read_text())
    assert persisted["tokens"]["refresh_token"] == "rt-rotated"
    assert persisted["tokens"]["access_token"] == new_access


def test_refresh_failure_raises_actionable_error(codex_home):
    _write_auth(codex_home, exp_in=60)
    creds = load_codex_credentials()

    fake_response = type("R", (), {
        "status_code": 400,
        "json": staticmethod(lambda: {"error": "invalid_grant"}),
        "text": '{"error":"invalid_grant"}',
    })()

    with patch("tradingagents.llm_clients.codex_auth.httpx.post", return_value=fake_response):
        with pytest.raises(CodexAuthError, match="HTTP 400"):
            creds.access_token()


def test_refresh_loop_guard_raises_after_repeated_bad_tokens(codex_home):
    # If the OAuth endpoint hands back an access token that already
    # looks expired, we'd otherwise re-refresh on every request. The
    # guard caps consecutive failures and raises a hard error so the
    # user sees the upstream issue instead of an infinite refresh.
    _write_auth(codex_home, exp_in=60)
    creds = load_codex_credentials()

    expired_access = _make_jwt({"exp": int(time.time()) - 1000})
    fake_response = type("R", (), {
        "status_code": 200,
        "json": staticmethod(lambda: {
            "access_token": expired_access,
            "refresh_token": "rt-rotated",
        }),
        "text": "",
    })()

    with patch("tradingagents.llm_clients.codex_auth.httpx.post", return_value=fake_response):
        # Each call raises individually first (soft fail); after enough
        # consecutive failures the guard switches to a hard error.
        for _ in range(3):
            with pytest.raises(CodexAuthError):
                creds.access_token()
        with pytest.raises(CodexAuthError, match="Refusing to loop"):
            creds.access_token()


def test_persisted_auth_file_is_mode_0o600(codex_home):
    # Defence in depth: the tmp file is created with O_CREAT|O_EXCL
    # and mode 0o600 atomically, so credentials are never briefly
    # visible under the user's umask-default mode (e.g. 0o644).
    _write_auth(codex_home, exp_in=60)
    creds = load_codex_credentials()

    new_access = _make_jwt({"exp": int(time.time()) + 3600})
    fake_response = type("R", (), {
        "status_code": 200,
        "json": staticmethod(lambda: {"access_token": new_access, "refresh_token": "rt-rotated"}),
        "text": "",
    })()

    with patch("tradingagents.llm_clients.codex_auth.httpx.post", return_value=fake_response):
        creds.access_token()

    mode = stat.S_IMODE((codex_home / "auth.json").stat().st_mode)
    assert mode == 0o600, f"expected mode 0o600 after refresh, got 0o{mode:o}"


def test_force_refresh_validates_session_eagerly(codex_home):
    # The CLI preflight uses this to detect a revoked refresh token
    # *before* a 30-minute analysis run starts. A bare account_id read
    # is not enough — it would pass even when the refresh token has
    # been rotated/revoked since `codex login`.
    _write_auth(codex_home, exp_in=3600)  # access still fresh!
    creds = load_codex_credentials()

    fake_response = type("R", (), {
        "status_code": 401,
        "json": staticmethod(lambda: {"error": "invalid_grant"}),
        "text": '{"error":"invalid_grant"}',
    })()

    with patch("tradingagents.llm_clients.codex_auth.httpx.post", return_value=fake_response):
        with pytest.raises(CodexAuthError, match="HTTP 401"):
            creds.force_refresh()


# ---- WHAM payload adapter --------------------------------------------------


def test_adapter_lifts_system_messages_to_instructions():
    # WHAM rejects payloads without a top-level `instructions` field
    # (live-confirmed 400: "Instructions are required"). Langchain
    # sends system messages inline as input items; we lift them.
    payload = {
        "input": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
    }
    out = _adapt_payload_for_wham(payload)
    assert out["instructions"] == "be terse"
    # System item must be removed from input — duplicating it would
    # confuse the model (or be rejected by WHAM).
    assert all(item.get("role") != "system" for item in out["input"])
    assert out["input"] == [{"role": "user", "content": "hi"}]


def test_adapter_concatenates_multiple_system_items():
    payload = {
        "input": [
            {"role": "system", "content": "rule 1"},
            {"role": "user", "content": "q1"},
            {"role": "developer", "content": "rule 2"},
        ],
    }
    out = _adapt_payload_for_wham(payload)
    # Both system and developer roles get lifted; order preserved.
    assert "rule 1" in out["instructions"]
    assert "rule 2" in out["instructions"]


def test_adapter_handles_typed_content_blocks():
    # Responses API content can be a list of typed blocks; we flatten
    # input_text blocks into the instructions string.
    payload = {
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "be terse"}]},
            {"role": "user", "content": "hi"},
        ],
    }
    out = _adapt_payload_for_wham(payload)
    assert out["instructions"] == "be terse"


def test_adapter_supplies_placeholder_instructions_when_missing():
    payload = {"input": [{"role": "user", "content": "hi"}]}
    out = _adapt_payload_for_wham(payload)
    # WHAM requires *some* instructions value; a placeholder beats a 400.
    assert out["instructions"]


def test_adapter_forces_stream_true_and_store_false():
    # WHAM hard-requires streaming and rejects server-side persistence.
    payload = {"input": [{"role": "user", "content": "hi"}], "store": True, "stream": False}
    out = _adapt_payload_for_wham(payload)
    assert out["stream"] is True
    assert out["store"] is False


# ---- LangChain integration -------------------------------------------------


def test_factory_returns_codex_client(codex_home):
    _write_auth(codex_home)
    client = create_llm_client(provider="codex", model="gpt-5.5")
    assert isinstance(client, CodexClient)


def test_get_llm_wires_subscription_backend(codex_home):
    _write_auth(codex_home)
    client = create_llm_client(provider="codex", model="gpt-5.5")
    llm = client.get_llm()

    assert isinstance(llm, ChatCodexSubscription)
    # Bound to the ChatGPT subscription backend, not api.openai.com.
    assert llm.openai_api_base == CODEX_BASE_URL
    # ChatGPT-Account-Id header rides along on every request.
    headers = (
        getattr(llm, "default_headers", None)
        or getattr(llm.client, "_custom_headers", None)
        or {}
    )
    assert headers.get("ChatGPT-Account-Id") == "acct-123"


def test_get_llm_pins_base_url_ignoring_caller_override(codex_home):
    # A stale TRADINGAGENTS_LLM_BACKEND_URL would otherwise route
    # ChatGPT-account auth to api.openai.com (401 at best, token leak
    # at worst). The provider must pin the WHAM endpoint unconditionally.
    _write_auth(codex_home)
    client = create_llm_client(
        provider="codex",
        model="gpt-5.5",
        base_url="https://api.openai.com/v1",
    )
    llm = client.get_llm()
    assert llm.openai_api_base == CODEX_BASE_URL


def test_construction_fails_fast_on_missing_auth(codex_home):
    # CodexClient now loads credentials in __init__ (not get_llm), so
    # broken auth surfaces at config time rather than 30 minutes into
    # a graph execution.
    with pytest.raises(CodexAuthError, match="codex login"):
        create_llm_client(provider="codex", model="gpt-5.5")


def test_payload_rotates_root_client_api_key(codex_home):
    # The CRITICAL bug the GAN review caught: rewriting self.client
    # or self.async_client api_key is a no-op because LangChain
    # dispatches through root_client.responses, and the openai SDK
    # reads the bearer token from root_client.api_key at request time.
    # This test pins the contract so a future refactor can't regress it.
    _write_auth(codex_home, exp_in=3600)  # fresh; construction won't refresh
    client = create_llm_client(provider="codex", model="gpt-5.5")
    llm = client.get_llm()

    # Now force the next request to trigger a refresh by replacing the
    # cached access token with an expired one. The credentials object
    # is shared (singleton via load()), so editing it here is the same
    # state the request-time refresh sees.
    now = int(time.time())
    expired = _make_jwt({"exp": now - 10})
    llm.codex_credentials._tokens["access_token"] = expired

    new_access = _make_jwt({"exp": now + 3600})
    fake_response = type("R", (), {
        "status_code": 200,
        "json": staticmethod(lambda: {"access_token": new_access, "refresh_token": "rt-rotated"}),
        "text": "",
    })()

    with patch("tradingagents.llm_clients.codex_auth.httpx.post", return_value=fake_response):
        # _get_request_payload runs both refresh and adapter steps.
        llm._get_request_payload(
            [{"role": "user", "content": "ping"}],
        )

    assert llm.root_client.api_key == new_access
    assert llm.root_async_client.api_key == new_access


# ---- registration smoke tests ----------------------------------------------


def test_codex_validator_accepts_any_model():
    # Subscription models rotate faster than this catalog; the validator
    # must not gate user-typed IDs (matches ollama/openrouter behaviour).
    assert validate_model("codex", "gpt-5.5") is True
    assert validate_model("codex", "some-future-model") is True


def test_codex_api_key_env_is_none():
    # Credentials come from ~/.codex/auth.json, not an env var — so
    # ensure_api_key must NOT prompt the user when codex is selected.
    assert get_api_key_env("codex") is None
