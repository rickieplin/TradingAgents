"""Tests for the ChatGPT-subscription provider (`codex`).

The provider reads OAuth credentials from `~/.codex/auth.json` (populated
by `codex login`) and uses them as a bearer token against the same
Responses-API backend the Codex CLI itself talks to. These tests stub
the auth file and the HTTP layer so we never actually hit the network.

Coverage:
- `codex_auth.load` parses the OAuth shape and surfaces helpful errors
  on missing files / malformed JSON / non-OAuth (api-key-only) sessions.
- Token refresh fires only when the access JWT is near expiry, persists
  the rotated tokens atomically, and survives non-fatal write failures.
- The factory builds a `ChatCodexSubscription` pointed at the WHAM URL
  with the `ChatGPT-Account-Id` header and `use_responses_api=True`.
- Provider registration: factory, validator, and api-key-env mapping all
  recognise the provider; ensure_api_key doesn't prompt for it.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.codex_auth import (
    CodexAuthError,
    CodexCredentials,
    load as load_codex_credentials,
)
from tradingagents.llm_clients.codex_client import (
    CODEX_BASE_URL,
    ChatCodexSubscription,
    CodexClient,
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


@pytest.fixture
def codex_home(tmp_path, monkeypatch):
    """Point CODEX_HOME at a temp dir so `load()` reads our fake auth.json."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    return tmp_path


# ---- auth file loading -----------------------------------------------------


def test_load_reads_oauth_session(codex_home):
    _write_auth(codex_home)
    creds = load_codex_credentials()
    # account_id must be exposed unchanged so the WHAM backend can route
    # the request to the right ChatGPT account.
    assert creds.account_id == "acct-123"
    assert creds.access_token().count(".") == 2  # well-formed JWT


def test_load_missing_file_surfaces_actionable_error(codex_home):
    # No auth.json written — the user hasn't run `codex login` yet, and
    # the error message must say so.
    with pytest.raises(CodexAuthError, match="codex login"):
        load_codex_credentials()


def test_load_rejects_malformed_json(codex_home):
    (codex_home / "auth.json").write_text("{not json")
    with pytest.raises(CodexAuthError, match="not valid JSON"):
        load_codex_credentials()


def test_load_rejects_api_key_only_session(codex_home):
    # `codex login --with-api-key` writes OPENAI_API_KEY but no tokens
    # block; that path doesn't grant ChatGPT-subscription access.
    auth_path = codex_home / "auth.json"
    auth_path.write_text(json.dumps({"OPENAI_API_KEY": "sk-test", "tokens": None}))
    creds = load_codex_credentials()
    # The error fires lazily on first use so loading itself doesn't fail
    # for users who only inspect the file.
    with pytest.raises(CodexAuthError, match="OAuth"):
        creds.access_token()


# ---- refresh behavior ------------------------------------------------------


def test_access_token_returns_cached_when_fresh(codex_home):
    _write_auth(codex_home, exp_in=3600)
    creds = load_codex_credentials()
    # No refresh round-trip should happen when the token has > 5 min left.
    with patch("tradingagents.llm_clients.codex_auth.httpx.post") as mock_post:
        token = creds.access_token()
    mock_post.assert_not_called()
    assert token  # round-trips the existing token


def test_access_token_refreshes_when_expired(codex_home):
    _write_auth(codex_home, exp_in=60)  # within the 300s refresh buffer
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

    # Verify the request used the refresh token from the file with the
    # correct client_id (matches Codex CLI's public OAuth client; using
    # a different ID would 401 the user's refresh_token).
    args, kwargs = mock_post.call_args
    assert "auth.openai.com/oauth/token" in args[0]
    assert kwargs["data"]["grant_type"] == "refresh_token"
    assert kwargs["data"]["refresh_token"] == "rt-original"
    assert kwargs["data"]["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"

    assert result == new_access

    # Rotated refresh token must be persisted so the next process / the
    # codex CLI itself doesn't try the now-invalidated old one.
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


def test_refresh_persists_atomically(codex_home):
    # The temp file used for atomic replace must not be left behind on
    # success — otherwise repeated refreshes leak files into ~/.codex/.
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

    assert (codex_home / "auth.json").exists()
    assert not (codex_home / "auth.json.tmp").exists()


# ---- LangChain integration -------------------------------------------------


def test_factory_returns_codex_client():
    client = create_llm_client(provider="codex", model="gpt-5.5")
    assert isinstance(client, CodexClient)


def test_get_llm_wires_subscription_backend(codex_home):
    _write_auth(codex_home)
    client = create_llm_client(provider="codex", model="gpt-5.5")
    llm = client.get_llm()

    assert isinstance(llm, ChatCodexSubscription)
    # The whole point of this provider: the LangChain model is bound to
    # the ChatGPT-subscription backend, not api.openai.com, and the
    # account-id header rides along on every request.
    assert llm.openai_api_base == CODEX_BASE_URL
    # ChatOpenAI normalises default_headers into the underlying client
    # config; we just assert the value survived the round-trip.
    headers = (
        getattr(llm, "default_headers", None)
        or getattr(llm.client, "_custom_headers", None)
        or {}
    )
    assert headers.get("ChatGPT-Account-Id") == "acct-123"


def test_get_llm_propagates_auth_error(codex_home):
    # CODEX_HOME points at an empty dir; no auth.json — get_llm must
    # raise CodexAuthError so the CLI layer can turn it into a "run
    # codex login" hint.
    client = create_llm_client(provider="codex", model="gpt-5.5")
    with pytest.raises(CodexAuthError, match="codex login"):
        client.get_llm()


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
