"""Regression tests for the ``effort`` → extended-thinking translation.

The CLI prompts the user for an effort level (low/medium/high) and stashes
it in client kwargs. Anthropic's API has no ``effort`` parameter — it has
``thinking={"type":"enabled","budget_tokens":N}``. Earlier code listed
``effort`` in ``_PASSTHROUGH_KWARGS`` and forwarded it verbatim, which
ChatAnthropic accepted into ``model_kwargs`` and then handed to
``messages.create(**payload)``, where the SDK raised
``TypeError: Messages.create() got an unexpected keyword argument 'effort'``.

These tests pin the helper's contract and verify both providers stop
leaking ``effort`` into ``model_kwargs``.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from tradingagents.llm_clients.anthropic_client import (
    AnthropicClient,
    effort_to_thinking_kwargs,
)


# ---- helper unit tests -----------------------------------------------------


def test_effort_none_returns_empty():
    assert effort_to_thinking_kwargs(None) == {}


def test_effort_unknown_returns_empty():
    assert effort_to_thinking_kwargs("garbage") == {}


def test_effort_low_sets_budget_and_max_tokens():
    assert effort_to_thinking_kwargs("low") == {
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "max_tokens": 3072,
    }


def test_effort_uppercase_normalised():
    out = effort_to_thinking_kwargs("HIGH")
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 16000}
    assert out["max_tokens"] == 17024


def test_explicit_max_tokens_above_budget_is_preserved():
    out = effort_to_thinking_kwargs("medium", explicit_max_tokens=50000)
    assert out == {"thinking": {"type": "enabled", "budget_tokens": 8000}}
    assert "max_tokens" not in out


def test_explicit_max_tokens_below_budget_is_bumped():
    out = effort_to_thinking_kwargs("medium", explicit_max_tokens=1024)
    assert out["max_tokens"] == 9024
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 8000}


# ---- behaviour: AnthropicClient --------------------------------------------


def test_anthropic_client_translates_effort_and_clears_passthrough():
    client = AnthropicClient(
        model="claude-opus-4-5",
        api_key="placeholder",
        effort="high",
    )
    llm = client.get_llm()
    assert llm.thinking == {"type": "enabled", "budget_tokens": 16000}
    # The crashing-path regression: model_kwargs must never carry `effort`
    # because ChatAnthropic forwards model_kwargs verbatim into
    # messages.create() and the Anthropic SDK rejects unknown kwargs.
    assert "effort" not in llm.model_kwargs


# ---- behaviour: ClaudeSubscriptionClient -----------------------------------


def _write_credentials(path, exp_in_seconds=3600):
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "at-test",
                    "refreshToken": "rt-test",
                    "expiresAt": int((time.time() + exp_in_seconds) * 1000),
                    "scopes": ["user:inference"],
                    "subscriptionType": "pro",
                }
            }
        )
    )
    return path


@pytest.fixture
def credentials_path(tmp_path, monkeypatch):
    cred = tmp_path / ".credentials.json"
    monkeypatch.setenv("CLAUDE_CREDENTIALS_PATH", str(cred))
    return cred


@pytest.fixture(autouse=True)
def _reset_credentials_cache():
    from tradingagents.llm_clients.claude_auth import _reset_cache_for_tests

    _reset_cache_for_tests()
    yield
    _reset_cache_for_tests()


def test_claude_subscription_client_translates_effort(credentials_path):
    _write_credentials(credentials_path)
    # Keep _read_keychain away from the real macOS keychain.
    with patch(
        "tradingagents.llm_clients.claude_auth._read_keychain",
        return_value=None,
    ):
        from tradingagents.llm_clients.claude_subscription_client import (
            ClaudeSubscriptionClient,
        )

        client = ClaudeSubscriptionClient(
            model="claude-haiku-4-5",
            effort="medium",
        )
        llm = client.get_llm()

    assert llm.thinking == {"type": "enabled", "budget_tokens": 8000}
    assert "effort" not in llm.model_kwargs
