"""Claude Pro/Max subscription provider — uses the user's `claude` OAuth session.

The Claude Code CLI maintains an OAuth session in the macOS keychain
(or ``~/.claude/.credentials.json`` on other OSes) after the user signs
in to their Claude Pro/Max subscription. This provider reuses that
session as the bearer token against the standard Anthropic API
(``https://api.anthropic.com``), so TradingAgents nodes route through
the user's subscription quota instead of a Platform API key — same
model quality, same tool-calling and structured-output flows, no
per-token billing.

Auth and refresh live in :mod:`tradingagents.llm_clients.claude_auth`;
this module is just the LangChain glue.

Why we override _generate / _stream / _agenerate / _astream
----------------------------------------------------------
``langchain-anthropic`` builds two ``anthropic`` SDK clients (a sync
``_client`` and an async ``_async_client``) at construction time using
whatever ``api_key`` was passed to ``__init__``. The SDK's
``Anthropic.auth_headers`` property emits either ``X-Api-Key`` (if
``api_key`` is set) or ``Authorization: Bearer ...`` (if ``auth_token``
is set, with ``api_key=None``).

For OAuth we want the Bearer form *only* — sending both headers is
either rejected outright or treated as Platform-API auth on some
versions, neither of which we want. So after construction we null the
SDK's ``api_key`` and write our rotated OAuth token into ``auth_token``
before each request. Doing this in the request-path methods (rather
than only in ``__init__``) is what handles token expiry between calls
in a long-running agent loop.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .anthropic_client import NormalizedChatAnthropic, effort_to_thinking_kwargs
from .base_client import BaseLLMClient
from .claude_auth import (
    ClaudeAuthError,
    ClaudeCredentials,
    load as load_claude_credentials,
)
from .validators import validate_model

logger = logging.getLogger(__name__)


# Pinned base URL — ChatGPT-account credentials must never be sent
# anywhere else, and the same rule applies here. A stale
# ``backend_url`` from another provider would otherwise route OAuth
# tokens to an unintended host.
CLAUDE_SUBSCRIPTION_BASE_URL = "https://api.anthropic.com"

# The Anthropic API requires this beta header for OAuth-based auth.
# Without it the API returns 4xx even with a valid bearer token.
_OAUTH_BETA_HEADER = "oauth-2025-04-20"

# Pydantic on ChatAnthropic requires a non-empty ``anthropic_api_key``
# in some versions. The actual auth header is built from
# ``_client.auth_token`` after construction, so the value here is
# placeholder-only and we null ``_client.api_key`` immediately to
# guarantee no ``X-Api-Key`` ever ships on the wire.
_PLACEHOLDER_API_KEY = "oauth-bearer-placeholder"


class ChatClaudeSubscription(NormalizedChatAnthropic):
    """ChatAnthropic bound to the user's Claude Code OAuth session.

    Inherits all the standard tool-binding / structured-output behaviour
    from :class:`NormalizedChatAnthropic` — the wire protocol is the
    same Anthropic API, only the auth scheme differs — and overrides
    the four request-entry methods (``_generate``, ``_agenerate``,
    ``_stream``, ``_astream``) to rotate the OAuth bearer token before
    each outgoing request.

    Where the token actually lives matters. ``langchain-anthropic``
    dispatches via ``self._client`` / ``self._async_client``; the
    ``anthropic`` SDK reads the bearer token off ``self._client.auth_token``
    at request time when ``api_key`` is ``None``. So token rotation
    writes those attributes — writing ``self.anthropic_api_key`` is a
    no-op because the SDK clients were built once and don't re-read
    that field.
    """

    # Declared so Pydantic accepts the field on construction; the
    # parent class is a Pydantic model.
    claude_credentials: Optional[ClaudeCredentials] = None

    model_config = {"arbitrary_types_allowed": True}

    def _rotate_token(self) -> None:
        """Stamp the latest OAuth bearer token onto both SDK clients.

        Called before each request entry point. We also null
        ``api_key`` defensively in case some code path tried to set it
        — the SDK's ``auth_headers`` property prefers ``api_key`` over
        ``auth_token`` when both are present, which would silently
        switch us to ``X-Api-Key`` and break OAuth auth.
        """
        creds = self.claude_credentials
        if creds is None:
            return
        fresh = creds.access_token()
        client = getattr(self, "_client", None)
        if client is not None:
            if hasattr(client, "api_key"):
                client.api_key = None
            if hasattr(client, "auth_token"):
                client.auth_token = fresh
        async_client = getattr(self, "_async_client", None)
        if async_client is not None:
            if hasattr(async_client, "api_key"):
                async_client.api_key = None
            if hasattr(async_client, "auth_token"):
                async_client.auth_token = fresh

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._rotate_token()
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._rotate_token()
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self._rotate_token()
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._rotate_token()
        async for chunk in super()._astream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk


class ClaudeSubscriptionClient(BaseLLMClient):
    """Factory wrapper that wires a Claude Pro/Max subscription chat model.

    Behaves like :class:`~tradingagents.llm_clients.anthropic_client.AnthropicClient`
    from the graph's perspective — same Messages API, same tool /
    structured-output capabilities — except the bearer token comes from
    the user's Claude Code OAuth session instead of an env-var API key,
    and the ``anthropic-beta: oauth-2025-04-20`` header is added so the
    API accepts OAuth bearer auth.

    Auth is validated eagerly in ``__init__`` rather than ``get_llm`` so
    a misconfigured environment fails at construction with a clear
    ``ClaudeAuthError`` — before the graph wires it up and a long
    analysis run discovers the problem mid-flight.
    """

    # Forwarded ChatAnthropic kwargs (mirrors the Anthropic client's list).
    _PASSTHROUGH_KWARGS = (
        "timeout", "max_retries", "max_tokens",
        "callbacks", "http_client", "http_async_client",
    )

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = "claude_subscription"

        if base_url and base_url != CLAUDE_SUBSCRIPTION_BASE_URL:
            # A caller-provided base_url is almost always a leftover
            # from another provider. Routing the OAuth bearer token
            # anywhere other than api.anthropic.com is at best useless
            # and at worst an unintended credential leak, so we refuse.
            logger.warning(
                "Ignoring caller-provided base_url %r for claude_subscription "
                "provider; OAuth bearer auth only works against the Anthropic "
                "API and will be pinned to %s.",
                base_url, CLAUDE_SUBSCRIPTION_BASE_URL,
            )

        # Fail-fast: load credentials now so a missing/broken session
        # surfaces at construction rather than at first request.
        try:
            self._credentials = load_claude_credentials()
        except ClaudeAuthError as exc:
            raise ClaudeAuthError(
                f"Cannot initialise the Claude subscription provider: {exc}"
            ) from exc

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()

        # Touch the credentials once so any expired-token refresh has
        # already happened by the time the first request goes out.
        access_token = self._credentials.access_token()

        # ``anthropic-beta`` is mandatory for OAuth auth on Anthropic's
        # API — without it the server rejects the bearer token. We pass
        # it via ``default_headers`` so the SDK merges it into every
        # outgoing request.
        default_headers = {"anthropic-beta": _OAUTH_BETA_HEADER}

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            # Pinned, NOT honouring self.base_url — see __init__ rationale.
            "anthropic_api_url": CLAUDE_SUBSCRIPTION_BASE_URL,
            # Placeholder only — Pydantic on ChatAnthropic rejects an
            # empty ``anthropic_api_key`` in some versions. We null the
            # SDK's api_key right after construction and rely on
            # ``auth_token`` instead.
            "anthropic_api_key": _PLACEHOLDER_API_KEY,
            "default_headers": default_headers,
            # Stash credentials on the model so each request can refresh
            # before sending. This is the field declared on
            # ChatClaudeSubscription above.
            "claude_credentials": self._credentials,
        }

        for key in self._PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        thinking_kwargs = effort_to_thinking_kwargs(
            self.kwargs.get("effort"),
            llm_kwargs.get("max_tokens"),
        )
        llm_kwargs.update(thinking_kwargs)

        llm = ChatClaudeSubscription(**llm_kwargs)

        # Critical: null the SDK's api_key and stamp the OAuth token
        # before any request goes out. Without this, the auth_headers
        # property would emit ``X-Api-Key: oauth-bearer-placeholder``,
        # which the Anthropic API would reject (or treat as Platform-
        # API auth and silently bill the wrong account).
        for client_attr in ("_client", "_async_client"):
            client = getattr(llm, client_attr, None)
            if client is None:
                continue
            if hasattr(client, "api_key"):
                client.api_key = None
            if hasattr(client, "auth_token"):
                client.auth_token = access_token

        return llm

    def validate_model(self) -> bool:
        return validate_model("claude_subscription", self.model)
