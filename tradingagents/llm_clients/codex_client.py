"""ChatGPT-subscription provider — uses the user's `codex login` session.

The Codex CLI maintains an OAuth session at ``~/.codex/auth.json`` after
the user runs ``codex login``. This provider reuses that session to talk
directly to the same OpenAI backend the CLI uses
(``https://chatgpt.com/backend-api/codex``), which is a Responses-API
variant that supports the standard tool-calling and structured-output
flows. The result: TradingAgents nodes route through the user's ChatGPT
subscription rather than the Platform API — same model quality, no
per-token billing — while still emitting OpenAI ``tool_calls`` so the
analyst loop and Pydantic-bound managers work unchanged.

Auth and refresh live in :mod:`tradingagents.llm_clients.codex_auth`;
this module is just the LangChain glue.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .base_client import BaseLLMClient
from .codex_auth import CodexAuthError, CodexCredentials, load as load_codex_credentials
from .openai_client import NormalizedChatOpenAI
from .validators import validate_model

logger = logging.getLogger(__name__)


# The Responses-API endpoint the Codex CLI itself talks to. Path is
# ``/responses`` — ChatOpenAI appends that automatically when
# ``use_responses_api=True``, so we point it at the base only.
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Default model when the user doesn't override. Tracks the OpenYak
# reference implementation's curated subscription model list — these
# IDs are what the ChatGPT backend accepts, NOT the Platform-API model
# slugs (e.g. it's ``gpt-5.5``, not ``gpt-5-turbo``).
DEFAULT_CODEX_MODEL = "gpt-5.5"


class ChatCodexSubscription(NormalizedChatOpenAI):
    """ChatOpenAI bound to the ChatGPT-subscription backend.

    Inherits all the standard structured-output / tool-binding behavior
    from :class:`NormalizedChatOpenAI` (it's just OpenAI's own Responses
    API at a different URL with a different auth scheme), and overrides
    the request payload to inject the ``ChatGPT-Account-Id`` header that
    the backend requires alongside the bearer token.

    Token refresh is handled at construction time by
    :class:`CodexCredentials`. The token can live for an hour, so for
    long-running graph executions we refresh through a thin httpx
    ``event_hook`` that re-reads the credentials before each request.
    """

    # Pydantic-mode subclassing: declare the field so the parent
    # ``__init__`` accepts it instead of rejecting it as unknown.
    codex_credentials: Optional[CodexCredentials] = None

    model_config = {"arbitrary_types_allowed": True}

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        # Refresh the bearer token before every request. The base class
        # binds api_key once at construction; we override here so a
        # long-running graph that crosses the token-expiry boundary
        # picks up the new token transparently.
        if self.codex_credentials is not None:
            fresh = self.codex_credentials.access_token()
            # ChatOpenAI stores the key on the underlying client's
            # ``api_key`` SecretStr; rewriting it is the supported way
            # to rotate the credential mid-flight.
            try:
                self.client.api_key = fresh
                self.async_client.api_key = fresh
            except AttributeError:
                # Some langchain-openai versions stash these differently;
                # the default_headers path below is the load-bearing one.
                pass
        return super()._get_request_payload(input_, stop=stop, **kwargs)


class CodexClient(BaseLLMClient):
    """Factory wrapper that wires a Codex/ChatGPT-subscription chat model.

    Behaves like :class:`~tradingagents.llm_clients.openai_client.OpenAIClient`
    from the graph's perspective — same Responses-API path, same tool /
    structured-output capabilities — except the bearer token comes from
    ``~/.codex/auth.json`` and the request includes the
    ``ChatGPT-Account-Id`` header that the subscription backend requires.
    """

    # Forwarded ChatOpenAI kwargs (mirrors the OpenAI client's list).
    _PASSTHROUGH_KWARGS = (
        "timeout", "max_retries", "reasoning_effort",
        "callbacks", "http_client", "http_async_client",
    )

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = "codex"

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()

        try:
            credentials = load_codex_credentials()
        except CodexAuthError as exc:
            # The CLI flow turns this into a user-facing message; the
            # graph flow lets the exception propagate so the operator
            # sees exactly why init failed.
            raise CodexAuthError(
                f"Cannot initialise the Codex/ChatGPT-subscription provider: {exc}"
            ) from exc

        access_token = credentials.access_token()
        account_id = credentials.account_id

        # The ChatGPT backend requires both Authorization (handled by
        # ChatOpenAI's bearer-token logic) AND a custom account-id
        # header. ChatOpenAI exposes default_headers for exactly this
        # case; LangChain passes them through to the underlying openai
        # SDK client.
        default_headers = {"ChatGPT-Account-Id": account_id}

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            "base_url": self.base_url or CODEX_BASE_URL,
            "api_key": access_token,
            "default_headers": default_headers,
            # The subscription backend speaks the Responses API just
            # like native OpenAI, so this stays consistent with the
            # OpenAI provider's wiring.
            "use_responses_api": True,
            # Stash credentials on the model so each request can refresh
            # before sending. This is the field declared on
            # ChatCodexSubscription above.
            "codex_credentials": credentials,
        }

        for key in self._PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return ChatCodexSubscription(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("codex", self.model)
