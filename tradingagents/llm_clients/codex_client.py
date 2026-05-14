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

from langchain_core.outputs import ChatGeneration, ChatResult

from .base_client import BaseLLMClient
from .codex_auth import CodexAuthError, CodexCredentials, load as load_codex_credentials
from .openai_client import NormalizedChatOpenAI
from .validators import validate_model

logger = logging.getLogger(__name__)


# The Responses-API endpoint the Codex CLI itself talks to. We pin this
# unconditionally inside ``get_llm`` — a stale ``backend_url`` left over
# from another provider would otherwise silently route ChatGPT-account
# credentials to the wrong host and 401 (or worse, leak the bearer
# token to an unintended endpoint).
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Default model when the user doesn't override. Tracks the OpenYak
# reference implementation's curated subscription model list — these
# IDs are what the ChatGPT backend accepts, NOT the Platform-API model
# slugs (e.g. it's ``gpt-5.5``, not ``gpt-5-turbo``).
DEFAULT_CODEX_MODEL = "gpt-5.5"


def _adapt_payload_for_wham(payload: dict) -> dict:
    """Translate a stock Responses-API payload to the shape WHAM accepts.

    WHAM (``chatgpt.com/backend-api/codex/responses``) is a Responses-API
    *variant*, not a strict superset. Two known divergences from native
    OpenAI Responses we hit live:

    1. ``instructions`` is mandatory. WHAM 400s with
       ``{"detail": "Instructions are required"}`` if the field is
       absent, even though native Responses treats the field as
       optional and accepts a ``system``-role item in ``input``
       instead. Langchain-openai sends system messages inline as input
       items (or as ``developer`` role on o-series models), never as
       a top-level ``instructions`` string, so we lift them ourselves.

    2. WHAM also doesn't want ``store: true``. Native Responses accepts
       this for server-side conversation persistence; on the WHAM
       endpoint it has no meaning (and the OpenYak reference impl
       explicitly forces ``store: false``). We force the same default
       — callers can still override via langchain kwargs if a future
       WHAM version changes the policy.

    Anything else we leave untouched; the tool-spec format, schema
    binding, and reasoning fields appear to round-trip cleanly.
    """
    # Extract and concatenate system / developer items from the input
    # array. We walk in order so the prompt sequence is preserved.
    remaining: list = []
    system_chunks: list[str] = []
    for item in payload.get("input", []) or []:
        if not isinstance(item, dict):
            remaining.append(item)
            continue
        role = item.get("role")
        if role in ("system", "developer"):
            content = item.get("content")
            system_chunks.append(_extract_text(content))
            continue
        remaining.append(item)

    if system_chunks:
        existing = payload.get("instructions") or ""
        merged = "\n\n".join(c for c in [existing, *system_chunks] if c)
        payload["instructions"] = merged
        payload["input"] = remaining

    # WHAM requires *some* instructions value. If after the lift we
    # still have nothing, supply a benign placeholder — sending an
    # empty user-only prompt is legal here, the rejection is purely on
    # the missing field shape.
    if not payload.get("instructions"):
        payload["instructions"] = "You are a helpful assistant."

    # WHAM rejects server-side conversation persistence.
    payload["store"] = False

    # WHAM only accepts streaming requests — non-stream calls are
    # rejected with ``{"detail": "Stream must be set to true"}``. The
    # OpenYak reference impl hardcodes ``stream: true`` for the same
    # reason. Langchain-openai's SDK handles SSE responses regardless
    # of whether the caller used ``.invoke`` or ``.stream``, so flipping
    # this flag is safe at the transport level — the higher-level
    # buffering happens inside ChatOpenAI's _generate path.
    payload["stream"] = True

    return payload


def _extract_text(content) -> str:
    """Flatten a Responses-API content value to a plain string.

    Inputs may be a bare string (langchain's simplest shape) or a list
    of typed blocks (``{"type": "input_text", "text": "..."}``). We
    only care about ``input_text`` blocks for the system-message lift;
    anything else (images, function-call output, etc.) shouldn't be
    routed into ``instructions`` and is dropped here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("input_text", "text"):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "\n".join(p for p in parts if p)
    return ""


class ChatCodexSubscription(NormalizedChatOpenAI):
    """ChatOpenAI bound to the ChatGPT-subscription backend.

    Inherits all the standard structured-output / tool-binding behavior
    from :class:`NormalizedChatOpenAI` — it's the same Responses API at
    a different URL with a different auth scheme — and overrides
    ``_get_request_payload`` only to refresh the bearer token before
    each outgoing request.

    Where the token actually lives matters. ``langchain-openai`` sends
    Responses calls through ``self.root_client.responses.create``
    (see ``langchain_openai/chat_models/base.py``), and the OpenAI SDK
    builds the ``Authorization`` header from ``root_client.api_key``
    at request time (``openai/_client.py`` ``auth_headers``). The
    sub-resources accessed via ``self.client`` /  ``self.async_client``
    share auth with their root and don't carry their own key. So token
    rotation has to write ``root_client.api_key`` (and the async
    counterpart) — writing ``self.client.api_key`` is a no-op.
    """

    # Declared so Pydantic accepts the field on construction; the
    # parent class is a Pydantic model.
    codex_credentials: Optional[CodexCredentials] = None

    model_config = {"arbitrary_types_allowed": True}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Aggregate the streaming Responses-API output into a ChatResult.

        WHAM only accepts ``stream=true`` requests (a 400 is returned
        otherwise — confirmed by live probe). Langchain-openai's
        ``_generate`` method calls ``responses.create`` and expects a
        unary ``Response`` object, which collides with WHAM's streaming
        requirement and crashes with ``'Stream' object has no
        attribute 'error'``. The corresponding streaming method is
        ``_stream_responses``, so we route ``.invoke()`` callers
        through that and collect the chunks into one message.

        This is the canonical "stream-then-aggregate" pattern documented
        for LangChain chat models that only support streaming I/O.
        """
        chunks = list(
            self._stream_responses(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        )
        if not chunks:
            return ChatResult(generations=[])

        message = chunks[0].message
        for chunk in chunks[1:]:
            message = message + chunk.message

        # Use the final chunk's generation_info (carries usage / finish_reason).
        info = chunks[-1].generation_info or {}
        return ChatResult(
            generations=[ChatGeneration(message=message, generation_info=info)]
        )

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        creds = self.codex_credentials
        if creds is not None:
            fresh = creds.access_token()
            # Rotate the credential on the root clients — that's the
            # only place the SDK reads it from when assembling the
            # Authorization header. Guard each write with hasattr so
            # the rotation degrades gracefully if a future
            # langchain-openai release moves these fields around.
            root = getattr(self, "root_client", None)
            if root is not None and hasattr(root, "api_key"):
                root.api_key = fresh
            root_async = getattr(self, "root_async_client", None)
            if root_async is not None and hasattr(root_async, "api_key"):
                root_async.api_key = fresh

        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        return _adapt_payload_for_wham(payload)


class CodexClient(BaseLLMClient):
    """Factory wrapper that wires a Codex/ChatGPT-subscription chat model.

    Behaves like :class:`~tradingagents.llm_clients.openai_client.OpenAIClient`
    from the graph's perspective — same Responses-API path, same tool /
    structured-output capabilities — except the bearer token comes from
    ``~/.codex/auth.json`` and the request includes the
    ``ChatGPT-Account-Id`` header that the subscription backend requires.

    Auth is validated eagerly in ``__init__`` rather than ``get_llm`` so
    a misconfigured environment fails at construction with a clear
    ``CodexAuthError`` — before the graph wires it up and a 30-minute
    analysis run discovers the problem mid-flight.
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

        if base_url and base_url != CODEX_BASE_URL:
            # A caller-provided base_url is almost always a leftover
            # from another provider (e.g. TRADINGAGENTS_LLM_BACKEND_URL
            # pointing at api.openai.com). Routing ChatGPT-account
            # credentials there 401s at best and exfiltrates the token
            # at worst, so we refuse to use it.
            logger.warning(
                "Ignoring caller-provided base_url %r for codex provider; "
                "ChatGPT-subscription auth only works against the WHAM "
                "endpoint and will be pinned to %s.",
                base_url, CODEX_BASE_URL,
            )

        # Fail-fast: load credentials now so a missing/broken session
        # surfaces at construction rather than at first request.
        try:
            self._credentials = load_codex_credentials()
        except CodexAuthError as exc:
            raise CodexAuthError(
                f"Cannot initialise the Codex/ChatGPT-subscription provider: {exc}"
            ) from exc

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()

        access_token = self._credentials.access_token()
        account_id = self._credentials.account_id

        # The ChatGPT backend requires both Authorization (handled by
        # ChatOpenAI's bearer-token logic) AND a custom account-id
        # header. ChatOpenAI exposes default_headers for exactly this
        # case; LangChain passes them through to the underlying openai
        # SDK client.
        default_headers = {"ChatGPT-Account-Id": account_id}

        llm_kwargs: dict[str, Any] = {
            "model": self.model,
            # Pinned, NOT honouring self.base_url — see __init__ rationale.
            "base_url": CODEX_BASE_URL,
            "api_key": access_token,
            "default_headers": default_headers,
            # The subscription backend speaks the Responses API just
            # like native OpenAI, so this stays consistent with the
            # OpenAI provider's wiring.
            "use_responses_api": True,
            # Stash credentials on the model so each request can refresh
            # before sending. This is the field declared on
            # ChatCodexSubscription above.
            "codex_credentials": self._credentials,
        }

        for key in self._PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        return ChatCodexSubscription(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("codex", self.model)
