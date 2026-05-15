from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client",
)


_EFFORT_TO_BUDGET = {"low": 2048, "medium": 8000, "high": 16000}


def effort_to_thinking_kwargs(
    effort: Optional[str],
    explicit_max_tokens: Optional[int] = None,
) -> dict:
    """Translate an effort label (low/medium/high) to Anthropic's thinking config.

    Returns a kwargs dict to be merged into ChatAnthropic construction. Empty
    dict when effort is None or unknown — callers can spread it unconditionally.

    Bumps max_tokens to budget_tokens + 1024 when the caller didn't pass a
    larger value, because Anthropic's API rejects requests where
    thinking.budget_tokens >= max_tokens.
    """
    if not effort:
        return {}
    budget = _EFFORT_TO_BUDGET.get(effort.lower())
    if budget is None:
        return {}
    out: dict = {"thinking": {"type": "enabled", "budget_tokens": budget}}
    needed_max = budget + 1024
    if explicit_max_tokens is None or explicit_max_tokens < needed_max:
        out["max_tokens"] = needed_max
    return out


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        thinking_kwargs = effort_to_thinking_kwargs(
            self.kwargs.get("effort"),
            llm_kwargs.get("max_tokens"),
        )
        llm_kwargs.update(thinking_kwargs)

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
