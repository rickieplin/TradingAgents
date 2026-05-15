"""Model name validators for each provider."""

from .model_catalog import get_known_models


VALID_MODELS = {
    provider: models
    for provider, models in get_known_models().items()
    if provider not in ("ollama", "openrouter", "codex", "claude_subscription")
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For ollama, openrouter, codex, claude_subscription - any model is
    accepted (subscription / pull-based providers rotate supported model
    IDs faster than this catalog is updated, and users may legitimately
    name custom ones).
    """
    provider_lower = provider.lower()

    if provider_lower in ("ollama", "openrouter", "codex", "claude_subscription"):
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]
