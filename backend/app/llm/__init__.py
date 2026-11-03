"""Etapa 3a: abstração de LLM + provedores. A escolha vem de LLM_PROVIDER."""

from app.config import get_settings
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMError, LLMProvider
from app.llm.ollama_provider import OllamaProvider


def create_llm(provider: str | None = None) -> LLMProvider:
    """Instancia o provedor configurado (ou o indicado em `provider`)."""
    settings = get_settings()
    provider = provider or settings.llm_provider
    if provider == "anthropic":
        api_key = settings.anthropic_api_key
        return AnthropicProvider(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            effort=settings.anthropic_effort,
            api_key=api_key.get_secret_value() if api_key else None,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaProvider(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            max_tokens=settings.llm_max_tokens,
            num_ctx=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise ValueError(f"LLM_PROVIDER desconhecido: '{provider}' (use 'ollama' ou 'anthropic')")


__all__ = ["AnthropicProvider", "LLMError", "LLMProvider", "OllamaProvider", "create_llm"]
