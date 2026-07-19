"""Claude Opus 4.8 as the primary model, with Claude Haiku 4.5 as an automatic
fallback via LangChain's `.with_fallbacks(...)`.

Fallback triggers on the primary model raising (rate limit, overload, timeout,
5xx, etc.) per LangChain's default `RunnableWithFallbacks` behavior -- not on
content-based judgments. Tools must be bound to *both* models before the
fallback is attached, otherwise a fallback invocation silently loses tool
access.

API key resolution has two modes (`ANTHROPIC_AUTH_MODE`):
  - "env" (default, local dev): read `ANTHROPIC_API_KEY` directly.
  - "agentcore_identity" (deployed): resolve the key per-call from an
    AgentCore Identity API-key credential provider, so the key never has to
    be baked into the container image or set as a plain runtime env var.
    Create the provider once with `agentcore identity add-api-key-provider`
    and set BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME to its name (see
    deployment/README.md).
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from agent_demo.config import settings


async def _resolve_api_key() -> str:
    if settings.anthropic_auth_mode == "agentcore_identity":
        from bedrock_agentcore.identity.auth import requires_api_key

        @requires_api_key(provider_name=settings.agentcore_model_provider_api_key_name)
        async def _fetch(*, api_key: str) -> str:
            return api_key

        return await _fetch()

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set and ANTHROPIC_AUTH_MODE is 'env'. "
            "Either set ANTHROPIC_API_KEY, or set ANTHROPIC_AUTH_MODE=agentcore_identity "
            "to resolve it from AgentCore Identity instead."
        )
    return settings.anthropic_api_key


async def _base_model(model_name: str) -> ChatAnthropic:
    api_key = await _resolve_api_key()
    return ChatAnthropic(
        model=model_name,
        api_key=api_key,
        max_tokens=8000,
        timeout=120,
        max_retries=2,
    )


async def build_chat_model_with_fallback(tools: list[BaseTool]) -> Runnable:
    """Return an Opus-4.8 model bound to `tools`, falling back to Haiku 4.5
    (also bound to `tools`) if the primary call raises."""
    primary: BaseChatModel = (await _base_model(settings.primary_model)).bind_tools(tools)
    fallback: BaseChatModel = (await _base_model(settings.fallback_model)).bind_tools(tools)
    return primary.with_fallbacks([fallback])
