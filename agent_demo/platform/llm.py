"""Every model call, for every agent, goes through an LLM gateway (LiteLLM
proxy -- see `litellm_config.yaml`) rather than calling a provider directly.
The gateway holds the real per-provider credentials and does the actual
routing/retries/rate limiting/spend tracking; this process only ever needs a
single gateway key. `ChatOpenAI` is the client here (not `ChatAnthropic`)
because the gateway speaks the OpenAI-compatible `/chat/completions` API
regardless of which upstream provider `model` resolves to -- swapping
providers or adding new ones is a change to `litellm_config.yaml`, not to
this file or to any agent's code.

Which model names to request (`primary_model`/`fallback_model`) is each
agent's own choice, supplied as parameters here -- `AgentSpec` implementations
own that choice in their own config, not this module, since it's a business
decision (cost/quality tradeoff per agent), not a deployment one. What *isn't*
a per-agent choice is whether the call reaches the gateway at all: there is no
hook anywhere in `AgentSpec` for an agent to construct its own model client,
so `build_tools`/`build_initial_domain_state`/etc. never see one.

Fallback triggers on the primary model raising (rate limit, overload,
timeout, 5xx, etc.) per LangChain's default `RunnableWithFallbacks` behavior
-- not on content-based judgments. Tools must be bound to *both* models
before the fallback is attached, otherwise a fallback invocation silently
loses tool access.

Gateway key resolution has two modes (`LLM_GATEWAY_AUTH_MODE`):
  - "env" (default, local dev): read `LLM_GATEWAY_API_KEY` directly.
  - "agentcore_identity" (deployed): resolve the key per-call from an
    AgentCore Identity API-key credential provider, so the key never has to
    be baked into the container image or set as a plain runtime env var.
    Create the provider once with `agentcore identity add-api-key-provider`
    and set BEDROCK_AGENTCORE_MODEL_PROVIDER_API_KEY_NAME to its name (see
    deployment/README.md).
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from agent_demo.platform.config import platform_settings


async def _resolve_gateway_api_key() -> str:
    if platform_settings.llm_gateway_auth_mode == "agentcore_identity":
        from bedrock_agentcore.identity.auth import requires_api_key

        @requires_api_key(provider_name=platform_settings.agentcore_model_provider_api_key_name)
        async def _fetch(*, api_key: str) -> str:
            return api_key

        return await _fetch()

    if not platform_settings.llm_gateway_api_key:
        raise RuntimeError(
            "LLM_GATEWAY_API_KEY is not set and LLM_GATEWAY_AUTH_MODE is 'env'. "
            "Either set LLM_GATEWAY_API_KEY, or set LLM_GATEWAY_AUTH_MODE=agentcore_identity "
            "to resolve it from AgentCore Identity instead."
        )
    return platform_settings.llm_gateway_api_key


async def _base_model(model_name: str) -> ChatOpenAI:
    api_key = await _resolve_gateway_api_key()
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=platform_settings.llm_gateway_base_url,
        max_tokens=8000,
        timeout=120,
        max_retries=2,
    )


async def build_chat_model_with_fallback(
    tools: list[BaseTool], *, primary_model: str, fallback_model: str
) -> Runnable:
    """Return a `primary_model` model bound to `tools`, falling back to
    `fallback_model` (also bound to `tools`) if the primary call raises."""
    primary: BaseChatModel = (await _base_model(primary_model)).bind_tools(tools)
    fallback: BaseChatModel = (await _base_model(fallback_model)).bind_tools(tools)
    return primary.with_fallbacks([fallback])
