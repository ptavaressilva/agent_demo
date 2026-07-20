"""Arize AX tracing via OpenTelemetry + OpenInference's LangChain
auto-instrumentation. Called once per process by `harness.run()` before the
graph is built -- instrumentation patches `langchain_core` globally, so every
LLM call, tool call, and graph node in every session after that point is
captured as a span and exported to Arize AX, for whichever agent this
process is running.

Every agent gets this unconditionally: there is no hook on `AgentSpec` for an
agent to opt out or substitute a different tracing backend.
"""

from __future__ import annotations

import logging

from agent_demo.platform.config import platform_settings

logger = logging.getLogger(__name__)

_configured = False


def configure_tracing(project_name: str) -> None:
    global _configured
    if _configured:
        return
    if not platform_settings.tracing_enabled:
        logger.info("Tracing disabled (TRACING_ENABLED=false); skipping Arize AX setup.")
        return
    if not platform_settings.arize_space_id or not platform_settings.arize_api_key:
        logger.warning(
            "ARIZE_SPACE_ID / ARIZE_API_KEY not set; skipping Arize AX tracing setup."
        )
        return

    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        space_id=platform_settings.arize_space_id,
        api_key=platform_settings.arize_api_key,
        project_name=project_name,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _configured = True
    logger.info("Arize AX tracing configured for project %r.", project_name)
