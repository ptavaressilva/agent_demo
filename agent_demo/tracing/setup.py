"""Arize AX tracing via OpenTelemetry + OpenInference's LangChain
auto-instrumentation. Call `configure_tracing()` once at process startup,
before the graph is built -- instrumentation patches `langchain_core`
globally, so every LLM call, tool call, and graph node in every session
after that point is captured as a span and exported to Arize AX.
"""

from __future__ import annotations

import logging

from agent_demo.config import settings

logger = logging.getLogger(__name__)

_configured = False


def configure_tracing() -> None:
    global _configured
    if _configured:
        return
    if not settings.tracing_enabled:
        logger.info("Tracing disabled (TRACING_ENABLED=false); skipping Arize AX setup.")
        return
    if not settings.arize_space_id or not settings.arize_api_key:
        logger.warning(
            "ARIZE_SPACE_ID / ARIZE_API_KEY not set; skipping Arize AX tracing setup."
        )
        return

    from arize.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    tracer_provider = register(
        space_id=settings.arize_space_id,
        api_key=settings.arize_api_key,
        project_name=settings.arize_project_name,
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _configured = True
    logger.info("Arize AX tracing configured for project %r.", settings.arize_project_name)
