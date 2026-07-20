"""`FaqAgent`: the minimal second agent. Compare its size to
`agent_demo.agents.house_search.spec.HouseSearchAgent` -- it has no Postgres
pool, no MCP tools, no human-in-the-loop tool, and no bespoke config beyond
its own model choice and budget defaults, yet it is onboarded onto the exact
same platform harness (kill switch, tracing, gateway routing, budget
ceilings) with zero changes to `agent_demo/platform/` or `main.py`.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import Field

from agent_demo.agents.faq_agent.config import faq_agent_config
from agent_demo.agents.faq_agent.prompts import SYSTEM_PROMPT
from agent_demo.agents.faq_agent.state import FaqAgentState
from agent_demo.agents.faq_agent.tools import build_faq_tools
from agent_demo.platform.envelope import BaseInvokeEnvelope
from agent_demo.platform.spec import RequestResources


class FaqAgentRequest(BaseInvokeEnvelope):
    topic: str | None = Field(default=None, max_length=200)


class FaqAgent:
    agent_id = "faq-agent"
    request_schema = FaqAgentRequest
    state_schema = FaqAgentState

    primary_model = faq_agent_config.primary_model
    fallback_model = faq_agent_config.fallback_model

    default_max_react_steps = faq_agent_config.default_max_react_steps
    default_max_self_correction_retries = faq_agent_config.default_max_self_correction_retries

    def build_initial_domain_state(self, request: FaqAgentRequest) -> dict:
        return {"topic": request.topic}

    def render_system_prompt(self, state: FaqAgentState) -> str:
        return SYSTEM_PROMPT

    async def build_tools(self, resources: RequestResources) -> list[BaseTool]:
        return build_faq_tools()
