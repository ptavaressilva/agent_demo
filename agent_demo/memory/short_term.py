"""Short-term memory: per-session conversation state.

Backed by LangGraph's `MongoDBSaver` checkpointer. This persists the full
message history and graph state for a `thread_id` (== our `session_id`), so a
session can be resumed across process restarts / AgentCore invocations.
Scope is a single ongoing job search conversation -- it is *not* shared
across sessions (that's what long-term memory is for).
"""

from __future__ import annotations

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from agent_demo.config import settings


def build_checkpointer(client: MongoClient) -> MongoDBSaver:
    return MongoDBSaver(
        client=client,
        db_name=settings.mongo_db_name,
        checkpoint_collection_name="short_term_checkpoints",
        writes_collection_name="short_term_checkpoint_writes",
    )
