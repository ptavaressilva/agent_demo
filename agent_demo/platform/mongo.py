"""Process-wide Mongo client, shared by the kill switch, memory backends, and
main.py's admin kill-switch endpoint -- there's exactly one Mongo connection
pool per process, not one per caller or per agent.
"""

from __future__ import annotations

from pymongo import MongoClient

from agent_demo.platform.config import platform_settings

_mongo_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(platform_settings.mongo_uri)
    return _mongo_client
