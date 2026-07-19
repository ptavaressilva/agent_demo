"""Local tools backed by Postgres: persist discovered house listings, record
the agent's fit rating for each one, and save draft (never auto-sent)
viewing requests.

Tools are built per-session via `build_listing_tools(pool, session_id)` so the
LLM never has to pass `session_id` itself -- it's bound from the graph's
invocation context, which keeps a listing's identity tied to the
conversation that discovered it.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
from langchain_core.tools import BaseTool, tool

from agent_demo.config import settings

_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "db" / "schema.sql"

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Process-wide connection pool, created lazily on first use."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=settings.postgres_dsn, min_size=1, max_size=5)
        await ensure_schema(_pool)
    return _pool


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA_PATH.read_text())


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def build_listing_tools(pool: asyncpg.Pool, session_id: str) -> list[BaseTool]:
    """Return the four listing-persistence tools bound to `session_id`."""

    @tool
    async def save_house_listing(
        source_url: str,
        title: str,
        price: str = "",
        location: str = "",
        description: str = "",
        property_type: str = "",
        listed_at: str = "",
        raw_snippet: str = "",
    ) -> str:
        """Persist a house listing you found so it can be rated and, if it's
        a good fit, drafted into a viewing request. Call this once per
        distinct listing before rating or drafting for it. Returns the
        listing's internal id (an integer as a string) -- use that id in
        rate_listing and draft_viewing_request.
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO house_listings
                    (session_id, source_url, title, price, location,
                     description, property_type, listed_at, raw_snippet)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (session_id, source_url) DO UPDATE SET
                    title = EXCLUDED.title,
                    price = EXCLUDED.price,
                    location = EXCLUDED.location,
                    description = EXCLUDED.description,
                    property_type = EXCLUDED.property_type,
                    listed_at = EXCLUDED.listed_at,
                    raw_snippet = EXCLUDED.raw_snippet
                RETURNING id
                """,
                session_id,
                source_url,
                title,
                price,
                location,
                description,
                property_type,
                listed_at,
                raw_snippet,
            )
            return str(row["id"])

    @tool
    async def rate_listing(
        listing_id: str,
        score: int,
        rationale: str,
        matched_features: list[str] | None = None,
        missing_features: list[str] | None = None,
    ) -> str:
        """Record your fit assessment of a saved house listing against the
        buyer's profile. `score` is 0-100 (100 = perfect fit).
        `rationale` should explain the score in 1-3 sentences.
        listing_id must come from a prior save_house_listing call.
        """
        if not 0 <= score <= 100:
            return f"Error: score must be between 0 and 100, got {score}."
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO listing_ratings
                        (listing_id, score, rationale, matched_features, missing_features)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (listing_id) DO UPDATE SET
                        score = EXCLUDED.score,
                        rationale = EXCLUDED.rationale,
                        matched_features = EXCLUDED.matched_features,
                        missing_features = EXCLUDED.missing_features,
                        rated_at = now()
                    """,
                    int(listing_id),
                    score,
                    rationale,
                    matched_features or [],
                    missing_features or [],
                )
            except (ValueError, asyncpg.ForeignKeyViolationError):
                return (
                    f"Error: no house listing with id {listing_id!r} for this session. "
                    "Call save_house_listing first."
                )
        return f"Rated listing {listing_id}: {score}/100."

    @tool
    async def list_rated_listings(min_score: int = 0) -> str:
        """List house listings saved in this session, with their fit score
        if rated, sorted best-fit first. Use this to decide which listings
        are worth drafting a viewing request for.
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT l.id, l.title, l.price, l.location, l.source_url,
                       r.score, r.rationale
                FROM house_listings l
                LEFT JOIN listing_ratings r ON r.listing_id = l.id
                WHERE l.session_id = $1 AND COALESCE(r.score, 0) >= $2
                ORDER BY r.score DESC NULLS LAST, l.discovered_at DESC
                """,
                session_id,
                min_score,
            )
        if not rows:
            return "No house listings saved yet for this session."
        lines = [
            f"- id={r['id']} score={r['score'] if r['score'] is not None else 'unrated'} "
            f"{r['title']!r} at {r['location']!r} ({r['price'] or 'price n/a'}) "
            f"({r['source_url']}) "
            f"{'- ' + r['rationale'] if r['rationale'] else ''}"
            for r in rows
        ]
        return "\n".join(lines)

    @tool
    async def draft_viewing_request(
        listing_id: str,
        inquiry_message: str,
        buyer_highlights: str,
        notes_for_buyer: str = "",
    ) -> str:
        """Save a draft inquiry message and buyer highlights tailored to a
        specific house listing for the BUYER to review and send themselves.
        This tool never contacts the listing agent or schedules anything --
        it only persists a draft. Only draft for listings the buyer would
        want to view (check list_rated_listings first).
        """
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO viewing_requests
                        (listing_id, inquiry_message, buyer_highlights, notes_for_buyer)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (listing_id) DO UPDATE SET
                        inquiry_message = EXCLUDED.inquiry_message,
                        buyer_highlights = EXCLUDED.buyer_highlights,
                        notes_for_buyer = EXCLUDED.notes_for_buyer,
                        status = 'draft',
                        created_at = now()
                    """,
                    int(listing_id),
                    inquiry_message,
                    buyer_highlights,
                    notes_for_buyer,
                )
            except (ValueError, asyncpg.ForeignKeyViolationError):
                return (
                    f"Error: no house listing with id {listing_id!r} for this session. "
                    "Call save_house_listing first."
                )
        return (
            f"Saved a draft viewing request for listing {listing_id}. "
            "The buyer must review and send it themselves -- nothing was sent."
        )

    return [save_house_listing, rate_listing, list_rated_listings, draft_viewing_request]
