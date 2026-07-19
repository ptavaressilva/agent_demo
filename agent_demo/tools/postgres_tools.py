"""Local tools backed by Postgres: persist discovered job offers, record the
agent's fit rating for each one, and save draft (never auto-submitted)
application materials.

Tools are built per-session via `build_job_tools(pool, session_id)` so the
LLM never has to pass `session_id` itself -- it's bound from the graph's
invocation context, which keeps a job offer's identity tied to the
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


def build_job_tools(pool: asyncpg.Pool, session_id: str) -> list[BaseTool]:
    """Return the four job-persistence tools bound to `session_id`."""

    @tool
    async def save_job_offer(
        source_url: str,
        title: str,
        company: str = "",
        location: str = "",
        description: str = "",
        salary_range: str = "",
        posted_at: str = "",
        raw_snippet: str = "",
    ) -> str:
        """Persist a job posting you found so it can be rated and, if it's a
        good fit, drafted into an application. Call this once per distinct
        job posting before rating or drafting for it. Returns the job's
        internal id (an integer as a string) -- use that id in
        rate_job_offer and draft_application_materials.
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO job_offers
                    (session_id, source_url, title, company, location,
                     description, salary_range, posted_at, raw_snippet)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (session_id, source_url) DO UPDATE SET
                    title = EXCLUDED.title,
                    company = EXCLUDED.company,
                    location = EXCLUDED.location,
                    description = EXCLUDED.description,
                    salary_range = EXCLUDED.salary_range,
                    posted_at = EXCLUDED.posted_at,
                    raw_snippet = EXCLUDED.raw_snippet
                RETURNING id
                """,
                session_id,
                source_url,
                title,
                company,
                location,
                description,
                salary_range,
                posted_at,
                raw_snippet,
            )
            return str(row["id"])

    @tool
    async def rate_job_offer(
        job_offer_id: str,
        score: int,
        rationale: str,
        matched_skills: list[str] | None = None,
        missing_skills: list[str] | None = None,
    ) -> str:
        """Record your fit assessment of a saved job offer against the
        candidate's profile. `score` is 0-100 (100 = perfect fit).
        `rationale` should explain the score in 1-3 sentences.
        job_offer_id must come from a prior save_job_offer call.
        """
        if not 0 <= score <= 100:
            return f"Error: score must be between 0 and 100, got {score}."
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO job_ratings
                        (job_offer_id, score, rationale, matched_skills, missing_skills)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (job_offer_id) DO UPDATE SET
                        score = EXCLUDED.score,
                        rationale = EXCLUDED.rationale,
                        matched_skills = EXCLUDED.matched_skills,
                        missing_skills = EXCLUDED.missing_skills,
                        rated_at = now()
                    """,
                    int(job_offer_id),
                    score,
                    rationale,
                    matched_skills or [],
                    missing_skills or [],
                )
            except (ValueError, asyncpg.ForeignKeyViolationError):
                return (
                    f"Error: no job offer with id {job_offer_id!r} for this session. "
                    "Call save_job_offer first."
                )
        return f"Rated job {job_offer_id}: {score}/100."

    @tool
    async def list_rated_job_offers(min_score: int = 0) -> str:
        """List job offers saved in this session, with their fit score if
        rated, sorted best-fit first. Use this to decide which jobs are
        worth drafting an application for.
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT o.id, o.title, o.company, o.source_url,
                       r.score, r.rationale
                FROM job_offers o
                LEFT JOIN job_ratings r ON r.job_offer_id = o.id
                WHERE o.session_id = $1 AND COALESCE(r.score, 0) >= $2
                ORDER BY r.score DESC NULLS LAST, o.discovered_at DESC
                """,
                session_id,
                min_score,
            )
        if not rows:
            return "No job offers saved yet for this session."
        lines = [
            f"- id={r['id']} score={r['score'] if r['score'] is not None else 'unrated'} "
            f"{r['title']!r} at {r['company']!r} ({r['source_url']}) "
            f"{'- ' + r['rationale'] if r['rationale'] else ''}"
            for r in rows
        ]
        return "\n".join(lines)

    @tool
    async def draft_application_materials(
        job_offer_id: str,
        cover_letter: str,
        resume_highlights: str,
        notes_for_candidate: str = "",
    ) -> str:
        """Save a draft cover letter and resume highlights tailored to a
        specific job offer for the CANDIDATE to review and submit
        themselves. This tool never submits anything anywhere -- it only
        persists a draft. Only draft for jobs the candidate would want to
        apply to (check list_rated_job_offers first).
        """
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO application_drafts
                        (job_offer_id, cover_letter, resume_highlights, notes_for_candidate)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (job_offer_id) DO UPDATE SET
                        cover_letter = EXCLUDED.cover_letter,
                        resume_highlights = EXCLUDED.resume_highlights,
                        notes_for_candidate = EXCLUDED.notes_for_candidate,
                        status = 'draft',
                        created_at = now()
                    """,
                    int(job_offer_id),
                    cover_letter,
                    resume_highlights,
                    notes_for_candidate,
                )
            except (ValueError, asyncpg.ForeignKeyViolationError):
                return (
                    f"Error: no job offer with id {job_offer_id!r} for this session. "
                    "Call save_job_offer first."
                )
        return (
            f"Saved a draft application for job {job_offer_id}. "
            "The candidate must review and submit it themselves -- nothing was sent."
        )

    return [save_job_offer, rate_job_offer, list_rated_job_offers, draft_application_materials]
