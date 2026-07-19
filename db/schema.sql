-- Postgres schema for job offers, ratings, and (draft-only) application materials.
-- Applied automatically by agent_demo.tools.postgres_tools.ensure_schema() on first
-- use, and can also be applied manually with:
--   psql "$POSTGRES_DSN" -f db/schema.sql

CREATE TABLE IF NOT EXISTS job_offers (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    title           TEXT NOT NULL,
    company         TEXT,
    location        TEXT,
    description     TEXT,
    salary_range    TEXT,
    posted_at       TEXT,
    raw_snippet     TEXT,
    discovered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, source_url)
);

CREATE TABLE IF NOT EXISTS job_ratings (
    id              BIGSERIAL PRIMARY KEY,
    job_offer_id    BIGINT NOT NULL REFERENCES job_offers(id) ON DELETE CASCADE,
    score           SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
    rationale       TEXT NOT NULL,
    matched_skills  TEXT[] NOT NULL DEFAULT '{}',
    missing_skills  TEXT[] NOT NULL DEFAULT '{}',
    rated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_offer_id)
);

-- "Apply" is draft-only by design: the agent prepares tailored materials for a
-- human to review and submit. No submission ever happens from this table.
CREATE TABLE IF NOT EXISTS application_drafts (
    id                  BIGSERIAL PRIMARY KEY,
    job_offer_id        BIGINT NOT NULL REFERENCES job_offers(id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'reviewed', 'discarded')),
    cover_letter        TEXT NOT NULL,
    resume_highlights   TEXT NOT NULL,
    notes_for_candidate TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_offer_id)
);

CREATE INDEX IF NOT EXISTS idx_job_offers_session ON job_offers (session_id);
CREATE INDEX IF NOT EXISTS idx_job_ratings_score ON job_ratings (score DESC);
