-- Postgres schema for house listings, ratings, and (draft-only) viewing
-- requests. Applied automatically by agent_demo.tools.postgres_tools.ensure_schema()
-- on first use, and can also be applied manually with:
--   psql "$POSTGRES_DSN" -f db/schema.sql

CREATE TABLE IF NOT EXISTS house_listings (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    title           TEXT NOT NULL,
    price           TEXT,
    location        TEXT,
    description     TEXT,
    property_type   TEXT,
    listed_at       TEXT,
    raw_snippet     TEXT,
    discovered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, source_url)
);

CREATE TABLE IF NOT EXISTS listing_ratings (
    id                  BIGSERIAL PRIMARY KEY,
    listing_id          BIGINT NOT NULL REFERENCES house_listings(id) ON DELETE CASCADE,
    score               SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
    rationale           TEXT NOT NULL,
    matched_features    TEXT[] NOT NULL DEFAULT '{}',
    missing_features    TEXT[] NOT NULL DEFAULT '{}',
    rated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (listing_id)
);

-- A "viewing request" is draft-only by design: the agent prepares a tailored
-- inquiry message for a human to review and send. No message is ever sent
-- from this table.
CREATE TABLE IF NOT EXISTS viewing_requests (
    id                  BIGSERIAL PRIMARY KEY,
    listing_id          BIGINT NOT NULL REFERENCES house_listings(id) ON DELETE CASCADE,
    status              TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'reviewed', 'discarded')),
    inquiry_message     TEXT NOT NULL,
    buyer_highlights    TEXT NOT NULL,
    notes_for_buyer     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (listing_id)
);

CREATE INDEX IF NOT EXISTS idx_house_listings_session ON house_listings (session_id);
CREATE INDEX IF NOT EXISTS idx_listing_ratings_score ON listing_ratings (score DESC);
