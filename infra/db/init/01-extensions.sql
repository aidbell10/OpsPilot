-- Enable the extensions OpsPilot relies on. Alembic migration 0001 also runs
-- CREATE EXTENSION IF NOT EXISTS, so this is belt-and-braces for fresh volumes
-- and for tools that connect before migrations run.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
