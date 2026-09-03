# Deployment

> Status: containerisation exists from Phase 1; full deployment story is **Phase 14**.

## Local

`docker compose up` runs:

- `db` — `pgvector/pgvector:pg16`, healthchecked, named volume, extensions pre-created
- `api` — built from `backend/Dockerfile`, runs `alembic upgrade head` then `uvicorn`

The API image uses the deterministic fake providers by default, so it starts with no keys.
Set `OPSPILOT_LLM_PROVIDER=anthropic` + `OPSPILOT_ANTHROPIC_API_KEY` and
`OPSPILOT_EMBEDDING_PROVIDER=local` for real behaviour.

## Target (AWS-straightforward, no Kubernetes)

```
        ┌──────────────┐        ┌────────────────────┐
 users ─┤ frontend      │──────▶ │ FastAPI container   │
        │ (static/SSR)  │  HTTPS │ (ECS Fargate / App  │
        └──────────────┘        │  Runner)            │
                                └─────────┬──────────┘
                                          │
                                ┌─────────▼──────────┐
                                │ PostgreSQL+pgvector │
                                │ (RDS / Aurora)      │
                                └────────────────────┘
```

- Config + secrets via environment / SSM Parameter Store / Secrets Manager.
- Migrations run as a one-off task on deploy.
- Embedding model baked into the image (or a cached layer / EFS mount).
- No infrastructure added purely for résumé keywords.
