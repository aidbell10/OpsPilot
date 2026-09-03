# Security

OpsPilot is an AI system that reads a knowledge base and (later) drives an agent. The threat
model treats **all retrieved content as untrusted** and constrains what the system can do.

## Threats and mitigations

| Threat | Mitigation | Phase |
|--------|------------|-------|
| **Prompt injection inside retrieved documents** ("ignore previous instructions…") | Retrieved text is inserted as clearly-delimited *data*, never as instructions; the system prompt states that document content is untrusted; adversarial eval cases with injected instructions verify the model does not comply | 3 / 12 |
| **LLM invents citations** | Citations must resolve to ids that were actually in the retrieved set; the citation verifier drops/flags the rest; unresolved-claim ratio is a tracked metric | 3 / 4 |
| **Malformed LLM output** | All model output is parsed through Pydantic schemas; parse failure → structured error / abstention, never a crash or a silent pass | 3 |
| **Arbitrary tool execution by the agent** | Tools are an explicit allowlist, all read-only and returning typed objects; no shell, no arbitrary SQL, no deploys, no infra mutation | 8 |
| **Runaway agent loops / cost** | Hard budgets: max tool calls, max investigation cost, per-call timeouts, retry limits, token caps; exceeding a budget is an explicit terminal state | 8 |
| **SQL injection** | No string-built SQL; SQLAlchemy Core/ORM with bound parameters everywhere; agent tools take typed arguments, not query fragments | 1+ |
| **Unsafe document ingestion** | Ingestion only reads files from the project's own `data/` tree; size/type limits; no code execution during parsing | 3 |
| **Secret leakage** | Secrets come only from env vars; never logged (structlog processors), never attached to OpenTelemetry spans, never returned in API responses or error details | 1 / 11 |
| **Unrestricted API use** | Request size limits, input length caps on every endpoint; rate limiting at deploy time | 3 / 14 |
| **Remediation actions applied automatically** | OpsPilot only *recommends*; any action requires explicit human approval | 8 |

## Practices

- `.env` is git-ignored; `.env.example` documents variables with no real values.
- Containers run as a non-root user.
- CI runs static/security checks (`ruff`, `bandit`, `pip-audit`) on every PR (Phase 13).
- Dependencies are pinned via `uv.lock`.
