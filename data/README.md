# data/

| Path | Contents | Phase |
|------|----------|-------|
| `generator/` | deterministic (seeded) synthetic knowledge-base generator | 2 |
| `generated/` | generator output — **git-ignored**, rebuilt with `python -m data.generator --seed 42` | 2 |
| `evaluation/` | versioned evaluation dataset (`vN/…`) — committed | 4 |

The synthetic company (auth / payments / checkout / notification / user-profile / inventory /
orders services) is generated as a coherent environment with planted ground-truth chains, so
retrieval and generation can be evaluated against known-correct evidence.
