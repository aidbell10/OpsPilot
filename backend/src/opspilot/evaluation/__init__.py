"""Evaluation harness (Phase 4).

Versioned dataset loading, metric implementations (Recall@K, evidence
coverage@K, MRR, nDCG@10, groundedness, hallucination rate, citation
precision, abstention P/R/F1, latency percentiles, token/cost), run
persistence with full configuration + git SHA, and comparison reports.

Metrics are computed from real system output only — never fabricated.
"""
