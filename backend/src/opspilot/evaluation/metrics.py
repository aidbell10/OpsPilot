"""Pure evaluation metric functions.

No DB, no I/O, no LLM calls — every function here is a hand-verifiable
computation over plain Python values, exercised directly by
``tests/unit/test_evaluation_metrics.py`` against hand-computed expected
numbers.

Convention: a case's "required" set (documents / evidence snippets) can be
empty (the 4 unanswerable cases in the current dataset have none). An empty
required set is treated as vacuously satisfied — the retrieval-quality
metrics below report ``1.0`` for it, because there is nothing to recall,
cover, or rank. Whether the *system* behaved correctly on such a case is
judged by the abstention metrics instead, not by these.

All bounded metrics live in ``[0.0, 1.0]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def recall_at_k(retrieved_keys: list[str], required_keys: list[str]) -> float:
    """Fraction of ``required_keys`` present anywhere in ``retrieved_keys``."""
    required = set(required_keys)
    if not required:
        return 1.0
    retrieved = set(retrieved_keys)
    return len(required & retrieved) / len(required)


def evidence_coverage(retrieved_texts: list[str], required_snippets: list[str]) -> float:
    """Fraction of ``required_snippets`` found as a substring in any retrieved text.

    Case-insensitive: the snippets are exact tokens (error codes, function
    names, version strings) that should appear verbatim in the corpus, but
    matching case-insensitively is more forgiving of incidental case drift
    without weakening the check (these tokens are distinctive enough that a
    case-insensitive false positive is not a realistic concern).
    """
    if not required_snippets:
        return 1.0
    haystack = "\n".join(retrieved_texts).lower()
    found = sum(1 for snippet in required_snippets if snippet.lower() in haystack)
    return found / len(required_snippets)


def mrr(retrieved_keys_ordered: list[str], required_keys: list[str]) -> float:
    """Reciprocal rank (1-indexed) of the first retrieved key that is required."""
    required = set(required_keys)
    if not required:
        return 1.0
    for rank, key in enumerate(retrieved_keys_ordered, start=1):
        if key in required:
            return 1.0 / rank
    return 0.0


def ndcg_at_10(retrieved_keys_ordered: list[str], required_keys: list[str]) -> float:
    """nDCG@10 with binary relevance (a retrieved key is relevant iff it is required).

    IDCG is built from the count of relevant *items actually retrieved*
    (capped at 10), not from ``len(required_keys)``: several retrieved keys
    (chunks) can belong to the same required document, in which case the
    number of relevant retrieved items exceeds the number of required
    documents. Capping IDCG at the required-document count would then let
    DCG exceed IDCG, producing an nDCG above 1.0 — capping at the actual
    relevant-item count keeps IDCG a true upper bound on DCG.
    """
    required = set(required_keys)
    if not required:
        return 1.0
    top = retrieved_keys_ordered[:10]
    relevances = [1.0 if key in required else 0.0 for key in top]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    relevant_total = sum(1 for key in retrieved_keys_ordered if key in required)
    ideal_count = min(relevant_total, 10)
    if ideal_count == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    return dcg / idcg


def citation_precision(cited_keys: list[str], required_keys: list[str]) -> float | None:
    """Fraction of cited evidence keys that belong to a required document.

    The citations passed in are assumed already verified against the
    retrieved evidence set (see ``opspilot.generation.parser.parse_llm_output``)
    — this measures *relevance* of citations, not fabrication, which is
    handled upstream.

    Returns ``None`` (undefined) when there is nothing to score: no
    citations were made, or the case has no required documents (e.g. an
    unanswerable case). Callers must exclude ``None`` from aggregation
    rather than treating it as ``0`` — averaging in a 0 would penalize
    correctly-abstained cases for citing nothing, which is the opposite of
    what abstention is supposed to do.
    """
    if not cited_keys or not required_keys:
        return None
    required = set(required_keys)
    hits = sum(1 for key in cited_keys if key in required)
    return hits / len(cited_keys)


def hallucinated_forbidden_claim(text: str, forbidden_claims: list[str]) -> bool:
    """True if any forbidden-claim substring (case-insensitive) appears in ``text``.

    A deliberately simple, judge-free heuristic — see ``docs/evaluation.md``
    for why this is not a substitute for a semantic faithfulness judge.
    """
    if not forbidden_claims:
        return False
    lowered = text.lower()
    return any(claim.lower() in lowered for claim in forbidden_claims)


@dataclass(frozen=True, slots=True)
class AbstentionCounts:
    """A confusion-matrix tally where the positive class is "should abstain"."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) else 0.0


def abstention_counts(*, should_abstain: bool, did_abstain: bool) -> AbstentionCounts:
    if should_abstain and did_abstain:
        return AbstentionCounts(true_positive=1)
    if not should_abstain and did_abstain:
        return AbstentionCounts(false_positive=1)
    if should_abstain and not did_abstain:
        return AbstentionCounts(false_negative=1)
    return AbstentionCounts(true_negative=1)


def sum_abstention_counts(counts: list[AbstentionCounts]) -> AbstentionCounts:
    return AbstentionCounts(
        true_positive=sum(c.true_positive for c in counts),
        false_positive=sum(c.false_positive for c in counts),
        false_negative=sum(c.false_negative for c in counts),
        true_negative=sum(c.true_negative for c in counts),
    )
