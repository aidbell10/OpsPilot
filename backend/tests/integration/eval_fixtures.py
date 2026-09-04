"""A tiny, hand-written ground-truth set matching ``corpus_fixtures.tiny_corpus()``.

Deliberately not the real ``data/generated/ground_truth.json`` (git-ignored,
may not exist in a fresh checkout) — small, deterministic, self-contained.
"""

from __future__ import annotations

from opspilot.evaluation.loader import GroundTruthCase


def tiny_ground_truth() -> list[GroundTruthCase]:
    return [
        GroundTruthCase(
            case_id="gt-checkout-500-01",
            query=(
                "Checkout is returning HTTP 500 after a deploy that touched promotion codes. Why?"
            ),
            answerable=True,
            difficulty="straightforward",
            category="root_cause",
            service_name="checkout",
            expected_root_cause="promotion validation raised on empty discount code",
            acceptable_actions=["Patch validate_cart to allow an empty discount code"],
            forbidden_claims=["the database was down"],
            required_document_source_paths=["runbooks/checkout/500s.md"],
            required_evidence_snippets=["promotion validation", "validate_cart"],
            chain_id="chain-checkout-500s",
            related_deployment_version="v2.14.0",
        ),
        GroundTruthCase(
            case_id="unans-checkout-cpu-01",
            query="What is the current CPU utilisation of the checkout database?",
            answerable=False,
            difficulty="unanswerable",
            category="abstention",
            service_name="checkout",
            expected_root_cause=None,
            acceptable_actions=["State that the available evidence is insufficient to answer"],
            forbidden_claims=["% CPU"],
            required_document_source_paths=[],
            required_evidence_snippets=[],
            chain_id=None,
            related_deployment_version=None,
        ),
    ]
