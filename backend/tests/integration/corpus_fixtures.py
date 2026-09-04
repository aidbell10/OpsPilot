"""A tiny, hand-written corpus shared by ingestion and endpoint integration tests.

Deliberately not the real ``data/generated`` output (which is git-ignored and
may not exist in a fresh checkout) — small, deterministic, and self-contained.
"""

from __future__ import annotations

import datetime as dt

from opspilot.ingestion.loader import (
    Corpus,
    DeploymentRecord,
    DocumentRecord,
    HistoricalIncidentRecord,
    ServiceRecord,
)


def tiny_corpus() -> Corpus:
    return Corpus(
        services=[
            ServiceRecord(
                name="checkout", description="cart + checkout", tier=1, owning_team="commerce"
            ),
        ],
        documents=[
            DocumentRecord(
                doc_id="runbook-checkout-500s",
                document_type="runbook",
                title="Checkout 500s troubleshooting",
                content=(
                    "If checkout returns HTTP 500 after a deploy, check promotion validation. "
                    "The validate_cart function raises when a promotion code is empty but "
                    "present in the request payload. "
                )
                * 10,
                service_name="checkout",
                version="v2.14.0",
                environment="production",
                source_path="runbooks/checkout/500s.md",
                doc_timestamp=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            ),
        ],
        deployments=[
            DeploymentRecord(
                service_name="checkout",
                version="v2.14.0",
                environment="production",
                deployed_at=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                status="succeeded",
                change_summary="promotion validation rewrite",
                changed_components={"functions": ["validate_cart"]},
            ),
        ],
        historical_incidents=[
            HistoricalIncidentRecord(
                incident_id="inc-checkout-500s-01",
                title="Checkout 500s after promotion rewrite",
                service_name="checkout",
                occurred_at=dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
                severity="sev2",
                symptoms="Checkout returns HTTP 500 for carts with a promotion applied.",
                root_cause="promotion validation raised on empty discount code",
                resolution="patched validation to allow empty discount code",
                related_deployment_version="v2.14.0",
                linked_document_source_path="runbooks/checkout/500s.md",
            ),
        ],
    )
