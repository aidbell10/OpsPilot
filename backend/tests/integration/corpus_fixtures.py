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


def agent_corpus() -> Corpus:
    """A small multi-service corpus for the Phase 8 agent tool tests.

    Adds what ``tiny_corpus`` deliberately doesn't need: a service dependency
    graph (``get_service_dependencies``) and a historical incident carrying
    ``meta.error_code``/``meta.log_lines`` (``get_incident_logs``) — both real
    columns/fields, populated by the Phase 2 generator the same way.
    """
    base = tiny_corpus()
    return Corpus(
        services=[
            ServiceRecord(
                name="checkout",
                description="cart + checkout",
                tier=1,
                owning_team="commerce",
                depends_on=["payments"],
            ),
            ServiceRecord(name="payments", description="payments", tier=1, depends_on=["auth"]),
            ServiceRecord(name="auth", description="auth", tier=1),
        ],
        documents=base.documents,
        deployments=base.deployments,
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
                meta={
                    "error_code": "CHK-50500",
                    "log_lines": [
                        "2025-01-02T10:00:00+00:00 ERROR checkout.validate_cart "
                        "CHK-50500 promotion code empty but present",
                        "2025-01-02T10:00:05+00:00 WARN  checkout.place_order "
                        "returning 500 to client",
                    ],
                },
            ),
        ],
    )
