"""The fictional company: Meridian Retail's internal platform.

Seven services with a real dependency graph, owning teams, and — critically — a
per-service *catalog* of exact identifiers (function names, error-code prefixes,
pinned dependency versions, config keys). Chains embed these verbatim so lexical
retrieval sometimes beats vector search, which is the whole point of the Phase 5
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPANY_NAME = "Meridian Retail"
PLATFORM_NAME = "Meridian"
GITHUB_ORG = "meridian-retail"


@dataclass(frozen=True, slots=True)
class ServiceCatalog:
    name: str
    description: str
    tier: int
    owning_team: str
    depends_on: tuple[str, ...]
    error_prefix: str
    functions: tuple[str, ...]
    config_keys: tuple[str, ...]
    datastore: str

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{GITHUB_ORG}/{self.name}"


SERVICES: tuple[ServiceCatalog, ...] = (
    ServiceCatalog(
        name="auth",
        description=(
            "Authentication and session tokens: password and OAuth login, JWT issuance, "
            "JWKS rotation, refresh-token exchange, and session revocation."
        ),
        tier=1,
        owning_team="identity",
        depends_on=(),
        error_prefix="AUTH",
        functions=(
            "verify_token",
            "issue_token",
            "refresh_session",
            "rotate_jwks",
            "revoke_token",
            "validate_audience",
        ),
        config_keys=(
            "AUTH_JWKS_CACHE_TTL_S",
            "AUTH_ACCESS_TOKEN_TTL_S",
            "AUTH_REFRESH_TOKEN_TTL_S",
            "AUTH_CLOCK_SKEW_S",
            "AUTH_DB_POOL_SIZE",
        ),
        datastore="postgres (auth-db)",
    ),
    ServiceCatalog(
        name="user-profile",
        description=(
            "User profile, contact details, saved addresses and notification preferences. "
            "Owns PII masking for downstream consumers."
        ),
        tier=2,
        owning_team="profile",
        depends_on=("auth",),
        error_prefix="PROF",
        functions=(
            "get_profile",
            "update_profile",
            "list_addresses",
            "add_address",
            "resolve_preferences",
            "mask_pii",
        ),
        config_keys=(
            "PROFILE_PII_MASK_MODE",
            "PROFILE_CACHE_TTL_S",
            "PROFILE_MAX_ADDRESSES",
            "PROFILE_AUTH_TIMEOUT_MS",
            "PROFILE_READ_REPLICA_ENABLED",
        ),
        datastore="postgres (profile-db) + redis cache",
    ),
    ServiceCatalog(
        name="payments",
        description=(
            "Card authorization, capture, refunds and nightly settlement reconciliation "
            "against the payment service provider."
        ),
        tier=1,
        owning_team="payments-core",
        depends_on=("auth",),
        error_prefix="PAY",
        functions=(
            "authorize_payment",
            "capture_funds",
            "refund_payment",
            "reconcile_settlement",
            "validate_idempotency_key",
            "tokenize_card",
        ),
        config_keys=(
            "PAYMENTS_AUTH_TIMEOUT_MS",
            "PAYMENTS_IDEMPOTENCY_TTL_S",
            "PAYMENTS_CAPTURE_RETRY_MAX",
            "PAYMENTS_SETTLEMENT_BATCH_SIZE",
            "PAYMENTS_PSP_POOL_SIZE",
        ),
        datastore="postgres (payments-db)",
    ),
    ServiceCatalog(
        name="inventory",
        description=(
            "Stock levels, reservations during checkout, and warehouse synchronisation "
            "over the fulfilment event bus."
        ),
        tier=2,
        owning_team="fulfilment",
        depends_on=(),
        error_prefix="INV",
        functions=(
            "reserve_stock",
            "release_reservation",
            "adjust_level",
            "sync_warehouse",
            "check_availability",
            "expire_reservations",
        ),
        config_keys=(
            "INVENTORY_RESERVATION_TTL_S",
            "INVENTORY_KAFKA_MAX_POLL_RECORDS",
            "INVENTORY_KAFKA_SESSION_TIMEOUT_MS",
            "INVENTORY_WAREHOUSE_SYNC_INTERVAL_S",
            "INVENTORY_OVERSELL_GUARD_ENABLED",
        ),
        datastore="postgres (inventory-db) + kafka",
    ),
    ServiceCatalog(
        name="orders",
        description=(
            "Order lifecycle and fulfilment orchestration: state machine, order history, "
            "and the events that drive downstream fulfilment."
        ),
        tier=1,
        owning_team="commerce",
        depends_on=("payments", "inventory", "notification"),
        error_prefix="ORD",
        functions=(
            "create_order",
            "transition_state",
            "cancel_order",
            "orchestrate_fulfilment",
            "compute_totals",
            "emit_order_event",
        ),
        config_keys=(
            "ORDERS_EVENT_ORDERING_MODE",
            "ORDERS_STATE_LOCK_TIMEOUT_MS",
            "ORDERS_FULFILMENT_RETRY_MAX",
            "ORDERS_CANCEL_GRACE_S",
            "ORDERS_KAFKA_ACKS",
        ),
        datastore="postgres (orders-db) + kafka",
    ),
    ServiceCatalog(
        name="checkout",
        description=(
            "Cart and checkout session, price and promotion calculation, quote refresh, "
            "and final order placement."
        ),
        tier=1,
        owning_team="commerce",
        depends_on=("payments", "inventory", "orders", "user-profile"),
        error_prefix="CHK",
        functions=(
            "create_session",
            "calculate_pricing",
            "apply_promotion",
            "place_order",
            "validate_cart",
            "refresh_quote",
        ),
        config_keys=(
            "CHECKOUT_SESSION_TTL_S",
            "CHECKOUT_QUOTE_TTL_S",
            "CHECKOUT_REDIS_TIMEOUT_MS",
            "CHECKOUT_MAX_PROMOTIONS",
            "CHECKOUT_PRICING_PRECISION",
        ),
        datastore="redis (session) + postgres (checkout-db)",
    ),
    ServiceCatalog(
        name="notification",
        description=(
            "Transactional email, SMS and push delivery: template rendering, provider "
            "fan-out, and retry of failed deliveries."
        ),
        tier=3,
        owning_team="growth",
        depends_on=("user-profile",),
        error_prefix="NOTIF",
        functions=(
            "send_email",
            "send_sms",
            "send_push",
            "render_template",
            "enqueue_delivery",
            "retry_failed",
        ),
        config_keys=(
            "NOTIF_TEMPLATE_ENGINE",
            "NOTIF_DELIVERY_TIMEOUT_MS",
            "NOTIF_RETRY_MAX",
            "NOTIF_RATE_LIMIT_PER_MIN",
            "NOTIF_PROVIDER_FALLBACK_ENABLED",
        ),
        datastore="postgres (notification-db) + sqs",
    ),
)

SERVICE_BY_NAME: dict[str, ServiceCatalog] = {s.name: s for s in SERVICES}
SERVICE_NAMES: tuple[str, ...] = tuple(s.name for s in SERVICES)


# Third-party / internal libraries with the versions pinned during the world
# window. Dependency-bump chains move one of these.
DEPENDENCIES: dict[str, str] = {
    "meridian-common": "4.12.0",
    "httpx": "0.27.0",
    "psycopg": "3.2.1",
    "redis": "5.0.4",
    "kafka-python": "2.0.2",
    "pydantic": "2.9.1",
    "boto3": "1.34.128",
    "sqlalchemy": "2.0.34",
    "tenacity": "8.5.0",
    "orjson": "3.10.7",
}


def bumped_version(current: str) -> str:
    major, minor, _patch = (int(p) for p in current.split("."))
    return f"{major}.{minor + 1}.0"


CROSS_CUTTING_ARCH_DOCS: tuple[tuple[str, str], ...] = (
    ("login-and-token-flow", "Login and token flow across auth, user-profile and the edge"),
    ("checkout-sequence", "End-to-end checkout sequence: cart to placed order"),
    ("fulfilment-event-bus", "The fulfilment event bus: topics, ordering and consumers"),
    ("data-stores-and-ownership", "Data stores, ownership boundaries and PII handling"),
    ("payment-settlement-pipeline", "Payment authorization, capture and nightly settlement"),
)
