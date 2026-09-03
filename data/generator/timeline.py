"""Build each service's release history and bind the ground-truth chains to it.

Strategy: generate a spread of *routine* releases, then for every chain insert
three dedicated releases (pre-change / trigger / fix) around a chosen date,
merge, sort, and assign monotonic semver. Routine releases are kept at least
14 days away from any chain trigger so the ``prev`` / ``fix`` versions the
narrative refers to are unambiguous.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from data.generator.blueprints import BLUEPRINTS, ChainBlueprint
from data.generator.company import (
    DEPENDENCIES,
    SERVICE_BY_NAME,
    ServiceCatalog,
    bumped_version,
)
from data.generator.rng import Rng
from data.generator.spec import GeneratorSpec

ROUTINE_SUMMARIES: tuple[str, ...] = (
    "Dependency updates and observability improvements in {func}.",
    "Performance tuning for {func}; no behaviour change.",
    "Add structured logging and metrics around {func}.",
    "Refactor {func} for readability; covered by existing tests.",
    "Bug fixes in {func} and minor error-message improvements.",
    "Raise timeouts and add retries in {func}.",
    "Documentation and OpenAPI schema updates; {func} unchanged.",
)

KIND_INTENT: dict[str, str] = {
    "upstream-timeout": "moves an auth check inline for stronger verification",
    "stale-cache": "reduces load on the JWKS endpoint",
    "off-by-one": "batches reservations for fewer round-trips",
    "pricing-regression": "lets customers stack eligible promotions",
    "template-regression": "adopts the strict template engine",
    "race-condition": "raises event throughput with per-partition ordering",
    "idempotency": "reduces Redis memory usage",
    "pii-exposure": "makes masked values readable for support tooling",
    "dependency-bump": "routine dependency upgrade",
    "pool-exhaustion": "cost-optimisation pass on connection pools",
    "batch-oom": "makes settlement reconciliation a single pass",
}


@dataclass(frozen=True, slots=True)
class Release:
    service_name: str
    version: str
    deployed_at: dt.datetime
    environment: str
    status: str
    change_summary: str
    changed_components: dict[str, object]
    role: str = "routine"  # routine | pre | trigger | fix
    chain_id: str | None = None


@dataclass(frozen=True, slots=True)
class BoundChain:
    blueprint: ChainBlueprint
    service: ServiceCatalog
    prev_version: str
    trigger_version: str
    fix_version: str
    trigger_at: dt.datetime
    incident_at: dt.datetime
    fix_at: dt.datetime
    error_code: str
    dep_from: str | None
    dep_to: str | None
    log_lines: list[str] = field(default_factory=list)

    @property
    def incident_id(self) -> str:
        return self.blueprint.chain_id.replace("chain-", "inc-")

    @property
    def subs(self) -> dict[str, str]:
        bp = self.blueprint
        return {
            "service": self.service.name,
            "version": self.trigger_version,
            "prev_version": self.prev_version,
            "fix_version": self.fix_version,
            "error_code": self.error_code,
            "func": bp.changed_function,
            "config_key": bp.config_key or "",
            "dependency": bp.dependency or "",
            "dep_from": self.dep_from or "",
            "dep_to": self.dep_to or "",
            "date": _human(self.trigger_at),
        }

    def fill(self, template: str) -> str:
        return template.format(**self.subs)


def _human(ts: dt.datetime) -> str:
    return f"{ts.strftime('%B')} {ts.day}, {ts.year}"


def _dt_in(rng: Rng, start: dt.date, end: dt.date) -> dt.datetime:
    span = (end - start).days
    day = rng.int(0, max(0, span))
    base = dt.datetime.combine(start, dt.time(), tzinfo=dt.UTC)
    return base + dt.timedelta(days=day, hours=rng.int(6, 21), minutes=rng.int(0, 59))


def _bump(version: tuple[int, int, int], kind: str) -> tuple[int, int, int]:
    major, minor, patch = version
    if kind == "major":
        return major + 1, 0, 0
    if kind == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def _routine_components(rng: Rng, service: ServiceCatalog) -> dict[str, object]:
    funcs = rng.sample(service.functions, k=rng.int(1, 2))
    components: dict[str, object] = {"functions": funcs}
    if rng.chance(0.4):
        components["config"] = [rng.choice(service.config_keys)]
    return components


def build_timeline(
    spec: GeneratorSpec, rng: Rng
) -> tuple[dict[str, list[Release]], list[BoundChain]]:
    chains_by_service: dict[str, list[ChainBlueprint]] = {}
    for bp in BLUEPRINTS:
        chains_by_service.setdefault(bp.service, []).append(bp)

    releases: dict[str, list[Release]] = {}
    bound: list[BoundChain] = []

    win_start, win_end = spec.world_start, spec.world_end
    total_days = (win_end - win_start).days

    for service in SERVICE_BY_NAME.values():
        svc_rng = rng.child("timeline", service.name)
        service_chains = chains_by_service.get(service.name, [])

        # --- chain releases -------------------------------------------------
        chain_slots: list[
            tuple[ChainBlueprint, dt.datetime, dt.datetime, dt.datetime, dt.datetime]
        ] = []
        n = max(1, len(service_chains))
        for j, bp in enumerate(service_chains):
            c_rng = svc_rng.child("chain", bp.chain_id)
            lo = win_start + dt.timedelta(days=20 + (total_days - 40) * j // n)
            hi = win_start + dt.timedelta(days=20 + (total_days - 40) * (j + 1) // n)
            trigger_at = _dt_in(c_rng, lo, max(lo + dt.timedelta(days=1), hi))
            prev_at = trigger_at - dt.timedelta(days=c_rng.int(4, 12), hours=c_rng.int(0, 12))
            incident_at = trigger_at + dt.timedelta(hours=c_rng.int(5, 40))
            fix_at = incident_at + dt.timedelta(hours=c_rng.int(3, 12))
            chain_slots.append((bp, prev_at, trigger_at, incident_at, fix_at))

        raw: list[tuple[dt.datetime, str, ChainBlueprint | None]] = []
        for bp, prev_at, trigger_at, _incident_at, fix_at in chain_slots:
            raw.append((prev_at, "pre", bp))
            raw.append((trigger_at, "trigger", bp))
            raw.append((fix_at, "fix", bp))

        # --- routine releases (kept away from chain triggers) --------------
        trigger_dates = [t for _, _, t, _, _ in chain_slots]
        want_routine = svc_rng.int(*spec.releases_per_service)
        attempts = 0
        routine_dates: list[dt.datetime] = []
        while len(routine_dates) < want_routine and attempts < want_routine * 20:
            attempts += 1
            cand = _dt_in(svc_rng.child("routine", attempts), win_start, win_end)
            if all(abs((cand - td).days) > 14 for td in trigger_dates):
                routine_dates.append(cand)
        for d in routine_dates:
            raw.append((d, "routine", None))

        raw.sort(key=lambda item: item[0])

        # --- assign monotonic semver -------------------------------------
        version = (svc_rng.int(2, 4), svc_rng.int(0, 4), svc_rng.int(0, 3))
        svc_releases: list[Release] = []
        version_by_index: list[str] = []
        for i in range(len(raw)):
            if i > 0:
                kind = svc_rng.child("bump", i).weighted(
                    [("patch", 0.70), ("minor", 0.27), ("major", 0.03)]
                )
                version = _bump(version, kind)
            version_by_index.append(f"v{version[0]}.{version[1]}.{version[2]}")

        for i, (when, role, slot_bp) in enumerate(raw):
            vstr = version_by_index[i]
            env = "staging" if svc_rng.child("env", i).chance(0.12) else "production"
            status = "succeeded"
            if role == "routine" and svc_rng.child("status", i).chance(0.06):
                status = svc_rng.child("status2", i).choice(("rolled_back", "failed"))

            if role in {"routine", "pre"}:
                summary = (
                    svc_rng.child("sum", i)
                    .choice(ROUTINE_SUMMARIES)
                    .format(func=svc_rng.child("sumf", i).choice(service.functions))
                )
                components = _routine_components(svc_rng.child("comp", i), service)
            elif role == "trigger":
                assert slot_bp is not None
                summary = _trigger_summary(slot_bp, service)
                components = _chain_components(slot_bp, service, kind="trigger")
            else:  # fix
                assert slot_bp is not None
                summary = (
                    f"Remediation for incident {slot_bp.chain_id.replace('chain-', 'inc-')} "
                    f"({service.error_prefix}-{slot_bp.error_number}); see postmortem."
                )
                components = _chain_components(slot_bp, service, kind="fix")

            svc_releases.append(
                Release(
                    service_name=service.name,
                    version=vstr,
                    deployed_at=when,
                    environment=env,
                    status=status,
                    change_summary=summary,
                    changed_components=components,
                    role=role,
                    chain_id=slot_bp.chain_id if slot_bp is not None else None,
                )
            )

        releases[service.name] = svc_releases

        # --- read back prev/trigger/fix versions and bind ---------------
        by_role_chain = {
            (r.chain_id, r.role): r for r in svc_releases if r.role in {"pre", "trigger", "fix"}
        }
        for bp, _prev_at, trigger_at, incident_at, fix_at in chain_slots:
            prev_r = by_role_chain[(bp.chain_id, "pre")]
            trig_r = by_role_chain[(bp.chain_id, "trigger")]
            fix_r = by_role_chain[(bp.chain_id, "fix")]
            dep_from = DEPENDENCIES.get(bp.dependency) if bp.dependency else None
            dep_to = bumped_version(dep_from) if dep_from else None
            error_code = f"{service.error_prefix}-{bp.error_number}"
            chain = BoundChain(
                blueprint=bp,
                service=service,
                prev_version=prev_r.version,
                trigger_version=trig_r.version,
                fix_version=fix_r.version,
                trigger_at=trigger_at,
                incident_at=incident_at,
                fix_at=fix_at,
                error_code=error_code,
                dep_from=dep_from,
                dep_to=dep_to,
            )
            chain.log_lines.extend(_render_logs(bp, chain, svc_rng.child("logs", bp.chain_id)))
            bound.append(chain)

    bound.sort(key=lambda c: c.incident_at)
    return releases, bound


def _trigger_summary(bp: ChainBlueprint, service: ServiceCatalog) -> str:
    parts = [f"{bp.changed_function} {KIND_INTENT.get(bp.kind, 'behaviour change')}."]
    if bp.config_key:
        parts.append(f"Config touched: {bp.config_key}.")
    if bp.dependency:
        frm = DEPENDENCIES[bp.dependency]
        parts.append(f"Bumped {bp.dependency} {frm} -> {bumped_version(frm)}.")
    return " ".join(parts)


def _chain_components(
    bp: ChainBlueprint, service: ServiceCatalog, *, kind: str
) -> dict[str, object]:
    components: dict[str, object] = {"functions": [bp.changed_function]}
    if bp.config_key:
        components["config"] = [bp.config_key]
    if bp.dependency:
        frm = DEPENDENCIES[bp.dependency]
        if kind == "trigger":
            components["dependencies"] = [f"{bp.dependency} {frm}->{bumped_version(frm)}"]
        else:
            components["dependencies"] = [f"{bp.dependency} pinned {frm}"]
    if kind == "fix":
        components["note"] = f"remediation for {bp.chain_id}"
    return components


def _render_logs(bp: ChainBlueprint, chain: BoundChain, rng: Rng) -> list[str]:
    out: list[str] = []
    for i, template in enumerate(bp.log_templates):
        ts = chain.incident_at + dt.timedelta(minutes=rng.int(1, 220) + i * 7)
        line = template.format(
            ts=ts.replace(microsecond=0).isoformat(),
            error_code=chain.error_code,
            n=rng.int(100000, 999999),
            age=rng.int(1200, 3400),
            ttl=chain.blueprint.error_number % 900 + 100,
            amt=f"{rng.int(4, 60)}.{rng.int(10, 99)}",
        )
        out.append(line)
    return out
