"""Assemble the whole synthetic corpus from a :class:`GeneratorSpec`."""

from __future__ import annotations

from dataclasses import dataclass, field

from data.generator import GENERATOR_VERSION
from data.generator.company import CROSS_CUTTING_ARCH_DOCS, SERVICES
from data.generator.documents import (
    _GENERIC_RUNBOOKS,
    background_incidents,
    chain_architecture,
    chain_incident_record,
    chain_known_error,
    chain_postmortem,
    chain_runbook,
    cross_cutting_architecture,
    deployment_doc,
    deployment_records,
    generic_runbook,
    ground_truth_for_chain,
    service_architecture,
    service_records,
    unanswerable_cases,
)
from data.generator.rng import Rng
from data.generator.schema import (
    Chain,
    DeploymentRecord,
    DocumentRecord,
    GroundTruthCase,
    HistoricalIncidentRecord,
    ServiceRecord,
)
from data.generator.spec import GeneratorSpec
from data.generator.timeline import Release, build_timeline


@dataclass(slots=True)
class World:
    spec: GeneratorSpec
    services: list[ServiceRecord] = field(default_factory=list)
    deployments: list[DeploymentRecord] = field(default_factory=list)
    documents: list[DocumentRecord] = field(default_factory=list)
    historical_incidents: list[HistoricalIncidentRecord] = field(default_factory=list)
    ground_truth: list[GroundTruthCase] = field(default_factory=list)
    chains: list[Chain] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        by_type: dict[str, int] = {}
        for d in self.documents:
            by_type[d.document_type] = by_type.get(d.document_type, 0) + 1
        return {
            "services": len(self.services),
            "deployments": len(self.deployments),
            "documents": len(self.documents),
            **{f"documents.{k}": v for k, v in sorted(by_type.items())},
            "historical_incidents": len(self.historical_incidents),
            "ground_truth_cases": len(self.ground_truth),
            "ground_truth.answerable": sum(1 for c in self.ground_truth if c.answerable),
            "ground_truth.unanswerable": sum(1 for c in self.ground_truth if not c.answerable),
            "chains": len(self.chains),
        }


def build_world(spec: GeneratorSpec) -> World:
    root = Rng(spec.seed, "opspilot", GENERATOR_VERSION)
    world = World(spec=spec)

    world.services = service_records()

    releases, bound_chains = build_timeline(spec, root)
    world.deployments = deployment_records(releases)

    # --- chain artefacts ------------------------------------------------
    for chain in bound_chains:
        d_rng = root.child("docs", chain.blueprint.chain_id)
        world.historical_incidents.append(chain_incident_record(chain))
        world.documents.append(chain_postmortem(chain, d_rng.child("pm")))
        world.documents.append(chain_runbook(chain, d_rng.child("rb")))
        world.documents.append(chain_known_error(chain))
        arch = chain_architecture(chain)
        if arch is not None:
            world.documents.append(arch)
        cases, chain_record = ground_truth_for_chain(chain)
        world.ground_truth.extend(cases)
        world.chains.append(chain_record)

    # --- background incidents (noise) ---------------------------------
    bg_count = root.child("bg-count").int(*spec.background_incidents)
    world.historical_incidents.extend(
        background_incidents(releases, bg_count, root.child("bg-incidents"))
    )

    # --- architecture docs -------------------------------------------
    for service in SERVICES:
        world.documents.append(service_architecture(service, root.child("arch", service.name)))
    for slug, title in CROSS_CUTTING_ARCH_DOCS:
        world.documents.append(cross_cutting_architecture(slug, title, root.child("arch-x", slug)))

    # --- generic runbooks ------------------------------------------
    for service in SERVICES:
        s_rng = root.child("generic-rb", service.name)
        k = s_rng.int(*spec.generic_runbooks_per_service)
        for slug, title in s_rng.sample(_GENERIC_RUNBOOKS, k=k):
            world.documents.append(generic_runbook(service, slug, title, s_rng.child(slug)))

    # --- deployment docs for the most notable releases ------------
    notable: list[Release] = []
    for svc_releases in releases.values():
        notable.extend(r for r in svc_releases if r.role in {"trigger", "fix"})
    remaining = spec.deployment_docs - len(notable)
    if remaining > 0:
        routine = [r for rs in releases.values() for r in rs if r.role in {"routine", "pre"}]
        notable.extend(root.child("notable").sample(routine, k=min(remaining, len(routine))))
    seen: set[tuple[str, str]] = set()
    for r in sorted(notable, key=lambda x: (x.service_name, x.version)):
        key = (r.service_name, r.version)
        if key in seen:
            continue
        seen.add(key)
        world.documents.append(deployment_doc(r, root.child("depdoc", *key)))

    world.ground_truth.extend(unanswerable_cases())

    _stable_sort(world)
    return world


def _stable_sort(world: World) -> None:
    world.services.sort(key=lambda s: s.name)
    world.deployments.sort(key=lambda d: (d.service_name, d.deployed_at, d.version))
    world.documents.sort(key=lambda d: d.source_path)
    world.historical_incidents.sort(key=lambda i: (i.occurred_at, i.incident_id))
    world.ground_truth.sort(key=lambda c: c.case_id)
    world.chains.sort(key=lambda c: c.chain_id)
