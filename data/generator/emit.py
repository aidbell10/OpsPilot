"""Serialise a :class:`World` to JSON and verify its invariants.

``generate`` is pure: seed in, ``{filename: text}`` out. ``write`` is the only
function that touches the filesystem. ``check`` regenerates in memory and
compares against what is already on disk (used by CI and the ``--check`` flag).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from data.generator import GENERATOR_VERSION
from data.generator.schema import DIFFICULTIES, DOCUMENT_TYPES
from data.generator.spec import GeneratorSpec
from data.generator.world import World, build_world

DATA_FILES = (
    "services.json",
    "deployments.json",
    "documents.json",
    "historical_incidents.json",
    "ground_truth.json",
    "chains.json",
)
MANIFEST_FILE = "manifest.json"


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _spec_dict(spec: GeneratorSpec) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(spec):
        value = getattr(spec, f.name)
        out[f.name] = value.isoformat() if hasattr(value, "isoformat") else value
    return out


def generate(spec: GeneratorSpec) -> dict[str, str]:
    """Return ``{filename: file_text}`` for the whole corpus, deterministically."""
    world = build_world(spec)
    payload: dict[str, list[dict[str, Any]]] = {
        "services.json": [r.to_dict() for r in world.services],
        "deployments.json": [r.to_dict() for r in world.deployments],
        "documents.json": [r.to_dict() for r in world.documents],
        "historical_incidents.json": [r.to_dict() for r in world.historical_incidents],
        "ground_truth.json": [r.to_dict() for r in world.ground_truth],
        "chains.json": [r.to_dict() for r in world.chains],
    }
    verify(world)

    files: dict[str, str] = {name: _dump(rows) for name, rows in payload.items()}
    file_hashes = {name: _sha256(text) for name, text in files.items()}
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": spec.seed,
        "spec": _spec_dict(spec),
        "counts": world.counts(),
        "files": file_hashes,
        "corpus_sha256": _sha256("".join(file_hashes[n] for n in DATA_FILES)),
    }
    files[MANIFEST_FILE] = _dump(manifest)
    return files


def write(spec: GeneratorSpec, out_dir: Path) -> dict[str, str]:
    files = generate(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out_dir / name).write_text(text, encoding="utf-8", newline="\n")
    return files


def check(spec: GeneratorSpec, out_dir: Path) -> list[str]:
    """Return a list of human-readable differences (empty == corpus is current)."""
    fresh = generate(spec)
    problems: list[str] = []
    for name, text in fresh.items():
        path = out_dir / name
        if not path.exists():
            problems.append(f"{name}: missing on disk")
            continue
        on_disk = path.read_text(encoding="utf-8")
        if on_disk != text:
            problems.append(f"{name}: on-disk content differs from regenerated content")
    extra = {p.name for p in out_dir.glob("*.json")} - set(fresh)
    problems.extend(f"{name}: unexpected file on disk" for name in sorted(extra))
    return problems


# --------------------------------------------------------------------------
# invariants
# --------------------------------------------------------------------------
class VerificationError(AssertionError):
    pass


def verify(world: World) -> None:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    doc_paths = {d.source_path for d in world.documents}
    doc_ids = [d.doc_id for d in world.documents]
    require(len(doc_ids) == len(set(doc_ids)), "duplicate document doc_id")
    require(len(doc_paths) == len(world.documents), "duplicate document source_path")

    for d in world.documents:
        require(d.document_type in DOCUMENT_TYPES, f"bad document_type {d.document_type!r}")
        require(bool(d.content.strip()), f"empty document {d.source_path}")

    dep_keys = [(d.service_name, d.version) for d in world.deployments]
    require(len(dep_keys) == len(set(dep_keys)), "duplicate (service, version) deployment")

    # every planted chain must be fully wired
    chain_ids = {c.chain_id for c in world.chains}
    for chain in world.chains:
        for role, path in chain.document_source_paths.items():
            require(path in doc_paths, f"{chain.chain_id}: missing {role} doc {path}")
        trigger_doc = f"deployments/{chain.service_name}/{chain.trigger_deployment_version}.md"
        require(
            trigger_doc in doc_paths,
            f"{chain.chain_id}: no deployment doc for trigger release {trigger_doc}",
        )
        require(
            any(chain.error_code in line for line in chain.symptom_log_lines),
            f"{chain.chain_id}: error code absent from log lines",
        )
        deploy_versions = {
            d.version for d in world.deployments if d.service_name == chain.service_name
        }
        for v in (
            chain.trigger_deployment_version,
            chain.fix_deployment_version,
            chain.narrative["prev_version"],
        ):
            require(v in deploy_versions, f"{chain.chain_id}: version {v} not in deployments")

    incident_ids = [i.incident_id for i in world.historical_incidents]
    require(len(incident_ids) == len(set(incident_ids)), "duplicate historical incident id")
    for inc in world.historical_incidents:
        if inc.linked_document_source_path is not None:
            require(
                inc.linked_document_source_path in doc_paths,
                f"{inc.incident_id}: linked doc missing",
            )

    case_ids = [c.case_id for c in world.ground_truth]
    require(len(case_ids) == len(set(case_ids)), "duplicate ground-truth case_id")
    for case in world.ground_truth:
        require(case.difficulty in DIFFICULTIES, f"{case.case_id}: bad difficulty")
        if case.answerable:
            require(bool(case.required_document_source_paths), f"{case.case_id}: no required docs")
            for p in case.required_document_source_paths:
                require(p in doc_paths, f"{case.case_id}: required doc {p} does not exist")
            for snippet in case.required_evidence_snippets:
                hit = any(
                    snippet in d.content
                    for d in world.documents
                    if d.source_path in case.required_document_source_paths
                )
                require(hit, f"{case.case_id}: snippet {snippet!r} not found in any required doc")
            if case.chain_id is not None:
                require(
                    case.chain_id in chain_ids, f"{case.case_id}: unknown chain {case.chain_id}"
                )
        else:
            require(
                not case.required_document_source_paths,
                f"{case.case_id}: unanswerable case must not require docs",
            )

    answerable = sum(1 for c in world.ground_truth if c.answerable)
    unanswerable = sum(1 for c in world.ground_truth if not c.answerable)
    require(answerable >= 12, f"expected >=12 answerable cases, got {answerable}")
    require(unanswerable >= 3, f"expected >=3 unanswerable cases, got {unanswerable}")
    require(
        any(c.difficulty == "adversarial" for c in world.ground_truth),
        "no adversarial ground-truth case",
    )

    if errors:
        raise VerificationError("corpus verification failed:\n  - " + "\n  - ".join(errors))
