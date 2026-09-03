"""CLI entry point.

python -m data.generator                    # build into data/generated/
python -m data.generator --seed 7 --out /tmp/kb
python -m data.generator --check            # verify data/generated/ is current
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.generator.emit import check, generate, write
from data.generator.spec import DEFAULT_SEED, GeneratorSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "generated"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="data.generator", description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and diff against --out; exit 1 on any difference",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and verify but write nothing; print the manifest",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = GeneratorSpec(seed=args.seed)

    if args.check:
        problems = check(spec, args.out)
        if problems:
            print(f"stale corpus in {args.out} (seed {args.seed}):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print("\nregenerate with:  python -m data.generator", file=sys.stderr)
            return 1
        print(f"ok: {args.out} matches seed {args.seed}")
        return 0

    if args.dry_run:
        files = generate(spec)
        print(files["manifest.json"])
        return 0

    files = write(spec, args.out)
    manifest = json.loads(files["manifest.json"])
    print(f"wrote {len(files)} files to {args.out}  (seed {args.seed})")
    print(f"corpus_sha256 {manifest['corpus_sha256']}")
    width = max(len(k) for k in manifest["counts"])
    for key, value in manifest["counts"].items():
        print(f"  {key.ljust(width)}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
