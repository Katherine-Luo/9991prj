"""Katana P0 storage preflight utilities."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lidc_baseline.audit import write_json


def directory_size(path: Path) -> int:
    """Return the total size of regular files without following symlinks."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )


def run_preflight(
    scratch: Path,
    quota_bytes: int,
    min_remaining_bytes: int,
    max_workset_bytes: int,
    worksets: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    """Evaluate the pre-registered Katana storage gates."""
    scratch_used = directory_size(scratch)
    remaining = max(0, quota_bytes - scratch_used)
    workset_bytes = sum(directory_size(path) for path in worksets)
    remaining_pass = remaining >= min_remaining_bytes
    workset_pass = workset_bytes <= max_workset_bytes
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if remaining_pass and workset_pass else "BLOCKED",
        "scratch_path": str(scratch),
        "scratch_quota_bytes": quota_bytes,
        "scratch_used_bytes": scratch_used,
        "scratch_remaining_bytes": remaining,
        "minimum_remaining_bytes": min_remaining_bytes,
        "remaining_space_pass": remaining_pass,
        "p0_workset_paths": [str(path) for path in worksets],
        "p0_workset_bytes": workset_bytes,
        "maximum_p0_workset_bytes": max_workset_bytes,
        "p0_workset_pass": workset_pass,
    }
    write_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Check storage gates")
    preflight.add_argument("--scratch", type=Path, required=True)
    preflight.add_argument("--quota-bytes", type=int, required=True)
    preflight.add_argument("--min-remaining-bytes", type=int, required=True)
    preflight.add_argument("--max-workset-bytes", type=int, required=True)
    preflight.add_argument("--workset", type=Path, action="append", default=[])
    preflight.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Katana utility commands."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        report = run_preflight(
            scratch=arguments.scratch,
            quota_bytes=arguments.quota_bytes,
            min_remaining_bytes=arguments.min_remaining_bytes,
            max_workset_bytes=arguments.max_workset_bytes,
            worksets=arguments.workset,
            output=arguments.output,
        )
        return 0 if report["status"] == "PASS" else 1
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
