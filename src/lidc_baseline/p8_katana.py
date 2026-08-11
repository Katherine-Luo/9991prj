"""Prepare and verify the exact P8 Katana code delta."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p4_prepare import canonical_json_bytes, sha256_bytes, sha256_file
from lidc_baseline.p7_katana import (
    P7_TRANSFER_MANIFEST,
    verify_transfer_manifest as verify_p7_transfer_manifest,
)
from lidc_baseline.p8_gam import validate_p8_execution_config


SCHEMA_VERSION = 1
P8_TRANSFER_MANIFEST = Path(
    "artifacts/baseline_v2/manifests/p8_stage_a_transfer_manifest.json"
)
P8_DELTA_FILES = (
    "configs/experiments/baseline_v2_p8_gam_h200.yaml",
    "configs/experiments/baseline_v2_p8_gam_h200.resolved.yaml",
    "configs/experiments/baseline_v2_p8_gam_h200.sha256",
    "scripts/katana/p8_fold.pbs",
    "scripts/katana/p8_oof.pbs",
    "scripts/katana/p8_stage_a.pbs",
    "src/lidc_baseline/p8_audit.py",
    "src/lidc_baseline/p8_gam.py",
    "src/lidc_baseline/p8_gam_lifecycle.py",
    "src/lidc_baseline/p8_katana.py",
)


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("transfer_manifest_sha256", None)
    return sha256_bytes(canonical_json_bytes(unhashed))


def _safe_relative_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"P8_TRANSFER_UNSAFE_PATH:{relative}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"P8_TRANSFER_OUTSIDE_REPOSITORY:{relative}") from error
    if not path.is_file():
        raise FileNotFoundError(f"P8_TRANSFER_FILE_MISSING:{relative}")
    return path


def build_transfer_manifest(
    repository_root: Path,
    scientific_config_path: Path,
    p8_execution_config_path: Path,
    output_path: Path = P8_TRANSFER_MANIFEST,
) -> dict[str, Any]:
    """Build the private exact P8 delta on the verified P7/P6/P5/P4 base."""
    root = repository_root.resolve()
    scientific = load_config(root / scientific_config_path)
    _p8_config, p8_hash = validate_p8_execution_config(
        root / p8_execution_config_path
    )
    p7_manifest_path = root / P7_TRANSFER_MANIFEST
    p7 = verify_p7_transfer_manifest(root, P7_TRANSFER_MANIFEST)
    entries = []
    for relative in sorted(P8_DELTA_FILES):
        path = _safe_relative_file(root, relative)
        entries.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": scientific["protocol"]["version"],
        "scientific_config_sha256": compute_config_sha256(scientific),
        "p8_execution_config_sha256": p8_hash,
        "p7_base_transfer_manifest_sha256": p7["transfer_manifest_sha256"],
        "p7_base_transfer_manifest_file_sha256": sha256_file(p7_manifest_path),
        "total_files": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "files": entries,
    }
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    write_json(root / output_path, payload)
    return payload


def read_transfer_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("transfer_manifest_sha256") != _manifest_digest(payload):
        raise ValueError("P8_TRANSFER_MANIFEST_HASH_MISMATCH")
    return payload


def verify_transfer_manifest(
    repository_root: Path, manifest_path: Path
) -> dict[str, Any]:
    root = repository_root.resolve()
    payload = read_transfer_manifest(root / manifest_path)
    seen: set[str] = set()
    observed_bytes = 0
    for entry in payload["files"]:
        relative = str(entry["relative_path"])
        if relative in seen:
            raise ValueError("P8_TRANSFER_DUPLICATE_PATH")
        seen.add(relative)
        path = _safe_relative_file(root, relative)
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"P8_TRANSFER_FILE_SIZE_MISMATCH:{relative}")
        if sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"P8_TRANSFER_FILE_HASH_MISMATCH:{relative}")
        observed_bytes += path.stat().st_size
    expected = set(P8_DELTA_FILES)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(
            "P8_TRANSFER_FILE_SET_MISMATCH:"
            f"missing={','.join(missing)}:extra={','.join(extra)}"
        )
    if (
        len(seen) != int(payload["total_files"])
        or observed_bytes != int(payload["total_bytes"])
    ):
        raise ValueError("P8_TRANSFER_TOTAL_MISMATCH")
    return {
        "status": "PASS",
        "verified_files": len(seen),
        "verified_bytes": observed_bytes,
        "transfer_manifest_sha256": payload["transfer_manifest_sha256"],
        "transfer_manifest_file_sha256": sha256_file(root / manifest_path),
    }


def transfer_file_list(repository_root: Path, manifest_path: Path) -> list[str]:
    root = repository_root.resolve()
    payload = read_transfer_manifest(root / manifest_path)
    paths = [str(entry["relative_path"]) for entry in payload["files"]]
    paths.append(manifest_path.as_posix())
    if len(paths) != len(set(paths)):
        raise ValueError("P8_TRANSFER_LIST_DUPLICATE")
    for relative in paths:
        _safe_relative_file(root, relative)
    return sorted(paths)


def verify_stage_a(
    repository_root: Path,
    scientific_config_path: Path,
    p8_execution_config_path: Path,
    p8_manifest_path: Path,
) -> dict[str, Any]:
    """Verify the immutable P7 base and exact P8 delta locally or remotely."""
    root = repository_root.resolve()
    scientific = load_config(root / scientific_config_path)
    _p8_config, p8_hash = validate_p8_execution_config(
        root / p8_execution_config_path
    )
    p7 = verify_p7_transfer_manifest(root, P7_TRANSFER_MANIFEST)
    p8 = verify_transfer_manifest(root, p8_manifest_path)
    payload = read_transfer_manifest(root / p8_manifest_path)
    if payload["p7_base_transfer_manifest_sha256"] != p7[
        "transfer_manifest_sha256"
    ]:
        raise ValueError("P8_REMOTE_P7_BASE_MANIFEST_MISMATCH")
    if payload["p7_base_transfer_manifest_file_sha256"] != p7[
        "transfer_manifest_file_sha256"
    ]:
        raise ValueError("P8_REMOTE_P7_BASE_FILE_MISMATCH")
    if payload["scientific_config_sha256"] != compute_config_sha256(scientific):
        raise ValueError("P8_REMOTE_SCIENTIFIC_CONFIG_MISMATCH")
    if payload["p8_execution_config_sha256"] != p8_hash:
        raise ValueError("P8_REMOTE_EXECUTION_CONFIG_MISMATCH")
    return {
        "status": "PASS",
        "p7_base": p7,
        "p8_delta": p8,
        "scientific_config_sha256": compute_config_sha256(scientific),
        "p8_execution_config_sha256": p8_hash,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repository-root", type=Path, default=Path("."))
    common.add_argument(
        "--config", type=Path, default=Path("configs/baseline_v2.yaml")
    )
    common.add_argument(
        "--p8-execution-config",
        type=Path,
        default=Path("configs/experiments/baseline_v2_p8_gam_h200.yaml"),
    )
    build = subparsers.add_parser("build-transfer-manifest", parents=[common])
    build.add_argument("--output", type=Path, default=P8_TRANSFER_MANIFEST)
    verify = subparsers.add_parser("verify-transfer", parents=[common])
    verify.add_argument(
        "--transfer-manifest", type=Path, default=P8_TRANSFER_MANIFEST
    )
    listing = subparsers.add_parser("transfer-list", parents=[common])
    listing.add_argument(
        "--transfer-manifest", type=Path, default=P8_TRANSFER_MANIFEST
    )
    stage = subparsers.add_parser("verify-stage-a", parents=[common])
    stage.add_argument(
        "--transfer-manifest", type=Path, default=P8_TRANSFER_MANIFEST
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.repository_root
    if arguments.command == "build-transfer-manifest":
        result = build_transfer_manifest(
            root, arguments.config, arguments.p8_execution_config, arguments.output
        )
        printable = {key: value for key, value in result.items() if key != "files"}
    elif arguments.command == "verify-transfer":
        printable = verify_transfer_manifest(root, arguments.transfer_manifest)
    elif arguments.command == "transfer-list":
        for relative in transfer_file_list(root, arguments.transfer_manifest):
            print(relative)
        return 0
    elif arguments.command == "verify-stage-a":
        printable = verify_stage_a(
            root,
            arguments.config,
            arguments.p8_execution_config,
            arguments.transfer_manifest,
        )
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(printable).decode("utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
