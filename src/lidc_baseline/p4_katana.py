"""Prepare and verify the private P4 Katana transfer and CUDA smoke."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p3_roi import assert_deidentified_audit
from lidc_baseline.p4_prepare import (
    CONSUMERS,
    build_encoder,
    canonical_json_bytes,
    load_shared_encoder_initialization,
    read_split,
    sha256_bytes,
    sha256_file,
    validate_encoder_artifact,
    verify as verify_p4,
)


SCHEMA_VERSION = 1
TRANSFER_MANIFEST_RELATIVE_PATH = Path("artifacts/baseline_v2/manifests/p4_transfer_manifest.json")


def _relative_file(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"TRANSFER_FILE_OUTSIDE_REPOSITORY:{path}") from error
    if not resolved_path.is_file():
        raise FileNotFoundError(f"TRANSFER_FILE_MISSING:{relative}")
    return relative.as_posix()


def _source_files(root: Path) -> list[Path]:
    fixed = [
        root / "pyproject.toml",
        root / "configs/baseline_v2.yaml",
        root / "configs/baseline_v2.resolved.yaml",
        root / "configs/baseline_v2.sha256",
        root / "environment/katana-cuda.yml",
        root / "environment/locks/katana-linux-conda-explicit.txt",
        root / "environment/locks/katana-linux-pip-freeze.txt",
        root / "scripts/katana/p4_cuda_smoke.pbs",
    ]
    source = sorted((root / "src/lidc_baseline").rglob("*.py"))
    return fixed + source


def transfer_files(root: Path, config: Mapping[str, Any]) -> list[Path]:
    """Return the exact files needed for the P4 Katana smoke."""
    repository = root.resolve()
    manifest = repository / str(config["paths"]["manifest"])
    roi_index = repository / "artifacts/baseline_v2/manifests/roi_index.parquet"
    split_root = repository / str(config["paths"]["split_directory"])
    encoder_root = repository / str(config["paths"]["encoder_initialization_directory"])
    roi_root = repository / str(config["paths"]["roi_directory"])
    fold_count = int(config["splits"]["outer_folds"])
    paths = _source_files(repository) + [manifest, roi_index]
    paths.extend(split_root / f"fold_{fold}.json" for fold in range(fold_count))
    paths.extend(encoder_root / f"fold_{fold}.pt" for fold in range(fold_count))
    roi_paths = sorted(roi_root.glob("*.npz"))
    expected_rois = int(config["cohort"]["primary_regression"]["nodules"])
    if len(roi_paths) != expected_rois:
        raise ValueError(f"TRANSFER_ROI_COUNT_MISMATCH:{len(roi_paths)}")
    paths.extend(roi_paths)
    relative_paths = [_relative_file(repository, path) for path in paths]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("TRANSFER_FILE_DUPLICATE")
    forbidden = (".git/", "lidc_data/", "reports/", "runs/", "__pycache__/", ".pytest_cache/")
    if any(any(token in relative for token in forbidden) for relative in relative_paths):
        raise ValueError("TRANSFER_FORBIDDEN_PATH")
    return [repository / relative for relative in sorted(relative_paths)]


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("transfer_manifest_sha256", None)
    return sha256_bytes(canonical_json_bytes(unhashed))


def build_transfer_manifest(
    repository_root: Path,
    config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create a private file-level transfer manifest after full local P4 verify."""
    root = repository_root.resolve()
    config = load_config(root / config_path)
    local_verification = verify_p4(root / config_path, root / manifest_path, root / roi_index_path)
    files = transfer_files(root, config)
    entries = [
        {
            "relative_path": _relative_file(root, path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    categories = {
        "source_and_environment_files": sum(
            entry["relative_path"].startswith(("src/", "scripts/", "configs/", "environment/"))
            or entry["relative_path"] == "pyproject.toml"
            for entry in entries
        ),
        "manifest_files": sum("/manifests/" in entry["relative_path"] for entry in entries),
        "roi_files": sum(entry["relative_path"].startswith("artifacts/baseline_v2/rois/") for entry in entries),
        "split_files": sum(entry["relative_path"].startswith("artifacts/baseline_v2/splits/fold_") for entry in entries),
        "encoder_files": sum(entry["relative_path"].startswith("artifacts/baseline_v2/encoder_initializations/fold_") for entry in entries),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": config["protocol"]["version"],
        "config_sha256": compute_config_sha256(config),
        "manifest_sha256": sha256_file(root / manifest_path),
        "roi_index_sha256": sha256_file(root / roi_index_path),
        "primary_nodules": local_verification["primary_nodules"],
        "primary_patients": local_verification["primary_patients"],
        "roi_set_sha256": local_verification["roi_integrity"]["roi_set_sha256"],
        "total_files": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "categories": categories,
        "files": entries,
    }
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    destination = root / output_path
    write_json(destination, payload)
    return payload


def read_transfer_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = payload.get("transfer_manifest_sha256")
    if declared != _manifest_digest(payload):
        raise ValueError("TRANSFER_MANIFEST_HASH_MISMATCH")
    return payload


def verify_transfer_manifest(repository_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every transferred file without allowing absolute manifest paths."""
    root = repository_root.resolve()
    payload = read_transfer_manifest(root / manifest_path)
    observed_bytes = 0
    relative_paths: list[str] = []
    for entry in payload["files"]:
        relative = str(entry["relative_path"])
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("TRANSFER_MANIFEST_UNSAFE_PATH")
        path = root / candidate
        _relative_file(root, path)
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"TRANSFER_FILE_SIZE_MISMATCH:{relative}")
        if sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"TRANSFER_FILE_HASH_MISMATCH:{relative}")
        observed_bytes += path.stat().st_size
        relative_paths.append(relative)
    if len(relative_paths) != len(set(relative_paths)) or len(relative_paths) != int(payload["total_files"]):
        raise ValueError("TRANSFER_MANIFEST_FILE_SET_MISMATCH")
    if observed_bytes != int(payload["total_bytes"]):
        raise ValueError("TRANSFER_MANIFEST_TOTAL_BYTES_MISMATCH")
    return {
        "status": "PASS",
        "transfer_manifest_sha256": payload["transfer_manifest_sha256"],
        "transfer_manifest_file_sha256": sha256_file(root / manifest_path),
        "verified_files": len(relative_paths),
        "verified_bytes": observed_bytes,
        "categories": payload["categories"],
    }


def transfer_file_list(repository_root: Path, manifest_path: Path) -> list[str]:
    """Return the exact rsync whitelist, including the manifest anchor file."""
    root = repository_root.resolve()
    payload = read_transfer_manifest(root / manifest_path)
    relative_manifest = _relative_file(root, root / manifest_path)
    paths = [str(entry["relative_path"]) for entry in payload["files"]] + [relative_manifest]
    if len(paths) != len(set(paths)):
        raise ValueError("TRANSFER_WHITELIST_DUPLICATE")
    for relative in paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("TRANSFER_WHITELIST_UNSAFE_PATH")
        _relative_file(root, root / candidate)
    return sorted(paths)


def load_roi_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one real P3 ROI and enforce the frozen standard interface."""
    with np.load(path, allow_pickle=False) as archive:
        image = archive["image"]
        mask = archive["mask"]
    if image.shape != (1, 64, 64, 64) or image.dtype != np.float32:
        raise ValueError("KATANA_ROI_IMAGE_INTERFACE_MISMATCH")
    if mask.shape != (1, 64, 64, 64) or mask.dtype != np.uint8:
        raise ValueError("KATANA_ROI_MASK_INTERFACE_MISMATCH")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError("KATANA_ROI_IMAGE_VALUE_MISMATCH")
    if not mask.any() or not set(np.unique(mask)).issubset({0, 1}):
        raise ValueError("KATANA_ROI_MASK_VALUE_MISMATCH")
    return image, mask


def _driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines()[0]


def run_cuda_smoke(
    repository_root: Path,
    config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    transfer_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run the P4 no-training CUDA loading/hash/forward smoke on Katana."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("P4_KATANA_CUDA_UNAVAILABLE")
    root = repository_root.resolve()
    transfer = verify_transfer_manifest(root, transfer_manifest_path)
    config = load_config(root / config_path)
    local = verify_p4(root / config_path, root / manifest_path, root / roi_index_path)
    roi_index = pd.read_parquet(root / roi_index_path).set_index("nodule_uid", drop=False)
    split_root = root / str(config["paths"]["split_directory"])
    encoder_root = root / str(config["paths"]["encoder_initialization_directory"])
    roi_root = root / str(config["paths"]["roi_directory"])
    fold_reports: list[dict[str, Any]] = []
    for fold_index in range(int(config["splits"]["outer_folds"])):
        split = read_split(split_root / f"fold_{fold_index}.json")
        partition_samples: dict[str, dict[str, Any]] = {}
        loaded_images: dict[str, np.ndarray] = {}
        for partition in ("train", "validation", "test"):
            uid = sorted(split["partitions"][partition]["nodule_uids"])[0]
            row = roi_index.loc[uid]
            path = roi_root.parent / str(row["relative_roi_path"])
            image, mask = load_roi_sample(path)
            loaded_images[partition] = image
            partition_samples[partition] = {
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "mask_shape": list(mask.shape),
                "mask_dtype": str(mask.dtype),
            }
        artifact_path = encoder_root / f"fold_{fold_index}.pt"
        validated = validate_encoder_artifact(artifact_path, config, split)
        encoders = [build_encoder() for _ in CONSUMERS]
        consumer_hashes = [
            load_shared_encoder_initialization(encoder, artifact_path, config, split)
            for encoder in encoders
        ]
        if len({id(encoder) for encoder in encoders}) != len(CONSUMERS) or len(set(consumer_hashes)) != 1:
            raise ValueError(f"KATANA_SHARED_ENCODER_HASH_MISMATCH:{fold_index}")
        forward_encoder = encoders[0].eval().to("cuda")
        input_tensor = torch.from_numpy(loaded_images["train"]).unsqueeze(0).to("cuda")
        with torch.no_grad():
            output = forward_encoder(input_tensor)
        if not torch.isfinite(output).all():
            raise ValueError(f"KATANA_CUDA_FORWARD_NONFINITE:{fold_index}")
        fold_reports.append({
            "fold_index": fold_index,
            "split_sha256": split["split_sha256"],
            "encoder_state_sha256": validated["metadata"]["encoder_state_sha256"],
            "encoder_file_sha256": sha256_file(artifact_path),
            "consumer_count": len(CONSUMERS),
            "consumer_hashes_equal": True,
            "partition_samples": partition_samples,
            "cuda_forward_input_shape": list(input_tensor.shape),
            "cuda_forward_output_shape": list(output.shape),
            "cuda_forward_finite": True,
        })
        del output, input_tensor, forward_encoder, encoders
        torch.cuda.empty_cache()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": config["protocol"]["version"],
        "config_sha256": compute_config_sha256(config),
        "transfer": transfer,
        "primary_nodules": local["primary_nodules"],
        "primary_patients": local["primary_patients"],
        "roi_integrity": local["roi_integrity"],
        "folds": fold_reports,
        "runtime": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "monai": importlib.metadata.version("monai"),
            "numpy": importlib.metadata.version("numpy"),
            "cuda_runtime": str(torch.version.cuda),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_driver": _driver_version(),
            "pbs_job_id": os.environ.get("PBS_JOBID", "interactive"),
        },
        "training_operations": {
            "optimizer_created": False,
            "backward_called": False,
            "parameter_update": False,
        },
    }
    write_json(root / output_path, report)
    return report


def write_aggregate_audit(
    repository_root: Path,
    config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    remote_report_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write the deidentified tracked P4 aggregate audit after remote PASS."""
    root = repository_root.resolve()
    config = load_config(root / config_path)
    local = verify_p4(root / config_path, root / manifest_path, root / roi_index_path)
    remote = json.loads((root / remote_report_path).read_text(encoding="utf-8"))
    if remote.get("status") != "PASS" or remote.get("config_sha256") != compute_config_sha256(config):
        raise ValueError("KATANA_REMOTE_REPORT_MISMATCH")
    if remote.get("primary_nodules") != local["primary_nodules"] or remote.get("roi_integrity") != local["roi_integrity"]:
        raise ValueError("KATANA_REMOTE_DATASET_MISMATCH")
    local_transfer_path = root / TRANSFER_MANIFEST_RELATIVE_PATH
    local_transfer = read_transfer_manifest(local_transfer_path)
    if remote.get("transfer", {}).get("transfer_manifest_sha256") != local_transfer["transfer_manifest_sha256"]:
        raise ValueError("KATANA_TRANSFER_MANIFEST_MISMATCH")
    if remote.get("transfer", {}).get("transfer_manifest_file_sha256") != sha256_file(local_transfer_path):
        raise ValueError("KATANA_TRANSFER_MANIFEST_FILE_MISMATCH")
    output = root / output_root
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": config["protocol"]["version"],
        "config_sha256": compute_config_sha256(config),
        "primary_nodules": local["primary_nodules"],
        "primary_patients": local["primary_patients"],
        "roi_integrity": local["roi_integrity"],
        "fold_count": len(local["folds"]),
        "patient_leakage": 0,
        "oof_nodule_coverage": local["primary_nodules"],
        "oof_patient_coverage": local["primary_patients"],
        "katana_cuda_status": remote["status"],
        "katana_transfer_manifest_sha256": remote["transfer"]["transfer_manifest_sha256"],
        "katana_verified_files": remote["transfer"]["verified_files"],
        "katana_verified_bytes": remote["transfer"]["verified_bytes"],
        "katana_runtime": remote["runtime"],
        "training_operations": remote["training_operations"],
    }
    write_json(output / "summary.json", summary)
    with (output / "folds.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "fold", "split_sha256", "train_nodules", "train_patients",
                "validation_nodules", "validation_patients", "test_nodules", "test_patients",
                "validation_low", "validation_high", "test_low", "test_high",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        split_root = root / str(config["paths"]["split_directory"])
        for fold in range(int(config["splits"]["outer_folds"])):
            split = read_split(split_root / f"fold_{fold}.json")
            partitions = split["partitions"]
            writer.writerow({
                "fold": fold,
                "split_sha256": split["split_sha256"],
                "train_nodules": partitions["train"]["summary"]["nodules"],
                "train_patients": partitions["train"]["summary"]["patients"],
                "validation_nodules": partitions["validation"]["summary"]["nodules"],
                "validation_patients": partitions["validation"]["summary"]["patients"],
                "test_nodules": partitions["test"]["summary"]["nodules"],
                "test_patients": partitions["test"]["summary"]["patients"],
                "validation_low": partitions["validation"]["summary"]["extremes"]["low"],
                "validation_high": partitions["validation"]["summary"]["extremes"]["high"],
                "test_low": partitions["test"]["summary"]["extremes"]["low"],
                "test_high": partitions["test"]["summary"]["extremes"]["high"],
            })
    with (output / "initializations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["fold", "encoder_state_sha256", "encoder_file_sha256", "consumer_count", "consumer_hashes_equal", "cuda_forward_finite"],
            lineterminator="\n",
        )
        writer.writeheader()
        for fold in remote["folds"]:
            writer.writerow({
                "fold": fold["fold_index"],
                "encoder_state_sha256": fold["encoder_state_sha256"],
                "encoder_file_sha256": fold["encoder_file_sha256"],
                "consumer_count": fold["consumer_count"],
                "consumer_hashes_equal": fold["consumer_hashes_equal"],
                "cuda_forward_finite": fold["cuda_forward_finite"],
            })
    raw = pd.read_parquet(root / manifest_path)
    forbidden: set[str] = set(raw["nodule_uid"].astype(str))
    for column in ("patient_id", "study_instance_uid", "series_instance_uid"):
        if column in raw:
            forbidden.update(raw[column].dropna().astype(str))
    for artifact in (output / "summary.json", output / "folds.csv", output / "initializations.csv"):
        assert_deidentified_audit(artifact, forbidden)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repository-root", type=Path, default=Path("."))
    common.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    common.add_argument("--manifest", type=Path, default=Path("artifacts/baseline_v2/manifests/nodules.parquet"))
    common.add_argument("--roi-index", type=Path, default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"))
    build_parser = subparsers.add_parser("build-transfer-manifest", parents=[common])
    build_parser.add_argument("--output", type=Path, default=TRANSFER_MANIFEST_RELATIVE_PATH)
    verify_parser = subparsers.add_parser("verify-transfer", parents=[common])
    verify_parser.add_argument("--transfer-manifest", type=Path, default=TRANSFER_MANIFEST_RELATIVE_PATH)
    list_parser = subparsers.add_parser("transfer-list", parents=[common])
    list_parser.add_argument("--transfer-manifest", type=Path, default=TRANSFER_MANIFEST_RELATIVE_PATH)
    smoke_parser = subparsers.add_parser("cuda-smoke", parents=[common])
    smoke_parser.add_argument("--transfer-manifest", type=Path, default=TRANSFER_MANIFEST_RELATIVE_PATH)
    smoke_parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_v2/audit/p4/katana_cuda.json"))
    audit_parser = subparsers.add_parser("build-audit", parents=[common])
    audit_parser.add_argument("--remote-report", type=Path, required=True)
    audit_parser.add_argument("--output-root", type=Path, default=Path("artifacts/baseline_v2/audit/p4"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build-transfer-manifest":
        result = build_transfer_manifest(arguments.repository_root, arguments.config, arguments.manifest, arguments.roi_index, arguments.output)
        printable = {key: value for key, value in result.items() if key != "files"}
    elif arguments.command == "verify-transfer":
        result = verify_transfer_manifest(arguments.repository_root, arguments.transfer_manifest)
        printable = result
    elif arguments.command == "transfer-list":
        for relative in transfer_file_list(arguments.repository_root, arguments.transfer_manifest):
            print(relative)
        return 0
    elif arguments.command == "cuda-smoke":
        result = run_cuda_smoke(arguments.repository_root, arguments.config, arguments.manifest, arguments.roi_index, arguments.transfer_manifest, arguments.output)
        printable = result
    elif arguments.command == "build-audit":
        result = write_aggregate_audit(arguments.repository_root, arguments.config, arguments.manifest, arguments.roi_index, arguments.remote_report, arguments.output_root)
        printable = result
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(printable).decode("utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
