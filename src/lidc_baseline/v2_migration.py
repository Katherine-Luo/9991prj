"""Rematerialize Baseline-v2 task semantics from the private P2 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lidc_baseline import __version__
from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.regression import (
    extreme_binary_label,
    malignancy_stratum,
    normalize_malignancy_target,
)

REQUIRED_SOURCE_COLUMNS = {
    "nodule_uid",
    "patient_id",
    "mean_malignancy",
    "all_required_targets_valid",
    "missing_required_target_fields",
    "cohort_status",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _validate_source(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Source manifest is missing columns: {missing}")
    if frame["nodule_uid"].isna().any() or frame["nodule_uid"].duplicated().any():
        raise ValueError("Source nodule_uid values must be present and unique")
    valid_means = frame.loc[frame["mean_malignancy"].notna(), "mean_malignancy"]
    if ((valid_means < 1.0) | (valid_means > 5.0)).any():
        raise ValueError("Source mean_malignancy values must be in [1, 5]")


def apply_v2_task_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with V2 regression and extreme-subset fields."""
    _validate_source(frame)
    result = frame.copy(deep=True)
    primary = result["all_required_targets_valid"].fillna(False).astype(bool)
    if result.loc[primary, "mean_malignancy"].isna().any():
        raise ValueError("Primary regression rows require mean_malignancy")

    result["protocol_version"] = "Baseline-v2"
    result["primary_regression_eligible"] = primary
    result["malignancy_target_normalized"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result.loc[primary, "malignancy_target_normalized"] = result.loc[
        primary, "mean_malignancy"
    ].map(normalize_malignancy_target)
    result["malignancy_stratum"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[primary, "malignancy_stratum"] = result.loc[
        primary, "mean_malignancy"
    ].map(malignancy_stratum)

    labels = result.loc[primary, "mean_malignancy"].map(extreme_binary_label)
    extreme = pd.Series(False, index=result.index, dtype=bool)
    extreme.loc[primary] = labels.notna()
    result["extreme_binary_eligible"] = extreme
    result["extreme_binary_label"] = pd.Series(pd.NA, index=result.index, dtype="Int8")
    result.loc[extreme, "extreme_binary_label"] = labels.loc[labels.notna()].astype("int8")
    result["cohort_status_v2"] = "EXCLUDED_MISSING_REQUIRED_TARGET"
    result.loc[primary, "cohort_status_v2"] = "PRIMARY_REGRESSION"
    return result


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    primary = frame[frame["primary_regression_eligible"]]
    extreme = frame[frame["extreme_binary_eligible"]]
    return {
        "physical_clusters": int(len(frame)),
        "primary_regression_nodules": int(len(primary)),
        "primary_regression_patients": int(primary["patient_id"].nunique()),
        "secondary_extreme_nodules": int(len(extreme)),
        "secondary_extreme_patients": int(extreme["patient_id"].nunique()),
        "secondary_low_nodules": int((extreme["extreme_binary_label"] == 0).sum()),
        "secondary_high_nodules": int((extreme["extreme_binary_label"] == 1).sum()),
        "middle_spectrum_primary_nodules": int(
            (primary["extreme_binary_eligible"] == False).sum()  # noqa: E712
        ),
        "excluded_missing_required_target": int(
            (frame["cohort_status_v2"] == "EXCLUDED_MISSING_REQUIRED_TARGET").sum()
        ),
    }


def _assert_expected_counts(counts: Mapping[str, int], expected: Mapping[str, int]) -> None:
    mismatches = {
        key: {"expected": int(value), "observed": counts.get(key)}
        for key, value in expected.items()
        if counts.get(key) != int(value)
    }
    if mismatches:
        raise ValueError(f"V2 cohort count mismatch: {mismatches}")


def _records_sha256(frame: pd.DataFrame) -> str:
    rows = frame.sort_values("nodule_uid")[
        [
            "nodule_uid",
            "primary_regression_eligible",
            "malignancy_target_normalized",
            "malignancy_stratum",
            "extreme_binary_eligible",
            "extreme_binary_label",
            "cohort_status_v2",
        ]
    ].astype(object)
    canonical_rows = rows.where(pd.notna(rows), None).to_dict(orient="records")
    payload = json.dumps(
        canonical_rows,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rematerialize_v2_manifest(
    source_manifest: str | Path,
    output_manifest: str | Path,
    audit_directory: str | Path,
    config: Mapping[str, Any],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Create the private V2 manifest and deterministic deidentified audit."""
    source_path = Path(source_manifest)
    output_path = Path(output_manifest)
    audit_path = Path(audit_directory)
    source = pd.read_parquet(source_path)
    result = apply_v2_task_semantics(source)
    counts = _counts(result)
    if expected_counts is not None:
        _assert_expected_counts(counts, expected_counts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("nodule_uid").to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    primary = result[result["primary_regression_eligible"]]
    stratum_rows = []
    for stratum in (
        "mean_le_2",
        "mean_gt_2_lt_3",
        "mean_eq_3",
        "mean_gt_3_lt_4",
        "mean_ge_4",
    ):
        subset = primary[primary["malignancy_stratum"] == stratum]
        stratum_rows.append(
            {
                "stratum": stratum,
                "nodules": int(len(subset)),
                "patients": int(subset["patient_id"].nunique()),
            }
        )

    reconciliation_rows = [
        {"stage": "physical_clusters", "count": counts["physical_clusters"]},
        {"stage": "primary_regression", "count": counts["primary_regression_nodules"]},
        {"stage": "middle_spectrum_retained", "count": counts["middle_spectrum_primary_nodules"]},
        {"stage": "secondary_extreme", "count": counts["secondary_extreme_nodules"]},
        {"stage": "secondary_low", "count": counts["secondary_low_nodules"]},
        {"stage": "secondary_high", "count": counts["secondary_high_nodules"]},
        {"stage": "excluded_missing_required_target", "count": counts["excluded_missing_required_target"]},
        {"stage": "reference_nodules", "count": int(config["cohort"]["reference_reconciliation"]["nodules"])},
    ]
    _write_csv(audit_path / "reconciliation.csv", ["stage", "count"], reconciliation_rows)
    _write_csv(audit_path / "strata.csv", ["stratum", "nodules", "patients"], stratum_rows)

    summary = {
        "audit": "Baseline-v2 cohort rematerialization",
        "schema_version": 1,
        "program_version": __version__,
        "protocol_version": "Baseline-v2",
        "config_sha256": compute_config_sha256(config),
        "source_manifest_sha256": _sha256_file(source_path),
        "v2_records_sha256": _records_sha256(result),
        "physical_identity_reused": True,
        "nodule_uid_changed": False,
        "raw_source_data_accessed": False,
        "task": {
            "primary": "continuous_regression",
            "target_normalization": "(mean_malignancy - 1) / 4",
            "secondary": "extreme_binary_evaluation_only",
            "independent_binary_head": False,
        },
        "counts": counts,
        "reference_reconciliation": {
            "nodules": int(config["cohort"]["reference_reconciliation"]["nodules"]),
            "patients": int(config["cohort"]["reference_reconciliation"]["patients"]),
            "hard_gate": False,
        },
        "privacy": "Tracked reports contain aggregate counts and cryptographic fingerprints only; the full manifest is local-only.",
    }
    write_json(audit_path / "summary.json", summary)
    return summary


def _expected_counts(config: Mapping[str, Any]) -> dict[str, int]:
    primary = config["cohort"]["primary_regression"]
    secondary = config["cohort"]["secondary_extreme_binary"]
    return {
        "physical_clusters": 2634,
        "primary_regression_nodules": int(primary["nodules"]),
        "primary_regression_patients": int(primary["patients"]),
        "secondary_extreme_nodules": int(secondary["total_nodules"]),
        "secondary_extreme_patients": int(secondary["patients"]),
        "secondary_low_nodules": int(secondary["low_nodules"]),
        "secondary_high_nodules": int(secondary["high_nodules"]),
        "middle_spectrum_primary_nodules": 1560,
        "excluded_missing_required_target": 1,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--audit-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the V2 cohort rematerialization command."""
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    paths = config["paths"]
    rematerialize_v2_manifest(
        source_manifest=arguments.source_manifest or paths["source_v1_manifest"],
        output_manifest=arguments.output_manifest or paths["manifest"],
        audit_directory=arguments.audit_directory or Path(paths["audit_directory"]) / "p2",
        config=config,
        expected_counts=_expected_counts(config),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
