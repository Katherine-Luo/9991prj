from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lidc_baseline.config import load_config
from lidc_baseline.v2_migration import (
    apply_v2_task_semantics,
    rematerialize_v2_manifest,
)


def _source_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "nodule_uid": "a" * 64,
                "patient_id": "patient-a",
                "mean_malignancy": 2.0,
                "all_required_targets_valid": True,
                "missing_required_target_fields": [],
                "cohort_status": "PRIMARY_BINARY",
            },
            {
                "nodule_uid": "b" * 64,
                "patient_id": "patient-b",
                "mean_malignancy": 3.0,
                "all_required_targets_valid": True,
                "missing_required_target_fields": [],
                "cohort_status": "EXCLUDED_UNCERTAIN_MALIGNANCY",
            },
            {
                "nodule_uid": "c" * 64,
                "patient_id": "patient-c",
                "mean_malignancy": 4.0,
                "all_required_targets_valid": True,
                "missing_required_target_fields": [],
                "cohort_status": "PRIMARY_BINARY",
            },
            {
                "nodule_uid": "d" * 64,
                "patient_id": "patient-d",
                "mean_malignancy": 3.0,
                "all_required_targets_valid": False,
                "missing_required_target_fields": ["texture"],
                "cohort_status": "EXCLUDED_MISSING_REQUIRED_TARGET",
            },
        ]
    )


def test_v2_semantics_retain_middle_spectrum_for_primary_regression() -> None:
    source = _source_fixture()
    result = apply_v2_task_semantics(source)
    assert result["nodule_uid"].tolist() == source["nodule_uid"].tolist()
    assert result["primary_regression_eligible"].tolist() == [True, True, True, False]
    assert result["extreme_binary_eligible"].tolist() == [True, False, True, False]
    assert result["extreme_binary_label"].tolist()[:3] == [0, pd.NA, 1]
    assert result["malignancy_target_normalized"].tolist()[:3] == [0.25, 0.5, 0.75]
    assert result["cohort_status_v2"].tolist() == [
        "PRIMARY_REGRESSION",
        "PRIMARY_REGRESSION",
        "PRIMARY_REGRESSION",
        "EXCLUDED_MISSING_REQUIRED_TARGET",
    ]


def test_v2_rematerialization_is_deterministic_and_deidentified(tmp_path: Path) -> None:
    source_path = tmp_path / "source.parquet"
    _source_fixture().to_parquet(source_path, index=False)
    config = load_config("configs/baseline_v2.yaml")
    expected = {
        "physical_clusters": 4,
        "primary_regression_nodules": 3,
        "primary_regression_patients": 3,
        "secondary_extreme_nodules": 2,
        "secondary_extreme_patients": 2,
        "secondary_low_nodules": 1,
        "secondary_high_nodules": 1,
        "middle_spectrum_primary_nodules": 1,
        "excluded_missing_required_target": 1,
    }
    audits = []
    for name in ("first", "second"):
        audit = tmp_path / name / "audit"
        rematerialize_v2_manifest(
            source_path,
            tmp_path / name / "manifest.parquet",
            audit,
            config,
            expected,
        )
        audits.append(audit)

    assert (audits[0] / "summary.json").read_bytes() == (
        audits[1] / "summary.json"
    ).read_bytes()
    assert (audits[0] / "reconciliation.csv").read_bytes() == (
        audits[1] / "reconciliation.csv"
    ).read_bytes()
    summary = json.loads((audits[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["nodule_uid_changed"] is False
    assert summary["raw_source_data_accessed"] is False
    assert "patient-a" not in json.dumps(summary)
    assert str(tmp_path) not in json.dumps(summary)


def test_tracked_v2_cohort_audit_matches_pre_registered_counts() -> None:
    summary = json.loads(
        Path("artifacts/baseline_v2/audit/p2/summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["counts"] == {
        "excluded_missing_required_target": 1,
        "middle_spectrum_primary_nodules": 1560,
        "physical_clusters": 2634,
        "primary_regression_nodules": 2633,
        "primary_regression_patients": 868,
        "secondary_extreme_nodules": 1073,
        "secondary_extreme_patients": 578,
        "secondary_high_nodules": 291,
        "secondary_low_nodules": 782,
    }
    assert summary["nodule_uid_changed"] is False
    assert summary["raw_source_data_accessed"] is False
