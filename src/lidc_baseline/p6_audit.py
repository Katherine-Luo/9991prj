"""Build private P6 OOF predictions and deidentified scientific evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.audit import write_json
from lidc_baseline.p3_roi import assert_deidentified_audit
from lidc_baseline.p4_prepare import patient_key, sha256_file
from lidc_baseline.p5_blackbox import regression_metrics
from lidc_baseline.p6_standard_cbm import (
    CONCEPT_GROUP_ORDER,
    P6_EXECUTION_CONFIG_DEFAULT,
    _atomic_parquet,
    run_directory,
    verify_all,
    verify_fold,
)


SCHEMA_VERSION = 1
EXPECTED_FOLD_TEST_COUNTS = (479, 502, 539, 549, 564)
EXPECTED_OOF_NODULES = 2633
EXPECTED_OOF_PATIENTS = 868
MODEL_NAME = "standard_cbm"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P6_AUDIT_EXPECTED_OBJECT:{path.name}")
    return payload


def contribution_reconstruction_errors(frame: pd.DataFrame) -> dict[str, float]:
    raw = frame["raw_bias"].to_numpy(dtype=np.float64).copy()
    rating = frame["rating_scale_bias"].to_numpy(dtype=np.float64).copy()
    scale_error = np.zeros(len(frame), dtype=np.float64)
    for group in CONCEPT_GROUP_ORDER:
        raw_group = frame[f"{group}_raw_contribution"].to_numpy(dtype=np.float64)
        rating_group = frame[f"{group}_rating_point_contribution"].to_numpy(
            dtype=np.float64
        )
        raw += raw_group
        rating += rating_group
        scale_error = np.maximum(scale_error, np.abs(rating_group - 4.0 * raw_group))
    raw_error = np.abs(raw - frame["malignancy_raw_score"].to_numpy(dtype=np.float64))
    rating_error = np.abs(
        rating - frame["malignancy_score_1_to_5"].to_numpy(dtype=np.float64)
    )
    bias_error = np.abs(
        frame["rating_scale_bias"].to_numpy(dtype=np.float64)
        - (1.0 + 4.0 * frame["raw_bias"].to_numpy(dtype=np.float64))
    )
    return {
        "normalized_max_absolute_error": float(raw_error.max(initial=0.0)),
        "rating_scale_max_absolute_error": float(rating_error.max(initial=0.0)),
        "contribution_scale_max_absolute_error": float(scale_error.max(initial=0.0)),
        "bias_scale_max_absolute_error": float(bias_error.max(initial=0.0)),
    }


def _private_run_storage(run: Path) -> dict[str, int]:
    files = [path for path in run.rglob("*") if path.is_file() and not path.name.startswith(".p6_")]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def _forbidden_source_values(manifest_path: Path) -> set[str]:
    frame = pd.read_parquet(manifest_path)
    forbidden: set[str] = set()
    for column in (
        "nodule_uid",
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "scan_id",
    ):
        if column in frame:
            forbidden.update(frame[column].dropna().astype(str))
    return forbidden


def validate_manifest_patient_mapping(
    pooled: pd.DataFrame, manifest_path: Path
) -> int:
    """Independently verify prediction patient keys from the private manifest."""
    manifest = pd.read_parquet(manifest_path)
    required = {"nodule_uid", "patient_id", "primary_regression_eligible"}
    if not required.issubset(manifest.columns):
        raise ValueError("P6_AUDIT_MANIFEST_PATIENT_SCHEMA_MISSING")
    primary = manifest.loc[
        manifest["primary_regression_eligible"].astype(bool),
        ["nodule_uid", "patient_id"],
    ].copy()
    primary["nodule_uid"] = primary["nodule_uid"].astype(str)
    if (
        len(primary) != EXPECTED_OOF_NODULES
        or primary["nodule_uid"].nunique() != EXPECTED_OOF_NODULES
    ):
        raise ValueError("P6_AUDIT_MANIFEST_PRIMARY_SET_MISMATCH")
    expected = {
        str(row.nodule_uid): patient_key(str(row.patient_id))
        for row in primary.itertuples(index=False)
    }
    observed_uids = pooled["nodule_uid"].astype(str)
    if set(observed_uids) != set(expected):
        raise ValueError("P6_AUDIT_MANIFEST_OOF_UID_SET_MISMATCH")
    observed_keys = pooled["patient_key"].astype(str)
    mismatched = [
        uid
        for uid, key in zip(observed_uids, observed_keys, strict=True)
        if key != expected[uid]
    ]
    if mismatched:
        raise ValueError("P6_AUDIT_PATIENT_KEY_MAPPING_MISMATCH")
    patient_count = len(set(expected.values()))
    if patient_count != EXPECTED_OOF_PATIENTS:
        raise ValueError("P6_AUDIT_MANIFEST_PATIENT_COUNT_MISMATCH")
    return patient_count


def _fold_report(
    *,
    fold_index: int,
    verified: Mapping[str, Any],
    run_root: Path,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    run = run_directory(fold_index, run_root)
    concept = _read_json(run / "concept_stage" / "training_complete.json")
    task = _read_json(run / "task_stage" / "training_complete.json")
    sequential = _read_json(run / "sequential_training_complete.json")
    evaluation = _read_json(run / "test_evaluation.json")
    before = str(sequential["frozen_predictor_semantic_sha256_before_task"])
    after = str(sequential["frozen_predictor_semantic_sha256_after_task"])
    bn_before = str(sequential["frozen_batchnorm_state_sha256_before_task"])
    bn_after = str(sequential["frozen_batchnorm_state_sha256_after_task"])
    if before != after or bn_before != bn_after:
        raise ValueError(f"P6_AUDIT_FROZEN_PREDICTOR_CHANGED:{fold_index}")
    errors = contribution_reconstruction_errors(predictions)
    if max(errors.values()) > 1e-6:
        raise ValueError(f"P6_AUDIT_CONTRIBUTION_RECONSTRUCTION_FAILED:{fold_index}")
    if evaluation.get("status") != "TEST_EVALUATED_EXACTLY_ONCE":
        raise ValueError(f"P6_AUDIT_TEST_NOT_EXACTLY_ONCE:{fold_index}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": sequential["protocol_version"],
        "model": MODEL_NAME,
        "fold_index": fold_index,
        "concept_epochs": int(verified["concept_epochs"]),
        "task_epochs": int(verified["task_epochs"]),
        "train_samples_per_epoch": int(verified["train_samples_per_epoch"]),
        "test_samples": int(verified["test_samples"]),
        "concept_best_epoch_index": int(concept["best_epoch_index"]),
        "concept_best_validation_loss": float(concept["best_validation_objective"]),
        "task_best_epoch_index": int(task["best_epoch_index"]),
        "task_best_validation_mse": float(task["best_validation_objective"]),
        "test_evaluated_once": bool(verified["test_evaluated_once"]),
        "test_inference_transactions": int(evaluation["test_inference_transactions"]),
        "frozen_predictor_semantic_hash_unchanged": before == after,
        "frozen_batchnorm_state_hash_unchanged": bn_before == bn_after,
        "contribution_reconstruction": errors,
        "task_metrics": regression_metrics(predictions.to_dict("records")),
        "scientific_config_sha256": sequential["scientific_config_sha256"],
        "execution_config_sha256": sequential["execution_config_sha256"],
        "p6_execution_config_sha256": sequential["p6_execution_config_sha256"],
        "split_sha256": sequential["split_sha256"],
        "encoder_initialization_sha256": sequential[
            "encoder_initialization_sha256"
        ],
        "concept_best_checkpoint_sha256": sequential[
            "concept_best_checkpoint_sha256"
        ],
        "task_best_checkpoint_sha256": predictions[
            "task_best_checkpoint_sha256"
        ].astype(str).iloc[0],
        "test_predictions_sha256": sha256_file(run / "test_predictions.parquet"),
        "test_evaluation_sha256": sha256_file(run / "test_evaluation.json"),
        "private_run_storage": _private_run_storage(run),
    }


def build_oof_audit(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    run_root: Path,
    audit_root: Path,
    oof_predictions_path: Path,
) -> dict[str, Any]:
    """Verify five completed folds and build private OOF plus tracked evidence."""
    verified_all = verify_all(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        p6_execution_config_path=p6_execution_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        output_root=run_root,
    )
    if verified_all.get("status") != "PASS":
        raise ValueError("P6_AUDIT_FINAL_VERIFY_NOT_PASS")
    verified_by_fold = {int(item["fold_index"]): item for item in verified_all["folds"]}
    if set(verified_by_fold) != set(range(5)):
        raise ValueError("P6_AUDIT_FOLD_SET_MISMATCH")

    frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    audit_root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        verify_fold(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p6_execution_config_path=p6_execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold,
            output_root=run_root,
            require_test=True,
        )
        predictions = pd.read_parquet(
            run_directory(fold, run_root) / "test_predictions.parquet"
        )
        report = _fold_report(
            fold_index=fold,
            verified=verified_by_fold[fold],
            run_root=run_root,
            predictions=predictions,
        )
        write_json(audit_root / f"fold_{fold}.json", report)
        fold_reports.append(report)
        frames.append(predictions)

    pooled = (
        pd.concat(frames, ignore_index=True)
        .sort_values("nodule_uid", kind="stable")
        .reset_index(drop=True)
    )
    if len(pooled) != EXPECTED_OOF_NODULES or pooled["nodule_uid"].astype(str).nunique() != EXPECTED_OOF_NODULES:
        raise ValueError("P6_AUDIT_OOF_NODULE_SET_MISMATCH")
    manifest_patient_count = validate_manifest_patient_mapping(pooled, manifest_path)
    if pooled["patient_key"].astype(str).nunique() != manifest_patient_count:
        raise ValueError("P6_AUDIT_OOF_PATIENT_SET_MISMATCH")
    if int(pooled.groupby("patient_key")["fold_index"].nunique().max()) != 1:
        raise ValueError("P6_AUDIT_PATIENT_LEAKAGE")
    fold_counts = tuple(
        int(value)
        for value in pooled["fold_index"]
        .astype(int)
        .value_counts()
        .reindex(range(5), fill_value=0)
        .tolist()
    )
    if fold_counts != EXPECTED_FOLD_TEST_COUNTS:
        raise ValueError("P6_AUDIT_FOLD_TEST_COUNTS_MISMATCH")
    pooled_errors = contribution_reconstruction_errors(pooled)
    if max(pooled_errors.values()) > 1e-6:
        raise ValueError("P6_AUDIT_POOLED_RECONSTRUCTION_FAILED")
    _atomic_parquet(oof_predictions_path, pooled)

    pooled_task = regression_metrics(pooled.to_dict("records"))
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": fold_reports[0]["protocol_version"],
        "model": MODEL_NAME,
        "folds": 5,
        "oof_nodules": EXPECTED_OOF_NODULES,
        "oof_unique_nodules": EXPECTED_OOF_NODULES,
        "oof_patients": EXPECTED_OOF_PATIENTS,
        "patient_leakage": 0,
        "fold_test_counts": list(EXPECTED_FOLD_TEST_COUNTS),
        "test_evaluated_once_all_folds": all(
            item["test_evaluated_once"]
            and item["test_inference_transactions"] == 1
            for item in fold_reports
        ),
        "concept_predictor_semantic_hash_unchanged_all_folds": all(
            item["frozen_predictor_semantic_hash_unchanged"]
            for item in fold_reports
        ),
        "batchnorm_state_hash_unchanged_all_folds": all(
            item["frozen_batchnorm_state_hash_unchanged"]
            for item in fold_reports
        ),
        "pooled_oof_task_metrics": pooled_task,
        "pooled_contribution_reconstruction": pooled_errors,
        "concept_best_epoch_index_by_fold": [
            item["concept_best_epoch_index"] for item in fold_reports
        ],
        "concept_best_validation_loss_by_fold": [
            item["concept_best_validation_loss"] for item in fold_reports
        ],
        "task_best_epoch_index_by_fold": [
            item["task_best_epoch_index"] for item in fold_reports
        ],
        "task_best_validation_mse_by_fold": [
            item["task_best_validation_mse"] for item in fold_reports
        ],
        "scientific_config_sha256": fold_reports[0]["scientific_config_sha256"],
        "execution_config_sha256": fold_reports[0]["execution_config_sha256"],
        "p6_execution_config_sha256": fold_reports[0][
            "p6_execution_config_sha256"
        ],
        "split_sha256_by_fold": [item["split_sha256"] for item in fold_reports],
        "encoder_initialization_sha256_by_fold": [
            item["encoder_initialization_sha256"] for item in fold_reports
        ],
        "oof_predictions_sha256": sha256_file(oof_predictions_path),
        "private_run_storage": {
            "file_count": sum(
                int(item["private_run_storage"]["file_count"])
                for item in fold_reports
            ),
            "total_bytes": sum(
                int(item["private_run_storage"]["total_bytes"])
                for item in fold_reports
            ),
        },
    }
    output_path = audit_root / "summary.json"
    write_json(output_path, report)
    forbidden = _forbidden_source_values(manifest_path)
    for path in [*(audit_root / f"fold_{fold}.json" for fold in range(5)), output_path]:
        assert_deidentified_audit(path, forbidden)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-oof")
    build.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    build.add_argument(
        "--execution-config",
        type=Path,
        default=Path(
            "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
        ),
    )
    build.add_argument(
        "--p6-execution-config", type=Path, default=P6_EXECUTION_CONFIG_DEFAULT
    )
    build.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/baseline_v2/manifests/nodules.parquet"),
    )
    build.add_argument(
        "--roi-index",
        type=Path,
        default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"),
    )
    build.add_argument(
        "--run-root", type=Path, default=Path("runs/baseline_v2/standard_cbm")
    )
    build.add_argument(
        "--audit-root", type=Path, default=Path("artifacts/baseline_v2/audit/p6")
    )
    build.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path("runs/baseline_v2/standard_cbm/oof_predictions.parquet"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command != "build-oof":  # pragma: no cover
        raise AssertionError(arguments.command)
    report = build_oof_audit(
        scientific_config_path=arguments.config,
        execution_config_path=arguments.execution_config,
        p6_execution_config_path=arguments.p6_execution_config,
        manifest_path=arguments.manifest,
        roi_index_path=arguments.roi_index,
        run_root=arguments.run_root,
        audit_root=arguments.audit_root,
        oof_predictions_path=arguments.oof_predictions,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
