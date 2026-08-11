"""Build private P8 OOF predictions and deidentified aggregate evidence."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.audit import write_json
from lidc_baseline.p3_roi import assert_deidentified_audit
from lidc_baseline.p4_prepare import patient_key, sha256_file
from lidc_baseline.p5_blackbox import regression_metrics
from lidc_baseline.p6_standard_cbm import CONCEPT_GROUP_ORDER
from lidc_baseline.p8_gam import MODEL_NAME, P8_EXECUTION_CONFIG_DEFAULT
from lidc_baseline.p8_gam_lifecycle import (
    _atomic_parquet,
    run_directory,
    verify_all,
    verify_fold,
)


SCHEMA_VERSION = 1
EXPECTED_FOLD_TEST_COUNTS = (479, 502, 539, 549, 564)
EXPECTED_OOF_NODULES = 2633
EXPECTED_OOF_PATIENTS = 868


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P8_AUDIT_EXPECTED_OBJECT:{path.name}")
    return payload


def contribution_reconstruction_errors(frame: pd.DataFrame) -> dict[str, float]:
    raw = frame["raw_bias"].to_numpy(dtype=np.float64).copy()
    rating = frame["rating_scale_bias"].to_numpy(dtype=np.float64).copy()
    contribution_scale_error = np.zeros(len(frame), dtype=np.float64)
    for group in CONCEPT_GROUP_ORDER:
        raw_group = frame[f"{group}_raw_contribution"].to_numpy(dtype=np.float64)
        rating_group = frame[f"{group}_rating_contribution"].to_numpy(
            dtype=np.float64
        )
        raw += raw_group
        rating += rating_group
        contribution_scale_error = np.maximum(
            contribution_scale_error, np.abs(rating_group - 4.0 * raw_group)
        )
    return {
        "normalized_max_absolute_error": float(
            np.abs(
                raw - frame["malignancy_raw_score"].to_numpy(dtype=np.float64)
            ).max(initial=0.0)
        ),
        "rating_scale_max_absolute_error": float(
            np.abs(
                rating
                - frame["malignancy_score_1_to_5"].to_numpy(dtype=np.float64)
            ).max(initial=0.0)
        ),
        "contribution_scale_max_absolute_error": float(
            contribution_scale_error.max(initial=0.0)
        ),
        "bias_scale_max_absolute_error": float(
            np.abs(
                frame["rating_scale_bias"].to_numpy(dtype=np.float64)
                - (1.0 + 4.0 * frame["raw_bias"].to_numpy(dtype=np.float64))
            ).max(initial=0.0)
        ),
    }


def _private_run_storage(run: Path) -> dict[str, int]:
    files = [
        path
        for path in run.rglob("*")
        if path.is_file() and not path.name.startswith(".p8_")
    ]
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
    """Independently bind each prediction patient key to the private manifest."""
    manifest = pd.read_parquet(manifest_path)
    required = {"nodule_uid", "patient_id", "primary_regression_eligible"}
    if not required.issubset(manifest.columns):
        raise ValueError("P8_AUDIT_MANIFEST_PATIENT_SCHEMA_MISSING")
    primary = manifest.loc[
        manifest["primary_regression_eligible"].astype(bool),
        ["nodule_uid", "patient_id"],
    ].copy()
    primary["nodule_uid"] = primary["nodule_uid"].astype(str)
    if (
        len(primary) != EXPECTED_OOF_NODULES
        or primary["nodule_uid"].nunique() != EXPECTED_OOF_NODULES
    ):
        raise ValueError("P8_AUDIT_MANIFEST_PRIMARY_SET_MISMATCH")
    expected = {
        str(row.nodule_uid): patient_key(str(row.patient_id))
        for row in primary.itertuples(index=False)
    }
    observed_uids = pooled["nodule_uid"].astype(str)
    if set(observed_uids) != set(expected):
        raise ValueError("P8_AUDIT_MANIFEST_OOF_UID_SET_MISMATCH")
    if any(
        observed_key != expected[uid]
        for uid, observed_key in zip(
            observed_uids, pooled["patient_key"].astype(str), strict=True
        )
    ):
        raise ValueError("P8_AUDIT_PATIENT_KEY_MAPPING_MISMATCH")
    patient_count = len(set(expected.values()))
    if patient_count != EXPECTED_OOF_PATIENTS:
        raise ValueError("P8_AUDIT_MANIFEST_PATIENT_COUNT_MISMATCH")
    return patient_count


def _alpha_at_best(history: pd.DataFrame, best_epoch: int) -> dict[str, Any]:
    matches = history.loc[history["epoch_index"].astype(int) == int(best_epoch)]
    if len(matches) != 1:
        raise ValueError("P8_AUDIT_BEST_ALPHA_EPOCH_MISMATCH")
    row = matches.iloc[0]
    logits: OrderedDict[str, list[float]] = OrderedDict()
    weights: OrderedDict[str, list[float]] = OrderedDict()
    gradients: OrderedDict[str, float] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        group_logits = list(map(float, json.loads(row[f"alpha_{group}_logits"])))
        group_weights = list(map(float, json.loads(row[f"alpha_{group}_weights"])))
        if len(group_logits) != 5 or len(group_weights) != 5:
            raise ValueError(f"P8_AUDIT_ALPHA_SHAPE_MISMATCH:{group}")
        if (
            not np.isfinite(group_logits).all()
            or not np.isfinite(group_weights).all()
            or np.any(np.asarray(group_weights) < 0.0)
            or not np.isclose(sum(group_weights), 1.0, atol=1e-7, rtol=0.0)
        ):
            raise ValueError(f"P8_AUDIT_ALPHA_INVALID:{group}")
        logits[group] = group_logits
        weights[group] = group_weights
        gradients[group] = float(row[f"alpha_{group}_gradient_l1"])
        if not np.isfinite(gradients[group]) or gradients[group] <= 0.0:
            raise ValueError(f"P8_AUDIT_ALPHA_GRADIENT_INVALID:{group}")
    return {
        "logits": logits,
        "weights": weights,
        "gradient_l1_at_best_epoch": gradients,
    }


def _fold_report(
    *,
    fold_index: int,
    verified: Mapping[str, Any],
    run_root: Path,
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    run = run_directory(fold_index, run_root)
    completion = _read_json(run / "training_complete.json")
    evaluation = _read_json(run / "test_evaluation.json")
    history = pd.read_csv(run / "history.csv")
    errors = contribution_reconstruction_errors(predictions)
    if max(errors.values()) > 1e-6:
        raise ValueError(f"P8_AUDIT_CONTRIBUTION_RECONSTRUCTION_FAILED:{fold_index}")
    if (
        evaluation.get("status") != "TEST_EVALUATED_ONCE"
        or int(evaluation.get("test_transaction_count", -1)) != 1
    ):
        raise ValueError(f"P8_AUDIT_TEST_NOT_EXACTLY_ONCE:{fold_index}")
    best_epoch = int(completion["best_epoch_index"])
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": completion["protocol_version"],
        "model": MODEL_NAME,
        "fold_index": fold_index,
        "epochs": int(verified["epochs"]),
        "train_samples_per_epoch": int(verified["train_samples_per_epoch"]),
        "test_samples": int(verified["test_samples"]),
        "best_epoch_index": best_epoch,
        "best_validation_total_loss": float(
            completion["best_validation_total_loss"]
        ),
        "test_evaluated_once": bool(verified["test_evaluated_once"]),
        "test_transaction_count": int(evaluation["test_transaction_count"]),
        "alpha_gradient_and_update_gate": verified[
            "alpha_gradient_and_update_gate"
        ],
        "learned_alpha_at_best_epoch": _alpha_at_best(history, best_epoch),
        "best_alpha_snapshot_sha256": completion[
            "best_alpha_snapshot_sha256"
        ],
        "final_alpha_snapshot_sha256": completion[
            "final_alpha_snapshot_sha256"
        ],
        "contribution_reconstruction": errors,
        "task_metrics": regression_metrics(predictions.to_dict("records")),
        "scientific_config_sha256": completion["scientific_config_sha256"],
        "execution_config_sha256": completion["execution_config_sha256"],
        "p8_execution_config_sha256": completion[
            "p8_execution_config_sha256"
        ],
        "split_sha256": completion["split_sha256"],
        "encoder_initialization_sha256": completion[
            "encoder_initialization_sha256"
        ],
        "combined_gam_initialization_sha256": completion[
            "combined_gam_initialization_sha256"
        ],
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "test_predictions_sha256": sha256_file(run / "test_predictions.parquet"),
        "test_evaluation_sha256": sha256_file(run / "test_evaluation.json"),
        "private_run_storage": _private_run_storage(run),
    }


def build_oof_audit(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    run_root: Path,
    audit_root: Path,
    oof_predictions_path: Path,
) -> dict[str, Any]:
    """Verify five existing folds and build private OOF plus tracked evidence."""
    verified_all = verify_all(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        p8_config_path=p8_execution_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        output_root=run_root,
    )
    if verified_all.get("status") != "PASS":
        raise ValueError("P8_AUDIT_FINAL_VERIFY_NOT_PASS")
    verified_by_fold = {
        int(item["fold_index"]): item for item in verified_all["folds"]
    }
    if set(verified_by_fold) != set(range(5)):
        raise ValueError("P8_AUDIT_FOLD_SET_MISMATCH")
    frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    audit_root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        verified = verify_fold(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p8_config_path=p8_execution_config_path,
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
            verified=verified,
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
    if (
        len(pooled) != EXPECTED_OOF_NODULES
        or pooled["nodule_uid"].astype(str).nunique() != EXPECTED_OOF_NODULES
    ):
        raise ValueError("P8_AUDIT_OOF_NODULE_SET_MISMATCH")
    patients = validate_manifest_patient_mapping(pooled, manifest_path)
    if pooled["patient_key"].astype(str).nunique() != patients:
        raise ValueError("P8_AUDIT_OOF_PATIENT_SET_MISMATCH")
    if int(pooled.groupby("patient_key")["fold_index"].nunique().max()) != 1:
        raise ValueError("P8_AUDIT_PATIENT_LEAKAGE")
    fold_counts = tuple(
        int(value)
        for value in pooled["fold_index"]
        .astype(int)
        .value_counts()
        .reindex(range(5), fill_value=0)
        .tolist()
    )
    if fold_counts != EXPECTED_FOLD_TEST_COUNTS:
        raise ValueError("P8_AUDIT_FOLD_TEST_COUNTS_MISMATCH")
    errors = contribution_reconstruction_errors(pooled)
    if max(errors.values()) > 1e-6:
        raise ValueError("P8_AUDIT_POOLED_RECONSTRUCTION_FAILED")
    _atomic_parquet(oof_predictions_path, pooled)
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
            and item["test_transaction_count"] == 1
            for item in fold_reports
        ),
        "pooled_oof_task_metrics": regression_metrics(pooled.to_dict("records")),
        "pooled_contribution_reconstruction": errors,
        "best_epoch_index_by_fold": [
            item["best_epoch_index"] for item in fold_reports
        ],
        "best_validation_total_loss_by_fold": [
            item["best_validation_total_loss"] for item in fold_reports
        ],
        "learned_alpha_at_best_epoch_by_fold": [
            item["learned_alpha_at_best_epoch"] for item in fold_reports
        ],
        "best_alpha_snapshot_sha256_by_fold": [
            item["best_alpha_snapshot_sha256"] for item in fold_reports
        ],
        "final_alpha_snapshot_sha256_by_fold": [
            item["final_alpha_snapshot_sha256"] for item in fold_reports
        ],
        "scientific_config_sha256": fold_reports[0]["scientific_config_sha256"],
        "execution_config_sha256": fold_reports[0]["execution_config_sha256"],
        "p8_execution_config_sha256": fold_reports[0][
            "p8_execution_config_sha256"
        ],
        "split_sha256_by_fold": [item["split_sha256"] for item in fold_reports],
        "encoder_initialization_sha256_by_fold": [
            item["encoder_initialization_sha256"] for item in fold_reports
        ],
        "combined_gam_initialization_sha256_by_fold": [
            item["combined_gam_initialization_sha256"] for item in fold_reports
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
    for path in [
        *(audit_root / f"fold_{fold}.json" for fold in range(5)),
        output_path,
    ]:
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
    build.add_argument("--p8-config", type=Path, default=P8_EXECUTION_CONFIG_DEFAULT)
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
    build.add_argument("--run-root", type=Path, default=Path("runs/baseline_v2/gam"))
    build.add_argument(
        "--audit-root", type=Path, default=Path("artifacts/baseline_v2/audit/p8")
    )
    build.add_argument(
        "--oof-output",
        type=Path,
        default=Path("runs/baseline_v2/gam/oof_predictions.parquet"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build_oof_audit(
        scientific_config_path=arguments.config,
        execution_config_path=arguments.execution_config,
        p8_execution_config_path=arguments.p8_config,
        manifest_path=arguments.manifest,
        roi_index_path=arguments.roi_index,
        run_root=arguments.run_root,
        audit_root=arguments.audit_root,
        oof_predictions_path=arguments.oof_output,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
