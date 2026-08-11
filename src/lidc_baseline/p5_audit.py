"""Build deidentified aggregate evidence for completed P5 folds."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.audit import write_json
from lidc_baseline.p3_roi import assert_deidentified_audit
from lidc_baseline.p5_blackbox import (
    EXECUTION_CONFIG_DEFAULT,
    regression_metrics,
    run_directory,
    verify_all,
    verify_fold,
)
from lidc_baseline.p4_prepare import sha256_file


SCHEMA_VERSION = 1
EXPECTED_FOLD_TEST_COUNTS = (479, 502, 539, 549, 564)
METRIC_KEYS = (
    "samples",
    "normalized_mae",
    "normalized_rmse",
    "original_scale_mae",
    "original_scale_rmse",
    "pearson",
    "spearman",
    "prediction_min",
    "prediction_max",
    "prediction_below_0_rate",
    "prediction_above_1_rate",
    "prediction_below_1_on_original_scale_rate",
    "prediction_above_5_on_original_scale_rate",
)
PROVENANCE_KEYS = (
    "protocol_version",
    "scientific_config_sha256",
    "execution_config_sha256",
    "execution_profile_id",
    "formal_gpu_model",
    "torch_use_deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "split_sha256",
    "fold_index",
    "model",
    "task_output",
    "task_loss",
    "fold_seed",
    "encoder_initialization_sha256",
    "encoder_artifact_file_sha256",
    "head_initialization_seed",
    "head_initialization_sha256",
    "head_seed_derivation",
)
STAGE_A_PROVENANCE_KEYS = (
    "scientific_config_sha256",
    "execution_config_sha256",
    "execution_profile_id",
    "formal_gpu_model",
    "torch_use_deterministic_algorithms",
    "deterministic_algorithms_warn_only",
    "split_sha256",
    "fold_index",
    "fold_seed",
    "encoder_initialization_sha256",
    "encoder_artifact_file_sha256",
    "head_initialization_seed",
    "head_initialization_sha256",
    "head_seed_derivation",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P5_AUDIT_EXPECTED_OBJECT:{path.name}")
    return payload


def _require_equal_provenance(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    source: str,
    keys: Sequence[str] = PROVENANCE_KEYS,
) -> None:
    for key in keys:
        if observed.get(key) != expected.get(key):
            raise ValueError(f"P5_AUDIT_PROVENANCE_MISMATCH:{source}:{key}")


def _runtime_summary(runtime: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "device_type",
        "gpu_name",
        "python_version",
        "torch_version",
        "monai_version",
        "numpy_version",
        "cuda_runtime",
        "fp32",
        "amp_enabled",
        "bfloat16_enabled",
        "cuda_matmul_tf32_enabled",
        "cudnn_tf32_enabled",
        "torch_use_deterministic_algorithms",
        "deterministic_algorithms_warn_only",
        "epochs_total",
        "sum_epoch_seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    return {key: runtime.get(key) for key in allowed if key in runtime}


def _private_run_storage(run: Path) -> dict[str, int]:
    files = [path for path in run.rglob("*") if path.is_file() and not path.name.startswith(".p5_")]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        temporary = Path(temporary_name)
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


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


def build_fold_audit(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    run_root: Path,
    fold_index: int,
    output_path: Path,
    require_stage_a: bool,
) -> dict[str, Any]:
    """Verify a private fold run and write only aggregate tracked evidence."""
    if fold_index not in range(5):
        raise ValueError("P5_AUDIT_INVALID_FOLD")
    verified = verify_fold(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        fold_index=fold_index,
        output_root=run_root,
        require_test=True,
    )
    run = run_directory(fold_index, run_root)
    completion = _read_json(run / "training_complete.json")
    evaluation = _read_json(run / "test_evaluation.json")
    metrics = _read_json(run / "metrics.json")
    runtime = _read_json(run / "runtime.json")
    _require_equal_provenance(verified, completion, "training_complete")
    _require_equal_provenance(verified, evaluation, "test_evaluation")
    _require_equal_provenance(verified, runtime, "runtime")
    if set(metrics) != set(METRIC_KEYS):
        raise ValueError("P5_AUDIT_METRIC_SCHEMA_MISMATCH")
    if int(metrics["samples"]) != int(verified["test_samples"]):
        raise ValueError("P5_AUDIT_METRIC_SAMPLE_MISMATCH")
    expected_gpu = str(runtime.get("formal_gpu_model", ""))
    if (
        runtime.get("device_type") != "cuda"
        or not expected_gpu
        or expected_gpu not in str(runtime.get("gpu_name", ""))
    ):
        raise ValueError("P5_AUDIT_FORMAL_RUNTIME_NOT_PROFILE_CUDA")
    expected_precision = {
        "fp32": True,
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
    }
    if any(runtime.get(key) is not value for key, value in expected_precision.items()):
        raise ValueError("P5_AUDIT_FORMAL_PRECISION_MISMATCH")

    stage_a: dict[str, Any] | None = None
    if require_stage_a:
        if fold_index != 0:
            raise ValueError("P5_AUDIT_STAGE_A_ONLY_FOLD_ZERO")
        overfit = _read_json(run / "overfit_sanity.json")
        preflight = _read_json(run / "preflight.json")
        _require_equal_provenance(
            verified,
            overfit,
            "overfit_sanity",
            STAGE_A_PROVENANCE_KEYS,
        )
        _require_equal_provenance(
            verified,
            preflight,
            "preflight",
            STAGE_A_PROVENANCE_KEYS,
        )
        if overfit.get("status") != "PASS" or preflight.get("status") != "PASS":
            raise ValueError("P5_AUDIT_STAGE_A_NOT_PASS")
        if float(overfit["final_mse"]) >= float(overfit["initial_mse"]):
            raise ValueError("P5_AUDIT_OVERFIT_NOT_IMPROVED")
        if int(preflight.get("batch_size", -1)) != 16:
            raise ValueError("P5_AUDIT_PREFLIGHT_BATCH_MISMATCH")
        if not all(preflight.get(key) is True for key in ("forward", "backward", "adam_step")):
            raise ValueError("P5_AUDIT_PREFLIGHT_OPERATION_MISMATCH")
        stage_a = {
            "overfit_status": overfit["status"],
            "overfit_samples": int(overfit["samples"]),
            "overfit_steps": int(overfit["steps"]),
            "overfit_initial_mse": float(overfit["initial_mse"]),
            "overfit_final_mse": float(overfit["final_mse"]),
            "preflight_status": preflight["status"],
            "preflight_batch_size": int(preflight["batch_size"]),
            "preflight_forward": True,
            "preflight_backward": True,
            "preflight_adam_step": True,
            "preflight_peak_reserved_bytes": int(preflight["peak_reserved_bytes"]),
            "preflight_gpu_total_bytes": int(preflight["gpu_total_bytes"]),
            "preflight_peak_reserved_fraction": float(preflight["peak_reserved_fraction"]),
            "preflight_maximum_allowed_fraction": float(preflight["maximum_allowed_fraction"]),
            "gpu_name": preflight.get("gpu_name"),
        }

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": verified["protocol_version"],
        "model": "blackbox",
        "fold_index": fold_index,
        "epochs": int(verified["epochs"]),
        "train_samples_per_epoch": int(verified["train_samples_per_epoch"]),
        "test_samples": int(verified["test_samples"]),
        "best_epoch_index": int(verified["best_epoch_index"]),
        "best_validation_mse": float(verified["best_validation_mse"]),
        "test_evaluated_once": bool(verified["test_evaluated_once"]),
        "metrics": {key: metrics[key] for key in METRIC_KEYS},
        "runtime": _runtime_summary(runtime),
        "private_run_storage": _private_run_storage(run),
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "test_evaluation_sha256": completion["test_evaluation_sha256"],
        **{key: verified[key] for key in PROVENANCE_KEYS},
    }
    if stage_a is not None:
        report["stage_a"] = stage_a
    write_json(output_path, report)
    assert_deidentified_audit(output_path, _forbidden_source_values(manifest_path))
    return report


def build_oof_audit(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    run_root: Path,
    audit_root: Path,
    oof_predictions_path: Path,
) -> dict[str, Any]:
    """Verify all formal folds and materialize private pooled OOF predictions."""
    verified = verify_all(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        output_root=run_root,
    )
    if verified.get("status") != "PASS":
        raise ValueError("P5_OOF_VERIFY_NOT_PASS")
    if tuple(verified.get("fold_test_counts", ())) != EXPECTED_FOLD_TEST_COUNTS:
        raise ValueError("P5_OOF_FOLD_COUNTS_MISMATCH")

    fold_reports = [
        build_fold_audit(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            run_root=run_root,
            fold_index=fold,
            output_path=audit_root / f"fold_{fold}.json",
            require_stage_a=fold == 0,
        )
        for fold in range(5)
    ]
    frames = [
        pd.read_parquet(run_directory(fold, run_root) / "test_predictions.parquet")
        for fold in range(5)
    ]
    pooled = pd.concat(frames, ignore_index=True).sort_values("nodule_uid", kind="stable").reset_index(drop=True)
    if len(pooled) != 2633 or pooled["nodule_uid"].astype(str).nunique() != 2633:
        raise ValueError("P5_OOF_NODULE_SET_MISMATCH")
    if pooled["patient_key"].astype(str).nunique() != 868:
        raise ValueError("P5_OOF_PATIENT_SET_MISMATCH")
    if int(pooled.groupby("patient_key")["fold_index"].nunique().max()) != 1:
        raise ValueError("P5_OOF_PATIENT_LEAKAGE")
    observed_counts = tuple(
        int(value)
        for value in pooled["fold_index"].astype(int).value_counts().reindex(range(5), fill_value=0).tolist()
    )
    if observed_counts != EXPECTED_FOLD_TEST_COUNTS:
        raise ValueError("P5_OOF_FOLD_COUNTS_MISMATCH")
    if pooled["model"].astype(str).nunique() != 1 or str(pooled["model"].iloc[0]) != "blackbox":
        raise ValueError("P5_OOF_MODEL_MISMATCH")
    if any(token in column.lower() for column in pooled.columns for token in ("probability", "logit", "concept", "mask")):
        raise ValueError("P5_OOF_FORBIDDEN_COLUMN")

    metrics = regression_metrics(pooled.to_dict("records"))
    _atomic_parquet(oof_predictions_path, pooled)
    fold_metric_values = {
        key: np.asarray([float(report["metrics"][key]) for report in fold_reports], dtype=np.float64)
        for key in METRIC_KEYS
        if key != "samples"
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "protocol_version": fold_reports[0]["protocol_version"],
        "model": "blackbox",
        "folds": 5,
        "oof_nodules": 2633,
        "oof_patients": 868,
        "fold_test_counts": list(EXPECTED_FOLD_TEST_COUNTS),
        "test_evaluated_once_all_folds": True,
        "pooled_oof_metrics": {key: metrics[key] for key in METRIC_KEYS},
        "fold_metric_mean": {key: float(values.mean()) for key, values in fold_metric_values.items()},
        "fold_metric_sample_standard_deviation": {
            key: float(values.std(ddof=1)) for key, values in fold_metric_values.items()
        },
        "scientific_config_sha256": fold_reports[0]["scientific_config_sha256"],
        "execution_config_sha256": fold_reports[0]["execution_config_sha256"],
        "execution_profile_id": fold_reports[0]["execution_profile_id"],
        "formal_gpu_model": fold_reports[0]["formal_gpu_model"],
        "split_sha256_by_fold": [report["split_sha256"] for report in fold_reports],
        "encoder_initialization_sha256_by_fold": [
            report["encoder_initialization_sha256"] for report in fold_reports
        ],
        "head_initialization_seed_by_fold": [
            int(report["head_initialization_seed"]) for report in fold_reports
        ],
        "head_initialization_sha256_by_fold": [
            report["head_initialization_sha256"] for report in fold_reports
        ],
        "best_epoch_index_by_fold": [int(report["best_epoch_index"]) for report in fold_reports],
        "best_validation_mse_by_fold": [
            float(report["best_validation_mse"]) for report in fold_reports
        ],
        "oof_predictions_sha256": sha256_file(oof_predictions_path),
        "private_run_storage": {
            "file_count": sum(int(report["private_run_storage"]["file_count"]) for report in fold_reports),
            "total_bytes": sum(int(report["private_run_storage"]["total_bytes"]) for report in fold_reports),
        },
    }
    audit_root.mkdir(parents=True, exist_ok=True)
    output_path = audit_root / "summary.json"
    write_json(output_path, report)
    assert_deidentified_audit(output_path, _forbidden_source_values(manifest_path))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    parser.add_argument("--execution-config", type=Path, default=EXECUTION_CONFIG_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/baseline_v2/manifests/nodules.parquet"))
    parser.add_argument("--roi-index", type=Path, default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/baseline_v2/blackbox"))
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--fold", type=int, choices=range(5))
    scope.add_argument("--scope", choices=("all",))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-root", type=Path, default=Path("artifacts/baseline_v2/audit/p5"))
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        default=Path("runs/baseline_v2/blackbox/oof_predictions.parquet"),
    )
    parser.add_argument("--require-stage-a", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.scope == "all":
        report = build_oof_audit(
            scientific_config_path=arguments.config,
            execution_config_path=arguments.execution_config,
            manifest_path=arguments.manifest,
            roi_index_path=arguments.roi_index,
            run_root=arguments.run_root,
            audit_root=arguments.audit_root,
            oof_predictions_path=arguments.oof_predictions,
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    output = arguments.output or Path(f"artifacts/baseline_v2/audit/p5/fold_{arguments.fold}.json")
    report = build_fold_audit(
        scientific_config_path=arguments.config,
        execution_config_path=arguments.execution_config,
        manifest_path=arguments.manifest,
        roi_index_path=arguments.roi_index,
        run_root=arguments.run_root,
        fold_index=arguments.fold,
        output_path=output,
        require_stage_a=arguments.require_stage_a,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
