"""Build deidentified aggregate evidence for completed P5 folds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lidc_baseline.audit import write_json
from lidc_baseline.p3_roi import assert_deidentified_audit
from lidc_baseline.p5_blackbox import (
    EXECUTION_CONFIG_DEFAULT,
    run_directory,
    verify_fold,
)


SCHEMA_VERSION = 1
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    parser.add_argument("--execution-config", type=Path, default=EXECUTION_CONFIG_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/baseline_v2/manifests/nodules.parquet"))
    parser.add_argument("--roi-index", type=Path, default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/baseline_v2/blackbox"))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-stage-a", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
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
