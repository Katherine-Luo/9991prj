from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lidc_baseline.p5_audit import METRIC_KEYS, build_fold_audit


def _provenance() -> dict[str, object]:
    return {
        "protocol_version": "Baseline-v2",
        "scientific_config_sha256": "scientific",
        "execution_config_sha256": "execution",
        "split_sha256": "split",
        "fold_index": 0,
        "model": "blackbox",
        "task_output": "unconstrained_linear_raw_score",
        "task_loss": "mean_squared_error_on_normalized_target",
        "fold_seed": 20260808,
        "encoder_initialization_sha256": "encoder",
        "encoder_artifact_file_sha256": "encoder-file",
        "head_initialization_seed": 123,
        "head_initialization_sha256": "head",
        "head_seed_derivation": "test-derivation",
    }


def _stage_a_provenance() -> dict[str, object]:
    formal = _provenance()
    for key in ("protocol_version", "model", "task_output", "task_loss"):
        formal.pop(key)
    return formal


def _write_private_fold(root: Path) -> None:
    run = root / "fold_0"
    run.mkdir(parents=True)
    provenance = _provenance()
    completion = {
        **provenance,
        "best_checkpoint_sha256": "best",
        "test_evaluation_sha256": "evaluation",
    }
    evaluation = {**provenance}
    runtime = {
        **provenance,
        "device_type": "cuda",
        "gpu_name": "NVIDIA L40S",
        "epochs_total": 80,
        "peak_reserved_bytes": 100,
        "fp32": True,
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "private_path": "/srv/scratch/private",
    }
    metrics = {key: 479 if key == "samples" else 0.25 for key in METRIC_KEYS}
    overfit = {
        **_stage_a_provenance(),
        "status": "PASS",
        "samples": 8,
        "steps": 40,
        "initial_mse": 0.5,
        "final_mse": 0.1,
    }
    preflight = {
        **_stage_a_provenance(),
        "status": "PASS",
        "batch_size": 16,
        "forward": True,
        "backward": True,
        "adam_step": True,
        "peak_reserved_bytes": 100,
        "gpu_total_bytes": 1000,
        "peak_reserved_fraction": 0.1,
        "maximum_allowed_fraction": 0.85,
        "device_type": "cuda",
        "gpu_name": "NVIDIA L40S",
    }
    for name, payload in {
        "training_complete.json": completion,
        "test_evaluation.json": evaluation,
        "metrics.json": metrics,
        "runtime.json": runtime,
        "overfit_sanity.json": overfit,
        "preflight.json": preflight,
    }.items():
        (run / name).write_text(json.dumps(payload), encoding="utf-8")


def test_fold_audit_is_aggregate_and_deidentified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    _write_private_fold(run_root)
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "nodule_uid": ["private-nodule"],
            "patient_id": ["private-patient"],
            "series_instance_uid": ["1.2.3.4"],
        }
    ).to_parquet(manifest, index=False)
    verified = {
        **_provenance(),
        "epochs": 80,
        "train_samples_per_epoch": 1882,
        "test_samples": 479,
        "best_epoch_index": 7,
        "best_validation_mse": 0.02,
        "test_evaluated_once": True,
    }
    monkeypatch.setattr("lidc_baseline.p5_audit.verify_fold", lambda **_kwargs: verified)
    output = tmp_path / "audit" / "fold_0.json"
    report = build_fold_audit(
        scientific_config_path=tmp_path / "config.yaml",
        execution_config_path=tmp_path / "execution.yaml",
        manifest_path=manifest,
        roi_index_path=tmp_path / "roi.parquet",
        run_root=run_root,
        fold_index=0,
        output_path=output,
        require_stage_a=True,
    )
    assert report["status"] == "PASS"
    assert report["epochs"] == 80
    assert report["test_samples"] == 479
    assert report["stage_a"]["preflight_batch_size"] == 16
    text = output.read_text(encoding="utf-8")
    for forbidden in ("private-nodule", "private-patient", "1.2.3.4", "/srv/scratch", str(tmp_path)):
        assert forbidden not in text


def test_fold_audit_rejects_stage_a_provenance_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    _write_private_fold(run_root)
    preflight_path = run_root / "fold_0" / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["execution_config_sha256"] = "wrong"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame({"nodule_uid": ["private"]}).to_parquet(manifest, index=False)
    verified = {
        **_provenance(),
        "epochs": 80,
        "train_samples_per_epoch": 1,
        "test_samples": 479,
        "best_epoch_index": 0,
        "best_validation_mse": 0.1,
        "test_evaluated_once": True,
    }
    monkeypatch.setattr("lidc_baseline.p5_audit.verify_fold", lambda **_kwargs: verified)
    with pytest.raises(ValueError, match="P5_AUDIT_PROVENANCE_MISMATCH:preflight"):
        build_fold_audit(
            scientific_config_path=tmp_path / "config.yaml",
            execution_config_path=tmp_path / "execution.yaml",
            manifest_path=manifest,
            roi_index_path=tmp_path / "roi.parquet",
            run_root=run_root,
            fold_index=0,
            output_path=tmp_path / "audit.json",
            require_stage_a=True,
        )
