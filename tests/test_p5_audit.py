from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lidc_baseline.p5_audit import (
    EXPECTED_FOLD_TEST_COUNTS,
    METRIC_KEYS,
    build_fold_audit,
    build_oof_audit,
)


def _provenance() -> dict[str, object]:
    return {
        "protocol_version": "Baseline-v2",
        "scientific_config_sha256": "scientific",
        "execution_config_sha256": "execution",
        "execution_profile_id": "baseline-v2-formal-h200-warn-only",
        "formal_gpu_model": "H200",
        "torch_use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
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
        "gpu_name": "NVIDIA H200",
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
        "gpu_name": "NVIDIA H200",
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


def test_oof_audit_materializes_private_predictions_and_deidentified_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    patient_counts = (171, 180, 169, 174, 174)
    private_nodules: list[str] = []
    private_patients: list[str] = []
    for fold, (nodule_count, patient_count) in enumerate(zip(EXPECTED_FOLD_TEST_COUNTS, patient_counts, strict=True)):
        fold_dir = run_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        rows = []
        for index in range(nodule_count):
            nodule = f"private-nodule-{fold}-{index}"
            patient = f"private-patient-{fold}-{index % patient_count}"
            target = float((index % 5) / 4)
            score = target + 0.01
            private_nodules.append(nodule)
            private_patients.append(patient)
            rows.append(
                {
                    "nodule_uid": nodule,
                    "patient_key": patient,
                    "target_normalized": target,
                    "target_1_to_5": 1.0 + 4.0 * target,
                    "malignancy_raw_score": score,
                    "malignancy_score_normalized": score,
                    "malignancy_score_1_to_5": 1.0 + 4.0 * score,
                    "extreme_binary_eligible": target in (0.0, 1.0),
                    "extreme_binary_label": 0.0 if target == 0.0 else 1.0 if target == 1.0 else None,
                    "fold_index": fold,
                    "model": "blackbox",
                }
            )
        pd.DataFrame(rows).to_parquet(fold_dir / "test_predictions.parquet", index=False)
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(
        {
            "nodule_uid": private_nodules,
            "patient_id": private_patients,
        }
    ).to_parquet(manifest, index=False)
    verified = {
        "status": "PASS",
        "oof_nodules": 2633,
        "oof_patients": 868,
        "fold_test_counts": list(EXPECTED_FOLD_TEST_COUNTS),
        "folds": [],
    }
    monkeypatch.setattr("lidc_baseline.p5_audit.verify_all", lambda **_kwargs: verified)

    def fake_fold_audit(**kwargs: object) -> dict[str, object]:
        fold = int(kwargs["fold_index"])
        report = {
            **_provenance(),
            "fold_index": fold,
            "split_sha256": f"split-{fold}",
            "encoder_initialization_sha256": f"encoder-{fold}",
            "head_initialization_seed": 100 + fold,
            "head_initialization_sha256": f"head-{fold}",
            "best_epoch_index": fold,
            "best_validation_mse": 0.02 + fold / 1000,
            "metrics": {key: EXPECTED_FOLD_TEST_COUNTS[fold] if key == "samples" else 0.25 for key in METRIC_KEYS},
            "private_run_storage": {"file_count": 10, "total_bytes": 1000},
        }
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr("lidc_baseline.p5_audit.build_fold_audit", fake_fold_audit)
    audit_root = tmp_path / "audit"
    oof_path = run_root / "oof_predictions.parquet"
    report = build_oof_audit(
        scientific_config_path=tmp_path / "config.yaml",
        execution_config_path=tmp_path / "execution.yaml",
        manifest_path=manifest,
        roi_index_path=tmp_path / "roi.parquet",
        run_root=run_root,
        audit_root=audit_root,
        oof_predictions_path=oof_path,
    )
    assert report["status"] == "PASS"
    assert report["oof_nodules"] == 2633
    assert report["oof_patients"] == 868
    assert report["fold_test_counts"] == list(EXPECTED_FOLD_TEST_COUNTS)
    assert report["pooled_oof_metrics"]["original_scale_mae"] == pytest.approx(0.04)
    assert pd.read_parquet(oof_path)["nodule_uid"].is_monotonic_increasing
    text = (audit_root / "summary.json").read_text(encoding="utf-8")
    for forbidden in (private_nodules[0], private_patients[0], str(tmp_path)):
        assert forbidden not in text
