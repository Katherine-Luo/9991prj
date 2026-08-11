from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.p6_audit import (
    build_oof_audit,
    contribution_reconstruction_errors,
    validate_manifest_patient_mapping,
)
from lidc_baseline.p6_standard_cbm import CONCEPT_GROUP_ORDER
from lidc_baseline.p4_prepare import patient_key


def _prediction_frame(fold: int, count: int = 2) -> pd.DataFrame:
    rows = []
    for index in range(count):
        target = float(index) / max(1, count - 1)
        score = 0.1 + 0.7 * target
        internal_target = [1.0, 0.0, 0.0, 0.0]
        internal_tie = False
        if index == count - 1:
            internal_target = [0.5, 0.5, 0.0, 0.0]
            internal_tie = True
        calcification_target = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        continuous_target = 0.2 + 0.6 * target
        targets = {
            "subtlety": [continuous_target],
            "internalStructure": internal_target,
            "calcification": calcification_target,
            "sphericity": [continuous_target],
            "margin": [continuous_target],
            "lobulation": [continuous_target],
            "spiculation": [continuous_target],
            "texture": [continuous_target],
        }
        row = {
            "nodule_uid": f"nodule-{fold}-{index}",
            "patient_key": patient_key(f"raw-patient-{fold}-{index}"),
            "fold_index": fold,
            "target_normalized": target,
            "target_1_to_5": 1.0 + 4.0 * target,
            "malignancy_raw_score": score,
            "malignancy_score_normalized": score,
            "malignancy_score_1_to_5": 1.0 + 4.0 * score,
            "raw_bias": 0.05,
            "rating_scale_bias": 1.2,
            "concept_targets": json.dumps(targets, sort_keys=True),
            "internalStructure_modal_tie": internal_tie,
            "calcification_modal_tie": False,
            "task_best_checkpoint_sha256": "a" * 64,
        }
        contribution = (score - 0.05) / 8.0
        for group in CONCEPT_GROUP_ORDER:
            row[f"{group}_raw_contribution"] = contribution
            row[f"{group}_rating_point_contribution"] = 4.0 * contribution
        for group in (
            "subtlety",
            "sphericity",
            "margin",
            "lobulation",
            "spiculation",
            "texture",
        ):
            row[f"{group}_activated_prediction"] = json.dumps(
                [continuous_target + (0.05 if index == 0 else -0.05)]
            )
        row["internalStructure_activated_prediction"] = json.dumps(
            [0.7, 0.2, 0.05, 0.05]
        )
        row["calcification_activated_prediction"] = json.dumps(
            [0.05, 0.7, 0.05, 0.05, 0.1, 0.05]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_p6_contribution_reconstruction_reports_maximum_errors() -> None:
    frame = _prediction_frame(0, count=3)
    errors = contribution_reconstruction_errors(frame)
    assert max(errors.values()) < 1e-12

    tampered = frame.copy()
    tampered.loc[1, "margin_raw_contribution"] += 0.01
    errors = contribution_reconstruction_errors(tampered)
    assert errors["normalized_max_absolute_error"] == pytest.approx(0.01)
    assert errors["contribution_scale_max_absolute_error"] == pytest.approx(0.04)


def _write_fold(root: Path, fold: int, frame: pd.DataFrame) -> None:
    run = root / f"fold_{fold}"
    (run / "concept_stage").mkdir(parents=True)
    (run / "task_stage").mkdir(parents=True)
    concept = {
        "best_epoch_index": fold,
        "best_validation_objective": 0.1 + fold * 0.01,
    }
    task = {
        "best_epoch_index": fold + 1,
        "best_validation_objective": 0.2 + fold * 0.01,
    }
    sequential = {
        "protocol_version": "Baseline-v2",
        "scientific_config_sha256": "1" * 64,
        "execution_config_sha256": "2" * 64,
        "p6_execution_config_sha256": "3" * 64,
        "split_sha256": str(fold) * 64,
        "encoder_initialization_sha256": str(fold + 1) * 64,
        "concept_best_checkpoint_sha256": "4" * 64,
        "frozen_predictor_semantic_sha256_before_task": "5" * 64,
        "frozen_predictor_semantic_sha256_after_task": "5" * 64,
        "frozen_batchnorm_state_sha256_before_task": "6" * 64,
        "frozen_batchnorm_state_sha256_after_task": "6" * 64,
    }
    evaluation = {
        "status": "TEST_EVALUATED_EXACTLY_ONCE",
        "test_inference_transactions": 1,
    }
    for path, payload in (
        (run / "concept_stage" / "training_complete.json", concept),
        (run / "task_stage" / "training_complete.json", task),
        (run / "sequential_training_complete.json", sequential),
        (run / "test_evaluation.json", evaluation),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    frame.to_parquet(run / "test_predictions.parquet", index=False)


def test_p6_build_oof_audit_materializes_private_and_deidentified_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lidc_baseline.p6_audit as audit

    run_root = tmp_path / "runs"
    for fold in range(5):
        _write_fold(run_root, fold, _prediction_frame(fold))
    manifest = pd.DataFrame(
        {
            "nodule_uid": [f"nodule-{fold}-{index}" for fold in range(5) for index in range(2)],
            "patient_id": [f"raw-patient-{fold}-{index}" for fold in range(5) for index in range(2)],
            "primary_regression_eligible": [True] * 10,
        }
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    verified_folds = [
        {
            "fold_index": fold,
            "concept_epochs": 80,
            "task_epochs": 80,
            "train_samples_per_epoch": 10,
            "test_evaluated_once": True,
            "test_samples": 2,
        }
        for fold in range(5)
    ]
    monkeypatch.setattr(audit, "EXPECTED_FOLD_TEST_COUNTS", (2, 2, 2, 2, 2))
    monkeypatch.setattr(audit, "EXPECTED_OOF_NODULES", 10)
    monkeypatch.setattr(audit, "EXPECTED_OOF_PATIENTS", 10)
    monkeypatch.setattr(
        audit, "verify_all", lambda **_kwargs: {"status": "PASS", "folds": verified_folds}
    )
    monkeypatch.setattr(audit, "verify_fold", lambda **_kwargs: {"status": "PASS"})

    audit_root = tmp_path / "audit"
    oof_path = tmp_path / "private" / "oof.parquet"
    report = build_oof_audit(
        scientific_config_path=tmp_path / "scientific.yaml",
        execution_config_path=tmp_path / "execution.yaml",
        p6_execution_config_path=tmp_path / "p6.yaml",
        manifest_path=manifest_path,
        roi_index_path=tmp_path / "roi.parquet",
        run_root=run_root,
        audit_root=audit_root,
        oof_predictions_path=oof_path,
    )
    assert report["status"] == "PASS"
    assert report["oof_nodules"] == 10
    assert report["patient_leakage"] == 0
    assert report["test_evaluated_once_all_folds"] is True
    assert report["concept_predictor_semantic_hash_unchanged_all_folds"] is True
    assert oof_path.is_file()
    assert len(pd.read_parquet(oof_path)) == 10
    tracked = "".join(path.read_text(encoding="utf-8") for path in audit_root.glob("*.json"))
    assert "nodule-0-0" not in tracked
    assert "raw-patient-0-0" not in tracked


def test_p6_manifest_mapping_rejects_swapped_patient_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lidc_baseline.p6_audit as audit

    frame = _prediction_frame(0, count=2)
    manifest = pd.DataFrame(
        {
            "nodule_uid": ["nodule-0-0", "nodule-0-1"],
            "patient_id": ["raw-patient-0-0", "raw-patient-0-1"],
            "primary_regression_eligible": [True, True],
        }
    )
    path = tmp_path / "manifest.parquet"
    manifest.to_parquet(path, index=False)
    monkeypatch.setattr(audit, "EXPECTED_OOF_NODULES", 2)
    monkeypatch.setattr(audit, "EXPECTED_OOF_PATIENTS", 2)
    frame.loc[0, "patient_key"], frame.loc[1, "patient_key"] = (
        frame.loc[1, "patient_key"],
        frame.loc[0, "patient_key"],
    )
    with pytest.raises(ValueError, match="PATIENT_KEY_MAPPING_MISMATCH"):
        validate_manifest_patient_mapping(frame, path)


def test_p6_oof_audit_rejects_patient_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lidc_baseline.p6_audit as audit

    run_root = tmp_path / "runs"
    for fold in range(5):
        frame = _prediction_frame(fold)
        if fold in (0, 1):
            frame.loc[0, "patient_key"] = patient_key("shared-patient")
        _write_fold(run_root, fold, frame)
    manifest_path = tmp_path / "manifest.parquet"
    rows = [
        {
            "nodule_uid": f"nodule-{fold}-{index}",
            "patient_id": (
                "shared-patient" if index == 0 and fold in (0, 1) else f"raw-patient-{fold}-{index}"
            ),
            "primary_regression_eligible": True,
        }
        for fold in range(5)
        for index in range(2)
    ]
    pd.DataFrame(rows).to_parquet(manifest_path, index=False)
    verified = [
        {
            "fold_index": fold,
            "concept_epochs": 80,
            "task_epochs": 80,
            "train_samples_per_epoch": 10,
            "test_evaluated_once": True,
            "test_samples": 2,
        }
        for fold in range(5)
    ]
    monkeypatch.setattr(audit, "EXPECTED_FOLD_TEST_COUNTS", (2, 2, 2, 2, 2))
    monkeypatch.setattr(audit, "EXPECTED_OOF_NODULES", 10)
    monkeypatch.setattr(audit, "EXPECTED_OOF_PATIENTS", 9)
    monkeypatch.setattr(audit, "verify_all", lambda **_kwargs: {"status": "PASS", "folds": verified})
    monkeypatch.setattr(audit, "verify_fold", lambda **_kwargs: {"status": "PASS"})
    with pytest.raises(ValueError, match="PATIENT_LEAKAGE"):
        build_oof_audit(
            scientific_config_path=tmp_path / "scientific.yaml",
            execution_config_path=tmp_path / "execution.yaml",
            p6_execution_config_path=tmp_path / "p6.yaml",
            manifest_path=manifest_path,
            roi_index_path=tmp_path / "roi.parquet",
            run_root=run_root,
            audit_root=tmp_path / "audit",
            oof_predictions_path=tmp_path / "oof.parquet",
        )
