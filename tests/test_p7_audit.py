from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lidc_baseline.p4_prepare import patient_key
from lidc_baseline.p7_audit import (
    build_oof_audit,
    contribution_reconstruction_errors,
    validate_manifest_patient_mapping,
)
from lidc_baseline.p7_mixed_cem import CONCEPT_GROUP_ORDER


def _predictions(fold: int, count: int = 2) -> pd.DataFrame:
    rows = []
    for index in range(count):
        target = float(index) / max(1, count - 1)
        score = 0.1 + 0.7 * target
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
        }
        contribution = (score - 0.05) / 8.0
        for group in CONCEPT_GROUP_ORDER:
            row[f"{group}_raw_contribution"] = contribution
            row[f"{group}_rating_contribution"] = 4.0 * contribution
        rows.append(row)
    return pd.DataFrame(rows)


def _write_fold(root: Path, fold: int, frame: pd.DataFrame) -> None:
    run = root / f"fold_{fold}"
    run.mkdir(parents=True)
    completion = {
        "protocol_version": "Baseline-v2",
        "method_label": "A project-specific mixed-type extension of the original CEM.",
        "best_epoch_index": fold,
        "best_validation_total_loss": 0.1 + fold * 0.01,
        "scientific_config_sha256": "1" * 64,
        "execution_config_sha256": "2" * 64,
        "p7_execution_config_sha256": "3" * 64,
        "split_sha256": str(fold) * 64,
        "encoder_initialization_sha256": str(fold + 1) * 64,
        "combined_cem_initialization_sha256": "4" * 64,
        "best_checkpoint_sha256": "5" * 64,
    }
    evaluation = {"status": "TEST_EVALUATED_ONCE"}
    (run / "training_complete.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (run / "test_evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    frame.to_parquet(run / "test_predictions.parquet", index=False)


def test_p7_contribution_reconstruction_reports_tampering() -> None:
    frame = _predictions(0, count=3)
    assert max(contribution_reconstruction_errors(frame).values()) < 1e-12
    tampered = frame.copy()
    tampered.loc[1, "margin_raw_contribution"] += 0.01
    errors = contribution_reconstruction_errors(tampered)
    assert errors["normalized_max_absolute_error"] == pytest.approx(0.01)
    assert errors["contribution_scale_max_absolute_error"] == pytest.approx(0.04)


def test_p7_build_oof_materializes_private_and_deidentified_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lidc_baseline.p7_audit as audit

    run_root = tmp_path / "runs"
    for fold in range(5):
        _write_fold(run_root, fold, _predictions(fold))
    manifest = pd.DataFrame(
        {
            "nodule_uid": [
                f"nodule-{fold}-{index}" for fold in range(5) for index in range(2)
            ],
            "patient_id": [
                f"raw-patient-{fold}-{index}"
                for fold in range(5)
                for index in range(2)
            ],
            "primary_regression_eligible": [True] * 10,
        }
    )
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    verified = [
        {
            "fold_index": fold,
            "epochs": 80,
            "train_samples_per_epoch": 10,
            "test_evaluated_once": True,
            "test_samples": 2,
            "intervention_rates": {"overall_decision_rate": 0.25},
        }
        for fold in range(5)
    ]
    monkeypatch.setattr(audit, "EXPECTED_FOLD_TEST_COUNTS", (2, 2, 2, 2, 2))
    monkeypatch.setattr(audit, "EXPECTED_OOF_NODULES", 10)
    monkeypatch.setattr(audit, "EXPECTED_OOF_PATIENTS", 10)
    monkeypatch.setattr(
        audit, "verify_all", lambda **_kwargs: {"status": "PASS", "folds": verified}
    )
    monkeypatch.setattr(
        audit,
        "verify_fold",
        lambda fold_index, **_kwargs: verified[fold_index],
    )
    audit_root = tmp_path / "audit"
    oof_path = tmp_path / "private" / "oof.parquet"
    report = build_oof_audit(
        scientific_config_path=tmp_path / "scientific.yaml",
        execution_config_path=tmp_path / "execution.yaml",
        p7_execution_config_path=tmp_path / "p7.yaml",
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
    assert len(pd.read_parquet(oof_path)) == 10
    tracked = "".join(
        path.read_text(encoding="utf-8") for path in audit_root.glob("*.json")
    )
    assert "nodule-0-0" not in tracked
    assert "raw-patient-0-0" not in tracked


def test_p7_manifest_mapping_rejects_swapped_patient_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lidc_baseline.p7_audit as audit

    frame = _predictions(0, count=2)
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
