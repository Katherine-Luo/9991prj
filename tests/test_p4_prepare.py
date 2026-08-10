from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.config import load_config
from lidc_baseline.p4_prepare import (
    CONSUMERS,
    build_encoder,
    build_split_payloads,
    encoder_state_sha256,
    load_shared_encoder_initialization,
    patient_key,
    read_split,
    sha256_file,
    train_only_rows,
    train_statistics,
    validate_encoder_artifact,
    validate_roi_files,
    write_encoder_artifact,
)


STRATA = ("mean_le_2", "mean_gt_2_lt_3", "mean_eq_3", "mean_gt_3_lt_4", "mean_ge_4")
RATINGS = (1.5, 2.5, 3.0, 3.5, 4.5)


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for index in range(200):
        stratum_index = index % 5
        row = {
            "nodule_uid": f"nodule-{index:03d}",
            "patient_id": f"patient-{index:03d}",
            "primary_regression_eligible": True,
            "malignancy_stratum": STRATA[stratum_index],
            "mean_malignancy": RATINGS[stratum_index],
            "malignancy_target_normalized": (RATINGS[stratum_index] - 1.0) / 4.0,
            "malignancy_valid_reader_count": 2,
        }
        for concept in ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"):
            row[f"{concept}_target"] = (index % 5) / 4.0
            row[f"{concept}_valid_reader_count"] = 2
        row.update({
            "internalStructure_vote_distribution": json.dumps([0.5, 0.5, 0.0, 0.0]),
            "internalStructure_valid_reader_count": 2,
            "calcification_vote_distribution": json.dumps([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
            "calcification_valid_reader_count": 2,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_roi_index(manifest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "nodule_uid": manifest["nodule_uid"],
        "status": "WRITTEN",
        "relative_roi_path": manifest["nodule_uid"].map(lambda uid: f"artifacts/baseline_v2/rois/{uid}.npz"),
        "roi_file_sha256": manifest["nodule_uid"].map(lambda uid: f"sha-{uid}"),
    })


def synthetic_config() -> dict:
    config = copy.deepcopy(load_config("configs/baseline_v2.yaml"))
    config["cohort"]["primary_regression"]["nodules"] = 200
    config["cohort"]["primary_regression"]["patients"] = 200
    return config


def test_patient_key_is_stable_domain_separated_and_private() -> None:
    first = patient_key("LIDC-IDRI-0001")
    assert first == patient_key("LIDC-IDRI-0001")
    assert first != patient_key("LIDC-IDRI-0002")
    assert "LIDC" not in first
    assert len(first) == 64


def test_split_is_deterministic_under_input_reordering() -> None:
    manifest = synthetic_manifest()
    roi_index = synthetic_roi_index(manifest)
    config = synthetic_config()
    first = build_split_payloads(manifest, roi_index, config, "manifest", "roi")
    second = build_split_payloads(
        manifest.sample(frac=1.0, random_state=99),
        roi_index.sample(frac=1.0, random_state=98),
        config,
        "manifest",
        "roi",
    )
    assert first == second
    assert len(first) == 5
    pooled_test = []
    pooled_patients = []
    for split in first:
        partitions = split["partitions"]
        nodule_sets = {name: set(partitions[name]["nodule_uids"]) for name in partitions}
        patient_sets = {name: set(partitions[name]["patient_keys"]) for name in partitions}
        assert not nodule_sets["train"] & nodule_sets["validation"]
        assert not nodule_sets["train"] & nodule_sets["test"]
        assert not nodule_sets["validation"] & nodule_sets["test"]
        assert not patient_sets["train"] & patient_sets["validation"]
        assert not patient_sets["train"] & patient_sets["test"]
        assert not patient_sets["validation"] & patient_sets["test"]
        assert partitions["validation"]["summary"]["extremes"]["low"] > 0
        assert partitions["validation"]["summary"]["extremes"]["high"] > 0
        assert partitions["test"]["summary"]["extremes"]["low"] > 0
        assert partitions["test"]["summary"]["extremes"]["high"] > 0
        pooled_test.extend(partitions["test"]["nodule_uids"])
        pooled_patients.extend(partitions["test"]["patient_keys"])
    assert len(pooled_test) == len(set(pooled_test)) == 200
    assert len(pooled_patients) == len(set(pooled_patients)) == 200


def _write_split(path: Path, split: dict) -> None:
    path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_train_only_statistics_reject_validation_or_unknown_uids(tmp_path: Path) -> None:
    manifest = synthetic_manifest()
    split = build_split_payloads(manifest, synthetic_roi_index(manifest), synthetic_config(), "manifest", "roi")[0]
    path = tmp_path / "fold.json"
    _write_split(path, split)
    train = split["partitions"]["train"]["nodule_uids"]
    validation_uid = split["partitions"]["validation"]["nodule_uids"][0]
    selected = train_only_rows(manifest, path, train[:4], expected_manifest_sha256="manifest")
    assert selected["nodule_uid"].tolist() == sorted(train[:4])
    with pytest.raises(ValueError, match="TRAIN_ONLY_STATISTICS_LEAKAGE"):
        train_only_rows(manifest, path, [validation_uid])
    with pytest.raises(ValueError, match="TRAIN_ONLY_STATISTICS_LEAKAGE"):
        train_only_rows(manifest, path, ["missing"])
    with pytest.raises(ValueError, match="TRAIN_ONLY_SPLIT_MANIFEST_MISMATCH"):
        train_only_rows(manifest, path, expected_manifest_sha256="forged")

    tampered = copy.deepcopy(split)
    tampered["partitions"]["train"]["nodule_uids"] = split["partitions"]["validation"]["nodule_uids"]
    tampered_path = tmp_path / "tampered.json"
    _write_split(tampered_path, tampered)
    with pytest.raises(ValueError, match="SPLIT_HASH_MISMATCH"):
        train_only_rows(manifest, tampered_path)


def test_train_statistics_match_only_registered_train_rows(tmp_path: Path) -> None:
    manifest = synthetic_manifest()
    split = build_split_payloads(manifest, synthetic_roi_index(manifest), synthetic_config(), "manifest", "roi")[0]
    path = tmp_path / "fold.json"
    _write_split(path, split)
    train_uids = split["partitions"]["train"]["nodule_uids"]
    result = train_statistics(manifest, path, expected_manifest_sha256="manifest")
    expected = manifest[manifest["nodule_uid"].isin(train_uids)]["malignancy_target_normalized"].mean()
    assert result["scope"] == "train_only"
    assert result["nodule_count"] == len(train_uids)
    assert result["targets"]["malignancy_target_normalized"]["mean"] == pytest.approx(expected)
    assert sum(result["targets"]["internalStructure"]["mean_vote_distribution"]) == pytest.approx(1.0)
    assert result["model_dependent_contribution_means"] == "deferred_until_model_inference"


def test_split_hash_detects_tampering(tmp_path: Path) -> None:
    manifest = synthetic_manifest()
    split = build_split_payloads(manifest, synthetic_roi_index(manifest), synthetic_config(), "manifest", "roi")[0]
    path = tmp_path / "fold_0.json"
    path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    assert read_split(path)["split_sha256"] == split["split_sha256"]
    split["fold_index"] = 99
    path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="SPLIT_HASH_MISMATCH"):
        read_split(path)


@pytest.mark.integration
def test_encoder_artifact_is_deterministic_and_shared(tmp_path: Path) -> None:
    config = synthetic_config()
    manifest = synthetic_manifest()
    split = build_split_payloads(manifest, synthetic_roi_index(manifest), config, "manifest", "roi")[0]
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    first = write_encoder_artifact(first_path, config, split)
    original_bytes = first_path.read_bytes()
    reused = write_encoder_artifact(first_path, config, split)
    second = write_encoder_artifact(second_path, config, split)
    assert reused["status"] == "REUSED"
    assert first_path.read_bytes() == original_bytes
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["file_sha256"] == second["file_sha256"]
    assert first["encoder_state_sha256"] == second["encoder_state_sha256"]
    assert first["serialization"] == "torch_legacy_canonical_storage_keys"
    hashes = [load_shared_encoder_initialization(build_encoder(), first_path, config, split) for _ in CONSUMERS]
    assert len(set(hashes)) == 1
    assert hashes[0] == validate_encoder_artifact(first_path, config, split)["metadata"]["encoder_state_sha256"]


@pytest.mark.integration
def test_different_fold_seed_changes_encoder_state(tmp_path: Path) -> None:
    config = synthetic_config()
    manifest = synthetic_manifest()
    splits = build_split_payloads(manifest, synthetic_roi_index(manifest), config, "manifest", "roi")
    first = write_encoder_artifact(tmp_path / "fold0.pt", config, splits[0])
    second = write_encoder_artifact(tmp_path / "fold1.pt", config, splits[1])
    assert first["encoder_state_sha256"] != second["encoder_state_sha256"]


@pytest.mark.integration
def test_encoder_artifact_corruption_and_provenance_mismatch_block(tmp_path: Path) -> None:
    config = synthetic_config()
    manifest = synthetic_manifest()
    splits = build_split_payloads(manifest, synthetic_roi_index(manifest), config, "manifest", "roi")
    path = tmp_path / "fold.pt"
    write_encoder_artifact(path, config, splits[0])
    with pytest.raises(ValueError, match="PROVENANCE_MISMATCH"):
        validate_encoder_artifact(path, config, splits[1])
    path.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        validate_encoder_artifact(path, config, splits[0])
    repaired = write_encoder_artifact(path, config, splits[0], overwrite=True)
    assert encoder_state_sha256(validate_encoder_artifact(path, config, splits[0])["encoder_state_dict"]) == repaired["encoder_state_sha256"]


def test_roi_directory_requires_exact_set_and_file_hashes(tmp_path: Path) -> None:
    roi_root = tmp_path / "rois"
    roi_root.mkdir()
    primary = pd.DataFrame({"nodule_uid": ["a", "b"]})
    for uid, content in (("a", b"first"), ("b", b"second")):
        (roi_root / f"{uid}.npz").write_bytes(content)
    index = pd.DataFrame({
        "nodule_uid": ["a", "b"],
        "status": ["WRITTEN", "REUSED"],
        "relative_roi_path": ["rois/a.npz", "rois/b.npz"],
        "roi_file_sha256": [sha256_file(roi_root / "a.npz"), sha256_file(roi_root / "b.npz")],
    })
    assert validate_roi_files(primary, index, roi_root)["roi_files"] == 2

    (roi_root / "extra.npz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="ROI_DIRECTORY_SET_MISMATCH"):
        validate_roi_files(primary, index, roi_root)
    (roi_root / "extra.npz").unlink()

    (roi_root / "a.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="ROI_FILE_HASH_MISMATCH"):
        validate_roi_files(primary, index, roi_root)
    (roi_root / "a.npz").unlink()
    with pytest.raises(ValueError, match="ROI_FILE_MISSING"):
        validate_roi_files(primary, index, roi_root)
