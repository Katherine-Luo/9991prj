from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lidc_baseline import p9_spatial as spatial


def test_all_28_target_paths_and_categorical_tie_break() -> None:
    assert len(spatial.all_stage_a_target_paths()) == 28
    assert len(spatial.target_specs("blackbox")) == 1
    assert all(len(spatial.target_specs(model)) == 9 for model in spatial.MODEL_ORDER[1:])
    logits = torch.tensor([[1.0, 1.0, 0.0]])
    assert spatial.predicted_class_index(logits).item() == 0


def test_gradcam_formula_is_spatial_mean_weighted_relu_and_trilinear() -> None:
    activation = torch.tensor(
        [[[[[1.0, -1.0], [2.0, -2.0]], [[3.0, -3.0], [4.0, -4.0]]]]]
    )
    gradient = torch.full_like(activation, 2.0)
    result = spatial.gradcam_from_activation_and_gradient(
        activation, gradient, output_shape=(2, 2, 2)
    )
    expected = torch.relu(2.0 * activation[:, 0])
    assert result.dtype == torch.float32
    assert torch.equal(result, expected)
    with pytest.raises(ValueError, match="INTERFACE_MISMATCH"):
        spatial.gradcam_from_activation_and_gradient(
            activation.half(), gradient.half(), output_shape=(2, 2, 2)
        )
    with pytest.raises(ValueError, match="INTERFACE_MISMATCH"):
        spatial.gradcam_from_activation_and_gradient(
            activation.double(), gradient.double(), output_shape=(2, 2, 2)
        )


def test_map_status_requires_raw_fp32_post_relu_and_marks_zero_undefined() -> None:
    zeros = np.zeros(spatial.MAP_SHAPE, dtype=np.float32)
    assert spatial.map_status(zeros) == "undefined"
    zeros[0, 0, 0] = 1.0
    assert spatial.map_status(zeros) == "valid"
    with pytest.raises(ValueError, match="NEGATIVE"):
        spatial.map_status(-zeros)
    with pytest.raises(ValueError, match="INTERFACE"):
        spatial.map_status(zeros.astype(np.float16))


def test_stable_topk_uses_value_descending_then_flat_index_ascending() -> None:
    heatmap = np.zeros(spatial.MAP_SHAPE, dtype=np.float32)
    heatmap.reshape(-1)[: spatial.SALIENCY_VOXELS + 2] = 1.0
    selected = spatial.stable_topk_indices(heatmap)
    assert np.array_equal(selected, np.arange(spatial.SALIENCY_VOXELS))
    with pytest.raises(ValueError, match="UNDEFINED_MAP"):
        spatial.stable_topk_indices(np.zeros(spatial.MAP_SHAPE, dtype=np.float32))


def test_random_masks_are_reproducible_model_independent_and_unique() -> None:
    first = spatial.matched_random_mask_indices(
        base_seed=9, fold_index=1, nodule_uid="uid", target="malignancy"
    )
    second = spatial.matched_random_mask_indices(
        base_seed=9, fold_index=1, nodule_uid="uid", target="malignancy"
    )
    other = spatial.matched_random_mask_indices(
        base_seed=9, fold_index=1, nodule_uid="uid", target="subtlety"
    )
    assert len(first) == 20
    assert all(len(mask) == spatial.SALIENCY_VOXELS for mask in first)
    assert all(len(np.unique(mask)) == spatial.SALIENCY_VOXELS for mask in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    assert not np.array_equal(first[0], other[0])


def test_occlusion_is_not_inplace_and_uses_exact_mask() -> None:
    image = np.ones((1, *spatial.MAP_SHAPE), dtype=np.float32)
    before = image.copy()
    mask = np.arange(spatial.SALIENCY_VOXELS, dtype=np.int64)
    result = spatial.occlude_image_copy(image, mask)
    assert np.array_equal(image, before)
    assert np.sum(result == 0.0) == spatial.SALIENCY_VOXELS


def test_both_faithfulness_quantities_include_positive_and_negative_error_change() -> None:
    worsened = spatial.faithfulness_quantities(0.8, 0.5, 1.0)
    improved = spatial.faithfulness_quantities(0.8, 0.9, 1.0)
    assert worsened["output_sensitivity"] == pytest.approx(0.3)
    assert worsened["error_increase"] == pytest.approx(0.3)
    assert improved["output_sensitivity"] == pytest.approx(0.1)
    assert improved["error_increase"] == pytest.approx(-0.1)


def test_faithfulness_record_retains_all_20_values_and_aggregates() -> None:
    scores = np.linspace(0.0, 1.0, 20)
    result = spatial.build_faithfulness_record(
        score_original=0.5,
        target_normalized=1.0,
        saliency_score_occluded=0.0,
        random_scores_occluded=scores,
    )
    assert len(result["random_output_sensitivity_values"]) == 20
    assert len(result["random_error_increase_values"]) == 20
    assert set(result["random_output_sensitivity"]) == {"mean", "sd", "median", "min", "max"}
    assert result["saliency_error_increase"] > 0
    aggregate = spatial.aggregate_faithfulness_records([result], "error_increase")
    assert aggregate["sample_count"] == 1


def _map_record(value: float = 1.0) -> dict[str, object]:
    return {
        "nodule_uid": "private-uid",
        "model": "blackbox",
        "fold_index": 0,
        "target": "malignancy",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "map": np.full(spatial.MAP_SHAPE, value, dtype=np.float32),
    }


def test_raw_float32_zstd_parquet_shard_roundtrip_and_tamper_detection(tmp_path) -> None:
    path = tmp_path / "maps.parquet"
    seal = spatial.write_map_shard(path, [_map_record()])
    assert seal["records"] == 1
    restored = spatial.read_and_verify_map_shard(path)
    assert np.array_equal(restored[0]["map"], _map_record()["map"])
    seal_path = path.with_suffix(".parquet.json")
    payload = json.loads(seal_path.read_text())
    payload["file_sha256"] = "0" * 64
    seal_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="FILE_HASH_MISMATCH"):
        spatial.read_and_verify_map_shard(path)


def test_shard_counts_unique_nodules_not_map_records_and_validates_targets(tmp_path) -> None:
    records = []
    for nodule_index in range(16):
        for target in spatial.target_specs("standard_cbm"):
            record = _map_record(float(nodule_index + 1))
            record.update(
                nodule_uid=f"n{nodule_index}",
                model="standard_cbm",
                target=target.name,
            )
            records.append(record)
    path = tmp_path / "concept-maps.parquet"
    seal = spatial.write_map_shard(path, records)
    assert seal["nodule_count"] == 16
    assert seal["records"] == 16 * 9
    assert len(spatial.read_and_verify_map_shard(path)) == 16 * 9
    extra = dict(records[0], nodule_uid="n16")
    with pytest.raises(ValueError, match="NODULE_COUNT_INVALID"):
        spatial.write_map_shard(tmp_path / "too-many.parquet", [*records, extra])
    with pytest.raises(ValueError, match="DUPLICATE_TARGET_RECORD"):
        spatial.write_map_shard(tmp_path / "duplicate.parquet", [records[0], records[0]])
    invalid = dict(records[0], target="not-a-target")
    with pytest.raises(ValueError, match="TARGET_INVALID"):
        spatial.write_map_shard(tmp_path / "invalid.parquet", [invalid])


def test_shard_seal_schema_tampering_is_rejected(tmp_path) -> None:
    path = tmp_path / "maps.parquet"
    spatial.write_map_shard(path, [_map_record()])
    seal_path = path.with_suffix(".parquet.json")
    payload = json.loads(seal_path.read_text())
    payload["schema_version"] = 999
    seal_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="FILE_HASH_MISMATCH"):
        spatial.read_and_verify_map_shard(path)


def test_formal_gate_defaults_to_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(spatial.SPATIAL_APPROVAL_ENV, raising=False)
    with pytest.raises(PermissionError, match="USER_APPROVAL_REQUIRED"):
        spatial.require_formal_spatial_approval()
    monkeypatch.setenv(spatial.SPATIAL_APPROVAL_ENV, "1")
    spatial.require_formal_spatial_approval()
