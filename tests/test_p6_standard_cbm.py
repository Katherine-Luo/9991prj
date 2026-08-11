from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.p6_standard_cbm import (
    CANONICAL_VECTOR_SLICES,
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    StandardCBMConceptPredictor,
    ConceptRecord,
    TaskCacheRecord,
    activate_concept_logits,
    batchnorm_state_sha256,
    build_deterministic_concept_heads,
    build_deterministic_task_head,
    canonical_concept_vector,
    concept_group_loss_sums,
    concept_head_seed,
    concept_loss,
    evaluate_concept_records,
    evaluate_task_records,
    ensure_predicted_cache_features,
    freeze_concept_predictor,
    module_state_sha256,
    task_head_seed,
    task_cache_records,
    task_optimizer,
    task_predictions_and_contributions,
    train_concept_one_epoch,
    train_task_one_epoch,
    run_training_stage,
    verify_training_stage,
    _test_prediction_rows,
    _validate_test_predictions,
    _validate_test_claim,
    _validate_metrics_file,
    stage_resume_requested,
    stage_a_preflight_steps,
    validate_sequential_completion,
    validate_p6_execution_config,
    validate_cache_provenance,
    verify_train_validation_caches,
    write_train_validation_caches,
    predict_concept_cache_frame,
)


def _logits(batch_size: int = 3) -> OrderedDict[str, object]:
    import torch

    return OrderedDict(
        (
            group,
            torch.linspace(-1.5, 1.5, batch_size * size, dtype=torch.float32).reshape(
                batch_size, size
            ),
        )
        for group, size in CONCEPT_OUTPUT_SIZES.items()
    )


def _targets(batch_size: int = 3) -> OrderedDict[str, object]:
    import torch

    result = OrderedDict()
    for group, size in CONCEPT_OUTPUT_SIZES.items():
        if size == 1:
            result[group] = torch.linspace(0.1, 0.9, batch_size).reshape(-1, 1)
        else:
            value = torch.arange(1, size + 1, dtype=torch.float32)
            value = value / value.sum()
            result[group] = value.repeat(batch_size, 1)
    return result


def _outputs(batch_size: int = 3) -> dict[str, object]:
    logits = _logits(batch_size)
    activated = activate_concept_logits(logits)
    return {
        "logits": logits,
        "activated": activated,
        "canonical_vector": canonical_concept_vector(activated),
    }


def test_p6_execution_supplement_is_enforced() -> None:
    config, digest = validate_p6_execution_config()
    assert digest == "792f544aef33d30f122054ba40bdf8f185cea71e516614545ba3f85879ed3bc3"
    assert config["common_execution_profile"]["formal_gpu_model"] == "H200"


def test_concept_head_seeds_follow_domain_separated_definition() -> None:
    fold_seed = 20260808
    group = "subtlety"
    material = (
        b"Baseline-v2/P6/standard-cbm-concept-head/subtlety"
        + fold_seed.to_bytes(8, "big")
    )
    expected = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & (
        (1 << 63) - 1
    )
    assert concept_head_seed(group, fold_seed) == expected
    task_material = b"Baseline-v2/P6/standard-cbm-task-head" + fold_seed.to_bytes(8, "big")
    assert task_head_seed(fold_seed) == (
        int.from_bytes(hashlib.sha256(task_material).digest()[:8], "big")
        & ((1 << 63) - 1)
    )


def test_eight_linear_heads_are_deterministic_order_independent_and_rng_isolated() -> None:
    import torch

    torch.manual_seed(55)
    state_before = torch.get_rng_state().clone()
    first, first_meta = build_deterministic_concept_heads(20260808)
    assert torch.equal(torch.get_rng_state(), state_before)
    second, second_meta = build_deterministic_concept_heads(20260808)
    other, other_meta = build_deterministic_concept_heads(20260809)
    assert list(first) == list(CONCEPT_GROUP_ORDER)
    assert all(isinstance(first[group], torch.nn.Linear) for group in CONCEPT_GROUP_ORDER)
    assert all(first[group].in_features == 1024 for group in CONCEPT_GROUP_ORDER)
    assert [first[group].out_features for group in CONCEPT_GROUP_ORDER] == [1, 4, 6, 1, 1, 1, 1, 1]
    assert first_meta == second_meta
    assert first_meta["combined_concept_head_initialization_sha256"] == module_state_sha256(second)
    assert (
        first_meta["combined_concept_head_initialization_sha256"]
        != other_meta["combined_concept_head_initialization_sha256"]
    )


def test_task_head_is_deterministic_unconstrained_linear_and_rng_isolated() -> None:
    import torch

    torch.manual_seed(77)
    state_before = torch.get_rng_state().clone()
    first, first_meta = build_deterministic_task_head(20260808)
    assert torch.equal(torch.get_rng_state(), state_before)
    second, second_meta = build_deterministic_task_head(20260808)
    other, other_meta = build_deterministic_task_head(20260809)
    assert isinstance(first, torch.nn.Linear)
    assert (first.in_features, first.out_features) == (16, 1)
    assert first_meta == second_meta
    assert first_meta["task_head_initialization_sha256"] == module_state_sha256(second)
    assert first_meta["task_head_initialization_sha256"] != other_meta["task_head_initialization_sha256"]


def test_activated_predictions_form_exact_canonical_16d_vector() -> None:
    import torch

    logits = _logits(batch_size=2)
    activated = activate_concept_logits(logits)
    vector = canonical_concept_vector(activated)
    assert vector.shape == (2, 16)
    assert torch.all((activated["subtlety"] > 0.0) & (activated["subtlety"] < 1.0))
    assert torch.allclose(activated["internalStructure"].sum(dim=1), torch.ones(2))
    assert torch.allclose(activated["calcification"].sum(dim=1), torch.ones(2))
    for group, vector_slice in CANONICAL_VECTOR_SLICES.items():
        assert torch.equal(vector[:, vector_slice], activated[group])
    assert not torch.equal(vector[:, :1], logits["subtlety"])


def test_concept_predictor_has_no_hidden_heads_and_returns_activated_vector() -> None:
    import torch

    class Encoder(torch.nn.Module):
        def forward(self, image: object) -> object:
            tensor = image
            return tensor.repeat(1, 1024, 1, 1, 1)

    heads, _metadata = build_deterministic_concept_heads(20260808)
    model = StandardCBMConceptPredictor.build(Encoder(), heads)
    result = model(torch.ones((2, 1, 1, 1, 1), dtype=torch.float32))
    assert result["canonical_vector"].shape == (2, 16)
    assert list(model.concept_heads.modules())[0] is model.concept_heads
    assert sum(isinstance(module, torch.nn.Linear) for module in model.concept_heads.modules()) == 8


def test_concept_loss_is_exact_arithmetic_mean_of_eight_group_means() -> None:
    import torch

    outputs = _outputs(batch_size=3)
    total, means = concept_loss(outputs, _targets(batch_size=3))
    assert tuple(means) == CONCEPT_GROUP_ORDER
    assert total == pytest.approx(
        float(torch.stack(tuple(means.values())).mean()), rel=0.0, abs=1e-7
    )
    categorical = -(
        _targets(3)["internalStructure"]
        * torch.log_softmax(outputs["logits"]["internalStructure"], dim=1)
    ).sum(dim=1).mean()
    assert means["internalStructure"] == pytest.approx(float(categorical))


def test_epoch_group_aggregation_is_sample_weighted_for_partial_batch() -> None:
    full_outputs = _outputs(batch_size=5)
    full_targets = _targets(batch_size=5)
    full_sums, full_count = concept_group_loss_sums(full_outputs, full_targets)
    combined = {group: 0.0 for group in CONCEPT_GROUP_ORDER}
    combined_count = 0
    for selection in (slice(0, 3), slice(3, 5)):
        logits = OrderedDict((group, value[selection]) for group, value in full_outputs["logits"].items())
        activated = activate_concept_logits(logits)
        batch_outputs = {
            "logits": logits,
            "activated": activated,
            "canonical_vector": canonical_concept_vector(activated),
        }
        batch_targets = OrderedDict((group, value[selection]) for group, value in full_targets.items())
        sums, count = concept_group_loss_sums(batch_outputs, batch_targets)
        for group, value in sums.items():
            combined[group] += float(value)
        combined_count += count
    assert combined_count == full_count == 5
    for group in CONCEPT_GROUP_ORDER:
        assert combined[group] / combined_count == pytest.approx(float(full_sums[group]) / full_count)


def test_soft_targets_are_required_and_modal_labels_are_not_used() -> None:
    import torch

    targets = _targets(batch_size=2)
    targets["internalStructure"] = torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(2, 1)
    with pytest.raises(ValueError, match="CATEGORICAL_TARGET_SUM_MISMATCH"):
        concept_loss(_outputs(batch_size=2), targets)


def test_freezing_preserves_predictor_and_batchnorm_state() -> None:
    import torch

    model = torch.nn.Sequential(
        OrderedDict(
            (
                ("linear", torch.nn.Linear(4, 4)),
                ("batchnorm", torch.nn.BatchNorm1d(4)),
            )
        )
    )
    model.train()
    before = module_state_sha256(model)
    bn_before = batchnorm_state_sha256(model)
    frozen = freeze_concept_predictor(model)
    assert frozen == before
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert batchnorm_state_sha256(model) == bn_before


def test_task_contributions_reconstruct_normalized_and_rating_outputs() -> None:
    import torch

    head, _metadata = build_deterministic_task_head(20260808)
    vector = canonical_concept_vector(activate_concept_logits(_logits(batch_size=4)))
    result = task_predictions_and_contributions(head, vector)
    raw = result["raw_bias"] + torch.stack(
        tuple(result["raw_group_contributions"].values()), dim=0
    ).sum(dim=0)
    rating = result["rating_scale_bias"] + torch.stack(
        tuple(result["rating_point_contributions"].values()), dim=0
    ).sum(dim=0)
    assert torch.allclose(raw, result["malignancy_raw_score"], atol=1e-6, rtol=0.0)
    assert torch.allclose(rating, result["malignancy_score_1_to_5"], atol=1e-6, rtol=0.0)


def test_task_cache_accepts_only_activated_frozen_predictions() -> None:
    vector = canonical_concept_vector(activate_concept_logits(_logits(batch_size=2))).numpy()
    frame = pd.DataFrame(
        {
            "nodule_uid": ["a", "b"],
            "canonical_activated_concepts": [json.dumps(row.tolist()) for row in vector],
            "feature_source": ["frozen_predicted_activated_concepts"] * 2,
            "feature_dimension": [16, 16],
        }
    )
    assert np.array_equal(ensure_predicted_cache_features(frame), vector.astype(np.float32))
    injected = copy.deepcopy(frame)
    injected["feature_source"] = "ground_truth_concepts"
    with pytest.raises(ValueError, match="GROUND_TRUTH_CONCEPT_INJECTION_FORBIDDEN"):
        ensure_predicted_cache_features(injected)

    negative_probability = frame.copy(deep=True)
    negative_vector = vector.copy()
    negative_vector[0, CANONICAL_VECTOR_SLICES["internalStructure"]] = [
        -0.5,
        1.5,
        0.0,
        0.0,
    ]
    negative_probability.loc[0, "canonical_activated_concepts"] = json.dumps(
        negative_vector[0].tolist()
    )
    with pytest.raises(ValueError, match="TASK_CACHE_ACTIVATION_INVARIANT_FAILED"):
        ensure_predicted_cache_features(negative_probability)

    above_one_probability = frame.copy(deep=True)
    above_one_vector = vector.copy()
    above_one_vector[1, CANONICAL_VECTOR_SLICES["calcification"]] = [
        1.25,
        -0.25,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    above_one_probability.loc[1, "canonical_activated_concepts"] = json.dumps(
        above_one_vector[1].tolist()
    )
    with pytest.raises(ValueError, match="TASK_CACHE_ACTIVATION_INVARIANT_FAILED"):
        ensure_predicted_cache_features(above_one_probability)


def _concept_record(tmp_path: object, uid: str, target: float) -> ConceptRecord:
    path = tmp_path / f"{uid}.npz"
    np.savez(path, image=np.full((1, 64, 64, 64), target, dtype=np.float32))
    target_1_to_5 = 1.0 + 4.0 * target
    extreme = target_1_to_5 <= 2.0 or target_1_to_5 >= 4.0
    label = 0 if target_1_to_5 <= 2.0 else (1 if target_1_to_5 >= 4.0 else None)
    return ConceptRecord(
        nodule_uid=uid,
        patient_key=f"patient-{uid}",
        roi_path=path,
        target_normalized=target,
        target_1_to_5=target_1_to_5,
        extreme_binary_eligible=extreme,
        extreme_binary_label=label,
        continuous_targets=(target,) * 6,
        internal_structure_target=(0.4, 0.3, 0.2, 0.1),
        calcification_target=(0.3, 0.2, 0.2, 0.1, 0.1, 0.1),
        valid_reader_counts=(2,) * 8,
        categorical_ties=(False, False),
    )


def _tiny_concept_predictor() -> object:
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.2))
            self.bn = torch.nn.BatchNorm1d(1)

        def forward(self, image: object) -> dict[str, object]:
            batch_size = image.shape[0]
            base = image.mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1)
            base = self.bn(base) * self.scale
            logits = OrderedDict(
                (
                    group,
                    base.repeat(1, size)
                    + torch.linspace(-0.2, 0.2, size, dtype=torch.float32),
                )
                for group, size in CONCEPT_OUTPUT_SIZES.items()
            )
            activated = activate_concept_logits(logits)
            return {
                "logits": logits,
                "activated": activated,
                "canonical_vector": canonical_concept_vector(activated),
            }

    return Model()


def test_stage_a_preflight_exercises_true_batch_16_and_frozen_task_boundary(
    tmp_path: object,
) -> None:
    import torch
    from lidc_baseline.p5_blackbox import validate_execution_config

    execution, _digest = validate_execution_config(
        "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
    )
    train = [
        _concept_record(tmp_path, f"train-{index:02d}", 0.1 + index * 0.04)
        for index in range(16)
    ]
    validation = [
        _concept_record(tmp_path, f"validation-{index:02d}", 0.12 + index * 0.04)
        for index in range(16)
    ]
    report = stage_a_preflight_steps(
        _tiny_concept_predictor(),
        train,
        validation,
        execution,
        fold_seed=20260808,
        base_seed=20260808,
        fold_index=0,
        device=torch.device("cpu"),
    )
    assert report["batch_size"] == 16
    assert report["concept_forward"] is True
    assert report["concept_eight_group_loss"] is True
    assert report["concept_backward"] is True
    assert report["concept_adam_step"] is True
    assert set(report["concept_group_losses"]) == set(CONCEPT_GROUP_ORDER)
    assert report["train_cache_samples"] == 16
    assert report["validation_cache_samples"] == 16
    assert report["canonical_task_input_dimension"] == 16
    assert report["task_features"] == "frozen_predicted_activated_concepts"
    assert report["task_forward"] is True
    assert report["task_backward"] is True
    assert report["task_adam_step"] is True
    assert report["predictor_unchanged_during_task_step"] is True
    assert (
        report["predictor_semantic_sha256_before_task"]
        == report["predictor_semantic_sha256_after_task"]
    )
    assert (
        report["batchnorm_state_sha256_before_task"]
        == report["batchnorm_state_sha256_after_task"]
    )


def _cache_provenance() -> dict[str, object]:
    digest = "a" * 64
    return {
        "scientific_config_sha256": digest,
        "execution_config_sha256": digest,
        "p6_execution_config_sha256": digest,
        "split_sha256": digest,
        "fold_index": 0,
        "encoder_initialization_sha256": digest,
        "encoder_artifact_file_sha256": digest,
        "concept_head_initialization_sha256": OrderedDict(
            (group, digest) for group in CONCEPT_GROUP_ORDER
        ),
        "combined_concept_head_initialization_sha256": digest,
        "concept_best_checkpoint_sha256": digest,
        "predictor_semantic_sha256": digest,
        "batchnorm_state_sha256": digest,
        "source_manifest_sha256": digest,
        "source_roi_index_sha256": digest,
    }


def test_concept_epoch_uses_all_samples_and_sample_weighted_group_losses(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import lidc_baseline.p6_standard_cbm as p6

    monkeypatch.setattr(p6, "apply_training_augmentation", lambda image, _parameters: image)
    records = [_concept_record(tmp_path, f"n-{index}", 0.1 + index * 0.1) for index in range(5)]
    model = _tiny_concept_predictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    report = train_concept_one_epoch(
        model,
        records,
        optimizer,
        torch.device("cpu"),
        base_seed=20260808,
        fold_index=0,
        epoch_index=0,
        batch_size=3,
        num_workers=0,
    )
    assert report["sample_count"] == 5
    assert tuple(report["group_losses"]) == CONCEPT_GROUP_ORDER
    assert report["concept_loss"] == pytest.approx(
        np.mean(list(report["group_losses"].values()))
    )
    validation = evaluate_concept_records(
        model,
        records,
        torch.device("cpu"),
        batch_size=3,
        num_workers=0,
    )
    assert validation["sample_count"] == 5


def test_cache_generation_requires_frozen_predictor_and_delays_test(
    tmp_path: object,
) -> None:
    import torch

    records = [_concept_record(tmp_path, f"n-{index}", 0.2 + index * 0.1) for index in range(2)]
    model = _tiny_concept_predictor()
    with pytest.raises(ValueError, match="CACHE_REQUIRES_FROZEN_EVAL_PREDICTOR"):
        predict_concept_cache_frame(
            model,
            records,
            torch.device("cpu"),
            partition="train",
            batch_size=2,
            num_workers=0,
        )
    freeze_concept_predictor(model)
    train = predict_concept_cache_frame(
        model,
        records,
        torch.device("cpu"),
        partition="train",
        batch_size=2,
        num_workers=0,
    )
    assert set(train["feature_source"]) == {"frozen_predicted_activated_concepts"}
    assert "subtlety_logits" in train
    assert "concept_targets" in train
    with pytest.raises(PermissionError, match="BEFORE_TASK_BEST_FORBIDDEN"):
        predict_concept_cache_frame(
            model,
            records,
            torch.device("cpu"),
            partition="test",
            batch_size=2,
            num_workers=0,
        )
    test = predict_concept_cache_frame(
        model,
        records,
        torch.device("cpu"),
        partition="test",
        batch_size=2,
        num_workers=0,
        task_best_checkpoint_sha256="a" * 64,
    )
    assert set(test["partition"]) == {"test"}

    model.bn.train()
    assert model.training is False
    with pytest.raises(ValueError, match="CACHE_REQUIRES_FROZEN_EVAL_PREDICTOR"):
        predict_concept_cache_frame(
            model,
            records,
            torch.device("cpu"),
            partition="train",
            batch_size=2,
            num_workers=0,
        )


def test_cache_generation_detects_any_frozen_state_mutation(tmp_path: object) -> None:
    import torch

    class MutatingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.register_buffer("calls", torch.tensor(0, dtype=torch.int64))

        def forward(self, image: object) -> dict[str, object]:
            self.calls.add_(1)
            batch_size = image.shape[0]
            logits = OrderedDict(
                (group, torch.zeros((batch_size, size)))
                for group, size in CONCEPT_OUTPUT_SIZES.items()
            )
            activated = activate_concept_logits(logits)
            return {
                "logits": logits,
                "activated": activated,
                "canonical_vector": canonical_concept_vector(activated),
            }

    record = _concept_record(tmp_path, "n-0", 0.2)
    model = MutatingModel()
    model.eval()
    with pytest.raises(ValueError, match="FROZEN_PREDICTOR_CHANGED"):
        predict_concept_cache_frame(
            model,
            [record],
            torch.device("cpu"),
            partition="train",
            batch_size=1,
            num_workers=0,
        )


def test_train_validation_cache_bundle_checks_sets_hashes_and_provenance(
    tmp_path: object,
) -> None:
    import torch

    model = _tiny_concept_predictor()
    freeze_concept_predictor(model)
    train_records = [_concept_record(tmp_path, f"train-{index}", 0.2 + index * 0.1) for index in range(3)]
    validation_records = [_concept_record(tmp_path, f"val-{index}", 0.3 + index * 0.1) for index in range(2)]
    frames = {
        "train": predict_concept_cache_frame(
            model,
            train_records,
            torch.device("cpu"),
            partition="train",
            batch_size=2,
            num_workers=0,
        ),
        "validation": predict_concept_cache_frame(
            model,
            validation_records,
            torch.device("cpu"),
            partition="validation",
            batch_size=2,
            num_workers=0,
        ),
    }
    expected = {
        "train": [record.nodule_uid for record in train_records],
        "validation": [record.nodule_uid for record in validation_records],
    }
    provenance = _cache_provenance()
    provenance["predictor_semantic_sha256"] = module_state_sha256(model)
    provenance["batchnorm_state_sha256"] = batchnorm_state_sha256(model)
    directory = tmp_path / "cache"
    manifest = write_train_validation_caches(directory, frames, expected, provenance)
    assert manifest["test_cache_generated"] is False
    loaded, verified = verify_train_validation_caches(directory, expected, provenance)
    assert set(loaded) == {"train", "validation"}
    assert verified == manifest
    first_train_hash = manifest["partitions"]["train"]["cache_file_sha256"]
    reused = write_train_validation_caches(directory, frames, expected, provenance)
    assert reused["partitions"]["train"]["cache_file_sha256"] == first_train_hash

    with pytest.raises(ValueError, match="CACHE_UID_SET_MISMATCH"):
        verify_train_validation_caches(
            directory,
            {**expected, "validation": [*expected["validation"], "extra"]},
            provenance,
        )
    with pytest.raises(ValueError, match="CACHE_MANIFEST_PROVENANCE_MISMATCH"):
        verify_train_validation_caches(
            directory, expected, {**provenance, "split_sha256": "b" * 64}
        )
    train_path = directory / "train.parquet"
    train_path.write_bytes(train_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="CACHE_FILE_HASH_MISMATCH"):
        verify_train_validation_caches(directory, expected, provenance)


def test_partial_cache_bundle_is_not_silently_overwritten(tmp_path: object) -> None:
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "train.parquet").write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="PARTIAL_CACHE_BUNDLE_REQUIRES_AUDIT"):
        write_train_validation_caches(
            directory,
            {"train": pd.DataFrame(), "validation": pd.DataFrame()},
            {"train": [], "validation": []},
            _cache_provenance(),
        )


def test_cache_provenance_requires_all_source_and_initialization_hashes() -> None:
    provenance = _cache_provenance()
    assert validate_cache_provenance(provenance) == provenance
    missing = dict(provenance)
    del missing["source_manifest_sha256"]
    with pytest.raises(ValueError, match="REQUIRED_FIELDS_MISSING"):
        validate_cache_provenance(missing)
    wrong_heads = dict(provenance)
    wrong_heads["concept_head_initialization_sha256"] = {"subtlety": "a" * 64}
    with pytest.raises(ValueError, match="HEAD_HASHES_INVALID"):
        validate_cache_provenance(wrong_heads)


def test_duplicate_expected_or_evaluation_uids_are_rejected(tmp_path: object) -> None:
    import torch

    vector = canonical_concept_vector(activate_concept_logits(_logits(batch_size=2))).numpy()
    frame = pd.DataFrame(
        {
            "nodule_uid": ["a", "b"],
            "canonical_activated_concepts": [json.dumps(row.tolist()) for row in vector],
            "feature_source": ["frozen_predicted_activated_concepts"] * 2,
            "feature_dimension": [16, 16],
            "target_normalized": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError, match="TASK_CACHE_UID_SET_MISMATCH"):
        task_cache_records(frame, ["a", "a", "b"])
    records = [
        TaskCacheRecord("a", tuple(vector[0]), 0.2),
        TaskCacheRecord("a", tuple(vector[0]), 0.2),
    ]
    head, _metadata = build_deterministic_task_head(20260808)
    with pytest.raises(ValueError, match="DUPLICATE_TASK_CACHE_UID"):
        evaluate_task_records(
            head,
            records,
            torch.device("cpu"),
            batch_size=2,
            num_workers=0,
        )
    concept_record = _concept_record(tmp_path, "duplicate", 0.2)
    with pytest.raises(ValueError, match="DUPLICATE_CONCEPT_RECORD_UID"):
        evaluate_concept_records(
            _tiny_concept_predictor(),
            [concept_record, concept_record],
            torch.device("cpu"),
            batch_size=2,
            num_workers=0,
        )


def test_task_stage_reads_predicted_cache_only_and_cannot_change_predictor(
    tmp_path: object,
) -> None:
    import torch
    from lidc_baseline.config import load_config

    predictor = _tiny_concept_predictor()
    freeze_concept_predictor(predictor)
    predictor_before = module_state_sha256(predictor)
    bn_before = batchnorm_state_sha256(predictor)
    source_records = [_concept_record(tmp_path, f"n-{index}", 0.1 + index * 0.1) for index in range(5)]
    frame = predict_concept_cache_frame(
        predictor,
        source_records,
        torch.device("cpu"),
        partition="train",
        batch_size=3,
        num_workers=0,
    )
    records = task_cache_records(frame, [record.nodule_uid for record in source_records])
    head, _metadata = build_deterministic_task_head(20260808)
    optimizer = task_optimizer(
        head,
        load_config("configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"),
    )
    report = train_task_one_epoch(
        head,
        records,
        optimizer,
        torch.device("cpu"),
        base_seed=20260808,
        fold_index=0,
        epoch_index=0,
        batch_size=3,
        num_workers=0,
    )
    assert report["sample_count"] == 5
    validation = evaluate_task_records(
        head,
        records,
        torch.device("cpu"),
        batch_size=3,
        num_workers=0,
    )
    assert validation["sample_count"] == 5
    assert module_state_sha256(predictor) == predictor_before
    assert batchnorm_state_sha256(predictor) == bn_before
    assert {id(parameter) for parameter in head.parameters()} == {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_checkpointed_stage_resumes_at_epoch_boundary_and_selects_earliest_minimum(
    tmp_path: object,
) -> None:
    import torch
    from lidc_baseline.config import load_config
    from lidc_baseline.p5_blackbox import _optimizer, _scheduler

    execution = load_config(
        "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
    )
    provenance = {
        "schema_version": 1,
        "fold_index": 0,
        "stage": "task",
        "config": "a" * 64,
    }
    output = tmp_path / "task_stage"
    first_module = torch.nn.Linear(1, 1)
    first_optimizer = _optimizer(first_module, execution)
    first_scheduler = _scheduler(first_optimizer, execution)
    current = {"epoch": -1}

    def callbacks(module: object, optimizer: object) -> tuple[object, object]:
        def train(epoch: int) -> dict[str, object]:
            current["epoch"] = epoch
            optimizer.zero_grad(set_to_none=True)
            loss = module(torch.ones((3, 1))).square().mean()
            loss.backward()
            optimizer.step()
            return {
                "mse": float(loss.detach()),
                "sample_count": 3,
                "nodule_set_sha256": "b" * 64,
            }

        def validate() -> dict[str, object]:
            epoch = current["epoch"]
            objective = 0.5 - 0.1 * min(epoch, 3)
            return {
                "mse": objective,
                "sample_count": 2,
                "nodule_set_sha256": "c" * 64,
            }

        return train, validate

    first_train, first_validate = callbacks(first_module, first_optimizer)
    interrupted = run_training_stage(
        stage_name="task",
        module=first_module,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        provenance=provenance,
        output_directory=output,
        epochs=80,
        objective_key="mse",
        expected_train_samples=3,
        train_epoch=first_train,
        validate_epoch=first_validate,
        device=torch.device("cpu"),
        resume=False,
        _stop_after_epoch_for_test=2,
    )
    assert interrupted["epoch_index"] == 2

    resumed_module = torch.nn.Linear(1, 1)
    resumed_optimizer = _optimizer(resumed_module, execution)
    resumed_scheduler = _scheduler(resumed_optimizer, execution)
    resumed_train, resumed_validate = callbacks(resumed_module, resumed_optimizer)
    completion = run_training_stage(
        stage_name="task",
        module=resumed_module,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        provenance=provenance,
        output_directory=output,
        epochs=80,
        objective_key="mse",
        expected_train_samples=3,
        train_epoch=resumed_train,
        validate_epoch=resumed_validate,
        device=torch.device("cpu"),
        resume=True,
    )
    assert completion["best_epoch_index"] == 3
    assert completion["best_validation_objective"] == pytest.approx(0.2)
    verified = verify_training_stage(
        stage_name="task",
        output_directory=output,
        expected_provenance=provenance,
        expected_epochs=80,
        objective_key="mse",
        expected_train_samples=3,
    )
    assert verified == completion
    with pytest.raises(ValueError, match="FORMAL_RUNTIME_POLICY_MISMATCH"):
        verify_training_stage(
            stage_name="task",
            output_directory=output,
            expected_provenance=provenance,
            expected_epochs=80,
            objective_key="mse",
            expected_train_samples=3,
            require_h200_runtime=True,
        )


def test_resume_is_stage_aware_for_pristine_downstream_stage(tmp_path: object) -> None:
    concept = tmp_path / "concept_stage"
    task = tmp_path / "task_stage"
    concept.mkdir()
    task.mkdir()
    (concept / "last.pt").write_bytes(b"checkpoint")
    assert stage_resume_requested(True, concept) is True
    assert stage_resume_requested(True, task) is False
    (concept / "training_complete.json").write_text("{}", encoding="utf-8")
    assert stage_resume_requested(True, concept) is False
    (task / "history.csv").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="PARTIAL_STATE_WITHOUT_LAST"):
        stage_resume_requested(True, task)


def test_stage_verifier_rejects_real_best_objective_mismatch(tmp_path: object) -> None:
    import torch
    from lidc_baseline.p5_blackbox import _atomic_csv, _atomic_json, _atomic_torch_save

    provenance = {"schema_version": 1, "fold_index": 0, "stage": "task"}
    output = tmp_path / "stage"
    output.mkdir()
    rows = [
        {
            "epoch_index": index,
            "validation_mse": 0.2 if index == 4 else 0.3 + index / 1000,
            "train_sample_count": 3,
        }
        for index in range(80)
    ]
    _atomic_csv(output / "history.csv", rows, list(rows[0]))
    module = torch.nn.Linear(1, 1)
    payload = {
        "schema_version": 1,
        "provenance": provenance,
        "epoch_index": 5,
        "validation_objective": 0.19,
        "best_epoch_index": 5,
        "best_validation_objective": 0.19,
        "module_state_dict": module.state_dict(),
    }
    _atomic_torch_save(output / "best.pt", payload)
    _atomic_torch_save(output / "last.pt", payload)
    _atomic_json(output / "runtime.json", {"status": "runtime"})
    from lidc_baseline.p4_prepare import sha256_file

    _atomic_json(
        output / "training_complete.json",
        {
            **provenance,
            "status": "STAGE_TRAINING_COMPLETE",
            "stage": "task",
            "epochs_completed": 80,
            "objective_key": "mse",
            "best_epoch_index": 5,
            "best_validation_objective": 0.19,
            "best_checkpoint_sha256": sha256_file(output / "best.pt"),
            "last_checkpoint_sha256": sha256_file(output / "last.pt"),
            "history_sha256": sha256_file(output / "history.csv"),
            "runtime_sha256": sha256_file(output / "runtime.json"),
            "test_evaluated": False,
        },
    )
    with pytest.raises(ValueError, match="BEST_OBJECTIVE_MISMATCH"):
        verify_training_stage(
            stage_name="task",
            output_directory=output,
            expected_provenance=provenance,
            expected_epochs=80,
            objective_key="mse",
            expected_train_samples=3,
        )


def test_test_prediction_rows_keep_concepts_and_exact_contributions(tmp_path: object) -> None:
    import torch

    model = _tiny_concept_predictor()
    freeze_concept_predictor(model)
    records = [
        _concept_record(tmp_path, "a", 0.2),
        _concept_record(tmp_path, "b", 0.8),
    ]
    concept_frame = predict_concept_cache_frame(
        model,
        records,
        torch.device("cpu"),
        partition="test",
        batch_size=2,
        num_workers=0,
        task_best_checkpoint_sha256="a" * 64,
    )
    head, _metadata = build_deterministic_task_head(20260808)
    provenance = {"fold_index": 0, "split_sha256": "a" * 64}
    predictions = _test_prediction_rows(
        concept_frame, head, torch.device("cpu"), provenance
    )
    _validate_test_predictions(predictions, ["a", "b"], provenance)
    assert "subtlety_raw_contribution" in predictions
    assert "internalStructure_rating_point_contribution" in predictions
    tampered = predictions.copy()
    tampered.loc[0, "subtlety_raw_contribution"] += 1e-3
    with pytest.raises(ValueError, match="NORMALIZED_RECONSTRUCTION_FAILED"):
        _validate_test_predictions(tampered, ["a", "b"], provenance)
    missing_logits = predictions.drop(columns=["subtlety_logits"])
    with pytest.raises(ValueError, match="SCHEMA_MISSING"):
        _validate_test_predictions(missing_logits, ["a", "b"], provenance)
    missing_extreme = predictions.drop(columns=["extreme_binary_eligible"])
    with pytest.raises(ValueError, match="SCHEMA_MISSING"):
        _validate_test_predictions(missing_extreme, ["a", "b"], provenance)
    bad_shape = predictions.copy()
    bad_shape.loc[0, "calcification_logits"] = json.dumps([0.0, 1.0])
    with pytest.raises(ValueError, match="CONCEPT_SHAPE_MISMATCH"):
        _validate_test_predictions(bad_shape, ["a", "b"], provenance)
    wrong_scale = predictions.copy()
    wrong_scale.loc[0, "malignancy_score_1_to_5"] += 1e-3
    with pytest.raises(ValueError, match="SCORE_SCALE_CONVERSION_FAILED"):
        _validate_test_predictions(wrong_scale, ["a", "b"], provenance)
    wrong_contribution_scale = predictions.copy()
    wrong_contribution_scale.loc[0, "subtlety_rating_point_contribution"] += 1e-3
    with pytest.raises(ValueError, match="RATING_RECONSTRUCTION_FAILED|CONTRIBUTION_SCALE"):
        _validate_test_predictions(wrong_contribution_scale, ["a", "b"], provenance)
    wrong_tie = predictions.copy()
    wrong_tie.loc[0, "internalStructure_modal_tie"] = True
    with pytest.raises(ValueError, match="CATEGORICAL_TIE_SEMANTIC_MISMATCH"):
        _validate_test_predictions(wrong_tie, ["a", "b"], provenance)
    wrong_extreme = predictions.copy()
    wrong_extreme.loc[0, "extreme_binary_eligible"] = False
    wrong_extreme.loc[0, "extreme_binary_label"] = np.nan
    with pytest.raises(ValueError, match="EXTREME_LOW_LABEL_MISMATCH"):
        _validate_test_predictions(wrong_extreme, ["a", "b"], provenance)
    nonbinary_low = predictions.copy().astype({"extreme_binary_label": "object"})
    nonbinary_low.loc[0, "extreme_binary_label"] = 0.9
    with pytest.raises(ValueError, match="EXTREME_LOW_LABEL_MISMATCH"):
        _validate_test_predictions(nonbinary_low, ["a", "b"], provenance)
    nonbinary_high = predictions.copy().astype({"extreme_binary_label": "object"})
    nonbinary_high.loc[1, "extreme_binary_label"] = 1.9
    with pytest.raises(ValueError, match="EXTREME_HIGH_LABEL_MISMATCH"):
        _validate_test_predictions(nonbinary_high, ["a", "b"], provenance)
    boolean_low = predictions.copy().astype({"extreme_binary_label": "object"})
    boolean_low.loc[0, "extreme_binary_label"] = False
    with pytest.raises(ValueError, match="EXTREME_LOW_LABEL_MISMATCH"):
        _validate_test_predictions(boolean_low, ["a", "b"], provenance)
    infinite_high = predictions.copy().astype({"extreme_binary_label": "object"})
    infinite_high.loc[1, "extreme_binary_label"] = np.inf
    with pytest.raises(ValueError, match="EXTREME_HIGH_LABEL_MISMATCH"):
        _validate_test_predictions(infinite_high, ["a", "b"], provenance)


def test_claim_and_metrics_are_reconstructed_not_trusted(
    tmp_path: object,
) -> None:
    from lidc_baseline.p5_blackbox import _atomic_json, regression_metrics

    provenance = {"fold_index": 0, "split_sha256": "a" * 64}
    claim_path = tmp_path / "test_claim.json"
    _atomic_json(
        claim_path,
        {
            "schema_version": 1,
            **provenance,
            "status": "TEST_INFERENCE_CLAIMED",
            "expected_test_samples": 2,
        },
    )
    _claim, claim_hash = _validate_test_claim(claim_path, provenance, 2)
    assert len(claim_hash) == 64
    tampered = json.loads(claim_path.read_text(encoding="utf-8"))
    tampered["expected_test_samples"] = 3
    _atomic_json(claim_path, tampered)
    with pytest.raises(ValueError, match="CLAIM_PROVENANCE_MISMATCH"):
        _validate_test_claim(claim_path, provenance, 2)

    predictions = pd.DataFrame(
        {
            "target_normalized": [0.2, 0.8, 0.5],
            "malignancy_raw_score": [0.3, 0.7, 0.4],
        }
    )
    metrics_path = tmp_path / "metrics.json"
    metrics = regression_metrics(predictions.to_dict(orient="records"))
    _atomic_json(metrics_path, metrics)
    assert _validate_metrics_file(metrics_path, predictions) == metrics
    metrics["normalized_mae"] += 1e-3
    _atomic_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="TEST_METRIC_MISMATCH"):
        _validate_metrics_file(metrics_path, predictions)


def test_sequential_completion_binds_full_provenance_and_frozen_state() -> None:
    digest = "a" * 64
    provenance = {
        "scientific_config_sha256": digest,
        "execution_config_sha256": digest,
        "split_sha256": digest,
        "fold_index": 0,
        "model": "standard_cbm",
        "stage": "sequential",
    }
    payload = {
        **provenance,
        "status": "SEQUENTIAL_TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "concept_completion_sha256": digest,
        "cache_manifest_sha256": digest,
        "task_completion_sha256": digest,
        "frozen_predictor_semantic_sha256_before_task": digest,
        "frozen_predictor_semantic_sha256_after_task": digest,
        "frozen_batchnorm_state_sha256_before_task": digest,
        "frozen_batchnorm_state_sha256_after_task": digest,
        "cache_partitions": ["train", "validation"],
        "test_concepts_generated": False,
        "test_evaluated": False,
    }
    validate_sequential_completion(
        payload,
        provenance,
        concept_completion_sha256=digest,
        cache_manifest_sha256=digest,
        task_completion_sha256=digest,
        predictor_semantic_sha256=digest,
        batchnorm_state_sha256_value=digest,
    )
    changed = dict(payload)
    changed["split_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="SEQUENTIAL_TRAINING_PROVENANCE_MISMATCH"):
        validate_sequential_completion(
            changed,
            provenance,
            concept_completion_sha256=digest,
            cache_manifest_sha256=digest,
            task_completion_sha256=digest,
            predictor_semantic_sha256=digest,
            batchnorm_state_sha256_value=digest,
        )


def test_fold_orchestration_resumes_each_stage_and_reuses_completed_seal(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import lidc_baseline.p6_standard_cbm as p6

    digest = "a" * 64
    scientific = {
        "protocol": {"version": "Baseline-v2"},
        "reproducibility": {"base_seed": 20260808},
    }
    execution = {
        "reference_reported": {"epochs": 80},
        "project_preregistered": {
            "batching": {"micro_batch_size": 16},
            "reproducibility": {
                "torch_use_deterministic_algorithms": True,
                "warn_only": True,
            },
        },
        "execution_profile": {
            "profile_id": "test-h200",
            "formal_gpu_model": "H200",
        },
    }
    split = {
        "split_sha256": digest,
        "fold_index": 0,
        "partitions": {
            "train": {"nodule_uids": ["train"], "summary": {"nodules": 1}},
            "validation": {"nodule_uids": ["validation"], "summary": {"nodules": 1}},
            "test": {"nodule_uids": ["test"], "summary": {"nodules": 1}},
        },
    }
    initialization = {
        "fold_seed": 20260808,
        "encoder_initialization_sha256": digest,
        "encoder_artifact_file_sha256": digest,
        "concept_head_initialization_seeds": {group: 1 for group in CONCEPT_GROUP_ORDER},
        "concept_head_initialization_sha256": OrderedDict(
            (group, digest) for group in CONCEPT_GROUP_ORDER
        ),
        "combined_concept_head_initialization_sha256": digest,
        "concept_head_seed_derivation": "test",
    }

    class Predictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

    monkeypatch.setattr(
        p6,
        "_load_p6_sources",
        lambda *_args, **_kwargs: (
            scientific,
            execution,
            digest,
            {},
            digest,
            split,
            pd.DataFrame(),
            pd.DataFrame(),
            tmp_path / "encoder.pt",
        ),
    )
    monkeypatch.setattr(p6, "require_formal_gpu_for_cuda", lambda *_args: None)
    monkeypatch.setattr(p6, "configure_fp32_determinism", lambda *_args: None)
    monkeypatch.setattr(
        p6,
        "build_initialized_concept_predictor",
        lambda *_args: (Predictor(), copy.deepcopy(initialization)),
    )
    monkeypatch.setattr(p6, "_optimizer", lambda *_args: object())
    monkeypatch.setattr(p6, "_scheduler", lambda *_args: object())
    monkeypatch.setattr(
        p6, "build_partition_concept_records", lambda *_args: [object()]
    )
    monkeypatch.setattr(
        p6,
        "_load_concept_best",
        lambda **_kwargs: (
            (lambda model: (freeze_concept_predictor(model), model)[1])(Predictor()),
            copy.deepcopy(initialization),
        ),
    )
    monkeypatch.setattr(
        p6, "_cache_provenance_from_run", lambda **_kwargs: _cache_provenance()
    )
    cache_frame = pd.DataFrame(
        {
            "nodule_uid": ["train"],
            "feature_source": ["frozen_predicted_activated_concepts"],
            "feature_dimension": [16],
            "canonical_activated_concepts": [json.dumps([0.5] * 16)],
            "target_normalized": [0.2],
        }
    )
    monkeypatch.setattr(
        p6, "predict_concept_cache_frame", lambda *_args, **_kwargs: cache_frame
    )

    def write_caches(directory: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "train.parquet").write_bytes(b"train")
        (directory / "validation.parquet").write_bytes(b"validation")
        (directory / "cache_manifest.json").write_text("{}", encoding="utf-8")
        return {"allowed_partitions": ["train", "validation"]}

    monkeypatch.setattr(p6, "write_train_validation_caches", write_caches)
    monkeypatch.setattr(
        p6,
        "verify_train_validation_caches",
        lambda *_args, **_kwargs: (
            {"train": cache_frame, "validation": cache_frame},
            {"allowed_partitions": ["train", "validation"]},
        ),
    )
    monkeypatch.setattr(
        p6,
        "task_cache_records",
        lambda *_args, **_kwargs: [TaskCacheRecord("train", tuple([0.5] * 16), 0.2)],
    )
    monkeypatch.setattr(p6, "task_optimizer", lambda *_args: object())

    calls: list[tuple[str, bool]] = []
    call_counts = {"concept": 0, "task": 0}

    def fake_stage(**kwargs: object) -> dict[str, object]:
        stage = str(kwargs["stage_name"])
        resume = bool(kwargs["resume"])
        calls.append((stage, resume))
        call_counts[stage] += 1
        directory = kwargs["output_directory"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "last.pt").write_bytes(f"{stage}-last".encode())
        if (stage == "concept" and call_counts[stage] == 1) or (
            stage == "task" and call_counts[stage] == 1
        ):
            return {
                "status": "INTERRUPTED_AT_EPOCH_BOUNDARY_FOR_TEST",
                "stage": stage,
                "epoch_index": 0,
            }
        (directory / "best.pt").write_bytes(f"{stage}-best".encode())
        (directory / "training_complete.json").write_text("{}", encoding="utf-8")
        return {"status": "STAGE_TRAINING_COMPLETE", "epochs_completed": 80}

    monkeypatch.setattr(p6, "run_training_stage", fake_stage)
    output_root = tmp_path / "runs"
    arguments = {
        "scientific_config_path": tmp_path / "scientific.yaml",
        "execution_config_path": tmp_path / "execution.yaml",
        "p6_execution_config_path": tmp_path / "p6.yaml",
        "manifest_path": tmp_path / "manifest.parquet",
        "roi_index_path": tmp_path / "roi.parquet",
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": output_root,
    }
    first = p6._train_fold_locked(**arguments, resume=False, _stop_concept_after_epoch_for_test=None, _stop_task_after_epoch_for_test=None)
    assert first["stage"] == "concept"
    second = p6._train_fold_locked(**arguments, resume=True, _stop_concept_after_epoch_for_test=None, _stop_task_after_epoch_for_test=None)
    assert second["stage"] == "task"
    third = p6._train_fold_locked(**arguments, resume=True, _stop_concept_after_epoch_for_test=None, _stop_task_after_epoch_for_test=None)
    assert third["status"] == "SEQUENTIAL_TRAINING_COMPLETE_TEST_NOT_EVALUATED"
    assert calls == [
        ("concept", False),
        ("concept", True),
        ("task", False),
        ("concept", False),
        ("task", True),
    ]
    before = list(calls)
    monkeypatch.setattr(
        p6,
        "_prepare_trained_context",
        lambda **_kwargs: {
            "sequential": json.loads(
                (
                    output_root
                    / "fold_0"
                    / "sequential_training_complete.json"
                ).read_text(encoding="utf-8")
            )
        },
    )
    reused = p6._train_fold_locked(**arguments, resume=True, _stop_concept_after_epoch_for_test=None, _stop_task_after_epoch_for_test=None)
    assert reused["status"] == "SEQUENTIAL_TRAINING_COMPLETE_TEST_NOT_EVALUATED"
    assert calls == before


def test_test_transaction_runs_inference_once_recovers_and_blocks_claim_only(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import lidc_baseline.p6_standard_cbm as p6
    from lidc_baseline.p5_blackbox import _atomic_json

    records = [
        _concept_record(tmp_path, "a", 0.2),
        _concept_record(tmp_path, "b", 0.5),
        _concept_record(tmp_path, "c", 0.9),
    ]
    concept_model = _tiny_concept_predictor()
    freeze_concept_predictor(concept_model)
    concept_frame = predict_concept_cache_frame(
        concept_model,
        records,
        torch.device("cpu"),
        partition="test",
        batch_size=2,
        num_workers=0,
        task_best_checkpoint_sha256="a" * 64,
    )
    task_head, task_metadata = build_deterministic_task_head(20260808)
    digest = "a" * 64
    output = tmp_path / "runs" / "fold_0"
    output.mkdir(parents=True)
    context = {
        "scientific": {
            "protocol": {"version": "Baseline-v2"},
            "reproducibility": {"base_seed": 20260808},
        },
        "execution": {"project_preregistered": {"batching": {"micro_batch_size": 16}}},
        "execution_hash": digest,
        "p6_hash": digest,
        "split": {
            "split_sha256": digest,
            "fold_index": 0,
            "partitions": {
                "test": {
                    "nodule_uids": [record.nodule_uid for record in records],
                    "summary": {"nodules": 3},
                }
            },
        },
        "initialization": {
            "fold_seed": 20260808,
            "encoder_initialization_sha256": digest,
            "encoder_artifact_file_sha256": digest,
            "concept_head_initialization_sha256": OrderedDict(
                (group, digest) for group in CONCEPT_GROUP_ORDER
            ),
            "combined_concept_head_initialization_sha256": digest,
        },
        "cache_provenance": {
            "concept_best_checkpoint_sha256": digest,
            "predictor_semantic_sha256": digest,
            "source_manifest_sha256": digest,
            "source_roi_index_sha256": digest,
        },
        "task_provenance": {
            **task_metadata,
        },
        "output": output,
        "concept_model": concept_model,
        "task_head": task_head,
        "manifest": pd.DataFrame(),
        "roi_index": pd.DataFrame(),
    }
    (output / "concept_cache").mkdir()
    (output / "concept_cache" / "cache_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    (output / "task_stage").mkdir()
    (output / "task_stage" / "best.pt").write_bytes(b"task-best")
    monkeypatch.setattr(p6, "_prepare_trained_context", lambda **_kwargs: context)
    monkeypatch.setattr(p6, "build_partition_concept_records", lambda *_args: records)
    inference_calls = {"count": 0}

    def predict(*_args: object, **_kwargs: object) -> pd.DataFrame:
        inference_calls["count"] += 1
        return concept_frame

    monkeypatch.setattr(p6, "predict_concept_cache_frame", predict)
    arguments = {
        "scientific_config_path": tmp_path / "scientific.yaml",
        "execution_config_path": tmp_path / "execution.yaml",
        "p6_execution_config_path": tmp_path / "p6.yaml",
        "manifest_path": tmp_path / "manifest.parquet",
        "roi_index_path": tmp_path / "roi.parquet",
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": tmp_path / "runs",
    }
    first = p6._evaluate_test_once_locked(**arguments)
    assert first["status"] == "TEST_EVALUATED_EXACTLY_ONCE"
    assert inference_calls["count"] == 1
    second = p6._evaluate_test_once_locked(**arguments)
    assert second == first
    assert inference_calls["count"] == 1

    (output / "test_evaluation.json").unlink()
    (output / "metrics.json").unlink()
    recovered = p6._evaluate_test_once_locked(**arguments)
    assert recovered["status"] == "TEST_EVALUATED_EXACTLY_ONCE"
    assert inference_calls["count"] == 1

    claim_only_output = tmp_path / "claim-only" / "fold_0"
    claim_only_output.mkdir(parents=True)
    claim_context = dict(context)
    claim_context["output"] = claim_only_output
    (claim_only_output / "concept_cache").mkdir()
    (claim_only_output / "concept_cache" / "cache_manifest.json").write_text(
        "{}", encoding="utf-8"
    )
    (claim_only_output / "task_stage").mkdir()
    (claim_only_output / "task_stage" / "best.pt").write_bytes(b"task-best")
    provenance = p6._test_prediction_provenance(claim_context)
    _atomic_json(
        claim_only_output / "test_claim.json",
        {
            "schema_version": 1,
            **provenance,
            "status": "TEST_INFERENCE_CLAIMED",
            "expected_test_samples": 3,
        },
    )
    monkeypatch.setattr(
        p6, "_prepare_trained_context", lambda **_kwargs: claim_context
    )
    claim_arguments = dict(arguments)
    claim_arguments["output_root"] = tmp_path / "claim-only"
    with pytest.raises(RuntimeError, match="CLAIM_EXISTS_WITHOUT_PREDICTIONS"):
        p6._evaluate_test_once_locked(**claim_arguments)
    assert inference_calls["count"] == 1
