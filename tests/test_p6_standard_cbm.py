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
    activate_concept_logits,
    batchnorm_state_sha256,
    build_deterministic_concept_heads,
    build_deterministic_task_head,
    canonical_concept_vector,
    concept_group_loss_sums,
    concept_head_seed,
    concept_loss,
    ensure_predicted_cache_features,
    freeze_concept_predictor,
    module_state_sha256,
    task_head_seed,
    task_predictions_and_contributions,
    validate_p6_execution_config,
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
