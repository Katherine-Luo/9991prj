from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from lidc_baseline.p6_standard_cbm import CONCEPT_GROUP_ORDER, CONTINUOUS_CONCEPTS
from lidc_baseline.p7_mixed_cem import (
    MixedTypeCEM,
    apply_intervention_weights,
    batch_shared_intervention_mask,
    build_deterministic_cem_components,
    cem_losses,
    task_predictions_and_contributions,
    validate_p7_execution_config,
)


class TinyEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(2, 3, 4), keepdim=True)
        return pooled.repeat(1, 1024, 2, 2, 2)


def targets(batch_size: int) -> dict[str, object]:
    concepts: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        if group == "internalStructure":
            concepts[group] = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).repeat(batch_size, 1)
        elif group == "calcification":
            concepts[group] = torch.tensor(
                [[0.05, 0.1, 0.15, 0.2, 0.2, 0.3]]
            ).repeat(batch_size, 1)
        else:
            concepts[group] = torch.full((batch_size, 1), 0.35)
    return {
        "concepts": concepts,
        "malignancy": torch.full((batch_size, 1), 0.6),
    }


def build_model(fold_seed: int = 20260808) -> torch.nn.Module:
    components, _metadata = build_deterministic_cem_components(fold_seed)
    return MixedTypeCEM.build(TinyEncoder(), components)


def test_execution_config_enforces_mixed_cem_identity() -> None:
    config, digest = validate_p7_execution_config()
    assert config["method_declaration"]["label"] == (
        "A project-specific mixed-type extension of the original CEM."
    )
    assert digest == "60e84612eec0ce60b0d17284f6888ddea3627778ab39bcee4c0c6ee3b0c63a2c"


def test_dynamic_states_shapes_probabilities_and_shared_scorers() -> None:
    model = build_model()
    h_x = torch.stack((torch.zeros(1024), torch.ones(1024)))
    outputs = model.forward_from_features(h_x)
    assert model.continuous_scorer.in_features == 32
    assert model.categorical_scorer.in_features == 16
    assert outputs["flat_mixed_embedding"].shape == (2, 128)
    for group in CONCEPT_GROUP_ORDER:
        states = outputs["states"][group]
        probabilities = outputs["activated"][group]
        if group in CONTINUOUS_CONCEPTS:
            assert states.shape == (2, 2, 16)
            assert probabilities.shape == (2, 1)
            assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
        else:
            classes = 4 if group == "internalStructure" else 6
            assert states.shape == (2, classes, 16)
            assert probabilities.shape == (2, classes)
            assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
        assert not torch.equal(states[0], states[1])
    assert not any("state_table" in name for name, _parameter in model.named_parameters())


def test_fixed_probabilities_with_changed_features_change_dynamic_states() -> None:
    model = build_model()
    h_a = torch.zeros(1, 1024)
    h_b = torch.ones(1, 1024)
    a = model.states_and_probabilities(h_a)
    b = model.states_and_probabilities(h_b)
    fixed = OrderedDict(
        (group, a["activated"][group]) for group in CONCEPT_GROUP_ORDER
    )
    mixed_a = model.mix_states(a["states"], fixed)
    mixed_b = model.mix_states(b["states"], fixed)
    assert any(
        not torch.equal(mixed_a[group], mixed_b[group])
        for group in CONCEPT_GROUP_ORDER
    )


def test_joint_loss_is_task_plus_point_zero_one_equal_group_loss() -> None:
    model = build_model()
    outputs = model.forward_from_features(torch.randn(3, 1024))
    losses = cem_losses(outputs, targets(3))
    assert tuple(losses["group_losses"]) == CONCEPT_GROUP_ORDER
    expected_concept = torch.stack(tuple(losses["group_losses"].values())).mean()
    assert torch.allclose(losses["concept_loss"], expected_concept)
    assert torch.allclose(
        losses["total_loss"], losses["task_loss"] + 0.01 * losses["concept_loss"]
    )


def test_randint_intervention_mask_is_batch_shared_and_resume_deterministic() -> None:
    arguments = {
        "base_seed": 20260808,
        "fold_index": 2,
        "epoch_index": 7,
        "batch_index": 11,
    }
    first = batch_shared_intervention_mask(**arguments)
    second = batch_shared_intervention_mask(**arguments)
    changed = batch_shared_intervention_mask(**{**arguments, "batch_index": 12})
    assert first.dtype == torch.bool
    assert first.shape == (8,)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)


def test_randint_intervention_decision_rates_match_preregistered_gates() -> None:
    masks = torch.stack(
        [
            batch_shared_intervention_mask(
                base_seed=20260808,
                fold_index=0,
                epoch_index=batch_index // 128,
                batch_index=batch_index,
            )
            for batch_index in range(4096)
        ]
    ).float()
    assert 0.24 <= float(masks.mean()) <= 0.26
    assert torch.all((masks.mean(dim=0) >= 0.23) & (masks.mean(dim=0) <= 0.27))


def test_intervention_replaces_weights_only_and_preserves_sample_states() -> None:
    model = build_model()
    h_x = torch.randn(2, 1024)
    generated = model.states_and_probabilities(h_x)
    mask = torch.tensor([True, False, True, False, True, False, True, False])
    concept_targets = targets(2)["concepts"]
    effective = apply_intervention_weights(generated["activated"], concept_targets, mask)
    for index, group in enumerate(CONCEPT_GROUP_ORDER):
        expected = concept_targets[group] if bool(mask[index]) else generated["activated"][group]
        assert effective[group] is expected
    intervened = model.forward_from_features(
        h_x,
        intervention_targets=concept_targets,
        intervention_mask=mask,
    )
    for group in CONCEPT_GROUP_ORDER:
        assert torch.equal(intervened["states"][group], generated["states"][group])
    report = task_predictions_and_contributions(model, intervened)
    assert report["normalized_reconstruction_max_abs_error"] <= 1e-6
    assert report["rating_reconstruction_max_abs_error"] <= 1e-6


def test_initialization_is_isolated_reproducible_and_fold_specific() -> None:
    torch.manual_seed(123)
    _ = torch.rand(9)
    _first, first = build_deterministic_cem_components(20260808)
    _ = torch.rand(17)
    _second, second = build_deterministic_cem_components(20260808)
    _third, third = build_deterministic_cem_components(20260809)
    assert first["combined_cem_initialization_sha256"] == second[
        "combined_cem_initialization_sha256"
    ]
    assert first["combined_cem_initialization_sha256"] != third[
        "combined_cem_initialization_sha256"
    ]
    assert first["state_generator_initialization_sha256"] == second[
        "state_generator_initialization_sha256"
    ]


def test_task_output_is_unconstrained_and_contributions_reconstruct() -> None:
    model = build_model()
    with torch.no_grad():
        model.task_head.weight.fill_(0.01)
        model.task_head.bias.fill_(1.25)
    outputs = model.forward_from_features(torch.randn(4, 1024))
    report = task_predictions_and_contributions(model, outputs)
    assert torch.any(report["malignancy_raw_score"] > 1.0)
    assert torch.equal(
        report["malignancy_raw_score"], report["malignancy_score_normalized"]
    )
    assert report["normalized_reconstruction_max_abs_error"] <= 1e-6
    assert report["rating_reconstruction_max_abs_error"] <= 1e-6


def test_invalid_partial_intervention_is_rejected() -> None:
    model = build_model()
    with pytest.raises(ValueError, match="P7_PARTIAL_INTERVENTION_ARGUMENTS"):
        model.forward_from_features(
            torch.zeros(1, 1024),
            intervention_targets=targets(1)["concepts"],
        )
