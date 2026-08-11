from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from lidc_baseline.p6_standard_cbm import (
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    CONTINUOUS_CONCEPTS,
)
from lidc_baseline.p8_gam import (
    EXPERTS_PER_GROUP,
    LearnedSoftmaxGAM,
    build_deterministic_gam_components,
    gam_losses,
    task_predictions_and_contributions,
    validate_p8_execution_config,
)


class IdentityEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


def _model(fold_seed: int = 20260808) -> torch.nn.Module:
    components, _ = build_deterministic_gam_components(fold_seed)
    return LearnedSoftmaxGAM.build(IdentityEncoder(), components)


def _features(batch_size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(101)
    return torch.randn(batch_size, 1024, generator=generator)


def _targets(outputs: dict[str, object]) -> dict[str, object]:
    batch_size = int(outputs["malignancy_raw_score"].shape[0])
    concepts: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        size = CONCEPT_OUTPUT_SIZES[group]
        if group in CONTINUOUS_CONCEPTS:
            concepts[group] = torch.full((batch_size, size), 0.4)
        else:
            concepts[group] = torch.full((batch_size, size), 1.0 / size)
    return {
        "malignancy": torch.full((batch_size, 1), 0.5),
        "concepts": concepts,
    }


def test_p8_execution_config_runtime_guard() -> None:
    config, digest = validate_p8_execution_config()
    assert config["phase"] == "P8"
    assert len(digest) == 64


def test_gam_has_eight_groups_and_five_independent_local_experts() -> None:
    model = _model()
    assert tuple(model.experts) == CONCEPT_GROUP_ORDER
    parameter_ids: set[int] = set()
    for group in CONCEPT_GROUP_ORDER:
        experts = model.experts[group]
        assert len(experts) == EXPERTS_PER_GROUP
        expected_input = CONCEPT_OUTPUT_SIZES[group]
        for expert in experts:
            assert isinstance(expert[0], torch.nn.Linear)
            assert expert[0].in_features == expected_input
            assert expert[0].out_features == 32
            assert isinstance(expert[1], torch.nn.ReLU)
            assert expert[2].in_features == 32
            assert expert[2].out_features == 16
            assert isinstance(expert[3], torch.nn.ReLU)
            assert expert[4].in_features == 16
            assert expert[4].out_features == 1
            for parameter in expert.parameters():
                assert id(parameter) not in parameter_ids
                parameter_ids.add(id(parameter))
    assert len(parameter_ids) == 40 * 6


def test_zero_alpha_logits_produce_exact_uniform_weights() -> None:
    model = _model()
    outputs = model.forward_from_features(_features())
    for group in CONCEPT_GROUP_ORDER:
        assert torch.equal(model.alpha_logits[group], torch.zeros(5))
        assert torch.equal(outputs["alpha_weights"][group], torch.full((5,), 0.2))
        assert outputs["alpha_weights"][group].requires_grad


def test_task_path_is_group_local_and_uses_activated_predictions() -> None:
    model = _model()
    features = _features()
    baseline = model.forward_from_features(features)
    with torch.no_grad():
        model.concept_heads["subtlety"].bias.add_(10.0)
    changed = model.forward_from_features(features)
    assert not torch.equal(
        baseline["expert_outputs"]["subtlety"],
        changed["expert_outputs"]["subtlety"],
    )
    for group in CONCEPT_GROUP_ORDER:
        if group != "subtlety":
            assert torch.equal(
                baseline["expert_outputs"][group], changed["expert_outputs"][group]
            )
    expected = torch.cat(tuple(changed["activated"].values()), dim=1)
    assert torch.equal(changed["canonical_vector"], expected)


def test_alpha_receives_gradient_and_updates_for_every_group() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    initial = {group: model.alpha_logits[group].detach().clone() for group in CONCEPT_GROUP_ORDER}
    outputs = model.forward_from_features(_features(16))
    losses = gam_losses(outputs, _targets(outputs))
    losses["total_loss"].backward()
    for group in CONCEPT_GROUP_ORDER:
        gradient = model.alpha_logits[group].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0
    optimizer.step()
    for group in CONCEPT_GROUP_ORDER:
        assert not torch.equal(model.alpha_logits[group].detach(), initial[group])


def test_gam_loss_is_task_plus_equal_weight_concept_loss() -> None:
    model = _model()
    outputs = model.forward_from_features(_features(7))
    losses = gam_losses(outputs, _targets(outputs))
    assert torch.allclose(
        losses["concept_loss"],
        torch.stack(tuple(losses["group_losses"].values())).mean(),
    )
    assert torch.allclose(
        losses["total_loss"], losses["task_loss"] + losses["concept_loss"]
    )
    assert losses["batch_size"] == 7


def test_deterministic_initialization_is_order_isolated_and_fold_specific() -> None:
    first, first_metadata = build_deterministic_gam_components(20260808)
    torch.manual_seed(7)
    _ = torch.randn(500)
    second, second_metadata = build_deterministic_gam_components(20260808)
    _, other_metadata = build_deterministic_gam_components(20260809)
    assert first_metadata == second_metadata
    assert first_metadata["combined_gam_initialization_sha256"] != (
        other_metadata["combined_gam_initialization_sha256"]
    )
    assert first_metadata["subnetwork_initialization_sha256"] == (
        second_metadata["subnetwork_initialization_sha256"]
    )
    assert first_metadata["initial_alpha_logits_sha256"] == (
        second_metadata["initial_alpha_logits_sha256"]
    )
    assert first_metadata["initial_raw_bias_sha256"] == (
        second_metadata["initial_raw_bias_sha256"]
    )
    assert first["concept_heads"] is not second["concept_heads"]


def test_unbounded_output_and_contribution_reconstruction() -> None:
    model = _model()
    with torch.no_grad():
        model.global_raw_bias.fill_(2.0)
    outputs = model.forward_from_features(_features(5))
    result = task_predictions_and_contributions(model, outputs)
    assert torch.all(result["malignancy_raw_score"] > 1.0)
    assert torch.equal(
        result["malignancy_raw_score"], result["malignancy_score_normalized"]
    )
    assert torch.allclose(
        result["malignancy_score_1_to_5"],
        1.0 + 4.0 * result["malignancy_raw_score"],
    )
    assert result["normalized_reconstruction_max_abs_error"] <= 1e-6
    assert result["rating_reconstruction_max_abs_error"] <= 1e-6


def test_contribution_guard_rejects_tampering() -> None:
    model = _model()
    outputs = model.forward_from_features(_features(3))
    outputs["group_contributions"]["subtlety"] = (
        outputs["group_contributions"]["subtlety"] + 0.01
    )
    with pytest.raises(ValueError, match="P8_NORMALIZED_CONTRIBUTION"):
        task_predictions_and_contributions(model, outputs)
