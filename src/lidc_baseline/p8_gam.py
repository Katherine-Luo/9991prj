"""Train and verify the Baseline-v2 end-to-end learned-softmax GAM."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p4_prepare import (
    build_encoder,
    encoder_state_sha256,
    load_shared_encoder_initialization,
    sha256_file,
    validate_encoder_artifact,
)
from lidc_baseline.p6_standard_cbm import (
    CATEGORICAL_CONCEPTS,
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    activate_concept_logits,
    canonical_concept_vector,
    concept_group_loss_sums,
    module_state_sha256,
)


SCHEMA_VERSION = 1
MODEL_NAME = "end_to_end_cbm_learned_softmax_gam"
P8_EXECUTION_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_p8_gam_h200.yaml"
)
EXPERTS_PER_GROUP = 5
CONCEPT_HEAD_SEED_DOMAIN = "Baseline-v2/P8/gam-concept-head"
SUBNETWORK_SEED_DOMAIN = "Baseline-v2/P8/gam-subnetwork"
RECONSTRUCTION_TOLERANCE = 1e-6


def _torch() -> Any:
    import torch

    return torch


def validate_p8_execution_config(
    config_path: str | Path = P8_EXECUTION_CONFIG_DEFAULT,
    digest_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and enforce the frozen P8 execution supplement."""
    source = Path(config_path)
    config = load_config(source)
    observed = compute_config_sha256(config)
    digest = Path(digest_path) if digest_path is not None else source.with_suffix(
        ".sha256"
    )
    if digest.read_text(encoding="ascii").strip() != observed:
        raise ValueError("P8_EXECUTION_CONFIG_HASH_MISMATCH")
    project = config.get("project_preregistered", {})
    architecture = project.get("architecture", {})
    gam = project.get("gam", {})
    alpha = gam.get("alpha", {})
    loss = project.get("loss", {})
    intervention = project.get("intervention", {})
    if (
        config.get("protocol_version") != "Baseline-v2"
        or config.get("phase") != "P8"
        or config.get("model") != MODEL_NAME
    ):
        raise ValueError("P8_EXECUTION_CONFIG_IDENTITY_MISMATCH")
    if (
        tuple(architecture.get("group_order", ())) != CONCEPT_GROUP_ORDER
        or architecture.get("task_input") != "activated_predicted_concepts_only"
        or architecture.get("preactivation_logits_task_input") != "forbidden"
        or architecture.get("ground_truth_concepts_task_input") != "forbidden"
        or architecture.get("dense_feature_bypass") is not False
        or architecture.get("independent_binary_head") is not False
        or architecture.get("task_output_activation") != "none"
        or architecture.get("task_output_constraint") != "unbounded"
    ):
        raise ValueError("P8_ARCHITECTURE_POLICY_MISMATCH")
    if (
        gam.get("groups") != 8
        or gam.get("experts_per_group") != EXPERTS_PER_GROUP
        or gam.get("total_independent_experts") != 40
        or gam.get("concept_local_inputs_only") is not True
        or gam.get("cross_concept_inputs") != "forbidden"
        or gam.get("simple_expert_average") != "forbidden"
        or gam.get("expert_architecture", {}).get("hidden_sizes") != [32, 16]
        or gam.get("expert_architecture", {}).get("hidden_activation") != "relu"
        or gam.get("expert_architecture", {}).get("output_activation") != "none"
    ):
        raise ValueError("P8_GAM_POLICY_MISMATCH")
    if (
        alpha.get("scope") != "fold_level_global_trainable_parameter"
        or alpha.get("sample_conditioned") is not False
        or alpha.get("logits_per_group") != EXPERTS_PER_GROUP
        or alpha.get("logits_initialization") != "all_zeros"
        or alpha.get("weights_transform") != "softmax"
        or alpha.get("initial_weight_per_expert") != 0.2
    ):
        raise ValueError("P8_ALPHA_POLICY_MISMATCH")
    if (
        loss.get("concept_weight") != 1.0
        or loss.get("total") != "task_loss_plus_concept_loss"
        or intervention
        != {"training": False, "validation": False, "test": False, "reserved_phase": "P9"}
    ):
        raise ValueError("P8_OBJECTIVE_OR_INTERVENTION_POLICY_MISMATCH")
    return config, observed


def _seed_from_material(material: str, fold_seed: int) -> int:
    payload = material.encode("utf-8") + int(fold_seed).to_bytes(
        8, "big", signed=False
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def concept_head_seed(group: str, fold_seed: int) -> int:
    if group not in CONCEPT_OUTPUT_SIZES:
        raise ValueError(f"P8_UNKNOWN_CONCEPT_GROUP:{group}")
    return _seed_from_material(f"{CONCEPT_HEAD_SEED_DOMAIN}/{group}", fold_seed)


def subnetwork_seed(group: str, expert_index: int, fold_seed: int) -> int:
    if group not in CONCEPT_OUTPUT_SIZES:
        raise ValueError(f"P8_UNKNOWN_CONCEPT_GROUP:{group}")
    if expert_index < 0 or expert_index >= EXPERTS_PER_GROUP:
        raise ValueError(f"P8_EXPERT_INDEX_OUT_OF_RANGE:{expert_index}")
    return _seed_from_material(
        f"{SUBNETWORK_SEED_DOMAIN}/{group}/{expert_index}", fold_seed
    )


def _isolated_linear(in_features: int, out_features: int, seed: int) -> Any:
    torch = _torch()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        return torch.nn.Linear(in_features, out_features)


def build_expert(input_size: int, seed: int) -> Any:
    """Build one deterministically initialized concept-local GAM expert."""
    torch = _torch()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(input_size, 32),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(16, 1),
        )


def build_deterministic_gam_components(
    fold_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build order-independent heads, experts, alpha logits, and raw bias."""
    torch = _torch()
    concept_heads = torch.nn.ModuleDict()
    concept_head_seeds: dict[str, int] = {}
    concept_head_hashes: dict[str, str] = {}
    experts = torch.nn.ModuleDict()
    expert_seeds: dict[str, list[int]] = {}
    expert_hashes: dict[str, list[str]] = {}
    for group in CONCEPT_GROUP_ORDER:
        head_seed = concept_head_seed(group, fold_seed)
        head = _isolated_linear(1024, CONCEPT_OUTPUT_SIZES[group], head_seed)
        concept_heads[group] = head
        concept_head_seeds[group] = head_seed
        concept_head_hashes[group] = module_state_sha256(head)
        group_experts = torch.nn.ModuleList()
        group_seeds: list[int] = []
        group_hashes: list[str] = []
        for expert_index in range(EXPERTS_PER_GROUP):
            seed = subnetwork_seed(group, expert_index, fold_seed)
            expert = build_expert(CONCEPT_OUTPUT_SIZES[group], seed)
            group_experts.append(expert)
            group_seeds.append(seed)
            group_hashes.append(module_state_sha256(expert))
        experts[group] = group_experts
        expert_seeds[group] = group_seeds
        expert_hashes[group] = group_hashes
    alpha_logits = torch.nn.ParameterDict(
        {
            group: torch.nn.Parameter(torch.zeros(EXPERTS_PER_GROUP, dtype=torch.float32))
            for group in CONCEPT_GROUP_ORDER
        }
    )
    raw_parameters = torch.nn.ParameterDict(
        {"global_raw_bias": torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))}
    )
    container = torch.nn.ModuleDict(
        {
            "concept_heads": concept_heads,
            "experts": experts,
            "alpha_logits": alpha_logits,
            "raw_parameters": raw_parameters,
        }
    )
    components = {
        "concept_heads": concept_heads,
        "experts": experts,
        "alpha_logits": alpha_logits,
        "raw_parameters": raw_parameters,
    }
    metadata = {
        "concept_head_initialization_seeds": concept_head_seeds,
        "concept_head_initialization_sha256": concept_head_hashes,
        "combined_concept_head_initialization_sha256": module_state_sha256(
            concept_heads
        ),
        "subnetwork_initialization_seeds": expert_seeds,
        "subnetwork_initialization_sha256": expert_hashes,
        "combined_subnetwork_initialization_sha256": module_state_sha256(experts),
        "initial_alpha_logit_group_sha256": {
            group: encoder_state_sha256(
                OrderedDict(
                    alpha_logits=alpha_logits[group].detach().cpu().contiguous()
                )
            )
            for group in CONCEPT_GROUP_ORDER
        },
        "initial_alpha_logits_sha256": module_state_sha256(alpha_logits),
        "initial_raw_bias_sha256": module_state_sha256(raw_parameters),
        "combined_gam_initialization_sha256": module_state_sha256(container),
        "initialization_seed_derivation": (
            "sha256(utf8(domain/group[/expert]) || fold_seed_u64be), "
            "first_8_bytes_u64be_mask_63_bits"
        ),
    }
    return components, metadata


class LearnedSoftmaxGAM:
    """Factory for the end-to-end concept bottleneck learned-softmax GAM."""

    @staticmethod
    def build(encoder: Any, components: Mapping[str, Any]) -> Any:
        torch = _torch()

        class Model(torch.nn.Module):
            def __init__(self, feature_encoder: Any, modules: Mapping[str, Any]) -> None:
                super().__init__()
                self.encoder = feature_encoder
                self.relu = torch.nn.ReLU(inplace=False)
                self.concept_heads = modules["concept_heads"]
                self.experts = modules["experts"]
                self.alpha_logits = modules["alpha_logits"]
                self.raw_parameters = modules["raw_parameters"]

            @property
            def global_raw_bias(self) -> Any:
                return self.raw_parameters["global_raw_bias"]

            def forward_from_features(self, h_x: Any) -> dict[str, Any]:
                if h_x.ndim != 2 or int(h_x.shape[1]) != 1024:
                    raise ValueError("P8_ENCODER_FEATURE_SHAPE_MISMATCH")
                logits = OrderedDict(
                    (group, self.concept_heads[group](h_x))
                    for group in CONCEPT_GROUP_ORDER
                )
                activated = activate_concept_logits(logits)
                expert_outputs: OrderedDict[str, Any] = OrderedDict()
                alpha_weights: OrderedDict[str, Any] = OrderedDict()
                contributions: OrderedDict[str, Any] = OrderedDict()
                for group in CONCEPT_GROUP_ORDER:
                    group_input = activated[group]
                    outputs = torch.cat(
                        tuple(expert(group_input) for expert in self.experts[group]),
                        dim=1,
                    )
                    if outputs.shape != (h_x.shape[0], EXPERTS_PER_GROUP):
                        raise ValueError(f"P8_EXPERT_OUTPUT_SHAPE_MISMATCH:{group}")
                    weights = torch.softmax(self.alpha_logits[group], dim=0)
                    contribution = (outputs * weights.reshape(1, -1)).sum(
                        dim=1, keepdim=True
                    )
                    expert_outputs[group] = outputs
                    alpha_weights[group] = weights
                    contributions[group] = contribution
                raw = self.global_raw_bias.reshape(1, 1) + torch.stack(
                    tuple(contributions.values()), dim=0
                ).sum(dim=0)
                return {
                    "logits": logits,
                    "activated": activated,
                    "canonical_vector": canonical_concept_vector(activated),
                    "expert_outputs": expert_outputs,
                    "alpha_logits": OrderedDict(
                        (group, self.alpha_logits[group])
                        for group in CONCEPT_GROUP_ORDER
                    ),
                    "alpha_weights": alpha_weights,
                    "group_contributions": contributions,
                    "malignancy_raw_score": raw,
                }

            def forward(self, image: Any) -> dict[str, Any]:
                features = self.relu(self.encoder(image))
                h_x = features.mean(dim=(2, 3, 4))
                result = self.forward_from_features(h_x)
                result["encoder_feature_h_x"] = h_x
                return result

        return Model(encoder, components)


def build_initialized_model(
    scientific_config: Mapping[str, Any],
    split: Mapping[str, Any],
    encoder_artifact_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Start P8 from the same immutable P4 fold encoder initialization."""
    encoder = build_encoder()
    encoder_hash = load_shared_encoder_initialization(
        encoder, encoder_artifact_path, scientific_config, split
    )
    validated = validate_encoder_artifact(
        Path(encoder_artifact_path), scientific_config, split
    )
    fold_seed = int(validated["metadata"]["fold_seed"])
    components, metadata = build_deterministic_gam_components(fold_seed)
    if encoder_state_sha256(encoder.state_dict()) != encoder_hash:
        raise ValueError("P8_ENCODER_HASH_CHANGED_BEFORE_TRAINING")
    model = LearnedSoftmaxGAM.build(encoder, components)
    return model, {
        "fold_seed": fold_seed,
        "encoder_initialization_sha256": encoder_hash,
        "encoder_artifact_file_sha256": sha256_file(encoder_artifact_path),
        **metadata,
    }


def gam_losses(
    outputs: Mapping[str, Any], targets: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute task, equal-group concept, and preregistered total GAM loss."""
    torch = _torch()
    raw = outputs["malignancy_raw_score"]
    malignancy = targets["malignancy"]
    if raw.shape != malignancy.shape:
        raise ValueError("P8_TASK_TARGET_SHAPE_MISMATCH")
    task = torch.nn.functional.mse_loss(raw, malignancy, reduction="mean")
    sums, batch_size = concept_group_loss_sums(outputs, targets["concepts"])
    group_losses = OrderedDict(
        (group, value / batch_size) for group, value in sums.items()
    )
    concept = torch.stack(tuple(group_losses.values())).mean()
    total = task + concept
    return {
        "task_loss": task,
        "concept_loss": concept,
        "total_loss": total,
        "group_losses": group_losses,
        "batch_size": batch_size,
    }


def task_predictions_and_contributions(
    model: Any, outputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return exact normalized and rating-scale GAM decompositions."""
    torch = _torch()
    contributions = outputs["group_contributions"]
    if tuple(contributions) != CONCEPT_GROUP_ORDER:
        raise ValueError("P8_GROUP_CONTRIBUTION_ORDER_MISMATCH")
    raw_contributions: OrderedDict[str, Any] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        value = contributions[group]
        if value.ndim != 2 or int(value.shape[1]) != 1:
            raise ValueError(f"P8_GROUP_CONTRIBUTION_SHAPE_MISMATCH:{group}")
        raw_contributions[group] = value.reshape(-1)
    raw = outputs["malignancy_raw_score"].reshape(-1)
    raw_bias = model.global_raw_bias.reshape(1)
    reconstruction = raw_bias + torch.stack(
        tuple(raw_contributions.values()), dim=1
    ).sum(dim=1)
    error = float((reconstruction - raw).abs().max().detach().cpu())
    if error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(f"P8_NORMALIZED_CONTRIBUTION_RECONSTRUCTION_FAILED:{error}")
    rating_contributions = OrderedDict(
        (group, 4.0 * value) for group, value in raw_contributions.items()
    )
    rating_bias = 1.0 + 4.0 * raw_bias
    rating = 1.0 + 4.0 * raw
    rating_reconstruction = rating_bias + torch.stack(
        tuple(rating_contributions.values()), dim=1
    ).sum(dim=1)
    rating_error = float(
        (rating_reconstruction - rating).abs().max().detach().cpu()
    )
    if rating_error > RECONSTRUCTION_TOLERANCE:
        raise ValueError(
            f"P8_RATING_CONTRIBUTION_RECONSTRUCTION_FAILED:{rating_error}"
        )
    return {
        "malignancy_raw_score": raw,
        "malignancy_score_normalized": raw,
        "malignancy_score_1_to_5": rating,
        "raw_bias": raw_bias,
        "raw_group_contributions": raw_contributions,
        "rating_scale_bias": rating_bias,
        "rating_group_contributions": rating_contributions,
        "normalized_reconstruction_max_abs_error": error,
        "rating_reconstruction_max_abs_error": rating_error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the P8 lifecycle command-line interface."""
    from lidc_baseline.p8_gam_lifecycle import main as lifecycle_main

    return lifecycle_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
