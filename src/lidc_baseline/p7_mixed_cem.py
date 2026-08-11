"""Train and verify the Baseline-v2 project-specific mixed-type CEM."""

from __future__ import annotations

import argparse
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
    CONTINUOUS_CONCEPTS,
    canonical_concept_vector,
    concept_group_loss_sums,
    module_state_sha256,
)


SCHEMA_VERSION = 1
MODEL_NAME = "mixed_type_cem"
METHOD_LABEL = "A project-specific mixed-type extension of the original CEM."
P7_EXECUTION_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_p7_mixed_cem_h200.yaml"
)
EMBEDDING_SIZE = 16
CONTINUOUS_STATE_COUNT = 2
INTERVENTION_SEED_DOMAIN = "Baseline-v2/P7/mixed-cem-intervention"
STATE_GENERATOR_SEED_DOMAIN = "Baseline-v2/P7/mixed-cem-state-generator"
CONTINUOUS_SCORER_SEED_DOMAIN = "Baseline-v2/P7/mixed-cem-continuous-scorer"
CATEGORICAL_SCORER_SEED_DOMAIN = "Baseline-v2/P7/mixed-cem-categorical-scorer"
TASK_HEAD_SEED_DOMAIN = "Baseline-v2/P7/mixed-cem-task-head"
INTERVENTION_SCHEMA_VERSION = 1


def _torch() -> Any:
    import torch

    return torch


def validate_p7_execution_config(
    config_path: str | Path = P7_EXECUTION_CONFIG_DEFAULT,
    digest_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and enforce the frozen P7 execution supplement."""
    source = Path(config_path)
    config = load_config(source)
    observed = compute_config_sha256(config)
    digest = Path(digest_path) if digest_path is not None else source.with_suffix(".sha256")
    if digest.read_text(encoding="ascii").strip() != observed:
        raise ValueError("P7_EXECUTION_CONFIG_HASH_MISMATCH")
    declaration = config.get("method_declaration", {})
    project = config.get("project_preregistered", {})
    architecture = project.get("architecture", {})
    states = project.get("dynamic_states", {})
    intervention = project.get("intervention", {})
    if (
        config.get("protocol_version") != "Baseline-v2"
        or config.get("phase") != "P7"
        or config.get("model") != MODEL_NAME
        or declaration.get("label") != METHOD_LABEL
    ):
        raise ValueError("P7_EXECUTION_CONFIG_IDENTITY_MISMATCH")
    if (
        architecture.get("mixed_embedding_shape") != [8, 16]
        or architecture.get("flattened_task_input_size") != 128
        or architecture.get("dense_feature_bypass") is not False
        or architecture.get("state_tables") != "forbidden"
        or tuple(architecture.get("group_order", ())) != CONCEPT_GROUP_ORDER
    ):
        raise ValueError("P7_ARCHITECTURE_POLICY_MISMATCH")
    if (
        states.get("continuous", {}).get("scorer") != "shared_linear_32_to_1"
        or states.get("categorical", {}).get("scorer")
        != "shared_linear_16_to_1_across_all_categorical_states"
    ):
        raise ValueError("P7_SHARED_SCORER_POLICY_MISMATCH")
    if (
        intervention.get("mode")
        != "training_only_batch_shared_group_independent_randint"
        or intervention.get("random_primitive") != "torch_randint"
        or intervention.get("randint_low_inclusive") != 0
        or intervention.get("randint_high_exclusive") != 4
        or intervention.get("intervene_when_value_equals") != 0
        or intervention.get("replace_mixture_weights_only") is not True
        or intervention.get("preserve_sample_conditioned_states") is not True
        or intervention.get("validation_and_test_intervention") is not False
    ):
        raise ValueError("P7_INTERVENTION_POLICY_MISMATCH")
    return config, observed


def _seed_from_material(material: str) -> int:
    return int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
    ) & ((1 << 63) - 1)


def initialization_seed(domain: str, fold_seed: int, group: str | None = None) -> int:
    material = domain if group is None else f"{domain}/{group}"
    return _seed_from_material(f"{material} || {int(fold_seed)}")


def intervention_seed(
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    batch_index: int,
    schema_version: int = INTERVENTION_SCHEMA_VERSION,
) -> int:
    material = " || ".join(
        (
            INTERVENTION_SEED_DOMAIN,
            str(int(base_seed)),
            str(int(fold_index)),
            str(int(epoch_index)),
            str(int(batch_index)),
            str(int(schema_version)),
        )
    )
    return _seed_from_material(material)


def batch_shared_intervention_mask(
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    batch_index: int,
    schema_version: int = INTERVENTION_SCHEMA_VERSION,
) -> Any:
    """Return the preregistered eight-group RandInt intervention decision."""
    torch = _torch()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        intervention_seed(
            base_seed=base_seed,
            fold_index=fold_index,
            epoch_index=epoch_index,
            batch_index=batch_index,
            schema_version=schema_version,
        )
    )
    values = torch.randint(
        low=0,
        high=4,
        size=(len(CONCEPT_GROUP_ORDER),),
        generator=generator,
        dtype=torch.int64,
    )
    return values.eq(0)


def _isolated_linear(in_features: int, out_features: int, seed: int) -> Any:
    torch = _torch()
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        return torch.nn.Linear(in_features, out_features)


def build_deterministic_cem_components(fold_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build order-independent dynamic generators, shared scorers, and task head."""
    torch = _torch()
    generators = torch.nn.ModuleDict()
    generator_seeds: dict[str, int] = {}
    generator_hashes: dict[str, str] = {}
    for group in CONCEPT_GROUP_ORDER:
        output_size = (
            CONTINUOUS_STATE_COUNT * EMBEDDING_SIZE
            if group in CONTINUOUS_CONCEPTS
            else CONCEPT_OUTPUT_SIZES[group] * EMBEDDING_SIZE
        )
        seed = initialization_seed(STATE_GENERATOR_SEED_DOMAIN, fold_seed, group)
        generator = _isolated_linear(1024, output_size, seed)
        generators[group] = generator
        generator_seeds[group] = seed
        generator_hashes[group] = module_state_sha256(generator)
    continuous_seed = initialization_seed(CONTINUOUS_SCORER_SEED_DOMAIN, fold_seed)
    categorical_seed = initialization_seed(CATEGORICAL_SCORER_SEED_DOMAIN, fold_seed)
    task_seed = initialization_seed(TASK_HEAD_SEED_DOMAIN, fold_seed)
    continuous_scorer = _isolated_linear(32, 1, continuous_seed)
    categorical_scorer = _isolated_linear(16, 1, categorical_seed)
    task_head = _isolated_linear(128, 1, task_seed)
    container = torch.nn.ModuleDict(
        {
            "state_generators": generators,
            "continuous_scorer": continuous_scorer,
            "categorical_scorer": categorical_scorer,
            "task_head": task_head,
        }
    )
    components = {
        "state_generators": generators,
        "continuous_scorer": continuous_scorer,
        "categorical_scorer": categorical_scorer,
        "task_head": task_head,
    }
    metadata = {
        "state_generator_initialization_seeds": generator_seeds,
        "state_generator_initialization_sha256": generator_hashes,
        "continuous_scorer_initialization_seed": continuous_seed,
        "continuous_scorer_initialization_sha256": module_state_sha256(
            continuous_scorer
        ),
        "categorical_scorer_initialization_seed": categorical_seed,
        "categorical_scorer_initialization_sha256": module_state_sha256(
            categorical_scorer
        ),
        "task_head_initialization_seed": task_seed,
        "task_head_initialization_sha256": module_state_sha256(task_head),
        "combined_cem_initialization_sha256": module_state_sha256(container),
        "initialization_seed_derivation": (
            "sha256(utf8(domain[/group] || ' || ' || fold_seed)), "
            "first_8_bytes_u64be_mask_63_bits"
        ),
    }
    return components, metadata


def apply_intervention_weights(
    predicted: Mapping[str, Any],
    targets: Mapping[str, Any],
    mask: Any,
) -> OrderedDict[str, Any]:
    """Replace only selected mixture weights with per-sample soft targets."""
    torch = _torch()
    if mask.shape != (len(CONCEPT_GROUP_ORDER),) or mask.dtype != torch.bool:
        raise ValueError("P7_INTERVENTION_MASK_SHAPE_OR_DTYPE_MISMATCH")
    effective: OrderedDict[str, Any] = OrderedDict()
    for index, group in enumerate(CONCEPT_GROUP_ORDER):
        prediction = predicted[group]
        target = targets[group]
        if prediction.shape != target.shape:
            raise ValueError(f"P7_INTERVENTION_TARGET_SHAPE_MISMATCH:{group}")
        effective[group] = target if bool(mask[index]) else prediction
    return effective


class MixedTypeCEM:
    """Factory for the P7 sample-conditioned mixed-type CEM."""

    @staticmethod
    def build(encoder: Any, components: Mapping[str, Any]) -> Any:
        torch = _torch()

        class Model(torch.nn.Module):
            def __init__(self, feature_encoder: Any, modules: Mapping[str, Any]) -> None:
                super().__init__()
                self.encoder = feature_encoder
                self.relu = torch.nn.ReLU(inplace=False)
                self.state_activation = torch.nn.LeakyReLU(0.01, inplace=False)
                self.state_generators = modules["state_generators"]
                self.continuous_scorer = modules["continuous_scorer"]
                self.categorical_scorer = modules["categorical_scorer"]
                self.task_head = modules["task_head"]

            def states_and_probabilities(self, h_x: Any) -> dict[str, Any]:
                if h_x.ndim != 2 or h_x.shape[1] != 1024:
                    raise ValueError("P7_ENCODER_FEATURE_SHAPE_MISMATCH")
                states: OrderedDict[str, Any] = OrderedDict()
                logits: OrderedDict[str, Any] = OrderedDict()
                activated: OrderedDict[str, Any] = OrderedDict()
                batch_size = int(h_x.shape[0])
                for group in CONCEPT_GROUP_ORDER:
                    generated = self.state_activation(self.state_generators[group](h_x))
                    if group in CONTINUOUS_CONCEPTS:
                        group_states = generated.reshape(
                            batch_size, CONTINUOUS_STATE_COUNT, EMBEDDING_SIZE
                        )
                        group_logits = self.continuous_scorer(
                            group_states.reshape(batch_size, 32)
                        )
                        probability = torch.sigmoid(group_logits)
                    else:
                        classes = CONCEPT_OUTPUT_SIZES[group]
                        group_states = generated.reshape(
                            batch_size, classes, EMBEDDING_SIZE
                        )
                        group_logits = self.categorical_scorer(group_states).squeeze(-1)
                        probability = torch.softmax(group_logits, dim=1)
                    states[group] = group_states
                    logits[group] = group_logits
                    activated[group] = probability
                return {"states": states, "logits": logits, "activated": activated}

            def mix_states(
                self,
                states: Mapping[str, Any],
                weights: Mapping[str, Any],
            ) -> OrderedDict[str, Any]:
                mixed: OrderedDict[str, Any] = OrderedDict()
                for group in CONCEPT_GROUP_ORDER:
                    group_states = states[group]
                    probability = weights[group]
                    if group in CONTINUOUS_CONCEPTS:
                        if probability.ndim != 2 or probability.shape[1] != 1:
                            raise ValueError(f"P7_CONTINUOUS_WEIGHT_SHAPE_MISMATCH:{group}")
                        mixed[group] = (
                            (1.0 - probability) * group_states[:, 0, :]
                            + probability * group_states[:, 1, :]
                        )
                    else:
                        if probability.shape != group_states.shape[:2]:
                            raise ValueError(f"P7_CATEGORICAL_WEIGHT_SHAPE_MISMATCH:{group}")
                        mixed[group] = (probability.unsqueeze(-1) * group_states).sum(dim=1)
                return mixed

            def forward_from_features(
                self,
                h_x: Any,
                *,
                intervention_targets: Mapping[str, Any] | None = None,
                intervention_mask: Any | None = None,
            ) -> dict[str, Any]:
                generated = self.states_and_probabilities(h_x)
                predicted = generated["activated"]
                if intervention_targets is None and intervention_mask is None:
                    effective = predicted
                elif intervention_targets is not None and intervention_mask is not None:
                    effective = apply_intervention_weights(
                        predicted, intervention_targets, intervention_mask
                    )
                else:
                    raise ValueError("P7_PARTIAL_INTERVENTION_ARGUMENTS")
                mixed = self.mix_states(generated["states"], effective)
                flat = torch.cat(tuple(mixed.values()), dim=1)
                if flat.shape != (h_x.shape[0], 128):
                    raise ValueError("P7_FLAT_MIXED_EMBEDDING_SHAPE_MISMATCH")
                raw = self.task_head(flat)
                return {
                    **generated,
                    "canonical_vector": canonical_concept_vector(predicted),
                    "effective_weights": effective,
                    "mixed_embeddings": mixed,
                    "flat_mixed_embedding": flat,
                    "malignancy_raw_score": raw,
                }

            def forward(
                self,
                image: Any,
                *,
                intervention_targets: Mapping[str, Any] | None = None,
                intervention_mask: Any | None = None,
            ) -> dict[str, Any]:
                features = self.relu(self.encoder(image))
                h_x = features.mean(dim=(2, 3, 4))
                result = self.forward_from_features(
                    h_x,
                    intervention_targets=intervention_targets,
                    intervention_mask=intervention_mask,
                )
                result["encoder_feature_h_x"] = h_x
                return result

        return Model(encoder, components)


def build_initialized_model(
    scientific_config: Mapping[str, Any],
    split: Mapping[str, Any],
    encoder_artifact_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    encoder = build_encoder()
    encoder_hash = load_shared_encoder_initialization(
        encoder, encoder_artifact_path, scientific_config, split
    )
    validated = validate_encoder_artifact(
        Path(encoder_artifact_path), scientific_config, split
    )
    fold_seed = int(validated["metadata"]["fold_seed"])
    components, metadata = build_deterministic_cem_components(fold_seed)
    if encoder_state_sha256(encoder.state_dict()) != encoder_hash:
        raise ValueError("P7_ENCODER_HASH_CHANGED_BEFORE_TRAINING")
    model = MixedTypeCEM.build(encoder, components)
    return model, {
        "fold_seed": fold_seed,
        "encoder_initialization_sha256": encoder_hash,
        "encoder_artifact_file_sha256": sha256_file(encoder_artifact_path),
        **metadata,
    }


def cem_losses(outputs: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    """Compute task, equal-group concept, and preregistered total CEM loss."""
    torch = _torch()
    raw = outputs["malignancy_raw_score"]
    malignancy = targets["malignancy"]
    if raw.shape != malignancy.shape:
        raise ValueError("P7_TASK_TARGET_SHAPE_MISMATCH")
    task = torch.nn.functional.mse_loss(raw, malignancy, reduction="mean")
    sums, batch_size = concept_group_loss_sums(outputs, targets["concepts"])
    group_losses = OrderedDict((group, value / batch_size) for group, value in sums.items())
    concept = torch.stack(tuple(group_losses.values())).mean()
    total = task + 0.01 * concept
    return {
        "task_loss": task,
        "concept_loss": concept,
        "total_loss": total,
        "group_losses": group_losses,
        "batch_size": batch_size,
    }


def task_predictions_and_contributions(model: Any, outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact eight-group normalized and rating-scale decompositions."""
    torch = _torch()
    mixed = outputs["mixed_embeddings"]
    if tuple(mixed) != CONCEPT_GROUP_ORDER:
        raise ValueError("P7_MIXED_EMBEDDING_ORDER_MISMATCH")
    weight = model.task_head.weight.reshape(-1)
    bias = model.task_head.bias.reshape(1)
    if weight.numel() != 128 or bias.numel() != 1:
        raise ValueError("P7_TASK_HEAD_SHAPE_MISMATCH")
    raw_contributions: OrderedDict[str, Any] = OrderedDict()
    for index, group in enumerate(CONCEPT_GROUP_ORDER):
        embedding = mixed[group]
        if embedding.ndim != 2 or embedding.shape[1] != 16:
            raise ValueError(f"P7_MIXED_EMBEDDING_SHAPE_MISMATCH:{group}")
        group_weight = weight[index * 16 : (index + 1) * 16]
        raw_contributions[group] = (embedding * group_weight).sum(dim=1)
    raw = outputs["malignancy_raw_score"].reshape(-1)
    reconstruction = bias + torch.stack(tuple(raw_contributions.values()), dim=1).sum(dim=1)
    error = float((reconstruction - raw).abs().max().detach().cpu())
    if error > 1e-6:
        raise ValueError(f"P7_NORMALIZED_CONTRIBUTION_RECONSTRUCTION_FAILED:{error}")
    rating_contributions = OrderedDict(
        (group, 4.0 * value) for group, value in raw_contributions.items()
    )
    rating_bias = 1.0 + 4.0 * bias
    rating = 1.0 + 4.0 * raw
    rating_reconstruction = rating_bias + torch.stack(
        tuple(rating_contributions.values()), dim=1
    ).sum(dim=1)
    rating_error = float((rating_reconstruction - rating).abs().max().detach().cpu())
    if rating_error > 1e-6:
        raise ValueError(f"P7_RATING_CONTRIBUTION_RECONSTRUCTION_FAILED:{rating_error}")
    return {
        "malignancy_raw_score": raw,
        "malignancy_score_normalized": raw,
        "malignancy_score_1_to_5": rating,
        "raw_bias": bias,
        "raw_group_contributions": raw_contributions,
        "rating_scale_bias": rating_bias,
        "rating_group_contributions": rating_contributions,
        "normalized_reconstruction_max_abs_error": error,
        "rating_reconstruction_max_abs_error": rating_error,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("overfit-check", "preflight", "train", "evaluate-test", "verify"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _parser().parse_args(argv)
    raise RuntimeError("P7_LIFECYCLE_NOT_IMPLEMENTED")


if __name__ == "__main__":
    raise SystemExit(main())
