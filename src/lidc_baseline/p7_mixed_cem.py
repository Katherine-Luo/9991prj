"""Train and verify the Baseline-v2 project-specific mixed-type CEM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p4_prepare import (
    build_encoder,
    canonical_json_bytes,
    encoder_state_sha256,
    load_shared_encoder_initialization,
    sha256_bytes,
    sha256_file,
    validate_encoder_artifact,
)
from lidc_baseline.p5_blackbox import (
    EXPECTED_FOLD_TEST_COUNTS,
    ValidationMSEPlateau,
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
    _atomic_torch_save,
    _loader,
    _optimizer,
    _prepare_sources,
    _runtime_environment,
    _scheduler,
    capture_rng_state,
    checkpoint_improves,
    configure_fp32_determinism,
    epoch_uid_order,
    exclusive_fold_lifecycle_lock,
    regression_metrics,
    require_formal_gpu_for_cuda,
    reproducibility_provenance,
    restore_rng_state,
    seed_training,
    serialized_float_consistent,
)
from lidc_baseline.p6_standard_cbm import (
    CATEGORICAL_CONCEPTS,
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    CONTINUOUS_CONCEPTS,
    ConceptROIDataset,
    ConceptRecord,
    _targets_to_device,
    build_partition_concept_records,
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
STATE_MIXTURE_NUMERIC_SCHEMA = "fp32_serialized_weighted_sum_v1"
STATE_MIXTURE_ABSOLUTE_FLOOR = 1e-6
STATE_MIXTURE_FLOAT32_OPERATION_FACTOR = 16.0
RECOVERY_BUG_ID = "BUG-P7-001"
RECOVERY_INVALIDATED_STATUS = "INVALIDATED_PRECOMMIT_TEST_ATTEMPT"
RECOVERY_MODEL_CHANGE_STATUS = "NONE"
RECOVERY_APPROVED_BEST_CHECKPOINT_SHA256 = (
    "e245f06f4d001a1450a35bdfd87dd053d0210bc8b5fc942194a6a6cd8e641a07"
)
RECOVERY_APPROVED_ORIGINAL_CLAIM_SHA256 = (
    "055125afba805186f3b1b282270cdd3ef56958255df4ae2942b1d3d4303bb091"
)


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


def _ordered_records(
    records: Sequence[ConceptRecord],
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
) -> list[ConceptRecord]:
    by_uid = {record.nodule_uid: record for record in records}
    if len(by_uid) != len(records):
        raise ValueError("P7_DUPLICATE_RECORD_UID")
    return [
        by_uid[uid]
        for uid in epoch_uid_order(by_uid, base_seed, fold_index, epoch_index)
    ]


def run_cem_epoch(
    model: Any,
    records: Sequence[ConceptRecord],
    device: Any,
    *,
    optimizer: Any | None,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Run one full partition with sample-weighted task and group losses."""
    torch = _torch()
    training = optimizer is not None
    ordered = (
        _ordered_records(
            records,
            base_seed=base_seed,
            fold_index=fold_index,
            epoch_index=epoch_index,
        )
        if training
        else sorted(records, key=lambda record: record.nodule_uid)
    )
    if not ordered:
        raise ValueError("P7_EMPTY_PARTITION")
    dataset = ConceptROIDataset.build(
        ordered,
        training=training,
        base_seed=base_seed,
        fold_index=fold_index,
        epoch_index=epoch_index,
    )
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.train(training)
    task_squared_error_sum = 0.0
    group_sums = OrderedDict((group, 0.0) for group in CONCEPT_GROUP_ORDER)
    decision_counts = OrderedDict((group, 0) for group in CONCEPT_GROUP_ORDER)
    sample_weighted_counts = OrderedDict((group, 0) for group in CONCEPT_GROUP_ORDER)
    sample_count = 0
    batch_count = 0
    observed_uids: list[str] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device=device, dtype=torch.float32)
            concepts = _targets_to_device(batch["targets"], device)
            malignancy = batch["target_normalized"].to(
                device=device, dtype=torch.float32
            )
            if training:
                mask = batch_shared_intervention_mask(
                    base_seed=base_seed,
                    fold_index=fold_index,
                    epoch_index=epoch_index,
                    batch_index=batch_index,
                ).to(device=device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    image,
                    intervention_targets=concepts,
                    intervention_mask=mask,
                )
            else:
                mask = torch.zeros(len(CONCEPT_GROUP_ORDER), dtype=torch.bool, device=device)
                outputs = model(image)
            losses = cem_losses(
                outputs,
                {"concepts": concepts, "malignancy": malignancy},
            )
            if not torch.isfinite(losses["total_loss"]):
                raise ValueError("P7_NONFINITE_TOTAL_LOSS")
            if training:
                losses["total_loss"].backward()
                optimizer.step()
            batch_samples = int(malignancy.shape[0])
            task_squared_error_sum += float(
                torch.nn.functional.mse_loss(
                    outputs["malignancy_raw_score"].detach(),
                    malignancy,
                    reduction="sum",
                ).cpu()
            )
            sums, observed_batch_size = concept_group_loss_sums(outputs, concepts)
            if observed_batch_size != batch_samples:
                raise ValueError("P7_BATCH_SAMPLE_COUNT_MISMATCH")
            for index, group in enumerate(CONCEPT_GROUP_ORDER):
                group_sums[group] += float(sums[group].detach().cpu())
                if training and bool(mask[index]):
                    decision_counts[group] += 1
                    sample_weighted_counts[group] += batch_samples
            sample_count += batch_samples
            batch_count += 1
            observed_uids.extend(map(str, batch["nodule_uid"]))
    expected_uids = [record.nodule_uid for record in ordered]
    if observed_uids != expected_uids or len(set(observed_uids)) != len(ordered):
        raise ValueError("P7_PARTITION_SAMPLE_COVERAGE_MISMATCH")
    if sample_count != len(ordered):
        raise ValueError("P7_PARTITION_SAMPLE_COUNT_MISMATCH")
    group_losses = OrderedDict(
        (group, value / sample_count) for group, value in group_sums.items()
    )
    task_loss = task_squared_error_sum / sample_count
    concept_loss = float(np.mean(tuple(group_losses.values())))
    total_loss = task_loss + 0.01 * concept_loss
    return {
        "task_loss": task_loss,
        "concept_loss": concept_loss,
        "total_loss": total_loss,
        "group_losses": group_losses,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "nodule_set_sha256": sha256_bytes(canonical_json_bytes(sorted(observed_uids))),
        "intervention_decision_counts": decision_counts,
        "intervention_sample_weighted_counts": sample_weighted_counts,
    }


def _load_sources(
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    str,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = (
        _prepare_sources(
            scientific_config_path,
            execution_config_path,
            manifest_path,
            roi_index_path,
            fold_index,
        )
    )
    p7_config, p7_hash = validate_p7_execution_config(p7_config_path)
    if p7_config["scientific_config"]["sha256"] != compute_config_sha256(scientific):
        raise ValueError("P7_SCIENTIFIC_CONFIG_REFERENCE_MISMATCH")
    if p7_config["common_execution_profile"]["resolved_sha256"] != execution_hash:
        raise ValueError("P7_COMMON_EXECUTION_REFERENCE_MISMATCH")
    return (
        scientific,
        execution,
        execution_hash,
        p7_config,
        p7_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    )


def _provenance(
    scientific: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_hash: str,
    p7_hash: str,
    split: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": scientific["protocol"]["version"],
        "scientific_config_sha256": compute_config_sha256(scientific),
        "execution_config_sha256": execution_hash,
        "p7_execution_config_sha256": p7_hash,
        "execution_profile_id": execution["execution_profile"]["profile_id"],
        "formal_gpu_model": execution["execution_profile"]["formal_gpu_model"],
        **reproducibility_provenance(execution),
        "split_sha256": split["split_sha256"],
        "fold_index": int(split["fold_index"]),
        "model": MODEL_NAME,
        "method_label": METHOD_LABEL,
        "task_output": "unconstrained_linear_raw_score",
        "task_loss": "mean_squared_error_on_normalized_target",
        "concept_loss": "equal_mean_of_six_mse_and_two_soft_cross_entropy_groups",
        "total_loss": "task_loss_plus_0.01_times_concept_loss",
        **dict(initialization),
    }


def run_directory(fold_index: int, root: str | Path = "runs/baseline_v2/cem") -> Path:
    return Path(root) / f"fold_{fold_index}"


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: ValidationMSEPlateau,
    *,
    epoch_index: int,
    validation_total_loss: float,
    best_epoch_index: int,
    best_validation_total_loss: float,
    provenance: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "epoch_index": int(epoch_index),
        "validation_total_loss": float(validation_total_loss),
        "best_epoch_index": int(best_epoch_index),
        "best_validation_total_loss": float(best_validation_total_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "history": list(history),
    }


def _load_checkpoint(path: Path, provenance: Mapping[str, Any]) -> dict[str, Any]:
    payload = _torch().load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("P7_CHECKPOINT_SCHEMA_MISMATCH")
    if payload.get("provenance") != dict(provenance):
        raise ValueError("P7_CHECKPOINT_PROVENANCE_MISMATCH")
    return payload


def _history_row(
    *,
    epoch_index: int,
    train_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    learning_rate_start: float,
    optimizer: Any,
    scheduler: ValidationMSEPlateau,
    scheduler_decayed: bool,
    epoch_seconds: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch_index": epoch_index,
        "train_task_loss": train_report["task_loss"],
        "train_concept_loss": train_report["concept_loss"],
        "train_total_loss": train_report["total_loss"],
        "validation_task_loss": validation_report["task_loss"],
        "validation_concept_loss": validation_report["concept_loss"],
        "validation_total_loss": validation_report["total_loss"],
        "learning_rate_start": learning_rate_start,
        "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
        "scheduler_decayed": bool(scheduler_decayed),
        "scheduler_best": scheduler.best,
        "scheduler_bad_epoch_counter": scheduler.bad_epoch_counter,
        "train_sample_count": int(train_report["sample_count"]),
        "validation_sample_count": int(validation_report["sample_count"]),
        "train_batch_count": int(train_report["batch_count"]),
        "train_nodule_set_sha256": train_report["nodule_set_sha256"],
        "validation_nodule_set_sha256": validation_report["nodule_set_sha256"],
        "epoch_seconds": epoch_seconds,
    }
    for group in CONCEPT_GROUP_ORDER:
        row[f"train_{group}_loss"] = train_report["group_losses"][group]
        row[f"validation_{group}_loss"] = validation_report["group_losses"][group]
        row[f"intervention_{group}_decisions"] = train_report[
            "intervention_decision_counts"
        ][group]
        row[f"intervention_{group}_sample_weighted"] = train_report[
            "intervention_sample_weighted_counts"
        ][group]
    return row


def _validate_resume_history(
    path: Path,
    expected: Sequence[Mapping[str, Any]],
) -> None:
    frame = pd.read_csv(path)
    if len(frame) != len(expected) or frame["epoch_index"].tolist() != list(
        range(len(expected))
    ):
        raise ValueError("P7_RESUME_HISTORY_MISMATCH")
    if not expected or set(frame.columns) != set(expected[0]):
        raise ValueError("P7_RESUME_HISTORY_SCHEMA_MISMATCH")
    for row_index, expected_row in enumerate(expected):
        for key, expected_value in expected_row.items():
            observed = frame.iloc[row_index][key]
            if isinstance(expected_value, (bool, np.bool_)):
                if bool(observed) is not bool(expected_value):
                    raise ValueError("P7_RESUME_HISTORY_VALUE_MISMATCH")
            elif isinstance(expected_value, (int, float, np.integer, np.floating)):
                if not serialized_float_consistent(float(observed), float(expected_value)):
                    raise ValueError("P7_RESUME_HISTORY_VALUE_MISMATCH")
            elif str(observed) != str(expected_value):
                raise ValueError("P7_RESUME_HISTORY_VALUE_MISMATCH")


def _validate_completion_artifacts(
    output: Path,
    completion: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if any(completion.get(key) != value for key, value in provenance.items()):
        raise ValueError("P7_COMPLETION_PROVENANCE_MISMATCH")
    if int(completion.get("epochs_completed", -1)) != 80:
        raise ValueError("P7_COMPLETION_EPOCH_MISMATCH")
    for filename, key in (
        ("best.pt", "best_checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
        ("history.csv", "history_sha256"),
        ("runtime.json", "runtime_sha256"),
    ):
        path = output / filename
        if not path.is_file() or completion.get(key) != sha256_file(path):
            raise ValueError(f"P7_ARTIFACT_HASH_MISMATCH:{filename}")


def train_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    resume: bool,
    _stop_after_epoch_for_test: int | None = None,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p7_lifecycle.lock"):
        return _train_fold_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p7_config_path=p7_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            device_name=device_name,
            num_workers=num_workers,
            output_root=output_root,
            resume=resume,
            _stop_after_epoch_for_test=_stop_after_epoch_for_test,
        )


def _train_fold_locked(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    resume: bool,
    _stop_after_epoch_for_test: int | None,
) -> dict[str, Any]:
    torch = _torch()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    (
        scientific,
        execution,
        execution_hash,
        _p7_config,
        p7_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p7_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    provenance = _provenance(
        scientific, execution, execution_hash, p7_hash, split, initialization
    )
    output = run_directory(fold_index, output_root)
    output.mkdir(parents=True, exist_ok=True)
    complete_path = output / "training_complete.json"
    if complete_path.exists():
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        _validate_completion_artifacts(output, completion, provenance)
        _validate_history_and_runtime(
            pd.read_csv(output / "history.csv"),
            json.loads((output / "runtime.json").read_text(encoding="utf-8")),
            split,
            provenance,
        )
        if not resume:
            raise FileExistsError("P7_TRAINING_ALREADY_COMPLETE")
        return completion
    model.to(device)
    seed_training(int(initialization["fold_seed"]))
    optimizer = _optimizer(model, execution)
    scheduler = _scheduler(optimizer, execution)
    train_records = build_partition_concept_records(
        manifest, roi_index, split, "train", roi_index_path
    )
    validation_records = build_partition_concept_records(
        manifest, roi_index, split, "validation", roi_index_path
    )
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    history_path = output / "history.csv"
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    best_validation = math.inf
    if resume:
        if not last_path.exists() or not history_path.exists():
            raise FileNotFoundError("P7_RESUME_ARTIFACT_MISSING")
        payload = _load_checkpoint(last_path, provenance)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
        history = list(payload["history"])
        start_epoch = int(payload["epoch_index"]) + 1
        best_epoch = int(payload["best_epoch_index"])
        best_validation = float(payload["best_validation_total_loss"])
        if len(history) != start_epoch:
            raise ValueError("P7_RESUME_HISTORY_MISMATCH")
        _validate_resume_history(history_path, history)
    elif any(path.exists() for path in (last_path, best_path, history_path)):
        raise FileExistsError("P7_RUN_EXISTS_USE_RESUME_OR_INVALIDATE")
    epochs = 80
    batch_size = int(execution["project_preregistered"]["batching"]["micro_batch_size"])
    base_seed = int(scientific["reproducibility"]["base_seed"])
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_report = run_cem_epoch(
            model,
            train_records,
            device,
            optimizer=optimizer,
            base_seed=base_seed,
            fold_index=fold_index,
            epoch_index=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        validation_report = run_cem_epoch(
            model,
            validation_records,
            device,
            optimizer=None,
            base_seed=base_seed,
            fold_index=fold_index,
            epoch_index=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        validation_total = float(validation_report["total_loss"])
        improved = checkpoint_improves(validation_total, best_validation)
        if improved:
            best_validation = validation_total
            best_epoch = epoch
        decayed = scheduler.step(validation_total)
        row = _history_row(
            epoch_index=epoch,
            train_report=train_report,
            validation_report=validation_report,
            learning_rate_start=current_lr,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_decayed=decayed,
            epoch_seconds=time.monotonic() - epoch_started,
        )
        history.append(row)
        _atomic_csv(history_path, history, list(row))
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch_index=epoch,
            validation_total_loss=validation_total,
            best_epoch_index=best_epoch,
            best_validation_total_loss=best_validation,
            provenance=provenance,
            history=history,
        )
        _atomic_torch_save(last_path, payload)
        if improved:
            _atomic_torch_save(best_path, payload)
        print(
            canonical_json_bytes(
                {"event": "P7_EPOCH_COMPLETE", "fold_index": fold_index, **row}
            ).decode("utf-8").strip(),
            flush=True,
        )
        if _stop_after_epoch_for_test is not None and epoch == _stop_after_epoch_for_test:
            return {
                **provenance,
                "status": "INTERRUPTED_AT_EPOCH_BOUNDARY_FOR_TEST",
                "epoch_index": epoch,
                "last_checkpoint_sha256": sha256_file(last_path),
            }
    if len(history) != epochs or not best_path.is_file():
        raise ValueError("P7_TRAINING_INCOMPLETE")
    expected_train = int(split["partitions"]["train"]["summary"]["nodules"])
    if any(int(row["train_sample_count"]) != expected_train for row in history):
        raise ValueError("P7_EPOCH_TRAIN_COVERAGE_MISMATCH")
    runtime = {
        **provenance,
        **_runtime_environment(device),
        "epochs_this_invocation": epochs - start_epoch,
        "epochs_total": epochs,
        "wall_seconds_this_invocation": time.monotonic() - started,
        "sum_epoch_seconds": float(sum(float(row["epoch_seconds"]) for row in history)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else None,
    }
    _atomic_json(output / "runtime.json", runtime)
    completion = {
        **provenance,
        "status": "TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "epochs_completed": epochs,
        "best_epoch_index": best_epoch,
        "best_validation_total_loss": best_validation,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history_sha256": sha256_file(history_path),
        "runtime_sha256": sha256_file(output / "runtime.json"),
        "test_evaluated": False,
    }
    _atomic_json(complete_path, completion)
    return completion


def _prediction_rows(
    model: Any,
    records: Sequence[ConceptRecord],
    device: Any,
    *,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    torch = _torch()
    ordered = sorted(records, key=lambda record: record.nodule_uid)
    dataset = ConceptROIDataset.build(
        ordered,
        training=False,
        base_seed=0,
        fold_index=0,
        epoch_index=0,
    )
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers)
    by_uid = {record.nodule_uid: record for record in ordered}
    model.eval()
    result: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            concepts = _targets_to_device(batch["targets"], device)
            outputs = model(image)
            report = task_predictions_and_contributions(model, outputs)
            raw = report["malignancy_raw_score"].detach().cpu().numpy()
            rating = report["malignancy_score_1_to_5"].detach().cpu().numpy()
            raw_bias = float(report["raw_bias"].detach().cpu().reshape(-1)[0])
            rating_bias = float(
                report["rating_scale_bias"].detach().cpu().reshape(-1)[0]
            )
            for sample_index, uid in enumerate(map(str, batch["nodule_uid"])):
                record = by_uid[uid]
                row: dict[str, Any] = {
                    "nodule_uid": uid,
                    "patient_key": record.patient_key,
                    "target_normalized": record.target_normalized,
                    "target_1_to_5": record.target_1_to_5,
                    "malignancy_raw_score": float(raw[sample_index]),
                    "malignancy_score_normalized": float(raw[sample_index]),
                    "malignancy_score_1_to_5": float(rating[sample_index]),
                    "extreme_binary_eligible": record.extreme_binary_eligible,
                    "extreme_binary_label": record.extreme_binary_label,
                    "raw_bias": raw_bias,
                    "rating_scale_bias": rating_bias,
                    "normalized_reconstruction_max_abs_error": report[
                        "normalized_reconstruction_max_abs_error"
                    ],
                    "rating_reconstruction_max_abs_error": report[
                        "rating_reconstruction_max_abs_error"
                    ],
                }
                for group_index, group in enumerate(CONCEPT_GROUP_ORDER):
                    row[f"{group}_logits"] = json.dumps(
                        outputs["logits"][group][sample_index]
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist(),
                        separators=(",", ":"),
                    )
                    row[f"{group}_activated_prediction"] = json.dumps(
                        outputs["activated"][group][sample_index]
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist(),
                        separators=(",", ":"),
                    )
                    row[f"{group}_target"] = json.dumps(
                        concepts[group][sample_index]
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist(),
                        separators=(",", ":"),
                    )
                    row[f"{group}_states"] = json.dumps(
                        outputs["states"][group][sample_index]
                        .detach()
                        .cpu()
                        .tolist(),
                        separators=(",", ":"),
                    )
                    row[f"{group}_mixed_embedding"] = json.dumps(
                        outputs["mixed_embeddings"][group][sample_index]
                        .detach()
                        .cpu()
                        .reshape(-1)
                        .tolist(),
                        separators=(",", ":"),
                    )
                    row[f"{group}_valid_reader_count"] = record.valid_reader_counts[
                        group_index
                    ]
                    row[f"{group}_raw_contribution"] = float(
                        report["raw_group_contributions"][group][sample_index]
                        .detach()
                        .cpu()
                    )
                    row[f"{group}_rating_contribution"] = float(
                        report["rating_group_contributions"][group][sample_index]
                        .detach()
                        .cpu()
                    )
                row["internalStructure_modal_tie"] = record.categorical_ties[0]
                row["calcification_modal_tie"] = record.categorical_ties[1]
                result.append(row)
    if [row["nodule_uid"] for row in result] != [record.nodule_uid for record in ordered]:
        raise ValueError("P7_TEST_PREDICTION_ORDER_MISMATCH")
    return result


def _json_vector(value: Any, size: int, code: str) -> np.ndarray:
    try:
        parsed = json.loads(str(value))
        array = np.asarray(parsed, dtype=np.float64)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(code) from error
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(code)
    return array.reshape(-1)


def _strict_bool(value: Any, code: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(code)
    return bool(value)


def _strict_positive_integer(value: Any, code: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(code)
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 1.0 or numeric != math.floor(numeric):
        raise ValueError(code)
    return int(numeric)


def _distribution_has_modal_tie(distribution: np.ndarray) -> bool:
    maximum = float(distribution.max())
    return bool(
        np.count_nonzero(np.isclose(distribution, maximum, atol=1e-12, rtol=0.0))
        > 1
    )


def _state_mixture_diagnostic(
    mixed: np.ndarray,
    activated: np.ndarray,
    states: np.ndarray,
    *,
    group: str,
    anonymous_row_index: int,
) -> dict[str, Any]:
    """Validate serialized FP32 mixture values with an explicit roundoff budget."""
    size = CONCEPT_OUTPUT_SIZES[group]
    if group in CATEGORICAL_CONCEPTS:
        products = activated.reshape(size, 1) * states.reshape(size, EMBEDDING_SIZE)
    else:
        reshaped_states = states.reshape(CONTINUOUS_STATE_COUNT, EMBEDDING_SIZE)
        products = np.stack(
            (
                (1.0 - activated[0]) * reshaped_states[0],
                activated[0] * reshaped_states[1],
            )
        )
    expected = products.sum(axis=0, dtype=np.float64)
    absolute_error = np.abs(mixed - expected)
    magnitude = np.maximum(np.abs(products).sum(axis=0), 1.0)
    allowed = (
        STATE_MIXTURE_ABSOLUTE_FLOOR
        + STATE_MIXTURE_FLOAT32_OPERATION_FACTOR
        * np.finfo(np.float32).eps
        * magnitude
    )
    maximum_index = int(np.argmax(absolute_error))
    report = {
        "schema": STATE_MIXTURE_NUMERIC_SCHEMA,
        "anonymous_row_index": int(anonymous_row_index),
        "group": group,
        "dimension_index": maximum_index,
        "expected_value": float(expected[maximum_index]),
        "actual_value": float(mixed[maximum_index]),
        "maximum_absolute_error": float(absolute_error[maximum_index]),
        "allowed_absolute_error": float(allowed[maximum_index]),
    }
    if np.any(absolute_error > allowed):
        raise ValueError(
            "P7_TEST_STATE_MIXTURE_MISMATCH:"
            + canonical_json_bytes(report).decode("utf-8")
        )
    return report


def _validate_test_predictions(
    frame: pd.DataFrame,
    records: Sequence[ConceptRecord],
    row_provenance: Mapping[str, Any],
    model: Any | None = None,
) -> dict[str, Any]:
    expected = {record.nodule_uid: record for record in records}
    required = {
        "nodule_uid",
        "patient_key",
        "target_normalized",
        "target_1_to_5",
        "malignancy_raw_score",
        "malignancy_score_normalized",
        "malignancy_score_1_to_5",
        "extreme_binary_eligible",
        "extreme_binary_label",
        "raw_bias",
        "rating_scale_bias",
        "internalStructure_modal_tie",
        "calcification_modal_tie",
        *row_provenance,
    }
    for group in CONCEPT_GROUP_ORDER:
        required.update(
            {
                f"{group}_logits",
                f"{group}_activated_prediction",
                f"{group}_target",
                f"{group}_states",
                f"{group}_mixed_embedding",
                f"{group}_valid_reader_count",
                f"{group}_raw_contribution",
                f"{group}_rating_contribution",
            }
        )
    if not required <= set(frame.columns):
        raise ValueError("P7_TEST_PREDICTION_SCHEMA_MISMATCH")
    if len(frame) != len(expected) or frame["nodule_uid"].astype(str).duplicated().any():
        raise ValueError("P7_TEST_PREDICTION_COUNT_MISMATCH")
    if set(frame["nodule_uid"].astype(str)) != set(expected):
        raise ValueError("P7_TEST_PREDICTION_UID_SET_MISMATCH")
    for key, expected_value in row_provenance.items():
        if not all(value == expected_value for value in frame[key].tolist()):
            raise ValueError(f"P7_TEST_PROVENANCE_MISMATCH:{key}")
    by_uid = frame.set_index(frame["nodule_uid"].astype(str), drop=False)
    task_weight = (
        model.task_head.weight.detach().cpu().reshape(-1).numpy()
        if model is not None
        else None
    )
    task_bias = (
        float(model.task_head.bias.detach().cpu().reshape(-1)[0])
        if model is not None
        else None
    )
    maximum_mixture_error = 0.0
    maximum_allowed_mixture_error = 0.0
    for anonymous_row_index, (uid, record) in enumerate(expected.items()):
        row = by_uid.loc[uid]
        if str(row["patient_key"]) != record.patient_key:
            raise ValueError("P7_TEST_PATIENT_KEY_MISMATCH")
        if not math.isclose(
            float(row["target_normalized"]),
            record.target_normalized,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(row["target_1_to_5"]),
            record.target_1_to_5,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("P7_TEST_TARGET_MISMATCH")
        if _strict_bool(
            row["extreme_binary_eligible"], "P7_TEST_EXTREME_ELIGIBILITY_INVALID"
        ) is not record.extreme_binary_eligible:
            raise ValueError("P7_TEST_EXTREME_ELIGIBILITY_MISMATCH")
        observed_extreme_label = row["extreme_binary_label"]
        if record.extreme_binary_label is None:
            if not pd.isna(observed_extreme_label):
                raise ValueError("P7_TEST_EXTREME_LABEL_MISMATCH")
        elif (
            isinstance(observed_extreme_label, (bool, np.bool_))
            or not isinstance(
                observed_extreme_label,
                (int, float, np.integer, np.floating),
            )
            or not math.isfinite(float(observed_extreme_label))
            or float(observed_extreme_label) != float(record.extreme_binary_label)
        ):
            raise ValueError("P7_TEST_EXTREME_LABEL_MISMATCH")
        raw = float(row["malignancy_raw_score"])
        if not math.isfinite(raw) or float(row["malignancy_score_normalized"]) != raw:
            raise ValueError("P7_TEST_SCORE_MISMATCH")
        if not math.isclose(
            float(row["malignancy_score_1_to_5"]),
            1.0 + 4.0 * raw,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("P7_TEST_RATING_SCALE_MISMATCH")
        contribution_sum = float(row["raw_bias"])
        rating_sum = float(row["rating_scale_bias"])
        if not math.isclose(
            float(row["rating_scale_bias"]),
            1.0 + 4.0 * float(row["raw_bias"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("P7_TEST_BIAS_SCALE_MISMATCH")
        if task_bias is not None and not math.isclose(
            float(row["raw_bias"]), task_bias, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError("P7_TEST_TASK_BIAS_MISMATCH")
        continuous_targets = dict(
            zip(CONTINUOUS_CONCEPTS, record.continuous_targets, strict=True)
        )
        for group_index, group in enumerate(CONCEPT_GROUP_ORDER):
            size = CONCEPT_OUTPUT_SIZES[group]
            states_size = size * 16 if group in CATEGORICAL_CONCEPTS else 32
            logits = _json_vector(row[f"{group}_logits"], size, "P7_TEST_LOGIT_INVALID")
            activated = _json_vector(
                row[f"{group}_activated_prediction"],
                size,
                "P7_TEST_ACTIVATION_INVALID",
            )
            target = _json_vector(row[f"{group}_target"], size, "P7_TEST_TARGET_INVALID")
            states = _json_vector(
                row[f"{group}_states"], states_size, "P7_TEST_STATE_INVALID"
            )
            mixed = _json_vector(
                row[f"{group}_mixed_embedding"], 16, "P7_TEST_MIXED_INVALID"
            )
            expected_target = (
                np.asarray(record.internal_structure_target, dtype=np.float64)
                if group == "internalStructure"
                else np.asarray(record.calcification_target, dtype=np.float64)
                if group == "calcification"
                else np.asarray([continuous_targets[group]], dtype=np.float64)
            )
            if not np.allclose(target, expected_target, atol=1e-7, rtol=0.0):
                raise ValueError("P7_TEST_CONCEPT_TARGET_MISMATCH")
            if _strict_positive_integer(
                row[f"{group}_valid_reader_count"],
                "P7_TEST_VALID_READER_COUNT_INVALID",
            ) != record.valid_reader_counts[group_index]:
                raise ValueError("P7_TEST_VALID_READER_COUNT_MISMATCH")
            if group in CATEGORICAL_CONCEPTS:
                if (
                    np.any(activated < 0.0)
                    or np.any(activated > 1.0)
                    or not np.isclose(activated.sum(), 1.0, atol=1e-6, rtol=0.0)
                    or not np.isclose(target.sum(), 1.0, atol=1e-6, rtol=0.0)
                ):
                    raise ValueError("P7_TEST_CATEGORICAL_PROBABILITY_INVALID")
                shifted = logits - logits.max()
                expected_activation = np.exp(shifted) / np.exp(shifted).sum()
                if not np.allclose(activated, expected_activation, atol=1e-6, rtol=0.0):
                    raise ValueError("P7_TEST_LOGIT_ACTIVATION_MISMATCH")
            else:
                expected_activation = 1.0 / (1.0 + np.exp(-logits))
                if not np.allclose(activated, expected_activation, atol=1e-6, rtol=0.0):
                    raise ValueError("P7_TEST_LOGIT_ACTIVATION_MISMATCH")
            mixture_diagnostic = _state_mixture_diagnostic(
                mixed,
                activated,
                states,
                group=group,
                anonymous_row_index=anonymous_row_index,
            )
            maximum_mixture_error = max(
                maximum_mixture_error,
                float(mixture_diagnostic["maximum_absolute_error"]),
            )
            maximum_allowed_mixture_error = max(
                maximum_allowed_mixture_error,
                float(mixture_diagnostic["allowed_absolute_error"]),
            )
            raw_contribution = float(row[f"{group}_raw_contribution"])
            rating_contribution = float(row[f"{group}_rating_contribution"])
            if not math.isclose(
                rating_contribution,
                4.0 * raw_contribution,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("P7_TEST_CONTRIBUTION_SCALE_MISMATCH")
            if task_weight is not None:
                expected_contribution = float(
                    np.dot(mixed, task_weight[group_index * 16 : (group_index + 1) * 16])
                )
                if not math.isclose(
                    raw_contribution,
                    expected_contribution,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError("P7_TEST_CONTRIBUTION_WEIGHT_MISMATCH")
            contribution_sum += raw_contribution
            rating_sum += rating_contribution
        internal_tie = _distribution_has_modal_tie(
            np.asarray(record.internal_structure_target, dtype=np.float64)
        )
        calcification_tie = _distribution_has_modal_tie(
            np.asarray(record.calcification_target, dtype=np.float64)
        )
        if record.categorical_ties != (internal_tie, calcification_tie):
            raise ValueError("P7_RECORD_TIE_FLAG_SEMANTIC_MISMATCH")
        if _strict_bool(
            row["internalStructure_modal_tie"], "P7_TEST_TIE_FLAG_INVALID"
        ) is not internal_tie:
            raise ValueError("P7_TEST_TIE_FLAG_MISMATCH")
        if _strict_bool(
            row["calcification_modal_tie"], "P7_TEST_TIE_FLAG_INVALID"
        ) is not calcification_tie:
            raise ValueError("P7_TEST_TIE_FLAG_MISMATCH")
        if not math.isclose(contribution_sum, raw, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("P7_TEST_NORMALIZED_RECONSTRUCTION_MISMATCH")
        if not math.isclose(
            rating_sum, 1.0 + 4.0 * raw, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("P7_TEST_RATING_RECONSTRUCTION_MISMATCH")
    return {
        "state_mixture_numeric_schema": STATE_MIXTURE_NUMERIC_SCHEMA,
        "state_mixture_maximum_absolute_error": maximum_mixture_error,
        "state_mixture_maximum_allowed_absolute_error": maximum_allowed_mixture_error,
    }


def _validate_evaluation_artifacts(
    *,
    output: Path,
    evaluation: Mapping[str, Any],
    claim: Mapping[str, Any],
    completion: Mapping[str, Any],
    provenance: Mapping[str, Any],
    records: Sequence[ConceptRecord],
    row_provenance: Mapping[str, Any],
    model: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    claim_path = output / "test_claim.json"
    predictions_path = output / "test_predictions.parquet"
    metrics_path = output / "metrics.json"
    if json.loads(claim_path.read_text(encoding="utf-8")) != dict(claim):
        raise ValueError("P7_TEST_CLAIM_MISMATCH")
    if evaluation.get("status") != "TEST_EVALUATED_ONCE":
        raise ValueError("P7_TEST_EVALUATION_STATUS_MISMATCH")
    if any(evaluation.get(key) != value for key, value in provenance.items()):
        raise ValueError("P7_TEST_EVALUATION_PROVENANCE_MISMATCH")
    if evaluation.get("best_checkpoint_sha256") != completion.get(
        "best_checkpoint_sha256"
    ):
        raise ValueError("P7_TEST_EVALUATION_CHECKPOINT_MISMATCH")
    if evaluation.get("test_claim_sha256") != sha256_file(claim_path):
        raise ValueError("P7_TEST_CLAIM_HASH_MISMATCH")
    if evaluation.get("test_predictions_sha256") != sha256_file(predictions_path):
        raise ValueError("P7_TEST_PREDICTIONS_HASH_MISMATCH")
    if evaluation.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("P7_TEST_METRICS_HASH_MISMATCH")
    frame = pd.read_parquet(predictions_path)
    mixture_diagnostics = _validate_test_predictions(
        frame, records, row_provenance, model
    )
    if int(evaluation.get("test_samples", -1)) != len(frame):
        raise ValueError("P7_TEST_EVALUATION_SAMPLE_COUNT_MISMATCH")
    metrics = regression_metrics(frame.to_dict("records"))
    stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics != stored_metrics:
        raise ValueError("P7_TEST_METRICS_MISMATCH")
    recovery_audit_name = evaluation.get("recovery_attempt_audit")
    if recovery_audit_name is None:
        expected_attempts = {
            "total_test_forward_attempts": 1,
            "invalidated_attempts": 0,
            "valid_committed_test_evaluations": 1,
            "test_driven_model_changes": RECOVERY_MODEL_CHANGE_STATUS,
        }
    else:
        if recovery_audit_name != "test_attempt_audit.json":
            raise ValueError("P7_TEST_RECOVERY_AUDIT_NAME_MISMATCH")
        audit_path = output / str(recovery_audit_name)
        if evaluation.get("recovery_attempt_audit_sha256") != sha256_file(audit_path):
            raise ValueError("P7_TEST_RECOVERY_AUDIT_HASH_MISMATCH")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_audit = {
            "schema_version": SCHEMA_VERSION,
            "bug_id": RECOVERY_BUG_ID,
            "fold_index": int(provenance["fold_index"]),
            "status": "RECOVERY_COMPLETE",
            "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
            "original_test_claim_sha256": evaluation[
                "original_test_claim_sha256"
            ],
            "recovery_authorization_sha256": evaluation[
                "recovery_authorization_sha256"
            ],
            "recovery_claim_sha256": evaluation["recovery_claim_sha256"],
            "test_predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
            "total_test_forward_attempts": 2,
            "invalidated_attempts": 1,
            "valid_committed_test_evaluations": 1,
            "test_driven_model_changes": RECOVERY_MODEL_CHANGE_STATUS,
            "invalidated_attempt": {
                "attempt_index": 1,
                "status": RECOVERY_INVALIDATED_STATUS,
                "reason": "VERIFIER_IMPLEMENTATION_BUG",
                "mismatch_code": "P7_TEST_STATE_MIXTURE_MISMATCH",
                "committed_predictions": False,
                "committed_metrics": False,
                "committed_evaluation": False,
            },
            "valid_committed_attempt": {
                "attempt_index": 2,
                "status": "VALID_COMMITTED_TEST_EVALUATION",
                "same_best_checkpoint": True,
                "same_split": True,
                "same_scientific_config": True,
                "same_execution_config": True,
                "same_p7_config": True,
            },
            **mixture_diagnostics,
        }
        if audit != expected_audit:
            raise ValueError("P7_TEST_RECOVERY_AUDIT_CONTENT_MISMATCH")
        expected_attempts = {
            "total_test_forward_attempts": 2,
            "invalidated_attempts": 1,
            "valid_committed_test_evaluations": 1,
            "test_driven_model_changes": RECOVERY_MODEL_CHANGE_STATUS,
        }
        if completion.get("test_evaluated") is True and completion.get(
            "recovery_attempt_audit_sha256"
        ) != evaluation.get("recovery_attempt_audit_sha256"):
            raise ValueError("P7_COMPLETION_RECOVERY_AUDIT_HASH_MISMATCH")
    if any(evaluation.get(key, value) != value for key, value in expected_attempts.items()):
        raise ValueError("P7_TEST_ATTEMPT_ACCOUNTING_MISMATCH")
    if recovery_audit_name is not None and completion.get("test_evaluated") is True and any(
        completion.get(key) != value for key, value in expected_attempts.items()
    ):
        raise ValueError("P7_COMPLETION_ATTEMPT_ACCOUNTING_MISMATCH")
    return frame, metrics


def _recovery_authorization(
    *,
    context: Mapping[str, Any],
    claim_path: Path,
    expected_best_checkpoint_sha256: str,
    expected_original_claim_sha256: str,
) -> dict[str, Any]:
    if int(context["provenance"]["fold_index"]) != 4:
        raise ValueError("P7_RECOVERY_FOLD_NOT_AUTHORIZED")
    if (
        expected_best_checkpoint_sha256
        != RECOVERY_APPROVED_BEST_CHECKPOINT_SHA256
        or expected_original_claim_sha256
        != RECOVERY_APPROVED_ORIGINAL_CLAIM_SHA256
    ):
        raise ValueError("P7_RECOVERY_APPROVED_ARTIFACT_ALLOWLIST_MISMATCH")
    completion = context["completion"]
    if completion.get("best_checkpoint_sha256") != expected_best_checkpoint_sha256:
        raise ValueError("P7_RECOVERY_BEST_CHECKPOINT_MISMATCH")
    if sha256_file(claim_path) != expected_original_claim_sha256:
        raise ValueError("P7_RECOVERY_ORIGINAL_CLAIM_HASH_MISMATCH")
    return {
        "schema_version": SCHEMA_VERSION,
        "bug_id": RECOVERY_BUG_ID,
        "fold_index": 4,
        "status": "USER_AUTHORIZED_CONTROLLED_RECOVERY",
        "best_checkpoint_sha256": expected_best_checkpoint_sha256,
        "original_test_claim_sha256": expected_original_claim_sha256,
        "invalidated_attempt_status": RECOVERY_INVALIDATED_STATUS,
        "invalidated_attempts": 1,
        "authorized_recovery_forward_attempts": 1,
        "required_valid_committed_test_evaluations": 1,
        "test_driven_model_changes": RECOVERY_MODEL_CHANGE_STATUS,
        "same_best_checkpoint_required": True,
        "same_split_required": True,
        "same_scientific_config_required": True,
        "same_execution_config_required": True,
        "same_p7_config_required": True,
    }


def _trained_context(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
) -> dict[str, Any]:
    (
        scientific,
        execution,
        execution_hash,
        _p7_config,
        p7_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p7_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    provenance = _provenance(
        scientific, execution, execution_hash, p7_hash, split, initialization
    )
    output = run_directory(fold_index, output_root)
    completion_path = output / "training_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    _validate_completion_artifacts(output, completion, provenance)
    best_path = output / "best.pt"
    if completion.get("best_checkpoint_sha256") != sha256_file(best_path):
        raise ValueError("P7_BEST_CHECKPOINT_HASH_MISMATCH")
    payload = _load_checkpoint(best_path, provenance)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return {
        "scientific": scientific,
        "execution": execution,
        "execution_hash": execution_hash,
        "p7_hash": p7_hash,
        "split": split,
        "manifest": manifest,
        "roi_index": roi_index,
        "model": model,
        "initialization": initialization,
        "provenance": provenance,
        "completion": completion,
        "completion_path": completion_path,
        "output": output,
    }


def evaluate_test_once(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    recovery_authorization: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p7_lifecycle.lock"):
        return _evaluate_test_once_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p7_config_path=p7_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            device_name=device_name,
            num_workers=num_workers,
            output_root=output_root,
            recovery_authorization=recovery_authorization,
        )


def _evaluate_test_once_locked(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    recovery_authorization: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    torch = _torch()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    context = _trained_context(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        p7_config_path=p7_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        fold_index=fold_index,
        output_root=output_root,
    )
    execution = context["execution"]
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    completion = context["completion"]
    if completion.get("status") not in (
        "TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "TRAINING_COMPLETE_TEST_EVALUATED",
    ):
        raise ValueError("P7_TRAINING_NOT_COMPLETE")
    output = context["output"]
    evaluation_path = output / "test_evaluation.json"
    predictions_path = output / "test_predictions.parquet"
    metrics_path = output / "metrics.json"
    claim_path = output / "test_claim.json"
    recovery_authorization_path = output / "test_recovery_authorization.json"
    recovery_claim_path = output / "test_recovery_claim.json"
    attempt_audit_path = output / "test_attempt_audit.json"
    row_provenance = {
        **context["provenance"],
        "checkpoint_sha256": completion["best_checkpoint_sha256"],
    }
    records = build_partition_concept_records(
        context["manifest"],
        context["roi_index"],
        context["split"],
        "test",
        roi_index_path,
    )
    if len(records) != EXPECTED_FOLD_TEST_COUNTS[fold_index]:
        raise ValueError("P7_TEST_COUNT_MISMATCH")
    claim = {
        **context["provenance"],
        "status": "TEST_EVALUATION_CLAIMED",
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "best_epoch_index": int(completion["best_epoch_index"]),
        "expected_test_samples": len(records),
    }
    if evaluation_path.exists():
        if not claim_path.exists() or not predictions_path.exists() or not metrics_path.exists():
            raise ValueError("P7_TEST_TRANSACTION_ARTIFACT_MISSING")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        frame, metrics = _validate_evaluation_artifacts(
            output=output,
            evaluation=evaluation,
            claim=claim,
            completion=completion,
            provenance=context["provenance"],
            records=records,
            row_provenance=row_provenance,
            model=context["model"],
        )
        if completion.get("test_evaluated") is True:
            raise FileExistsError("P7_TEST_ALREADY_EVALUATED")
        completion["status"] = "TRAINING_COMPLETE_TEST_EVALUATED"
        completion["test_evaluated"] = True
        completion["test_evaluation_sha256"] = sha256_file(evaluation_path)
        for key in (
            "total_test_forward_attempts",
            "invalidated_attempts",
            "valid_committed_test_evaluations",
            "test_driven_model_changes",
        ):
            if key in evaluation:
                completion[key] = evaluation[key]
        if "recovery_attempt_audit_sha256" in evaluation:
            completion["recovery_attempt_audit_sha256"] = evaluation[
                "recovery_attempt_audit_sha256"
            ]
        _atomic_json(context["completion_path"], completion)
        return {"evaluation": evaluation, "metrics": metrics, "recovered": True}
    if completion.get("test_evaluated") is True:
        raise ValueError("P7_COMPLETION_CLAIMS_MISSING_EVALUATION")
    controlled_recovery: dict[str, Any] | None = None
    recovery_claim_created = False
    if claim_path.exists():
        if json.loads(claim_path.read_text(encoding="utf-8")) != claim:
            raise ValueError("P7_TEST_CLAIM_MISMATCH")
        if not predictions_path.exists():
            if recovery_authorization is None:
                raise RuntimeError(
                    "P7_TEST_CLAIM_WITHOUT_PREDICTIONS_REQUIRES_AUDIT"
                )
            if set(recovery_authorization) != {
                "bug_id",
                "expected_best_checkpoint_sha256",
                "expected_original_claim_sha256",
            } or recovery_authorization.get("bug_id") != RECOVERY_BUG_ID:
                raise ValueError("P7_RECOVERY_AUTHORIZATION_SCHEMA_MISMATCH")
            controlled_recovery = _recovery_authorization(
                context=context,
                claim_path=claim_path,
                expected_best_checkpoint_sha256=recovery_authorization[
                    "expected_best_checkpoint_sha256"
                ],
                expected_original_claim_sha256=recovery_authorization[
                    "expected_original_claim_sha256"
                ],
            )
            if recovery_authorization_path.exists():
                if json.loads(
                    recovery_authorization_path.read_text(encoding="utf-8")
                ) != controlled_recovery:
                    raise ValueError("P7_RECOVERY_AUTHORIZATION_CONTENT_MISMATCH")
            else:
                _atomic_json(recovery_authorization_path, controlled_recovery)
            recovery_claim = {
                **context["provenance"],
                "status": "RECOVERY_TEST_EVALUATION_CLAIMED",
                "bug_id": RECOVERY_BUG_ID,
                "attempt_index": 2,
                "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
                "original_test_claim_sha256": sha256_file(claim_path),
                "recovery_authorization_sha256": sha256_file(
                    recovery_authorization_path
                ),
                "expected_test_samples": len(records),
            }
            if recovery_claim_path.exists():
                if json.loads(recovery_claim_path.read_text(encoding="utf-8")) != recovery_claim:
                    raise ValueError("P7_RECOVERY_CLAIM_MISMATCH")
                raise RuntimeError(
                    "P7_RECOVERY_CLAIM_WITHOUT_PREDICTIONS_REQUIRES_AUDIT"
                )
            _atomic_json(recovery_claim_path, recovery_claim)
            recovery_claim_created = True
    else:
        if recovery_authorization is not None:
            raise ValueError("P7_RECOVERY_REQUIRES_EXISTING_ORIGINAL_CLAIM")
        _atomic_json(claim_path, claim)
    context["model"].to(device)
    if predictions_path.exists():
        frame = pd.read_parquet(predictions_path)
        mixture_diagnostics = _validate_test_predictions(
            frame, records, row_provenance, context["model"]
        )
        predictions = frame.to_dict("records")
    else:
        if recovery_authorization is not None and not recovery_claim_created:
            raise RuntimeError("P7_RECOVERY_FORWARD_NOT_AUTHORIZED")
        predictions = _prediction_rows(
            context["model"],
            records,
            device,
            batch_size=int(
                execution["project_preregistered"]["batching"]["micro_batch_size"]
            ),
            num_workers=num_workers,
        )
        for row in predictions:
            row.update(row_provenance)
        frame = pd.DataFrame(predictions)
        mixture_diagnostics = _validate_test_predictions(
            frame, records, row_provenance, context["model"]
        )
        _atomic_parquet(predictions_path, frame)
    metrics = regression_metrics(predictions)
    _atomic_json(metrics_path, metrics)
    attempt_fields: dict[str, Any] = {
        "total_test_forward_attempts": 1,
        "invalidated_attempts": 0,
        "valid_committed_test_evaluations": 1,
        "test_driven_model_changes": RECOVERY_MODEL_CHANGE_STATUS,
    }
    recovery_evaluation_fields: dict[str, Any] = {}
    if recovery_authorization is not None:
        if controlled_recovery is None:
            controlled_recovery = json.loads(
                recovery_authorization_path.read_text(encoding="utf-8")
            )
        attempt_fields = {
            "total_test_forward_attempts": 2,
            "invalidated_attempts": 1,
            "valid_committed_test_evaluations": 1,
            "test_driven_model_changes": RECOVERY_MODEL_CHANGE_STATUS,
        }
        attempt_audit = {
            "schema_version": SCHEMA_VERSION,
            "bug_id": RECOVERY_BUG_ID,
            "fold_index": int(context["provenance"]["fold_index"]),
            "status": "RECOVERY_COMPLETE",
            "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
            "original_test_claim_sha256": sha256_file(claim_path),
            "recovery_authorization_sha256": sha256_file(
                recovery_authorization_path
            ),
            "recovery_claim_sha256": sha256_file(recovery_claim_path),
            "test_predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
            **attempt_fields,
            "invalidated_attempt": {
                "attempt_index": 1,
                "status": RECOVERY_INVALIDATED_STATUS,
                "reason": "VERIFIER_IMPLEMENTATION_BUG",
                "mismatch_code": "P7_TEST_STATE_MIXTURE_MISMATCH",
                "committed_predictions": False,
                "committed_metrics": False,
                "committed_evaluation": False,
            },
            "valid_committed_attempt": {
                "attempt_index": 2,
                "status": "VALID_COMMITTED_TEST_EVALUATION",
                "same_best_checkpoint": True,
                "same_split": True,
                "same_scientific_config": True,
                "same_execution_config": True,
                "same_p7_config": True,
            },
            **mixture_diagnostics,
        }
        _atomic_json(attempt_audit_path, attempt_audit)
        recovery_evaluation_fields = {
            "recovery_attempt_audit": attempt_audit_path.name,
            "recovery_attempt_audit_sha256": sha256_file(attempt_audit_path),
            "original_test_claim_sha256": sha256_file(claim_path),
            "recovery_authorization_sha256": sha256_file(
                recovery_authorization_path
            ),
            "recovery_claim_sha256": sha256_file(recovery_claim_path),
        }
    evaluation = {
        **context["provenance"],
        "status": "TEST_EVALUATED_ONCE",
        "best_epoch_index": int(completion["best_epoch_index"]),
        "best_validation_total_loss": float(
            completion["best_validation_total_loss"]
        ),
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "test_claim_sha256": sha256_file(claim_path),
        "test_predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "test_samples": len(frame),
        **attempt_fields,
        **recovery_evaluation_fields,
        **mixture_diagnostics,
    }
    _atomic_json(evaluation_path, evaluation)
    completion["status"] = "TRAINING_COMPLETE_TEST_EVALUATED"
    completion["test_evaluated"] = True
    completion["test_evaluation_sha256"] = sha256_file(evaluation_path)
    completion.update(attempt_fields)
    if recovery_authorization is not None:
        completion["recovery_attempt_audit_sha256"] = sha256_file(
            attempt_audit_path
        )
    _atomic_json(context["completion_path"], completion)
    return {"evaluation": evaluation, "metrics": metrics}


def recover_invalidated_precommit_test(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    bug_id: str,
    expected_best_checkpoint_sha256: str,
    expected_original_claim_sha256: str,
) -> dict[str, Any]:
    if bug_id != RECOVERY_BUG_ID:
        raise ValueError("P7_RECOVERY_BUG_ID_MISMATCH")
    return evaluate_test_once(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        p7_config_path=p7_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        fold_index=fold_index,
        device_name=device_name,
        num_workers=num_workers,
        output_root=output_root,
        recovery_authorization={
            "bug_id": bug_id,
            "expected_best_checkpoint_sha256": expected_best_checkpoint_sha256,
            "expected_original_claim_sha256": expected_original_claim_sha256,
        },
    )


def _intervention_rates(history: pd.DataFrame) -> dict[str, Any]:
    total_batches = int(history["train_batch_count"].sum())
    total_samples = int(history["train_sample_count"].sum())
    decision_rates = OrderedDict()
    sample_weighted_rates = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        decision_rates[group] = float(
            history[f"intervention_{group}_decisions"].sum() / total_batches
        )
        sample_weighted_rates[group] = float(
            history[f"intervention_{group}_sample_weighted"].sum() / total_samples
        )
    overall_decision = float(np.mean(tuple(decision_rates.values())))
    overall_sample_weighted = float(np.mean(tuple(sample_weighted_rates.values())))
    if not 0.24 <= overall_decision <= 0.26:
        raise ValueError("P7_OVERALL_INTERVENTION_DECISION_RATE_OUT_OF_RANGE")
    if not 0.24 <= overall_sample_weighted <= 0.26:
        raise ValueError("P7_OVERALL_INTERVENTION_SAMPLE_RATE_OUT_OF_RANGE")
    if any(not 0.23 <= value <= 0.27 for value in decision_rates.values()):
        raise ValueError("P7_GROUP_INTERVENTION_DECISION_RATE_OUT_OF_RANGE")
    if any(not 0.23 <= value <= 0.27 for value in sample_weighted_rates.values()):
        raise ValueError("P7_GROUP_INTERVENTION_SAMPLE_RATE_OUT_OF_RANGE")
    return {
        "decision_rates": decision_rates,
        "sample_weighted_rates": sample_weighted_rates,
        "overall_decision_rate": overall_decision,
        "overall_sample_weighted_rate": overall_sample_weighted,
    }


def _partition_uid_sha256(split: Mapping[str, Any], partition: str) -> str:
    uids = list(map(str, split["partitions"][partition]["nodule_uids"]))
    if len(uids) != len(set(uids)):
        raise ValueError(f"P7_SPLIT_DUPLICATE_UID:{partition}")
    return sha256_bytes(canonical_json_bytes(sorted(uids)))


def _validate_history_and_runtime(
    history: pd.DataFrame,
    runtime: Mapping[str, Any],
    split: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[int, int]:
    expected_train = int(split["partitions"]["train"]["summary"]["nodules"])
    expected_validation = int(
        split["partitions"]["validation"]["summary"]["nodules"]
    )
    expected_train_hash = _partition_uid_sha256(split, "train")
    expected_validation_hash = _partition_uid_sha256(split, "validation")
    if len(history) != 80 or history["epoch_index"].tolist() != list(range(80)):
        raise ValueError("P7_HISTORY_EPOCH_MISMATCH")
    if not (history["train_sample_count"] == expected_train).all():
        raise ValueError("P7_HISTORY_TRAIN_COVERAGE_MISMATCH")
    if not (history["validation_sample_count"] == expected_validation).all():
        raise ValueError("P7_HISTORY_VALIDATION_COVERAGE_MISMATCH")
    if not (history["train_nodule_set_sha256"] == expected_train_hash).all():
        raise ValueError("P7_HISTORY_TRAIN_UID_SET_MISMATCH")
    if not (
        history["validation_nodule_set_sha256"] == expected_validation_hash
    ).all():
        raise ValueError("P7_HISTORY_VALIDATION_UID_SET_MISMATCH")
    if any(runtime.get(key) != value for key, value in provenance.items()):
        raise ValueError("P7_RUNTIME_PROVENANCE_MISMATCH")
    if runtime.get("device_type") != "cuda" or "H200" not in str(
        runtime.get("gpu_name", "")
    ).upper():
        raise ValueError("P7_RUNTIME_H200_MISMATCH")
    precision = {
        "fp32": True,
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "torch_use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
    }
    for key, expected in precision.items():
        if runtime.get(key) is not expected:
            raise ValueError(f"P7_RUNTIME_PRECISION_POLICY_MISMATCH:{key}")
    if int(runtime.get("epochs_total", -1)) != 80:
        raise ValueError("P7_RUNTIME_EPOCH_MISMATCH")
    if not isinstance(runtime.get("peak_reserved_bytes"), int):
        raise ValueError("P7_RUNTIME_MEMORY_EVIDENCE_MISSING")
    return expected_train, expected_validation


def _verify_fold_locked(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    require_test: bool,
) -> dict[str, Any]:
    context = _trained_context(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        p7_config_path=p7_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        fold_index=fold_index,
        output_root=output_root,
    )
    output = context["output"]
    completion = context["completion"]
    history = pd.read_csv(output / "history.csv")
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    expected_train, _expected_validation = _validate_history_and_runtime(
        history, runtime, context["split"], context["provenance"]
    )
    minimum_index = int(history["validation_total_loss"].idxmin())
    minimum_epoch = int(history.iloc[minimum_index]["epoch_index"])
    if minimum_epoch != int(completion["best_epoch_index"]):
        raise ValueError("P7_CHECKPOINT_SELECTION_MISMATCH")
    if not serialized_float_consistent(
        float(completion["best_validation_total_loss"]),
        float(history.iloc[minimum_index]["validation_total_loss"]),
    ):
        raise ValueError("P7_BEST_OBJECTIVE_MISMATCH")
    best_payload = _load_checkpoint(output / "best.pt", context["provenance"])
    last_payload = _load_checkpoint(output / "last.pt", context["provenance"])
    if int(best_payload["epoch_index"]) != minimum_epoch:
        raise ValueError("P7_BEST_PAYLOAD_EPOCH_MISMATCH")
    if not serialized_float_consistent(
        float(best_payload["validation_total_loss"]),
        float(history.iloc[minimum_index]["validation_total_loss"]),
    ):
        raise ValueError("P7_BEST_PAYLOAD_OBJECTIVE_MISMATCH")
    if int(last_payload["epoch_index"]) != 79:
        raise ValueError("P7_LAST_PAYLOAD_EPOCH_MISMATCH")
    for filename, key in (
        ("best.pt", "best_checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
        ("history.csv", "history_sha256"),
        ("runtime.json", "runtime_sha256"),
    ):
        if completion.get(key) != sha256_file(output / filename):
            raise ValueError(f"P7_ARTIFACT_HASH_MISMATCH:{filename}")
    rates = _intervention_rates(history)
    report: dict[str, Any] = {
        **context["provenance"],
        "status": "PASS",
        "epochs": 80,
        "train_samples_per_epoch": expected_train,
        "best_epoch_index": minimum_epoch,
        "best_validation_total_loss": float(
            history.iloc[minimum_index]["validation_total_loss"]
        ),
        "intervention_rates": rates,
    }
    if require_test:
        if (
            completion.get("status") != "TRAINING_COMPLETE_TEST_EVALUATED"
            or completion.get("test_evaluated") is not True
        ):
            raise ValueError("P7_TEST_NOT_EVALUATED")
        evaluation_path = output / "test_evaluation.json"
        if completion.get("test_evaluation_sha256") != sha256_file(evaluation_path):
            raise ValueError("P7_TEST_EVALUATION_HASH_MISMATCH")
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        records = build_partition_concept_records(
            context["manifest"],
            context["roi_index"],
            context["split"],
            "test",
            roi_index_path,
        )
        row_provenance = {
            **context["provenance"],
            "checkpoint_sha256": completion["best_checkpoint_sha256"],
        }
        claim = {
            **context["provenance"],
            "status": "TEST_EVALUATION_CLAIMED",
            "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
            "best_epoch_index": int(completion["best_epoch_index"]),
            "expected_test_samples": len(records),
        }
        frame, _metrics = _validate_evaluation_artifacts(
            output=output,
            evaluation=evaluation,
            claim=claim,
            completion=completion,
            provenance=context["provenance"],
            records=records,
            row_provenance=row_provenance,
            model=context["model"],
        )
        if len(frame) != EXPECTED_FOLD_TEST_COUNTS[fold_index]:
            raise ValueError("P7_VERIFY_TEST_COUNT_MISMATCH")
        report["test_samples"] = len(frame)
        report["test_evaluated_once"] = True
        report["total_test_forward_attempts"] = int(
            evaluation.get("total_test_forward_attempts", 1)
        )
        report["invalidated_attempts"] = int(
            evaluation.get("invalidated_attempts", 0)
        )
        report["valid_committed_test_evaluations"] = int(
            evaluation.get("valid_committed_test_evaluations", 1)
        )
        report["test_driven_model_changes"] = evaluation.get(
            "test_driven_model_changes", RECOVERY_MODEL_CHANGE_STATUS
        )
        report["maximum_normalized_reconstruction_error"] = float(
            frame["normalized_reconstruction_max_abs_error"].max()
        )
        report["maximum_rating_reconstruction_error"] = float(
            frame["rating_reconstruction_max_abs_error"].max()
        )
    return report


def verify_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    require_test: bool = True,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p7_lifecycle.lock"):
        return _verify_fold_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p7_config_path=p7_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            output_root=output_root,
            require_test=require_test,
        )


def verify_all(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    reports = [
        verify_fold(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p7_config_path=p7_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold,
            output_root=output_root,
        )
        for fold in range(5)
    ]
    frames = [
        pd.read_parquet(run_directory(fold, output_root) / "test_predictions.parquet")
        for fold in range(5)
    ]
    pooled = pd.concat(frames, ignore_index=True)
    if len(pooled) != 2633 or pooled["nodule_uid"].nunique() != 2633:
        raise ValueError("P7_OOF_NODULE_SET_MISMATCH")
    if pooled["patient_key"].nunique() != 868:
        raise ValueError("P7_OOF_PATIENT_SET_MISMATCH")
    if pooled.groupby("patient_key")["fold_index"].nunique().max() != 1:
        raise ValueError("P7_OOF_PATIENT_LEAKAGE")
    return {
        "status": "PASS",
        "oof_nodules": 2633,
        "oof_patients": 868,
        "fold_test_counts": [len(frame) for frame in frames],
        "folds": reports,
    }


def overfit_check(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    samples: int,
    steps: int,
    output_path: Path,
) -> dict[str, Any]:
    torch = _torch()
    device = torch.device(device_name)
    (
        scientific,
        execution,
        execution_hash,
        _p7_config,
        p7_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p7_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    seed_training(int(initialization["fold_seed"]))
    model.to(device)
    optimizer = _optimizer(model, execution)
    records = sorted(
        build_partition_concept_records(
            manifest, roi_index, split, "train", roi_index_path
        ),
        key=lambda record: record.nodule_uid,
    )[:samples]
    dataset = ConceptROIDataset.build(
        records,
        training=False,
        base_seed=0,
        fold_index=fold_index,
        epoch_index=0,
    )
    batch = next(iter(_loader(dataset, batch_size=samples, num_workers=0)))
    image = batch["image"].to(device=device, dtype=torch.float32)
    concepts = _targets_to_device(batch["targets"], device)
    malignancy = batch["target_normalized"].to(device=device, dtype=torch.float32)
    losses: list[float] = []
    model.train()
    for step in range(steps):
        mask = batch_shared_intervention_mask(
            base_seed=int(scientific["reproducibility"]["base_seed"]),
            fold_index=fold_index,
            epoch_index=0,
            batch_index=step,
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            image, intervention_targets=concepts, intervention_mask=mask
        )
        loss = cem_losses(
            outputs, {"concepts": concepts, "malignancy": malignancy}
        )["total_loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    initial = float(np.mean(losses[:5]))
    final = float(np.mean(losses[-5:]))
    if not math.isfinite(final) or final >= initial:
        raise RuntimeError("P7_OVERFIT_SANITY_DID_NOT_IMPROVE")
    report = {
        **_provenance(
            scientific, execution, execution_hash, p7_hash, split, initialization
        ),
        **_runtime_environment(device),
        "status": "PASS",
        "scope": "train_only_controlled_overfit_sanity",
        "formal_run": False,
        "augmentation_enabled": False,
        "samples": samples,
        "steps": steps,
        "initial_five_step_mean_total_loss": initial,
        "final_five_step_mean_total_loss": final,
        "relative_final_loss": final / initial,
    }
    _atomic_json(output_path, report)
    return report


def preflight(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p7_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_path: Path,
) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    (
        scientific,
        execution,
        execution_hash,
        _p7_config,
        p7_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p7_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    seed_training(int(initialization["fold_seed"]))
    model.to(device)
    optimizer = _optimizer(model, execution)
    records = build_partition_concept_records(
        manifest, roi_index, split, "train", roi_index_path
    )
    ordered = _ordered_records(
        records,
        base_seed=int(scientific["reproducibility"]["base_seed"]),
        fold_index=fold_index,
        epoch_index=0,
    )[:16]
    dataset = ConceptROIDataset.build(
        ordered,
        training=True,
        base_seed=int(scientific["reproducibility"]["base_seed"]),
        fold_index=fold_index,
        epoch_index=0,
    )
    batch = next(iter(_loader(dataset, batch_size=16, num_workers=0)))
    image = batch["image"].to(device=device, dtype=torch.float32)
    concepts = _targets_to_device(batch["targets"], device)
    malignancy = batch["target_normalized"].to(device=device, dtype=torch.float32)
    mask = batch_shared_intervention_mask(
        base_seed=int(scientific["reproducibility"]["base_seed"]),
        fold_index=fold_index,
        epoch_index=0,
        batch_index=0,
    ).to(device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    predicted_outputs = model(image)
    predicted_reconstruction = task_predictions_and_contributions(
        model, predicted_outputs
    )
    outputs = model(image, intervention_targets=concepts, intervention_mask=mask)
    intervened_reconstruction = task_predictions_and_contributions(model, outputs)
    losses = cem_losses(outputs, {"concepts": concepts, "malignancy": malignancy})
    losses["total_loss"].backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    with torch.no_grad():
        features_a = torch.zeros(2, 1024, device=device)
        features_b = torch.ones(2, 1024, device=device)
        states_a = model.states_and_probabilities(features_a)["states"]
        states_b = model.states_and_probabilities(features_b)["states"]
        dynamic_states_changed = any(
            not torch.equal(states_a[group], states_b[group])
            for group in CONCEPT_GROUP_ORDER
        )
    reserved = int(torch.cuda.max_memory_reserved(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = reserved / total_memory
    if fraction > 0.85:
        raise RuntimeError(f"P7_PREFLIGHT_MEMORY_LIMIT_EXCEEDED:{fraction}")
    if not dynamic_states_changed:
        raise RuntimeError("P7_PREFLIGHT_DYNAMIC_STATES_DID_NOT_CHANGE")
    report = {
        **_provenance(
            scientific, execution, execution_hash, p7_hash, split, initialization
        ),
        "status": "PASS",
        "batch_size": int(image.shape[0]),
        "forward": True,
        "task_and_concept_losses": True,
        "batch_shared_intervention": True,
        "backward": True,
        "adam_step": True,
        "dynamic_states_changed_with_h_x": dynamic_states_changed,
        "task_loss": float(losses["task_loss"].detach().cpu()),
        "concept_loss": float(losses["concept_loss"].detach().cpu()),
        "total_loss": float(losses["total_loss"].detach().cpu()),
        "predicted_normalized_reconstruction_max_abs_error": predicted_reconstruction[
            "normalized_reconstruction_max_abs_error"
        ],
        "predicted_rating_reconstruction_max_abs_error": predicted_reconstruction[
            "rating_reconstruction_max_abs_error"
        ],
        "intervened_normalized_reconstruction_max_abs_error": intervened_reconstruction[
            "normalized_reconstruction_max_abs_error"
        ],
        "intervened_rating_reconstruction_max_abs_error": intervened_reconstruction[
            "rating_reconstruction_max_abs_error"
        ],
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": reserved,
        "gpu_total_bytes": total_memory,
        "peak_reserved_fraction": fraction,
        "maximum_allowed_fraction": 0.85,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": importlib.metadata.version("torch"),
        "monai_version": importlib.metadata.version("monai"),
        "cuda_runtime": torch.version.cuda,
        **_runtime_environment(device),
    }
    _atomic_json(output_path, report)
    return report


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path(
            "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
        ),
    )
    parser.add_argument("--p7-config", type=Path, default=P7_EXECUTION_CONFIG_DEFAULT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/baseline_v2/manifests/nodules.parquet"),
    )
    parser.add_argument(
        "--roi-index",
        type=Path,
        default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/baseline_v2/cem")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    overfit_parser = subparsers.add_parser("overfit-check")
    _common_arguments(overfit_parser)
    overfit_parser.add_argument("--fold", type=int, required=True, choices=range(5))
    overfit_parser.add_argument("--device", default="cuda")
    overfit_parser.add_argument("--samples", type=int, default=8)
    overfit_parser.add_argument("--steps", type=int, default=40)
    overfit_parser.add_argument(
        "--output", type=Path, default=Path("runs/baseline_v2/cem/fold_0/stage_a/overfit_sanity.json")
    )
    preflight_parser = subparsers.add_parser("preflight")
    _common_arguments(preflight_parser)
    preflight_parser.add_argument("--fold", type=int, required=True, choices=range(5))
    preflight_parser.add_argument(
        "--output", type=Path, default=Path("runs/baseline_v2/cem/fold_0/stage_a/preflight.json")
    )
    train_parser = subparsers.add_parser("train")
    _common_arguments(train_parser)
    train_parser.add_argument("--fold", type=int, required=True, choices=range(5))
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--resume", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate-test")
    _common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--fold", type=int, required=True, choices=range(5))
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--num-workers", type=int, default=4)
    recovery_parser = subparsers.add_parser("recover-test")
    _common_arguments(recovery_parser)
    recovery_parser.add_argument("--fold", type=int, required=True, choices=(4,))
    recovery_parser.add_argument("--device", default="cuda")
    recovery_parser.add_argument("--num-workers", type=int, default=4)
    recovery_parser.add_argument(
        "--bug-id", required=True, choices=(RECOVERY_BUG_ID,)
    )
    recovery_parser.add_argument(
        "--expected-best-checkpoint-sha256", required=True
    )
    recovery_parser.add_argument(
        "--expected-original-claim-sha256", required=True
    )
    verify_parser = subparsers.add_parser("verify")
    _common_arguments(verify_parser)
    scope = verify_parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--fold", type=int, choices=range(5))
    scope.add_argument("--scope", choices=("all",))
    verify_parser.add_argument("--training-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "scientific_config_path": arguments.config,
        "execution_config_path": arguments.execution_config,
        "p7_config_path": arguments.p7_config,
        "manifest_path": arguments.manifest,
        "roi_index_path": arguments.roi_index,
    }
    if arguments.command == "overfit-check":
        report = overfit_check(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            samples=arguments.samples,
            steps=arguments.steps,
            output_path=arguments.output,
        )
    elif arguments.command == "preflight":
        report = preflight(
            **common, fold_index=arguments.fold, output_path=arguments.output
        )
    elif arguments.command == "train":
        report = train_fold(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
            output_root=arguments.output_root,
            resume=arguments.resume,
        )
    elif arguments.command == "evaluate-test":
        report = evaluate_test_once(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
            output_root=arguments.output_root,
        )
    elif arguments.command == "recover-test":
        report = recover_invalidated_precommit_test(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
            output_root=arguments.output_root,
            bug_id=arguments.bug_id,
            expected_best_checkpoint_sha256=(
                arguments.expected_best_checkpoint_sha256
            ),
            expected_original_claim_sha256=(
                arguments.expected_original_claim_sha256
            ),
        )
    elif arguments.fold is not None:
        report = verify_fold(
            **common,
            fold_index=arguments.fold,
            output_root=arguments.output_root,
            require_test=not arguments.training_only,
        )
    else:
        report = verify_all(**common, output_root=arguments.output_root)
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
