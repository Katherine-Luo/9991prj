"""Train and verify the Baseline-v2 sequential Standard CBM regression model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    patient_key,
    sha256_bytes,
    sha256_file,
    validate_encoder_artifact,
)
from lidc_baseline.p5_blackbox import (
    _atomic_csv,
    _atomic_json,
    _atomic_parquet,
    _atomic_torch_save,
    _loader,
    _optimizer,
    _prepare_sources,
    _runtime_environment,
    _scheduler,
    apply_training_augmentation,
    augmentation_parameters,
    capture_rng_state,
    checkpoint_improves,
    configure_fp32_determinism,
    epoch_uid_order,
    exclusive_fold_lifecycle_lock,
    regression_metrics,
    require_formal_gpu_for_cuda,
    restore_rng_state,
    seed_training,
    serialized_float_consistent,
    validate_execution_config,
)


SCHEMA_VERSION = 1
MODEL_NAME = "standard_cbm"
P6_EXECUTION_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_p6_standard_cbm_h200.yaml"
)
CONTINUOUS_CONCEPTS = (
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
CATEGORICAL_CONCEPTS = OrderedDict(
    (("internalStructure", 4), ("calcification", 6))
)
CONCEPT_GROUP_ORDER = (
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
CONCEPT_OUTPUT_SIZES = OrderedDict(
    (
        ("subtlety", 1),
        ("internalStructure", 4),
        ("calcification", 6),
        ("sphericity", 1),
        ("margin", 1),
        ("lobulation", 1),
        ("spiculation", 1),
        ("texture", 1),
    )
)
CANONICAL_VECTOR_SLICES = OrderedDict(
    (
        ("subtlety", slice(0, 1)),
        ("internalStructure", slice(1, 5)),
        ("calcification", slice(5, 11)),
        ("sphericity", slice(11, 12)),
        ("margin", slice(12, 13)),
        ("lobulation", slice(13, 14)),
        ("spiculation", slice(14, 15)),
        ("texture", slice(15, 16)),
    )
)
CONCEPT_HEAD_SEED_DOMAIN = "Baseline-v2/P6/standard-cbm-concept-head"
TASK_HEAD_SEED_DOMAIN = "Baseline-v2/P6/standard-cbm-task-head"
CACHE_PROVENANCE_HASH_KEYS = (
    "scientific_config_sha256",
    "execution_config_sha256",
    "p6_execution_config_sha256",
    "split_sha256",
    "encoder_initialization_sha256",
    "encoder_artifact_file_sha256",
    "combined_concept_head_initialization_sha256",
    "concept_best_checkpoint_sha256",
    "predictor_semantic_sha256",
    "batchnorm_state_sha256",
    "source_manifest_sha256",
    "source_roi_index_sha256",
)


def _torch() -> Any:
    import torch

    return torch


def validate_p6_execution_config(
    config_path: str | Path = P6_EXECUTION_CONFIG_DEFAULT,
    digest_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load and enforce the independently frozen P6 execution supplement."""
    source = Path(config_path)
    config = load_config(source)
    observed = compute_config_sha256(config)
    digest = Path(digest_path) if digest_path is not None else source.with_suffix(".sha256")
    if digest.read_text(encoding="ascii").strip() != observed:
        raise ValueError("P6_EXECUTION_CONFIG_HASH_MISMATCH")
    project = config.get("project_preregistered", {})
    predictor = project.get("concept_predictor", {})
    sequential = project.get("sequential_training", {})
    if (
        config.get("protocol_version") != "Baseline-v2"
        or config.get("phase") != "P6"
        or config.get("model") != MODEL_NAME
    ):
        raise ValueError("P6_EXECUTION_CONFIG_IDENTITY_MISMATCH")
    if predictor.get("head_type") != "independent_linear_no_hidden_layer":
        raise ValueError("P6_CONCEPT_HEAD_POLICY_MISMATCH")
    if tuple(predictor.get("group_order", ())) != CONCEPT_GROUP_ORDER:
        raise ValueError("P6_CONCEPT_GROUP_ORDER_MISMATCH")
    vector = predictor.get("canonical_task_vector", {})
    if (
        vector.get("dimension") != 16
        or vector.get("source") != "activated_predictions"
        or vector.get("preactivation_logits_as_task_input") is not False
        or vector.get("malignancy_as_input_concept") is not False
    ):
        raise ValueError("P6_CANONICAL_VECTOR_POLICY_MISMATCH")
    cache = sequential.get("cache_gate", {})
    task = sequential.get("task_stage", {})
    if (
        cache.get("generated_before_task_training") != ["train", "validation"]
        or cache.get("forbidden_before_task_best") != ["test"]
        or cache.get("ground_truth_concepts_as_task_features") != "forbidden"
        or task.get("input") != "frozen_activated_predicted_concepts"
        or task.get("ground_truth_concepts") != "forbidden"
    ):
        raise ValueError("P6_SEQUENTIAL_LEAKAGE_POLICY_MISMATCH")
    return config, observed


def _seed_from_material(material: str, fold_seed: int) -> int:
    payload = material.encode("utf-8") + int(fold_seed).to_bytes(8, "big", signed=False)
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def concept_head_seed(group: str, fold_seed: int) -> int:
    if group not in CONCEPT_OUTPUT_SIZES:
        raise ValueError(f"P6_UNKNOWN_CONCEPT_GROUP:{group}")
    return _seed_from_material(f"{CONCEPT_HEAD_SEED_DOMAIN}/{group}", fold_seed)


def task_head_seed(fold_seed: int) -> int:
    return _seed_from_material(TASK_HEAD_SEED_DOMAIN, fold_seed)


def module_state_sha256(module: Any) -> str:
    state = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in module.state_dict().items()
    )
    return encoder_state_sha256(state)


def build_deterministic_concept_heads(fold_seed: int) -> tuple[Any, dict[str, Any]]:
    """Create each linear head under an isolated, order-independent CPU RNG."""
    torch = _torch()
    heads = torch.nn.ModuleDict()
    seeds: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for group, output_size in CONCEPT_OUTPUT_SIZES.items():
        seed = concept_head_seed(group, fold_seed)
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seed)
            head = torch.nn.Linear(1024, output_size)
        heads[group] = head
        seeds[group] = seed
        hashes[group] = module_state_sha256(head)
    return heads, {
        "concept_head_initialization_seeds": seeds,
        "concept_head_initialization_sha256": hashes,
        "combined_concept_head_initialization_sha256": module_state_sha256(heads),
        "concept_head_seed_derivation": (
            "sha256(utf8('Baseline-v2/P6/standard-cbm-concept-head/<group>') "
            "|| fold_seed_u64be), first_8_bytes_u64be_mask_63_bits"
        ),
    }


def build_deterministic_task_head(fold_seed: int) -> tuple[Any, dict[str, Any]]:
    torch = _torch()
    seed = task_head_seed(fold_seed)
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        head = torch.nn.Linear(16, 1)
    return head, {
        "task_head_initialization_seed": seed,
        "task_head_initialization_sha256": module_state_sha256(head),
        "task_head_seed_derivation": (
            "sha256(utf8('Baseline-v2/P6/standard-cbm-task-head') || fold_seed_u64be), "
            "first_8_bytes_u64be_mask_63_bits"
        ),
    }


def activate_concept_logits(logits: Mapping[str, Any]) -> OrderedDict[str, Any]:
    torch = _torch()
    if tuple(logits) != CONCEPT_GROUP_ORDER:
        raise ValueError("P6_CONCEPT_LOGIT_ORDER_MISMATCH")
    activated: OrderedDict[str, Any] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        value = logits[group]
        if value.ndim != 2 or value.shape[1] != CONCEPT_OUTPUT_SIZES[group]:
            raise ValueError(f"P6_CONCEPT_LOGIT_SHAPE_MISMATCH:{group}")
        activated[group] = (
            torch.softmax(value, dim=1)
            if group in CATEGORICAL_CONCEPTS
            else torch.sigmoid(value)
        )
    return activated


def canonical_concept_vector(activated: Mapping[str, Any]) -> Any:
    torch = _torch()
    if tuple(activated) != CONCEPT_GROUP_ORDER:
        raise ValueError("P6_ACTIVATED_CONCEPT_ORDER_MISMATCH")
    pieces = []
    batch_size: int | None = None
    for group in CONCEPT_GROUP_ORDER:
        value = activated[group]
        if value.ndim != 2 or value.shape[1] != CONCEPT_OUTPUT_SIZES[group]:
            raise ValueError(f"P6_ACTIVATED_CONCEPT_SHAPE_MISMATCH:{group}")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif int(value.shape[0]) != batch_size:
            raise ValueError("P6_ACTIVATED_CONCEPT_BATCH_MISMATCH")
        pieces.append(value)
    result = torch.cat(pieces, dim=1)
    if result.shape != (batch_size, 16):
        raise ValueError("P6_CANONICAL_VECTOR_SHAPE_MISMATCH")
    return result


class StandardCBMConceptPredictor:
    """Factory for the P6 encoder and eight independent linear heads."""

    @staticmethod
    def build(encoder: Any, heads: Any) -> Any:
        torch = _torch()

        class Model(torch.nn.Module):
            def __init__(self, feature_encoder: Any, concept_heads: Any) -> None:
                super().__init__()
                self.encoder = feature_encoder
                self.relu = torch.nn.ReLU(inplace=False)
                self.concept_heads = concept_heads

            def forward(self, image: Any) -> dict[str, Any]:
                features = self.relu(self.encoder(image))
                pooled = features.mean(dim=(2, 3, 4))
                logits = OrderedDict(
                    (group, self.concept_heads[group](pooled))
                    for group in CONCEPT_GROUP_ORDER
                )
                activated = activate_concept_logits(logits)
                return {
                    "logits": logits,
                    "activated": activated,
                    "canonical_vector": canonical_concept_vector(activated),
                }

        return Model(encoder, heads)


def build_initialized_concept_predictor(
    scientific_config: Mapping[str, Any],
    split: Mapping[str, Any],
    encoder_artifact_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    encoder = build_encoder()
    encoder_hash = load_shared_encoder_initialization(
        encoder, encoder_artifact_path, scientific_config, split
    )
    validated = validate_encoder_artifact(Path(encoder_artifact_path), scientific_config, split)
    fold_seed = int(validated["metadata"]["fold_seed"])
    heads, head_metadata = build_deterministic_concept_heads(fold_seed)
    if encoder_state_sha256(encoder.state_dict()) != encoder_hash:
        raise ValueError("P6_ENCODER_HASH_CHANGED_BEFORE_TRAINING")
    model = StandardCBMConceptPredictor.build(encoder, heads)
    return model, {
        "fold_seed": fold_seed,
        "encoder_initialization_sha256": encoder_hash,
        "encoder_artifact_file_sha256": sha256_file(encoder_artifact_path),
        **head_metadata,
    }


def concept_group_loss_sums(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[OrderedDict[str, Any], int]:
    """Return per-group sample sums so epoch aggregation is sample weighted."""
    torch = _torch()
    logits = outputs["logits"]
    activated = outputs["activated"]
    batch_size = int(outputs["canonical_vector"].shape[0])
    sums: OrderedDict[str, Any] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        target = targets[group]
        prediction = activated[group]
        if target.shape != prediction.shape:
            raise ValueError(f"P6_CONCEPT_TARGET_SHAPE_MISMATCH:{group}")
        if group in CATEGORICAL_CONCEPTS:
            if not torch.allclose(
                target.sum(dim=1),
                torch.ones(batch_size, dtype=target.dtype, device=target.device),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError(f"P6_CATEGORICAL_TARGET_SUM_MISMATCH:{group}")
            per_sample = -(target * torch.log_softmax(logits[group], dim=1)).sum(dim=1)
        else:
            per_sample = (prediction - target).square().reshape(batch_size)
        sums[group] = per_sample.sum()
    return sums, batch_size


def concept_loss(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> tuple[Any, OrderedDict[str, Any]]:
    torch = _torch()
    sums, batch_size = concept_group_loss_sums(outputs, targets)
    means = OrderedDict((group, value / batch_size) for group, value in sums.items())
    total = torch.stack(tuple(means.values())).mean()
    return total, means


@dataclass(frozen=True)
class ConceptRecord:
    nodule_uid: str
    patient_key: str
    roi_path: Path
    target_normalized: float
    target_1_to_5: float
    extreme_binary_eligible: bool
    extreme_binary_label: int | None
    continuous_targets: tuple[float, ...]
    internal_structure_target: tuple[float, ...]
    calcification_target: tuple[float, ...]
    valid_reader_counts: tuple[int, ...]
    categorical_ties: tuple[bool, bool]


def _parse_distribution(value: Any, size: int, group: str) -> tuple[float, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, np.ndarray):
        parsed = value.tolist()
    else:
        parsed = list(value)
    distribution = tuple(float(item) for item in parsed)
    if (
        len(distribution) != size
        or not np.isfinite(distribution).all()
        or min(distribution) < 0.0
        or not np.isclose(sum(distribution), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError(f"P6_CATEGORICAL_TARGET_INVALID:{group}")
    return distribution


def _boolean(value: Any) -> bool:
    if value is pd.NA or pd.isna(value):
        return False
    return bool(value)


def build_partition_concept_records(
    manifest: pd.DataFrame,
    roi_index: pd.DataFrame,
    split: Mapping[str, Any],
    partition: str,
    roi_index_path: str | Path,
) -> list[ConceptRecord]:
    """Resolve one split partition to source-verified P6 concept records."""
    if partition not in ("train", "validation", "test"):
        raise ValueError(f"P6_INVALID_PARTITION:{partition}")
    uids = list(map(str, split["partitions"][partition]["nodule_uids"]))
    if len(uids) != len(set(uids)):
        raise ValueError("P6_SPLIT_DUPLICATE_UID")
    primary = manifest[manifest["primary_regression_eligible"].astype(bool)].copy()
    primary["nodule_uid"] = primary["nodule_uid"].astype(str)
    primary = primary.set_index("nodule_uid", drop=False)
    index = roi_index.copy()
    index["nodule_uid"] = index["nodule_uid"].astype(str)
    index = index.set_index("nodule_uid", drop=False)
    if not set(uids) <= set(primary.index) or not set(uids) <= set(index.index):
        raise ValueError("P6_PARTITION_SOURCE_SET_MISMATCH")
    base = Path(roi_index_path).resolve().parent.parent
    records = []
    for uid in uids:
        row = primary.loc[uid]
        roi = index.loc[uid]
        path = (base / Path(str(roi["relative_roi_path"]))).resolve()
        if path.parent != (base / "rois").resolve() or not path.is_file():
            raise ValueError(f"P6_ROI_PATH_MISMATCH:{uid}")
        if sha256_file(path) != str(roi["roi_file_sha256"]):
            raise ValueError(f"P6_ROI_HASH_MISMATCH:{uid}")
        continuous = tuple(float(row[f"{group}_target"]) for group in CONTINUOUS_CONCEPTS)
        if not np.isfinite(continuous).all() or min(continuous) < 0.0 or max(continuous) > 1.0:
            raise ValueError(f"P6_CONTINUOUS_TARGET_INVALID:{uid}")
        counts = tuple(int(row[f"{group}_valid_reader_count"]) for group in CONCEPT_GROUP_ORDER)
        if min(counts) < 1:
            raise ValueError(f"P6_CONCEPT_VALID_READER_COUNT_INVALID:{uid}")
        extreme = _boolean(row["extreme_binary_eligible"])
        records.append(
            ConceptRecord(
                nodule_uid=uid,
                patient_key=patient_key(str(row["patient_id"])),
                roi_path=path,
                target_normalized=float(row["malignancy_target_normalized"]),
                target_1_to_5=float(row["mean_malignancy"]),
                extreme_binary_eligible=extreme,
                extreme_binary_label=int(row["extreme_binary_label"]) if extreme else None,
                continuous_targets=continuous,
                internal_structure_target=_parse_distribution(
                    row["internalStructure_vote_distribution"], 4, "internalStructure"
                ),
                calcification_target=_parse_distribution(
                    row["calcification_vote_distribution"], 6, "calcification"
                ),
                valid_reader_counts=counts,
                categorical_ties=(
                    _boolean(row["internalStructure_modal_tie"]),
                    _boolean(row["calcification_modal_tie"]),
                ),
            )
        )
    return records


class ConceptROIDataset:
    """Factory for P6 ROI samples and all eight concept targets."""

    @staticmethod
    def build(
        records: Sequence[ConceptRecord],
        *,
        training: bool,
        base_seed: int,
        fold_index: int,
        epoch_index: int,
    ) -> Any:
        torch = _torch()

        class Dataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(records)

            def __getitem__(self, index: int) -> dict[str, Any]:
                record = records[index]
                with np.load(record.roi_path, allow_pickle=False) as archive:
                    image = archive["image"]
                if image.shape != (1, 64, 64, 64) or image.dtype != np.float32:
                    raise ValueError(f"P6_ROI_INTERFACE_MISMATCH:{record.nodule_uid}")
                if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
                    raise ValueError(f"P6_ROI_VALUE_MISMATCH:{record.nodule_uid}")
                tensor = torch.from_numpy(np.array(image, copy=True))
                if training:
                    tensor = apply_training_augmentation(
                        tensor,
                        augmentation_parameters(
                            base_seed,
                            fold_index,
                            epoch_index,
                            record.nodule_uid,
                        ),
                    )
                continuous = dict(zip(CONTINUOUS_CONCEPTS, record.continuous_targets, strict=True))
                targets = {
                    "subtlety": torch.tensor([continuous["subtlety"]], dtype=torch.float32),
                    "internalStructure": torch.tensor(record.internal_structure_target, dtype=torch.float32),
                    "calcification": torch.tensor(record.calcification_target, dtype=torch.float32),
                    "sphericity": torch.tensor([continuous["sphericity"]], dtype=torch.float32),
                    "margin": torch.tensor([continuous["margin"]], dtype=torch.float32),
                    "lobulation": torch.tensor([continuous["lobulation"]], dtype=torch.float32),
                    "spiculation": torch.tensor([continuous["spiculation"]], dtype=torch.float32),
                    "texture": torch.tensor([continuous["texture"]], dtype=torch.float32),
                }
                return {
                    "image": tensor,
                    "targets": targets,
                    "target_normalized": torch.tensor([record.target_normalized], dtype=torch.float32),
                    "nodule_uid": record.nodule_uid,
                }

        return Dataset()


def freeze_concept_predictor(model: Any) -> str:
    """Freeze parameters and BatchNorm state before caching and task training."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return module_state_sha256(model)


def batchnorm_state_sha256(model: Any) -> str:
    torch = _torch()
    state: OrderedDict[str, Any] = OrderedDict()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for field in ("running_mean", "running_var", "num_batches_tracked"):
                value = getattr(module, field, None)
                if value is not None:
                    state[f"{module_name}.{field}"] = value.detach().cpu().contiguous()
    return encoder_state_sha256(state)


def task_predictions_and_contributions(task_head: Any, canonical_vector: Any) -> dict[str, Any]:
    """Compute the task score and exact eight-group additive decomposition."""
    torch = _torch()
    if canonical_vector.ndim != 2 or canonical_vector.shape[1] != 16:
        raise ValueError("P6_TASK_INPUT_SHAPE_MISMATCH")
    score = task_head(canonical_vector)
    weights = task_head.weight.reshape(-1)
    raw: OrderedDict[str, Any] = OrderedDict()
    for group, vector_slice in CANONICAL_VECTOR_SLICES.items():
        raw[group] = (canonical_vector[:, vector_slice] * weights[vector_slice]).sum(
            dim=1, keepdim=True
        )
    reconstructed = task_head.bias.reshape(1, 1) + torch.stack(tuple(raw.values()), dim=0).sum(dim=0)
    if not torch.allclose(score, reconstructed, rtol=0.0, atol=1e-6):
        raise ValueError("P6_NORMALIZED_CONTRIBUTION_RECONSTRUCTION_FAILED")
    rating = OrderedDict((group, 4.0 * value) for group, value in raw.items())
    rating_bias = 1.0 + 4.0 * task_head.bias.reshape(1, 1)
    score_1_to_5 = 1.0 + 4.0 * score
    rating_reconstructed = rating_bias + torch.stack(tuple(rating.values()), dim=0).sum(dim=0)
    if not torch.allclose(score_1_to_5, rating_reconstructed, rtol=0.0, atol=1e-6):
        raise ValueError("P6_RATING_CONTRIBUTION_RECONSTRUCTION_FAILED")
    return {
        "malignancy_raw_score": score,
        "malignancy_score_normalized": score,
        "malignancy_score_1_to_5": score_1_to_5,
        "raw_group_contributions": raw,
        "rating_point_contributions": rating,
        "raw_bias": task_head.bias.reshape(1, 1),
        "rating_scale_bias": rating_bias,
    }


def ensure_predicted_cache_features(frame: pd.DataFrame) -> np.ndarray:
    """Reject any task cache that is not explicitly activated predictor output."""
    required = {
        "nodule_uid",
        "canonical_activated_concepts",
        "feature_source",
        "feature_dimension",
    }
    if not required <= set(frame.columns):
        raise ValueError("P6_TASK_CACHE_SCHEMA_MISMATCH")
    if set(map(str, frame["feature_source"])) != {"frozen_predicted_activated_concepts"}:
        raise ValueError("P6_GROUND_TRUTH_CONCEPT_INJECTION_FORBIDDEN")
    if set(map(int, frame["feature_dimension"])) != {16}:
        raise ValueError("P6_TASK_CACHE_DIMENSION_MISMATCH")
    vectors = np.stack(
        [
            np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float32)
            for value in frame["canonical_activated_concepts"]
        ]
    )
    if vectors.shape != (len(frame), 16) or not np.isfinite(vectors).all():
        raise ValueError("P6_TASK_CACHE_VECTOR_INVALID")
    internal = vectors[:, CANONICAL_VECTOR_SLICES["internalStructure"]]
    calcification = vectors[:, CANONICAL_VECTOR_SLICES["calcification"]]
    continuous_indices = (0, 11, 12, 13, 14, 15)
    if (
        np.any(vectors[:, continuous_indices] < 0.0)
        or np.any(vectors[:, continuous_indices] > 1.0)
        or np.any(internal < 0.0)
        or np.any(internal > 1.0)
        or np.any(calcification < 0.0)
        or np.any(calcification > 1.0)
        or not np.allclose(internal.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        or not np.allclose(calcification.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("P6_TASK_CACHE_ACTIVATION_INVARIANT_FAILED")
    return vectors


def _ordered_concept_records(
    records: Sequence[ConceptRecord],
    base_seed: int,
    fold_index: int,
    epoch_index: int,
) -> list[ConceptRecord]:
    by_uid = {record.nodule_uid: record for record in records}
    if len(by_uid) != len(records):
        raise ValueError("P6_DUPLICATE_CONCEPT_RECORD_UID")
    return [
        by_uid[uid]
        for uid in epoch_uid_order(by_uid, base_seed, fold_index, epoch_index)
    ]


def _targets_to_device(targets: Mapping[str, Any], device: Any) -> OrderedDict[str, Any]:
    return OrderedDict(
        (group, targets[group].to(device=device, dtype=_torch().float32))
        for group in CONCEPT_GROUP_ORDER
    )


def train_concept_one_epoch(
    model: Any,
    records: Sequence[ConceptRecord],
    optimizer: Any,
    device: Any,
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Train one concept epoch and aggregate each group by sample count."""
    torch = _torch()
    ordered = _ordered_concept_records(
        records, base_seed, fold_index, epoch_index
    )
    dataset = ConceptROIDataset.build(
        ordered,
        training=True,
        base_seed=base_seed,
        fold_index=fold_index,
        epoch_index=epoch_index,
    )
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.train()
    group_sums = OrderedDict((group, 0.0) for group in CONCEPT_GROUP_ORDER)
    sample_count = 0
    observed_uids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        targets = _targets_to_device(batch["targets"], device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        loss, _group_means = concept_loss(outputs, targets)
        if not torch.isfinite(loss):
            raise ValueError("P6_NONFINITE_CONCEPT_TRAIN_LOSS")
        loss.backward()
        optimizer.step()
        sums, count = concept_group_loss_sums(outputs, targets)
        for group, value in sums.items():
            group_sums[group] += float(value.detach().cpu())
        sample_count += count
        observed_uids.extend(map(str, batch["nodule_uid"]))
    expected_uids = [record.nodule_uid for record in ordered]
    if observed_uids != expected_uids or len(set(observed_uids)) != len(records):
        raise ValueError("P6_CONCEPT_TRAIN_SAMPLE_COVERAGE_MISMATCH")
    if sample_count != len(records) or sample_count == 0:
        raise ValueError("P6_CONCEPT_TRAIN_SAMPLE_COUNT_MISMATCH")
    group_losses = OrderedDict(
        (group, value / sample_count) for group, value in group_sums.items()
    )
    return {
        "concept_loss": float(np.mean(tuple(group_losses.values()))),
        "group_losses": group_losses,
        "sample_count": sample_count,
        "nodule_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted(observed_uids))
        ),
    }


def evaluate_concept_records(
    model: Any,
    records: Sequence[ConceptRecord],
    device: Any,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Evaluate all concept groups without augmentation or mutable BN state."""
    torch = _torch()
    if len({record.nodule_uid for record in records}) != len(records):
        raise ValueError("P6_DUPLICATE_CONCEPT_RECORD_UID")
    ordered = sorted(records, key=lambda record: record.nodule_uid)
    dataset = ConceptROIDataset.build(
        ordered,
        training=False,
        base_seed=0,
        fold_index=0,
        epoch_index=0,
    )
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    group_sums = OrderedDict((group, 0.0) for group in CONCEPT_GROUP_ORDER)
    sample_count = 0
    observed_uids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            targets = _targets_to_device(batch["targets"], device)
            outputs = model(image)
            sums, count = concept_group_loss_sums(outputs, targets)
            for group, value in sums.items():
                group_sums[group] += float(value.cpu())
            sample_count += count
            observed_uids.extend(map(str, batch["nodule_uid"]))
    expected_uids = [record.nodule_uid for record in ordered]
    if observed_uids != expected_uids or sample_count != len(ordered) or not ordered:
        raise ValueError("P6_CONCEPT_EVALUATION_COVERAGE_MISMATCH")
    group_losses = OrderedDict(
        (group, value / sample_count) for group, value in group_sums.items()
    )
    return {
        "concept_loss": float(np.mean(tuple(group_losses.values()))),
        "group_losses": group_losses,
        "sample_count": sample_count,
        "nodule_set_sha256": sha256_bytes(canonical_json_bytes(sorted(observed_uids))),
    }


def _concept_record_targets(record: ConceptRecord) -> dict[str, Any]:
    continuous = dict(
        zip(CONTINUOUS_CONCEPTS, record.continuous_targets, strict=True)
    )
    return {
        "subtlety": continuous["subtlety"],
        "internalStructure": list(record.internal_structure_target),
        "calcification": list(record.calcification_target),
        "sphericity": continuous["sphericity"],
        "margin": continuous["margin"],
        "lobulation": continuous["lobulation"],
        "spiculation": continuous["spiculation"],
        "texture": continuous["texture"],
    }


def predict_concept_cache_frame(
    model: Any,
    records: Sequence[ConceptRecord],
    device: Any,
    *,
    partition: str,
    batch_size: int,
    num_workers: int,
    task_best_checkpoint_sha256: str | None = None,
) -> pd.DataFrame:
    """Generate no-augmentation frozen predictor outputs for one partition."""
    torch = _torch()
    if partition not in ("train", "validation", "test"):
        raise ValueError(f"P6_INVALID_CACHE_PARTITION:{partition}")
    if partition == "test":
        if (
            task_best_checkpoint_sha256 is None
            or len(task_best_checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in task_best_checkpoint_sha256
            )
        ):
            raise PermissionError(
                "P6_TEST_CONCEPT_GENERATION_BEFORE_TASK_BEST_FORBIDDEN"
            )
    elif task_best_checkpoint_sha256 is not None:
        raise ValueError("P6_TASK_BEST_PROOF_ONLY_VALID_FOR_TEST_PARTITION")
    if (
        any(module.training for module in model.modules())
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError("P6_CACHE_REQUIRES_FROZEN_EVAL_PREDICTOR")
    if len({record.nodule_uid for record in records}) != len(records):
        raise ValueError("P6_DUPLICATE_CONCEPT_RECORD_UID")
    semantic_before = module_state_sha256(model)
    batchnorm_before = batchnorm_state_sha256(model)
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
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            outputs = model(image)
            vector = outputs["canonical_vector"].detach().cpu().numpy()
            logits = {
                group: outputs["logits"][group].detach().cpu().numpy()
                for group in CONCEPT_GROUP_ORDER
            }
            activated = {
                group: outputs["activated"][group].detach().cpu().numpy()
                for group in CONCEPT_GROUP_ORDER
            }
            for batch_index, uid in enumerate(map(str, batch["nodule_uid"])):
                record = by_uid[uid]
                row: dict[str, Any] = {
                    "nodule_uid": uid,
                    "patient_key": record.patient_key,
                    "partition": partition,
                    "feature_source": "frozen_predicted_activated_concepts",
                    "feature_dimension": 16,
                    "canonical_activated_concepts": json.dumps(
                        vector[batch_index].tolist(), separators=(",", ":")
                    ),
                    "target_normalized": record.target_normalized,
                    "target_1_to_5": record.target_1_to_5,
                    "extreme_binary_eligible": record.extreme_binary_eligible,
                    "extreme_binary_label": record.extreme_binary_label,
                    "concept_targets": json.dumps(
                        _concept_record_targets(record), separators=(",", ":"), sort_keys=True
                    ),
                    "valid_reader_counts": json.dumps(
                        dict(
                            zip(
                                CONCEPT_GROUP_ORDER,
                                record.valid_reader_counts,
                                strict=True,
                            )
                        ),
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    "internalStructure_modal_tie": record.categorical_ties[0],
                    "calcification_modal_tie": record.categorical_ties[1],
                }
                for group in CONCEPT_GROUP_ORDER:
                    row[f"{group}_logits"] = json.dumps(
                        logits[group][batch_index].tolist(), separators=(",", ":")
                    )
                    row[f"{group}_activated_prediction"] = json.dumps(
                        activated[group][batch_index].tolist(), separators=(",", ":")
                    )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if list(map(str, frame.get("nodule_uid", []))) != [
        record.nodule_uid for record in ordered
    ]:
        raise ValueError("P6_CACHE_PREDICTION_COVERAGE_MISMATCH")
    if (
        module_state_sha256(model) != semantic_before
        or batchnorm_state_sha256(model) != batchnorm_before
    ):
        raise ValueError("P6_FROZEN_PREDICTOR_CHANGED_DURING_CACHE_GENERATION")
    ensure_predicted_cache_features(frame)
    return frame


def _uid_set_sha256(uids: Sequence[str]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(map(str, uids))))


def validate_cache_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the complete cache-to-source and frozen-predictor provenance."""
    missing = set(CACHE_PROVENANCE_HASH_KEYS) - set(provenance)
    if missing or "fold_index" not in provenance or "concept_head_initialization_sha256" not in provenance:
        raise ValueError("P6_CACHE_PROVENANCE_REQUIRED_FIELDS_MISSING")
    result = dict(provenance)
    for key in CACHE_PROVENANCE_HASH_KEYS:
        value = result[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"P6_CACHE_PROVENANCE_HASH_INVALID:{key}")
    fold_index = result["fold_index"]
    if not isinstance(fold_index, int) or isinstance(fold_index, bool) or fold_index not in range(5):
        raise ValueError("P6_CACHE_PROVENANCE_FOLD_INVALID")
    head_hashes = result["concept_head_initialization_sha256"]
    if not isinstance(head_hashes, Mapping) or tuple(head_hashes) != CONCEPT_GROUP_ORDER:
        raise ValueError("P6_CACHE_PROVENANCE_HEAD_HASHES_INVALID")
    for group, value in head_hashes.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"P6_CACHE_PROVENANCE_HEAD_HASH_INVALID:{group}")
    return result


def _cache_manifest_payload(
    cache_directory: Path,
    frames: Mapping[str, pd.DataFrame],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    validated_provenance = validate_cache_provenance(provenance)
    partitions: dict[str, Any] = {}
    for partition in ("train", "validation"):
        frame = frames[partition]
        path = cache_directory / f"{partition}.parquet"
        partitions[partition] = {
            "rows": int(len(frame)),
            "uid_set_sha256": _uid_set_sha256(frame["nodule_uid"].astype(str).tolist()),
            "cache_file_sha256": sha256_file(path),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "TRAIN_VALIDATION_FROZEN_PREDICTION_CACHE_COMPLETE",
        "allowed_partitions": ["train", "validation"],
        "test_cache_generated": False,
        "feature_source": "frozen_predicted_activated_concepts",
        "feature_dimension": 16,
        "provenance": validated_provenance,
        "partitions": partitions,
    }


def verify_train_validation_caches(
    cache_directory: str | Path,
    expected_uids: Mapping[str, Sequence[str]],
    expected_provenance: Mapping[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Verify exact sets, files, source semantics and provenance before Stage 2."""
    directory = Path(cache_directory)
    validated_provenance = validate_cache_provenance(expected_provenance)
    if (directory / "test.parquet").exists():
        raise ValueError("P6_PRETASK_TEST_CACHE_FORBIDDEN")
    manifest_path = directory / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("P6_CACHE_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status")
        != "TRAIN_VALIDATION_FROZEN_PREDICTION_CACHE_COMPLETE"
        or manifest.get("allowed_partitions") != ["train", "validation"]
        or manifest.get("test_cache_generated") is not False
        or manifest.get("feature_source")
        != "frozen_predicted_activated_concepts"
        or manifest.get("feature_dimension") != 16
        or manifest.get("provenance") != validated_provenance
    ):
        raise ValueError("P6_CACHE_MANIFEST_PROVENANCE_MISMATCH")
    frames: dict[str, pd.DataFrame] = {}
    for partition in ("train", "validation"):
        path = directory / f"{partition}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"P6_CACHE_FILE_MISSING:{partition}")
        expected_partition = manifest["partitions"][partition]
        if sha256_file(path) != expected_partition["cache_file_sha256"]:
            raise ValueError(f"P6_CACHE_FILE_HASH_MISMATCH:{partition}")
        frame = pd.read_parquet(path)
        ensure_predicted_cache_features(frame)
        observed = list(map(str, frame["nodule_uid"]))
        expected = list(map(str, expected_uids[partition]))
        if (
            len(observed) != len(set(observed))
            or len(expected) != len(set(expected))
            or set(observed) != set(expected)
            or len(observed) != len(expected)
            or int(expected_partition["rows"]) != len(observed)
            or expected_partition["uid_set_sha256"] != _uid_set_sha256(observed)
        ):
            raise ValueError(f"P6_CACHE_UID_SET_MISMATCH:{partition}")
        if set(map(str, frame["partition"])) != {partition}:
            raise ValueError(f"P6_CACHE_PARTITION_LABEL_MISMATCH:{partition}")
        frames[partition] = frame
    return frames, manifest


def write_train_validation_caches(
    cache_directory: str | Path,
    frames: Mapping[str, pd.DataFrame],
    expected_uids: Mapping[str, Sequence[str]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist only train/validation frozen-prediction caches."""
    if set(frames) != {"train", "validation"}:
        raise ValueError("P6_PRETASK_CACHE_PARTITIONS_MUST_BE_TRAIN_VALIDATION_ONLY")
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "test.parquet").exists():
        raise FileExistsError("P6_PRETASK_TEST_CACHE_FORBIDDEN")
    expected_paths = (
        directory / "train.parquet",
        directory / "validation.parquet",
        directory / "cache_manifest.json",
    )
    existing = tuple(path.is_file() for path in expected_paths)
    if any(existing):
        if not all(existing):
            raise FileExistsError("P6_PARTIAL_CACHE_BUNDLE_REQUIRES_AUDIT")
        _frames, manifest = verify_train_validation_caches(
            directory, expected_uids, provenance
        )
        return manifest
    for partition in ("train", "validation"):
        frame = frames[partition].copy()
        ensure_predicted_cache_features(frame)
        observed = list(map(str, frame["nodule_uid"]))
        expected = list(map(str, expected_uids[partition]))
        if (
            len(observed) != len(set(observed))
            or len(expected) != len(set(expected))
            or set(observed) != set(expected)
            or len(observed) != len(expected)
        ):
            raise ValueError(f"P6_CACHE_UID_SET_MISMATCH:{partition}")
        _atomic_parquet(directory / f"{partition}.parquet", frame)
    manifest = _cache_manifest_payload(directory, frames, provenance)
    _atomic_json(directory / "cache_manifest.json", manifest)
    verify_train_validation_caches(directory, expected_uids, provenance)
    return manifest


@dataclass(frozen=True)
class TaskCacheRecord:
    nodule_uid: str
    activated_concepts: tuple[float, ...]
    target_normalized: float


def task_cache_records(
    frame: pd.DataFrame,
    expected_uids: Sequence[str],
) -> list[TaskCacheRecord]:
    vectors = ensure_predicted_cache_features(frame)
    observed = list(map(str, frame["nodule_uid"]))
    expected = list(map(str, expected_uids))
    if (
        len(observed) != len(set(observed))
        or len(expected) != len(set(expected))
        or len(observed) != len(expected)
        or set(observed) != set(expected)
    ):
        raise ValueError("P6_TASK_CACHE_UID_SET_MISMATCH")
    by_uid = {
        uid: TaskCacheRecord(
            nodule_uid=uid,
            activated_concepts=tuple(map(float, vectors[index])),
            target_normalized=float(frame.iloc[index]["target_normalized"]),
        )
        for index, uid in enumerate(observed)
    }
    return [by_uid[uid] for uid in expected]


class TaskCacheDataset:
    @staticmethod
    def build(records: Sequence[TaskCacheRecord]) -> Any:
        torch = _torch()

        class Dataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return len(records)

            def __getitem__(self, index: int) -> dict[str, Any]:
                record = records[index]
                return {
                    "activated_concepts": torch.tensor(
                        record.activated_concepts, dtype=torch.float32
                    ),
                    "target": torch.tensor(
                        [record.target_normalized], dtype=torch.float32
                    ),
                    "nodule_uid": record.nodule_uid,
                }

        return Dataset()


def train_task_one_epoch(
    task_head: Any,
    records: Sequence[TaskCacheRecord],
    optimizer: Any,
    device: Any,
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Train only Linear(16,1) from frozen predicted activated concepts."""
    torch = _torch()
    by_uid = {record.nodule_uid: record for record in records}
    if len(by_uid) != len(records):
        raise ValueError("P6_DUPLICATE_TASK_CACHE_UID")
    order = epoch_uid_order(by_uid, base_seed, fold_index, epoch_index)
    ordered = [by_uid[uid] for uid in order]
    loader = _loader(
        TaskCacheDataset.build(ordered),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    task_head.train()
    squared_error_sum = 0.0
    sample_count = 0
    observed: list[str] = []
    for batch in loader:
        features = batch["activated_concepts"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        score = task_head(features)
        if score.shape != target.shape:
            raise ValueError("P6_TASK_OUTPUT_SHAPE_MISMATCH")
        loss = torch.nn.functional.mse_loss(score, target, reduction="mean")
        if not torch.isfinite(loss):
            raise ValueError("P6_NONFINITE_TASK_TRAIN_LOSS")
        loss.backward()
        optimizer.step()
        squared_error_sum += float(
            torch.nn.functional.mse_loss(
                score.detach(), target, reduction="sum"
            ).cpu()
        )
        sample_count += int(target.shape[0])
        observed.extend(map(str, batch["nodule_uid"]))
    if observed != order or sample_count != len(records) or len(set(observed)) != len(records):
        raise ValueError("P6_TASK_TRAIN_SAMPLE_COVERAGE_MISMATCH")
    return {
        "mse": squared_error_sum / sample_count,
        "sample_count": sample_count,
        "nodule_set_sha256": _uid_set_sha256(observed),
    }


def evaluate_task_records(
    task_head: Any,
    records: Sequence[TaskCacheRecord],
    device: Any,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    torch = _torch()
    if len({record.nodule_uid for record in records}) != len(records):
        raise ValueError("P6_DUPLICATE_TASK_CACHE_UID")
    ordered = sorted(records, key=lambda record: record.nodule_uid)
    loader = _loader(
        TaskCacheDataset.build(ordered),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    task_head.eval()
    squared_error_sum = 0.0
    observed: list[str] = []
    with torch.no_grad():
        for batch in loader:
            features = batch["activated_concepts"].to(
                device=device, dtype=torch.float32
            )
            target = batch["target"].to(device=device, dtype=torch.float32)
            score = task_head(features)
            if not torch.isfinite(score).all():
                raise ValueError("P6_NONFINITE_TASK_PREDICTION")
            squared_error_sum += float(
                torch.nn.functional.mse_loss(score, target, reduction="sum").cpu()
            )
            observed.extend(map(str, batch["nodule_uid"]))
    if observed != [record.nodule_uid for record in ordered] or not observed:
        raise ValueError("P6_TASK_EVALUATION_COVERAGE_MISMATCH")
    return {
        "mse": squared_error_sum / len(observed),
        "sample_count": len(observed),
        "nodule_set_sha256": _uid_set_sha256(observed),
    }


def task_optimizer(task_head: Any, execution_config: Mapping[str, Any]) -> Any:
    optimizer = _optimizer(task_head, execution_config)
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {id(parameter) for parameter in task_head.parameters()}
    if optimized != expected:
        raise ValueError("P6_TASK_OPTIMIZER_PARAMETER_SCOPE_MISMATCH")
    return optimizer


def _p6_provenance(
    scientific_config: Mapping[str, Any],
    execution_config: Mapping[str, Any],
    execution_config_sha256: str,
    p6_execution_config_sha256: str,
    split: Mapping[str, Any],
    initialization: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    from lidc_baseline.p5_blackbox import reproducibility_provenance

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": scientific_config["protocol"]["version"],
        "scientific_config_sha256": compute_config_sha256(scientific_config),
        "execution_config_sha256": execution_config_sha256,
        "p6_execution_config_sha256": p6_execution_config_sha256,
        "split_sha256": split["split_sha256"],
        "fold_index": int(split["fold_index"]),
        "model": MODEL_NAME,
        "stage": stage,
        "execution_profile_id": execution_config["execution_profile"]["profile_id"],
        "formal_gpu_model": execution_config["execution_profile"]["formal_gpu_model"],
        **reproducibility_provenance(execution_config),
        **dict(initialization),
    }


def _load_p6_sources(
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
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
    (
        scientific,
        execution,
        execution_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _prepare_sources(
        scientific_config_path,
        execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    p6_config, p6_hash = validate_p6_execution_config(p6_execution_config_path)
    if p6_config["scientific_config"] != {
        "path": str(scientific_config_path),
        "sha256": compute_config_sha256(scientific),
    }:
        raise ValueError("P6_SUPPLEMENT_SCIENTIFIC_CONFIG_MISMATCH")
    common = p6_config["common_execution_profile"]
    if (
        common["path"] != str(execution_config_path)
        or common["resolved_sha256"] != execution_hash
        or common["formal_gpu_model"]
        != execution["execution_profile"]["formal_gpu_model"]
    ):
        raise ValueError("P6_SUPPLEMENT_COMMON_EXECUTION_MISMATCH")
    return (
        scientific,
        execution,
        execution_hash,
        p6_config,
        p6_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    )


def run_directory(
    fold_index: int,
    root: str | Path = "runs/baseline_v2/standard_cbm",
) -> Path:
    return Path(root) / f"fold_{fold_index}"


def _stage_checkpoint_payload(
    module: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    epoch_index: int,
    validation_objective: float,
    best_epoch_index: int,
    best_validation_objective: float,
    provenance: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "epoch_index": int(epoch_index),
        "validation_objective": float(validation_objective),
        "best_epoch_index": int(best_epoch_index),
        "best_validation_objective": float(best_validation_objective),
        "module_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "history": [dict(row) for row in history],
    }


def _load_stage_checkpoint(
    path: Path,
    expected_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("P6_STAGE_CHECKPOINT_SCHEMA_MISMATCH")
    if payload.get("provenance") != dict(expected_provenance):
        raise ValueError("P6_STAGE_CHECKPOINT_PROVENANCE_MISMATCH")
    return payload


def _flatten_epoch_report(prefix: str, report: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, Mapping):
            for nested_key, nested_value in value.items():
                flattened[f"{prefix}_{nested_key}_loss"] = nested_value
        else:
            flattened[f"{prefix}_{key}"] = value
    return flattened


def run_training_stage(
    *,
    stage_name: str,
    module: Any,
    optimizer: Any,
    scheduler: Any,
    provenance: Mapping[str, Any],
    output_directory: Path,
    epochs: int,
    objective_key: str,
    expected_train_samples: int,
    train_epoch: Any,
    validate_epoch: Any,
    device: Any,
    resume: bool,
    _stop_after_epoch_for_test: int | None = None,
) -> dict[str, Any]:
    """Run one exact checkpointed stage with epoch-boundary deterministic resume."""
    torch = _torch()
    if epochs != 80:
        raise ValueError("P6_STAGE_EPOCH_BUDGET_MISMATCH")
    output_directory.mkdir(parents=True, exist_ok=True)
    last_path = output_directory / "last.pt"
    best_path = output_directory / "best.pt"
    history_path = output_directory / "history.csv"
    complete_path = output_directory / "training_complete.json"
    if complete_path.exists():
        return verify_training_stage(
            stage_name=stage_name,
            output_directory=output_directory,
            expected_provenance=provenance,
            expected_epochs=epochs,
            objective_key=objective_key,
            expected_train_samples=expected_train_samples,
        )
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    best_objective = math.inf
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"P6_{stage_name.upper()}_RESUME_CHECKPOINT_MISSING")
        payload = _load_stage_checkpoint(last_path, provenance)
        module.load_state_dict(payload["module_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
        history = [dict(row) for row in payload["history"]]
        start_epoch = int(payload["epoch_index"]) + 1
        best_epoch = int(payload["best_epoch_index"])
        best_objective = float(payload["best_validation_objective"])
        if len(history) != start_epoch:
            raise ValueError(f"P6_{stage_name.upper()}_RESUME_HISTORY_MISMATCH")
    elif any(path.exists() for path in (last_path, best_path, history_path)):
        raise FileExistsError(f"P6_{stage_name.upper()}_RUN_EXISTS_USE_RESUME")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch_index in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        learning_rate_start = float(optimizer.param_groups[0]["lr"])
        train_report = train_epoch(epoch_index)
        validation_report = validate_epoch()
        validation_objective = float(validation_report[objective_key])
        if not math.isfinite(validation_objective):
            raise ValueError(f"P6_{stage_name.upper()}_NONFINITE_VALIDATION_OBJECTIVE")
        improved = checkpoint_improves(validation_objective, best_objective)
        if improved:
            best_epoch = epoch_index
            best_objective = validation_objective
        decayed = scheduler.step(validation_objective)
        row = {
            "epoch_index": epoch_index,
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
            "scheduler_decayed": bool(decayed),
            "scheduler_best": scheduler.best,
            "scheduler_bad_epoch_counter": scheduler.bad_epoch_counter,
            **_flatten_epoch_report("train", train_report),
            **_flatten_epoch_report("validation", validation_report),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        if int(train_report["sample_count"]) != expected_train_samples:
            raise ValueError(f"P6_{stage_name.upper()}_TRAIN_COVERAGE_MISMATCH")
        if improved:
            _atomic_torch_save(
                best_path,
                _stage_checkpoint_payload(
                    module,
                    optimizer,
                    scheduler,
                    epoch_index=epoch_index,
                    validation_objective=validation_objective,
                    best_epoch_index=best_epoch,
                    best_validation_objective=best_objective,
                    provenance=provenance,
                    history=history,
                ),
            )
        _atomic_csv(history_path, history, list(row))
        _atomic_torch_save(
            last_path,
            _stage_checkpoint_payload(
                module,
                optimizer,
                scheduler,
                epoch_index=epoch_index,
                validation_objective=validation_objective,
                best_epoch_index=best_epoch,
                best_validation_objective=best_objective,
                provenance=provenance,
                history=history,
            ),
        )
        print(
            canonical_json_bytes(
                {
                    "event": "P6_STAGE_EPOCH_COMPLETE",
                    "stage": stage_name,
                    "fold_index": provenance["fold_index"],
                    **row,
                }
            ).decode("utf-8").strip(),
            flush=True,
        )
        if _stop_after_epoch_for_test == epoch_index:
            return {
                "status": "INTERRUPTED_AT_EPOCH_BOUNDARY_FOR_TEST",
                "stage": stage_name,
                "epoch_index": epoch_index,
                "last_checkpoint_sha256": sha256_file(last_path),
            }
    if len(history) != epochs or not best_path.is_file():
        raise ValueError(f"P6_{stage_name.upper()}_TRAINING_INCOMPLETE")
    runtime = {
        **dict(provenance),
        **_runtime_environment(device),
        "stage": stage_name,
        "epochs_total": epochs,
        "wall_seconds_this_invocation": time.monotonic() - started,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
    }
    _atomic_json(output_directory / "runtime.json", runtime)
    completion = {
        **dict(provenance),
        "status": "STAGE_TRAINING_COMPLETE",
        "stage": stage_name,
        "epochs_completed": epochs,
        "objective_key": objective_key,
        "best_epoch_index": best_epoch,
        "best_validation_objective": best_objective,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history_sha256": sha256_file(history_path),
        "runtime_sha256": sha256_file(output_directory / "runtime.json"),
        "test_evaluated": False,
    }
    _atomic_json(complete_path, completion)
    return completion


def stage_resume_requested(resume: bool, output_directory: Path) -> bool:
    """Resume only a stage that has its own valid epoch-boundary checkpoint."""
    complete = output_directory / "training_complete.json"
    last = output_directory / "last.pt"
    if complete.is_file():
        return False
    if not resume:
        return False
    if last.is_file():
        return True
    partial = tuple(
        path.exists()
        for path in (
            output_directory / "best.pt",
            output_directory / "history.csv",
            output_directory / "runtime.json",
        )
    )
    if any(partial):
        raise FileExistsError("P6_STAGE_PARTIAL_STATE_WITHOUT_LAST_CHECKPOINT")
    return False


def verify_training_stage(
    *,
    stage_name: str,
    output_directory: Path,
    expected_provenance: Mapping[str, Any],
    expected_epochs: int,
    objective_key: str,
    expected_train_samples: int,
    require_h200_runtime: bool = False,
    expected_train_nodule_set_sha256: str | None = None,
    expected_validation_samples: int | None = None,
    expected_validation_nodule_set_sha256: str | None = None,
) -> dict[str, Any]:
    """Reconstruct checkpoint selection and epoch coverage from saved artifacts."""
    complete_path = output_directory / "training_complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"P6_{stage_name.upper()}_COMPLETION_MISSING")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if any(completion.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError(f"P6_{stage_name.upper()}_COMPLETION_PROVENANCE_MISMATCH")
    if (
        completion.get("status") != "STAGE_TRAINING_COMPLETE"
        or completion.get("stage") != stage_name
        or completion.get("epochs_completed") != expected_epochs
        or completion.get("objective_key") != objective_key
        or completion.get("test_evaluated") is not False
    ):
        raise ValueError(f"P6_{stage_name.upper()}_COMPLETION_SCHEMA_MISMATCH")
    history_path = output_directory / "history.csv"
    with history_path.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    if len(history) != expected_epochs:
        raise ValueError(f"P6_{stage_name.upper()}_HISTORY_LENGTH_MISMATCH")
    objective_column = f"validation_{objective_key}"
    objectives = [float(row[objective_column]) for row in history]
    expected_best_epoch = int(np.argmin(np.asarray(objectives, dtype=np.float64)))
    expected_best = objectives[expected_best_epoch]
    if int(completion["best_epoch_index"]) != expected_best_epoch or not serialized_float_consistent(
        float(completion["best_validation_objective"]), expected_best
    ):
        raise ValueError(f"P6_{stage_name.upper()}_BEST_OBJECTIVE_MISMATCH")
    if any(int(row["train_sample_count"]) != expected_train_samples for row in history):
        raise ValueError(f"P6_{stage_name.upper()}_TRAIN_COVERAGE_MISMATCH")
    if expected_train_nodule_set_sha256 is not None and any(
        row["train_nodule_set_sha256"] != expected_train_nodule_set_sha256
        for row in history
    ):
        raise ValueError(f"P6_{stage_name.upper()}_TRAIN_UID_SET_HASH_MISMATCH")
    if expected_validation_samples is not None and any(
        int(row["validation_sample_count"]) != expected_validation_samples
        for row in history
    ):
        raise ValueError(f"P6_{stage_name.upper()}_VALIDATION_COVERAGE_MISMATCH")
    if expected_validation_nodule_set_sha256 is not None and any(
        row["validation_nodule_set_sha256"]
        != expected_validation_nodule_set_sha256
        for row in history
    ):
        raise ValueError(f"P6_{stage_name.upper()}_VALIDATION_UID_SET_HASH_MISMATCH")
    best_path = output_directory / "best.pt"
    last_path = output_directory / "last.pt"
    runtime_path = output_directory / "runtime.json"
    for path, field in (
        (best_path, "best_checkpoint_sha256"),
        (last_path, "last_checkpoint_sha256"),
        (history_path, "history_sha256"),
        (runtime_path, "runtime_sha256"),
    ):
        if sha256_file(path) != completion[field]:
            raise ValueError(f"P6_{stage_name.upper()}_ARTIFACT_HASH_MISMATCH:{field}")
    best = _load_stage_checkpoint(best_path, expected_provenance)
    if (
        int(best["epoch_index"]) != expected_best_epoch
        or not serialized_float_consistent(
            float(best["validation_objective"]), expected_best
        )
        or int(best["best_epoch_index"]) != expected_best_epoch
        or not serialized_float_consistent(
            float(best["best_validation_objective"]), expected_best
        )
    ):
        raise ValueError(f"P6_{stage_name.upper()}_BEST_CHECKPOINT_MISMATCH")
    last = _load_stage_checkpoint(last_path, expected_provenance)
    if (
        int(last["epoch_index"]) != expected_epochs - 1
        or len(last.get("history", [])) != expected_epochs
        or int(last["best_epoch_index"]) != expected_best_epoch
        or not serialized_float_consistent(
            float(last["best_validation_objective"]), expected_best
        )
    ):
        raise ValueError(f"P6_{stage_name.upper()}_LAST_CHECKPOINT_MISMATCH")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if any(runtime.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError(f"P6_{stage_name.upper()}_RUNTIME_PROVENANCE_MISMATCH")
    if require_h200_runtime and (
        runtime.get("device_type") != "cuda"
        or "H200" not in str(runtime.get("gpu_name", "")).upper()
        or runtime.get("fp32") is not True
        or runtime.get("amp_enabled") is not False
        or runtime.get("bfloat16_enabled") is not False
        or runtime.get("cuda_matmul_tf32_enabled") is not False
        or runtime.get("cudnn_tf32_enabled") is not False
        or runtime.get("torch_use_deterministic_algorithms") is not True
        or runtime.get("deterministic_algorithms_warn_only") is not True
        or runtime.get("epochs_total") != expected_epochs
    ):
        raise ValueError(f"P6_{stage_name.upper()}_FORMAL_RUNTIME_POLICY_MISMATCH")
    return completion


def _cache_provenance_from_run(
    *,
    scientific_config: Mapping[str, Any],
    execution_config_sha256: str,
    p6_execution_config_sha256: str,
    split: Mapping[str, Any],
    initialization: Mapping[str, Any],
    concept_best_checkpoint_path: Path,
    predictor: Any,
    manifest_path: Path,
    roi_index_path: Path,
) -> dict[str, Any]:
    return validate_cache_provenance(
        {
            "scientific_config_sha256": compute_config_sha256(scientific_config),
            "execution_config_sha256": execution_config_sha256,
            "p6_execution_config_sha256": p6_execution_config_sha256,
            "split_sha256": split["split_sha256"],
            "fold_index": int(split["fold_index"]),
            "encoder_initialization_sha256": initialization[
                "encoder_initialization_sha256"
            ],
            "encoder_artifact_file_sha256": initialization[
                "encoder_artifact_file_sha256"
            ],
            "concept_head_initialization_sha256": initialization[
                "concept_head_initialization_sha256"
            ],
            "combined_concept_head_initialization_sha256": initialization[
                "combined_concept_head_initialization_sha256"
            ],
            "concept_best_checkpoint_sha256": sha256_file(
                concept_best_checkpoint_path
            ),
            "predictor_semantic_sha256": module_state_sha256(predictor),
            "batchnorm_state_sha256": batchnorm_state_sha256(predictor),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_roi_index_sha256": sha256_file(roi_index_path),
        }
    )


def _load_concept_best(
    *,
    scientific: Mapping[str, Any],
    split: Mapping[str, Any],
    encoder_path: Path,
    concept_best_path: Path,
    provenance: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    model, initialization = build_initialized_concept_predictor(
        scientific, split, encoder_path
    )
    payload = _load_stage_checkpoint(concept_best_path, provenance)
    model.load_state_dict(payload["module_state_dict"], strict=True)
    freeze_concept_predictor(model)
    return model, initialization


def train_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    resume: bool,
    _stop_concept_after_epoch_for_test: int | None = None,
    _stop_task_after_epoch_for_test: int | None = None,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p6_lifecycle.lock"):
        return _train_fold_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p6_execution_config_path=p6_execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            device_name=device_name,
            num_workers=num_workers,
            output_root=output_root,
            resume=resume,
            _stop_concept_after_epoch_for_test=_stop_concept_after_epoch_for_test,
            _stop_task_after_epoch_for_test=_stop_task_after_epoch_for_test,
        )


def _train_fold_locked(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
    resume: bool,
    _stop_concept_after_epoch_for_test: int | None,
    _stop_task_after_epoch_for_test: int | None,
) -> dict[str, Any]:
    torch = _torch()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    (
        scientific,
        execution,
        execution_hash,
        _p6_config,
        p6_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_p6_sources(
        scientific_config_path,
        execution_config_path,
        p6_execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    concept_model, initialization = build_initialized_concept_predictor(
        scientific, split, encoder_path
    )
    seed_training(int(initialization["fold_seed"]))
    concept_model.to(device)
    concept_optimizer = _optimizer(concept_model, execution)
    concept_scheduler = _scheduler(concept_optimizer, execution)
    concept_provenance = _p6_provenance(
        scientific,
        execution,
        execution_hash,
        p6_hash,
        split,
        initialization,
        stage="concept",
    )
    train_records = build_partition_concept_records(
        manifest, roi_index, split, "train", roi_index_path
    )
    validation_records = build_partition_concept_records(
        manifest, roi_index, split, "validation", roi_index_path
    )
    output = run_directory(fold_index, output_root)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "sequential_training_complete.json").exists():
        context = _prepare_trained_context(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p6_execution_config_path=p6_execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            output_root=output_root,
            device=device,
        )
        return context["sequential"]
    batch_size = int(
        execution["project_preregistered"]["batching"]["micro_batch_size"]
    )
    epochs = int(execution["reference_reported"]["epochs"])
    concept_completion = run_training_stage(
        stage_name="concept",
        module=concept_model,
        optimizer=concept_optimizer,
        scheduler=concept_scheduler,
        provenance=concept_provenance,
        output_directory=output / "concept_stage",
        epochs=epochs,
        objective_key="concept_loss",
        expected_train_samples=len(train_records),
        train_epoch=lambda epoch: train_concept_one_epoch(
            concept_model,
            train_records,
            concept_optimizer,
            device,
            base_seed=int(scientific["reproducibility"]["base_seed"]),
            fold_index=fold_index,
            epoch_index=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        validate_epoch=lambda: evaluate_concept_records(
            concept_model,
            validation_records,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        device=device,
        resume=stage_resume_requested(resume, output / "concept_stage"),
        _stop_after_epoch_for_test=_stop_concept_after_epoch_for_test,
    )
    if concept_completion.get("status") != "STAGE_TRAINING_COMPLETE":
        return concept_completion
    concept_best_path = output / "concept_stage" / "best.pt"
    concept_model, _rebuilt_initialization = _load_concept_best(
        scientific=scientific,
        split=split,
        encoder_path=encoder_path,
        concept_best_path=concept_best_path,
        provenance=concept_provenance,
    )
    concept_model.to(device)
    frozen_predictor_hash = module_state_sha256(concept_model)
    frozen_bn_hash = batchnorm_state_sha256(concept_model)
    cache_provenance = _cache_provenance_from_run(
        scientific_config=scientific,
        execution_config_sha256=execution_hash,
        p6_execution_config_sha256=p6_hash,
        split=split,
        initialization=initialization,
        concept_best_checkpoint_path=concept_best_path,
        predictor=concept_model,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
    )
    cache_directory = output / "concept_cache"
    expected_cache_uids = {
        "train": list(map(str, split["partitions"]["train"]["nodule_uids"])),
        "validation": list(
            map(str, split["partitions"]["validation"]["nodule_uids"])
        ),
    }
    cache_paths = (
        cache_directory / "train.parquet",
        cache_directory / "validation.parquet",
        cache_directory / "cache_manifest.json",
    )
    if all(path.is_file() for path in cache_paths):
        cache_frames, cache_manifest = verify_train_validation_caches(
            cache_directory, expected_cache_uids, cache_provenance
        )
    elif any(path.exists() for path in cache_paths):
        raise FileExistsError("P6_PARTIAL_CACHE_BUNDLE_REQUIRES_AUDIT")
    else:
        cache_frames = {
            "train": predict_concept_cache_frame(
                concept_model,
                train_records,
                device,
                partition="train",
                batch_size=batch_size,
                num_workers=num_workers,
            ),
            "validation": predict_concept_cache_frame(
                concept_model,
                validation_records,
                device,
                partition="validation",
                batch_size=batch_size,
                num_workers=num_workers,
            ),
        }
        cache_manifest = write_train_validation_caches(
            cache_directory,
            cache_frames,
            expected_cache_uids,
            cache_provenance,
        )
    train_task_records = task_cache_records(
        cache_frames["train"], expected_cache_uids["train"]
    )
    validation_task_records = task_cache_records(
        cache_frames["validation"], expected_cache_uids["validation"]
    )
    task_head, task_initialization = build_deterministic_task_head(
        int(initialization["fold_seed"])
    )
    task_initialization.update(
        {
            "fold_seed": initialization["fold_seed"],
            "encoder_initialization_sha256": initialization[
                "encoder_initialization_sha256"
            ],
            "concept_head_initialization_sha256": initialization[
                "concept_head_initialization_sha256"
            ],
            "combined_concept_head_initialization_sha256": initialization[
                "combined_concept_head_initialization_sha256"
            ],
            "concept_best_checkpoint_sha256": sha256_file(concept_best_path),
            "frozen_predictor_semantic_sha256": frozen_predictor_hash,
            "frozen_batchnorm_state_sha256": frozen_bn_hash,
            "cache_manifest_sha256": sha256_file(
                cache_directory / "cache_manifest.json"
            ),
        }
    )
    task_provenance = _p6_provenance(
        scientific,
        execution,
        execution_hash,
        p6_hash,
        split,
        task_initialization,
        stage="task",
    )
    task_head.to(device)
    task_stage_optimizer = task_optimizer(task_head, execution)
    task_scheduler = _scheduler(task_stage_optimizer, execution)
    task_completion = run_training_stage(
        stage_name="task",
        module=task_head,
        optimizer=task_stage_optimizer,
        scheduler=task_scheduler,
        provenance=task_provenance,
        output_directory=output / "task_stage",
        epochs=epochs,
        objective_key="mse",
        expected_train_samples=len(train_task_records),
        train_epoch=lambda epoch: train_task_one_epoch(
            task_head,
            train_task_records,
            task_stage_optimizer,
            device,
            base_seed=int(scientific["reproducibility"]["base_seed"]),
            fold_index=fold_index,
            epoch_index=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        validate_epoch=lambda: evaluate_task_records(
            task_head,
            validation_task_records,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        device=device,
        resume=stage_resume_requested(resume, output / "task_stage"),
        _stop_after_epoch_for_test=_stop_task_after_epoch_for_test,
    )
    if task_completion.get("status") != "STAGE_TRAINING_COMPLETE":
        return task_completion
    if (
        module_state_sha256(concept_model) != frozen_predictor_hash
        or batchnorm_state_sha256(concept_model) != frozen_bn_hash
    ):
        raise ValueError("P6_CONCEPT_PREDICTOR_CHANGED_DURING_TASK_STAGE")
    sequential = {
        **_p6_provenance(
            scientific,
            execution,
            execution_hash,
            p6_hash,
            split,
            task_initialization,
            stage="sequential",
        ),
        "status": "SEQUENTIAL_TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "concept_completion_sha256": sha256_file(
            output / "concept_stage" / "training_complete.json"
        ),
        "cache_manifest_sha256": sha256_file(
            cache_directory / "cache_manifest.json"
        ),
        "task_completion_sha256": sha256_file(
            output / "task_stage" / "training_complete.json"
        ),
        "frozen_predictor_semantic_sha256_before_task": frozen_predictor_hash,
        "frozen_predictor_semantic_sha256_after_task": module_state_sha256(
            concept_model
        ),
        "frozen_batchnorm_state_sha256_before_task": frozen_bn_hash,
        "frozen_batchnorm_state_sha256_after_task": batchnorm_state_sha256(
            concept_model
        ),
        "cache_partitions": cache_manifest["allowed_partitions"],
        "test_concepts_generated": False,
        "test_evaluated": False,
    }
    _atomic_json(output / "sequential_training_complete.json", sequential)
    return sequential


def _prepare_trained_context(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    device: Any,
) -> dict[str, Any]:
    (
        scientific,
        execution,
        execution_hash,
        _p6_config,
        p6_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_p6_sources(
        scientific_config_path,
        execution_config_path,
        p6_execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    initial_model, initialization = build_initialized_concept_predictor(
        scientific, split, encoder_path
    )
    concept_provenance = _p6_provenance(
        scientific,
        execution,
        execution_hash,
        p6_hash,
        split,
        initialization,
        stage="concept",
    )
    output = run_directory(fold_index, output_root)
    concept_completion = verify_training_stage(
        stage_name="concept",
        output_directory=output / "concept_stage",
        expected_provenance=concept_provenance,
        expected_epochs=int(execution["reference_reported"]["epochs"]),
        objective_key="concept_loss",
        expected_train_samples=int(
            split["partitions"]["train"]["summary"]["nodules"]
        ),
        require_h200_runtime=True,
        expected_train_nodule_set_sha256=_uid_set_sha256(
            split["partitions"]["train"]["nodule_uids"]
        ),
        expected_validation_samples=int(
            split["partitions"]["validation"]["summary"]["nodules"]
        ),
        expected_validation_nodule_set_sha256=_uid_set_sha256(
            split["partitions"]["validation"]["nodule_uids"]
        ),
    )
    del initial_model
    concept_best_path = output / "concept_stage" / "best.pt"
    concept_model, _rebuilt = _load_concept_best(
        scientific=scientific,
        split=split,
        encoder_path=encoder_path,
        concept_best_path=concept_best_path,
        provenance=concept_provenance,
    )
    concept_model.to(device)
    cache_provenance = _cache_provenance_from_run(
        scientific_config=scientific,
        execution_config_sha256=execution_hash,
        p6_execution_config_sha256=p6_hash,
        split=split,
        initialization=initialization,
        concept_best_checkpoint_path=concept_best_path,
        predictor=concept_model,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
    )
    expected_cache_uids = {
        "train": list(map(str, split["partitions"]["train"]["nodule_uids"])),
        "validation": list(
            map(str, split["partitions"]["validation"]["nodule_uids"])
        ),
    }
    cache_directory = output / "concept_cache"
    cache_frames, cache_manifest = verify_train_validation_caches(
        cache_directory, expected_cache_uids, cache_provenance
    )
    task_head, task_initialization = build_deterministic_task_head(
        int(initialization["fold_seed"])
    )
    task_initialization.update(
        {
            "fold_seed": initialization["fold_seed"],
            "encoder_initialization_sha256": initialization[
                "encoder_initialization_sha256"
            ],
            "concept_head_initialization_sha256": initialization[
                "concept_head_initialization_sha256"
            ],
            "combined_concept_head_initialization_sha256": initialization[
                "combined_concept_head_initialization_sha256"
            ],
            "concept_best_checkpoint_sha256": sha256_file(concept_best_path),
            "frozen_predictor_semantic_sha256": module_state_sha256(
                concept_model
            ),
            "frozen_batchnorm_state_sha256": batchnorm_state_sha256(
                concept_model
            ),
            "cache_manifest_sha256": sha256_file(
                cache_directory / "cache_manifest.json"
            ),
        }
    )
    task_provenance = _p6_provenance(
        scientific,
        execution,
        execution_hash,
        p6_hash,
        split,
        task_initialization,
        stage="task",
    )
    task_completion = verify_training_stage(
        stage_name="task",
        output_directory=output / "task_stage",
        expected_provenance=task_provenance,
        expected_epochs=int(execution["reference_reported"]["epochs"]),
        objective_key="mse",
        expected_train_samples=int(
            split["partitions"]["train"]["summary"]["nodules"]
        ),
        require_h200_runtime=True,
        expected_train_nodule_set_sha256=_uid_set_sha256(
            split["partitions"]["train"]["nodule_uids"]
        ),
        expected_validation_samples=int(
            split["partitions"]["validation"]["summary"]["nodules"]
        ),
        expected_validation_nodule_set_sha256=_uid_set_sha256(
            split["partitions"]["validation"]["nodule_uids"]
        ),
    )
    task_best = _load_stage_checkpoint(output / "task_stage" / "best.pt", task_provenance)
    task_head.load_state_dict(task_best["module_state_dict"], strict=True)
    task_head.eval()
    task_head.to(device)
    sequential_path = output / "sequential_training_complete.json"
    if not sequential_path.is_file():
        raise FileNotFoundError("P6_SEQUENTIAL_TRAINING_COMPLETION_MISSING")
    sequential = json.loads(sequential_path.read_text(encoding="utf-8"))
    predictor_hash = module_state_sha256(concept_model)
    bn_hash = batchnorm_state_sha256(concept_model)
    sequential_provenance = _p6_provenance(
        scientific,
        execution,
        execution_hash,
        p6_hash,
        split,
        task_initialization,
        stage="sequential",
    )
    validate_sequential_completion(
        sequential,
        sequential_provenance,
        concept_completion_sha256=sha256_file(
            output / "concept_stage" / "training_complete.json"
        ),
        cache_manifest_sha256=sha256_file(
            cache_directory / "cache_manifest.json"
        ),
        task_completion_sha256=sha256_file(
            output / "task_stage" / "training_complete.json"
        ),
        predictor_semantic_sha256=predictor_hash,
        batchnorm_state_sha256_value=bn_hash,
    )
    return {
        "scientific": scientific,
        "execution": execution,
        "execution_hash": execution_hash,
        "p6_hash": p6_hash,
        "split": split,
        "manifest": manifest,
        "roi_index": roi_index,
        "encoder_path": encoder_path,
        "initialization": initialization,
        "concept_provenance": concept_provenance,
        "concept_completion": concept_completion,
        "concept_model": concept_model,
        "cache_provenance": cache_provenance,
        "cache_frames": cache_frames,
        "cache_manifest": cache_manifest,
        "task_provenance": task_provenance,
        "task_completion": task_completion,
        "task_head": task_head,
        "task_best": task_best,
        "sequential": sequential,
        "output": output,
    }


def validate_sequential_completion(
    sequential: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    *,
    concept_completion_sha256: str,
    cache_manifest_sha256: str,
    task_completion_sha256: str,
    predictor_semantic_sha256: str,
    batchnorm_state_sha256_value: str,
) -> None:
    if any(sequential.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("P6_SEQUENTIAL_TRAINING_PROVENANCE_MISMATCH")
    if (
        sequential.get("status")
        != "SEQUENTIAL_TRAINING_COMPLETE_TEST_NOT_EVALUATED"
        or sequential.get("test_concepts_generated") is not False
        or sequential.get("test_evaluated") is not False
        or sequential.get("cache_partitions") != ["train", "validation"]
        or sequential.get("concept_completion_sha256")
        != concept_completion_sha256
        or sequential.get("cache_manifest_sha256") != cache_manifest_sha256
        or sequential.get("task_completion_sha256") != task_completion_sha256
    ):
        raise ValueError("P6_SEQUENTIAL_TRAINING_COMPLETION_MISMATCH")
    if (
        sequential.get("frozen_predictor_semantic_sha256_before_task")
        != predictor_semantic_sha256
        or sequential.get("frozen_predictor_semantic_sha256_after_task")
        != predictor_semantic_sha256
        or sequential.get("frozen_batchnorm_state_sha256_before_task")
        != batchnorm_state_sha256_value
        or sequential.get("frozen_batchnorm_state_sha256_after_task")
        != batchnorm_state_sha256_value
    ):
        raise ValueError("P6_FROZEN_PREDICTOR_TASK_INVARIANT_MISMATCH")


def _test_prediction_provenance(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scientific_config_sha256": compute_config_sha256(context["scientific"]),
        "execution_config_sha256": context["execution_hash"],
        "p6_execution_config_sha256": context["p6_hash"],
        "split_sha256": context["split"]["split_sha256"],
        "fold_index": int(context["split"]["fold_index"]),
        "fold_seed": int(context["initialization"]["fold_seed"]),
        "encoder_initialization_sha256": context["initialization"][
            "encoder_initialization_sha256"
        ],
        "encoder_artifact_file_sha256": context["initialization"][
            "encoder_artifact_file_sha256"
        ],
        "concept_head_initialization_sha256": json.dumps(
            context["initialization"]["concept_head_initialization_sha256"],
            separators=(",", ":"),
            sort_keys=True,
        ),
        "combined_concept_head_initialization_sha256": context["initialization"][
            "combined_concept_head_initialization_sha256"
        ],
        "concept_best_checkpoint_sha256": context["cache_provenance"][
            "concept_best_checkpoint_sha256"
        ],
        "frozen_predictor_semantic_sha256": context["cache_provenance"][
            "predictor_semantic_sha256"
        ],
        "cache_manifest_sha256": sha256_file(
            context["output"] / "concept_cache" / "cache_manifest.json"
        ),
        "task_head_initialization_sha256": context["task_provenance"][
            "task_head_initialization_sha256"
        ],
        "task_head_initialization_seed": int(
            context["task_provenance"]["task_head_initialization_seed"]
        ),
        "task_best_checkpoint_sha256": sha256_file(
            context["output"] / "task_stage" / "best.pt"
        ),
        "source_manifest_sha256": context["cache_provenance"][
            "source_manifest_sha256"
        ],
        "source_roi_index_sha256": context["cache_provenance"][
            "source_roi_index_sha256"
        ],
    }


def _test_prediction_rows(
    concept_frame: pd.DataFrame,
    task_head: Any,
    device: Any,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    torch = _torch()
    vectors = torch.from_numpy(ensure_predicted_cache_features(concept_frame)).to(
        device=device, dtype=torch.float32
    )
    task_head.eval()
    with torch.no_grad():
        outputs = task_predictions_and_contributions(task_head, vectors)
    score = outputs["malignancy_raw_score"].detach().cpu().reshape(-1).numpy()
    score_1_to_5 = outputs["malignancy_score_1_to_5"].detach().cpu().reshape(-1).numpy()
    raw_bias = float(outputs["raw_bias"].detach().cpu().reshape(-1)[0])
    rating_bias = float(outputs["rating_scale_bias"].detach().cpu().reshape(-1)[0])
    rows: list[dict[str, Any]] = []
    for index, source in concept_frame.reset_index(drop=True).iterrows():
        row = dict(source)
        row.update(
            {
                "malignancy_raw_score": float(score[index]),
                "malignancy_score_normalized": float(score[index]),
                "malignancy_score_1_to_5": float(score_1_to_5[index]),
                "raw_bias": raw_bias,
                "rating_scale_bias": rating_bias,
                **dict(provenance),
            }
        )
        raw_sum = raw_bias
        rating_sum = rating_bias
        for group in CONCEPT_GROUP_ORDER:
            raw = float(
                outputs["raw_group_contributions"][group]
                .detach()
                .cpu()
                .reshape(-1)[index]
            )
            rating = float(
                outputs["rating_point_contributions"][group]
                .detach()
                .cpu()
                .reshape(-1)[index]
            )
            row[f"{group}_raw_contribution"] = raw
            row[f"{group}_rating_point_contribution"] = rating
            raw_sum += raw
            rating_sum += rating
        if abs(raw_sum - float(score[index])) > 1e-6:
            raise ValueError("P6_TEST_NORMALIZED_RECONSTRUCTION_FAILED")
        if abs(rating_sum - float(score_1_to_5[index])) > 1e-6:
            raise ValueError("P6_TEST_RATING_RECONSTRUCTION_FAILED")
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_test_predictions(
    frame: pd.DataFrame,
    expected_uids: Sequence[str],
    provenance: Mapping[str, Any],
) -> None:
    required_columns = {
        "nodule_uid",
        "patient_key",
        "canonical_activated_concepts",
        "feature_source",
        "feature_dimension",
        "concept_targets",
        "valid_reader_counts",
        "internalStructure_modal_tie",
        "calcification_modal_tie",
        "target_normalized",
        "target_1_to_5",
        "extreme_binary_eligible",
        "extreme_binary_label",
        "malignancy_raw_score",
        "malignancy_score_normalized",
        "malignancy_score_1_to_5",
        "raw_bias",
        "rating_scale_bias",
        *(
            f"{group}_{suffix}"
            for group in CONCEPT_GROUP_ORDER
            for suffix in (
                "logits",
                "activated_prediction",
                "raw_contribution",
                "rating_point_contribution",
            )
        ),
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError("P6_TEST_PREDICTION_SCHEMA_MISSING")
    vectors = ensure_predicted_cache_features(frame)
    observed = list(map(str, frame["nodule_uid"]))
    expected = list(map(str, expected_uids))
    if (
        len(observed) != len(set(observed))
        or len(expected) != len(set(expected))
        or set(observed) != set(expected)
        or len(observed) != len(expected)
    ):
        raise ValueError("P6_TEST_PREDICTION_UID_SET_MISMATCH")
    for key, value in provenance.items():
        if key not in frame or set(frame[key]) != {value}:
            raise ValueError(f"P6_TEST_PREDICTION_PROVENANCE_MISMATCH:{key}")
    if not np.array_equal(
        frame["malignancy_raw_score"].to_numpy(),
        frame["malignancy_score_normalized"].to_numpy(),
    ):
        raise ValueError("P6_TEST_SCORE_ALIAS_MISMATCH")
    for row_index, row in frame.reset_index(drop=True).iterrows():
        activated_pieces: list[float] = []
        targets = json.loads(str(row["concept_targets"]))
        valid_counts = json.loads(str(row["valid_reader_counts"]))
        if set(targets) != set(CONCEPT_GROUP_ORDER) or set(valid_counts) != set(
            CONCEPT_GROUP_ORDER
        ):
            raise ValueError("P6_TEST_CONCEPT_TARGET_SCHEMA_MISMATCH")
        for group, size in CONCEPT_OUTPUT_SIZES.items():
            logits = np.asarray(json.loads(str(row[f"{group}_logits"])), dtype=np.float64)
            activated = np.asarray(
                json.loads(str(row[f"{group}_activated_prediction"])),
                dtype=np.float64,
            )
            if logits.shape != (size,) or activated.shape != (size,):
                raise ValueError(f"P6_TEST_CONCEPT_SHAPE_MISMATCH:{group}")
            if not np.isfinite(logits).all() or not np.isfinite(activated).all():
                raise ValueError(f"P6_TEST_CONCEPT_NONFINITE:{group}")
            if np.any(activated < 0.0) or np.any(activated > 1.0):
                raise ValueError(f"P6_TEST_CONCEPT_ACTIVATION_RANGE:{group}")
            if group in CATEGORICAL_CONCEPTS:
                shifted = logits - logits.max()
                expected_activation = np.exp(shifted) / np.exp(shifted).sum()
            else:
                from scipy.special import expit

                expected_activation = expit(logits)
            if not np.allclose(
                activated, expected_activation, atol=1e-6, rtol=0.0
            ):
                raise ValueError(f"P6_TEST_LOGIT_ACTIVATION_MISMATCH:{group}")
            if group in CATEGORICAL_CONCEPTS and not np.isclose(
                activated.sum(), 1.0, atol=1e-6, rtol=0.0
            ):
                raise ValueError(f"P6_TEST_CONCEPT_PROBABILITY_SUM:{group}")
            target = np.asarray(targets[group], dtype=np.float64).reshape(-1)
            if target.shape != (size,) or not np.isfinite(target).all():
                raise ValueError(f"P6_TEST_CONCEPT_TARGET_SHAPE:{group}")
            if np.any(target < 0.0) or np.any(target > 1.0):
                raise ValueError(f"P6_TEST_CONCEPT_TARGET_RANGE:{group}")
            if group in CATEGORICAL_CONCEPTS and not np.isclose(
                target.sum(), 1.0, atol=1e-6, rtol=0.0
            ):
                raise ValueError(f"P6_TEST_CONCEPT_TARGET_SUM:{group}")
            count = valid_counts[group]
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError(f"P6_TEST_VALID_READER_COUNT:{group}")
            activated_pieces.extend(map(float, activated))
        if not np.allclose(
            np.asarray(activated_pieces, dtype=np.float32),
            vectors[row_index],
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("P6_TEST_CANONICAL_VECTOR_CONTENT_MISMATCH")
        if not isinstance(row["internalStructure_modal_tie"], (bool, np.bool_)) or not isinstance(
            row["calcification_modal_tie"], (bool, np.bool_)
        ):
            raise ValueError("P6_TEST_CATEGORICAL_TIE_TYPE_MISMATCH")
        for group, tie_column in (
            ("internalStructure", "internalStructure_modal_tie"),
            ("calcification", "calcification_modal_tie"),
        ):
            distribution = np.asarray(targets[group], dtype=np.float64)
            expected_tie = bool(
                np.count_nonzero(
                    np.isclose(
                        distribution,
                        distribution.max(),
                        atol=1e-12,
                        rtol=0.0,
                    )
                )
                > 1
            )
            if bool(row[tie_column]) != expected_tie:
                raise ValueError(f"P6_TEST_CATEGORICAL_TIE_SEMANTIC_MISMATCH:{group}")
        raw_score = float(row["malignancy_raw_score"])
        score_1_to_5 = float(row["malignancy_score_1_to_5"])
        target_normalized = float(row["target_normalized"])
        target_1_to_5 = float(row["target_1_to_5"])
        numeric = [
            raw_score,
            score_1_to_5,
            target_normalized,
            target_1_to_5,
            float(row["raw_bias"]),
            float(row["rating_scale_bias"]),
            *(
                float(row[f"{group}_{suffix}"])
                for group in CONCEPT_GROUP_ORDER
                for suffix in (
                    "raw_contribution",
                    "rating_point_contribution",
                )
            ),
        ]
        if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
            raise ValueError("P6_TEST_NUMERIC_NONFINITE")
        if not 0.0 <= target_normalized <= 1.0 or abs(
            target_1_to_5 - (1.0 + 4.0 * target_normalized)
        ) > 1e-6:
            raise ValueError("P6_TEST_TARGET_SCALE_CONVERSION_FAILED")
        eligibility = row["extreme_binary_eligible"]
        if not isinstance(eligibility, (bool, np.bool_)):
            raise ValueError("P6_TEST_EXTREME_ELIGIBILITY_TYPE_MISMATCH")
        label = row["extreme_binary_label"]
        if target_1_to_5 <= 2.0:
            if (
                not bool(eligibility)
                or isinstance(label, (bool, np.bool_))
                or not isinstance(label, (int, float, np.integer, np.floating))
                or not math.isfinite(float(label))
                or float(label) != 0.0
            ):
                raise ValueError("P6_TEST_EXTREME_LOW_LABEL_MISMATCH")
        elif target_1_to_5 >= 4.0:
            if (
                not bool(eligibility)
                or isinstance(label, (bool, np.bool_))
                or not isinstance(label, (int, float, np.integer, np.floating))
                or not math.isfinite(float(label))
                or float(label) != 1.0
            ):
                raise ValueError("P6_TEST_EXTREME_HIGH_LABEL_MISMATCH")
        elif bool(eligibility) or not pd.isna(label):
            raise ValueError("P6_TEST_EXTREME_MIDDLE_LABEL_MISMATCH")
        if abs(score_1_to_5 - (1.0 + 4.0 * raw_score)) > 1e-6:
            raise ValueError("P6_TEST_SCORE_SCALE_CONVERSION_FAILED")
        if abs(float(row["rating_scale_bias"]) - (1.0 + 4.0 * float(row["raw_bias"]))) > 1e-6:
            raise ValueError("P6_TEST_BIAS_SCALE_CONVERSION_FAILED")
        raw = float(row["raw_bias"]) + sum(
            float(row[f"{group}_raw_contribution"])
            for group in CONCEPT_GROUP_ORDER
        )
        rating = float(row["rating_scale_bias"]) + sum(
            float(row[f"{group}_rating_point_contribution"])
            for group in CONCEPT_GROUP_ORDER
        )
        if abs(raw - float(row["malignancy_raw_score"])) > 1e-6:
            raise ValueError("P6_TEST_NORMALIZED_RECONSTRUCTION_FAILED")
        if abs(rating - float(row["malignancy_score_1_to_5"])) > 1e-6:
            raise ValueError("P6_TEST_RATING_RECONSTRUCTION_FAILED")
        for group in CONCEPT_GROUP_ORDER:
            if abs(
                float(row[f"{group}_rating_point_contribution"])
                - 4.0 * float(row[f"{group}_raw_contribution"])
            ) > 1e-6:
                raise ValueError(
                    f"P6_TEST_CONTRIBUTION_SCALE_CONVERSION_FAILED:{group}"
                )


def _seal_test_evaluation(
    *,
    output: Path,
    predictions: pd.DataFrame,
    expected_uids: Sequence[str],
    provenance: Mapping[str, Any],
    claim_sha256: str,
) -> dict[str, Any]:
    _validate_test_predictions(predictions, expected_uids, provenance)
    metrics = regression_metrics(predictions.to_dict(orient="records"))
    _atomic_json(output / "metrics.json", metrics)
    evaluation = {
        **dict(provenance),
        "status": "TEST_EVALUATED_EXACTLY_ONCE",
        "test_samples": int(len(predictions)),
        "test_inference_transactions": 1,
        "test_claim_sha256": claim_sha256,
        "test_predictions_sha256": sha256_file(output / "test_predictions.parquet"),
        "metrics_sha256": sha256_file(output / "metrics.json"),
    }
    _atomic_json(output / "test_evaluation.json", evaluation)
    return evaluation


def _validate_test_claim(
    claim_path: Path,
    provenance: Mapping[str, Any],
    expected_test_samples: int,
) -> tuple[dict[str, Any], str]:
    if not claim_path.is_file():
        raise FileNotFoundError("P6_TEST_CLAIM_MISSING")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if (
        claim.get("schema_version") != SCHEMA_VERSION
        or claim.get("status") != "TEST_INFERENCE_CLAIMED"
        or claim.get("expected_test_samples") != expected_test_samples
        or any(claim.get(key) != value for key, value in provenance.items())
    ):
        raise ValueError("P6_TEST_CLAIM_PROVENANCE_MISMATCH")
    return claim, sha256_file(claim_path)


def _validate_metrics_file(path: Path, predictions: pd.DataFrame) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("P6_TEST_METRICS_MISSING")
    stored = json.loads(path.read_text(encoding="utf-8"))
    reconstructed = regression_metrics(predictions.to_dict(orient="records"))
    if set(stored) != set(reconstructed):
        raise ValueError("P6_TEST_METRICS_SCHEMA_MISMATCH")
    for key, expected in reconstructed.items():
        observed = stored[key]
        if isinstance(expected, int):
            if observed != expected:
                raise ValueError(f"P6_TEST_METRIC_MISMATCH:{key}")
        elif not math.isclose(
            float(observed), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"P6_TEST_METRIC_MISMATCH:{key}")
    return stored


def evaluate_test_once(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p6_lifecycle.lock"):
        return _evaluate_test_once_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p6_execution_config_path=p6_execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            device_name=device_name,
            num_workers=num_workers,
            output_root=output_root,
        )


def _evaluate_test_once_locked(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    torch = _torch()
    device = torch.device(device_name)
    context = _prepare_trained_context(
        scientific_config_path=scientific_config_path,
        execution_config_path=execution_config_path,
        p6_execution_config_path=p6_execution_config_path,
        manifest_path=manifest_path,
        roi_index_path=roi_index_path,
        fold_index=fold_index,
        output_root=output_root,
        device=device,
    )
    output = context["output"]
    expected_uids = list(
        map(str, context["split"]["partitions"]["test"]["nodule_uids"])
    )
    provenance = _test_prediction_provenance(context)
    evaluation_path = output / "test_evaluation.json"
    prediction_path = output / "test_predictions.parquet"
    claim_path = output / "test_claim.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        predictions = pd.read_parquet(prediction_path)
        _validate_test_predictions(predictions, expected_uids, provenance)
        _claim, claim_sha = _validate_test_claim(
            claim_path, provenance, len(expected_uids)
        )
        _validate_metrics_file(output / "metrics.json", predictions)
        if (
            evaluation.get("status") != "TEST_EVALUATED_EXACTLY_ONCE"
            or any(evaluation.get(key) != value for key, value in provenance.items())
            or evaluation.get("test_inference_transactions") != 1
            or evaluation.get("test_samples") != len(expected_uids)
            or evaluation.get("test_claim_sha256") != claim_sha
            or evaluation.get("test_predictions_sha256") != sha256_file(prediction_path)
            or evaluation.get("metrics_sha256") != sha256_file(output / "metrics.json")
        ):
            raise ValueError("P6_TEST_EVALUATION_MISMATCH")
        return evaluation
    if prediction_path.exists():
        _claim, claim_sha = _validate_test_claim(
            claim_path, provenance, len(expected_uids)
        )
        predictions = pd.read_parquet(prediction_path)
        return _seal_test_evaluation(
            output=output,
            predictions=predictions,
            expected_uids=expected_uids,
            provenance=provenance,
            claim_sha256=claim_sha,
        )
    if claim_path.exists():
        raise RuntimeError("P6_TEST_CLAIM_EXISTS_WITHOUT_PREDICTIONS_MANUAL_AUDIT_REQUIRED")
    claim = {
        "schema_version": SCHEMA_VERSION,
        **provenance,
        "status": "TEST_INFERENCE_CLAIMED",
        "expected_test_samples": len(expected_uids),
    }
    _atomic_json(claim_path, claim)
    _claim, claim_sha = _validate_test_claim(
        claim_path, provenance, len(expected_uids)
    )
    test_records = build_partition_concept_records(
        context["manifest"],
        context["roi_index"],
        context["split"],
        "test",
        roi_index_path,
    )
    batch_size = int(
        context["execution"]["project_preregistered"]["batching"][
            "micro_batch_size"
        ]
    )
    concept_frame = predict_concept_cache_frame(
        context["concept_model"],
        test_records,
        device,
        partition="test",
        batch_size=batch_size,
        num_workers=num_workers,
        task_best_checkpoint_sha256=sha256_file(
            context["output"] / "task_stage" / "best.pt"
        ),
    )
    predictions = _test_prediction_rows(
        concept_frame, context["task_head"], device, provenance
    )
    _atomic_parquet(prediction_path, predictions)
    return _seal_test_evaluation(
        output=output,
        predictions=predictions,
        expected_uids=expected_uids,
        provenance=provenance,
        claim_sha256=claim_sha,
    )


def stage_a_preflight_steps(
    concept_model: Any,
    train_records: Sequence[ConceptRecord],
    validation_records: Sequence[ConceptRecord],
    execution_config: Mapping[str, Any],
    *,
    fold_seed: int,
    base_seed: int,
    fold_index: int,
    device: Any,
) -> dict[str, Any]:
    """Exercise both P6 stages at true batch 16 without a formal run."""
    torch = _torch()
    batch_size = int(
        execution_config["project_preregistered"]["batching"]["micro_batch_size"]
    )
    if batch_size != 16:
        raise ValueError("P6_STAGE_A_TRUE_BATCH_SIZE_MISMATCH")
    if len(train_records) < batch_size or len(validation_records) < batch_size:
        raise ValueError("P6_STAGE_A_INSUFFICIENT_RECORDS")
    ordered_train = _ordered_concept_records(
        train_records, base_seed, fold_index, 0
    )[:batch_size]
    ordered_validation = sorted(
        validation_records, key=lambda record: record.nodule_uid
    )[:batch_size]
    dataset = ConceptROIDataset.build(
        ordered_train,
        training=True,
        base_seed=base_seed,
        fold_index=fold_index,
        epoch_index=0,
    )
    batch = next(
        iter(_loader(dataset, batch_size=batch_size, num_workers=0))
    )
    image = batch["image"].to(device=device, dtype=torch.float32)
    targets = _targets_to_device(batch["targets"], device)
    if tuple(image.shape) != (16, 1, 64, 64, 64):
        raise ValueError("P6_STAGE_A_CONCEPT_BATCH_INTERFACE_MISMATCH")
    concept_optimizer = _optimizer(concept_model, execution_config)
    concept_model.train()
    concept_optimizer.zero_grad(set_to_none=True)
    outputs = concept_model(image)
    loss, group_losses = concept_loss(outputs, targets)
    if not torch.isfinite(loss):
        raise ValueError("P6_STAGE_A_CONCEPT_LOSS_NONFINITE")
    loss.backward()
    concept_optimizer.step()

    frozen_before_task = freeze_concept_predictor(concept_model)
    batchnorm_before_task = batchnorm_state_sha256(concept_model)
    train_cache = predict_concept_cache_frame(
        concept_model,
        ordered_train,
        device,
        partition="train",
        batch_size=batch_size,
        num_workers=0,
    )
    validation_cache = predict_concept_cache_frame(
        concept_model,
        ordered_validation,
        device,
        partition="validation",
        batch_size=batch_size,
        num_workers=0,
    )
    train_uids = [record.nodule_uid for record in ordered_train]
    validation_uids = [record.nodule_uid for record in ordered_validation]
    train_task_records = task_cache_records(train_cache, train_uids)
    validation_task_records = task_cache_records(
        validation_cache, validation_uids
    )
    task_head, task_initialization = build_deterministic_task_head(fold_seed)
    task_head.to(device)
    task_stage_optimizer = task_optimizer(task_head, execution_config)
    task_report = train_task_one_epoch(
        task_head,
        train_task_records,
        task_stage_optimizer,
        device,
        base_seed=base_seed,
        fold_index=fold_index,
        epoch_index=0,
        batch_size=batch_size,
        num_workers=0,
    )
    validation_task_report = evaluate_task_records(
        task_head,
        validation_task_records,
        device,
        batch_size=batch_size,
        num_workers=0,
    )
    frozen_after_task = module_state_sha256(concept_model)
    batchnorm_after_task = batchnorm_state_sha256(concept_model)
    if (
        frozen_after_task != frozen_before_task
        or batchnorm_after_task != batchnorm_before_task
    ):
        raise ValueError("P6_STAGE_A_PREDICTOR_CHANGED_DURING_TASK_STEP")
    return {
        "batch_size": batch_size,
        "concept_forward": True,
        "concept_eight_group_loss": True,
        "concept_backward": True,
        "concept_adam_step": True,
        "concept_loss": float(loss.detach().cpu()),
        "concept_group_losses": {
            group: float(value.detach().cpu())
            for group, value in group_losses.items()
        },
        "train_cache_smoke": True,
        "validation_cache_smoke": True,
        "train_cache_samples": len(train_cache),
        "validation_cache_samples": len(validation_cache),
        "canonical_task_input_dimension": 16,
        "task_features": "frozen_predicted_activated_concepts",
        "task_forward": True,
        "task_mse": float(task_report["mse"]),
        "task_validation_mse": float(validation_task_report["mse"]),
        "task_backward": True,
        "task_adam_step": True,
        "predictor_semantic_sha256_before_task": frozen_before_task,
        "predictor_semantic_sha256_after_task": frozen_after_task,
        "batchnorm_state_sha256_before_task": batchnorm_before_task,
        "batchnorm_state_sha256_after_task": batchnorm_after_task,
        "predictor_unchanged_during_task_step": True,
        **task_initialization,
    }


def overfit_check(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    samples: int,
    steps: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run a controlled eight-group concept overfit without formal artifacts."""
    torch = _torch()
    if samples < 2 or steps < 1:
        raise ValueError("P6_INVALID_OVERFIT_CHECK_SIZE")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    (
        scientific,
        execution,
        execution_hash,
        _p6_config,
        p6_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_p6_sources(
        scientific_config_path,
        execution_config_path,
        p6_execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_concept_predictor(
        scientific, split, encoder_path
    )
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
    targets = _targets_to_device(batch["targets"], device)
    model.eval()
    with torch.no_grad():
        initial_loss, _ = concept_loss(model(image), targets)
        initial_value = float(initial_loss.cpu())
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = concept_loss(model(image), targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss, _ = concept_loss(model(image), targets)
        final_value = float(final_loss.cpu())
    if not math.isfinite(final_value) or final_value >= initial_value:
        raise RuntimeError("P6_OVERFIT_SANITY_DID_NOT_IMPROVE")
    report = {
        **_p6_provenance(
            scientific,
            execution,
            execution_hash,
            p6_hash,
            split,
            initialization,
            stage="stage_a_overfit",
        ),
        "status": "PASS",
        "scope": "train_only_eight_group_concept_overfit_sanity",
        "formal_run": False,
        "augmentation_enabled": False,
        "samples": samples,
        "steps": steps,
        "initial_concept_loss": initial_value,
        "final_concept_loss": final_value,
        "relative_final_concept_loss": final_value / initial_value,
        **_runtime_environment(device),
    }
    _atomic_json(output_path, report)
    return report


def preflight(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run the true-batch-16 H200 P6 Stage A operations without formal epochs."""
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    (
        scientific,
        execution,
        execution_hash,
        _p6_config,
        p6_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_p6_sources(
        scientific_config_path,
        execution_config_path,
        p6_execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_concept_predictor(
        scientific, split, encoder_path
    )
    seed_training(int(initialization["fold_seed"]))
    model.to(device)
    train_records = build_partition_concept_records(
        manifest, roi_index, split, "train", roi_index_path
    )
    validation_records = build_partition_concept_records(
        manifest, roi_index, split, "validation", roi_index_path
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    steps = stage_a_preflight_steps(
        model,
        train_records,
        validation_records,
        execution,
        fold_seed=int(initialization["fold_seed"]),
        base_seed=int(scientific["reproducibility"]["base_seed"]),
        fold_index=fold_index,
        device=device,
    )
    torch.cuda.synchronize(device)
    reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = reserved / total
    limit = float(
        execution["project_preregistered"]["preflight"][
            "maximum_peak_reserved_fraction"
        ]
    )
    if fraction > limit:
        raise RuntimeError(f"P6_PREFLIGHT_MEMORY_LIMIT_EXCEEDED:{fraction}")
    report = {
        **_p6_provenance(
            scientific,
            execution,
            execution_hash,
            p6_hash,
            split,
            initialization,
            stage="stage_a_preflight",
        ),
        "status": "PASS",
        "formal_run": False,
        **steps,
        "peak_reserved_bytes": reserved,
        "gpu_total_bytes": total,
        "peak_reserved_fraction": fraction,
        "maximum_allowed_fraction": limit,
        "deterministic_warning_policy": "warn_only",
        "expected_cuda_pooling_warnings_may_continue": True,
        **_runtime_environment(device),
    }
    _atomic_json(output_path, report)
    return report


def verify_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    require_test: bool,
) -> dict[str, Any]:
    torch = _torch()
    with exclusive_fold_lifecycle_lock(
        run_directory(fold_index, output_root) / ".p6_lifecycle.lock"
    ):
        context = _prepare_trained_context(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p6_execution_config_path=p6_execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            output_root=output_root,
            device=torch.device("cpu"),
        )
        result = {
            "status": "PASS",
            "fold_index": fold_index,
            "concept_epochs": context["concept_completion"]["epochs_completed"],
            "task_epochs": context["task_completion"]["epochs_completed"],
            "concept_best_epoch_index": context["concept_completion"][
                "best_epoch_index"
            ],
            "task_best_epoch_index": context["task_completion"]["best_epoch_index"],
            "train_samples_per_epoch": int(
                context["split"]["partitions"]["train"]["summary"]["nodules"]
            ),
            "test_evaluated_once": False,
        }
        if require_test:
            output = context["output"]
            evaluation_path = output / "test_evaluation.json"
            if not evaluation_path.is_file():
                raise FileNotFoundError("P6_TEST_EVALUATION_MISSING")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            predictions = pd.read_parquet(output / "test_predictions.parquet")
            expected_uids = list(
                map(
                    str,
                    context["split"]["partitions"]["test"]["nodule_uids"],
                )
            )
            provenance = _test_prediction_provenance(context)
            _validate_test_predictions(predictions, expected_uids, provenance)
            _claim, claim_sha = _validate_test_claim(
                output / "test_claim.json", provenance, len(expected_uids)
            )
            _validate_metrics_file(output / "metrics.json", predictions)
            if (
                evaluation.get("status") != "TEST_EVALUATED_EXACTLY_ONCE"
                or any(
                    evaluation.get(key) != value
                    for key, value in provenance.items()
                )
                or evaluation.get("test_inference_transactions") != 1
                or evaluation.get("test_samples") != len(expected_uids)
                or evaluation.get("test_claim_sha256") != claim_sha
                or evaluation.get("test_predictions_sha256")
                != sha256_file(output / "test_predictions.parquet")
                or evaluation.get("metrics_sha256")
                != sha256_file(output / "metrics.json")
            ):
                raise ValueError("P6_TEST_EVALUATION_MISMATCH")
            result["test_evaluated_once"] = True
            result["test_samples"] = len(expected_uids)
        return result


def verify_all(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p6_execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    folds = [
        verify_fold(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p6_execution_config_path=p6_execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold,
            output_root=output_root,
            require_test=True,
        )
        for fold in range(5)
    ]
    return {
        "status": "PASS",
        "folds": folds,
        "test_samples": sum(int(fold["test_samples"]) for fold in folds),
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path(
            "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
        ),
    )
    parser.add_argument(
        "--p6-execution-config",
        type=Path,
        default=P6_EXECUTION_CONFIG_DEFAULT,
    )
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
        "--output-root",
        type=Path,
        default=Path("runs/baseline_v2/standard_cbm"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    overfit_parser = commands.add_parser("overfit-check")
    _common_arguments(overfit_parser)
    overfit_parser.add_argument("--fold", type=int, choices=range(5), required=True)
    overfit_parser.add_argument("--device", default="cuda")
    overfit_parser.add_argument("--samples", type=int, default=8)
    overfit_parser.add_argument("--steps", type=int, default=40)
    overfit_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/baseline_v2/standard_cbm/fold_0/stage_a/overfit_sanity.json"
        ),
    )
    preflight_parser = commands.add_parser("preflight")
    _common_arguments(preflight_parser)
    preflight_parser.add_argument("--fold", type=int, choices=range(5), required=True)
    preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/baseline_v2/standard_cbm/fold_0/stage_a/preflight.json"
        ),
    )
    train_parser = commands.add_parser("train")
    _common_arguments(train_parser)
    train_parser.add_argument("--fold", type=int, choices=range(5), required=True)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--resume", action="store_true")
    evaluate_parser = commands.add_parser("evaluate-test")
    _common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--fold", type=int, choices=range(5), required=True)
    evaluate_parser.add_argument("--device", default="cuda")
    evaluate_parser.add_argument("--num-workers", type=int, default=4)
    verify_parser = commands.add_parser("verify")
    _common_arguments(verify_parser)
    scope = verify_parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--fold", type=int, choices=range(5))
    scope.add_argument("--scope", choices=("all",))
    verify_parser.add_argument("--require-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "scientific_config_path": args.config,
        "execution_config_path": args.execution_config,
        "p6_execution_config_path": args.p6_execution_config,
        "manifest_path": args.manifest,
        "roi_index_path": args.roi_index,
        "output_root": args.output_root,
    }
    if args.command == "overfit-check":
        stage_a_common = dict(common)
        stage_a_common.pop("output_root")
        result = overfit_check(
            **stage_a_common,
            fold_index=args.fold,
            device_name=args.device,
            samples=args.samples,
            steps=args.steps,
            output_path=args.output,
        )
    elif args.command == "preflight":
        stage_a_common = dict(common)
        stage_a_common.pop("output_root")
        result = preflight(
            **stage_a_common,
            fold_index=args.fold,
            output_path=args.output,
        )
    elif args.command == "train":
        result = train_fold(
            **common,
            fold_index=args.fold,
            device_name=args.device,
            num_workers=args.num_workers,
            resume=args.resume,
        )
    elif args.command == "evaluate-test":
        result = evaluate_test_once(
            **common,
            fold_index=args.fold,
            device_name=args.device,
            num_workers=args.num_workers,
        )
    elif args.scope == "all":
        if not args.require_test:
            raise ValueError("P6_VERIFY_ALL_REQUIRES_TEST")
        result = verify_all(**common)
    else:
        result = verify_fold(
            **common,
            fold_index=args.fold,
            require_test=args.require_test,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
