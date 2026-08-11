"""Train and verify the Baseline-v2 sequential Standard CBM regression model."""

from __future__ import annotations

import hashlib
import json
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
    _atomic_json,
    _atomic_parquet,
    _loader,
    _optimizer,
    apply_training_augmentation,
    augmentation_parameters,
    epoch_uid_order,
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
    allow_test_after_task_best: bool = False,
) -> pd.DataFrame:
    """Generate no-augmentation frozen predictor outputs for one partition."""
    torch = _torch()
    if partition not in ("train", "validation", "test"):
        raise ValueError(f"P6_INVALID_CACHE_PARTITION:{partition}")
    if partition == "test" and not allow_test_after_task_best:
        raise PermissionError("P6_TEST_CONCEPT_GENERATION_BEFORE_TASK_BEST_FORBIDDEN")
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
