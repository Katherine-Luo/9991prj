"""P9 Grad-CAM, occlusion, and private spatial-artifact primitives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.p4_prepare import sha256_file
from lidc_baseline.p6_standard_cbm import CONCEPT_GROUP_ORDER
from lidc_baseline.p9_evaluation import (
    MODEL_ORDER,
    P9_EXECUTION_CONFIG_DEFAULT,
    SCHEMA_VERSION,
    validate_p9_execution_config,
)


TARGET_LAYER = "encoder.denseblock4.denselayer16.layers.conv2"
MAP_SHAPE = (64, 64, 64)
SALIENCY_VOXELS = 26215
RANDOM_MASKS = 20
RECONSTRUCTION_TOLERANCE = 1e-6
SPATIAL_APPROVAL_ENV = "P9_SPATIAL_APPROVED"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    source: str


def target_specs(model: str) -> tuple[TargetSpec, ...]:
    if model not in MODEL_ORDER:
        raise ValueError(f"P9_SPATIAL_UNKNOWN_MODEL:{model}")
    targets = [TargetSpec("malignancy", "malignancy", "raw_task_score")]
    if model != "blackbox":
        for group in CONCEPT_GROUP_ORDER:
            if group in ("internalStructure", "calcification"):
                targets.append(
                    TargetSpec(group, "categorical_concept", "predicted_class_logit")
                )
            else:
                targets.append(
                    TargetSpec(group, "continuous_concept", "pre_sigmoid_logit")
                )
    return tuple(targets)


def all_stage_a_target_paths() -> tuple[tuple[str, TargetSpec], ...]:
    result = tuple(
        (model, target) for model in MODEL_ORDER for target in target_specs(model)
    )
    if len(result) != 28:
        raise AssertionError("P9_STAGE_A_TARGET_PATH_COUNT_MISMATCH")
    return result


def predicted_class_index(logits: Any) -> Any:
    """Return argmax; PyTorch/NumPy argmax already chooses the smaller tied index."""
    if hasattr(logits, "argmax"):
        return logits.argmax(dim=-1) if hasattr(logits, "dim") else logits.argmax(axis=-1)
    raise TypeError("P9_CATEGORICAL_LOGITS_UNSUPPORTED")


def gradcam_from_activation_and_gradient(
    activation: Any,
    gradient: Any,
    *,
    output_shape: tuple[int, int, int] = MAP_SHAPE,
) -> Any:
    """Compute raw post-ReLU 3D Grad-CAM using spatial-mean gradients."""
    import torch
    import torch.nn.functional as functional

    if (
        not isinstance(activation, torch.Tensor)
        or not isinstance(gradient, torch.Tensor)
        or activation.ndim != 5
        or activation.shape != gradient.shape
        or activation.dtype != torch.float32
        or gradient.dtype != torch.float32
    ):
        raise ValueError("P9_GRADCAM_TENSOR_INTERFACE_MISMATCH")
    weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
    raw = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    upsampled = functional.interpolate(
        raw,
        size=output_shape,
        mode="trilinear",
        align_corners=False,
    )
    return upsampled[:, 0].to(dtype=torch.float32)


def map_status(heatmap: np.ndarray) -> str:
    array = np.asarray(heatmap)
    if array.shape != MAP_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("P9_GRADCAM_MAP_INTERFACE_MISMATCH")
    if np.any(array < 0.0):
        raise ValueError("P9_GRADCAM_POST_RELU_NEGATIVE")
    return "undefined" if not np.any(array != 0.0) else "valid"


def stable_topk_indices(heatmap: np.ndarray, k: int = SALIENCY_VOXELS) -> np.ndarray:
    array = np.asarray(heatmap)
    if array.shape != MAP_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("P9_SALIENCY_MAP_INTERFACE_MISMATCH")
    if k != SALIENCY_VOXELS or not 0 < k < array.size:
        raise ValueError("P9_SALIENCY_VOXEL_COUNT_MISMATCH")
    if map_status(array) == "undefined":
        raise ValueError("P9_UNDEFINED_MAP_HAS_NO_OCCLUSION_MASK")
    flat = array.reshape(-1)
    indices = np.arange(flat.size, dtype=np.int64)
    order = np.lexsort((indices, -flat.astype(np.float64)))
    return order[:k]


def _seed_from_material(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def matched_random_mask_indices(
    *,
    base_seed: int,
    fold_index: int,
    nodule_uid: str,
    target: str,
    masks: int = RANDOM_MASKS,
    voxels: int = SALIENCY_VOXELS,
    total_voxels: int = 64**3,
) -> tuple[np.ndarray, ...]:
    if fold_index not in range(5) or masks != 20 or voxels != SALIENCY_VOXELS:
        raise ValueError("P9_RANDOM_MASK_POLICY_MISMATCH")
    if total_voxels != 64**3:
        raise ValueError("P9_RANDOM_MASK_VOLUME_MISMATCH")
    material = (
        "Baseline-v2/P9/spatial-random-mask|"
        f"{base_seed}|{fold_index}|{nodule_uid}|{target}|{SCHEMA_VERSION}"
    )
    generator = np.random.default_rng(_seed_from_material(material))
    return tuple(
        generator.choice(total_voxels, size=voxels, replace=False).astype(np.int64)
        for _ in range(masks)
    )


def occlude_image_copy(
    image: np.ndarray, flat_indices: np.ndarray, replacement: float = 0.0
) -> np.ndarray:
    source = np.asarray(image)
    if source.shape != (1, *MAP_SHAPE) or source.dtype != np.float32:
        raise ValueError("P9_OCCLUSION_IMAGE_INTERFACE_MISMATCH")
    indices = np.asarray(flat_indices)
    if (
        indices.ndim != 1
        or indices.dtype.kind not in "iu"
        or len(indices) != SALIENCY_VOXELS
        or len(np.unique(indices)) != len(indices)
        or np.min(indices) < 0
        or np.max(indices) >= 64**3
        or replacement != 0.0
    ):
        raise ValueError("P9_OCCLUSION_MASK_INTERFACE_MISMATCH")
    result = np.array(source, copy=True)
    result.reshape(-1)[indices] = np.float32(replacement)
    return result


def faithfulness_quantities(
    score_original: float, score_occluded: float, target_normalized: float
) -> dict[str, float]:
    values = np.asarray(
        [score_original, score_occluded, target_normalized], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("P9_FAITHFULNESS_NONFINITE")
    return {
        "output_sensitivity": float(abs(score_occluded - score_original)),
        "error_increase": float(
            abs(score_occluded - target_normalized)
            - abs(score_original - target_normalized)
        ),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.shape != (RANDOM_MASKS,) or not np.isfinite(values).all():
        raise ValueError("P9_RANDOM_FAITHFULNESS_VALUES_INVALID")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_faithfulness_record(
    *,
    score_original: float,
    target_normalized: float,
    saliency_score_occluded: float,
    random_scores_occluded: list[float] | tuple[float, ...] | np.ndarray,
) -> dict[str, Any]:
    random_scores = np.asarray(random_scores_occluded, dtype=np.float64)
    if random_scores.shape != (RANDOM_MASKS,) or not np.isfinite(random_scores).all():
        raise ValueError("P9_RANDOM_OCCLUDED_SCORE_COUNT_MISMATCH")
    saliency = faithfulness_quantities(
        score_original, saliency_score_occluded, target_normalized
    )
    random_output = np.abs(random_scores - score_original)
    random_error = np.abs(random_scores - target_normalized) - abs(
        score_original - target_normalized
    )
    output_summary = _summary(random_output)
    error_summary = _summary(random_error)
    return {
        "score_original": float(score_original),
        "target_normalized": float(target_normalized),
        "saliency_output_sensitivity": saliency["output_sensitivity"],
        "saliency_error_increase": saliency["error_increase"],
        "random_output_sensitivity_values": random_output.tolist(),
        "random_error_increase_values": random_error.tolist(),
        "random_output_sensitivity": output_summary,
        "random_error_increase": error_summary,
        "saliency_minus_random_mean_output_sensitivity": float(
            saliency["output_sensitivity"] - output_summary["mean"]
        ),
        "saliency_minus_random_mean_error_increase": float(
            saliency["error_increase"] - error_summary["mean"]
        ),
        "saliency_greater_than_random_mean_output_sensitivity": bool(
            saliency["output_sensitivity"] > output_summary["mean"]
        ),
        "saliency_greater_than_random_mean_error_increase": bool(
            saliency["error_increase"] > error_summary["mean"]
        ),
    }


def aggregate_faithfulness_records(
    records: list[dict[str, Any]], quantity: str
) -> dict[str, float | int]:
    if quantity not in ("output_sensitivity", "error_increase") or not records:
        raise ValueError("P9_FAITHFULNESS_AGGREGATE_INPUT_INVALID")
    saliency = np.asarray(
        [float(row[f"saliency_{quantity}"]) for row in records], dtype=np.float64
    )
    if not np.isfinite(saliency).all():
        raise ValueError("P9_FAITHFULNESS_AGGREGATE_NONFINITE")
    return {
        "sample_count": int(saliency.size),
        "mean": float(np.mean(saliency)),
        "sd": float(np.std(saliency, ddof=0)),
        "median": float(np.median(saliency)),
        "percentile_2_5": float(np.percentile(saliency, 2.5)),
        "percentile_97_5": float(np.percentile(saliency, 97.5)),
        "saliency_greater_than_matched_random_mean_rate": float(
            np.mean(
                [
                    bool(row[f"saliency_greater_than_random_mean_{quantity}"])
                    for row in records
                ]
            )
        ),
    }


def _map_row(record: dict[str, Any]) -> dict[str, Any]:
    map_value = np.asarray(record["map"])
    status = map_status(map_value)
    required = (
        "nodule_uid",
        "model",
        "fold_index",
        "target",
        "checkpoint_sha256",
        "config_sha256",
    )
    if any(key not in record for key in required):
        raise ValueError("P9_MAP_SHARD_METADATA_MISSING")
    if record["model"] not in MODEL_ORDER or int(record["fold_index"]) not in range(5):
        raise ValueError("P9_MAP_SHARD_MODEL_OR_FOLD_INVALID")
    return {
        **{key: record[key] for key in required},
        "shape": list(MAP_SHAPE),
        "dtype": "float32_le",
        "status": status,
        "map_bytes": map_value.astype("<f4", copy=False).tobytes(order="C"),
        "map_sha256": hashlib.sha256(
            map_value.astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest(),
    }


def write_map_shard(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    nodule_uids = {str(record.get("nodule_uid")) for record in records}
    if not records or len(nodule_uids) > 16:
        raise ValueError("P9_MAP_SHARD_NODULE_COUNT_INVALID")
    identities = [
        (
            str(record.get("model")),
            int(record.get("fold_index", -1)),
            str(record.get("nodule_uid")),
            str(record.get("target")),
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("P9_MAP_SHARD_DUPLICATE_TARGET_RECORD")
    for model, _, _, target in identities:
        if target not in {spec.name for spec in target_specs(model)}:
            raise ValueError(f"P9_MAP_SHARD_TARGET_INVALID:{model}:{target}")
    rows = [_map_row(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        temporary = Path(temporary_name)
        pd.DataFrame(rows).to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    file_sha256 = sha256_file(path)
    seal = {
        "schema_version": SCHEMA_VERSION,
        "status": "SPATIAL_MAP_SHARD_COMMITTED",
        "records": len(rows),
        "nodule_count": len(nodule_uids),
        "file": path.name,
        "file_sha256": file_sha256,
        "map_sha256": [row["map_sha256"] for row in rows],
    }
    seal_path = path.with_suffix(path.suffix + ".json")
    descriptor, seal_temporary_name = tempfile.mkstemp(
        dir=seal_path.parent, prefix=f".{seal_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(seal, sort_keys=True, separators=(",", ":")) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(seal_temporary_name, seal_path)
    finally:
        if os.path.exists(seal_temporary_name):
            os.unlink(seal_temporary_name)
    return seal


def read_and_verify_map_shard(path: Path) -> list[dict[str, Any]]:
    seal = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    if (
        seal.get("schema_version") != SCHEMA_VERSION
        or seal.get("status") != "SPATIAL_MAP_SHARD_COMMITTED"
        or seal.get("file") != path.name
        or seal.get("file_sha256") != sha256_file(path)
    ):
        raise ValueError("P9_MAP_SHARD_FILE_HASH_MISMATCH")
    frame = pd.read_parquet(path)
    if len(frame) != int(seal.get("records", -1)):
        raise ValueError("P9_MAP_SHARD_RECORD_COUNT_MISMATCH")
    identities = list(
        zip(
            frame["model"].astype(str),
            frame["fold_index"].astype(int),
            frame["nodule_uid"].astype(str),
            frame["target"].astype(str),
            strict=True,
        )
    )
    nodule_count = frame["nodule_uid"].astype(str).nunique()
    if (
        nodule_count != int(seal.get("nodule_count", -1))
        or nodule_count > 16
        or len(identities) != len(set(identities))
    ):
        raise ValueError("P9_MAP_SHARD_NODULE_OR_IDENTITY_MISMATCH")
    for model, _, _, target in identities:
        if target not in {spec.name for spec in target_specs(model)}:
            raise ValueError("P9_MAP_SHARD_TARGET_INVALID")
    results = []
    for index, row in frame.iterrows():
        if list(row["shape"]) != list(MAP_SHAPE) or row["dtype"] != "float32_le":
            raise ValueError("P9_MAP_SHARD_SCHEMA_MISMATCH")
        raw = bytes(row["map_bytes"])
        if hashlib.sha256(raw).hexdigest() != row["map_sha256"]:
            raise ValueError("P9_MAP_SHARD_MAP_HASH_MISMATCH")
        if row["map_sha256"] != seal["map_sha256"][index]:
            raise ValueError("P9_MAP_SHARD_SEAL_MAP_HASH_MISMATCH")
        heatmap = np.frombuffer(raw, dtype="<f4").reshape(MAP_SHAPE).copy()
        if map_status(heatmap) != row["status"]:
            raise ValueError("P9_MAP_SHARD_STATUS_MISMATCH")
        result = row.to_dict()
        result["map"] = heatmap
        results.append(result)
    return results


def require_formal_spatial_approval() -> None:
    if os.environ.get(SPATIAL_APPROVAL_ENV, "0") != "1":
        raise PermissionError("P9_FORMAL_SPATIAL_USER_APPROVAL_REQUIRED")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=P9_EXECUTION_CONFIG_DEFAULT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--fold", type=int, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--model", choices=MODEL_ORDER, required=True)
    run.add_argument("--fold", type=int, required=True)
    run.add_argument("--resume", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--model", choices=MODEL_ORDER)
    verify.add_argument("--fold", type=int)
    verify.add_argument("--scope", choices=("all",))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    validate_p9_execution_config(arguments.config)
    if arguments.command == "preflight":
        if arguments.fold != 0:
            raise ValueError("P9_STAGE_A_FOLD_MUST_BE_ZERO")
        raise RuntimeError("P9_SPATIAL_PREFLIGHT_LIFECYCLE_NOT_IMPLEMENTED")
    if arguments.command == "run":
        require_formal_spatial_approval()
        raise RuntimeError("P9_SPATIAL_FORMAL_LIFECYCLE_NOT_IMPLEMENTED")
    raise RuntimeError("P9_SPATIAL_VERIFY_LIFECYCLE_NOT_IMPLEMENTED")


if __name__ == "__main__":
    raise SystemExit(main())
