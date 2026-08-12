"""Read-only checkpoint loading and execution lifecycle for P9 spatial analysis."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lidc_baseline import p5_blackbox as p5
from lidc_baseline import p6_standard_cbm as p6
from lidc_baseline import p7_mixed_cem as p7
from lidc_baseline import p8_gam as p8
from lidc_baseline import p8_gam_lifecycle as p8l
from lidc_baseline.p4_prepare import sha256_file
from lidc_baseline.p6_standard_cbm import (
    CATEGORICAL_CONCEPTS,
    CONCEPT_GROUP_ORDER,
    CONTINUOUS_CONCEPTS,
    ConceptRecord,
    build_partition_concept_records,
    module_state_sha256,
)
from lidc_baseline.p9_evaluation import MODEL_ORDER, P9_EXECUTION_CONFIG_DEFAULT
from lidc_baseline.p9_spatial import (
    MAP_SHAPE,
    RANDOM_MASKS,
    SALIENCY_VOXELS,
    SPATIAL_APPROVAL_ENV,
    TargetSpec,
    aggregate_faithfulness_records,
    build_faithfulness_record,
    gradcam_from_activation_and_gradient,
    map_status,
    matched_random_mask_indices,
    occlude_image_copy,
    read_and_verify_map_shard,
    stable_topk_indices,
    target_specs,
    write_map_shard,
)


SCIENTIFIC_CONFIG_DEFAULT = Path("configs/baseline_v2.yaml")
COMMON_EXECUTION_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
)
P6_CONFIG_DEFAULT = Path("configs/experiments/baseline_v2_p6_standard_cbm_h200.yaml")
P7_CONFIG_DEFAULT = Path("configs/experiments/baseline_v2_p7_mixed_cem_h200.yaml")
P8_CONFIG_DEFAULT = Path("configs/experiments/baseline_v2_p8_gam_h200.yaml")
MANIFEST_DEFAULT = Path("artifacts/baseline_v2/manifests/nodules.parquet")
ROI_INDEX_DEFAULT = Path("artifacts/baseline_v2/manifests/roi_index.parquet")
RUN_ROOTS_DEFAULT = {
    "blackbox": Path("runs/baseline_v2/blackbox"),
    "standard_cbm": Path("runs/baseline_v2/standard_cbm"),
    "mixed_cem": Path("runs/baseline_v2/cem"),
    "learned_softmax_gam": Path("runs/baseline_v2/gam"),
}
P9_ROOT_DEFAULT = Path("runs/baseline_v2/p9")
SPATIAL_APPROVAL_DEFAULT = Path(
    "artifacts/baseline_v2/audit/p9/spatial_execution_approval.json"
)


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).with_name("p9_spatial.py"),
        P9_EXECUTION_CONFIG_DEFAULT,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class FrozenModelBundle:
    model_name: str
    fold_index: int
    model: Any
    task_head: Any | None
    split: Mapping[str, Any]
    manifest: pd.DataFrame
    roi_index: pd.DataFrame
    roi_index_path: Path
    checkpoint_sha256: str
    config_sha256: str
    source_output: Path
    source_verification: Mapping[str, Any]
    encoder_initialization_sha256: str


def _torch() -> Any:
    import torch

    return torch


def _source_common() -> dict[str, Path]:
    return {
        "scientific_config_path": SCIENTIFIC_CONFIG_DEFAULT,
        "execution_config_path": COMMON_EXECUTION_CONFIG_DEFAULT,
        "manifest_path": MANIFEST_DEFAULT,
        "roi_index_path": ROI_INDEX_DEFAULT,
    }


def _load_blackbox(fold_index: int, require_test: bool) -> FrozenModelBundle:
    common = _source_common()
    verification = p5.verify_fold(
        **common,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["blackbox"],
        require_test=require_test,
    )
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = (
        p5._prepare_sources(**common, fold_index=fold_index)
    )
    output = p5.run_directory(fold_index, RUN_ROOTS_DEFAULT["blackbox"])
    checkpoint = output / "best.pt"
    model, initialization, _ = p5._load_best_model(
        scientific, split, encoder_path, checkpoint, execution_hash, execution
    )
    model.eval()
    return FrozenModelBundle(
        "blackbox",
        fold_index,
        model,
        None,
        split,
        manifest,
        roi_index,
        ROI_INDEX_DEFAULT,
        sha256_file(checkpoint),
        execution_hash,
        output,
        verification,
        initialization["encoder_initialization_sha256"],
    )


def _load_standard_cbm(fold_index: int, require_test: bool) -> FrozenModelBundle:
    common = _source_common()
    verification = p6.verify_fold(
        **common,
        p6_execution_config_path=P6_CONFIG_DEFAULT,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["standard_cbm"],
        require_test=require_test,
    )
    context = p6._prepare_trained_context(
        **common,
        p6_execution_config_path=P6_CONFIG_DEFAULT,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["standard_cbm"],
        device=_torch().device("cpu"),
    )
    concept_checkpoint_sha256 = sha256_file(
        context["output"] / "concept_stage" / "best.pt"
    )
    task_checkpoint_sha256 = sha256_file(
        context["output"] / "task_stage" / "best.pt"
    )
    checkpoint_sha256 = hashlib.sha256(
        f"{concept_checkpoint_sha256}:{task_checkpoint_sha256}".encode("ascii")
    ).hexdigest()
    return FrozenModelBundle(
        "standard_cbm",
        fold_index,
        context["concept_model"],
        context["task_head"],
        context["split"],
        context["manifest"],
        context["roi_index"],
        ROI_INDEX_DEFAULT,
        checkpoint_sha256,
        context["p6_hash"],
        context["output"],
        verification,
        context["initialization"]["encoder_initialization_sha256"],
    )


def _load_mixed_cem(fold_index: int, require_test: bool) -> FrozenModelBundle:
    common = _source_common()
    verification = p7.verify_fold(
        **common,
        p7_config_path=P7_CONFIG_DEFAULT,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["mixed_cem"],
        require_test=require_test,
    )
    context = p7._trained_context(
        **common,
        p7_config_path=P7_CONFIG_DEFAULT,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["mixed_cem"],
    )
    context["model"].eval()
    return FrozenModelBundle(
        "mixed_cem",
        fold_index,
        context["model"],
        None,
        context["split"],
        context["manifest"],
        context["roi_index"],
        ROI_INDEX_DEFAULT,
        sha256_file(context["output"] / "best.pt"),
        context["p7_hash"],
        context["output"],
        verification,
        context["initialization"]["encoder_initialization_sha256"],
    )


def _load_gam(fold_index: int, require_test: bool) -> FrozenModelBundle:
    common = _source_common()
    verification = p8l.verify_fold(
        **common,
        p8_config_path=P8_CONFIG_DEFAULT,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["learned_softmax_gam"],
        require_test=require_test,
    )
    context = p8l._trained_context(
        **common,
        p8_config_path=P8_CONFIG_DEFAULT,
        fold_index=fold_index,
        output_root=RUN_ROOTS_DEFAULT["learned_softmax_gam"],
    )
    return FrozenModelBundle(
        "learned_softmax_gam",
        fold_index,
        context["model"],
        None,
        context["split"],
        context["manifest"],
        context["roi_index"],
        ROI_INDEX_DEFAULT,
        sha256_file(context["output"] / "best.pt"),
        context["provenance"]["p8_execution_config_sha256"],
        context["output"],
        verification,
        context["provenance"]["encoder_initialization_sha256"],
    )


def load_frozen_model_bundle(
    model_name: str, fold_index: int, *, require_test: bool
) -> FrozenModelBundle:
    if model_name not in MODEL_ORDER or fold_index not in range(5):
        raise ValueError("P9_MODEL_OR_FOLD_INVALID")
    loaders = {
        "blackbox": _load_blackbox,
        "standard_cbm": _load_standard_cbm,
        "mixed_cem": _load_mixed_cem,
        "learned_softmax_gam": _load_gam,
    }
    return loaders[model_name](fold_index, require_test)


def bundle_state_sha256(bundle: FrozenModelBundle) -> str:
    payload = module_state_sha256(bundle.model)
    if bundle.task_head is not None:
        payload = f"{payload}:{module_state_sha256(bundle.task_head)}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _uid_set_sha256(uids: Sequence[str]) -> str:
    normalized = sorted(map(str, uids))
    if len(normalized) != len(set(normalized)):
        raise ValueError("P9_AUXILIARY_DUPLICATE_UID")
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary_name, index=False, compression="zstd")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _resolve_target_layer(model: Any) -> Any:
    module = model
    for component in (
        "encoder",
        "denseblock4",
        "denselayer16",
        "layers",
        "conv2",
    ):
        if not hasattr(module, component):
            raise ValueError(f"P9_TARGET_LAYER_MISSING:{component}")
        module = getattr(module, component)
    return module


def _forward_bundle(bundle: FrozenModelBundle, image: Any) -> dict[str, Any]:
    if bundle.model_name == "blackbox":
        return {"malignancy_raw_score": bundle.model(image)}
    result = bundle.model(image)
    if bundle.model_name == "standard_cbm":
        task = p6.task_predictions_and_contributions(
            bundle.task_head, result["canonical_vector"]
        )
        result = {
            **result,
            "malignancy_raw_score": task["malignancy_raw_score"],
        }
    return result


def _target_scalar(
    outputs: Mapping[str, Any],
    target: TargetSpec,
    class_indices: Any | None = None,
) -> tuple[Any, Any | None]:
    torch = _torch()
    if target.kind == "malignancy":
        return outputs["malignancy_raw_score"].reshape(-1), None
    logits = outputs["logits"][target.name]
    if target.kind == "continuous_concept":
        return logits.reshape(-1), None
    if class_indices is None:
        class_indices = torch.argmax(outputs["activated"][target.name], dim=1)
    values = logits.gather(1, class_indices.reshape(-1, 1)).reshape(-1)
    return values, class_indices


def _target_expected(record: ConceptRecord, target: TargetSpec, class_index: int | None) -> float:
    if target.kind == "malignancy":
        return float(record.target_normalized)
    if target.name in CONTINUOUS_CONCEPTS:
        mapping = dict(zip(CONTINUOUS_CONCEPTS, record.continuous_targets, strict=True))
        return float(mapping[target.name])
    distribution = (
        record.internal_structure_target
        if target.name == "internalStructure"
        else record.calcification_target
    )
    if class_index is None:
        raise ValueError("P9_CATEGORICAL_TARGET_CLASS_INDEX_MISSING")
    return float(distribution[class_index])


def _read_image(record: Any) -> np.ndarray:
    with np.load(record.roi_path, allow_pickle=False) as archive:
        image = np.asarray(archive["image"])
    if image.shape != (1, *MAP_SHAPE) or image.dtype != np.float32:
        raise ValueError(f"P9_ROI_INTERFACE_MISMATCH:{record.nodule_uid}")
    if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"P9_ROI_VALUE_MISMATCH:{record.nodule_uid}")
    return np.array(image, copy=True)


def partition_records(bundle: FrozenModelBundle, partition: str) -> list[Any]:
    if bundle.model_name == "blackbox":
        return p5.build_partition_records(
            bundle.manifest,
            bundle.roi_index,
            bundle.split,
            partition,
            bundle.roi_index_path,
        )
    return build_partition_concept_records(
        bundle.manifest,
        bundle.roi_index,
        bundle.split,
        partition,
        bundle.roi_index_path,
    )


def _persisted_target_score(
    row: Mapping[str, Any], target: TargetSpec, class_index: int | None
) -> float:
    if target.kind == "malignancy":
        return float(row["malignancy_raw_score"])
    value = row[f"{target.name}_logits"]
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("P9_PERSISTED_LOGIT_JSON_INVALID") from error
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("P9_PERSISTED_LOGIT_VECTOR_INVALID")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("P9_PERSISTED_LOGIT_VECTOR_INVALID") from error
    expected_size = 1 if target.kind == "continuous_concept" else (
        4 if target.name == "internalStructure" else 6
    )
    if array.ndim != 1 or array.size != expected_size or not np.isfinite(array).all():
        raise ValueError("P9_PERSISTED_LOGIT_VECTOR_INVALID")
    if target.kind == "continuous_concept":
        return float(array[0])
    if class_index is None or class_index not in range(array.size):
        raise ValueError("P9_PERSISTED_CATEGORICAL_LOGIT_INDEX_MISMATCH")
    return float(array[class_index])


def _batch_forward_target(
    bundle: FrozenModelBundle,
    images: np.ndarray,
    target: TargetSpec,
    device: Any,
    class_indices: np.ndarray | None,
    batch_size: int = 16,
) -> tuple[np.ndarray, int]:
    torch = _torch()
    scores: list[np.ndarray] = []
    observed_max_batch_size = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            observed_max_batch_size = max(
                observed_max_batch_size, len(images[start : start + batch_size])
            )
            tensor = torch.from_numpy(images[start : start + batch_size]).to(device)
            outputs = _forward_bundle(bundle, tensor)
            indices = (
                torch.from_numpy(class_indices[start : start + batch_size]).to(device)
                if class_indices is not None
                else None
            )
            values, _ = _target_scalar(outputs, target, indices)
            scores.append(values.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(scores), observed_max_batch_size


def _contribution_outputs(
    bundle: FrozenModelBundle, outputs: Mapping[str, Any]
) -> dict[str, Any] | None:
    if bundle.model_name == "blackbox":
        return None
    if bundle.model_name == "standard_cbm":
        return p6.task_predictions_and_contributions(
            bundle.task_head, outputs["canonical_vector"]
        )
    if bundle.model_name == "mixed_cem":
        return p7.task_predictions_and_contributions(bundle.model, outputs)
    return p8.task_predictions_and_contributions(bundle.model, outputs)


def _record_target_columns(record: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_normalized": float(record.target_normalized),
        "target_1_to_5": float(record.target_1_to_5),
    }
    if isinstance(record, ConceptRecord):
        continuous = dict(
            zip(CONTINUOUS_CONCEPTS, record.continuous_targets, strict=True)
        )
        for group, value in continuous.items():
            row[f"{group}_target"] = float(value)
        row["internalStructure_target"] = list(
            map(float, record.internal_structure_target)
        )
        row["calcification_target"] = list(
            map(float, record.calcification_target)
        )
        row["internalStructure_modal_tie"] = bool(record.categorical_ties[0])
        row["calcification_modal_tie"] = bool(record.categorical_ties[1])
    return row


def _prediction_rows(
    bundle: FrozenModelBundle,
    records: Sequence[Any],
    *,
    device: Any,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = _torch()
    rows: list[dict[str, Any]] = []
    normalized_error = 0.0
    rating_error = 0.0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = np.stack([_read_image(record) for record in batch_records])
            outputs = _forward_bundle(
                bundle, torch.from_numpy(images).to(device=device, dtype=torch.float32)
            )
            raw = outputs["malignancy_raw_score"].reshape(-1)
            contributions = _contribution_outputs(bundle, outputs)
            if contributions is not None:
                normalized_error = max(
                    normalized_error,
                    float(
                        contributions.get(
                            "normalized_reconstruction_max_abs_error", 0.0
                        )
                    ),
                )
                rating_error = max(
                    rating_error,
                    float(
                        contributions.get("rating_reconstruction_max_abs_error", 0.0)
                    ),
                )
            for index, record in enumerate(batch_records):
                row = {
                    "nodule_uid": str(record.nodule_uid),
                    "fold_index": int(bundle.fold_index),
                    "model": bundle.model_name,
                    "malignancy_raw_score": float(raw[index].detach().cpu()),
                    "malignancy_score_1_to_5": float(
                        1.0 + 4.0 * raw[index].detach().cpu()
                    ),
                    **_record_target_columns(record),
                }
                if bundle.model_name != "blackbox":
                    for group in CONCEPT_GROUP_ORDER:
                        row[f"{group}_logits"] = (
                            outputs["logits"][group][index]
                            .detach()
                            .cpu()
                            .to(dtype=torch.float32)
                            .reshape(-1)
                            .tolist()
                        )
                        row[f"{group}_activated"] = (
                            outputs["activated"][group][index]
                            .detach()
                            .cpu()
                            .to(dtype=torch.float32)
                            .reshape(-1)
                            .tolist()
                        )
                    if contributions is None:
                        raise AssertionError("P9_CONTRIBUTION_OUTPUT_MISSING")
                    row["raw_bias"] = float(
                        contributions["raw_bias"].reshape(-1)[0].detach().cpu()
                    )
                    row["rating_bias"] = float(
                        contributions["rating_scale_bias"]
                        .reshape(-1)[0]
                        .detach()
                        .cpu()
                    )
                    for group in CONCEPT_GROUP_ORDER:
                        row[f"{group}_raw_contribution"] = float(
                            contributions["raw_group_contributions"][group]
                            .reshape(-1)[index]
                            .detach()
                            .cpu()
                        )
                        rating_key = (
                            "rating_point_contributions"
                            if bundle.model_name == "standard_cbm"
                            else "rating_group_contributions"
                        )
                        row[f"{group}_rating_contribution"] = float(
                            contributions[rating_key][group]
                            .reshape(-1)[index]
                            .detach()
                            .cpu()
                        )
                rows.append(row)
    if len(rows) != len(records):
        raise AssertionError("P9_AUXILIARY_PREDICTION_COUNT_MISMATCH")
    return rows, {
        "normalized_reconstruction_max_abs_error": normalized_error,
        "rating_reconstruction_max_abs_error": rating_error,
    }


def _verify_or_write_auxiliary_predictions(
    bundle: FrozenModelBundle,
    records: Sequence[Any],
    *,
    device: Any,
    output_path: Path,
    partition: str,
) -> dict[str, Any]:
    seal_path = output_path.with_suffix(output_path.suffix + ".json")
    expected_uids = [str(record.nodule_uid) for record in records]
    expected_uid_hash = _uid_set_sha256(expected_uids)
    if seal_path.exists():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        frame = pd.read_parquet(output_path)
        try:
            _validate_auxiliary_predictions(
                bundle,
                frame,
                seal,
                output_path=output_path,
                partition=partition,
                expected_uids=expected_uids,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("P9_AUXILIARY_PREDICTIONS_REUSE_MISMATCH") from error
        return seal
    rows, reconstruction = _prediction_rows(bundle, records, device=device)
    frame = pd.DataFrame(rows)
    _atomic_parquet(output_path, frame)
    seal = {
        "schema_version": 1,
        "status": "P9_AUXILIARY_PREDICTIONS_COMMITTED",
        "partition": partition,
        "model": bundle.model_name,
        "fold_index": bundle.fold_index,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "config_sha256": bundle.config_sha256,
        "sample_count": len(frame),
        "nodule_uid_set_sha256": expected_uid_hash,
        "file": output_path.name,
        "file_sha256": sha256_file(output_path),
        **reconstruction,
    }
    _atomic_json(seal_path, seal)
    _validate_auxiliary_predictions(
        bundle,
        frame,
        seal,
        output_path=output_path,
        partition=partition,
        expected_uids=expected_uids,
    )
    return seal


def _validate_auxiliary_predictions(
    bundle: FrozenModelBundle,
    frame: pd.DataFrame,
    seal: Mapping[str, Any],
    *,
    output_path: Path,
    partition: str,
    expected_uids: Sequence[str],
) -> None:
    required_columns = {
        "nodule_uid",
        "fold_index",
        "model",
        "malignancy_raw_score",
        "malignancy_score_1_to_5",
        "target_normalized",
        "target_1_to_5",
    }
    if not required_columns <= set(frame):
        raise ValueError("P9_AUXILIARY_PREDICTION_SCHEMA_MISMATCH")
    observed_uids = frame["nodule_uid"].astype(str).tolist()
    numeric = frame[
        [
            "fold_index",
            "malignancy_raw_score",
            "malignancy_score_1_to_5",
            "target_normalized",
            "target_1_to_5",
        ]
    ].to_numpy(dtype=np.float64)
    if (
        seal.get("schema_version") != 1
        or seal.get("status") != "P9_AUXILIARY_PREDICTIONS_COMMITTED"
        or seal.get("partition") != partition
        or seal.get("model") != bundle.model_name
        or int(seal.get("fold_index", -1)) != bundle.fold_index
        or seal.get("checkpoint_sha256") != bundle.checkpoint_sha256
        or seal.get("config_sha256") != bundle.config_sha256
        or seal.get("file") != output_path.name
        or seal.get("file_sha256") != sha256_file(output_path)
        or int(seal.get("sample_count", -1)) != len(expected_uids)
        or seal.get("nodule_uid_set_sha256") != _uid_set_sha256(expected_uids)
        or _uid_set_sha256(observed_uids) != _uid_set_sha256(expected_uids)
        or not np.isfinite(numeric).all()
        or set(frame["fold_index"].astype(int)) != {bundle.fold_index}
        or set(frame["model"].astype(str)) != {bundle.model_name}
        or not np.allclose(
            frame["malignancy_score_1_to_5"].to_numpy(dtype=np.float64),
            1.0 + 4.0 * frame["malignancy_raw_score"].to_numpy(dtype=np.float64),
            atol=1e-6,
            rtol=0.0,
        )
        or not np.allclose(
            frame["target_1_to_5"].to_numpy(dtype=np.float64),
            1.0 + 4.0 * frame["target_normalized"].to_numpy(dtype=np.float64),
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise ValueError("P9_AUXILIARY_PREDICTION_CONTENT_MISMATCH")
    if bundle.model_name != "blackbox":
        for group in CONCEPT_GROUP_ORDER:
            expected_size = 4 if group == "internalStructure" else (
                6 if group == "calcification" else 1
            )
            required = {f"{group}_logits", f"{group}_activated", f"{group}_target"}
            if not required <= set(frame):
                raise ValueError("P9_AUXILIARY_CONCEPT_SCHEMA_MISMATCH")
            logits = [
                np.asarray(value, dtype=np.float64).reshape(-1)
                for value in frame[f"{group}_logits"]
            ]
            activated = [
                np.asarray(value, dtype=np.float64).reshape(-1)
                for value in frame[f"{group}_activated"]
            ]
            if any(
                vector.size != expected_size or not np.isfinite(vector).all()
                for vector in (*logits, *activated)
            ):
                raise ValueError("P9_AUXILIARY_CONCEPT_VECTOR_INVALID")
            activated_matrix = np.stack(activated)
            if (
                np.any(activated_matrix < 0.0)
                or np.any(activated_matrix > 1.0)
            ):
                raise ValueError("P9_AUXILIARY_CONCEPT_ACTIVATION_INVALID")
            if group in CATEGORICAL_CONCEPTS:
                targets = [
                    np.asarray(value, dtype=np.float64).reshape(-1)
                    for value in frame[f"{group}_target"]
                ]
                tie_column = f"{group}_modal_tie"
                if tie_column not in frame or any(
                    type(value) not in (bool, np.bool_) for value in frame[tie_column]
                ):
                    raise ValueError("P9_AUXILIARY_CONCEPT_TIE_SCHEMA_INVALID")
                if any(
                    vector.size != expected_size or not np.isfinite(vector).all()
                    for vector in targets
                ):
                    raise ValueError("P9_AUXILIARY_CONCEPT_TARGET_INVALID")
                target_matrix = np.stack(targets)
                if (
                    np.any(target_matrix < 0.0)
                    or np.any(target_matrix > 1.0)
                    or not np.allclose(
                        activated_matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
                    )
                    or not np.allclose(
                        target_matrix.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
                    )
                ):
                    raise ValueError("P9_AUXILIARY_CATEGORICAL_SIMPLEX_INVALID")
            else:
                targets = frame[f"{group}_target"].to_numpy(dtype=np.float64)
                if (
                    not np.isfinite(targets).all()
                    or np.any(targets < 0.0)
                    or np.any(targets > 1.0)
                ):
                    raise ValueError("P9_AUXILIARY_CONTINUOUS_TARGET_INVALID")


def _centering_from_contribution_rows(
    bundle: FrozenModelBundle,
    rows: Sequence[Mapping[str, Any]],
    expected_uids: Sequence[str],
) -> dict[str, Any]:
    if not rows or {str(row["nodule_uid"]) for row in rows} != set(expected_uids):
        raise ValueError("P9_CENTERING_TRAIN_UID_MEMBERSHIP_MISMATCH")
    matrix = np.asarray(
        [
            [float(row[f"{group}_raw_contribution"]) for group in CONCEPT_GROUP_ORDER]
            for row in rows
        ],
        dtype=np.float64,
    )
    if matrix.shape != (len(expected_uids), 8) or not np.isfinite(matrix).all():
        raise ValueError("P9_CENTERING_CONTRIBUTION_MATRIX_INVALID")
    means = matrix.mean(axis=0)
    return {
        "schema_version": 1,
        "status": "P9_TRAIN_FOLD_CONTRIBUTION_CENTERING_COMMITTED",
        "model": bundle.model_name,
        "fold_index": bundle.fold_index,
        "partition": "train",
        "sample_count": len(expected_uids),
        "nodule_uid_set_sha256": _uid_set_sha256(expected_uids),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "config_sha256": bundle.config_sha256,
        "raw_group_means": {
            group: float(means[index])
            for index, group in enumerate(CONCEPT_GROUP_ORDER)
        },
        "rating_group_means": {
            group: float(4.0 * means[index])
            for index, group in enumerate(CONCEPT_GROUP_ORDER)
        },
    }


def _validate_centering_report(
    report: Mapping[str, Any],
    bundle: FrozenModelBundle,
    expected_uids: Sequence[str],
) -> None:
    raw = report.get("raw_group_means")
    rating = report.get("rating_group_means")
    if not isinstance(raw, Mapping) or not isinstance(rating, Mapping):
        raise ValueError("P9_CENTERING_GROUP_MEANS_MISSING")
    if set(raw) != set(CONCEPT_GROUP_ORDER) or set(rating) != set(CONCEPT_GROUP_ORDER):
        raise ValueError("P9_CENTERING_GROUP_SET_MISMATCH")
    for group in CONCEPT_GROUP_ORDER:
        raw_value = float(raw[group])
        rating_value = float(rating[group])
        if (
            not math.isfinite(raw_value)
            or not math.isfinite(rating_value)
            or not math.isclose(rating_value, 4.0 * raw_value, abs_tol=1e-12, rel_tol=0.0)
        ):
            raise ValueError("P9_CENTERING_GROUP_VALUE_MISMATCH")
    normalized_error = float(
        report.get("normalized_reconstruction_max_abs_error", math.inf)
    )
    rating_error = float(report.get("rating_reconstruction_max_abs_error", math.inf))
    if (
        report.get("schema_version") != 1
        or report.get("status")
        != "P9_TRAIN_FOLD_CONTRIBUTION_CENTERING_COMMITTED"
        or report.get("model") != bundle.model_name
        or int(report.get("fold_index", -1)) != bundle.fold_index
        or report.get("partition") != "train"
        or int(report.get("sample_count", -1)) != len(expected_uids)
        or report.get("nodule_uid_set_sha256") != _uid_set_sha256(expected_uids)
        or report.get("checkpoint_sha256") != bundle.checkpoint_sha256
        or report.get("config_sha256") != bundle.config_sha256
        or not math.isfinite(normalized_error)
        or not math.isfinite(rating_error)
        or max(normalized_error, rating_error) > 1e-6
    ):
        raise ValueError("P9_CENTERING_REPORT_PROVENANCE_MISMATCH")


def _verify_or_write_centering(
    bundle: FrozenModelBundle, *, device: Any, output_path: Path
) -> dict[str, Any] | None:
    if bundle.model_name == "blackbox":
        return None
    expected_uids = list(
        map(str, bundle.split["partitions"]["train"]["nodule_uids"])
    )
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        try:
            _validate_centering_report(report, bundle, expected_uids)
        except (TypeError, ValueError) as error:
            raise ValueError("P9_CENTERING_REUSE_MISMATCH") from error
        return report
    reconstruction = {
        "normalized_reconstruction_max_abs_error": 0.0,
        "rating_reconstruction_max_abs_error": 0.0,
    }
    if bundle.model_name == "standard_cbm":
        cache = pd.read_parquet(bundle.source_output / "concept_cache" / "train.parquet")
        vectors = p6.ensure_predicted_cache_features(cache)
        observed_uids = list(map(str, cache["nodule_uid"]))
        if _uid_set_sha256(observed_uids) != _uid_set_sha256(expected_uids):
            raise ValueError("P9_P6_CENTERING_CACHE_UID_MISMATCH")
        torch = _torch()
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for start in range(0, len(cache), 16):
                features = torch.from_numpy(vectors[start : start + 16]).to(device)
                contributions = p6.task_predictions_and_contributions(
                    bundle.task_head, features
                )
                raw_reconstructed = contributions["raw_bias"] + _torch().stack(
                    tuple(contributions["raw_group_contributions"].values()), dim=0
                ).sum(dim=0)
                rating_reconstructed = contributions["rating_scale_bias"] + _torch().stack(
                    tuple(contributions["rating_point_contributions"].values()), dim=0
                ).sum(dim=0)
                reconstruction["normalized_reconstruction_max_abs_error"] = max(
                    reconstruction["normalized_reconstruction_max_abs_error"],
                    float(
                        (
                            raw_reconstructed
                            - contributions["malignancy_raw_score"]
                        )
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    ),
                )
                reconstruction["rating_reconstruction_max_abs_error"] = max(
                    reconstruction["rating_reconstruction_max_abs_error"],
                    float(
                        (
                            rating_reconstructed
                            - contributions["malignancy_score_1_to_5"]
                        )
                        .abs()
                        .max()
                        .detach()
                        .cpu()
                    ),
                )
                for index, uid in enumerate(observed_uids[start : start + 16]):
                    rows.append(
                        {
                            "nodule_uid": uid,
                            **{
                                f"{group}_raw_contribution": float(
                                    contributions["raw_group_contributions"][group]
                                    .reshape(-1)[index]
                                    .detach()
                                    .cpu()
                                )
                                for group in CONCEPT_GROUP_ORDER
                            },
                        }
                    )
    else:
        records = partition_records(bundle, "train")
        rows, reconstruction = _prediction_rows(bundle, records, device=device)
        if max(reconstruction.values()) > 1e-6:
            raise ValueError("P9_CENTERING_RECONSTRUCTION_GATE_FAILED")
    report = _centering_from_contribution_rows(bundle, rows, expected_uids)
    report.update(reconstruction)
    if max(reconstruction.values()) > 1e-6:
        raise ValueError("P9_CENTERING_RECONSTRUCTION_GATE_FAILED")
    _validate_centering_report(report, bundle, expected_uids)
    _atomic_json(output_path, report)
    return report


def process_target_batch(
    bundle: FrozenModelBundle,
    records: Sequence[Any],
    target: TargetSpec,
    *,
    device: Any,
    base_seed: int,
    persisted_rows: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch = _torch()
    if not records or len(records) > 16:
        raise ValueError("P9_SPATIAL_BATCH_SIZE_INVALID")
    source_images = np.stack([_read_image(record) for record in records])
    tensor = torch.from_numpy(source_images).to(device).requires_grad_(True)
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["activation"] = output

    handle = _resolve_target_layer(bundle.model).register_forward_hook(hook)
    try:
        outputs = _forward_bundle(bundle, tensor)
        values, class_indices_tensor = _target_scalar(outputs, target)
        if "activation" not in captured:
            raise ValueError("P9_TARGET_LAYER_ACTIVATION_NOT_CAPTURED")
        gradient = torch.autograd.grad(values.sum(), captured["activation"])[0]
        maps = gradcam_from_activation_and_gradient(
            captured["activation"], gradient
        ).detach().cpu().numpy()
    finally:
        handle.remove()
    original_scores = values.detach().cpu().numpy().astype(np.float64)
    class_indices = (
        class_indices_tensor.detach().cpu().numpy().astype(np.int64)
        if class_indices_tensor is not None
        else None
    )
    output_records: list[dict[str, Any]] = []
    valid = 0
    undefined = 0
    maximum_persisted_error = 0.0
    maximum_occlusion_batch_size = 0
    for index, record in enumerate(records):
        status = map_status(maps[index])
        class_index = int(class_indices[index]) if class_indices is not None else None
        output: dict[str, Any] = {
            "nodule_uid": record.nodule_uid,
            "model": bundle.model_name,
            "fold_index": bundle.fold_index,
            "target": target.name,
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "config_sha256": bundle.config_sha256,
            "implementation_sha256": implementation_sha256(),
            "map": maps[index],
        }
        if class_index is not None:
            output["predicted_class_index"] = class_index
        if persisted_rows is not None:
            if record.nodule_uid not in persisted_rows:
                raise ValueError(f"P9_PERSISTED_OOF_UID_MISSING:{record.nodule_uid}")
            expected_score = _persisted_target_score(
                persisted_rows[record.nodule_uid], target, class_index
            )
            error = abs(expected_score - float(original_scores[index]))
            maximum_persisted_error = max(maximum_persisted_error, error)
            if error > 1e-6:
                raise ValueError(
                    f"P9_ORIGINAL_SCORE_OOF_MISMATCH:{bundle.model_name}:{target.name}"
                )
        if status == "undefined":
            undefined += 1
            output_records.append(output)
            continue
        valid += 1
        saliency = stable_topk_indices(maps[index])
        random_masks = matched_random_mask_indices(
            base_seed=base_seed,
            fold_index=bundle.fold_index,
            nodule_uid=record.nodule_uid,
            target=target.name,
        )
        masked = np.stack(
            [
                occlude_image_copy(source_images[index], mask)
                for mask in (saliency, *random_masks)
            ]
        )
        repeated_class = (
            np.full(len(masked), class_index, dtype=np.int64)
            if class_index is not None
            else None
        )
        occluded_scores, observed_batch_size = _batch_forward_target(
            bundle, masked, target, device, repeated_class, batch_size=16
        )
        expected = _target_expected(record, target, class_index)
        output["faithfulness"] = build_faithfulness_record(
            score_original=float(original_scores[index]),
            target_normalized=expected,
            saliency_score_occluded=float(occluded_scores[0]),
            random_scores_occluded=occluded_scores[1:],
        )
        output_records.append(output)
        maximum_occlusion_batch_size = max(
            maximum_occlusion_batch_size, observed_batch_size
        )
    return output_records, {
        "target": target.name,
        "sample_count": len(records),
        "valid_map_count": valid,
        "undefined_map_count": undefined,
        "maximum_original_vs_persisted_abs_error": maximum_persisted_error,
        "true_occlusion_batch_size_observed": maximum_occlusion_batch_size,
    }


def _runtime_policy(device: Any) -> dict[str, Any]:
    torch = _torch()
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("P9_H200_CUDA_REQUIRED")
    name = torch.cuda.get_device_name(device)
    if "H200" not in name:
        raise ValueError(f"P9_H200_REQUIRED:{name}")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("P9_FP32_DEFAULT_REQUIRED")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise ValueError("P9_TF32_FORBIDDEN")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("P9_DETERMINISTIC_ALGORITHMS_REQUIRED")
    warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    if not warn_only:
        raise ValueError("P9_DETERMINISTIC_WARN_ONLY_REQUIRED")
    return {
        "device_type": "cuda",
        "gpu_name": name,
        "fp32": True,
        "amp": False,
        "bf16": False,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": True,
    }


def _move_bundle(bundle: FrozenModelBundle, device: Any) -> None:
    bundle.model.to(device)
    bundle.model.eval()
    if bundle.task_head is not None:
        bundle.task_head.to(device)
        bundle.task_head.eval()


def preflight(
    *,
    fold_index: int,
    output_path: Path,
    p9_root: Path = P9_ROOT_DEFAULT,
    base_seed: int = 20260808,
) -> dict[str, Any]:
    torch = _torch()
    if fold_index != 0:
        raise ValueError("P9_STAGE_A_FOLD_MUST_BE_ZERO")
    device = torch.device("cuda")
    p5.configure_fp32_determinism(device, p5.validate_execution_config(COMMON_EXECUTION_CONFIG_DEFAULT)[0])
    runtime = _runtime_policy(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model_reports = []
    all_map_records: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_started = time.monotonic()
        bundle = load_frozen_model_bundle(model_name, fold_index, require_test=False)
        before = bundle_state_sha256(bundle)
        _move_bundle(bundle, device)
        records = partition_records(bundle, "validation")[:16]
        if len(records) != 16:
            raise ValueError("P9_STAGE_A_VALIDATION_BATCH_SIZE_MISMATCH")
        target_reports = []
        for target in target_specs(model_name):
            map_records, target_report = process_target_batch(
                bundle,
                records,
                target,
                device=device,
                base_seed=base_seed,
                persisted_rows=None,
            )
            all_map_records.extend(map_records)
            target_reports.append(target_report)
            if (
                int(target_report["valid_map_count"]) < 1
                or int(target_report["true_occlusion_batch_size_observed"]) != 16
            ):
                raise ValueError(
                    f"P9_STAGE_A_TARGET_OCCLUSION_NOT_EXERCISED:{model_name}:{target.name}"
                )
        after = bundle_state_sha256(bundle)
        if before != after:
            raise ValueError(f"P9_STAGE_A_MODEL_STATE_CHANGED:{model_name}")
        elapsed = time.monotonic() - model_started
        projected_hours = elapsed * max((479, 502, 539, 549, 564)) / 16 / 3600
        if projected_hours > 8.8:
            raise ValueError(f"P9_STAGE_A_RUNTIME_PROJECTION_EXCEEDED:{model_name}")
        model_reports.append(
            {
                "model": model_name,
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "implementation_sha256": implementation_sha256(),
                "p4_encoder_initialization_sha256": (
                    bundle.encoder_initialization_sha256
                ),
                "model_semantic_sha256_before": before,
                "model_semantic_sha256_after": after,
                "target_reports": target_reports,
                "elapsed_seconds": elapsed,
                "projected_slowest_fold_hours": projected_hours,
            }
        )
        bundle.model.to("cpu")
        if bundle.task_head is not None:
            bundle.task_head.to("cpu")
        torch.cuda.empty_cache()
    stage_a_root = p9_root / "stage_a"
    shard_path = stage_a_root / "raw_map_roundtrip.parquet"
    if shard_path.with_suffix(".parquet.json").exists():
        restored = read_and_verify_map_shard(shard_path)
    else:
        write_map_shard(shard_path, all_map_records)
        restored = read_and_verify_map_shard(shard_path)
    _validate_stage_a_roundtrip_identity(all_map_records, restored)
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    peak_fraction = peak_reserved / total_memory
    if peak_fraction > 0.85:
        raise ValueError("P9_STAGE_A_MEMORY_GATE_FAILED")
    map_count = sum((479, 502, 539, 549, 564)) * (1 + 3 * 9)
    projected_peak_bytes = int(map_count * (64**3) * 4 * 1.15)
    free_bytes = int(shutil.disk_usage(p9_root.parent).free)
    storage_ratio = free_bytes / projected_peak_bytes
    if storage_ratio < 1.2:
        raise ValueError("P9_STAGE_A_STORAGE_GATE_FAILED")
    actual_error_increases: list[float] = []
    for row in all_map_records:
        faithfulness = row.get("faithfulness")
        if faithfulness is None:
            continue
        actual_error_increases.append(float(faithfulness["saliency_error_increase"]))
        actual_error_increases.extend(
            map(float, faithfulness["random_error_increase_values"])
        )
    if (
        not actual_error_increases
        or not all(math.isfinite(value) for value in actual_error_increases)
        or not any(value > 0.0 for value in actual_error_increases)
        or not any(value < 0.0 for value in actual_error_increases)
    ):
        raise ValueError("P9_STAGE_A_ACTUAL_ERROR_INCREASE_SIGN_CASE_FAILED")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "fold_index": 0,
        "partition": "validation",
        "test_read": False,
        "optimizer_or_parameter_update": False,
        "target_path_count": sum(len(target_specs(model)) for model in MODEL_ORDER),
        "models": model_reports,
        "raw_fp32_shard_roundtrip": True,
        "true_batch_16_occlusion_forward": True,
        "saliency_voxel_count": SALIENCY_VOXELS,
        "matched_random_masks": RANDOM_MASKS,
        "actual_error_increase_value_count": len(actual_error_increases),
        "actual_positive_error_increase_count": sum(
            value > 0.0 for value in actual_error_increases
        ),
        "actual_negative_error_increase_count": sum(
            value < 0.0 for value in actual_error_increases
        ),
        "actual_error_increase_minimum": min(actual_error_increases),
        "actual_error_increase_maximum": max(actual_error_increases),
        "peak_reserved_bytes": peak_reserved,
        "total_gpu_memory_bytes": total_memory,
        "peak_reserved_fraction": peak_fraction,
        "projected_p9_peak_working_set_bytes": projected_peak_bytes,
        "scratch_free_bytes": free_bytes,
        "scratch_free_to_projected_peak_ratio": storage_ratio,
        "runtime": runtime,
        "implementation_sha256": implementation_sha256(),
        "wall_seconds": time.monotonic() - started,
        "p9_spatial_approved": os.environ.get(SPATIAL_APPROVAL_ENV, "0"),
    }
    if report["p9_spatial_approved"] != "0":
        raise ValueError("P9_STAGE_A_MUST_RUN_WITH_FORMAL_APPROVAL_ZERO")
    _atomic_json(output_path, report)
    return report


def _validate_approval_payload(approval: Mapping[str, Any]) -> dict[str, Any]:
    if (
        approval.get("schema_version") != 1
        or approval.get("phase") != "P9"
        or approval.get("status") != "USER_APPROVED_FORMAL_SPATIAL_EXECUTION"
        or approval.get("jobs") != 20
        or approval.get("models") != list(MODEL_ORDER)
        or approval.get("folds") != list(range(5))
        or not isinstance(approval.get("stage_a_preflight_sha256"), str)
        or len(approval["stage_a_preflight_sha256"]) != 64
    ):
        raise PermissionError("P9_FORMAL_SPATIAL_APPROVAL_RECORD_INVALID")
    return dict(approval)


def require_formal_approval_record(approval_path: Path = SPATIAL_APPROVAL_DEFAULT) -> dict[str, Any]:
    if os.environ.get(SPATIAL_APPROVAL_ENV, "0") != "1":
        raise PermissionError("P9_FORMAL_SPATIAL_USER_APPROVAL_REQUIRED")
    if not approval_path.is_file():
        raise PermissionError("P9_FORMAL_SPATIAL_APPROVAL_RECORD_MISSING")
    return _validate_approval_payload(
        json.loads(approval_path.read_text(encoding="utf-8"))
    )


def _validated_stage_a_artifact(
    p9_root: Path, approval: Mapping[str, Any]
) -> dict[str, Any]:
    stage_a_path = p9_root / "stage_a" / "preflight.json"
    if (
        not stage_a_path.is_file()
        or approval["stage_a_preflight_sha256"] != sha256_file(stage_a_path)
    ):
        raise PermissionError("P9_FORMAL_STAGE_A_ARTIFACT_APPROVAL_MISMATCH")
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8"))
    models = stage_a.get("models", [])
    runtime = stage_a.get("runtime", {})
    target_reports = [
        report for model in models for report in model.get("target_reports", [])
    ]
    if (
        stage_a.get("schema_version") != 1
        or stage_a.get("status") != "PASS"
        or stage_a.get("p9_spatial_approved") != "0"
        or stage_a.get("partition") != "validation"
        or stage_a.get("test_read") is not False
        or stage_a.get("optimizer_or_parameter_update") is not False
        or int(stage_a.get("target_path_count", -1)) != 28
        or stage_a.get("true_batch_16_occlusion_forward") is not True
        or stage_a.get("raw_fp32_shard_roundtrip") is not True
        or int(stage_a.get("saliency_voxel_count", -1)) != SALIENCY_VOXELS
        or int(stage_a.get("matched_random_masks", -1)) != RANDOM_MASKS
        or int(stage_a.get("actual_positive_error_increase_count", 0)) < 1
        or int(stage_a.get("actual_negative_error_increase_count", 0)) < 1
        or float(stage_a.get("peak_reserved_fraction", math.inf)) > 0.85
        or float(stage_a.get("scratch_free_to_projected_peak_ratio", -math.inf))
        < 1.2
        or len(models) != len(MODEL_ORDER)
        or {model.get("model") for model in models} != set(MODEL_ORDER)
        or len(target_reports) != 28
        or any(
            int(report.get("valid_map_count", 0)) < 1
            or int(report.get("true_occlusion_batch_size_observed", -1)) != 16
            for report in target_reports
        )
        or any(
            float(model.get("projected_slowest_fold_hours", math.inf)) > 8.8
            or model.get("implementation_sha256") != implementation_sha256()
            or not isinstance(model.get("checkpoint_sha256"), str)
            or len(model.get("checkpoint_sha256", "")) != 64
            or not isinstance(model.get("p4_encoder_initialization_sha256"), str)
            or len(model.get("p4_encoder_initialization_sha256", "")) != 64
            or model.get("model_semantic_sha256_before")
            != model.get("model_semantic_sha256_after")
            for model in models
        )
        or runtime.get("device_type") != "cuda"
        or "H200" not in str(runtime.get("gpu_name", ""))
        or runtime.get("fp32") is not True
        or runtime.get("amp") is not False
        or runtime.get("bf16") is not False
        or runtime.get("cuda_matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_warn_only") is not True
    ):
        raise PermissionError("P9_FORMAL_STAGE_A_GATES_NOT_PASS")
    return stage_a


def _validate_stage_a_roundtrip_identity(
    generated: Sequence[Mapping[str, Any]],
    restored: Sequence[Mapping[str, Any]],
) -> None:
    def identity(row: Mapping[str, Any]) -> tuple[str, int, str, str, str, str, str]:
        return (
            str(row["model"]),
            int(row["fold_index"]),
            str(row["nodule_uid"]),
            str(row["target"]),
            str(row["checkpoint_sha256"]),
            str(row["config_sha256"]),
            str(row["implementation_sha256"]),
        )

    expected = {
        identity(row): hashlib.sha256(
            np.asarray(row["map"], dtype="<f4").tobytes(order="C")
        ).hexdigest()
        for row in generated
    }
    observed = {
        identity(row): str(row["map_sha256"])
        for row in restored
    }
    if (
        len(generated) != len(restored)
        or len(expected) != len(generated)
        or len(observed) != len(restored)
        or observed != expected
    ):
        raise ValueError("P9_STAGE_A_SHARD_ROUNDTRIP_MISMATCH")


def _oof_path(model_name: str) -> Path:
    return RUN_ROOTS_DEFAULT[model_name] / "oof_predictions.parquet"


def _persisted_rows(model_name: str, fold_index: int) -> dict[str, dict[str, Any]]:
    frame = pd.read_parquet(_oof_path(model_name))
    fold = frame[frame["fold_index"].astype(int) == fold_index]
    if fold["nodule_uid"].astype(str).duplicated().any():
        raise ValueError("P9_SPATIAL_OOF_DUPLICATE_UID")
    return {
        str(row["nodule_uid"]): row.to_dict() for _, row in fold.iterrows()
    }


def run_model_fold(
    *,
    model_name: str,
    fold_index: int,
    p9_root: Path = P9_ROOT_DEFAULT,
    base_seed: int = 20260808,
    resume: bool = False,
    approval_path: Path = SPATIAL_APPROVAL_DEFAULT,
) -> dict[str, Any]:
    del resume  # completed shards are always verified and reused.
    approval = require_formal_approval_record(approval_path)
    _validated_stage_a_artifact(p9_root, approval)
    output = p9_root / "spatial" / model_name / f"fold_{fold_index}"
    completion_path = output / "spatial_complete.json"
    if completion_path.exists():
        verify_model_fold(
            model_name,
            fold_index,
            p9_root=p9_root,
            approval_path=approval_path,
        )
        return json.loads(completion_path.read_text(encoding="utf-8"))
    torch = _torch()
    device = torch.device("cuda")
    p5.configure_fp32_determinism(device, p5.validate_execution_config(COMMON_EXECUTION_CONFIG_DEFAULT)[0])
    runtime = _runtime_policy(device)
    bundle = load_frozen_model_bundle(model_name, fold_index, require_test=True)
    before = bundle_state_sha256(bundle)
    _move_bundle(bundle, device)
    output.mkdir(parents=True, exist_ok=True)
    validation_records = partition_records(bundle, "validation")
    validation_seal = _verify_or_write_auxiliary_predictions(
        bundle,
        validation_records,
        device=device,
        output_path=output / "validation_predictions.parquet",
        partition="validation",
    )
    centering_report = _verify_or_write_centering(
        bundle, device=device, output_path=output / "train_contribution_centering.json"
    )
    records = partition_records(bundle, "test")
    persisted = _persisted_rows(model_name, fold_index)
    if set(persisted) != {record.nodule_uid for record in records}:
        raise ValueError("P9_SPATIAL_TEST_OOF_UID_SET_MISMATCH")
    shard_reports = []
    target_names = {target.name for target in target_specs(model_name)}
    for shard_index, start in enumerate(range(0, len(records), 16)):
        shard_records = records[start : start + 16]
        path = output / f"shard_{shard_index:04d}.parquet"
        seal_path = path.with_suffix(".parquet.json")
        expected_uids = {record.nodule_uid for record in shard_records}
        if seal_path.exists():
            restored = read_and_verify_map_shard(path)
            identities = [
                (str(row["nodule_uid"]), str(row["target"])) for row in restored
            ]
            if (
                {str(row["nodule_uid"]) for row in restored} != expected_uids
                or {str(row["target"]) for row in restored} != target_names
                or len(identities) != len(expected_uids) * len(target_names)
                or len(identities) != len(set(identities))
                or any(
                    str(row["checkpoint_sha256"]) != bundle.checkpoint_sha256
                    or str(row["config_sha256"]) != bundle.config_sha256
                    or str(row["implementation_sha256"])
                    != implementation_sha256()
                    for row in restored
                )
            ):
                raise ValueError("P9_COMPLETED_SHARD_CONTENT_MISMATCH")
            shard_reports.append(json.loads(seal_path.read_text(encoding="utf-8")))
            continue
        produced: list[dict[str, Any]] = []
        for target in target_specs(model_name):
            target_rows, _ = process_target_batch(
                bundle,
                shard_records,
                target,
                device=device,
                base_seed=base_seed,
                persisted_rows=persisted,
            )
            produced.extend(target_rows)
        shard_reports.append(write_map_shard(path, produced))
    after = bundle_state_sha256(bundle)
    if before != after:
        raise ValueError("P9_FORMAL_MODEL_STATE_CHANGED")
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    peak_fraction = peak_reserved / total_memory
    if peak_fraction > 0.85:
        raise ValueError("P9_FORMAL_MEMORY_GATE_FAILED")
    completion = {
        "schema_version": 1,
        "status": "P9_SPATIAL_MODEL_FOLD_COMPLETE",
        "model": model_name,
        "fold_index": fold_index,
        "sample_count": len(records),
        "target_count_per_sample": len(target_names),
        "expected_map_records": len(records) * len(target_names),
        "shard_count": len(shard_reports),
        "shard_file_sha256": {
            report["file"]: report["file_sha256"] for report in shard_reports
        },
        "validation_predictions_file_sha256": validation_seal["file_sha256"],
        "validation_predictions_seal_sha256": sha256_file(
            (output / "validation_predictions.parquet").with_suffix(
                ".parquet.json"
            )
        ),
        "train_contribution_centering_sha256": (
            sha256_file(output / "train_contribution_centering.json")
            if centering_report is not None
            else None
        ),
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "p4_encoder_initialization_sha256": bundle.encoder_initialization_sha256,
        "implementation_sha256": implementation_sha256(),
        "source_oof_sha256": sha256_file(_oof_path(model_name)),
        "model_semantic_sha256_before": before,
        "model_semantic_sha256_after": after,
        "optimizer_or_parameter_update": False,
        "second_committed_test_evaluation": False,
        "approval_record_sha256": sha256_file(approval_path),
        "stage_a_preflight_sha256": approval["stage_a_preflight_sha256"],
        "peak_reserved_bytes": peak_reserved,
        "total_gpu_memory_bytes": total_memory,
        "peak_reserved_fraction": peak_fraction,
        "runtime": runtime,
    }
    _atomic_json(completion_path, completion)
    return completion


def _validate_faithfulness_payload(payload: Mapping[str, Any]) -> None:
    for quantity in ("output_sensitivity", "error_increase"):
        saliency = float(payload[f"saliency_{quantity}"])
        if not math.isfinite(saliency):
            raise ValueError("P9_FAITHFULNESS_SALIENCY_VALUE_INVALID")
        values = np.asarray(payload[f"random_{quantity}_values"], dtype=np.float64)
        if values.shape != (20,) or not np.isfinite(values).all():
            raise ValueError("P9_FAITHFULNESS_RANDOM_VALUES_INVALID")
        summary = payload[f"random_{quantity}"]
        expected = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
        if any(not math.isclose(float(summary[key]), value, abs_tol=1e-12, rel_tol=1e-12) for key, value in expected.items()):
            raise ValueError("P9_FAITHFULNESS_RANDOM_SUMMARY_MISMATCH")
        difference = float(payload[f"saliency_minus_random_mean_{quantity}"])
        greater = payload[f"saliency_greater_than_random_mean_{quantity}"]
        if (
            isinstance(greater, np.bool_)
            or type(greater) is not bool
            or not math.isclose(
                difference, saliency - expected["mean"], abs_tol=1e-12, rel_tol=1e-12
            )
            or greater is not (saliency > expected["mean"])
        ):
            raise ValueError("P9_FAITHFULNESS_SALIENCY_COMPARISON_MISMATCH")


def verify_model_fold(
    model_name: str,
    fold_index: int,
    *,
    p9_root: Path = P9_ROOT_DEFAULT,
    approval_path: Path = SPATIAL_APPROVAL_DEFAULT,
) -> dict[str, Any]:
    output = p9_root / "spatial" / model_name / f"fold_{fold_index}"
    completion_path = output / "spatial_complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("schema_version") != 1
        or completion.get("status") != "P9_SPATIAL_MODEL_FOLD_COMPLETE"
        or completion.get("model") != model_name
        or int(completion.get("fold_index", -1)) != fold_index
        or completion.get("optimizer_or_parameter_update") is not False
        or completion.get("second_committed_test_evaluation") is not False
        or completion.get("model_semantic_sha256_before")
        != completion.get("model_semantic_sha256_after")
        or int(completion.get("target_count_per_sample", -1))
        != len(target_specs(model_name))
        or int(completion.get("expected_map_records", -1))
        != int(completion.get("sample_count", -1)) * len(target_specs(model_name))
        or float(completion.get("peak_reserved_fraction", math.inf)) > 0.85
    ):
        raise ValueError("P9_SPATIAL_COMPLETION_MISMATCH")
    runtime = completion.get("runtime", {})
    if (
        runtime.get("device_type") != "cuda"
        or "H200" not in str(runtime.get("gpu_name", ""))
        or runtime.get("fp32") is not True
        or runtime.get("amp") is not False
        or runtime.get("bf16") is not False
        or runtime.get("cuda_matmul_tf32") is not False
        or runtime.get("cudnn_tf32") is not False
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("deterministic_warn_only") is not True
    ):
        raise ValueError("P9_SPATIAL_RUNTIME_POLICY_MISMATCH")
    if (
        not approval_path.is_file()
        or completion.get("approval_record_sha256") != sha256_file(approval_path)
    ):
        raise ValueError("P9_SPATIAL_APPROVAL_PROVENANCE_MISMATCH")
    approval = _validate_approval_payload(
        json.loads(approval_path.read_text(encoding="utf-8"))
    )
    if completion.get("stage_a_preflight_sha256") != approval.get(
        "stage_a_preflight_sha256"
    ):
        raise ValueError("P9_SPATIAL_STAGE_A_PROVENANCE_MISMATCH")
    _validated_stage_a_artifact(p9_root, approval)
    bundle = load_frozen_model_bundle(model_name, fold_index, require_test=True)
    expected_records = partition_records(bundle, "test")
    expected_uids = {str(record.nodule_uid) for record in expected_records}
    if (
        completion.get("checkpoint_sha256") != bundle.checkpoint_sha256
        or completion.get("p4_encoder_initialization_sha256")
        != bundle.encoder_initialization_sha256
        or completion.get("implementation_sha256") != implementation_sha256()
        or completion.get("source_oof_sha256") != sha256_file(_oof_path(model_name))
        or completion.get("model_semantic_sha256_before") != bundle_state_sha256(bundle)
        or int(completion.get("sample_count", -1)) != len(expected_records)
    ):
        raise ValueError("P9_SPATIAL_SOURCE_PROVENANCE_MISMATCH")
    validation_path = output / "validation_predictions.parquet"
    validation_seal_path = validation_path.with_suffix(".parquet.json")
    validation_seal = json.loads(validation_seal_path.read_text(encoding="utf-8"))
    validation_frame = pd.read_parquet(validation_path)
    expected_validation_uids = list(
        map(str, bundle.split["partitions"]["validation"]["nodule_uids"])
    )
    if (
        completion.get("validation_predictions_file_sha256")
        != sha256_file(validation_path)
        or completion.get("validation_predictions_seal_sha256")
        != sha256_file(validation_seal_path)
    ):
        raise ValueError("P9_SPATIAL_VALIDATION_AUXILIARY_MISMATCH")
    try:
        _validate_auxiliary_predictions(
            bundle,
            validation_frame,
            validation_seal,
            output_path=validation_path,
            partition="validation",
            expected_uids=expected_validation_uids,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("P9_SPATIAL_VALIDATION_AUXILIARY_MISMATCH") from error
    centering_path = output / "train_contribution_centering.json"
    if model_name == "blackbox":
        if completion.get("train_contribution_centering_sha256") is not None:
            raise ValueError("P9_BLACKBOX_CENTERING_MUST_BE_NOT_APPLICABLE")
    else:
        centering = json.loads(centering_path.read_text(encoding="utf-8"))
        expected_train_uids = list(
            map(str, bundle.split["partitions"]["train"]["nodule_uids"])
        )
        if completion.get("train_contribution_centering_sha256") != sha256_file(
            centering_path
        ):
            raise ValueError("P9_SPATIAL_CENTERING_ARTIFACT_MISMATCH")
        try:
            _validate_centering_report(centering, bundle, expected_train_uids)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("P9_SPATIAL_CENTERING_ARTIFACT_MISMATCH") from error
    all_rows = []
    expected_shards = {
        f"shard_{index:04d}.parquet"
        for index in range(math.ceil(len(expected_records) / 16))
    }
    if (
        set(completion.get("shard_file_sha256", {})) != expected_shards
        or int(completion.get("shard_count", -1)) != len(expected_shards)
    ):
        raise ValueError("P9_SPATIAL_SHARD_SET_MISMATCH")
    for filename, digest in completion["shard_file_sha256"].items():
        path = output / filename
        if sha256_file(path) != digest:
            raise ValueError("P9_SPATIAL_COMPLETION_SHARD_HASH_MISMATCH")
        all_rows.extend(read_and_verify_map_shard(path))
    expected_map_records = int(completion["expected_map_records"])
    if len(all_rows) != expected_map_records:
        raise ValueError("P9_SPATIAL_MAP_RECORD_COUNT_MISMATCH")
    by_uid: dict[str, set[str]] = {}
    identities: set[tuple[str, str]] = set()
    faithfulness_records: dict[str, list[dict[str, Any]]] = {
        "output_sensitivity": [],
        "error_increase": [],
    }
    undefined = 0
    for row in all_rows:
        uid = str(row["nodule_uid"])
        identity = (uid, str(row["target"]))
        if identity in identities:
            raise ValueError("P9_SPATIAL_DUPLICATE_GLOBAL_TARGET_RECORD")
        identities.add(identity)
        by_uid.setdefault(uid, set()).add(str(row["target"]))
        if (
            str(row["model"]) != model_name
            or int(row["fold_index"]) != fold_index
            or str(row["checkpoint_sha256"]) != bundle.checkpoint_sha256
            or str(row["config_sha256"]) != bundle.config_sha256
            or str(row["implementation_sha256"]) != implementation_sha256()
        ):
            raise ValueError("P9_SPATIAL_MAP_ROW_PROVENANCE_MISMATCH")
        if row["status"] == "undefined":
            undefined += 1
            if not pd.isna(row.get("faithfulness_json")):
                raise ValueError("P9_UNDEFINED_MAP_HAS_FAITHFULNESS")
            continue
        if not isinstance(row.get("faithfulness_json"), str):
            raise ValueError("P9_VALID_MAP_FAITHFULNESS_MISSING")
        payload = json.loads(row["faithfulness_json"])
        _validate_faithfulness_payload(payload)
        for quantity in faithfulness_records:
            faithfulness_records[quantity].append(payload)
    expected_targets = {target.name for target in target_specs(model_name)}
    if set(by_uid) != expected_uids or any(
        targets != expected_targets for targets in by_uid.values()
    ):
        raise ValueError("P9_SPATIAL_TARGET_COVERAGE_MISMATCH")
    aggregates = {
        quantity: aggregate_faithfulness_records(rows, quantity) if rows else None
        for quantity, rows in faithfulness_records.items()
    }
    return {
        "status": "PASS",
        "model": model_name,
        "fold_index": fold_index,
        "sample_count": len(by_uid),
        "map_record_count": len(all_rows),
        "valid_map_count": len(all_rows) - undefined,
        "undefined_map_count": undefined,
        "faithfulness": aggregates,
        "completion_sha256": sha256_file(completion_path),
    }


def verify_all(
    *,
    p9_root: Path = P9_ROOT_DEFAULT,
    approval_path: Path = SPATIAL_APPROVAL_DEFAULT,
) -> dict[str, Any]:
    reports = [
        verify_model_fold(
            model, fold, p9_root=p9_root, approval_path=approval_path
        )
        for model in MODEL_ORDER
        for fold in range(5)
    ]
    return {"status": "PASS", "jobs": len(reports), "reports": reports}
