"""Prepare and verify Baseline-v2 patient splits and shared encoders."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import random
import struct
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, fold_seed, load_config


SCHEMA_VERSION = 1
STRATA = (
    "mean_le_2",
    "mean_gt_2_lt_3",
    "mean_eq_3",
    "mean_gt_3_lt_4",
    "mean_ge_4",
)
CONTINUOUS_CONCEPTS = (
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
CATEGORICAL_CONCEPTS = {
    "internalStructure": 4,
    "calcification": 6,
}
CONSUMERS = ("blackbox", "standard_cbm", "cem", "gam")
EXPECTED_PARTITION_COUNTS = {
    0: {"train": (1882, 611), "validation": (272, 86), "test": (479, 171)},
    1: {"train": (1858, 602), "validation": (273, 86), "test": (502, 180)},
    2: {"train": (1853, 612), "validation": (241, 87), "test": (539, 169)},
    3: {"train": (1813, 608), "validation": (271, 86), "test": (549, 174)},
    4: {"train": (1811, 607), "validation": (258, 87), "test": (564, 174)},
}


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and private artifacts."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_key(patient_id: str) -> str:
    """Return a domain-separated patient key without exposing PatientID."""
    return sha256_bytes(b"Baseline-v2 patient split\0" + str(patient_id).encode("utf-8"))


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_immutable_bytes(path: Path, content: bytes, overwrite: bool) -> str:
    if path.exists():
        if path.read_bytes() == content:
            return "REUSED"
        if not overwrite:
            raise FileExistsError(f"ARTIFACT_PROVENANCE_MISMATCH:{path}")
    _atomic_bytes(path, content)
    return "WRITTEN"


def _parse_distribution(value: Any, expected_size: int) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    result = np.asarray(parsed, dtype=np.float64)
    if result.shape != (expected_size,) or not np.isfinite(result).all():
        raise ValueError("INVALID_CATEGORICAL_TARGET")
    if not math.isclose(float(result.sum()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("INVALID_CATEGORICAL_TARGET_SUM")
    return result


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("INVALID_TRAIN_STATISTIC_INPUT")
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _rows_for_registered_train(
    manifest: pd.DataFrame,
    train_uids: Iterable[str],
    requested_uids: Iterable[str] | None = None,
) -> pd.DataFrame:
    train = frozenset(map(str, train_uids))
    requested = train if requested_uids is None else frozenset(map(str, requested_uids))
    if not requested <= train:
        raise ValueError("TRAIN_ONLY_STATISTICS_LEAKAGE")
    by_uid = manifest.set_index(manifest["nodule_uid"].astype(str), drop=False)
    missing = requested - frozenset(by_uid.index)
    if missing:
        raise ValueError("TRAIN_ONLY_STATISTICS_UNKNOWN_UID")
    return by_uid.loc[sorted(requested)].reset_index(drop=True)


def _statistics_for_registered_train(manifest: pd.DataFrame, train_uids: Iterable[str]) -> dict[str, Any]:
    rows = _rows_for_registered_train(manifest, train_uids)
    target_stats: dict[str, Any] = {
        "malignancy_target_normalized": _summary(rows["malignancy_target_normalized"].astype(float)),
    }
    valid_reader_counts: dict[str, Any] = {
        "malignancy": _summary(rows["malignancy_valid_reader_count"].astype(float)),
    }
    for concept in CONTINUOUS_CONCEPTS:
        target_stats[concept] = _summary(rows[f"{concept}_target"].astype(float))
        valid_reader_counts[concept] = _summary(rows[f"{concept}_valid_reader_count"].astype(float))
    for concept, classes in CATEGORICAL_CONCEPTS.items():
        distributions = np.stack([_parse_distribution(value, classes) for value in rows[f"{concept}_vote_distribution"]])
        target_stats[concept] = {
            "count": int(len(distributions)),
            "mean_vote_distribution": [float(value) for value in distributions.mean(axis=0)],
        }
        valid_reader_counts[concept] = _summary(rows[f"{concept}_valid_reader_count"].astype(float))
    sorted_uids = sorted(rows["nodule_uid"].astype(str))
    return {
        "scope": "train_only",
        "nodule_count": len(sorted_uids),
        "train_nodule_set_sha256": sha256_bytes(canonical_json_bytes(sorted_uids)),
        "targets": target_stats,
        "valid_reader_counts": valid_reader_counts,
        "model_dependent_contribution_means": "deferred_until_model_inference",
    }


def train_only_rows(
    manifest: pd.DataFrame,
    split_path: str | Path,
    requested_uids: Iterable[str] | None = None,
    *,
    expected_config_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> pd.DataFrame:
    """Select samples from a hash-verified split's true train partition."""
    split = read_split(split_path)
    if expected_config_sha256 is not None and split.get("config_sha256") != expected_config_sha256:
        raise ValueError("TRAIN_ONLY_SPLIT_CONFIG_MISMATCH")
    if expected_manifest_sha256 is not None and split.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("TRAIN_ONLY_SPLIT_MANIFEST_MISMATCH")
    return _rows_for_registered_train(
        manifest,
        split["partitions"]["train"]["nodule_uids"],
        requested_uids,
    )


def train_statistics(
    manifest: pd.DataFrame,
    split_path: str | Path,
    *,
    expected_config_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Compute statistics only from a hash-verified split train partition."""
    rows = train_only_rows(
        manifest,
        split_path,
        expected_config_sha256=expected_config_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return _statistics_for_registered_train(manifest, rows["nodule_uid"].astype(str))


def _primary_manifest(manifest: pd.DataFrame, expected_nodules: int, expected_patients: int) -> pd.DataFrame:
    required = {
        "nodule_uid", "patient_id", "primary_regression_eligible", "malignancy_stratum",
        "mean_malignancy", "malignancy_target_normalized",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"MANIFEST_MISSING_COLUMNS:{sorted(missing)}")
    primary = manifest[manifest["primary_regression_eligible"].astype(bool)].copy()
    primary["nodule_uid"] = primary["nodule_uid"].astype(str)
    primary["patient_id"] = primary["patient_id"].astype(str)
    primary["malignancy_stratum"] = primary["malignancy_stratum"].astype(str)
    primary = primary.sort_values("nodule_uid").reset_index(drop=True)
    if primary["nodule_uid"].duplicated().any():
        raise ValueError("DUPLICATE_PRIMARY_NODULE_UID")
    if len(primary) != expected_nodules or primary["patient_id"].nunique() != expected_patients:
        raise ValueError("PRIMARY_COHORT_COUNT_MISMATCH")
    if set(primary["malignancy_stratum"]) != set(STRATA):
        raise ValueError("MALIGNANCY_STRATA_MISMATCH")
    return primary


def _validate_roi_index(primary: pd.DataFrame, roi_index: pd.DataFrame) -> None:
    required = {"nodule_uid", "status", "relative_roi_path", "roi_file_sha256"}
    if required - set(roi_index.columns):
        raise ValueError("ROI_INDEX_MISSING_COLUMNS")
    if roi_index["nodule_uid"].astype(str).duplicated().any():
        raise ValueError("DUPLICATE_ROI_INDEX_UID")
    if set(roi_index["nodule_uid"].astype(str)) != set(primary["nodule_uid"]):
        raise ValueError("ROI_INDEX_PRIMARY_SET_MISMATCH")
    if not roi_index["status"].isin(["WRITTEN", "REUSED"]).all():
        raise ValueError("ROI_INDEX_INVALID_STATUS")
    if roi_index["relative_roi_path"].isna().any() or roi_index["roi_file_sha256"].isna().any():
        raise ValueError("ROI_INDEX_INCOMPLETE")


def validate_roi_files(primary: pd.DataFrame, roi_index: pd.DataFrame, roi_root: str | Path) -> dict[str, Any]:
    """Require the private ROI directory and index to match byte-for-byte."""
    _validate_roi_index(primary, roi_index)
    root = Path(roi_root).resolve()
    actual_paths = {path.resolve() for path in root.glob("*.npz") if path.is_file()}
    expected_paths: set[Path] = set()
    for row in roi_index.to_dict("records"):
        relative = Path(str(row["relative_roi_path"]))
        path = (root.parent / relative).resolve()
        if path.parent != root:
            raise ValueError("ROI_INDEX_PATH_OUTSIDE_ROOT")
        expected_paths.add(path)
        if not path.is_file():
            raise ValueError(f"ROI_FILE_MISSING:{row['nodule_uid']}")
        if sha256_file(path) != str(row["roi_file_sha256"]):
            raise ValueError(f"ROI_FILE_HASH_MISMATCH:{row['nodule_uid']}")
    if actual_paths != expected_paths:
        raise ValueError("ROI_DIRECTORY_SET_MISMATCH")
    return {
        "roi_files": len(expected_paths),
        "roi_total_bytes": sum(path.stat().st_size for path in expected_paths),
        "roi_set_sha256": sha256_bytes(canonical_json_bytes(sorted(path.name for path in expected_paths))),
    }


def _partition_summary(rows: pd.DataFrame) -> dict[str, Any]:
    strata = rows["malignancy_stratum"].value_counts().to_dict()
    return {
        "nodules": int(len(rows)),
        "patients": int(rows["patient_id"].nunique()),
        "strata": {name: int(strata.get(name, 0)) for name in STRATA},
        "extremes": {
            "low": int((rows["mean_malignancy"].astype(float) <= 2.0).sum()),
            "high": int((rows["mean_malignancy"].astype(float) >= 4.0).sum()),
        },
    }


def _partition(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "nodule_uids": sorted(rows["nodule_uid"].astype(str)),
        "patient_keys": sorted({patient_key(value) for value in rows["patient_id"].astype(str)}),
        "summary": _partition_summary(rows),
    }


def _candidate_score(development: pd.DataFrame, validation: pd.DataFrame, candidate_index: int) -> tuple[float, float, float, int]:
    patient_fraction = validation["patient_id"].nunique() / development["patient_id"].nunique()
    nodule_fraction = len(validation) / len(development)
    development_counts = development["malignancy_stratum"].value_counts()
    validation_counts = validation["malignancy_stratum"].value_counts()
    stratum_deviation = sum(
        abs(validation_counts.get(name, 0) / development_counts[name] - 0.125)
        for name in STRATA
    )
    return abs(patient_fraction - 0.125), stratum_deviation, abs(nodule_fraction - 0.125), candidate_index


def build_split_payloads(
    manifest: pd.DataFrame,
    roi_index: pd.DataFrame,
    config: Mapping[str, Any],
    manifest_sha256: str,
    roi_index_sha256: str,
) -> list[dict[str, Any]]:
    split_config = config["splits"]
    cohort_config = config["cohort"]["primary_regression"]
    outer_folds = int(split_config["outer_folds"])
    base_seed = int(config["reproducibility"]["base_seed"])
    primary = _primary_manifest(manifest, int(cohort_config["nodules"]), int(cohort_config["patients"]))
    _validate_roi_index(primary, roi_index)
    outer = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=base_seed)
    payloads: list[dict[str, Any]] = []
    test_patient_keys: list[str] = []
    test_nodule_uids: list[str] = []
    for fold_index, (development_indices, test_indices) in enumerate(
        outer.split(np.zeros(len(primary)), primary["malignancy_stratum"], primary["patient_id"])
    ):
        development = primary.iloc[development_indices].sort_values("nodule_uid").reset_index(drop=True)
        test = primary.iloc[test_indices].sort_values("nodule_uid").reset_index(drop=True)
        inner_seed = fold_seed(base_seed, fold_index, outer_folds)
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=inner_seed)
        candidates: list[tuple[tuple[float, float, float, int], int, np.ndarray, np.ndarray]] = []
        for candidate_index, (train_indices, validation_indices) in enumerate(
            inner.split(np.zeros(len(development)), development["malignancy_stratum"], development["patient_id"])
        ):
            validation = development.iloc[validation_indices]
            summary = _partition_summary(validation)
            if summary["extremes"]["low"] == 0 or summary["extremes"]["high"] == 0:
                continue
            candidates.append((_candidate_score(development, validation, candidate_index), candidate_index, train_indices, validation_indices))
        if not candidates:
            raise ValueError(f"NO_VALID_INNER_SPLIT:{fold_index}")
        score, candidate_index, train_indices, validation_indices = min(candidates, key=lambda item: item[0])
        train = development.iloc[train_indices].sort_values("nodule_uid").reset_index(drop=True)
        validation = development.iloc[validation_indices].sort_values("nodule_uid").reset_index(drop=True)
        partitions = {"train": _partition(train), "validation": _partition(validation), "test": _partition(test)}
        observed = {name: (partitions[name]["summary"]["nodules"], partitions[name]["summary"]["patients"]) for name in partitions}
        if int(cohort_config["nodules"]) == 2633 and int(cohort_config["patients"]) == 868:
            expected = EXPECTED_PARTITION_COUNTS[fold_index]
            if observed != expected:
                raise ValueError(f"PRE_REGISTERED_SPLIT_COUNT_DRIFT:{fold_index}:{observed}")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": config["protocol"]["version"],
            "config_sha256": compute_config_sha256(config),
            "manifest_sha256": manifest_sha256,
            "roi_index_sha256": roi_index_sha256,
            "fold_index": fold_index,
            "outer_seed": base_seed,
            "inner_seed": inner_seed,
            "algorithm": {
                "implementation": "sklearn.model_selection.StratifiedGroupKFold",
                "scikit_learn_version": importlib.metadata.version("scikit-learn"),
                "input_order": "nodule_uid_ascending",
                "outer": {"n_splits": outer_folds, "shuffle": True},
                "inner": {"n_splits": 8, "shuffle": True, "selected_candidate_index": candidate_index},
                "inner_candidate_score": [float(value) if index < 3 else int(value) for index, value in enumerate(score)],
            },
            "partitions": partitions,
            "train_statistics": _statistics_for_registered_train(primary, partitions["train"]["nodule_uids"]),
        }
        payload["split_sha256"] = sha256_bytes(canonical_json_bytes(payload))
        payloads.append(payload)
        test_patient_keys.extend(partitions["test"]["patient_keys"])
        test_nodule_uids.extend(partitions["test"]["nodule_uids"])
    if len(test_patient_keys) != len(set(test_patient_keys)) or len(test_patient_keys) != int(cohort_config["patients"]):
        raise ValueError("OUTER_TEST_PATIENT_COVERAGE_MISMATCH")
    if len(test_nodule_uids) != len(set(test_nodule_uids)) or set(test_nodule_uids) != set(primary["nodule_uid"]):
        raise ValueError("OUTER_TEST_NODULE_COVERAGE_MISMATCH")
    return payloads


def seed_initialization(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_encoder() -> Any:
    from monai.networks.nets import DenseNet121

    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).features


def encoder_state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        digest.update(struct.pack(">I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack(">I", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">I", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _encoder_payload(config: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    fold_index = int(split["fold_index"])
    seed = fold_seed(int(config["reproducibility"]["base_seed"]), fold_index, int(config["splits"]["outer_folds"]))
    seed_initialization(seed)
    encoder = build_encoder()
    state = OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in encoder.state_dict().items())
    semantic_hash = encoder_state_sha256(state)
    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": config["protocol"]["version"],
            "config_sha256": compute_config_sha256(config),
            "split_sha256": split["split_sha256"],
            "fold_index": fold_index,
            "fold_seed": seed,
            "architecture": "MONAI DenseNet-121 3D features",
            "pretrained": False,
            "input_channels": 1,
            "intended_consumers": list(CONSUMERS),
            "serialization": "torch_legacy_canonical_storage_keys",
            "state_tensor_count": len(state),
            "state_value_count": parameter_count,
            "encoder_state_sha256": semantic_hash,
            "torch_version": importlib.metadata.version("torch"),
            "monai_version": importlib.metadata.version("monai"),
        },
        "encoder_state_dict": state,
    }


def _load_torch_artifact(path: Path) -> dict[str, Any]:
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or set(loaded) != {"metadata", "encoder_state_dict"}:
        raise ValueError("INVALID_ENCODER_ARTIFACT")
    return loaded


def _deterministic_legacy_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Write PyTorch legacy format with stable sequential storage keys."""
    import torch
    from torch.serialization import (
        INT_SIZE,
        LONG_SIZE,
        MAGIC_NUMBER,
        PROTOCOL_VERSION,
        SHORT_SIZE,
        _should_read_directly,
        location_tag,
        normalize_storage_type,
    )

    serialized_storages: dict[str, tuple[Any, Any]] = {}
    storage_keys: dict[int, str] = {}
    storage_dtypes: dict[int, Any] = {}

    def persistent_id(value: Any) -> tuple[Any, ...] | None:
        if not (isinstance(value, torch.storage.TypedStorage) or torch.is_storage(value)):
            return None
        if isinstance(value, torch.storage.TypedStorage):
            storage = value._untyped_storage
            storage_dtype = value.dtype
            storage_type = getattr(torch, value._pickle_storage_type())
            dtype = value.dtype
            storage_numel = value._size()
        elif isinstance(value, torch.UntypedStorage):
            storage = value
            storage_dtype = torch.uint8
            storage_type = normalize_storage_type(type(value))
            dtype = torch.uint8
            storage_numel = storage.nbytes()
        else:  # pragma: no cover - guarded by torch.is_storage
            raise TypeError(f"Unsupported storage: {type(value)}")
        data_pointer = storage.data_ptr()
        if data_pointer:
            previous_dtype = storage_dtypes.setdefault(data_pointer, storage_dtype)
            if previous_dtype != storage_dtype:
                raise RuntimeError("Cannot serialize shared storage with different dtypes")
        identity = int(storage._cdata)
        if identity not in storage_keys:
            storage_keys[identity] = f"{len(storage_keys):08d}"
        key = storage_keys[identity]
        if key not in serialized_storages:
            serialized_storages[key] = (storage, dtype)
        return ("storage", storage_type, key, location_tag(storage), storage_numel, None)

    system_info = {
        "protocol_version": PROTOCOL_VERSION,
        "little_endian": sys.byteorder == "little",
        "type_sizes": {"short": SHORT_SIZE, "int": INT_SIZE, "long": LONG_SIZE},
    }
    with path.open("wb") as stream:
        pickle.dump(MAGIC_NUMBER, stream, protocol=2)
        pickle.dump(PROTOCOL_VERSION, stream, protocol=2)
        pickle.dump(system_info, stream, protocol=2)
        pickler = pickle.Pickler(stream, protocol=2)
        pickler.persistent_id = persistent_id
        pickler.dump(payload)
        ordered_keys = sorted(serialized_storages)
        pickle.dump(ordered_keys, stream, protocol=2)
        stream.flush()
        for key in ordered_keys:
            storage, dtype = serialized_storages[key]
            storage._write_file(
                stream,
                _should_read_directly(stream),
                True,
                torch._utils._element_size(dtype),
            )
        stream.flush()
        os.fsync(stream.fileno())


def validate_encoder_artifact(path: Path, config: Mapping[str, Any], split: Mapping[str, Any]) -> dict[str, Any]:
    payload = _load_torch_artifact(path)
    metadata = payload["metadata"]
    expected = {
        "protocol_version": config["protocol"]["version"],
        "config_sha256": compute_config_sha256(config),
        "split_sha256": split["split_sha256"],
        "fold_index": int(split["fold_index"]),
        "fold_seed": fold_seed(int(config["reproducibility"]["base_seed"]), int(split["fold_index"]), int(config["splits"]["outer_folds"])),
        "architecture": "MONAI DenseNet-121 3D features",
        "pretrained": False,
        "input_channels": 1,
        "intended_consumers": list(CONSUMERS),
        "serialization": "torch_legacy_canonical_storage_keys",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"ENCODER_ARTIFACT_PROVENANCE_MISMATCH:{key}")
    observed_hash = encoder_state_sha256(payload["encoder_state_dict"])
    if metadata.get("encoder_state_sha256") != observed_hash:
        raise ValueError("ENCODER_ARTIFACT_STATE_HASH_MISMATCH")
    return payload


def write_encoder_artifact(path: Path, config: Mapping[str, Any], split: Mapping[str, Any], overwrite: bool = False) -> dict[str, Any]:
    import torch

    if path.exists():
        try:
            existing = validate_encoder_artifact(path, config, split)
        except Exception:
            if not overwrite:
                raise
        else:
            return {"status": "REUSED", "file_sha256": sha256_file(path), **existing["metadata"]}
    payload = _encoder_payload(config, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    try:
        _deterministic_legacy_torch_save(payload, Path(temporary_name))
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    validated = validate_encoder_artifact(path, config, split)
    return {"status": "WRITTEN", "file_sha256": sha256_file(path), **validated["metadata"]}


def load_shared_encoder_initialization(
    encoder: Any,
    artifact_path: str | Path,
    config: Mapping[str, Any],
    split: Mapping[str, Any],
) -> str:
    payload = validate_encoder_artifact(Path(artifact_path), config, split)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    observed = encoder_state_sha256(encoder.state_dict())
    expected = payload["metadata"]["encoder_state_sha256"]
    if observed != expected:
        raise ValueError("LOADED_ENCODER_HASH_MISMATCH")
    return observed


def read_split(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    declared = payload.pop("split_sha256", None)
    observed = sha256_bytes(canonical_json_bytes(payload))
    payload["split_sha256"] = declared
    if declared != observed:
        raise ValueError(f"SPLIT_HASH_MISMATCH:{path}")
    return payload


def _private_summary(payloads: Sequence[Mapping[str, Any]], encoder_reports: Sequence[Mapping[str, Any]], config_hash: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_hash,
        "folds": [
            {
                "fold_index": int(split["fold_index"]),
                "split_sha256": split["split_sha256"],
                "partitions": {name: split["partitions"][name]["summary"] for name in ("train", "validation", "test")},
                "encoder_state_sha256": encoder_reports[index]["encoder_state_sha256"],
                "encoder_file_sha256": encoder_reports[index]["file_sha256"],
            }
            for index, split in enumerate(payloads)
        ],
    }


def build(
    config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = pd.read_parquet(manifest_path)
    roi_index = pd.read_parquet(roi_index_path)
    manifest_hash = sha256_file(manifest_path)
    roi_index_hash = sha256_file(roi_index_path)
    primary = _primary_manifest(
        manifest,
        int(config["cohort"]["primary_regression"]["nodules"]),
        int(config["cohort"]["primary_regression"]["patients"]),
    )
    validate_roi_files(primary, roi_index, config["paths"]["roi_directory"])
    payloads = build_split_payloads(manifest, roi_index, config, manifest_hash, roi_index_hash)
    split_root = Path(config["paths"]["split_directory"])
    encoder_root = Path(config["paths"]["encoder_initialization_directory"])
    for payload in payloads:
        path = split_root / f"fold_{payload['fold_index']}.json"
        _write_immutable_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n", overwrite)
    reloaded = [read_split(split_root / f"fold_{index}.json") for index in range(len(payloads))]
    encoder_reports = [
        write_encoder_artifact(encoder_root / f"fold_{index}.pt", config, reloaded[index], overwrite=overwrite)
        for index in range(len(reloaded))
    ]
    summary = _private_summary(reloaded, encoder_reports, compute_config_sha256(config))
    write_json(split_root / "local_build_summary.json", summary)
    return summary


def verify(config_path: Path, manifest_path: Path, roi_index_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = pd.read_parquet(manifest_path)
    roi_index = pd.read_parquet(roi_index_path)
    primary = _primary_manifest(
        manifest,
        int(config["cohort"]["primary_regression"]["nodules"]),
        int(config["cohort"]["primary_regression"]["patients"]),
    )
    roi_report = validate_roi_files(primary, roi_index, config["paths"]["roi_directory"])
    split_root = Path(config["paths"]["split_directory"])
    encoder_root = Path(config["paths"]["encoder_initialization_directory"])
    splits = [read_split(split_root / f"fold_{index}.json") for index in range(int(config["splits"]["outer_folds"]))]
    if any(split["manifest_sha256"] != sha256_file(manifest_path) or split["roi_index_sha256"] != sha256_file(roi_index_path) for split in splits):
        raise ValueError("SPLIT_SOURCE_FINGERPRINT_MISMATCH")
    test_uids: list[str] = []
    test_patients: list[str] = []
    reports = []
    for index, split in enumerate(splits):
        partition_uid_sets = {name: set(split["partitions"][name]["nodule_uids"]) for name in ("train", "validation", "test")}
        partition_patient_sets = {name: set(split["partitions"][name]["patient_keys"]) for name in ("train", "validation", "test")}
        if any(partition_uid_sets[left] & partition_uid_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"NODULE_LEAKAGE:{index}")
        if any(partition_patient_sets[left] & partition_patient_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"PATIENT_LEAKAGE:{index}")
        if set.union(*partition_uid_sets.values()) != set(primary["nodule_uid"]):
            raise ValueError(f"FOLD_COHORT_COVERAGE_MISMATCH:{index}")
        for name in ("validation", "test"):
            extremes = split["partitions"][name]["summary"]["extremes"]
            if extremes["low"] <= 0 or extremes["high"] <= 0:
                raise ValueError(f"MISSING_EXTREME_CLASS:{index}:{name}")
        expected_stats = _statistics_for_registered_train(primary, split["partitions"]["train"]["nodule_uids"])
        if split["train_statistics"] != expected_stats:
            raise ValueError(f"TRAIN_STATISTICS_MISMATCH:{index}")
        artifact = encoder_root / f"fold_{index}.pt"
        validated = validate_encoder_artifact(artifact, config, split)
        consumer_hashes = []
        for _consumer in CONSUMERS:
            consumer_hashes.append(load_shared_encoder_initialization(build_encoder(), artifact, config, split))
        if len(set(consumer_hashes)) != 1:
            raise ValueError(f"SHARED_ENCODER_HASH_MISMATCH:{index}")
        test_uids.extend(split["partitions"]["test"]["nodule_uids"])
        test_patients.extend(split["partitions"]["test"]["patient_keys"])
        reports.append({
            "fold_index": index,
            "split_sha256": split["split_sha256"],
            "encoder_state_sha256": validated["metadata"]["encoder_state_sha256"],
            "encoder_file_sha256": sha256_file(artifact),
            "consumer_hashes_equal": True,
        })
    if len(test_uids) != len(set(test_uids)) or set(test_uids) != set(primary["nodule_uid"]):
        raise ValueError("OOF_NODULE_COVERAGE_MISMATCH")
    if len(test_patients) != len(set(test_patients)) or len(test_patients) != primary["patient_id"].nunique():
        raise ValueError("OOF_PATIENT_COVERAGE_MISMATCH")
    return {
        "status": "PASS",
        "config_sha256": compute_config_sha256(config),
        "primary_nodules": len(primary),
        "primary_patients": int(primary["patient_id"].nunique()),
        "roi_integrity": roi_report,
        "folds": reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
        subparser.add_argument("--manifest", type=Path, default=Path("artifacts/baseline_v2/manifests/nodules.parquet"))
        subparser.add_argument("--roi-index", type=Path, default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"))
        if command == "build":
            subparser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        result = build(arguments.config, arguments.manifest, arguments.roi_index, overwrite=arguments.overwrite)
    else:
        result = verify(arguments.config, arguments.manifest, arguments.roi_index)
    print(canonical_json_bytes(result).decode("utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
