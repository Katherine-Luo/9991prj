"""Train and verify the Baseline-v2 black-box regression baseline."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
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
    read_split,
    sha256_bytes,
    sha256_file,
    validate_encoder_artifact,
)


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCHEMA_VERSION = 1
AUGMENTATION_SCHEMA_VERSION = 1
HEAD_SEED_DOMAIN = b"Baseline-v2/P5/blackbox-head\0"
EPOCH_ORDER_DOMAIN = b"Baseline-v2/P5/epoch-order\0"
AUGMENTATION_DOMAIN = b"Baseline-v2/common-augmentation\0"
MODEL_NAME = "blackbox"
EXPECTED_FOLD_TEST_COUNTS = (479, 502, 539, 549, 564)
EXECUTION_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
)
SERIALIZED_FLOAT_REL_TOL = 1e-12
SERIALIZED_FLOAT_ABS_TOL = 1e-12


def _torch() -> Any:
    import torch

    return torch


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary_name)
        with Path(temporary_name).open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@contextmanager
def exclusive_fold_lifecycle_lock(path: Path) -> Iterable[None]:
    """Allow only one formal train/evaluate writer for a fold at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("P5_FOLD_LIFECYCLE_ALREADY_RUNNING") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _runtime_environment(device: Any) -> dict[str, Any]:
    torch = _torch()
    return {
        "hostname_sha256": sha256_bytes(platform.node().encode("utf-8")),
        "python_version": platform.python_version(),
        "torch_version": importlib.metadata.version("torch"),
        "monai_version": importlib.metadata.version("monai"),
        "numpy_version": importlib.metadata.version("numpy"),
        "device_type": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "fp32": True,
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32_enabled": bool(torch.backends.cudnn.allow_tf32),
    }


def _training_plot(history: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    epochs = [int(row["epoch_index"]) + 1 for row in history]
    figure, loss_axis = plt.subplots(figsize=(8, 5))
    loss_axis.plot(epochs, [float(row["train_mse"]) for row in history], label="train MSE")
    loss_axis.plot(epochs, [float(row["validation_mse"]) for row in history], label="validation MSE")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("MSE (normalized score scale)")
    loss_axis.legend(loc="upper right")
    learning_axis = loss_axis.twinx()
    learning_axis.plot(
        epochs,
        [float(row["learning_rate_end"]) for row in history],
        color="black",
        linestyle="--",
        alpha=0.5,
        label="learning rate",
    )
    learning_axis.set_ylabel("Learning rate")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _prediction_plot(predictions: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    target = [float(row["target_1_to_5"]) for row in predictions]
    score = [float(row["malignancy_score_1_to_5"]) for row in predictions]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(target, score, s=10, alpha=0.5)
    low = min(1.0, min(score))
    high = max(5.0, max(score))
    axis.plot([low, high], [low, high], color="black", linestyle="--")
    axis.set_xlabel("Reader-mean malignancy target (1–5)")
    axis.set_ylabel("Unclipped predicted malignancy score (1–5 scale)")
    axis.set_xlim(0.9, 5.1)
    axis.set_ylim(low, high)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def validate_execution_config(
    execution_config_path: str | Path,
    digest_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load the independent execution policy and enforce its frozen hash."""
    source = Path(execution_config_path)
    config = load_config(source)
    observed = compute_config_sha256(config)
    digest = (
        Path(digest_path)
        if digest_path is not None
        else source.with_suffix(".sha256")
    )
    if digest.read_text(encoding="ascii").strip() != observed:
        raise ValueError("EXECUTION_CONFIG_HASH_MISMATCH")
    reported = config.get("reference_reported", {})
    project = config.get("project_preregistered", {})
    expected = {
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "epochs": 80,
        "batch_size": 16,
    }
    if any(reported.get(key) != value for key, value in expected.items()):
        raise ValueError("EXECUTION_CONFIG_REFERENCE_POLICY_MISMATCH")
    if project.get("statement") != (
        "Baseline-v2 project pre-registered implementation choices, "
        "not exact hyperparameters reported by the reference paper."
    ):
        raise ValueError("EXECUTION_CONFIG_SOURCE_LABEL_MISSING")
    batching = project.get("batching", {})
    if (
        batching.get("micro_batch_size") != 16
        or batching.get("gradient_accumulation_steps") != 1
        or batching.get("effective_batch_size") != 16
        or batching.get("drop_last") is not False
    ):
        raise ValueError("EXECUTION_CONFIG_BATCH_POLICY_MISMATCH")
    profile = config.get("execution_profile", {})
    if (
        profile.get("profile_id") != "baseline-v2-formal-h200-warn-only"
        or profile.get("amendment_type") != "execution_reproducibility_profile"
        or profile.get("formal_gpu_model") != "H200"
        or profile.get("applies_to_formal_training")
        != ["blackbox", "standard_cbm", "cem", "gam"]
    ):
        raise ValueError("EXECUTION_CONFIG_H200_PROFILE_MISMATCH")
    if project.get("preflight", {}).get("device") != "NVIDIA_H200":
        raise ValueError("EXECUTION_CONFIG_H200_PREFLIGHT_MISMATCH")
    reproducibility = project.get("reproducibility", {})
    if (
        reproducibility.get("torch_use_deterministic_algorithms") is not True
        or reproducibility.get("warn_only") is not True
        or not isinstance(reproducibility.get("statement"), str)
    ):
        raise ValueError("EXECUTION_CONFIG_WARN_ONLY_REPRODUCIBILITY_MISMATCH")
    return config, observed


def reproducibility_provenance(execution_config: Mapping[str, Any]) -> dict[str, bool]:
    """Return the explicit deterministic-algorithm enforcement policy."""
    policy = execution_config["project_preregistered"]["reproducibility"]
    return {
        "torch_use_deterministic_algorithms": bool(
            policy["torch_use_deterministic_algorithms"]
        ),
        "deterministic_algorithms_warn_only": bool(policy["warn_only"]),
    }


def configure_fp32_determinism(device: Any, execution_config: Mapping[str, Any]) -> dict[str, bool]:
    """Apply FP32/TF32-off and the profile-bound deterministic policy."""
    torch = _torch()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    policy = reproducibility_provenance(execution_config)
    torch.use_deterministic_algorithms(
        policy["torch_use_deterministic_algorithms"],
        warn_only=policy["deterministic_algorithms_warn_only"],
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(torch.initial_seed())
    return policy


def require_formal_gpu_for_cuda(device: Any, execution_config: Mapping[str, Any]) -> None:
    """Block formal CUDA execution outside the frozen execution hardware profile."""
    if device.type != "cuda":
        return
    torch = _torch()
    name = torch.cuda.get_device_name(device)
    expected = str(execution_config["execution_profile"]["formal_gpu_model"])
    if expected.upper() not in name.upper():
        raise RuntimeError(f"P5_REQUIRES_NVIDIA_{expected.upper()}:{name}")


def seed_training(seed: int) -> None:
    """Seed process RNGs without changing the deterministic data schedule."""
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def head_initialization_seed(fold_seed: int) -> int:
    material = HEAD_SEED_DOMAIN + int(fold_seed).to_bytes(8, "big", signed=False)
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def head_state_sha256(head: Any) -> str:
    state = OrderedDict(
        (name, tensor.detach().cpu().contiguous())
        for name, tensor in head.state_dict().items()
    )
    return encoder_state_sha256(state)


def build_deterministic_head(fold_seed: int) -> tuple[Any, int, str]:
    """Create Linear(1024,1) with an isolated fold-specific CPU RNG."""
    torch = _torch()
    seed = head_initialization_seed(fold_seed)
    devices: list[int] = []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        head = torch.nn.Linear(1024, 1)
    return head, seed, head_state_sha256(head)


class BlackBoxRegressor:
    """Factory for the P5 encoder, global mean pool and linear task head."""

    @staticmethod
    def build(encoder: Any, head: Any) -> Any:
        torch = _torch()

        class Model(torch.nn.Module):
            def __init__(self, feature_encoder: Any, task_head: Any) -> None:
                super().__init__()
                self.encoder = feature_encoder
                self.relu = torch.nn.ReLU(inplace=False)
                self.task_head = task_head

            def forward(self, image: Any) -> Any:
                features = self.relu(self.encoder(image))
                pooled = features.mean(dim=(2, 3, 4))
                return self.task_head(pooled)

        return Model(encoder, head)


def build_initialized_model(
    scientific_config: Mapping[str, Any],
    split: Mapping[str, Any],
    encoder_artifact_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Load the P4 encoder and create a deterministic P5 head on CPU."""
    encoder = build_encoder()
    encoder_hash = load_shared_encoder_initialization(
        encoder,
        encoder_artifact_path,
        scientific_config,
        split,
    )
    validated = validate_encoder_artifact(
        Path(encoder_artifact_path), scientific_config, split
    )
    fold_seed = int(validated["metadata"]["fold_seed"])
    head, head_seed, head_hash = build_deterministic_head(fold_seed)
    if encoder_state_sha256(encoder.state_dict()) != encoder_hash:
        raise ValueError("P5_ENCODER_HASH_CHANGED_BEFORE_TRAINING")
    model = BlackBoxRegressor.build(encoder, head)
    return model, {
        "fold_seed": fold_seed,
        "head_initialization_seed": head_seed,
        "head_initialization_sha256": head_hash,
        "head_seed_derivation": "sha256(utf8_domain_null || fold_seed_u64be), first_8_bytes_u64be_mask_63_bits",
        "encoder_initialization_sha256": encoder_hash,
        "encoder_artifact_file_sha256": sha256_file(encoder_artifact_path),
    }


def _uniform(material: bytes, label: str) -> float:
    digest = hashlib.sha256(material + b"\0" + label.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def augmentation_material(
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    nodule_uid: str,
    schema_version: int = AUGMENTATION_SCHEMA_VERSION,
) -> bytes:
    payload = {
        "augmentation_schema_version": int(schema_version),
        "base_seed": int(base_seed),
        "epoch_index": int(epoch_index),
        "fold_index": int(fold_index),
        "nodule_uid": str(nodule_uid),
    }
    return AUGMENTATION_DOMAIN + canonical_json_bytes(payload)


def augmentation_parameters(
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    nodule_uid: str,
) -> dict[str, Any]:
    """Return model-independent deterministic parameters for one sample/epoch."""
    material = augmentation_material(
        base_seed,
        fold_index,
        epoch_index,
        nodule_uid,
    )
    return {
        "rotate": _uniform(material, "axial_rotation_apply") < 0.5,
        "angle_degrees": -15.0 + 30.0 * _uniform(material, "axial_rotation_angle"),
        "flip_h": _uniform(material, "h_axis_flip") < 0.5,
        "flip_w": _uniform(material, "w_axis_flip") < 0.5,
        "reverse_z": _uniform(material, "z_order_reversal") < 0.5,
        "operation_order": (
            "axial_rotation",
            "h_axis_flip",
            "w_axis_flip",
            "z_order_reversal",
        ),
    }


def apply_training_augmentation(image: Any, parameters: Mapping[str, Any]) -> Any:
    """Apply the pre-registered image-only augmentation in fixed order."""
    torch = _torch()
    if image.ndim != 4 or tuple(image.shape) != (1, 64, 64, 64):
        raise ValueError("P5_AUGMENTATION_IMAGE_INTERFACE_MISMATCH")
    output = image
    if bool(parameters["rotate"]):
        radians = math.radians(float(parameters["angle_degrees"]))
        cosine = math.cos(radians)
        sine = math.sin(radians)
        theta = torch.tensor(
            [
                [cosine, -sine, 0.0, 0.0],
                [sine, cosine, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=output.dtype,
            device=output.device,
        ).unsqueeze(0)
        batch = output.unsqueeze(0)
        grid = torch.nn.functional.affine_grid(
            theta,
            batch.shape,
            align_corners=False,
        )
        output = torch.nn.functional.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(0)
    if bool(parameters["flip_h"]):
        output = torch.flip(output, dims=(-2,))
    if bool(parameters["flip_w"]):
        output = torch.flip(output, dims=(-1,))
    if bool(parameters["reverse_z"]):
        output = torch.flip(output, dims=(-3,))
    return output.contiguous()


def epoch_uid_order(
    nodule_uids: Iterable[str],
    base_seed: int,
    fold_index: int,
    epoch_index: int,
) -> list[str]:
    """Return a deterministic permutation without mutable sampler RNG state."""
    prefix = (
        EPOCH_ORDER_DOMAIN
        + int(base_seed).to_bytes(8, "big", signed=False)
        + int(fold_index).to_bytes(4, "big", signed=False)
        + int(epoch_index).to_bytes(8, "big", signed=False)
    )
    return sorted(
        map(str, nodule_uids),
        key=lambda uid: (hashlib.sha256(prefix + uid.encode("ascii")).digest(), uid),
    )


@dataclass(frozen=True)
class SampleRecord:
    nodule_uid: str
    patient_key: str
    roi_path: Path
    target_normalized: float
    target_1_to_5: float
    extreme_binary_eligible: bool
    extreme_binary_label: int | None


class ROIDataset:
    """Factory for a PyTorch dataset with deterministic train augmentation."""

    @staticmethod
    def build(
        records: Sequence[SampleRecord],
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
                    raise ValueError(f"P5_ROI_INTERFACE_MISMATCH:{record.nodule_uid}")
                if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
                    raise ValueError(f"P5_ROI_VALUE_MISMATCH:{record.nodule_uid}")
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
                return {
                    "image": tensor,
                    "target": torch.tensor([record.target_normalized], dtype=torch.float32),
                    "nodule_uid": record.nodule_uid,
                }

        return Dataset()


def _boolean(value: Any) -> bool:
    if value is pd.NA or pd.isna(value):
        return False
    return bool(value)


def build_partition_records(
    manifest: pd.DataFrame,
    roi_index: pd.DataFrame,
    split: Mapping[str, Any],
    partition: str,
    roi_index_path: str | Path,
) -> list[SampleRecord]:
    """Resolve one hash-verified partition to private ROI records."""
    if partition not in ("train", "validation", "test"):
        raise ValueError(f"INVALID_PARTITION:{partition}")
    uids = list(map(str, split["partitions"][partition]["nodule_uids"]))
    if len(uids) != len(set(uids)):
        raise ValueError("P5_SPLIT_DUPLICATE_UID")
    primary = manifest[manifest["primary_regression_eligible"].astype(bool)].copy()
    primary["nodule_uid"] = primary["nodule_uid"].astype(str)
    primary = primary.set_index("nodule_uid", drop=False)
    index = roi_index.copy()
    index["nodule_uid"] = index["nodule_uid"].astype(str)
    index = index.set_index("nodule_uid", drop=False)
    if not set(uids) <= set(primary.index) or not set(uids) <= set(index.index):
        raise ValueError("P5_PARTITION_SOURCE_SET_MISMATCH")
    base = Path(roi_index_path).resolve().parent.parent
    records: list[SampleRecord] = []
    for uid in uids:
        row = primary.loc[uid]
        roi = index.loc[uid]
        relative = Path(str(roi["relative_roi_path"]))
        path = (base / relative).resolve()
        if path.parent != (base / "rois").resolve() or not path.is_file():
            raise ValueError(f"P5_ROI_PATH_MISMATCH:{uid}")
        if sha256_file(path) != str(roi["roi_file_sha256"]):
            raise ValueError(f"P5_ROI_HASH_MISMATCH:{uid}")
        extreme = _boolean(row["extreme_binary_eligible"])
        label = int(row["extreme_binary_label"]) if extreme else None
        records.append(
            SampleRecord(
                nodule_uid=uid,
                patient_key=patient_key(str(row["patient_id"])),
                roi_path=path,
                target_normalized=float(row["malignancy_target_normalized"]),
                target_1_to_5=float(row["mean_malignancy"]),
                extreme_binary_eligible=extreme,
                extreme_binary_label=label,
            )
        )
    return records


class ValidationMSEPlateau:
    """Exact four-bad-epoch multiplicative schedule from the P5 execution config."""

    def __init__(
        self,
        optimizer: Any,
        *,
        factor: float = 0.9,
        patience: int = 4,
        min_delta: float = 1e-4,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_lr = float(min_lr)
        self.best = math.inf
        self.bad_epoch_counter = 0

    def step(self, validation_mse: float) -> bool:
        value = float(validation_mse)
        if not math.isfinite(value):
            raise ValueError("NONFINITE_VALIDATION_MSE")
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epoch_counter = 0
            return False
        self.bad_epoch_counter += 1
        if self.bad_epoch_counter < self.patience:
            return False
        for group in self.optimizer.param_groups:
            group["lr"] = max(self.min_lr, float(group["lr"]) * self.factor)
        self.bad_epoch_counter = 0
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_lr": self.min_lr,
            "best": self.best,
            "bad_epoch_counter": self.bad_epoch_counter,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "factor": self.factor,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_lr": self.min_lr,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("P5_SCHEDULER_POLICY_MISMATCH")
        self.best = float(state["best"])
        self.bad_epoch_counter = int(state["bad_epoch_counter"])


def _loader(dataset: Any, *, batch_size: int, num_workers: int) -> Any:
    torch = _torch()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
    )


def _ordered_records(
    records: Sequence[SampleRecord],
    base_seed: int,
    fold_index: int,
    epoch_index: int,
) -> list[SampleRecord]:
    by_uid = {record.nodule_uid: record for record in records}
    order = epoch_uid_order(by_uid, base_seed, fold_index, epoch_index)
    return [by_uid[uid] for uid in order]


def train_one_epoch(
    model: Any,
    records: Sequence[SampleRecord],
    optimizer: Any,
    device: Any,
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    torch = _torch()
    ordered = _ordered_records(records, base_seed, fold_index, epoch_index)
    dataset = ROIDataset.build(
        ordered,
        training=True,
        base_seed=base_seed,
        fold_index=fold_index,
        epoch_index=epoch_index,
    )
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.train()
    squared_error_sum = 0.0
    sample_count = 0
    observed_uids: list[str] = []
    for batch in loader:
        image = batch["image"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        score = model(image)
        if score.shape != target.shape:
            raise ValueError("P5_MODEL_OUTPUT_SHAPE_MISMATCH")
        loss = torch.nn.functional.mse_loss(score, target, reduction="mean")
        if not torch.isfinite(loss):
            raise ValueError("NONFINITE_TRAIN_LOSS")
        loss.backward()
        optimizer.step()
        batch_count = int(target.shape[0])
        squared_error_sum += float(
            torch.nn.functional.mse_loss(score.detach(), target, reduction="sum").cpu()
        )
        sample_count += batch_count
        observed_uids.extend(map(str, batch["nodule_uid"]))
    expected_uids = [record.nodule_uid for record in ordered]
    if observed_uids != expected_uids or len(set(observed_uids)) != len(records):
        raise ValueError("P5_TRAIN_SAMPLE_COVERAGE_MISMATCH")
    return {
        "mse": squared_error_sum / sample_count,
        "sample_count": sample_count,
        "nodule_set_sha256": sha256_bytes(canonical_json_bytes(sorted(observed_uids))),
    }


def predict_records(
    model: Any,
    records: Sequence[SampleRecord],
    device: Any,
    *,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    torch = _torch()
    ordered = sorted(records, key=lambda record: record.nodule_uid)
    dataset = ROIDataset.build(
        ordered,
        training=False,
        base_seed=0,
        fold_index=0,
        epoch_index=0,
    )
    loader = _loader(dataset, batch_size=batch_size, num_workers=num_workers)
    model.eval()
    result: list[dict[str, Any]] = []
    by_uid = {record.nodule_uid: record for record in ordered}
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            score = model(image).detach().cpu().reshape(-1).numpy()
            if not np.isfinite(score).all():
                raise ValueError("NONFINITE_PREDICTION")
            for uid, raw in zip(map(str, batch["nodule_uid"]), score, strict=True):
                record = by_uid[uid]
                raw_value = float(raw)
                result.append(
                    {
                        "nodule_uid": uid,
                        "patient_key": record.patient_key,
                        "target_normalized": record.target_normalized,
                        "target_1_to_5": record.target_1_to_5,
                        "malignancy_raw_score": raw_value,
                        "malignancy_score_normalized": raw_value,
                        "malignancy_score_1_to_5": 1.0 + 4.0 * raw_value,
                        "extreme_binary_eligible": record.extreme_binary_eligible,
                        "extreme_binary_label": record.extreme_binary_label,
                    }
                )
    if [row["nodule_uid"] for row in result] != [record.nodule_uid for record in ordered]:
        raise ValueError("P5_PREDICTION_ORDER_MISMATCH")
    return result


def mse_from_predictions(predictions: Sequence[Mapping[str, Any]]) -> float:
    errors = np.asarray(
        [
            float(row["malignancy_raw_score"]) - float(row["target_normalized"])
            for row in predictions
        ],
        dtype=np.float64,
    )
    if not len(errors):
        raise ValueError("EMPTY_PREDICTIONS")
    return float(np.mean(np.square(errors)))


def checkpoint_improves(current_validation_mse: float, best_validation_mse: float) -> bool:
    """Use exact validation MSE and preserve the earlier epoch on a tie."""
    current = float(current_validation_mse)
    best = float(best_validation_mse)
    if not math.isfinite(current):
        raise ValueError("NONFINITE_VALIDATION_MSE")
    return current < best


def serialized_float_consistent(left: float, right: float) -> bool:
    """Compare equivalent finite values after JSON/CSV serialization round-trips."""
    left_value = float(left)
    right_value = float(right)
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False
    return math.isclose(
        left_value,
        right_value,
        rel_tol=SERIALIZED_FLOAT_REL_TOL,
        abs_tol=SERIALIZED_FLOAT_ABS_TOL,
    )


def regression_metrics(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr

    target = np.asarray([float(row["target_normalized"]) for row in predictions])
    score = np.asarray([float(row["malignancy_raw_score"]) for row in predictions])
    if not len(target) or not np.isfinite(target).all() or not np.isfinite(score).all():
        raise ValueError("INVALID_METRIC_INPUT")
    error = score - target
    original_error = 4.0 * error
    return {
        "samples": int(len(target)),
        "normalized_mae": float(np.mean(np.abs(error))),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "original_scale_mae": float(np.mean(np.abs(original_error))),
        "original_scale_rmse": float(np.sqrt(np.mean(np.square(original_error)))),
        "pearson": float(pearsonr(target, score).statistic),
        "spearman": float(spearmanr(target, score).statistic),
        "prediction_min": float(score.min()),
        "prediction_max": float(score.max()),
        "prediction_below_0_rate": float(np.mean(score < 0.0)),
        "prediction_above_1_rate": float(np.mean(score > 1.0)),
        "prediction_below_1_on_original_scale_rate": float(np.mean((1.0 + 4.0 * score) < 1.0)),
        "prediction_above_5_on_original_scale_rate": float(np.mean((1.0 + 4.0 * score) > 5.0)),
    }


def capture_rng_state() -> dict[str, Any]:
    torch = _torch()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    torch = _torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _provenance(
    scientific_config: Mapping[str, Any],
    execution_config_sha256: str,
    split: Mapping[str, Any],
    initialization: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": scientific_config["protocol"]["version"],
        "scientific_config_sha256": compute_config_sha256(scientific_config),
        "execution_config_sha256": execution_config_sha256,
        "split_sha256": split["split_sha256"],
        "fold_index": int(split["fold_index"]),
        "model": MODEL_NAME,
        "task_output": "unconstrained_linear_raw_score",
        "task_loss": "mean_squared_error_on_normalized_target",
        **dict(initialization),
    }
    if execution_config is not None:
        provenance["execution_profile_id"] = execution_config["execution_profile"]["profile_id"]
        provenance["formal_gpu_model"] = execution_config["execution_profile"]["formal_gpu_model"]
        provenance.update(reproducibility_provenance(execution_config))
    return provenance


def checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: ValidationMSEPlateau,
    *,
    epoch_index: int,
    validation_mse: float,
    best_epoch_index: int,
    best_validation_mse: float,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "epoch_index": int(epoch_index),
        "validation_mse": float(validation_mse),
        "best_epoch_index": int(best_epoch_index),
        "best_validation_mse": float(best_validation_mse),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
    }


def _load_checkpoint(path: Path, expected_provenance: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("P5_CHECKPOINT_SCHEMA_MISMATCH")
    if payload.get("provenance") != dict(expected_provenance):
        raise ValueError("P5_CHECKPOINT_PROVENANCE_MISMATCH")
    return payload


def _optimizer(model: Any, execution_config: Mapping[str, Any]) -> Any:
    torch = _torch()
    reported = execution_config["reference_reported"]
    project = execution_config["project_preregistered"]["optimizer"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(reported["learning_rate"]),
        betas=tuple(map(float, project["betas"])),
        eps=float(project["epsilon"]),
        weight_decay=float(project["weight_decay"]),
    )


def _scheduler(optimizer: Any, execution_config: Mapping[str, Any]) -> ValidationMSEPlateau:
    reported = execution_config["reference_reported"]["scheduler"]
    project = execution_config["project_preregistered"]["scheduler"]
    return ValidationMSEPlateau(
        optimizer,
        factor=float(reported["factor"]),
        patience=int(reported["bad_epochs_before_decay"]),
        min_delta=float(project["min_delta"]),
        min_lr=float(project["min_learning_rate"]),
    )


def _prepare_sources(
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    Path,
]:
    scientific = load_config(scientific_config_path)
    execution, execution_hash = validate_execution_config(execution_config_path)
    split_path = Path(scientific["paths"]["split_directory"]) / f"fold_{fold_index}.json"
    split = read_split(split_path)
    if int(split["fold_index"]) != int(fold_index):
        raise ValueError("P5_SPLIT_FOLD_MISMATCH")
    if split["config_sha256"] != compute_config_sha256(scientific):
        raise ValueError("P5_SPLIT_CONFIG_MISMATCH")
    if split["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("P5_SPLIT_MANIFEST_MISMATCH")
    if split["roi_index_sha256"] != sha256_file(roi_index_path):
        raise ValueError("P5_SPLIT_ROI_INDEX_MISMATCH")
    manifest = pd.read_parquet(manifest_path)
    roi_index = pd.read_parquet(roi_index_path)
    encoder_path = Path(scientific["paths"]["encoder_initialization_directory"]) / f"fold_{fold_index}.pt"
    return scientific, execution, execution_hash, split, manifest, roi_index, encoder_path


def run_directory(fold_index: int, root: str | Path = "runs/baseline_v2/blackbox") -> Path:
    return Path(root) / f"fold_{fold_index}"


def train_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
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
    with exclusive_fold_lifecycle_lock(output / ".p5_lifecycle.lock"):
        return _train_fold_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
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
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = _prepare_sources(
        scientific_config_path,
        execution_config_path,
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
    scheduler = _scheduler(optimizer, execution)
    provenance = _provenance(scientific, execution_hash, split, initialization, execution)
    train_records = build_partition_records(manifest, roi_index, split, "train", roi_index_path)
    validation_records = build_partition_records(manifest, roi_index, split, "validation", roi_index_path)
    output = run_directory(fold_index, output_root)
    output.mkdir(parents=True, exist_ok=True)
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    complete_path = output / "training_complete.json"
    if complete_path.exists():
        raise FileExistsError("P5_TRAINING_ALREADY_COMPLETE")
    history: list[dict[str, Any]] = []
    start_epoch = 0
    best_epoch = -1
    best_validation = math.inf
    if resume:
        if not last_path.exists():
            raise FileNotFoundError("P5_RESUME_CHECKPOINT_MISSING")
        payload = _load_checkpoint(last_path, provenance)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
        start_epoch = int(payload["epoch_index"]) + 1
        best_epoch = int(payload["best_epoch_index"])
        best_validation = float(payload["best_validation_mse"])
        history_path = output / "history.csv"
        history = list(payload.get("history", []))
        if len(history) != start_epoch:
            raise ValueError("P5_RESUME_HISTORY_MISMATCH")
        _atomic_csv(output / "history.csv", history, list(history[0]) if history else (
            "epoch_index",
            "train_mse",
            "validation_mse",
            "learning_rate_start",
            "learning_rate_end",
            "scheduler_decayed",
            "scheduler_best",
            "scheduler_bad_epoch_counter",
            "train_sample_count",
            "train_nodule_set_sha256",
        ))
    elif any(path.exists() for path in (last_path, best_path, output / "history.csv")):
        raise FileExistsError("P5_RUN_EXISTS_USE_RESUME_OR_INVALIDATE")

    epochs = int(execution["reference_reported"]["epochs"])
    batch_size = int(execution["project_preregistered"]["batching"]["micro_batch_size"])
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_report = train_one_epoch(
            model,
            train_records,
            optimizer,
            device,
            base_seed=int(scientific["reproducibility"]["base_seed"]),
            fold_index=fold_index,
            epoch_index=epoch,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        validation_predictions = predict_records(
            model,
            validation_records,
            device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        validation_mse = mse_from_predictions(validation_predictions)
        improved_checkpoint = checkpoint_improves(validation_mse, best_validation)
        if improved_checkpoint:
            best_validation = validation_mse
            best_epoch = epoch
        decayed = scheduler.step(validation_mse)
        if improved_checkpoint:
            best_payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch_index=epoch,
                validation_mse=validation_mse,
                best_epoch_index=best_epoch,
                best_validation_mse=best_validation,
                provenance=provenance,
            )
            _atomic_torch_save(best_path, best_payload)
        row = {
            "epoch_index": epoch,
            "train_mse": train_report["mse"],
            "validation_mse": validation_mse,
            "learning_rate_start": current_lr,
            "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
            "scheduler_decayed": bool(decayed),
            "scheduler_best": scheduler.best,
            "scheduler_bad_epoch_counter": scheduler.bad_epoch_counter,
            "train_sample_count": train_report["sample_count"],
            "train_nodule_set_sha256": train_report["nodule_set_sha256"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        _atomic_csv(output / "history.csv", history, list(row))
        last_payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch_index=epoch,
            validation_mse=validation_mse,
            best_epoch_index=best_epoch,
            best_validation_mse=best_validation,
            provenance=provenance,
        )
        last_payload["history"] = list(history)
        _atomic_torch_save(last_path, last_payload)
        print(
            canonical_json_bytes(
                {
                    "event": "P5_EPOCH_COMPLETE",
                    "fold_index": fold_index,
                    **row,
                }
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
        raise ValueError("P5_TRAINING_INCOMPLETE")
    expected_train = int(split["partitions"]["train"]["summary"]["nodules"])
    if any(int(row["train_sample_count"]) != expected_train for row in history):
        raise ValueError("P5_EPOCH_TRAIN_COVERAGE_MISMATCH")
    completion = {
        **provenance,
        "status": "TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "epochs_completed": epochs,
        "best_epoch_index": best_epoch,
        "best_validation_mse": best_validation,
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "history_sha256": sha256_file(output / "history.csv"),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "test_evaluated": False,
    }
    _training_plot(history, output / "training_curves.png")
    runtime = {
        **provenance,
        **_runtime_environment(device),
        "epochs_this_invocation": epochs - start_epoch,
        "epochs_total": epochs,
        "wall_seconds_this_invocation": time.monotonic() - started,
        "sum_epoch_seconds": float(sum(float(row["epoch_seconds"]) for row in history)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "training_curves_sha256": sha256_file(output / "training_curves.png"),
    }
    _atomic_json(output / "runtime.json", runtime)
    completion["runtime_sha256"] = sha256_file(output / "runtime.json")
    completion["training_curves_sha256"] = runtime["training_curves_sha256"]
    _atomic_json(complete_path, completion)
    return completion


def _load_best_model(
    scientific: Mapping[str, Any],
    split: Mapping[str, Any],
    encoder_path: Path,
    checkpoint_path: Path,
    execution_hash: str,
    execution_config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    provenance = _provenance(scientific, execution_hash, split, initialization, execution_config)
    payload = _load_checkpoint(checkpoint_path, provenance)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, initialization, payload


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        temporary = Path(temporary_name)
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _test_row_provenance(
    scientific: Mapping[str, Any],
    execution_hash: str,
    split: Mapping[str, Any],
    initialization: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "fold_index": int(split["fold_index"]),
        "model": MODEL_NAME,
        "scientific_config_sha256": compute_config_sha256(scientific),
        "execution_config_sha256": execution_hash,
        "split_sha256": split["split_sha256"],
        **dict(initialization),
        "checkpoint_sha256": checkpoint_sha256,
    }


def _validate_test_prediction_frame(
    frame: pd.DataFrame,
    records: Sequence[SampleRecord],
    row_provenance: Mapping[str, Any],
) -> None:
    forbidden = {"probability", "logit", "concept", "mask"}
    if any(any(token in column.lower() for token in forbidden) for column in frame.columns):
        raise ValueError("P5_TEST_PREDICTION_FORBIDDEN_COLUMN")
    expected = {record.nodule_uid: record for record in records}
    if len(frame) != len(expected) or frame["nodule_uid"].astype(str).duplicated().any():
        raise ValueError("P5_TEST_PREDICTION_COUNT_MISMATCH")
    if set(frame["nodule_uid"].astype(str)) != set(expected):
        raise ValueError("P5_TEST_PREDICTION_UID_SET_MISMATCH")
    for key, value in row_provenance.items():
        if key not in frame or not all(observed == value for observed in frame[key].tolist()):
            raise ValueError(f"P5_TEST_PREDICTION_PROVENANCE_MISMATCH:{key}")
    by_uid = frame.set_index(frame["nodule_uid"].astype(str), drop=False)
    for uid, record in expected.items():
        row = by_uid.loc[uid]
        if str(row["patient_key"]) != record.patient_key:
            raise ValueError("P5_TEST_PREDICTION_PATIENT_KEY_MISMATCH")
        if not math.isclose(float(row["target_normalized"]), record.target_normalized, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("P5_TEST_PREDICTION_TARGET_MISMATCH")
        if not math.isclose(float(row["target_1_to_5"]), record.target_1_to_5, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("P5_TEST_PREDICTION_TARGET_MISMATCH")
        raw = float(row["malignancy_raw_score"])
        if not math.isfinite(raw):
            raise ValueError("P5_TEST_PREDICTION_NONFINITE")
        if float(row["malignancy_score_normalized"]) != raw:
            raise ValueError("P5_TEST_PREDICTION_SCORE_ALIAS_MISMATCH")
        if not math.isclose(float(row["malignancy_score_1_to_5"]), 1.0 + 4.0 * raw, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("P5_TEST_PREDICTION_SCALE_MISMATCH")


def _validate_evaluation_artifacts(
    output: Path,
    evaluation: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
    completion: Mapping[str, Any],
    records: Sequence[SampleRecord],
    row_provenance: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if evaluation.get("status") != "TEST_EVALUATED_ONCE":
        raise ValueError("P5_TEST_EVALUATION_STATUS_MISMATCH")
    for key, value in expected_provenance.items():
        if evaluation.get(key) != value:
            raise ValueError(f"P5_TEST_EVALUATION_PROVENANCE_MISMATCH:{key}")
    if evaluation.get("best_checkpoint_sha256") != completion.get("best_checkpoint_sha256"):
        raise ValueError("P5_TEST_EVALUATION_CHECKPOINT_MISMATCH")
    predictions_path = output / "test_predictions.parquet"
    metrics_path = output / "metrics.json"
    plot_path = output / "prediction_vs_target.png"
    claim_path = output / "test_claim.json"
    if evaluation.get("test_claim_sha256") != sha256_file(claim_path):
        raise ValueError("P5_TEST_CLAIM_HASH_MISMATCH")
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    if claim.get("status") != "TEST_EVALUATION_CLAIMED":
        raise ValueError("P5_TEST_CLAIM_STATUS_MISMATCH")
    if claim.get("best_checkpoint_sha256") != completion.get("best_checkpoint_sha256"):
        raise ValueError("P5_TEST_CLAIM_CHECKPOINT_MISMATCH")
    for key, value in expected_provenance.items():
        if claim.get(key) != value:
            raise ValueError(f"P5_TEST_CLAIM_PROVENANCE_MISMATCH:{key}")
    if evaluation.get("test_predictions_sha256") != sha256_file(predictions_path):
        raise ValueError("P5_TEST_PREDICTIONS_HASH_MISMATCH")
    if evaluation.get("metrics_sha256") != sha256_file(metrics_path):
        raise ValueError("P5_TEST_METRICS_HASH_MISMATCH")
    if evaluation.get("prediction_plot_sha256") != sha256_file(plot_path):
        raise ValueError("P5_TEST_PLOT_HASH_MISMATCH")
    frame = pd.read_parquet(predictions_path)
    _validate_test_prediction_frame(frame, records, row_provenance)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    observed_metrics = regression_metrics(frame.to_dict("records"))
    if metrics != observed_metrics:
        raise ValueError("P5_TEST_METRICS_RECONSTRUCTION_MISMATCH")
    if int(evaluation.get("test_samples", -1)) != len(frame):
        raise ValueError("P5_TEST_EVALUATION_COUNT_MISMATCH")
    return frame, metrics


def evaluate_test_once(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p5_lifecycle.lock"):
        return _evaluate_test_once_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
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
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    torch = _torch()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = _prepare_sources(
        scientific_config_path,
        execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    output = run_directory(fold_index, output_root)
    evaluation_path = output / "test_evaluation.json"
    claim_path = output / "test_claim.json"
    predictions_path = output / "test_predictions.parquet"
    completion_path = output / "training_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError("P5_TRAINING_NOT_SEALED")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    best_path = output / "best.pt"
    if sha256_file(best_path) != completion["best_checkpoint_sha256"]:
        raise ValueError("P5_BEST_CHECKPOINT_SEAL_MISMATCH")
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization, checkpoint = _load_best_model(
        scientific,
        split,
        encoder_path,
        best_path,
        execution_hash,
        execution,
    )
    if int(checkpoint["epoch_index"]) != int(completion["best_epoch_index"]):
        raise ValueError("P5_BEST_EPOCH_SEAL_MISMATCH")
    expected_provenance = _provenance(scientific, execution_hash, split, initialization, execution)
    row_provenance = _test_row_provenance(
        scientific,
        execution_hash,
        split,
        initialization,
        completion["best_checkpoint_sha256"],
    )
    test_records = build_partition_records(manifest, roi_index, split, "test", roi_index_path)
    if len(test_records) != EXPECTED_FOLD_TEST_COUNTS[fold_index]:
        raise ValueError("P5_TEST_COUNT_MISMATCH")

    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        frame, metrics = _validate_evaluation_artifacts(
            output,
            evaluation,
            expected_provenance,
            completion,
            test_records,
            row_provenance,
        )
        if completion.get("test_evaluated") is True:
            if completion.get("status") != "TRAINING_COMPLETE_TEST_EVALUATED":
                raise ValueError("P5_COMPLETION_TEST_STATUS_MISMATCH")
            if completion.get("test_evaluation_sha256") != sha256_file(evaluation_path):
                raise ValueError("P5_COMPLETION_TEST_HASH_MISMATCH")
            raise FileExistsError("P5_TEST_ALREADY_EVALUATED")
        completion["status"] = "TRAINING_COMPLETE_TEST_EVALUATED"
        completion["test_evaluated"] = True
        completion["test_evaluation_sha256"] = sha256_file(evaluation_path)
        _atomic_json(completion_path, completion)
        return {
            "evaluation": evaluation,
            "metrics": metrics,
            "recovered_after_evaluation_seal": True,
            "test_samples": len(frame),
        }
    if completion.get("test_evaluated") is True:
        raise ValueError("P5_COMPLETION_CLAIMS_MISSING_TEST_EVALUATION")

    claim = {
        "status": "TEST_EVALUATION_CLAIMED",
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "best_epoch_index": int(completion["best_epoch_index"]),
        **expected_provenance,
    }
    if claim_path.exists():
        if json.loads(claim_path.read_text(encoding="utf-8")) != claim:
            raise ValueError("P5_TEST_CLAIM_PROVENANCE_MISMATCH")
    else:
        _atomic_json(claim_path, claim)

    if predictions_path.exists():
        prediction_frame = pd.read_parquet(predictions_path)
        _validate_test_prediction_frame(prediction_frame, test_records, row_provenance)
        predictions = prediction_frame.to_dict("records")
    else:
        model.to(device)
        predictions = predict_records(
            model,
            test_records,
            device,
            batch_size=int(execution["project_preregistered"]["batching"]["micro_batch_size"]),
            num_workers=num_workers,
        )
        for row in predictions:
            row.update(row_provenance)
        prediction_frame = pd.DataFrame(predictions)
        _validate_test_prediction_frame(prediction_frame, test_records, row_provenance)
        _atomic_parquet(predictions_path, prediction_frame)
    metrics = regression_metrics(predictions)
    _atomic_json(output / "metrics.json", metrics)
    _prediction_plot(predictions, output / "prediction_vs_target.png")
    evaluation = {
        "status": "TEST_EVALUATED_ONCE",
        "fold_index": fold_index,
        "best_epoch_index": int(completion["best_epoch_index"]),
        "best_validation_mse": float(completion["best_validation_mse"]),
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "test_predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(output / "metrics.json"),
        "prediction_plot_sha256": sha256_file(output / "prediction_vs_target.png"),
        "test_samples": len(predictions),
        "test_claim_sha256": sha256_file(claim_path),
        **expected_provenance,
    }
    _atomic_json(evaluation_path, evaluation)
    completion["status"] = "TRAINING_COMPLETE_TEST_EVALUATED"
    completion["test_evaluated"] = True
    completion["test_evaluation_sha256"] = sha256_file(evaluation_path)
    _atomic_json(completion_path, completion)
    return {"evaluation": evaluation, "metrics": metrics}


def overfit_check(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    samples: int,
    steps: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run a controlled train-only sanity fit without creating a formal run."""
    torch = _torch()
    if samples < 2 or steps < 1:
        raise ValueError("INVALID_OVERFIT_CHECK_SIZE")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = _prepare_sources(
        scientific_config_path,
        execution_config_path,
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
        build_partition_records(manifest, roi_index, split, "train", roi_index_path),
        key=lambda record: record.nodule_uid,
    )[:samples]
    dataset = ROIDataset.build(
        records,
        training=False,
        base_seed=0,
        fold_index=fold_index,
        epoch_index=0,
    )
    batch = next(iter(_loader(dataset, batch_size=samples, num_workers=0)))
    image = batch["image"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    model.train()
    with torch.no_grad():
        initial_mse = float(torch.nn.functional.mse_loss(model(image), target).cpu())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(image), target)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_mse = float(torch.nn.functional.mse_loss(model(image), target).cpu())
    if not math.isfinite(final_mse) or final_mse >= initial_mse:
        raise RuntimeError("P5_OVERFIT_SANITY_DID_NOT_IMPROVE")
    report = {
        "status": "PASS",
        "scope": "train_only_controlled_overfit_sanity",
        "formal_run": False,
        "augmentation_enabled": False,
        "fold_index": fold_index,
        "samples": samples,
        "steps": steps,
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "relative_final_mse": final_mse / initial_mse,
        "scientific_config_sha256": compute_config_sha256(scientific),
        "execution_config_sha256": execution_hash,
        "execution_profile_id": execution["execution_profile"]["profile_id"],
        "formal_gpu_model": execution["execution_profile"]["formal_gpu_model"],
        **reproducibility_provenance(execution),
        "split_sha256": split["split_sha256"],
        **initialization,
        **_runtime_environment(device),
    }
    _atomic_json(output_path, report)
    return report


def preflight(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_path: Path,
) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = _prepare_sources(
        scientific_config_path,
        execution_config_path,
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
    records = build_partition_records(manifest, roi_index, split, "train", roi_index_path)
    ordered = _ordered_records(
        records,
        int(scientific["reproducibility"]["base_seed"]),
        fold_index,
        0,
    )[:16]
    dataset = ROIDataset.build(
        ordered,
        training=True,
        base_seed=int(scientific["reproducibility"]["base_seed"]),
        fold_index=fold_index,
        epoch_index=0,
    )
    batch = next(iter(_loader(dataset, batch_size=16, num_workers=0)))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    image = batch["image"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    score = model(image)
    loss = torch.nn.functional.mse_loss(score, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = reserved / total
    limit = float(execution["project_preregistered"]["preflight"]["maximum_peak_reserved_fraction"])
    if fraction > limit:
        raise RuntimeError(f"P5_PREFLIGHT_MEMORY_LIMIT_EXCEEDED:{fraction}")
    report = {
        "status": "PASS",
        "fold_index": fold_index,
        "batch_size": int(image.shape[0]),
        "forward": True,
        "backward": True,
        "adam_step": True,
        "loss": float(loss.detach().cpu()),
        "peak_reserved_bytes": reserved,
        "gpu_total_bytes": total,
        "peak_reserved_fraction": fraction,
        "maximum_allowed_fraction": limit,
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": importlib.metadata.version("torch"),
        "monai_version": importlib.metadata.version("monai"),
        "cuda_runtime": torch.version.cuda,
        "scientific_config_sha256": compute_config_sha256(scientific),
        "execution_config_sha256": execution_hash,
        "execution_profile_id": execution["execution_profile"]["profile_id"],
        "formal_gpu_model": execution["execution_profile"]["formal_gpu_model"],
        **reproducibility_provenance(execution),
        "split_sha256": split["split_sha256"],
        **initialization,
    }
    _atomic_json(output_path, report)
    return report


def verify_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    require_test: bool = True,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p5_lifecycle.lock"):
        return _verify_fold_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            output_root=output_root,
            require_test=require_test,
        )


def _verify_fold_locked(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    require_test: bool = True,
) -> dict[str, Any]:
    scientific, execution, execution_hash, split, manifest, roi_index, encoder_path = _prepare_sources(
        scientific_config_path,
        execution_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    _model, initialization = build_initialized_model(scientific, split, encoder_path)
    provenance = _provenance(scientific, execution_hash, split, initialization, execution)
    output = run_directory(fold_index, output_root)
    completion = json.loads((output / "training_complete.json").read_text(encoding="utf-8"))
    if any(completion.get(key) != value for key, value in provenance.items()):
        raise ValueError("P5_COMPLETION_PROVENANCE_MISMATCH")
    history = pd.read_csv(output / "history.csv")
    expected_epochs = int(execution["reference_reported"]["epochs"])
    if len(history) != expected_epochs or history["epoch_index"].tolist() != list(range(expected_epochs)):
        raise ValueError("P5_HISTORY_EPOCH_MISMATCH")
    expected_train = int(split["partitions"]["train"]["summary"]["nodules"])
    if not (history["train_sample_count"] == expected_train).all():
        raise ValueError("P5_HISTORY_TRAIN_COVERAGE_MISMATCH")
    best_path = output / "best.pt"
    if sha256_file(best_path) != completion["best_checkpoint_sha256"]:
        raise ValueError("P5_BEST_CHECKPOINT_HASH_MISMATCH")
    best_payload = _load_checkpoint(best_path, provenance)
    minimum_index = int(history["validation_mse"].idxmin())
    minimum_epoch = int(history.iloc[minimum_index]["epoch_index"])
    if minimum_epoch != int(completion["best_epoch_index"]):
        raise ValueError("P5_CHECKPOINT_SELECTION_MISMATCH")
    if int(best_payload["epoch_index"]) != minimum_epoch:
        raise ValueError("P5_BEST_PAYLOAD_EPOCH_MISMATCH")
    if not serialized_float_consistent(
        float(completion["best_validation_mse"]),
        float(history.iloc[minimum_index]["validation_mse"]),
    ):
        raise ValueError("P5_BEST_OBJECTIVE_MISMATCH")
    if completion.get("history_sha256") != sha256_file(output / "history.csv"):
        raise ValueError("P5_HISTORY_HASH_MISMATCH")
    if completion.get("last_checkpoint_sha256") != sha256_file(output / "last.pt"):
        raise ValueError("P5_LAST_CHECKPOINT_HASH_MISMATCH")
    if completion.get("runtime_sha256") != sha256_file(output / "runtime.json"):
        raise ValueError("P5_RUNTIME_HASH_MISMATCH")
    if completion.get("training_curves_sha256") != sha256_file(output / "training_curves.png"):
        raise ValueError("P5_TRAINING_CURVE_HASH_MISMATCH")
    report: dict[str, Any] = {
        "status": "PASS",
        "fold_index": fold_index,
        "epochs": len(history),
        "train_samples_per_epoch": expected_train,
        "best_epoch_index": minimum_epoch,
        "best_validation_mse": float(history.iloc[minimum_index]["validation_mse"]),
        **provenance,
    }
    if require_test:
        evaluation = json.loads((output / "test_evaluation.json").read_text(encoding="utf-8"))
        if completion.get("status") != "TRAINING_COMPLETE_TEST_EVALUATED" or completion.get("test_evaluated") is not True:
            raise ValueError("P5_COMPLETION_TEST_STATUS_MISMATCH")
        evaluation_path = output / "test_evaluation.json"
        if completion.get("test_evaluation_sha256") != sha256_file(evaluation_path):
            raise ValueError("P5_COMPLETION_TEST_HASH_MISMATCH")
        test_records = build_partition_records(manifest, roi_index, split, "test", roi_index_path)
        row_provenance = _test_row_provenance(
            scientific,
            execution_hash,
            split,
            initialization,
            completion["best_checkpoint_sha256"],
        )
        predictions, _metrics = _validate_evaluation_artifacts(
            output,
            evaluation,
            provenance,
            completion,
            test_records,
            row_provenance,
        )
        expected_test = EXPECTED_FOLD_TEST_COUNTS[fold_index]
        if len(predictions) != expected_test:
            raise ValueError("P5_VERIFY_TEST_COUNT_MISMATCH")
        report["test_samples"] = len(predictions)
        report["test_evaluated_once"] = True
    return report


def verify_all(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    reports = [
        verify_fold(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold,
            output_root=output_root,
        )
        for fold in range(5)
    ]
    predictions = [
        pd.read_parquet(run_directory(fold, output_root) / "test_predictions.parquet")
        for fold in range(5)
    ]
    pooled = pd.concat(predictions, ignore_index=True)
    if len(pooled) != 2633 or pooled["nodule_uid"].nunique() != 2633:
        raise ValueError("P5_OOF_NODULE_SET_MISMATCH")
    if pooled["patient_key"].nunique() != 868:
        raise ValueError("P5_OOF_PATIENT_SET_MISMATCH")
    if pooled.groupby("patient_key")["fold_index"].nunique().max() != 1:
        raise ValueError("P5_OOF_PATIENT_LEAKAGE")
    return {
        "status": "PASS",
        "oof_nodules": 2633,
        "oof_patients": 868,
        "fold_test_counts": [int(len(frame)) for frame in predictions],
        "folds": reports,
    }


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_v2.yaml"))
    parser.add_argument("--execution-config", type=Path, default=EXECUTION_CONFIG_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/baseline_v2/manifests/nodules.parquet"))
    parser.add_argument("--roi-index", type=Path, default=Path("artifacts/baseline_v2/manifests/roi_index.parquet"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/baseline_v2/blackbox"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _common_arguments(preflight_parser)
    preflight_parser.add_argument("--fold", type=int, required=True, choices=range(5))
    preflight_parser.add_argument("--output", type=Path, default=Path("runs/baseline_v2/blackbox/fold_0/preflight.json"))

    overfit_parser = subparsers.add_parser("overfit-check")
    _common_arguments(overfit_parser)
    overfit_parser.add_argument("--fold", type=int, required=True, choices=range(5))
    overfit_parser.add_argument("--device", default="cuda")
    overfit_parser.add_argument("--samples", type=int, default=8)
    overfit_parser.add_argument("--steps", type=int, default=40)
    overfit_parser.add_argument("--output", type=Path, default=Path("runs/baseline_v2/blackbox/fold_0/overfit_sanity.json"))

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
        "manifest_path": arguments.manifest,
        "roi_index_path": arguments.roi_index,
    }
    if arguments.command == "preflight":
        result = preflight(
            **common,
            fold_index=arguments.fold,
            output_path=arguments.output,
        )
    elif arguments.command == "overfit-check":
        result = overfit_check(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            samples=arguments.samples,
            steps=arguments.steps,
            output_path=arguments.output,
        )
    elif arguments.command == "train":
        result = train_fold(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
            output_root=arguments.output_root,
            resume=arguments.resume,
        )
    elif arguments.command == "evaluate-test":
        result = evaluate_test_once(
            **common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
            output_root=arguments.output_root,
        )
    elif arguments.scope == "all":
        if arguments.training_only:
            raise ValueError("P5_ALL_SCOPE_REQUIRES_TEST_EVALUATIONS")
        result = verify_all(**common, output_root=arguments.output_root)
    else:
        result = verify_fold(
            **common,
            fold_index=arguments.fold,
            output_root=arguments.output_root,
            require_test=not arguments.training_only,
        )
    print(canonical_json_bytes(result).decode("utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
