"""Formal lifecycle for the Baseline-v2 P8 learned-softmax GAM."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.config import compute_config_sha256
from lidc_baseline.p4_prepare import (
    canonical_json_bytes,
    encoder_state_sha256,
    sha256_bytes,
    sha256_file,
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
    concept_group_loss_sums,
)
from lidc_baseline.p8_gam import (
    EXPERTS_PER_GROUP,
    MODEL_NAME,
    P8_EXECUTION_CONFIG_DEFAULT,
    build_initialized_model,
    gam_losses,
    task_predictions_and_contributions,
    validate_p8_execution_config,
)


SCHEMA_VERSION = 1
RUN_ROOT_DEFAULT = Path("runs/baseline_v2/gam")
NUMERIC_SCHEMA = "p8_fp32_serialization_scale_aware_v1"
NUMERIC_ABSOLUTE_FLOOR = 1e-6
NUMERIC_FLOAT32_OPERATION_FACTOR = 64.0


def _torch() -> Any:
    import torch

    return torch


def _alpha_snapshot(model: Any) -> dict[str, Any]:
    """Return canonical per-group logits, weights, and semantic hashes."""
    torch = _torch()
    logits: OrderedDict[str, list[float]] = OrderedDict()
    weights: OrderedDict[str, list[float]] = OrderedDict()
    group_hashes: OrderedDict[str, str] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        value = model.alpha_logits[group].detach().cpu().contiguous()
        logits[group] = [float(item) for item in value.tolist()]
        weights[group] = [
            float(item) for item in torch.softmax(value, dim=0).tolist()
        ]
        group_hashes[group] = encoder_state_sha256(
            OrderedDict(alpha_logits=value)
        )
    snapshot = {
        "logits": logits,
        "weights": weights,
        "group_sha256": group_hashes,
    }
    snapshot["combined_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    return snapshot


def _alpha_snapshot_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the alpha snapshot directly from a checkpoint state dict."""
    torch = _torch()
    logits: OrderedDict[str, list[float]] = OrderedDict()
    weights: OrderedDict[str, list[float]] = OrderedDict()
    group_hashes: OrderedDict[str, str] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        key = f"alpha_logits.{group}"
        if key not in state:
            raise ValueError(f"P8_CHECKPOINT_ALPHA_STATE_MISSING:{group}")
        value = state[key].detach().cpu().contiguous()
        if tuple(value.shape) != (EXPERTS_PER_GROUP,) or not torch.isfinite(value).all():
            raise ValueError(f"P8_CHECKPOINT_ALPHA_STATE_INVALID:{group}")
        logits[group] = [float(item) for item in value.tolist()]
        weights[group] = [
            float(item) for item in torch.softmax(value, dim=0).tolist()
        ]
        group_hashes[group] = encoder_state_sha256(
            OrderedDict(alpha_logits=value)
        )
    snapshot = {
        "logits": logits,
        "weights": weights,
        "group_sha256": group_hashes,
    }
    snapshot["combined_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    return snapshot


def _validate_checkpoint_metadata(
    payload: Mapping[str, Any], history_row: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind checkpoint objective metadata and alpha state to its history row."""
    for payload_key, history_key in (
        ("validation_task_loss", "validation_task_loss"),
        ("validation_concept_loss", "validation_concept_loss"),
        ("validation_total_loss", "validation_total_loss"),
    ):
        if not serialized_float_consistent(
            float(payload[payload_key]), float(history_row[history_key])
        ):
            raise ValueError(f"P8_CHECKPOINT_METADATA_MISMATCH:{payload_key}")
    groups = payload.get("validation_group_losses", {})
    if set(groups) != set(CONCEPT_GROUP_ORDER):
        raise ValueError("P8_CHECKPOINT_GROUP_LOSS_SCHEMA_MISMATCH")
    for group in CONCEPT_GROUP_ORDER:
        if not serialized_float_consistent(
            float(groups[group]), float(history_row[f"validation_{group}_loss"])
        ):
            raise ValueError(f"P8_CHECKPOINT_GROUP_LOSS_MISMATCH:{group}")
    observed = payload.get("alpha_snapshot")
    reconstructed = _alpha_snapshot_from_state(payload["model_state_dict"])
    if observed != reconstructed:
        raise ValueError("P8_CHECKPOINT_ALPHA_SNAPSHOT_MISMATCH")
    for group in CONCEPT_GROUP_ORDER:
        history_logits = _json_vector(
            history_row[f"alpha_{group}_logits"],
            EXPERTS_PER_GROUP,
            "P8_CHECKPOINT_ALPHA_HISTORY_MISMATCH",
        )
        history_weights = _json_vector(
            history_row[f"alpha_{group}_weights"],
            EXPERTS_PER_GROUP,
            "P8_CHECKPOINT_ALPHA_HISTORY_MISMATCH",
        )
        if not np.allclose(
            history_logits,
            np.asarray(reconstructed["logits"][group]),
            atol=1e-12,
            rtol=0.0,
        ) or not np.allclose(
            history_weights,
            np.asarray(reconstructed["weights"][group]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"P8_CHECKPOINT_ALPHA_HISTORY_MISMATCH:{group}")
    return reconstructed


def _numeric_tolerance(*values: float, factor: float = NUMERIC_FLOAT32_OPERATION_FACTOR) -> float:
    magnitude = max((abs(float(value)) for value in values), default=1.0)
    return NUMERIC_ABSOLUTE_FLOOR + factor * np.finfo(np.float32).eps * max(
        magnitude, 1.0
    )


def _numeric_diagnostic(
    *,
    expected: float,
    actual: float,
    anonymous_row_index: int,
    group: str,
    field: str,
    factor: float = NUMERIC_FLOAT32_OPERATION_FACTOR,
) -> dict[str, Any]:
    allowed = _numeric_tolerance(expected, actual, factor=factor)
    report = {
        "schema": NUMERIC_SCHEMA,
        "anonymous_row_index": int(anonymous_row_index),
        "group": group,
        "field": field,
        "expected_value": float(expected),
        "actual_value": float(actual),
        "absolute_error": float(abs(expected - actual)),
        "allowed_absolute_error": float(allowed),
    }
    if not math.isfinite(expected) or not math.isfinite(actual) or abs(expected - actual) > allowed:
        raise ValueError(
            "P8_TEST_NUMERIC_RECONSTRUCTION_MISMATCH:"
            + canonical_json_bytes(report).decode("utf-8")
        )
    return report


def _ordered_records(
    records: Sequence[ConceptRecord],
    *,
    base_seed: int,
    fold_index: int,
    epoch_index: int,
) -> list[ConceptRecord]:
    by_uid = {record.nodule_uid: record for record in records}
    if len(by_uid) != len(records):
        raise ValueError("P8_DUPLICATE_RECORD_UID")
    return [
        by_uid[uid]
        for uid in epoch_uid_order(by_uid, base_seed, fold_index, epoch_index)
    ]


def run_gam_epoch(
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
    """Run one full partition with exact sample-weighted epoch aggregation."""
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
        raise ValueError("P8_EMPTY_PARTITION")
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
    alpha_gradient_l1 = OrderedDict((group, 0.0) for group in CONCEPT_GROUP_ORDER)
    sample_count = 0
    batch_count = 0
    observed_uids: list[str] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            image = batch["image"].to(device=device, dtype=torch.float32)
            concepts = _targets_to_device(batch["targets"], device)
            malignancy = batch["target_normalized"].to(
                device=device, dtype=torch.float32
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(image)
            losses = gam_losses(
                outputs, {"concepts": concepts, "malignancy": malignancy}
            )
            if not torch.isfinite(losses["total_loss"]):
                raise ValueError("P8_NONFINITE_TOTAL_LOSS")
            if training:
                losses["total_loss"].backward()
                for group in CONCEPT_GROUP_ORDER:
                    gradient = model.alpha_logits[group].grad
                    if gradient is None or not torch.isfinite(gradient).all():
                        raise ValueError(f"P8_ALPHA_GRADIENT_INVALID:{group}")
                    alpha_gradient_l1[group] += float(
                        gradient.detach().abs().sum().cpu()
                    )
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
                raise ValueError("P8_BATCH_SAMPLE_COUNT_MISMATCH")
            for group in CONCEPT_GROUP_ORDER:
                group_sums[group] += float(sums[group].detach().cpu())
            sample_count += batch_samples
            batch_count += 1
            observed_uids.extend(map(str, batch["nodule_uid"]))
    expected_uids = [record.nodule_uid for record in ordered]
    if observed_uids != expected_uids or len(set(observed_uids)) != len(ordered):
        raise ValueError("P8_PARTITION_SAMPLE_COVERAGE_MISMATCH")
    if sample_count != len(ordered):
        raise ValueError("P8_PARTITION_SAMPLE_COUNT_MISMATCH")
    group_losses = OrderedDict(
        (group, value / sample_count) for group, value in group_sums.items()
    )
    task_loss = task_squared_error_sum / sample_count
    concept_loss = float(np.mean(tuple(group_losses.values())))
    return {
        "task_loss": task_loss,
        "concept_loss": concept_loss,
        "total_loss": task_loss + concept_loss,
        "group_losses": group_losses,
        "sample_count": sample_count,
        "batch_count": batch_count,
        "nodule_set_sha256": sha256_bytes(
            canonical_json_bytes(sorted(observed_uids))
        ),
        "alpha_gradient_l1": alpha_gradient_l1,
    }


def _load_sources(
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
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
    p8_config, p8_hash = validate_p8_execution_config(p8_config_path)
    if p8_config["scientific_config"]["sha256"] != compute_config_sha256(
        scientific
    ):
        raise ValueError("P8_SCIENTIFIC_CONFIG_REFERENCE_MISMATCH")
    if p8_config["common_execution_profile"]["resolved_sha256"] != execution_hash:
        raise ValueError("P8_COMMON_EXECUTION_REFERENCE_MISMATCH")
    return (
        scientific,
        execution,
        execution_hash,
        p8_config,
        p8_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    )


def _provenance(
    scientific: Mapping[str, Any],
    execution: Mapping[str, Any],
    execution_hash: str,
    p8_hash: str,
    split: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": scientific["protocol"]["version"],
        "scientific_config_sha256": compute_config_sha256(scientific),
        "execution_config_sha256": execution_hash,
        "p8_execution_config_sha256": p8_hash,
        "execution_profile_id": execution["execution_profile"]["profile_id"],
        "formal_gpu_model": execution["execution_profile"]["formal_gpu_model"],
        **reproducibility_provenance(execution),
        "split_sha256": split["split_sha256"],
        "fold_index": int(split["fold_index"]),
        "model": MODEL_NAME,
        "task_output": "unconstrained_additive_raw_score",
        "task_loss": "mean_squared_error_on_normalized_target",
        "concept_loss": "equal_mean_of_six_mse_and_two_soft_cross_entropy_groups",
        "total_loss": "task_loss_plus_concept_loss",
        **dict(initialization),
    }


def run_directory(
    fold_index: int, root: str | Path = RUN_ROOT_DEFAULT
) -> Path:
    return Path(root) / f"fold_{fold_index}"


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: ValidationMSEPlateau,
    *,
    epoch_index: int,
    validation_total_loss: float,
    validation_report: Mapping[str, Any],
    best_epoch_index: int,
    best_validation_total_loss: float,
    provenance: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    alpha = _alpha_snapshot(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance),
        "epoch_index": int(epoch_index),
        "validation_total_loss": float(validation_total_loss),
        "validation_task_loss": float(validation_report["task_loss"]),
        "validation_concept_loss": float(validation_report["concept_loss"]),
        "validation_group_losses": {
            group: float(validation_report["group_losses"][group])
            for group in CONCEPT_GROUP_ORDER
        },
        "alpha_snapshot": alpha,
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
        raise ValueError("P8_CHECKPOINT_SCHEMA_MISMATCH")
    if payload.get("provenance") != dict(provenance):
        raise ValueError("P8_CHECKPOINT_PROVENANCE_MISMATCH")
    return payload


def _history_row(
    *,
    model: Any,
    epoch_index: int,
    train_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    learning_rate_start: float,
    optimizer: Any,
    scheduler: ValidationMSEPlateau,
    scheduler_decayed: bool,
    epoch_seconds: float,
) -> dict[str, Any]:
    torch = _torch()
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
    with torch.no_grad():
        for group in CONCEPT_GROUP_ORDER:
            row[f"train_{group}_loss"] = train_report["group_losses"][group]
            row[f"validation_{group}_loss"] = validation_report["group_losses"][
                group
            ]
            row[f"alpha_{group}_gradient_l1"] = train_report[
                "alpha_gradient_l1"
            ][group]
            row[f"alpha_{group}_logits"] = json.dumps(
                model.alpha_logits[group].detach().cpu().tolist(), separators=(",", ":")
            )
            row[f"alpha_{group}_weights"] = json.dumps(
                torch.softmax(model.alpha_logits[group], dim=0)
                .detach()
                .cpu()
                .tolist(),
                separators=(",", ":"),
            )
    return row


def _partition_uid_sha256(split: Mapping[str, Any], partition: str) -> str:
    uids = list(map(str, split["partitions"][partition]["nodule_uids"]))
    if len(uids) != len(set(uids)):
        raise ValueError(f"P8_SPLIT_DUPLICATE_UID:{partition}")
    return sha256_bytes(canonical_json_bytes(sorted(uids)))


def _validate_resume_history(
    path: Path, expected: Sequence[Mapping[str, Any]]
) -> None:
    frame = pd.read_csv(path)
    if len(frame) != len(expected) or frame["epoch_index"].tolist() != list(
        range(len(expected))
    ):
        raise ValueError("P8_RESUME_HISTORY_MISMATCH")
    if not expected or set(frame.columns) != set(expected[0]):
        raise ValueError("P8_RESUME_HISTORY_SCHEMA_MISMATCH")
    for row_index, expected_row in enumerate(expected):
        for key, expected_value in expected_row.items():
            observed = frame.iloc[row_index][key]
            if isinstance(expected_value, (bool, np.bool_)):
                if bool(observed) is not bool(expected_value):
                    raise ValueError("P8_RESUME_HISTORY_VALUE_MISMATCH")
            elif isinstance(expected_value, (int, float, np.integer, np.floating)):
                if not serialized_float_consistent(
                    float(observed), float(expected_value)
                ):
                    raise ValueError("P8_RESUME_HISTORY_VALUE_MISMATCH")
            elif str(observed) != str(expected_value):
                raise ValueError("P8_RESUME_HISTORY_VALUE_MISMATCH")


def _validate_completion_artifacts(
    output: Path,
    completion: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if any(completion.get(key) != value for key, value in provenance.items()):
        raise ValueError("P8_COMPLETION_PROVENANCE_MISMATCH")
    if int(completion.get("epochs_completed", -1)) != 80:
        raise ValueError("P8_COMPLETION_EPOCH_MISMATCH")
    if any(
        not isinstance(completion.get(key), str)
        or len(str(completion.get(key))) != 64
        for key in ("best_alpha_snapshot_sha256", "final_alpha_snapshot_sha256")
    ) or set(completion.get("final_alpha_group_sha256", {})) != set(
        CONCEPT_GROUP_ORDER
    ) or any(
        not isinstance(value, str) or len(value) != 64
        for value in completion.get("final_alpha_group_sha256", {}).values()
    ):
        raise ValueError("P8_COMPLETION_ALPHA_PROVENANCE_MISMATCH")
    for filename, key in (
        ("best.pt", "best_checkpoint_sha256"),
        ("last.pt", "last_checkpoint_sha256"),
        ("history.csv", "history_sha256"),
        ("runtime.json", "runtime_sha256"),
    ):
        path = output / filename
        if not path.is_file() or completion.get(key) != sha256_file(path):
            raise ValueError(f"P8_ARTIFACT_HASH_MISMATCH:{filename}")


def train_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
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
    with exclusive_fold_lifecycle_lock(output / ".p8_lifecycle.lock"):
        return _train_fold_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p8_config_path=p8_config_path,
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
    p8_config_path: Path,
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
        _p8_config,
        p8_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p8_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    provenance = _provenance(
        scientific, execution, execution_hash, p8_hash, split, initialization
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
            raise FileExistsError("P8_TRAINING_ALREADY_COMPLETE")
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
            raise FileNotFoundError("P8_RESUME_ARTIFACT_MISSING")
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
            raise ValueError("P8_RESUME_HISTORY_MISMATCH")
        _validate_resume_history(history_path, history)
    elif any(path.exists() for path in (last_path, best_path, history_path)):
        raise FileExistsError("P8_RUN_EXISTS_USE_RESUME_OR_INVALIDATE")
    epochs = 80
    batch_size = int(execution["project_preregistered"]["batching"]["micro_batch_size"])
    base_seed = int(scientific["reproducibility"]["base_seed"])
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_report = run_gam_epoch(
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
        validation_report = run_gam_epoch(
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
            model=model,
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
            validation_report=validation_report,
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
                {"event": "P8_EPOCH_COMPLETE", "fold_index": fold_index, **row}
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
        raise ValueError("P8_TRAINING_INCOMPLETE")
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
    best_payload = _load_checkpoint(best_path, provenance)
    last_alpha_snapshot = _alpha_snapshot(model)
    completion = {
        **provenance,
        "status": "TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "epochs_completed": epochs,
        "best_epoch_index": best_epoch,
        "best_validation_total_loss": best_validation,
        "best_checkpoint_sha256": sha256_file(best_path),
        "best_alpha_snapshot_sha256": best_payload["alpha_snapshot"][
            "combined_sha256"
        ],
        "last_checkpoint_sha256": sha256_file(last_path),
        "final_alpha_snapshot_sha256": last_alpha_snapshot["combined_sha256"],
        "final_alpha_group_sha256": last_alpha_snapshot["group_sha256"],
        "history_sha256": sha256_file(history_path),
        "runtime_sha256": sha256_file(output / "runtime.json"),
        "test_evaluated": False,
    }
    _atomic_json(complete_path, completion)
    return completion


def _json_vector(value: Any, size: int, code: str) -> np.ndarray:
    try:
        array = np.asarray(json.loads(str(value)), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(code) from error
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(code)
    return array


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
        ordered, training=False, base_seed=0, fold_index=0, epoch_index=0
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
            rating_bias = float(report["rating_scale_bias"].detach().cpu().reshape(-1)[0])
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
                    for suffix, value in (
                        ("logits", outputs["logits"][group][sample_index]),
                        (
                            "activated_prediction",
                            outputs["activated"][group][sample_index],
                        ),
                        (
                            "expert_outputs",
                            outputs["expert_outputs"][group][sample_index],
                        ),
                        ("alpha_logits", outputs["alpha_logits"][group]),
                        ("alpha_weights", outputs["alpha_weights"][group]),
                        ("target", concepts[group][sample_index]),
                    ):
                        row[f"{group}_{suffix}"] = json.dumps(
                            value.detach().cpu().reshape(-1).tolist(),
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
        raise ValueError("P8_TEST_PREDICTION_ORDER_MISMATCH")
    return result


def _validate_test_predictions(
    frame: pd.DataFrame,
    records: Sequence[ConceptRecord],
    row_provenance: Mapping[str, Any],
    model: Any,
) -> dict[str, Any]:
    torch = _torch()
    expected = {record.nodule_uid: record for record in records}
    if len(expected) != len(records):
        raise ValueError("P8_TEST_EXPECTED_UID_DUPLICATE")
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
        "normalized_reconstruction_max_abs_error",
        "rating_reconstruction_max_abs_error",
        "internalStructure_modal_tie",
        "calcification_modal_tie",
        *row_provenance,
    }
    for group in CONCEPT_GROUP_ORDER:
        required.update(
            {
                f"{group}_logits",
                f"{group}_activated_prediction",
                f"{group}_expert_outputs",
                f"{group}_alpha_logits",
                f"{group}_alpha_weights",
                f"{group}_target",
                f"{group}_valid_reader_count",
                f"{group}_raw_contribution",
                f"{group}_rating_contribution",
            }
        )
    if frame.columns.duplicated().any() or set(frame.columns) != required:
        raise ValueError("P8_TEST_PREDICTION_SCHEMA_MISMATCH")
    if len(frame) != len(expected) or frame["nodule_uid"].astype(str).duplicated().any():
        raise ValueError("P8_TEST_PREDICTION_COUNT_MISMATCH")
    if set(frame["nodule_uid"].astype(str)) != set(expected):
        raise ValueError("P8_TEST_PREDICTION_UID_SET_MISMATCH")
    for key, expected_value in row_provenance.items():
        if not all(value == expected_value for value in frame[key].tolist()):
            raise ValueError(f"P8_TEST_PROVENANCE_MISMATCH:{key}")
    by_uid = frame.set_index(frame["nodule_uid"].astype(str), drop=False)
    maximum_numeric_error = 0.0
    maximum_allowed_error = 0.0
    for anonymous_row_index, (uid, record) in enumerate(expected.items()):
        row = by_uid.loc[uid]
        if str(row["patient_key"]) != record.patient_key:
            raise ValueError("P8_TEST_PATIENT_KEY_MISMATCH")
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
            raise ValueError("P8_TEST_TARGET_MISMATCH")
        raw = float(row["malignancy_raw_score"])
        if not math.isfinite(raw) or float(row["malignancy_score_normalized"]) != raw:
            raise ValueError("P8_TEST_SCORE_MISMATCH")
        if not math.isclose(
            float(row["malignancy_score_1_to_5"]),
            1.0 + 4.0 * raw,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("P8_TEST_RATING_SCALE_MISMATCH")
        if _strict_bool(
            row["extreme_binary_eligible"], "P8_TEST_EXTREME_ELIGIBILITY_INVALID"
        ) is not record.extreme_binary_eligible:
            raise ValueError("P8_TEST_EXTREME_ELIGIBILITY_MISMATCH")
        observed_label = row["extreme_binary_label"]
        if record.extreme_binary_label is None:
            if not pd.isna(observed_label):
                raise ValueError("P8_TEST_EXTREME_LABEL_MISMATCH")
        elif (
            isinstance(observed_label, (bool, np.bool_))
            or not isinstance(observed_label, (int, float, np.integer, np.floating))
            or not math.isfinite(float(observed_label))
            or float(observed_label) != float(record.extreme_binary_label)
        ):
            raise ValueError("P8_TEST_EXTREME_LABEL_MISMATCH")
        raw_sum = float(row["raw_bias"])
        rating_sum = float(row["rating_scale_bias"])
        if not math.isclose(
            rating_sum, 1.0 + 4.0 * raw_sum, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("P8_TEST_BIAS_SCALE_MISMATCH")
        model_bias = float(model.global_raw_bias.detach().cpu().reshape(-1)[0])
        if not math.isclose(raw_sum, model_bias, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("P8_TEST_RAW_BIAS_MODEL_MISMATCH")
        continuous_targets = dict(
            zip(CONTINUOUS_CONCEPTS, record.continuous_targets, strict=True)
        )
        for group_index, group in enumerate(CONCEPT_GROUP_ORDER):
            size = CONCEPT_OUTPUT_SIZES[group]
            logits = _json_vector(row[f"{group}_logits"], size, "P8_TEST_LOGIT_INVALID")
            activated = _json_vector(
                row[f"{group}_activated_prediction"], size, "P8_TEST_ACTIVATION_INVALID"
            )
            target = _json_vector(row[f"{group}_target"], size, "P8_TEST_TARGET_INVALID")
            expert_outputs = _json_vector(
                row[f"{group}_expert_outputs"],
                EXPERTS_PER_GROUP,
                "P8_TEST_EXPERT_OUTPUT_INVALID",
            )
            alpha_logits = _json_vector(
                row[f"{group}_alpha_logits"], EXPERTS_PER_GROUP, "P8_TEST_ALPHA_INVALID"
            )
            alpha_weights = _json_vector(
                row[f"{group}_alpha_weights"],
                EXPERTS_PER_GROUP,
                "P8_TEST_ALPHA_INVALID",
            )
            expected_target = (
                np.asarray(record.internal_structure_target)
                if group == "internalStructure"
                else np.asarray(record.calcification_target)
                if group == "calcification"
                else np.asarray([continuous_targets[group]])
            )
            if not np.allclose(target, expected_target, atol=1e-7, rtol=0.0):
                raise ValueError("P8_TEST_CONCEPT_TARGET_MISMATCH")
            if _strict_positive_integer(
                row[f"{group}_valid_reader_count"],
                "P8_TEST_VALID_READER_COUNT_INVALID",
            ) != record.valid_reader_counts[group_index]:
                raise ValueError("P8_TEST_VALID_READER_COUNT_MISMATCH")
            shifted = logits - logits.max()
            expected_activation = (
                np.exp(shifted) / np.exp(shifted).sum()
                if group in CATEGORICAL_CONCEPTS
                else 1.0 / (1.0 + np.exp(-logits))
            )
            if not np.allclose(activated, expected_activation, atol=1e-6, rtol=0.0):
                raise ValueError("P8_TEST_LOGIT_ACTIVATION_MISMATCH")
            shifted_alpha = alpha_logits - alpha_logits.max()
            expected_alpha = np.exp(shifted_alpha) / np.exp(shifted_alpha).sum()
            if (
                np.any(alpha_weights < 0.0)
                or not np.isclose(alpha_weights.sum(), 1.0, atol=1e-6, rtol=0.0)
                or not np.allclose(alpha_weights, expected_alpha, atol=1e-6, rtol=0.0)
            ):
                raise ValueError("P8_TEST_ALPHA_SOFTMAX_MISMATCH")
            model_alpha = model.alpha_logits[group].detach().cpu().numpy()
            if not np.allclose(alpha_logits, model_alpha, atol=1e-7, rtol=0.0):
                raise ValueError("P8_TEST_ALPHA_MODEL_MISMATCH")
            with torch.no_grad():
                expert_parameter = next(model.experts[group][0].parameters())
                concept_input = torch.tensor(
                    activated,
                    dtype=expert_parameter.dtype,
                    device=expert_parameter.device,
                ).reshape(1, -1)
                expected_experts = torch.cat(
                    tuple(expert(concept_input) for expert in model.experts[group]), dim=1
                ).cpu().numpy().reshape(-1)
            for expert_index, (expected_expert, actual_expert) in enumerate(
                zip(expected_experts, expert_outputs, strict=True)
            ):
                diagnostic = _numeric_diagnostic(
                    expected=float(expected_expert),
                    actual=float(actual_expert),
                    anonymous_row_index=anonymous_row_index,
                    group=group,
                    field=f"expert_output_{expert_index}",
                    factor=128.0,
                )
                maximum_numeric_error = max(
                    maximum_numeric_error, float(diagnostic["absolute_error"])
                )
                maximum_allowed_error = max(
                    maximum_allowed_error,
                    float(diagnostic["allowed_absolute_error"]),
                )
            contribution = float(row[f"{group}_raw_contribution"])
            expected_contribution = float(
                np.sum(
                    alpha_weights.astype(np.float32)
                    * expert_outputs.astype(np.float32),
                    dtype=np.float32,
                )
            )
            diagnostic = _numeric_diagnostic(
                expected=expected_contribution,
                actual=contribution,
                anonymous_row_index=anonymous_row_index,
                group=group,
                field="alpha_weighted_group_contribution",
            )
            maximum_numeric_error = max(
                maximum_numeric_error, float(diagnostic["absolute_error"])
            )
            maximum_allowed_error = max(
                maximum_allowed_error, float(diagnostic["allowed_absolute_error"])
            )
            rating_contribution = float(row[f"{group}_rating_contribution"])
            if not math.isclose(
                rating_contribution, 4.0 * contribution, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError("P8_TEST_CONTRIBUTION_SCALE_MISMATCH")
            raw_sum += contribution
            rating_sum += rating_contribution
        internal_tie = _distribution_has_modal_tie(
            np.asarray(record.internal_structure_target)
        )
        calcification_tie = _distribution_has_modal_tie(
            np.asarray(record.calcification_target)
        )
        if record.categorical_ties != (internal_tie, calcification_tie):
            raise ValueError("P8_RECORD_TIE_FLAG_SEMANTIC_MISMATCH")
        if _strict_bool(
            row["internalStructure_modal_tie"], "P8_TEST_TIE_FLAG_INVALID"
        ) is not internal_tie or _strict_bool(
            row["calcification_modal_tie"], "P8_TEST_TIE_FLAG_INVALID"
        ) is not calcification_tie:
            raise ValueError("P8_TEST_TIE_FLAG_MISMATCH")
        raw_diagnostic = _numeric_diagnostic(
            expected=raw,
            actual=raw_sum,
            anonymous_row_index=anonymous_row_index,
            group="all",
            field="normalized_score_reconstruction",
        )
        rating_diagnostic = _numeric_diagnostic(
            expected=1.0 + 4.0 * raw,
            actual=rating_sum,
            anonymous_row_index=anonymous_row_index,
            group="all",
            field="rating_score_reconstruction",
        )
        for diagnostic in (raw_diagnostic, rating_diagnostic):
            maximum_numeric_error = max(
                maximum_numeric_error, float(diagnostic["absolute_error"])
            )
            maximum_allowed_error = max(
                maximum_allowed_error, float(diagnostic["allowed_absolute_error"])
            )
        if (
            not math.isfinite(float(row["normalized_reconstruction_max_abs_error"]))
            or float(row["normalized_reconstruction_max_abs_error"]) > 1e-6
            or not math.isfinite(float(row["rating_reconstruction_max_abs_error"]))
            or float(row["rating_reconstruction_max_abs_error"]) > 1e-6
        ):
            raise ValueError("P8_TEST_RECONSTRUCTION_DIAGNOSTIC_MISMATCH")
    return {
        "numeric_reconstruction_schema": NUMERIC_SCHEMA,
        "numeric_reconstruction_maximum_absolute_error": maximum_numeric_error,
        "numeric_reconstruction_maximum_allowed_absolute_error": maximum_allowed_error,
    }


def _trained_context(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
) -> dict[str, Any]:
    (
        scientific,
        execution,
        execution_hash,
        _p8_config,
        p8_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p8_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    provenance = _provenance(
        scientific, execution, execution_hash, p8_hash, split, initialization
    )
    output = run_directory(fold_index, output_root)
    completion_path = output / "training_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError("P8_TRAINING_COMPLETION_MISSING")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    _validate_completion_artifacts(output, completion, provenance)
    payload = _load_checkpoint(output / "best.pt", provenance)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return {
        "scientific": scientific,
        "execution": execution,
        "split": split,
        "manifest": manifest,
        "roi_index": roi_index,
        "model": model,
        "provenance": provenance,
        "completion": completion,
        "completion_path": completion_path,
        "output": output,
    }


def _validate_committed_test_artifacts(
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
    if (
        not claim_path.is_file()
        or json.loads(claim_path.read_text(encoding="utf-8")) != dict(claim)
    ):
        raise ValueError("P8_TEST_CLAIM_MISMATCH")
    if (
        evaluation.get("status") != "TEST_EVALUATED_ONCE"
        or int(evaluation.get("test_transaction_count", -1)) != 1
        or any(evaluation.get(key) != value for key, value in provenance.items())
        or evaluation.get("best_checkpoint_sha256")
        != completion.get("best_checkpoint_sha256")
        or int(evaluation.get("best_epoch_index", -1))
        != int(completion.get("best_epoch_index", -2))
        or not serialized_float_consistent(
            float(evaluation.get("best_validation_total_loss", math.nan)),
            float(completion.get("best_validation_total_loss", math.nan)),
        )
        or evaluation.get("test_claim_sha256") != sha256_file(claim_path)
        or not predictions_path.is_file()
        or evaluation.get("test_predictions_sha256") != sha256_file(predictions_path)
        or not metrics_path.is_file()
        or evaluation.get("metrics_sha256") != sha256_file(metrics_path)
    ):
        raise ValueError("P8_TEST_EVALUATION_ARTIFACT_MISMATCH")
    frame = pd.read_parquet(predictions_path)
    diagnostics = _validate_test_predictions(frame, records, row_provenance, model)
    if int(evaluation.get("test_samples", -1)) != len(frame):
        raise ValueError("P8_TEST_EVALUATION_SAMPLE_COUNT_MISMATCH")
    recomputed = regression_metrics(frame.to_dict("records"))
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    if recomputed != stored:
        raise ValueError("P8_TEST_METRICS_MISMATCH")
    if evaluation.get("numeric_reconstruction_schema") != diagnostics[
        "numeric_reconstruction_schema"
    ]:
        raise ValueError("P8_TEST_NUMERIC_SCHEMA_MISMATCH")
    for key in (
        "numeric_reconstruction_maximum_absolute_error",
        "numeric_reconstruction_maximum_allowed_absolute_error",
    ):
        if not serialized_float_consistent(
            float(evaluation.get(key, math.nan)), float(diagnostics[key])
        ):
            raise ValueError("P8_TEST_NUMERIC_DIAGNOSTIC_MISMATCH")
    return frame, stored


def evaluate_test_once(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    num_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p8_lifecycle.lock"):
        return _evaluate_test_once_locked(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p8_config_path=p8_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            device_name=device_name,
            num_workers=num_workers,
            output_root=output_root,
        )


def _evaluate_test_once_locked(**arguments: Any) -> dict[str, Any]:
    torch = _torch()
    context = _trained_context(
        **{key: value for key, value in arguments.items() if key != "device_name" and key != "num_workers"}
    )
    device = torch.device(arguments["device_name"])
    require_formal_gpu_for_cuda(device, context["execution"])
    configure_fp32_determinism(device, context["execution"])
    output = context["output"]
    completion = context["completion"]
    evaluation_path = output / "test_evaluation.json"
    claim_path = output / "test_claim.json"
    predictions_path = output / "test_predictions.parquet"
    metrics_path = output / "metrics.json"
    completion_claims_test_complete = (
        completion.get("status") == "TRAINING_COMPLETE_TEST_EVALUATED"
        or completion.get("test_evaluated") is True
        or int(completion.get("test_transaction_count", 0) or 0) == 1
    )
    if completion_claims_test_complete and not evaluation_path.is_file():
        raise ValueError("P8_COMPLETION_CLAIMS_MISSING_EVALUATION")
    records = build_partition_concept_records(
        context["manifest"],
        context["roi_index"],
        context["split"],
        "test",
        arguments["roi_index_path"],
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
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        frame, metrics = _validate_committed_test_artifacts(
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
            raise FileExistsError("P8_TEST_ALREADY_EVALUATED")
        completion["status"] = "TRAINING_COMPLETE_TEST_EVALUATED"
        completion["test_evaluated"] = True
        completion["test_evaluation_sha256"] = sha256_file(evaluation_path)
        completion["test_transaction_count"] = 1
        _atomic_json(context["completion_path"], completion)
        return {
            "evaluation": evaluation,
            "metrics": metrics,
            "recovered_without_inference": True,
            "test_samples": len(frame),
        }
    if claim_path.exists():
        if json.loads(claim_path.read_text(encoding="utf-8")) != claim:
            raise ValueError("P8_TEST_CLAIM_MISMATCH")
        if not predictions_path.exists():
            raise RuntimeError("P8_TEST_CLAIM_WITHOUT_PREDICTIONS_REQUIRES_AUDIT")
    else:
        _atomic_json(claim_path, claim)
    context["model"].to(device)
    if predictions_path.exists():
        frame = pd.read_parquet(predictions_path)
        numeric_diagnostics = _validate_test_predictions(
            frame, records, row_provenance, context["model"]
        )
        predictions = frame.to_dict("records")
    else:
        predictions = _prediction_rows(
            context["model"],
            records,
            device,
            batch_size=int(
                context["execution"]["project_preregistered"]["batching"][
                    "micro_batch_size"
                ]
            ),
            num_workers=int(arguments["num_workers"]),
        )
        for row in predictions:
            row.update(row_provenance)
        frame = pd.DataFrame(predictions)
        numeric_diagnostics = _validate_test_predictions(
            frame, records, row_provenance, context["model"]
        )
        _atomic_parquet(predictions_path, frame)
    metrics = regression_metrics(predictions)
    _atomic_json(metrics_path, metrics)
    evaluation = {
        **context["provenance"],
        "status": "TEST_EVALUATED_ONCE",
        "best_epoch_index": int(completion["best_epoch_index"]),
        "best_validation_total_loss": float(completion["best_validation_total_loss"]),
        "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
        "test_claim_sha256": sha256_file(claim_path),
        "test_predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "test_samples": len(frame),
        "test_transaction_count": 1,
        **numeric_diagnostics,
    }
    _atomic_json(evaluation_path, evaluation)
    completion["status"] = "TRAINING_COMPLETE_TEST_EVALUATED"
    completion["test_evaluated"] = True
    completion["test_evaluation_sha256"] = sha256_file(evaluation_path)
    completion["test_transaction_count"] = 1
    _atomic_json(context["completion_path"], completion)
    return {"evaluation": evaluation, "metrics": metrics}


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
    if len(history) != 80 or history["epoch_index"].tolist() != list(range(80)):
        raise ValueError("P8_HISTORY_EPOCH_MISMATCH")
    if not (history["train_sample_count"] == expected_train).all():
        raise ValueError("P8_HISTORY_TRAIN_COVERAGE_MISMATCH")
    if not (history["validation_sample_count"] == expected_validation).all():
        raise ValueError("P8_HISTORY_VALIDATION_COVERAGE_MISMATCH")
    if not (
        history["train_nodule_set_sha256"] == _partition_uid_sha256(split, "train")
    ).all():
        raise ValueError("P8_HISTORY_TRAIN_UID_SET_MISMATCH")
    if not (
        history["validation_nodule_set_sha256"]
        == _partition_uid_sha256(split, "validation")
    ).all():
        raise ValueError("P8_HISTORY_VALIDATION_UID_SET_MISMATCH")
    for group in CONCEPT_GROUP_ORDER:
        gradients = history[f"alpha_{group}_gradient_l1"].astype(float)
        if (
            not np.isfinite(gradients).all()
            or (gradients < 0.0).any()
            or float(gradients.sum()) <= 0.0
        ):
            raise ValueError(f"P8_ALPHA_GRADIENT_EVIDENCE_INVALID:{group}")
        final_logits: np.ndarray | None = None
        for row_index in range(len(history)):
            logits = _json_vector(
                history.iloc[row_index][f"alpha_{group}_logits"],
                EXPERTS_PER_GROUP,
                "P8_ALPHA_HISTORY_INVALID",
            )
            weights = _json_vector(
                history.iloc[row_index][f"alpha_{group}_weights"],
                EXPERTS_PER_GROUP,
                "P8_ALPHA_HISTORY_INVALID",
            )
            shifted = logits - logits.max()
            expected_weights = np.exp(shifted) / np.exp(shifted).sum()
            if (
                np.any(weights < 0.0)
                or np.any(weights > 1.0)
                or not np.isclose(weights.sum(), 1.0, atol=1e-7, rtol=0.0)
                or not np.allclose(
                    weights, expected_weights, atol=1e-7, rtol=0.0
                )
            ):
                raise ValueError(f"P8_ALPHA_HISTORY_SOFTMAX_MISMATCH:{group}")
            final_logits = logits
        if final_logits is None:
            raise ValueError(f"P8_ALPHA_HISTORY_INVALID:{group}")
        if np.allclose(final_logits, np.zeros(EXPERTS_PER_GROUP), atol=0.0, rtol=0.0):
            raise ValueError(f"P8_ALPHA_UPDATE_EVIDENCE_MISSING:{group}")
    if any(runtime.get(key) != value for key, value in provenance.items()):
        raise ValueError("P8_RUNTIME_PROVENANCE_MISMATCH")
    if runtime.get("device_type") != "cuda" or "H200" not in str(
        runtime.get("gpu_name", "")
    ).upper():
        raise ValueError("P8_RUNTIME_H200_MISMATCH")
    for key, expected in {
        "fp32": True,
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "torch_use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
    }.items():
        if runtime.get(key) is not expected:
            raise ValueError(f"P8_RUNTIME_PRECISION_POLICY_MISMATCH:{key}")
    if int(runtime.get("epochs_total", -1)) != 80:
        raise ValueError("P8_RUNTIME_EPOCH_MISMATCH")
    if not isinstance(runtime.get("peak_reserved_bytes"), int):
        raise ValueError("P8_RUNTIME_MEMORY_EVIDENCE_MISSING")
    return expected_train, expected_validation


def verify_fold(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_root: Path,
    require_test: bool = True,
) -> dict[str, Any]:
    output = run_directory(fold_index, output_root)
    with exclusive_fold_lifecycle_lock(output / ".p8_lifecycle.lock"):
        context = _trained_context(
            scientific_config_path=scientific_config_path,
            execution_config_path=execution_config_path,
            p8_config_path=p8_config_path,
            manifest_path=manifest_path,
            roi_index_path=roi_index_path,
            fold_index=fold_index,
            output_root=output_root,
        )
        history = pd.read_csv(output / "history.csv")
        runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
        expected_train, _ = _validate_history_and_runtime(
            history, runtime, context["split"], context["provenance"]
        )
        minimum_index = int(history["validation_total_loss"].idxmin())
        minimum_epoch = int(history.iloc[minimum_index]["epoch_index"])
        completion = context["completion"]
        if minimum_epoch != int(completion["best_epoch_index"]):
            raise ValueError("P8_CHECKPOINT_SELECTION_MISMATCH")
        if not serialized_float_consistent(
            float(completion["best_validation_total_loss"]),
            float(history.iloc[minimum_index]["validation_total_loss"]),
        ):
            raise ValueError("P8_BEST_OBJECTIVE_MISMATCH")
        best = _load_checkpoint(output / "best.pt", context["provenance"])
        last = _load_checkpoint(output / "last.pt", context["provenance"])
        if int(best["epoch_index"]) != minimum_epoch or int(last["epoch_index"]) != 79:
            raise ValueError("P8_CHECKPOINT_EPOCH_MISMATCH")
        if not serialized_float_consistent(
            float(best["validation_total_loss"]),
            float(history.iloc[minimum_index]["validation_total_loss"]),
        ):
            raise ValueError("P8_BEST_PAYLOAD_OBJECTIVE_MISMATCH")
        best_alpha = _validate_checkpoint_metadata(
            best, history.iloc[minimum_index].to_dict()
        )
        last_alpha = _validate_checkpoint_metadata(last, history.iloc[-1].to_dict())
        if (
            completion.get("best_alpha_snapshot_sha256")
            != best_alpha["combined_sha256"]
            or completion.get("final_alpha_snapshot_sha256")
            != last_alpha["combined_sha256"]
            or completion.get("final_alpha_group_sha256")
            != last_alpha["group_sha256"]
        ):
            raise ValueError("P8_COMPLETION_ALPHA_PROVENANCE_MISMATCH")
        for filename, key in (
            ("best.pt", "best_checkpoint_sha256"),
            ("last.pt", "last_checkpoint_sha256"),
            ("history.csv", "history_sha256"),
            ("runtime.json", "runtime_sha256"),
        ):
            if completion.get(key) != sha256_file(output / filename):
                raise ValueError(f"P8_ARTIFACT_HASH_MISMATCH:{filename}")
        report: dict[str, Any] = {
            **context["provenance"],
            "status": "PASS",
            "epochs": 80,
            "train_samples_per_epoch": expected_train,
            "best_epoch_index": minimum_epoch,
            "best_validation_total_loss": float(
                history.iloc[minimum_index]["validation_total_loss"]
            ),
            "alpha_gradient_and_update_gate": "PASS",
            "best_alpha_snapshot_sha256": best_alpha["combined_sha256"],
            "final_alpha_snapshot_sha256": last_alpha["combined_sha256"],
            "final_alpha_group_sha256": last_alpha["group_sha256"],
        }
        if require_test:
            if (
                completion.get("status") != "TRAINING_COMPLETE_TEST_EVALUATED"
                or completion.get("test_evaluated") is not True
                or int(completion.get("test_transaction_count", -1)) != 1
            ):
                raise ValueError("P8_TEST_NOT_EVALUATED_EXACTLY_ONCE")
            evaluation_path = output / "test_evaluation.json"
            if completion.get("test_evaluation_sha256") != sha256_file(evaluation_path):
                raise ValueError("P8_TEST_EVALUATION_HASH_MISMATCH")
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
            expected_claim = {
                **context["provenance"],
                "status": "TEST_EVALUATION_CLAIMED",
                "best_checkpoint_sha256": completion["best_checkpoint_sha256"],
                "best_epoch_index": int(completion["best_epoch_index"]),
                "expected_test_samples": len(records),
            }
            frame, _stored = _validate_committed_test_artifacts(
                output=output,
                evaluation=evaluation,
                claim=expected_claim,
                completion=completion,
                provenance=context["provenance"],
                records=records,
                row_provenance=row_provenance,
                model=context["model"],
            )
            if len(frame) != EXPECTED_FOLD_TEST_COUNTS[fold_index]:
                raise ValueError("P8_VERIFY_TEST_COUNT_MISMATCH")
            report["test_samples"] = len(frame)
            report["test_evaluated_once"] = True
            report["maximum_normalized_reconstruction_error"] = float(
                frame["normalized_reconstruction_max_abs_error"].max()
            )
            report["maximum_rating_reconstruction_error"] = float(
                frame["rating_reconstruction_max_abs_error"].max()
            )
        return report


def verify_all(**arguments: Any) -> dict[str, Any]:
    reports = [verify_fold(**arguments, fold_index=fold) for fold in range(5)]
    frames = [
        pd.read_parquet(
            run_directory(fold, arguments["output_root"]) / "test_predictions.parquet"
        )
        for fold in range(5)
    ]
    pooled = pd.concat(frames, ignore_index=True)
    if len(pooled) != 2633 or pooled["nodule_uid"].nunique() != 2633:
        raise ValueError("P8_OOF_NODULE_SET_MISMATCH")
    if pooled["patient_key"].nunique() != 868:
        raise ValueError("P8_OOF_PATIENT_SET_MISMATCH")
    if pooled.groupby("patient_key")["fold_index"].nunique().max() != 1:
        raise ValueError("P8_OOF_PATIENT_LEAKAGE")
    return {
        "status": "PASS",
        "oof_nodules": 2633,
        "oof_patients": 868,
        "fold_test_counts": [len(frame) for frame in frames],
        "folds": reports,
    }


def _stage_a_structure_report(model: Any) -> dict[str, Any]:
    """Verify that all 40 experts are independent and concept-local."""
    parameter_ids: set[int] = set()
    groups: OrderedDict[str, Any] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        experts = model.experts[group]
        expected_input = int(CONCEPT_OUTPUT_SIZES[group])
        if len(experts) != EXPERTS_PER_GROUP:
            raise ValueError(f"P8_STAGE_A_EXPERT_COUNT_MISMATCH:{group}")
        for expert in experts:
            if (
                int(expert[0].in_features) != expected_input
                or int(expert[0].out_features) != 32
                or int(expert[2].in_features) != 32
                or int(expert[2].out_features) != 16
                or int(expert[4].in_features) != 16
                or int(expert[4].out_features) != 1
            ):
                raise ValueError(f"P8_STAGE_A_EXPERT_ARCHITECTURE_MISMATCH:{group}")
            for parameter in expert.parameters():
                identity = id(parameter)
                if identity in parameter_ids:
                    raise ValueError("P8_STAGE_A_EXPERT_PARAMETER_SHARING")
                parameter_ids.add(identity)
        groups[group] = {
            "experts": EXPERTS_PER_GROUP,
            "input_dimensions": [expected_input] * EXPERTS_PER_GROUP,
            "architecture": [expected_input, 32, 16, 1],
            "concept_local_input_only": True,
        }
    if len(parameter_ids) != len(CONCEPT_GROUP_ORDER) * EXPERTS_PER_GROUP * 6:
        raise ValueError("P8_STAGE_A_EXPERT_PARAMETER_COUNT_MISMATCH")
    return {
        "status": "PASS",
        "groups": groups,
        "independent_experts": len(CONCEPT_GROUP_ORDER) * EXPERTS_PER_GROUP,
        "shared_expert_parameters": 0,
    }


def overfit_check(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    device_name: str,
    samples: int,
    steps: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run the non-formal eight-sample Stage A overfit sanity check."""
    if samples != 8 or steps != 40:
        raise ValueError("P8_STAGE_A_OVERFIT_SCOPE_MISMATCH")
    torch = _torch()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    (
        scientific,
        execution,
        execution_hash,
        _p8_config,
        p8_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p8_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    structure = _stage_a_structure_report(model)
    seed_training(int(initialization["fold_seed"]))
    model.to(device)
    optimizer = _optimizer(model, execution)
    records = sorted(
        build_partition_concept_records(
            manifest, roi_index, split, "train", roi_index_path
        ),
        key=lambda record: record.nodule_uid,
    )[:samples]
    if len(records) != samples:
        raise ValueError("P8_STAGE_A_OVERFIT_SAMPLE_COUNT_MISMATCH")
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
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(image)
        total = gam_losses(
            outputs, {"concepts": concepts, "malignancy": malignancy}
        )["total_loss"]
        if not torch.isfinite(total):
            raise ValueError("P8_STAGE_A_OVERFIT_NONFINITE_LOSS")
        total.backward()
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    initial = float(np.mean(losses[:5]))
    final = float(np.mean(losses[-5:]))
    if not math.isfinite(final) or final >= initial:
        raise RuntimeError("P8_OVERFIT_SANITY_DID_NOT_IMPROVE")
    report = {
        **_provenance(
            scientific, execution, execution_hash, p8_hash, split, initialization
        ),
        **_runtime_environment(device),
        "status": "PASS",
        "scope": "train_only_controlled_overfit_sanity",
        "formal_run": False,
        "test_inference": False,
        "augmentation_enabled": False,
        "samples": samples,
        "steps": steps,
        "initial_five_step_mean_total_loss": initial,
        "final_five_step_mean_total_loss": final,
        "relative_final_loss": final / initial,
        "expert_structure": structure,
    }
    _atomic_json(output_path, report)
    return report


def preflight(
    *,
    scientific_config_path: Path,
    execution_config_path: Path,
    p8_config_path: Path,
    manifest_path: Path,
    roi_index_path: Path,
    fold_index: int,
    output_path: Path,
) -> dict[str, Any]:
    """Run one true-batch-16 H200 forward/backward/Adam Stage A step."""
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    (
        scientific,
        execution,
        execution_hash,
        _p8_config,
        p8_hash,
        split,
        manifest,
        roi_index,
        encoder_path,
    ) = _load_sources(
        scientific_config_path,
        execution_config_path,
        p8_config_path,
        manifest_path,
        roi_index_path,
        fold_index,
    )
    require_formal_gpu_for_cuda(device, execution)
    configure_fp32_determinism(device, execution)
    model, initialization = build_initialized_model(scientific, split, encoder_path)
    structure = _stage_a_structure_report(model)
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
    if len(ordered) != 16:
        raise ValueError("P8_STAGE_A_PREFLIGHT_SAMPLE_COUNT_MISMATCH")
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
    if int(image.shape[0]) != 16:
        raise ValueError("P8_STAGE_A_PREFLIGHT_TRUE_BATCH_MISMATCH")
    before_logits = {
        group: model.alpha_logits[group].detach().clone()
        for group in CONCEPT_GROUP_ORDER
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(image)
    reconstruction = task_predictions_and_contributions(model, outputs)
    losses = gam_losses(outputs, {"concepts": concepts, "malignancy": malignancy})
    losses["total_loss"].backward()
    gradient_l1: OrderedDict[str, float] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        gradient = model.alpha_logits[group].grad
        if (
            gradient is None
            or not torch.isfinite(gradient).all()
            or int(torch.count_nonzero(gradient).detach().cpu()) == 0
        ):
            raise ValueError(f"P8_STAGE_A_ALPHA_GRADIENT_INVALID:{group}")
        gradient_l1[group] = float(gradient.detach().abs().sum().cpu())
    optimizer.step()
    torch.cuda.synchronize(device)
    alpha_updated: OrderedDict[str, bool] = OrderedDict(
        (
            group,
            not torch.equal(
                before_logits[group], model.alpha_logits[group].detach()
            ),
        )
        for group in CONCEPT_GROUP_ORDER
    )
    if not all(alpha_updated.values()):
        raise ValueError("P8_STAGE_A_ALPHA_UPDATE_MISSING")
    with torch.no_grad():
        alpha_valid = all(
            bool((outputs["alpha_weights"][group] >= 0).all())
            and torch.allclose(
                outputs["alpha_weights"][group].sum(),
                torch.tensor(1.0, device=device),
                atol=1e-7,
                rtol=0.0,
            )
            for group in CONCEPT_GROUP_ORDER
        )
    if not alpha_valid:
        raise ValueError("P8_STAGE_A_ALPHA_SIMPLEX_INVALID")
    reserved = int(torch.cuda.max_memory_reserved(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    fraction = reserved / total_memory
    if fraction > 0.85:
        raise RuntimeError(f"P8_PREFLIGHT_MEMORY_LIMIT_EXCEEDED:{fraction}")
    report = {
        **_provenance(
            scientific, execution, execution_hash, p8_hash, split, initialization
        ),
        **_runtime_environment(device),
        "status": "PASS",
        "formal_run": False,
        "test_inference": False,
        "batch_size": int(image.shape[0]),
        "forward": True,
        "task_concept_and_total_losses": True,
        "backward": True,
        "adam_step": True,
        "expert_structure": structure,
        "alpha_simplex_gate": "PASS",
        "alpha_gradient_l1": gradient_l1,
        "alpha_updated": alpha_updated,
        "task_loss": float(losses["task_loss"].detach().cpu()),
        "concept_loss": float(losses["concept_loss"].detach().cpu()),
        "total_loss": float(losses["total_loss"].detach().cpu()),
        "normalized_reconstruction_max_abs_error": reconstruction[
            "normalized_reconstruction_max_abs_error"
        ],
        "rating_reconstruction_max_abs_error": reconstruction[
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
    parser.add_argument("--p8-config", type=Path, default=P8_EXECUTION_CONFIG_DEFAULT)
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
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT_DEFAULT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    overfit = subparsers.add_parser("overfit-check")
    _common_arguments(overfit)
    overfit.add_argument("--fold", type=int, required=True, choices=range(5))
    overfit.add_argument("--device", default="cuda")
    overfit.add_argument("--samples", type=int, default=8)
    overfit.add_argument("--steps", type=int, default=40)
    overfit.add_argument(
        "--output",
        type=Path,
        default=Path("runs/baseline_v2/gam/fold_0/stage_a/overfit_sanity.json"),
    )
    preflight_parser = subparsers.add_parser("preflight")
    _common_arguments(preflight_parser)
    preflight_parser.add_argument(
        "--fold", type=int, required=True, choices=range(5)
    )
    preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/baseline_v2/gam/fold_0/stage_a/preflight.json"),
    )
    train = subparsers.add_parser("train")
    _common_arguments(train)
    train.add_argument("--fold", type=int, required=True, choices=range(5))
    train.add_argument("--device", default="cuda")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--resume", action="store_true")
    evaluate = subparsers.add_parser("evaluate-test")
    _common_arguments(evaluate)
    evaluate.add_argument("--fold", type=int, required=True, choices=range(5))
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--num-workers", type=int, default=4)
    verify = subparsers.add_parser("verify")
    _common_arguments(verify)
    verify.add_argument("--fold", type=int, choices=range(5))
    verify.add_argument("--scope", choices=("all",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    source_common = {
        "scientific_config_path": arguments.config,
        "execution_config_path": arguments.execution_config,
        "p8_config_path": arguments.p8_config,
        "manifest_path": arguments.manifest,
        "roi_index_path": arguments.roi_index,
    }
    lifecycle_common = {
        **source_common,
        "output_root": arguments.output_root,
    }
    if arguments.command == "overfit-check":
        report = overfit_check(
            **source_common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            samples=arguments.samples,
            steps=arguments.steps,
            output_path=arguments.output,
        )
    elif arguments.command == "preflight":
        report = preflight(
            **source_common,
            fold_index=arguments.fold,
            output_path=arguments.output,
        )
    elif arguments.command == "train":
        report = train_fold(
            **lifecycle_common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
            resume=arguments.resume,
        )
    elif arguments.command == "evaluate-test":
        report = evaluate_test_once(
            **lifecycle_common,
            fold_index=arguments.fold,
            device_name=arguments.device,
            num_workers=arguments.num_workers,
        )
    elif arguments.command == "verify":
        if (arguments.fold is None) == (arguments.scope is None):
            raise ValueError("P8_VERIFY_REQUIRES_EXACTLY_ONE_FOLD_OR_SCOPE")
        report = (
            verify_all(**lifecycle_common)
            if arguments.scope == "all"
            else verify_fold(**lifecycle_common, fold_index=arguments.fold)
        )
    else:
        raise AssertionError(arguments.command)
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0
