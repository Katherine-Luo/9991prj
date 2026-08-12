"""Build private P9 aggregate results and deidentified tracked evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from lidc_baseline.audit import write_json
from lidc_baseline.p3_roi import assert_deidentified_audit
from lidc_baseline.p4_prepare import sha256_file
from lidc_baseline.p6_standard_cbm import (
    CATEGORICAL_CONCEPTS,
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    CONTINUOUS_CONCEPTS,
)
from lidc_baseline.p9_evaluation import (
    BOOTSTRAP_DRAWS,
    MODEL_ORDER,
    MODEL_PAIRS,
    OOF_FILENAMES,
    P4_SPLIT_ROOT_DEFAULT,
    P9_EXECUTION_CONFIG_DEFAULT,
    VALIDATION_FILENAMES,
    bootstrap_draw_sha256,
    build_task_results,
    canonical_oof_frame,
    categorical_concept_metrics,
    continuous_concept_metrics,
    error_first_order,
    extreme_labels,
    intervention_deltas,
    patient_cluster_bootstrap_draws,
    secondary_patient_bootstrap_draws,
    shared_intervention_permutations,
    validate_p9_execution_config,
    verify_inputs,
)
from lidc_baseline.p9_spatial import aggregate_faithfulness_records, target_specs
from lidc_baseline.p9_spatial_lifecycle import (
    MANIFEST_DEFAULT,
    P9_ROOT_DEFAULT,
    RUN_ROOTS_DEFAULT,
    SPATIAL_APPROVAL_DEFAULT,
    load_frozen_model_bundle,
    read_and_verify_map_shard,
    verify_all,
)


SCHEMA_VERSION = 1
AUDIT_ROOT_DEFAULT = Path("artifacts/baseline_v2/audit/p9")
PRIVATE_OOF_ROOT_NAME = "canonical_oof"
PRIVATE_VALIDATION_ROOT_NAME = "validation_auxiliary"
BASE_SEED = 20260808


def _json_vector(value: Any, size: int, code: str) -> np.ndarray:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        result = np.asarray(parsed, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(code) from error
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(code)
    return result


def _json_object(value: Any, code: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(code) from error
    if not isinstance(parsed, dict):
        raise ValueError(code)
    return parsed


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def materialize_canonical_inputs(p9_root: Path) -> dict[str, Any]:
    """Copy frozen OOF and validated spatial validation auxiliaries privately."""
    oof_root = p9_root / PRIVATE_OOF_ROOT_NAME
    validation_root = p9_root / PRIVATE_VALIDATION_ROOT_NAME
    oof_hashes: dict[str, str] = {}
    validation_hashes: dict[str, str] = {}
    for model in MODEL_ORDER:
        source = RUN_ROOTS_DEFAULT[model] / "oof_predictions.parquet"
        destination = oof_root / OOF_FILENAMES[model]
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ValueError(f"P9_CANONICAL_OOF_REUSE_MISMATCH:{model}")
        else:
            _atomic_copy(source, destination)
        oof_hashes[model] = sha256_file(destination)
        frames = []
        for fold in range(5):
            auxiliary = (
                p9_root
                / "spatial"
                / model
                / f"fold_{fold}"
                / "validation_predictions.parquet"
            )
            frame = pd.read_parquet(auxiliary)
            if "fold_index" not in frame:
                frame["fold_index"] = fold
            elif not np.all(frame["fold_index"].astype(int) == fold):
                raise ValueError(f"P9_VALIDATION_AUXILIARY_FOLD_MISMATCH:{model}:{fold}")
            frames.append(frame)
        pooled = pd.concat(frames, ignore_index=True)
        destination = validation_root / VALIDATION_FILENAMES[model]
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        pooled.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(destination)
        validation_hashes[model] = sha256_file(destination)
    return {
        "oof_root": oof_root,
        "validation_root": validation_root,
        "oof_sha256": oof_hashes,
        "validation_sha256": validation_hashes,
    }


def _concept_arrays(
    frame: pd.DataFrame, model: str, group: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    size = int(CONCEPT_OUTPUT_SIZES[group])
    prediction = np.stack(
        [
            _json_vector(value, size, f"P9_CONCEPT_PREDICTION_INVALID:{model}:{group}")
            for value in frame[f"{group}_activated_prediction"]
        ]
    )
    if model == "standard_cbm":
        targets = [
            _json_object(value, "P9_STANDARD_CBM_TARGET_OBJECT_INVALID")[group]
            for value in frame["concept_targets"]
        ]
    else:
        targets = frame[f"{group}_target"].tolist()
    target = np.stack(
        [
            _json_vector(value, size, f"P9_CONCEPT_TARGET_INVALID:{model}:{group}")
            for value in targets
        ]
    )
    ties = None
    if group in CATEGORICAL_CONCEPTS:
        column = f"{group}_modal_tie"
        values = frame[column].to_numpy()
        if values.dtype != np.bool_:
            if not all(type(value) is bool for value in values):
                raise ValueError(f"P9_CONCEPT_TIE_TYPE_INVALID:{model}:{group}")
            values = values.astype(bool)
        ties = values
    return prediction, target, ties


def concept_results(oof_frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in MODEL_ORDER[1:]:
        frame = oof_frames[model]
        scopes: dict[str, Any] = {"folds": []}
        for fold in range(5):
            subset = frame[frame["fold_index"].astype(int) == fold]
            scopes["folds"].append(
                {
                    "fold_index": fold,
                    "concepts": _concept_scope(subset, model),
                }
            )
        scopes["pooled"] = _concept_scope(frame, model)
        output[model] = scopes
    return output


def _concept_scope(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in CONCEPT_GROUP_ORDER:
        prediction, target, ties = _concept_arrays(frame, model, group)
        if group in CONTINUOUS_CONCEPTS:
            result[group] = continuous_concept_metrics(
                prediction.reshape(-1), target.reshape(-1)
            )
        else:
            result[group] = categorical_concept_metrics(prediction, target, ties)
    return result


def _percentile_interval(values: Sequence[float]) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (BOOTSTRAP_DRAWS,) or not np.isfinite(vector).all():
        raise ValueError("P9_BOOTSTRAP_RESULT_INVALID")
    return {
        "estimate_mean": float(vector.mean()),
        "percentile_2_5": float(np.percentile(vector, 2.5)),
        "percentile_97_5": float(np.percentile(vector, 97.5)),
    }


def _draw_indices(frame: pd.DataFrame, draw: Sequence[str], *, extreme: bool) -> np.ndarray:
    by_patient: dict[str, np.ndarray] = {}
    for patient, rows in frame.groupby("patient_key", sort=False):
        indices = rows.index.to_numpy(dtype=np.int64)
        if extreme:
            eligible, _ = extreme_labels(rows["target_1_to_5"])
            indices = indices[eligible]
        by_patient[str(patient)] = indices
    return np.concatenate([by_patient[str(patient)] for patient in draw])


def bootstrap_results(
    oof_frames: Mapping[str, pd.DataFrame], *, base_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    canonical = {
        model: canonical_oof_frame(frame, model)
        .sort_values("nodule_uid")
        .reset_index(drop=True)
        for model, frame in oof_frames.items()
    }
    reference = canonical[MODEL_ORDER[0]]
    primary_draws = patient_cluster_bootstrap_draws(
        reference["patient_key"], base_seed
    )
    eligible, labels = extreme_labels(reference["target_1_to_5"])
    patient_labels: dict[str, list[int]] = defaultdict(list)
    for patient, label, is_eligible in zip(
        reference["patient_key"], labels, eligible, strict=True
    ):
        if is_eligible:
            patient_labels[str(patient)].append(int(label))
    secondary_draws = secondary_patient_bootstrap_draws(patient_labels, base_seed)
    primary_values = {
        model: {
            "mae": [],
            "rmse": [],
            "normalized_mae": [],
            "pearson": [],
            "spearman": [],
        }
        for model in MODEL_ORDER
    }
    secondary_values = {
        model: {"auroc": [], "auprc": []} for model in MODEL_ORDER
    }
    paired_mae = {f"{a}__{b}": [] for a, b in MODEL_PAIRS}
    paired_auroc = {f"{a}__{b}": [] for a, b in MODEL_PAIRS}
    for primary_draw, secondary_draw in zip(
        primary_draws, secondary_draws, strict=True
    ):
        primary_index = _draw_indices(reference, primary_draw, extreme=False)
        secondary_index = _draw_indices(reference, secondary_draw, extreme=True)
        draw_mae: dict[str, float] = {}
        draw_auc: dict[str, float] = {}
        for model in MODEL_ORDER:
            frame = canonical[model]
            error = 4.0 * (
                frame.loc[primary_index, "malignancy_raw_score"].to_numpy(dtype=float)
                - frame.loc[primary_index, "target_normalized"].to_numpy(dtype=float)
            )
            mae = float(np.mean(np.abs(error)))
            rmse = float(np.sqrt(np.mean(np.square(error))))
            normalized_mae = float(np.mean(np.abs(error / 4.0)))
            draw_target = frame.loc[
                primary_index, "target_normalized"
            ].to_numpy(dtype=float)
            draw_score = frame.loc[
                primary_index, "malignancy_raw_score"
            ].to_numpy(dtype=float)
            pearson = float(stats.pearsonr(draw_target, draw_score).statistic)
            spearman = float(stats.spearmanr(draw_target, draw_score).statistic)
            primary_values[model]["mae"].append(mae)
            primary_values[model]["rmse"].append(rmse)
            primary_values[model]["normalized_mae"].append(normalized_mae)
            primary_values[model]["pearson"].append(pearson)
            primary_values[model]["spearman"].append(spearman)
            draw_mae[model] = mae
            secondary_frame = frame.loc[secondary_index]
            _, draw_labels = extreme_labels(secondary_frame["target_1_to_5"])
            score = secondary_frame["malignancy_raw_score"].to_numpy(dtype=float)
            auc = float(metrics.roc_auc_score(draw_labels, score))
            auprc = float(metrics.average_precision_score(draw_labels, score))
            secondary_values[model]["auroc"].append(auc)
            secondary_values[model]["auprc"].append(auprc)
            draw_auc[model] = auc
        for first, second in MODEL_PAIRS:
            paired_mae[f"{first}__{second}"].append(
                draw_mae[first] - draw_mae[second]
            )
            paired_auroc[f"{first}__{second}"].append(
                draw_auc[second] - draw_auc[first]
            )
    tracked = {
        "draws": BOOTSTRAP_DRAWS,
        "primary_draw_sha256": bootstrap_draw_sha256(primary_draws),
        "secondary_draw_sha256": bootstrap_draw_sha256(secondary_draws),
        "models": {
            model: {
                "original_scale_mae": _percentile_interval(
                    primary_values[model]["mae"]
                ),
                "original_scale_rmse": _percentile_interval(
                    primary_values[model]["rmse"]
                ),
                "normalized_mae": _percentile_interval(
                    primary_values[model]["normalized_mae"]
                ),
                "pearson": _percentile_interval(
                    primary_values[model]["pearson"]
                ),
                "spearman": _percentile_interval(
                    primary_values[model]["spearman"]
                ),
                "auroc": _percentile_interval(
                    secondary_values[model]["auroc"]
                ),
                "auprc": _percentile_interval(
                    secondary_values[model]["auprc"]
                ),
            }
            for model in MODEL_ORDER
        },
        "paired_mae_A_minus_B": {
            key: _percentile_interval(value) for key, value in paired_mae.items()
        },
        "paired_auroc_B_minus_A": {
            key: _percentile_interval(value) for key, value in paired_auroc.items()
        },
    }
    private = {
        "primary_draws": primary_draws,
        "secondary_draws": secondary_draws,
    }
    return tracked, private


def _all_concept_values(frame: pd.DataFrame, model: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prediction: dict[str, np.ndarray] = {}
    target: dict[str, np.ndarray] = {}
    for group in CONCEPT_GROUP_ORDER:
        predicted, expected, _ = _concept_arrays(frame, model, group)
        prediction[group] = predicted
        target[group] = expected
    return prediction, target


def _intervention_group_contributions(
    frame: pd.DataFrame, model: str, fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return persisted predicted and exact GT-intervened group contributions."""
    predicted = np.stack(
        [
            frame[f"{group}_raw_contribution"].to_numpy(dtype=np.float64)
            for group in CONCEPT_GROUP_ORDER
        ],
        axis=1,
    )
    concept_prediction, concept_target = _all_concept_values(frame, model)
    bundle = load_frozen_model_bundle(model, fold, require_test=True)
    target_contribution = np.empty_like(predicted)
    if model == "standard_cbm":
        weight = bundle.task_head.weight.detach().cpu().numpy().reshape(-1)
        offset = 0
        for index, group in enumerate(CONCEPT_GROUP_ORDER):
            width = int(CONCEPT_OUTPUT_SIZES[group])
            target_contribution[:, index] = concept_target[group] @ weight[
                offset : offset + width
            ]
            offset += width
    elif model == "mixed_cem":
        weight = bundle.model.task_head.weight.detach().cpu().numpy().reshape(8, 16)
        for index, group in enumerate(CONCEPT_GROUP_ORDER):
            classes = 2 if group in CONTINUOUS_CONCEPTS else int(
                CONCEPT_OUTPUT_SIZES[group]
            )
            states = np.stack(
                [
                    _json_vector(
                        value,
                        classes * 16,
                        f"P9_CEM_STATE_INVALID:{group}",
                    ).reshape(classes, 16)
                    for value in frame[f"{group}_states"]
                ]
            )
            if group in CONTINUOUS_CONCEPTS:
                probability = concept_target[group].reshape(-1, 1)
                mixed = (1.0 - probability) * states[:, 0] + probability * states[:, 1]
            else:
                mixed = np.sum(concept_target[group][..., None] * states, axis=1)
            target_contribution[:, index] = mixed @ weight[index]
    elif model == "learned_softmax_gam":
        import torch

        bundle.model.eval()
        with torch.no_grad():
            for index, group in enumerate(CONCEPT_GROUP_ORDER):
                values = torch.from_numpy(concept_target[group].astype(np.float32))
                expert = torch.cat(
                    [network(values) for network in bundle.model.experts[group]], dim=1
                )
                alpha = torch.softmax(bundle.model.alpha_logits[group], dim=0)
                target_contribution[:, index] = (
                    (expert * alpha.reshape(1, -1))
                    .sum(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
    else:  # pragma: no cover
        raise ValueError(f"P9_INTERVENTION_MODEL_INVALID:{model}")
    if not np.isfinite(target_contribution).all():
        raise ValueError(f"P9_INTERVENTION_NONFINITE:{model}:{fold}")
    return predicted, target_contribution


def _curve_metrics(score: np.ndarray, target: np.ndarray, target_rating: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(4.0 * (score - target))))
    eligible, labels = extreme_labels(target_rating)
    auc = float(metrics.roc_auc_score(labels[eligible], score[eligible]))
    return mae, auc


def intervention_results(
    oof_frames: Mapping[str, pd.DataFrame], *, base_seed: int,
    private_root: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODEL_ORDER[1:]:
        frame = canonical_oof_frame(oof_frames[model], model).copy()
        frame = frame.sort_values("nodule_uid").reset_index(drop=True)
        baseline = frame["malignancy_raw_score"].to_numpy(dtype=np.float64)
        target = frame["target_normalized"].to_numpy(dtype=np.float64)
        target_rating = frame["target_1_to_5"].to_numpy(dtype=np.float64)
        deltas = np.empty((len(frame), 8), dtype=np.float64)
        random_orders = np.empty((len(frame), 100, 8), dtype=np.int64)
        error_orders = np.empty((len(frame), 8), dtype=np.int64)
        for fold in range(5):
            mask = frame["fold_index"].astype(int).to_numpy() == fold
            subset = frame.loc[mask]
            persisted, intervened = _intervention_group_contributions(
                subset, model, fold
            )
            deltas[mask] = intervened - persisted
            permutations = shared_intervention_permutations(base_seed, fold)
            random_orders[mask] = np.broadcast_to(
                permutations, (len(subset), *permutations.shape)
            )
            predictions, targets = _all_concept_values(subset, model)
            for local_index, global_index in enumerate(np.flatnonzero(mask)):
                continuous_prediction = {
                    group: predictions[group][local_index, 0]
                    for group in CONTINUOUS_CONCEPTS
                }
                continuous_target = {
                    group: targets[group][local_index, 0]
                    for group in CONTINUOUS_CONCEPTS
                }
                categorical_prediction = {
                    group: predictions[group][local_index]
                    for group in CATEGORICAL_CONCEPTS
                }
                categorical_target = {
                    group: targets[group][local_index]
                    for group in CATEGORICAL_CONCEPTS
                }
                ordered = error_first_order(
                    continuous_prediction,
                    continuous_target,
                    categorical_prediction,
                    categorical_target,
                )
                error_orders[global_index] = [
                    CONCEPT_GROUP_ORDER.index(group) for group in ordered
                ]
        random_mae = np.zeros((100, 9), dtype=np.float64)
        random_auc = np.zeros((100, 9), dtype=np.float64)
        error_mae = np.zeros(9, dtype=np.float64)
        error_auc = np.zeros(9, dtype=np.float64)
        for permutation_index in range(100):
            score = baseline.copy()
            random_mae[permutation_index, 0], random_auc[permutation_index, 0] = (
                _curve_metrics(score, target, target_rating)
            )
            for k in range(1, 9):
                group_indices = random_orders[:, permutation_index, k - 1]
                score = score + deltas[np.arange(len(frame)), group_indices]
                random_mae[permutation_index, k], random_auc[permutation_index, k] = (
                    _curve_metrics(score, target, target_rating)
                )
        score = baseline.copy()
        error_mae[0], error_auc[0] = _curve_metrics(score, target, target_rating)
        for k in range(1, 9):
            group_indices = error_orders[:, k - 1]
            score = score + deltas[np.arange(len(frame)), group_indices]
            error_mae[k], error_auc[k] = _curve_metrics(score, target, target_rating)
        baseline_mae, baseline_auc = _curve_metrics(
            baseline, target, target_rating
        )
        if not np.allclose(random_mae[:, 0], baseline_mae, atol=0.0, rtol=0.0) or not np.allclose(
            random_auc[:, 0], baseline_auc, atol=0.0, rtol=0.0
        ):
            raise ValueError(f"P9_INTERVENTION_K_ZERO_MISMATCH:{model}")
        x = np.arange(9, dtype=np.float64) / 8.0
        mean_mae = random_mae.mean(axis=0)
        mean_auc = random_auc.mean(axis=0)
        # k=0 is a direct persisted-score reuse, not a floating reduction over
        # 100 numerically identical copies.
        mean_mae[0] = baseline_mae
        mean_auc[0] = baseline_auc
        imae = float(np.trapz(mean_mae, x))
        iauc = float(np.trapz(mean_auc, x))
        fold_secondary = []
        for fold in range(5):
            mask = frame["fold_index"].astype(int).to_numpy() == fold
            fold_mae = np.zeros((100, 9), dtype=np.float64)
            fold_auc = np.zeros((100, 9), dtype=np.float64)
            for permutation_index in range(100):
                fold_score = baseline[mask].copy()
                fold_target = target[mask]
                fold_rating = target_rating[mask]
                fold_mae[permutation_index, 0], fold_auc[permutation_index, 0] = (
                    _curve_metrics(fold_score, fold_target, fold_rating)
                )
                for k in range(1, 9):
                    group_indices = random_orders[mask, permutation_index, k - 1]
                    fold_score = fold_score + deltas[mask][
                        np.arange(int(mask.sum())), group_indices
                    ]
                    fold_mae[permutation_index, k], fold_auc[permutation_index, k] = (
                        _curve_metrics(fold_score, fold_target, fold_rating)
                    )
            fold_secondary.append(
                {
                    "fold_index": fold,
                    "original_scale_mae_mean": fold_mae.mean(axis=0).tolist(),
                    "original_scale_mae_sd_across_permutations": fold_mae.std(
                        axis=0
                    ).tolist(),
                    "auroc_mean": fold_auc.mean(axis=0).tolist(),
                    "auroc_sd_across_permutations": fold_auc.std(axis=0).tolist(),
                    "iMAE": float(np.trapz(fold_mae.mean(axis=0), x)),
                    "iAUC": float(np.trapz(fold_auc.mean(axis=0), x)),
                }
            )
        result[model] = {
            "random_permutations": {
                "count": 100,
                "k": list(range(9)),
                "x": x.tolist(),
                "pooled_original_scale_mae_mean": mean_mae.tolist(),
                "pooled_original_scale_mae_sd": random_mae.std(axis=0).tolist(),
                "pooled_auroc_mean": mean_auc.tolist(),
                "pooled_auroc_sd": random_auc.std(axis=0).tolist(),
                "iMAE": imae,
                "iAUC": iauc,
                **intervention_deltas(baseline_mae, imae, baseline_auc, iauc),
            },
            "error_first": {
                "k": list(range(9)),
                "pooled_original_scale_mae": error_mae.tolist(),
                "pooled_auroc": error_auc.tolist(),
                "iMAE": float(np.trapz(error_mae, x)),
                "iAUC": float(np.trapz(error_auc, x)),
                **intervention_deltas(
                    baseline_mae,
                    float(np.trapz(error_mae, x)),
                    baseline_auc,
                    float(np.trapz(error_auc, x)),
                ),
            },
            "baseline_original_scale_mae": baseline_mae,
            "baseline_auroc": baseline_auc,
            "fold_secondary": fold_secondary,
            "fold_secondary_iMAE_mean": float(
                np.mean([item["iMAE"] for item in fold_secondary])
            ),
            "fold_secondary_iMAE_sd": float(
                np.std([item["iMAE"] for item in fold_secondary], ddof=0)
            ),
            "fold_secondary_iAUC_mean": float(
                np.mean([item["iAUC"] for item in fold_secondary])
            ),
            "fold_secondary_iAUC_sd": float(
                np.std([item["iAUC"] for item in fold_secondary], ddof=0)
            ),
        }
        if private_root is not None:
            private_root.mkdir(parents=True, exist_ok=True)
            path = private_root / f"{model}_intervention_curves.npz"
            temporary = path.with_name(f".{path.name}.tmp.npz")
            np.savez_compressed(
                temporary,
                random_original_scale_mae=random_mae,
                random_auroc=random_auc,
                error_first_original_scale_mae=error_mae,
                error_first_auroc=error_auc,
            )
            temporary.replace(path)
            result[model]["private_curve_sha256"] = sha256_file(path)
    return result


def learned_alpha_results(oof_frame: pd.DataFrame) -> dict[str, Any]:
    """Verify and report fold-level learned GAM alpha and gradient evidence."""
    result = []
    for fold in range(5):
        subset = oof_frame[oof_frame["fold_index"].astype(int) == fold]
        completion_path = RUN_ROOTS_DEFAULT["learned_softmax_gam"] / f"fold_{fold}" / "training_complete.json"
        history_path = RUN_ROOTS_DEFAULT["learned_softmax_gam"] / f"fold_{fold}" / "history.csv"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        history = pd.read_csv(history_path)
        best_epoch = int(completion["best_epoch_index"])
        best_rows = history[history["epoch_index"].astype(int) == best_epoch]
        if len(best_rows) != 1:
            raise ValueError(f"P9_GAM_ALPHA_BEST_EPOCH_MISMATCH:{fold}")
        best = best_rows.iloc[0]
        groups = {}
        for group in CONCEPT_GROUP_ORDER:
            logits = np.stack(
                [
                    _json_vector(value, 5, f"P9_GAM_ALPHA_LOGITS_INVALID:{fold}:{group}")
                    for value in subset[f"{group}_alpha_logits"]
                ]
            )
            weights = np.stack(
                [
                    _json_vector(value, 5, f"P9_GAM_ALPHA_WEIGHTS_INVALID:{fold}:{group}")
                    for value in subset[f"{group}_alpha_weights"]
                ]
            )
            if not np.allclose(logits, logits[0], atol=0.0, rtol=0.0) or not np.allclose(
                weights, weights[0], atol=0.0, rtol=0.0
            ):
                raise ValueError(f"P9_GAM_ALPHA_NOT_FOLD_LEVEL:{fold}:{group}")
            shifted = logits[0].astype(np.float32) - np.max(logits[0].astype(np.float32))
            reconstructed = np.exp(shifted).astype(np.float32)
            reconstructed = reconstructed / reconstructed.sum(dtype=np.float32)
            if not np.allclose(
                weights[0].astype(np.float32),
                reconstructed,
                atol=2e-7,
                rtol=1e-6,
            ):
                raise ValueError(f"P9_GAM_ALPHA_SOFTMAX_MISMATCH:{fold}:{group}")
            gradient = float(best[f"alpha_{group}_gradient_l1"])
            if not math.isfinite(gradient) or gradient <= 0.0:
                raise ValueError(f"P9_GAM_ALPHA_GRADIENT_INVALID:{fold}:{group}")
            groups[group] = {
                "logits": logits[0].tolist(),
                "weights": weights[0].tolist(),
                "gradient_l1_at_best_epoch": gradient,
            }
        result.append(
            {
                "fold_index": fold,
                "best_epoch_index": best_epoch,
                "groups": groups,
                "best_alpha_snapshot_sha256": completion[
                    "best_alpha_snapshot_sha256"
                ],
            }
        )
    return {
        "initial_weights_per_expert": [0.2] * 5,
        "folds": result,
    }


def centering_results(p9_root: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for model in MODEL_ORDER[1:]:
        folds = []
        for fold in range(5):
            path = (
                p9_root
                / "spatial"
                / model
                / f"fold_{fold}"
                / "train_contribution_centering.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            folds.append(
                {
                    "fold_index": fold,
                    "train_sample_count": int(payload["sample_count"]),
                    "train_group_means_raw_normalized_units": payload[
                        "raw_group_means"
                    ],
                    "train_group_means_rating_point_units": payload[
                        "rating_group_means"
                    ],
                    "normalized_reconstruction_max_abs_error": float(
                        payload["normalized_reconstruction_max_abs_error"]
                    ),
                    "rating_reconstruction_max_abs_error": float(
                        payload["rating_reconstruction_max_abs_error"]
                    ),
                    "artifact_sha256": sha256_file(path),
                }
            )
        report[model] = {"folds": folds}
    return report


def _faithfulness_aggregate(
    rows: Sequence[Mapping[str, Any]], quantity: str
) -> dict[str, Any] | None:
    if not rows:
        return None
    aggregate = aggregate_faithfulness_records(list(rows), quantity)
    comparison = np.asarray(
        [
            float(row[f"saliency_minus_random_mean_{quantity}"])
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.isfinite(comparison).all():
        raise ValueError("P9_SPATIAL_COMPARISON_NONFINITE")
    aggregate["saliency_minus_matched_random_mean"] = {
        "mean": float(comparison.mean()),
        "sd": float(comparison.std(ddof=0)),
        "median": float(np.median(comparison)),
        "percentile_2_5": float(np.percentile(comparison, 2.5)),
        "percentile_97_5": float(np.percentile(comparison, 97.5)),
    }
    return aggregate


def spatial_results(p9_root: Path) -> dict[str, Any]:
    verified = verify_all(p9_root=p9_root, approval_path=SPATIAL_APPROVAL_DEFAULT)
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    counts: dict[tuple[str, int, str], dict[str, int]] = defaultdict(
        lambda: {"valid": 0, "undefined": 0}
    )
    for model in MODEL_ORDER:
        for fold in range(5):
            output = p9_root / "spatial" / model / f"fold_{fold}"
            for path in sorted(output.glob("shard_*.parquet")):
                for row in read_and_verify_map_shard(path):
                    key = (model, fold, str(row["target"]))
                    if row["status"] == "undefined":
                        counts[key]["undefined"] += 1
                        continue
                    counts[key]["valid"] += 1
                    grouped[key].append(
                        json.loads(row["faithfulness_json"])
                    )
    by_model: dict[str, Any] = {}
    for model in MODEL_ORDER:
        canonical_targets = [spec.name for spec in target_specs(model)]
        fold_reports = []
        pooled_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pooled_all_targets: list[dict[str, Any]] = []
        for fold in range(5):
            targets = {}
            fold_all_targets: list[dict[str, Any]] = []
            for target in canonical_targets:
                rows = grouped.get((model, fold, target), [])
                targets[target] = {
                    "valid_map_count": counts[(model, fold, target)]["valid"],
                    "undefined_map_count": counts[(model, fold, target)][
                        "undefined"
                    ],
                    **{
                        quantity: _faithfulness_aggregate(rows, quantity)
                        for quantity in ("output_sensitivity", "error_increase")
                    },
                }
                pooled_by_target[target].extend(rows)
                fold_all_targets.extend(rows)
                pooled_all_targets.extend(rows)
            fold_reports.append(
                {
                    "fold_index": fold,
                    "targets": targets,
                    "all_targets": {
                        quantity: _faithfulness_aggregate(
                            fold_all_targets, quantity
                        )
                        for quantity in ("output_sensitivity", "error_increase")
                    },
                }
            )
        by_model[model] = {
            "folds": fold_reports,
            "pooled_targets": {
                target: {
                    "valid_map_count": sum(
                        counts[(model, fold, target)]["valid"]
                        for fold in range(5)
                    ),
                    "undefined_map_count": sum(
                        counts[(model, fold, target)]["undefined"]
                        for fold in range(5)
                    ),
                    **{
                        quantity: _faithfulness_aggregate(rows, quantity)
                        for quantity in ("output_sensitivity", "error_increase")
                    },
                }
                for target, rows in (
                    (target, pooled_by_target[target])
                    for target in canonical_targets
                )
            },
            "pooled_all_targets": {
                quantity: _faithfulness_aggregate(pooled_all_targets, quantity)
                for quantity in ("output_sensitivity", "error_increase")
            },
        }
    return {
        "jobs": int(verified["jobs"]),
        "models": by_model,
        "output_sensitivity_interpretation": "output sensitivity only; not evidence that prediction worsened",
        "error_increase_interpretation": "positive values mean prediction error worsened after occlusion",
    }


def _private_storage(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }


def _write_private_draws(path: Path, draws: Mapping[str, Any]) -> str:
    rows = []
    for kind, values in draws.items():
        for draw_index, draw in enumerate(values):
            rows.append(
                {
                    "kind": kind,
                    "draw_index": draw_index,
                    "patient_keys_json": json.dumps(
                        list(map(str, draw)), separators=(",", ":")
                    ),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return sha256_file(path)


def _forbidden_values(manifest_path: Path) -> set[str]:
    frame = pd.read_parquet(manifest_path)
    values: set[str] = set()
    for column in (
        "nodule_uid",
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "scan_id",
    ):
        if column in frame:
            values.update(frame[column].dropna().astype(str))
    return values


def build_audit(
    *,
    p9_root: Path = P9_ROOT_DEFAULT,
    audit_root: Path = AUDIT_ROOT_DEFAULT,
    split_root: Path = P4_SPLIT_ROOT_DEFAULT,
    manifest_path: Path = MANIFEST_DEFAULT,
    base_seed: int = BASE_SEED,
) -> dict[str, Any]:
    validate_p9_execution_config(P9_EXECUTION_CONFIG_DEFAULT)
    inputs = materialize_canonical_inputs(p9_root)
    integrity = verify_inputs(inputs["oof_root"], split_root)
    task = build_task_results(
        inputs["oof_root"], inputs["validation_root"], split_root
    )
    oof_frames = {
        model: pd.read_parquet(inputs["oof_root"] / OOF_FILENAMES[model])
        for model in MODEL_ORDER
    }
    concepts = concept_results(oof_frames)
    bootstrap, private_draws = bootstrap_results(oof_frames, base_seed=base_seed)
    interventions = intervention_results(
        oof_frames,
        base_seed=base_seed,
        private_root=p9_root / "intervention",
    )
    learned_alpha = learned_alpha_results(oof_frames["learned_softmax_gam"])
    centering = centering_results(p9_root)
    spatial = spatial_results(p9_root)
    draw_path = p9_root / "bootstrap" / "patient_cluster_draws.parquet"
    draw_sha = _write_private_draws(draw_path, private_draws)
    reports = {
        "integrity.json": integrity,
        "task.json": task,
        "concept.json": concepts,
        "contribution_centering.json": centering,
        "intervention.json": interventions,
        "bootstrap.json": bootstrap,
        "learned_alpha.json": learned_alpha,
        "spatial.json": spatial,
    }
    for filename, payload in reports.items():
        write_json(
            audit_root / filename,
            {"schema_version": SCHEMA_VERSION, "status": "PASS", **payload},
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "phase": "P9",
        "models": list(MODEL_ORDER),
        "oof_nodules": int(integrity["unique_nodules"]),
        "oof_patients": int(integrity["unique_patients"]),
        "patient_leakage": int(integrity["patient_leakage"]),
        "fold_counts": integrity["fold_counts"],
        "spatial_jobs_verified": int(spatial["jobs"]),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_private_draws_sha256": draw_sha,
        "canonical_oof_sha256": inputs["oof_sha256"],
        "validation_auxiliary_sha256": inputs["validation_sha256"],
        "tracked_report_sha256": {
            filename: sha256_file(audit_root / filename) for filename in reports
        },
        "private_storage": _private_storage(p9_root),
        "p5_through_p8_artifacts_modified": False,
        "second_committed_test_evaluation": False,
        "p10_started": False,
    }
    write_json(audit_root / "summary.json", summary)
    forbidden = _forbidden_values(manifest_path)
    for path in audit_root.glob("*.json"):
        assert_deidentified_audit(path, forbidden)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--p9-root", type=Path, default=P9_ROOT_DEFAULT)
    build.add_argument("--audit-root", type=Path, default=AUDIT_ROOT_DEFAULT)
    build.add_argument("--split-root", type=Path, default=P4_SPLIT_ROOT_DEFAULT)
    build.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    build.add_argument("--base-seed", type=int, default=BASE_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build_audit(
        p9_root=arguments.p9_root,
        audit_root=arguments.audit_root,
        split_root=arguments.split_root,
        manifest_path=arguments.manifest,
        base_seed=arguments.base_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
