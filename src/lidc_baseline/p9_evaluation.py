"""Unified P9 evaluation primitives for frozen Baseline-v2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p4_prepare import read_split, sha256_file
from lidc_baseline.p6_standard_cbm import (
    CATEGORICAL_CONCEPTS,
    CONCEPT_GROUP_ORDER,
    CONTINUOUS_CONCEPTS,
)


SCHEMA_VERSION = 1
MODEL_ORDER = ("blackbox", "standard_cbm", "mixed_cem", "learned_softmax_gam")
MODEL_PAIRS = tuple(itertools.combinations(MODEL_ORDER, 2))
FOLD_TEST_COUNTS = (479, 502, 539, 549, 564)
EXPECTED_NODULES = 2633
EXPECTED_PATIENTS = 868
BOOTSTRAP_DRAWS = 2000
INTERVENTION_PERMUTATIONS = 100
P9_EXECUTION_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_p9_evaluation_h200.yaml"
)
P4_SPLIT_ROOT_DEFAULT = Path("artifacts/baseline_v2/splits")
OOF_FILENAMES = {
    "blackbox": "blackbox_oof_predictions.parquet",
    "standard_cbm": "standard_cbm_oof_predictions.parquet",
    "mixed_cem": "cem_oof_predictions.parquet",
    "learned_softmax_gam": "gam_oof_predictions.parquet",
}
VALIDATION_FILENAMES = {
    model: f"{model}_validation_predictions.parquet" for model in MODEL_ORDER
}


def validate_p9_execution_config(
    config_path: str | Path = P9_EXECUTION_CONFIG_DEFAULT,
    digest_path: str | Path | None = None,
) -> tuple[dict[str, Any], str]:
    source = Path(config_path)
    config = load_config(source)
    observed = compute_config_sha256(config)
    digest = Path(digest_path) if digest_path is not None else source.with_suffix(
        ".sha256"
    )
    if digest.read_text(encoding="ascii").strip() != observed:
        raise ValueError("P9_EXECUTION_CONFIG_HASH_MISMATCH")
    project = config.get("project_preregistered", {})
    frozen = project.get("frozen_inputs", {})
    secondary = project.get("task_evaluation", {}).get("secondary_extreme", {})
    intervention = project.get("intervention", {})
    occlusion = project.get("occlusion", {})
    formal = project.get("formal_spatial_execution", {})
    if (
        config.get("protocol_version") != "Baseline-v2"
        or config.get("phase") != "P9"
        or tuple(frozen.get("model_order", ())) != MODEL_ORDER
        or frozen.get("access") != "read_only"
        or frozen.get("retraining") != "forbidden"
        or frozen.get("second_committed_test_evaluation") != "forbidden"
        or frozen.get("artifact_rewrite") != "forbidden"
    ):
        raise ValueError("P9_EXECUTION_IDENTITY_OR_READ_ONLY_POLICY_MISMATCH")
    if (
        secondary.get("youden_selection_partition")
        != "fold_specific_validation_extreme_subset_only"
        or secondary.get("middle_samples_in_threshold_selection") is not False
        or secondary.get("tie_break") != "largest_threshold"
    ):
        raise ValueError("P9_YOUDEN_POLICY_MISMATCH")
    if (
        intervention.get("delta_iMAE") != "baseline_MAE_minus_iMAE"
        or intervention.get("delta_iAUC") != "iAUC_minus_baseline_AUROC"
        or intervention.get("positive_delta_means_improvement") is not True
    ):
        raise ValueError("P9_INTERVENTION_DELTA_POLICY_MISMATCH")
    if (
        occlusion.get("random_masks_per_target") != 20
        or occlusion.get("retain_individual_random_output_sensitivity_values") != 20
        or occlusion.get("retain_individual_random_error_increase_values") != 20
        or occlusion.get("output_sensitivity_is_prediction_worsening_evidence")
        is not False
    ):
        raise ValueError("P9_DOUBLE_FAITHFULNESS_POLICY_MISMATCH")
    if (
        formal.get("default_approval_value") != 0
        or formal.get("required_approval_value") != 1
        or formal.get("user_approval_after_stage_a_required") is not True
        or formal.get("jobs") != 20
    ):
        raise ValueError("P9_FORMAL_APPROVAL_POLICY_MISMATCH")
    return config, observed


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"P9_INVALID_FINITE_VECTOR:{name}")
    return result


def _correlations(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    if np.unique(target).size < 2 or np.unique(prediction).size < 2:
        return math.nan, math.nan
    return (
        float(stats.pearsonr(target, prediction).statistic),
        float(stats.spearmanr(target, prediction).statistic),
    )


def regression_metrics(
    score_normalized: Sequence[float], target_normalized: Sequence[float]
) -> dict[str, float | list[float]]:
    """Compute P9 primary metrics from authoritative unclipped normalized scores."""
    score = _finite_vector(score_normalized, "score_normalized")
    target = _finite_vector(target_normalized, "target_normalized")
    if score.shape != target.shape:
        raise ValueError("P9_TASK_METRIC_SHAPE_MISMATCH")
    error = score - target
    pearson, spearman = _correlations(target, score)
    return {
        "original_scale_mae": float(np.mean(np.abs(4.0 * error))),
        "original_scale_rmse": float(np.sqrt(np.mean(np.square(4.0 * error)))),
        "normalized_mae": float(np.mean(np.abs(error))),
        "pearson": pearson,
        "spearman": spearman,
        "prediction_range_normalized": [float(score.min()), float(score.max())],
        "below_zero_rate": float(np.mean(score < 0.0)),
        "above_one_rate": float(np.mean(score > 1.0)),
        "prediction_range_1_to_5": [
            float(1.0 + 4.0 * score.min()),
            float(1.0 + 4.0 * score.max()),
        ],
        "below_one_rate": float(np.mean((1.0 + 4.0 * score) < 1.0)),
        "above_five_rate": float(np.mean((1.0 + 4.0 * score) > 5.0)),
        "sample_count": int(score.size),
    }


def extreme_labels(target_1_to_5: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    target = _finite_vector(target_1_to_5, "target_1_to_5")
    eligible = (target <= 2.0) | (target >= 4.0)
    labels = (target >= 4.0).astype(np.int64)
    return eligible, labels


def select_youden_threshold(
    validation_scores: Sequence[float], validation_target_1_to_5: Sequence[float]
) -> dict[str, float | int]:
    """Select Youden-J on validation extremes only; ties use largest threshold."""
    score = _finite_vector(validation_scores, "validation_scores")
    target = _finite_vector(validation_target_1_to_5, "validation_target_1_to_5")
    if score.shape != target.shape:
        raise ValueError("P9_YOUDEN_SHAPE_MISMATCH")
    eligible, all_labels = extreme_labels(target)
    extreme_scores = score[eligible]
    labels = all_labels[eligible]
    if extreme_scores.size == 0 or np.unique(labels).size != 2:
        raise ValueError("P9_YOUDEN_VALIDATION_EXTREMES_REQUIRE_BOTH_CLASSES")
    candidates = np.unique(extreme_scores[np.isfinite(extreme_scores)])
    if candidates.size == 0:
        raise ValueError("P9_YOUDEN_NO_FINITE_THRESHOLD")
    best_threshold = math.nan
    best_j = -math.inf
    for threshold in candidates:
        predicted = extreme_scores >= threshold
        sensitivity = float(np.mean(predicted[labels == 1]))
        specificity = float(np.mean(~predicted[labels == 0]))
        youden = sensitivity + specificity - 1.0
        if youden > best_j or (youden == best_j and threshold > best_threshold):
            best_j = youden
            best_threshold = float(threshold)
    return {
        "threshold": best_threshold,
        "youden_j": float(best_j),
        "validation_extreme_sample_count": int(extreme_scores.size),
        "validation_low_count": int(np.sum(labels == 0)),
        "validation_high_count": int(np.sum(labels == 1)),
        "validation_middle_excluded_count": int(np.sum(~eligible)),
    }


def secondary_metrics(
    scores: Sequence[float], target_1_to_5: Sequence[float], threshold: float
) -> dict[str, float | int]:
    score = _finite_vector(scores, "secondary_scores")
    target = _finite_vector(target_1_to_5, "secondary_target")
    if score.shape != target.shape or not math.isfinite(threshold):
        raise ValueError("P9_SECONDARY_INPUT_MISMATCH")
    eligible, labels_all = extreme_labels(target)
    score = score[eligible]
    labels = labels_all[eligible]
    if score.size == 0 or np.unique(labels).size != 2:
        raise ValueError("P9_SECONDARY_EXTREMES_REQUIRE_BOTH_CLASSES")
    predicted = (score >= threshold).astype(np.int64)
    fixed = (score >= 0.5).astype(np.int64)
    return {
        "auroc": float(metrics.roc_auc_score(labels, score)),
        "auprc": float(metrics.average_precision_score(labels, score)),
        "accuracy": float(metrics.accuracy_score(labels, predicted)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(metrics.recall_score(labels, predicted, pos_label=1)),
        "specificity": float(metrics.recall_score(labels, predicted, pos_label=0)),
        "macro_f1": float(metrics.f1_score(labels, predicted, average="macro")),
        "fixed_0_5_sensitivity": float(metrics.recall_score(labels, fixed, pos_label=1)),
        "sample_count": int(score.size),
        "low_count": int(np.sum(labels == 0)),
        "high_count": int(np.sum(labels == 1)),
    }


def continuous_concept_metrics(
    prediction: Sequence[float], target: Sequence[float]
) -> dict[str, float | int]:
    predicted = _finite_vector(prediction, "continuous_concept_prediction")
    expected = _finite_vector(target, "continuous_concept_target")
    if predicted.shape != expected.shape:
        raise ValueError("P9_CONTINUOUS_CONCEPT_SHAPE_MISMATCH")
    error = predicted - expected
    pearson, spearman = _correlations(expected, predicted)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": pearson,
        "spearman": spearman,
        "sample_count": int(predicted.size),
    }


def categorical_concept_metrics(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    true_ties: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    predicted = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    if (
        predicted.ndim != 2
        or predicted.shape != expected.shape
        or predicted.shape[0] == 0
        or not np.isfinite(predicted).all()
        or not np.isfinite(expected).all()
        or np.min(predicted) < 0.0
        or np.min(expected) < 0.0
        or not np.allclose(predicted.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        or not np.allclose(expected.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("P9_CATEGORICAL_CONCEPT_INPUT_INVALID")
    recomputed_ties = np.sum(
        np.isclose(expected, expected.max(axis=1, keepdims=True), atol=1e-12, rtol=0.0),
        axis=1,
    ) > 1
    if true_ties is not None:
        observed_ties = np.asarray(true_ties)
        if observed_ties.dtype != np.bool_ or not np.array_equal(
            observed_ties, recomputed_ties
        ):
            raise ValueError("P9_CATEGORICAL_TRUE_TIE_MISMATCH")
    hard_mask = ~recomputed_ties
    hard_n = int(np.sum(hard_mask))
    macro_f1 = (
        float(
            metrics.f1_score(
                np.argmax(expected[hard_mask], axis=1),
                np.argmax(predicted[hard_mask], axis=1),
                labels=np.arange(predicted.shape[1]),
                average="macro",
                zero_division=0,
            )
        )
        if hard_n
        else math.nan
    )
    return {
        "soft_cross_entropy": float(
            np.mean(-np.sum(expected * np.log(np.clip(predicted, 1e-12, 1.0)), axis=1))
        ),
        "multiclass_brier": float(np.mean(np.sum(np.square(predicted - expected), axis=1))),
        "hard_modal_macro_f1": macro_f1,
        "soft_sample_count": int(predicted.shape[0]),
        "hard_sample_count": hard_n,
        "true_tie_count": int(np.sum(recomputed_ties)),
    }


def center_contributions(
    train_contributions: Sequence[Sequence[float]],
    evaluation_contributions: Sequence[Sequence[float]],
    raw_bias: float | Sequence[float],
    raw_score: Sequence[float],
    *,
    train_uids: Sequence[str],
    expected_train_uids: Sequence[str],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    train = np.asarray(train_contributions, dtype=np.float64)
    evaluation = np.asarray(evaluation_contributions, dtype=np.float64)
    score = _finite_vector(raw_score, "centering_raw_score")
    bias = np.asarray(raw_bias, dtype=np.float64)
    observed_uids = tuple(map(str, train_uids))
    expected_uids = tuple(map(str, expected_train_uids))
    if (
        not observed_uids
        or len(observed_uids) != len(set(observed_uids))
        or len(expected_uids) != len(set(expected_uids))
        or len(observed_uids) != len(expected_uids)
        or set(observed_uids) != set(expected_uids)
        or len(observed_uids) != train.shape[0]
    ):
        raise ValueError("P9_CENTERING_TRAIN_UID_MEMBERSHIP_MISMATCH")
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != 8:
        raise ValueError("P9_CENTERING_CONTRIBUTION_SHAPE_MISMATCH")
    if evaluation.shape != (score.size, 8) or bias.ndim > 1:
        raise ValueError("P9_CENTERING_EVALUATION_SHAPE_MISMATCH")
    if bias.ndim == 0:
        bias = np.full(score.size, float(bias), dtype=np.float64)
    if bias.shape != score.shape or not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError("P9_CENTERING_NONFINITE_OR_BIAS_MISMATCH")
    mu = train.mean(axis=0)
    centered = evaluation - mu
    centered_bias = bias + float(mu.sum())
    reconstructed = centered_bias + centered.sum(axis=1)
    raw_error = float(np.max(np.abs(reconstructed - score)))
    rating_reconstructed = 1.0 + 4.0 * centered_bias + (4.0 * centered).sum(axis=1)
    rating_error = float(np.max(np.abs(rating_reconstructed - (1.0 + 4.0 * score))))
    if raw_error > tolerance or rating_error > tolerance:
        raise ValueError("P9_CENTERING_RECONSTRUCTION_MISMATCH")
    return {
        "train_group_means": mu,
        "centered_contributions": centered,
        "centered_bias": centered_bias,
        "normalized_reconstruction_max_abs_error": raw_error,
        "rating_reconstruction_max_abs_error": rating_error,
    }


def _seed_from_material(material: str) -> int:
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def shared_intervention_permutations(
    base_seed: int, fold_index: int, *, permutations: int = INTERVENTION_PERMUTATIONS
) -> np.ndarray:
    if fold_index not in range(5) or permutations != 100:
        raise ValueError("P9_INTERVENTION_PERMUTATION_POLICY_MISMATCH")
    result = []
    for index in range(permutations):
        material = (
            "Baseline-v2/P9/intervention-permutation|"
            f"{base_seed}|{fold_index}|{index}|{SCHEMA_VERSION}"
        )
        result.append(np.random.default_rng(_seed_from_material(material)).permutation(8))
    return np.asarray(result, dtype=np.int64)


def error_first_order(
    continuous_prediction: Mapping[str, float],
    continuous_target: Mapping[str, float],
    categorical_prediction: Mapping[str, Sequence[float]],
    categorical_target: Mapping[str, Sequence[float]],
) -> tuple[str, ...]:
    distances: dict[str, float] = {}
    for group in CONTINUOUS_CONCEPTS:
        distances[group] = abs(
            float(continuous_prediction[group]) - float(continuous_target[group])
        )
    for group in CATEGORICAL_CONCEPTS:
        predicted = np.asarray(categorical_prediction[group], dtype=np.float64)
        target = np.asarray(categorical_target[group], dtype=np.float64)
        if predicted.shape != target.shape:
            raise ValueError("P9_ERROR_FIRST_CATEGORICAL_SHAPE_MISMATCH")
        distances[group] = float(0.5 * np.sum(np.abs(predicted - target)))
    canonical = {group: index for index, group in enumerate(CONCEPT_GROUP_ORDER)}
    return tuple(sorted(CONCEPT_GROUP_ORDER, key=lambda g: (-distances[g], canonical[g])))


def intervention_deltas(
    baseline_mae: float,
    intervention_mae: float,
    baseline_auroc: float,
    intervention_auroc: float,
) -> dict[str, float]:
    values = np.asarray(
        [baseline_mae, intervention_mae, baseline_auroc, intervention_auroc],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("P9_INTERVENTION_DELTA_NONFINITE")
    return {
        "Delta_iMAE": float(baseline_mae - intervention_mae),
        "Delta_iAUC": float(intervention_auroc - baseline_auroc),
    }


def patient_cluster_bootstrap_draws(
    patient_keys: Sequence[str], base_seed: int, *, draws: int = BOOTSTRAP_DRAWS
) -> list[np.ndarray]:
    patients = np.asarray(sorted(set(map(str, patient_keys))), dtype=object)
    if patients.size == 0 or draws != 2000:
        raise ValueError("P9_BOOTSTRAP_POLICY_MISMATCH")
    rng = np.random.default_rng(
        _seed_from_material(f"Baseline-v2/P9/bootstrap|{base_seed}|primary|{SCHEMA_VERSION}")
    )
    return [rng.choice(patients, size=patients.size, replace=True) for _ in range(draws)]


def secondary_patient_bootstrap_draws(
    patient_labels: Mapping[str, Sequence[int]],
    base_seed: int,
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[np.ndarray]:
    patients = np.asarray(sorted(map(str, patient_labels)), dtype=object)
    labels = {
        str(key): tuple(int(value) for value in values)
        for key, values in patient_labels.items()
    }
    flattened = [label for values in labels.values() for label in values]
    if (
        patients.size == 0
        or not all(values and set(values) <= {0, 1} for values in labels.values())
        or set(flattened) != {0, 1}
        or draws != 2000
    ):
        raise ValueError("P9_SECONDARY_BOOTSTRAP_POLICY_MISMATCH")
    rng = np.random.default_rng(
        _seed_from_material(f"Baseline-v2/P9/bootstrap|{base_seed}|secondary|{SCHEMA_VERSION}")
    )
    accepted: list[np.ndarray] = []
    while len(accepted) < draws:
        draw = rng.choice(patients, size=patients.size, replace=True)
        drawn_labels = {
            label for patient in draw for label in labels[str(patient)]
        }
        if drawn_labels == {0, 1}:
            accepted.append(draw)
    return accepted


def bootstrap_draw_sha256(draws: Sequence[Sequence[str]]) -> str:
    payload = json.dumps(
        [list(map(str, draw)) for draw in draws],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def all_model_pairs() -> tuple[tuple[str, str], ...]:
    if len(MODEL_PAIRS) != 6 or len(set(MODEL_PAIRS)) != 6:
        raise AssertionError("P9_MODEL_PAIR_ENUMERATION_INVALID")
    return MODEL_PAIRS


def paired_differences(
    metric_by_model: Mapping[str, float], *, metric: str
) -> dict[str, float]:
    if set(metric_by_model) != set(MODEL_ORDER):
        raise ValueError("P9_PAIRED_MODEL_SET_MISMATCH")
    result = {}
    for first, second in all_model_pairs():
        if metric == "mae":
            value = float(metric_by_model[first] - metric_by_model[second])
        elif metric == "auroc":
            value = float(metric_by_model[second] - metric_by_model[first])
        else:
            raise ValueError(f"P9_UNKNOWN_PAIRED_METRIC:{metric}")
        result[f"{first}__{second}"] = value
    return result


def canonical_oof_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    if model not in MODEL_ORDER:
        raise ValueError(f"P9_UNKNOWN_MODEL:{model}")
    aliases = {
        "malignancy_raw_score": ("malignancy_raw_score", "raw_task_score"),
        "target_normalized": ("target_normalized", "malignancy_target_normalized"),
        "target_1_to_5": ("target_1_to_5", "mean_malignancy"),
    }
    result = frame.copy()
    for canonical, candidates in aliases.items():
        if canonical in result:
            continue
        found = next((name for name in candidates if name in result), None)
        if found is None:
            raise ValueError(f"P9_OOF_REQUIRED_COLUMN_MISSING:{model}:{canonical}")
        result[canonical] = result[found]
    required = {"nodule_uid", "patient_key", "fold_index", *aliases}
    if not required <= set(result):
        raise ValueError(f"P9_OOF_REQUIRED_COLUMN_MISSING:{model}")
    result["nodule_uid"] = result["nodule_uid"].astype(str)
    result["patient_key"] = result["patient_key"].astype(str)
    return result


def verify_oof_equality(
    frames: Mapping[str, pd.DataFrame],
    expected_fold_uids: Mapping[int, Sequence[str]] | None = None,
) -> dict[str, Any]:
    if set(frames) != set(MODEL_ORDER):
        raise ValueError("P9_OOF_MODEL_SET_MISMATCH")
    canonical = {model: canonical_oof_frame(frames[model], model) for model in MODEL_ORDER}
    reference = canonical[MODEL_ORDER[0]].sort_values("nodule_uid").reset_index(drop=True)
    if (
        len(reference) != EXPECTED_NODULES
        or reference["nodule_uid"].duplicated().any()
        or reference["patient_key"].nunique() != EXPECTED_PATIENTS
        or set(reference["fold_index"].astype(int)) != set(range(5))
        or tuple(reference.groupby("fold_index").size().sort_index()) != FOLD_TEST_COUNTS
        or int(reference.groupby("patient_key")["fold_index"].nunique().max()) != 1
    ):
        raise ValueError("P9_REFERENCE_OOF_INTEGRITY_MISMATCH")
    if expected_fold_uids is not None:
        if set(expected_fold_uids) != set(range(5)):
            raise ValueError("P9_EXPECTED_FOLD_SET_MISMATCH")
        for fold_index in range(5):
            observed = set(
                reference.loc[
                    reference["fold_index"].astype(int) == fold_index, "nodule_uid"
                ]
            )
            expected = tuple(map(str, expected_fold_uids[fold_index]))
            if len(expected) != len(set(expected)) or observed != set(expected):
                raise ValueError(f"P9_P4_TEST_MEMBERSHIP_MISMATCH:{fold_index}")
    for model in MODEL_ORDER[1:]:
        observed = canonical[model].sort_values("nodule_uid").reset_index(drop=True)
        for column in ("nodule_uid", "patient_key", "fold_index"):
            if not np.array_equal(observed[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"P9_OOF_IDENTITY_MISMATCH:{model}:{column}")
        for column in ("target_normalized", "target_1_to_5"):
            if not np.allclose(
                observed[column].astype(float), reference[column].astype(float), atol=1e-12, rtol=0.0
            ):
                raise ValueError(f"P9_OOF_TARGET_MISMATCH:{model}:{column}")
    return {
        "status": "PASS",
        "unique_nodules": EXPECTED_NODULES,
        "unique_patients": EXPECTED_PATIENTS,
        "fold_counts": list(FOLD_TEST_COUNTS),
        "patient_leakage": 0,
    }


def load_oof_frames(oof_root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(oof_root)
    return {
        model: pd.read_parquet(root / filename)
        for model, filename in OOF_FILENAMES.items()
    }


def load_validation_frames(validation_root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(validation_root)
    return {
        model: pd.read_parquet(root / filename)
        for model, filename in VALIDATION_FILENAMES.items()
    }


def load_expected_partition_uids(
    split_root: str | Path = P4_SPLIT_ROOT_DEFAULT,
) -> dict[str, dict[int, tuple[str, ...]]]:
    membership = load_expected_partition_membership(split_root)
    return {
        partition: {
            fold_index: values["nodule_uids"]
            for fold_index, values in folds.items()
        }
        for partition, folds in membership.items()
    }


def load_expected_partition_membership(
    split_root: str | Path = P4_SPLIT_ROOT_DEFAULT,
) -> dict[str, dict[int, dict[str, tuple[str, ...]]]]:
    root = Path(split_root)
    result: dict[str, dict[int, dict[str, tuple[str, ...]]]] = {
        "validation": {},
        "test": {},
    }
    for fold_index in range(5):
        payload = read_split(root / f"fold_{fold_index}.json")
        if int(payload.get("fold_index", -1)) != fold_index:
            raise ValueError(f"P9_P4_SPLIT_FOLD_ID_MISMATCH:{fold_index}")
        for partition in result:
            registered = payload["partitions"][partition]
            result[partition][fold_index] = {
                "nodule_uids": tuple(map(str, registered["nodule_uids"])),
                "patient_keys": tuple(map(str, registered["patient_keys"])),
            }
    return result


def load_expected_fold_uids(
    split_root: str | Path = P4_SPLIT_ROOT_DEFAULT,
) -> dict[int, tuple[str, ...]]:
    return load_expected_partition_uids(split_root)["test"]


def verify_validation_membership(
    frames: Mapping[str, pd.DataFrame],
    expected_validation_uids: Mapping[int, Sequence[str]],
    *,
    expected_test_uids: Mapping[int, Sequence[str]] | None = None,
    expected_validation_patient_keys: Mapping[int, Sequence[str]] | None = None,
    expected_test_patient_keys: Mapping[int, Sequence[str]] | None = None,
) -> None:
    if set(frames) != set(MODEL_ORDER) or set(expected_validation_uids) != set(
        range(5)
    ):
        raise ValueError("P9_VALIDATION_MODEL_OR_FOLD_SET_MISMATCH")
    optional_fold_mappings = (
        expected_test_uids,
        expected_validation_patient_keys,
        expected_test_patient_keys,
    )
    if any(
        mapping is not None and set(mapping) != set(range(5))
        for mapping in optional_fold_mappings
    ):
        raise ValueError("P9_VALIDATION_MODEL_OR_FOLD_SET_MISMATCH")
    if (expected_validation_patient_keys is None) != (
        expected_test_patient_keys is None
    ):
        raise ValueError("P9_VALIDATION_PATIENT_MEMBERSHIP_INCOMPLETE")
    reference: pd.DataFrame | None = None
    for model in MODEL_ORDER:
        frame = _validation_frame(frames[model], model)
        for fold_index in range(5):
            observed_rows = frame[frame["fold_index"].astype(int) == fold_index]
            observed = tuple(observed_rows["nodule_uid"])
            expected = tuple(map(str, expected_validation_uids[fold_index]))
            if (
                len(expected) != len(set(expected))
                or len(observed) != len(expected)
                or set(observed) != set(expected)
            ):
                raise ValueError(
                    f"P9_P4_VALIDATION_MEMBERSHIP_MISMATCH:{model}:{fold_index}"
                )
            if expected_test_uids is not None:
                expected_test = tuple(map(str, expected_test_uids[fold_index]))
                if (
                    len(expected_test) != len(set(expected_test))
                    or set(expected) & set(expected_test)
                ):
                    raise ValueError(
                        f"P9_VALIDATION_TEST_NODULE_OVERLAP:{fold_index}"
                    )
            if expected_validation_patient_keys is not None:
                validation_patients = tuple(
                    map(str, expected_validation_patient_keys[fold_index])
                )
                test_patients = tuple(map(str, expected_test_patient_keys[fold_index]))
                if (
                    len(validation_patients) != len(set(validation_patients))
                    or len(test_patients) != len(set(test_patients))
                    or set(validation_patients) & set(test_patients)
                ):
                    raise ValueError(
                        f"P9_VALIDATION_TEST_PATIENT_OVERLAP:{fold_index}"
                    )
        ordered = frame.sort_values(["fold_index", "nodule_uid"]).reset_index(
            drop=True
        )
        if reference is None:
            reference = ordered
            continue
        for column in ("fold_index", "nodule_uid"):
            if not np.array_equal(ordered[column], reference[column]):
                raise ValueError(
                    f"P9_VALIDATION_UID_IDENTITY_MISMATCH:{model}:{column}"
                )
        if not np.allclose(
            ordered["target_1_to_5"].astype(float),
            reference["target_1_to_5"].astype(float),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"P9_VALIDATION_TARGET_MISMATCH:{model}")


def verify_inputs(
    oof_root: str | Path, split_root: str | Path = P4_SPLIT_ROOT_DEFAULT
) -> dict[str, Any]:
    frames = load_oof_frames(oof_root)
    report = verify_oof_equality(frames, load_expected_fold_uids(split_root))
    report["oof_file_sha256"] = {
        model: sha256_file(Path(oof_root) / filename)
        for model, filename in OOF_FILENAMES.items()
    }
    return report


def _validation_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    required = {"fold_index", "malignancy_raw_score", "target_1_to_5", "nodule_uid"}
    if not required <= set(frame):
        raise ValueError(f"P9_VALIDATION_REQUIRED_COLUMN_MISSING:{model}")
    result = frame.copy()
    result["nodule_uid"] = result["nodule_uid"].astype(str)
    fold_values = result["fold_index"].to_numpy(dtype=float)
    if (
        not np.isfinite(fold_values).all()
        or not np.array_equal(fold_values, fold_values.astype(np.int64))
        or set(fold_values.astype(np.int64)) != set(range(5))
    ):
        raise ValueError(f"P9_VALIDATION_FOLD_SET_MISMATCH:{model}")
    result["fold_index"] = fold_values.astype(np.int64)
    if (
        result.duplicated(subset=["fold_index", "nodule_uid"]).any()
        or not np.isfinite(
            result[["malignancy_raw_score", "target_1_to_5"]].to_numpy(dtype=float)
        ).all()
        or ("model" in result and set(result["model"].astype(str)) != {model})
    ):
        raise ValueError(f"P9_VALIDATION_FRAME_INVALID:{model}")
    return result


def build_task_results(
    oof_root: str | Path,
    validation_root: str | Path,
    split_root: str | Path = P4_SPLIT_ROOT_DEFAULT,
) -> dict[str, Any]:
    frames = {
        model: canonical_oof_frame(frame, model)
        for model, frame in load_oof_frames(oof_root).items()
    }
    expected_membership = load_expected_partition_membership(split_root)
    expected_partition_uids = {
        partition: {
            fold_index: values["nodule_uids"]
            for fold_index, values in folds.items()
        }
        for partition, folds in expected_membership.items()
    }
    verify_oof_equality(frames, expected_partition_uids["test"])
    validation_raw = load_validation_frames(validation_root)
    verify_validation_membership(
        validation_raw,
        expected_partition_uids["validation"],
        expected_test_uids=expected_partition_uids["test"],
        expected_validation_patient_keys={
            fold_index: values["patient_keys"]
            for fold_index, values in expected_membership["validation"].items()
        },
        expected_test_patient_keys={
            fold_index: values["patient_keys"]
            for fold_index, values in expected_membership["test"].items()
        },
    )
    validation = {
        model: _validation_frame(frame, model)
        for model, frame in validation_raw.items()
    }
    model_results: dict[str, Any] = {}
    for model in MODEL_ORDER:
        test_frame = frames[model]
        validation_frame = validation[model]
        fold_reports = []
        pooled_labels: list[int] = []
        pooled_predictions: list[int] = []
        pooled_scores: list[float] = []
        for fold_index in range(5):
            fold_validation = validation_frame[
                validation_frame["fold_index"].astype(int) == fold_index
            ]
            threshold_report = select_youden_threshold(
                fold_validation["malignancy_raw_score"],
                fold_validation["target_1_to_5"],
            )
            fold_test = test_frame[test_frame["fold_index"].astype(int) == fold_index]
            threshold = float(threshold_report["threshold"])
            fold_secondary = secondary_metrics(
                fold_test["malignancy_raw_score"], fold_test["target_1_to_5"], threshold
            )
            eligible, labels = extreme_labels(fold_test["target_1_to_5"])
            fold_scores = fold_test["malignancy_raw_score"].to_numpy(dtype=float)[eligible]
            pooled_labels.extend(labels[eligible].tolist())
            pooled_scores.extend(fold_scores.tolist())
            pooled_predictions.extend((fold_scores >= threshold).astype(int).tolist())
            fold_reports.append(
                {
                    "fold_index": fold_index,
                    "threshold_selection": threshold_report,
                    "task": regression_metrics(
                        fold_test["malignancy_raw_score"], fold_test["target_normalized"]
                    ),
                    "secondary": fold_secondary,
                }
            )
        labels_array = np.asarray(pooled_labels, dtype=int)
        predictions_array = np.asarray(pooled_predictions, dtype=int)
        scores_array = np.asarray(pooled_scores, dtype=float)
        model_results[model] = {
            "folds": fold_reports,
            "pooled": regression_metrics(
                test_frame["malignancy_raw_score"], test_frame["target_normalized"]
            ),
            "pooled_secondary": {
                "auroc": float(metrics.roc_auc_score(labels_array, scores_array)),
                "auprc": float(metrics.average_precision_score(labels_array, scores_array)),
                "accuracy": float(metrics.accuracy_score(labels_array, predictions_array)),
                "balanced_accuracy": float(
                    metrics.balanced_accuracy_score(labels_array, predictions_array)
                ),
                "sensitivity": float(
                    metrics.recall_score(labels_array, predictions_array, pos_label=1)
                ),
                "specificity": float(
                    metrics.recall_score(labels_array, predictions_array, pos_label=0)
                ),
                "macro_f1": float(
                    metrics.f1_score(labels_array, predictions_array, average="macro")
                ),
                "fixed_0_5_sensitivity": float(
                    metrics.recall_score(
                        labels_array, (scores_array >= 0.5).astype(int), pos_label=1
                    )
                ),
                "sample_count": int(labels_array.size),
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "models": model_results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=P9_EXECUTION_CONFIG_DEFAULT)
    parser.add_argument("--split-root", type=Path, default=P4_SPLIT_ROOT_DEFAULT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--oof-root", type=Path, required=True)
    build = subparsers.add_parser("build-results")
    build.add_argument("--oof-root", type=Path, required=True)
    build.add_argument("--validation-root", type=Path, required=True)
    build.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    validate_p9_execution_config(arguments.config)
    report = (
        verify_inputs(arguments.oof_root, arguments.split_root)
        if arguments.command == "verify-inputs"
        else build_task_results(
            arguments.oof_root, arguments.validation_root, arguments.split_root
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    output = getattr(arguments, "output", None)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
