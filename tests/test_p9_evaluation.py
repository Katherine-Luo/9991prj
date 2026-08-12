from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from lidc_baseline import p9_evaluation as p9


def test_unclipped_regression_metrics_preserve_range_and_out_of_bounds() -> None:
    report = p9.regression_metrics([-0.1, 0.5, 1.2], [0.0, 0.25, 1.0])
    assert report["prediction_range_normalized"] == [-0.1, 1.2]
    assert report["below_zero_rate"] == pytest.approx(1 / 3)
    assert report["above_one_rate"] == pytest.approx(1 / 3)
    assert report["prediction_range_1_to_5"] == pytest.approx([0.6, 5.8])
    assert report["below_one_rate"] == pytest.approx(1 / 3)
    assert report["above_five_rate"] == pytest.approx(1 / 3)
    assert report["original_scale_mae"] == pytest.approx(
        4 * np.mean([0.1, 0.25, 0.2])
    )


def test_youden_uses_validation_extremes_only_and_largest_tie_threshold() -> None:
    result = p9.select_youden_threshold(
        [0.1, 0.4, 0.6, 0.9, 100.0],
        [1.0, 2.0, 4.0, 5.0, 3.0],
    )
    assert result["threshold"] == 0.6
    assert result["validation_middle_excluded_count"] == 1
    assert result["validation_extreme_sample_count"] == 4


def test_youden_tie_break_prefers_largest_finite_threshold() -> None:
    result = p9.select_youden_threshold([0.1, 0.2, 0.2, 0.3], [1, 2, 4, 5])
    assert np.isfinite(result["threshold"])
    assert result["threshold"] == 0.3


def test_secondary_metrics_exclude_middle_targets() -> None:
    report = p9.secondary_metrics([0.1, 0.9, 0.0], [1.0, 5.0, 3.0], 0.5)
    assert report["sample_count"] == 2
    assert report["accuracy"] == 1.0
    assert report["auroc"] == 1.0


def test_continuous_and_categorical_concept_metrics_preserve_ties() -> None:
    continuous = p9.continuous_concept_metrics([0.0, 1.0], [0.0, 0.5])
    assert continuous["mae"] == 0.25
    categorical = p9.categorical_concept_metrics(
        [[0.8, 0.2], [0.4, 0.6]],
        [[1.0, 0.0], [0.5, 0.5]],
        [False, True],
    )
    assert categorical["soft_sample_count"] == 2
    assert categorical["hard_sample_count"] == 1
    assert categorical["true_tie_count"] == 1
    with pytest.raises(ValueError, match="TRUE_TIE_MISMATCH"):
        p9.categorical_concept_metrics(
            [[0.8, 0.2], [0.4, 0.6]],
            [[1.0, 0.0], [0.5, 0.5]],
            [False, False],
        )


def test_train_only_centering_reconstructs_both_scales() -> None:
    train = np.arange(32, dtype=float).reshape(4, 8) / 10
    evaluation = np.arange(16, dtype=float).reshape(2, 8) / 10
    bias = 0.25
    score = bias + evaluation.sum(axis=1)
    result = p9.center_contributions(
        train,
        evaluation,
        bias,
        score,
        train_uids=["a", "b", "c", "d"],
        expected_train_uids=["d", "c", "b", "a"],
    )
    assert result["normalized_reconstruction_max_abs_error"] <= 1e-12
    assert result["rating_reconstruction_max_abs_error"] <= 1e-12
    assert np.allclose(result["train_group_means"], train.mean(axis=0))
    with pytest.raises(ValueError, match="TRAIN_UID_MEMBERSHIP_MISMATCH"):
        p9.center_contributions(
            np.empty((0, 8)),
            evaluation,
            bias,
            score,
            train_uids=[],
            expected_train_uids=[],
        )
    with pytest.raises(ValueError, match="TRAIN_UID_MEMBERSHIP_MISMATCH"):
        p9.center_contributions(
            train,
            evaluation,
            bias,
            score,
            train_uids=["a", "b", "c", "x"],
            expected_train_uids=["a", "b", "c", "d"],
        )


def test_shared_permutations_are_model_independent_and_reproducible() -> None:
    first = p9.shared_intervention_permutations(20260808, 2)
    second = p9.shared_intervention_permutations(20260808, 2)
    other = p9.shared_intervention_permutations(20260808, 3)
    assert first.shape == (100, 8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert all(set(row) == set(range(8)) for row in first)


def test_error_first_uses_absolute_and_total_variation_with_canonical_ties() -> None:
    continuous_prediction = {
        group: 0.5 for group in ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
    }
    continuous_target = dict(continuous_prediction)
    continuous_target["subtlety"] = 0.0
    categorical_prediction = {
        "internalStructure": [1.0, 0.0],
        "calcification": [0.0, 1.0],
    }
    categorical_target = {
        "internalStructure": [0.5, 0.5],
        "calcification": [0.5, 0.5],
    }
    order = p9.error_first_order(
        continuous_prediction,
        continuous_target,
        categorical_prediction,
        categorical_target,
    )
    assert order[:3] == ("subtlety", "internalStructure", "calcification")


def test_intervention_delta_signs_positive_for_improvement() -> None:
    delta = p9.intervention_deltas(0.5, 0.4, 0.7, 0.8)
    assert delta == {"Delta_iMAE": pytest.approx(0.1), "Delta_iAUC": pytest.approx(0.1)}


def test_patient_bootstrap_is_deterministic_and_secondary_redraws_single_class() -> None:
    primary = p9.patient_cluster_bootstrap_draws(["p2", "p1", "p1"], 7)
    assert len(primary) == 2000
    assert all(len(draw) == 2 for draw in primary)
    assert p9.bootstrap_draw_sha256(primary) == p9.bootstrap_draw_sha256(
        p9.patient_cluster_bootstrap_draws(["p1", "p2"], 7)
    )
    secondary = p9.secondary_patient_bootstrap_draws(
        {"p1": [0, 1], "p2": [0]}, 7
    )
    assert len(secondary) == 2000
    assert all("p1" in set(draw) for draw in secondary)


def test_all_six_model_pairs_and_directional_differences() -> None:
    pairs = p9.all_model_pairs()
    assert len(pairs) == 6
    assert set(pairs) == set(itertools.combinations(p9.MODEL_ORDER, 2))
    values = {model: float(index) for index, model in enumerate(p9.MODEL_ORDER)}
    mae = p9.paired_differences(values, metric="mae")
    auroc = p9.paired_differences(values, metric="auroc")
    assert mae["blackbox__standard_cbm"] == -1.0
    assert auroc["blackbox__standard_cbm"] == 1.0


def _oof_frame() -> pd.DataFrame:
    rows = []
    fold_counts = p9.FOLD_TEST_COUNTS
    patient_index = 0
    for fold, count in enumerate(fold_counts):
        for index in range(count):
            rows.append(
                {
                    "nodule_uid": f"n{len(rows):04d}",
                    "patient_key": f"p{patient_index % p9.EXPECTED_PATIENTS:03d}",
                    "fold_index": fold,
                    "malignancy_raw_score": 0.5,
                    "target_normalized": 0.5,
                    "target_1_to_5": 3.0,
                }
            )
            patient_index += 1
    frame = pd.DataFrame(rows)
    # Make every synthetic patient fold-isolated while retaining exactly 868 keys.
    partitions = []
    for fold, group in frame.groupby("fold_index", sort=True):
        keys = [f"f{fold}_p{index:03d}" for index in range([158, 165, 178, 180, 187][fold])]
        partitions.extend(keys[index % len(keys)] for index in range(len(group)))
    frame["patient_key"] = partitions
    assert frame["patient_key"].nunique() == p9.EXPECTED_PATIENTS
    return frame


def test_oof_equality_checks_counts_targets_and_patient_isolation() -> None:
    reference = _oof_frame()
    frames = {model: reference.copy() for model in p9.MODEL_ORDER}
    assert p9.verify_oof_equality(frames)["status"] == "PASS"
    frames["mixed_cem"].loc[0, "target_normalized"] = 0.4
    with pytest.raises(ValueError, match="TARGET_MISMATCH"):
        p9.verify_oof_equality(frames)


def test_oof_equality_binds_reference_to_p4_fold_membership() -> None:
    reference = _oof_frame()
    frames = {model: reference.copy() for model in p9.MODEL_ORDER}
    expected = {
        fold: tuple(reference.loc[reference["fold_index"] == fold, "nodule_uid"])
        for fold in range(5)
    }
    assert p9.verify_oof_equality(frames, expected)["status"] == "PASS"
    changed = dict(expected)
    changed[0] = ("not-a-real-nodule", *changed[0][1:])
    with pytest.raises(ValueError, match="P4_TEST_MEMBERSHIP_MISMATCH"):
        p9.verify_oof_equality(frames, changed)


def test_build_task_results_uses_fold_youden_not_fixed_half(monkeypatch: pytest.MonkeyPatch) -> None:
    test_rows = []
    validation_rows = []
    for fold in range(5):
        test_rows.extend(
            [
                {
                    "nodule_uid": f"test-{fold}-low",
                    "patient_key": f"pl-{fold}",
                    "fold_index": fold,
                    "malignancy_raw_score": 0.3,
                    "target_normalized": 0.0,
                    "target_1_to_5": 1.0,
                },
                {
                    "nodule_uid": f"test-{fold}-high",
                    "patient_key": f"ph-{fold}",
                    "fold_index": fold,
                    "malignancy_raw_score": 0.7,
                    "target_normalized": 1.0,
                    "target_1_to_5": 5.0,
                },
            ]
        )
        validation_rows.extend(
            [
                {
                    "nodule_uid": f"val-{fold}-low",
                    "fold_index": fold,
                    "malignancy_raw_score": 0.6,
                    "target_1_to_5": 1.0,
                },
                {
                    "nodule_uid": f"val-{fold}-high",
                    "fold_index": fold,
                    "malignancy_raw_score": 0.8,
                    "target_1_to_5": 5.0,
                },
                {
                    "nodule_uid": f"val-{fold}-middle",
                    "fold_index": fold,
                    "malignancy_raw_score": -100.0,
                    "target_1_to_5": 3.0,
                },
            ]
        )
    test_frame = pd.DataFrame(test_rows)
    validation_frame = pd.DataFrame(validation_rows)
    monkeypatch.setattr(
        p9, "load_oof_frames", lambda _: {model: test_frame.copy() for model in p9.MODEL_ORDER}
    )
    monkeypatch.setattr(
        p9,
        "load_validation_frames",
        lambda _: {model: validation_frame.copy() for model in p9.MODEL_ORDER},
    )
    monkeypatch.setattr(
        p9,
        "load_expected_partition_uids",
        lambda _: {
            "test": {},
            "validation": {
                fold: tuple(
                    validation_frame.loc[
                        validation_frame["fold_index"] == fold, "nodule_uid"
                    ]
                )
                for fold in range(5)
            },
        },
    )
    monkeypatch.setattr(p9, "verify_oof_equality", lambda *_: {"status": "PASS"})
    report = p9.build_task_results("unused", "unused", "unused")
    thresholds = [
        fold["threshold_selection"]["threshold"]
        for fold in report["models"]["blackbox"]["folds"]
    ]
    assert thresholds == [0.8] * 5
    assert report["models"]["blackbox"]["pooled_secondary"]["accuracy"] == 0.5
    assert report["models"]["blackbox"]["pooled_secondary"]["fixed_0_5_sensitivity"] == 1.0


def test_validation_threshold_inputs_are_bound_to_p4_membership() -> None:
    rows = []
    expected = {}
    for fold in range(5):
        fold_rows = [
            {
                "nodule_uid": f"v-{fold}-low",
                "fold_index": fold,
                "malignancy_raw_score": 0.2,
                "target_1_to_5": 1.0,
            },
            {
                "nodule_uid": f"v-{fold}-high",
                "fold_index": fold,
                "malignancy_raw_score": 0.8,
                "target_1_to_5": 5.0,
            },
        ]
        rows.extend(fold_rows)
        expected[fold] = tuple(row["nodule_uid"] for row in fold_rows)
    frame = pd.DataFrame(rows)
    frames = {model: frame.copy() for model in p9.MODEL_ORDER}
    p9.verify_validation_membership(frames, expected)
    frames["standard_cbm"].loc[0, "nodule_uid"] = "test-or-other-uid"
    with pytest.raises(ValueError, match="P4_VALIDATION_MEMBERSHIP_MISMATCH"):
        p9.verify_validation_membership(frames, expected)
