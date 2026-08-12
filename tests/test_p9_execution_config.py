from __future__ import annotations

from pathlib import Path

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config


SOURCE = Path("configs/experiments/baseline_v2_p9_evaluation_h200.yaml")
RESOLVED = Path("configs/experiments/baseline_v2_p9_evaluation_h200.resolved.yaml")
DIGEST = Path("configs/experiments/baseline_v2_p9_evaluation_h200.sha256")


def _project() -> dict[str, object]:
    return load_config(SOURCE)["project_preregistered"]


def test_p9_execution_supplement_is_canonical_and_frozen() -> None:
    source = load_config(SOURCE)
    assert RESOLVED.read_bytes() == canonical_yaml(source)
    assert DIGEST.read_text(encoding="ascii") == f"{compute_config_sha256(source)}\n"


def test_p9_inputs_are_read_only_and_test_is_not_recommitted() -> None:
    frozen = _project()["frozen_inputs"]
    assert frozen["model_order"] == [
        "blackbox",
        "standard_cbm",
        "mixed_cem",
        "learned_softmax_gam",
    ]
    assert frozen["access"] == "read_only"
    assert frozen["retraining"] == "forbidden"
    assert frozen["second_committed_test_evaluation"] == "forbidden"
    assert frozen["artifact_rewrite"] == "forbidden"


def test_p9_youden_uses_validation_extremes_only() -> None:
    secondary = _project()["task_evaluation"]["secondary_extreme"]
    assert secondary["low_condition"] == "mean_malignancy_le_2"
    assert secondary["high_condition"] == "mean_malignancy_ge_4"
    assert secondary["middle_samples_in_primary_regression"] is True
    assert secondary["middle_samples_in_threshold_selection"] is False
    assert secondary["youden_selection_partition"] == (
        "fold_specific_validation_extreme_subset_only"
    )
    assert secondary["threshold_candidates"] == "finite_validation_extreme_scores"
    assert secondary["tie_break"] == "largest_threshold"


def test_p9_intervention_and_delta_signs_are_explicit() -> None:
    intervention = _project()["intervention"]
    assert intervention["permutations_per_fold"] == 100
    assert intervention["shared_order_across_models"] is True
    assert intervention["permutation_seed_excludes_model"] is True
    assert intervention["error_first"]["malignancy_target_in_ranking"] == "forbidden"
    assert intervention["k_zero"]["exact_baseline_reproduction"] == "required"
    assert intervention["delta_iMAE"] == "baseline_MAE_minus_iMAE"
    assert intervention["delta_iAUC"] == "iAUC_minus_baseline_AUROC"
    assert intervention["positive_delta_means_improvement"] is True


def test_p9_gradcam_and_raw_shard_contract_is_frozen() -> None:
    project = _project()
    gradcam = project["gradcam"]
    assert gradcam["target_layer"] == (
        "encoder.denseblock4.denselayer16.layers.conv2"
    )
    assert gradcam["gradient_weights"] == "spatial_mean"
    assert gradcam["post_combination_activation"] == "relu"
    assert gradcam["upsample"] == {
        "mode": "trilinear",
        "output_shape": [64, 64, 64],
        "align_corners": False,
    }
    assert gradcam["maps"]["dtype"] == "float32"
    assert gradcam["maps"]["normalization"] == "none"
    assert gradcam["maps"]["all_zero_status"] == "undefined"
    storage = project["spatial_storage"]
    assert storage["format"] == "parquet"
    assert storage["compression"] == "zstd"
    assert storage["nodules_per_shard"] == 16
    assert storage["map_encoding"] == "raw_little_endian_float32_bytes"


def test_p9_occlusion_preserves_both_faithfulness_quantities() -> None:
    occlusion = _project()["occlusion"]
    assert occlusion["valid_map_voxels"] == 26215
    assert occlusion["random_masks_per_target"] == 20
    assert occlusion["random_seed_excludes_model"] is True
    assert occlusion["output_sensitivity"] == (
        "abs_score_occluded_minus_score_original"
    )
    assert occlusion["error_increase"] == (
        "abs_score_occluded_minus_target_normalized_minus_abs_score_original_minus_target_normalized"
    )
    assert occlusion["output_sensitivity_is_prediction_worsening_evidence"] is False
    assert occlusion["positive_error_increase_means_prediction_worsened"] is True
    assert occlusion["negative_error_increase_means_prediction_improved"] is True
    assert occlusion["retain_individual_random_output_sensitivity_values"] == 20
    assert occlusion["retain_individual_random_error_increase_values"] == 20
    assert occlusion["random_aggregates"] == ["mean", "sd", "median", "min", "max"]


def test_p9_stage_a_and_explicit_formal_gate_are_frozen() -> None:
    project = _project()
    stage_a = project["stage_a"]
    assert stage_a["fold"] == 0
    assert stage_a["partition"] == "validation_only"
    assert stage_a["reads_test"] is False
    assert stage_a["all_gradcam_target_paths"] == 28
    assert stage_a["true_occlusion_batch_size"] == 16
    assert stage_a["peak_reserved_fraction_limit"] == 0.85
    assert stage_a["projected_slowest_model_fold_hours_limit"] == 8.8
    assert stage_a["scratch_free_space_to_projected_peak_ratio_minimum"] == 1.2
    formal = project["formal_spatial_execution"]
    assert formal["approval_environment_variable"] == "P9_SPATIAL_APPROVED"
    assert formal["default_approval_value"] == 0
    assert formal["required_approval_value"] == 1
    assert formal["user_approval_after_stage_a_required"] is True
    assert formal["submit_once_after_explicit_user_approval"] is True
    assert formal["jobs"] == 20
    assert formal["queue"] == "csegpu12"
    assert formal["gpu_model"] == "H200"
    assert formal["walltime"] == "11:00:00"


def test_p9_bootstrap_is_patient_clustered_and_all_pairs_are_retained() -> None:
    bootstrap = _project()["bootstrap"]
    assert bootstrap["draws"] == 2000
    assert bootstrap["unit"] == "patient_cluster"
    assert bootstrap["primary_shared_draws_across_models"] is True
    assert bootstrap["secondary_extreme_patients"] == 578
    assert bootstrap["secondary_single_class_draw"] == "discard_and_redraw"
    assert bootstrap["secondary_valid_draws_required"] == 2000
    assert bootstrap["paired_model_pairs"] == 6
    assert bootstrap["paired_mae_difference"] == "MAE_A_minus_MAE_B"
    assert bootstrap["paired_auroc_difference"] == "AUROC_B_minus_AUROC_A"
