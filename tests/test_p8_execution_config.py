from __future__ import annotations

from pathlib import Path

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config


SOURCE = Path("configs/experiments/baseline_v2_p8_gam_h200.yaml")
RESOLVED = Path("configs/experiments/baseline_v2_p8_gam_h200.resolved.yaml")
DIGEST = Path("configs/experiments/baseline_v2_p8_gam_h200.sha256")


def test_p8_execution_supplement_is_canonical_and_frozen() -> None:
    source = load_config(SOURCE)
    assert RESOLVED.read_bytes() == canonical_yaml(source)
    assert DIGEST.read_text(encoding="ascii") == f"{compute_config_sha256(source)}\n"


def test_p8_task_path_is_predicted_concept_only_and_unbounded() -> None:
    architecture = load_config(SOURCE)["project_preregistered"]["architecture"]
    assert architecture["task_input"] == "activated_predicted_concepts_only"
    assert architecture["preactivation_logits_task_input"] == "forbidden"
    assert architecture["ground_truth_concepts_task_input"] == "forbidden"
    assert architecture["dense_feature_bypass"] is False
    assert architecture["independent_binary_head"] is False
    assert architecture["task_output_activation"] == "none"
    assert architecture["task_output_constraint"] == "unbounded"


def test_p8_has_five_learned_softmax_experts_per_group() -> None:
    gam = load_config(SOURCE)["project_preregistered"]["gam"]
    assert gam["groups"] == 8
    assert gam["experts_per_group"] == 5
    assert gam["total_independent_experts"] == 40
    assert gam["expert_architecture"] == {
        "hidden_sizes": [32, 16],
        "hidden_activation": "relu",
        "output_size": 1,
        "output_activation": "none",
        "output_type": "linear_scalar",
    }
    assert gam["concept_local_inputs_only"] is True
    assert gam["cross_concept_inputs"] == "forbidden"
    assert gam["simple_expert_average"] == "forbidden"


def test_p8_alpha_and_joint_loss_are_frozen() -> None:
    preregistered = load_config(SOURCE)["project_preregistered"]
    alpha = preregistered["gam"]["alpha"]
    assert alpha["scope"] == "fold_level_global_trainable_parameter"
    assert alpha["sample_conditioned"] is False
    assert alpha["logits_initialization"] == "all_zeros"
    assert alpha["weights_transform"] == "softmax"
    assert alpha["initial_weight_per_expert"] == 0.2
    loss = preregistered["loss"]
    assert loss["concept_weight"] == 1.0
    assert loss["total"] == "task_loss_plus_concept_loss"
    assert preregistered["intervention"] == {
        "training": False,
        "validation": False,
        "test": False,
        "reserved_phase": "P9",
    }


def test_p8_stage_a_allows_one_time_five_fold_submission() -> None:
    gate = load_config(SOURCE)["project_preregistered"]["execution_gate"]
    assert gate["stage_a_formal_training"] is False
    assert gate["true_batch_size"] == 16
    assert gate["alpha_gradient_required_per_group"] is True
    assert gate["alpha_update_required_per_group"] is True
    assert gate["submit_all_five_formal_folds_after_stage_a_pass"] is True
    assert gate["intermediate_fold_0_approval_required"] is False
    assert gate["formal_folds"] == [0, 1, 2, 3, 4]
    assert gate["formal_gpu_model"] == "H200"


def test_p8_initialization_domains_and_reconstruction_are_frozen() -> None:
    preregistered = load_config(SOURCE)["project_preregistered"]
    initialization = preregistered["deterministic_initialization"]
    assert initialization["concept_head_seed_material"] == (
        "Baseline-v2/P8/gam-concept-head/<group> || fold_seed"
    )
    assert initialization["subnetwork_seed_material"] == (
        "Baseline-v2/P8/gam-subnetwork/<group>/<expert_index> || fold_seed"
    )
    assert initialization["alpha_logits"] == "all_zeros"
    assert initialization["global_raw_bias"] == "zero"
    contributions = preregistered["contributions"]
    assert contributions["normalized_reconstruction_tolerance"] == 1.0e-6
    assert contributions["rating_reconstruction_tolerance"] == 1.0e-6
