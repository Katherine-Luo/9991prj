from __future__ import annotations

from pathlib import Path

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config


SOURCE = Path("configs/experiments/baseline_v2_p7_mixed_cem_h200.yaml")
RESOLVED = Path("configs/experiments/baseline_v2_p7_mixed_cem_h200.resolved.yaml")
DIGEST = Path("configs/experiments/baseline_v2_p7_mixed_cem_h200.sha256")


def test_p7_execution_supplement_is_canonical_and_frozen() -> None:
    source = load_config(SOURCE)
    assert RESOLVED.read_bytes() == canonical_yaml(source)
    assert DIGEST.read_text(encoding="ascii") == f"{compute_config_sha256(source)}\n"


def test_p7_declares_project_specific_mixed_type_extension() -> None:
    config = load_config(SOURCE)
    declaration = config["method_declaration"]
    assert declaration["label"] == (
        "A project-specific mixed-type extension of the original CEM."
    )
    assert declaration["original_cem_elements"] == [
        "sample_conditioned_active_inactive_embeddings",
        "shared_scoring_functions",
        "concept_intervention",
    ]
    architecture = config["project_preregistered"]["architecture"]
    assert architecture["state_tables"] == "forbidden"
    assert architecture["dense_feature_bypass"] is False
    assert architecture["independent_binary_head"] is False


def test_p7_dynamic_states_and_mixed_task_interface_are_fixed() -> None:
    preregistered = load_config(SOURCE)["project_preregistered"]
    architecture = preregistered["architecture"]
    assert architecture["mixed_embedding_shape"] == [8, 16]
    assert architecture["flattened_task_input_size"] == 128
    assert architecture["task_head"] == {
        "type": "unconstrained_linear",
        "input_size": 128,
        "output_size": 1,
        "output_activation": "none",
    }
    states = preregistered["dynamic_states"]
    assert states["sample_conditioned_source"] == "encoder_feature_h_x"
    assert states["continuous"]["scorer"] == "shared_linear_32_to_1"
    assert states["categorical"]["scorer"] == (
        "shared_linear_16_to_1_across_all_categorical_states"
    )


def test_p7_intervention_and_joint_objective_are_fixed() -> None:
    preregistered = load_config(SOURCE)["project_preregistered"]
    loss = preregistered["loss"]
    assert loss["concept_weight"] == 0.01
    assert loss["total"] == "task_loss_plus_0.01_times_concept_loss"
    intervention = preregistered["intervention"]
    assert intervention["mode"] == (
        "training_only_batch_shared_group_independent_randint"
    )
    assert intervention["group_probability"] == 0.25
    assert intervention["shared_across_batch_samples"] is True
    assert intervention["independent_across_groups"] is True
    assert intervention["random_primitive"] == "torch_randint"
    assert intervention["randint_low_inclusive"] == 0
    assert intervention["randint_high_exclusive"] == 4
    assert intervention["intervene_when_value_equals"] == 0
    assert intervention["randint_dtype"] == "int64"
    assert intervention["replace_mixture_weights_only"] is True
    assert intervention["preserve_sample_conditioned_states"] is True
    assert intervention["concept_loss_uses_unintervened_predictions"] is True
    assert intervention["validation_and_test_intervention"] is False


def test_p7_stage_a_allows_one_time_five_fold_submission() -> None:
    gate = load_config(SOURCE)["project_preregistered"]["execution_gate"]
    assert gate["stage_a_formal_training"] is False
    assert gate["submit_all_five_formal_folds_after_stage_a_pass"] is True
    assert gate["intermediate_fold_0_approval_required"] is False
    assert gate["formal_folds"] == [0, 1, 2, 3, 4]
    assert gate["formal_gpu_model"] == "H200"
