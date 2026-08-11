from __future__ import annotations

from pathlib import Path

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config


SOURCE = Path("configs/experiments/baseline_v2_p6_standard_cbm_h200.yaml")
RESOLVED = Path("configs/experiments/baseline_v2_p6_standard_cbm_h200.resolved.yaml")
DIGEST = Path("configs/experiments/baseline_v2_p6_standard_cbm_h200.sha256")


def test_p6_execution_supplement_is_canonical_and_frozen() -> None:
    source = load_config(SOURCE)
    assert RESOLVED.read_bytes() == canonical_yaml(source)
    assert DIGEST.read_text(encoding="ascii") == f"{compute_config_sha256(source)}\n"


def test_p6_canonical_vector_uses_activated_predictions_only() -> None:
    config = load_config(SOURCE)["project_preregistered"]["concept_predictor"]
    assert config["head_type"] == "independent_linear_no_hidden_layer"
    assert config["group_order"] == [
        "subtlety",
        "internalStructure",
        "calcification",
        "sphericity",
        "margin",
        "lobulation",
        "spiculation",
        "texture",
    ]
    vector = config["canonical_task_vector"]
    assert vector["dimension"] == 16
    assert vector["source"] == "activated_predictions"
    assert vector["preactivation_logits_as_task_input"] is False
    assert vector["malignancy_as_input_concept"] is False
    assert config["logits"] == {
        "retain_for_audit_and_future_gradcam": True,
        "task_head_input": False,
    }


def test_p6_cache_and_task_stage_are_leakage_safe() -> None:
    sequential = load_config(SOURCE)["project_preregistered"]["sequential_training"]
    assert sequential["concept_stage"]["epochs"] == 80
    assert sequential["task_stage"]["epochs"] == 80
    cache = sequential["cache_gate"]
    assert cache["generated_before_task_training"] == ["train", "validation"]
    assert cache["forbidden_before_task_best"] == ["test"]
    assert cache["values"] == "activated_predicted_concepts_float32"
    assert cache["ground_truth_concepts_as_task_features"] == "forbidden"
    task = sequential["task_stage"]
    assert task["input"] == "frozen_activated_predicted_concepts"
    assert task["ground_truth_concepts"] == "forbidden"
    assert task["reads_images"] is False
    assert task["reads_encoder_features"] is False
    assert sequential["test_transaction"]["generate_test_concepts_after_task_best_only"] is True


def test_p6_stage_a_allows_one_time_five_fold_submission() -> None:
    gate = load_config(SOURCE)["project_preregistered"]["execution_gate"]
    assert gate["stage_a_formal_training"] is False
    assert gate["submit_all_five_formal_folds_after_stage_a_pass"] is True
    assert gate["intermediate_fold_0_approval_required"] is False
    assert gate["formal_folds"] == [0, 1, 2, 3, 4]
    assert gate["formal_gpu_model"] == "H200"
