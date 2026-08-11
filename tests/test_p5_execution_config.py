from __future__ import annotations

from pathlib import Path

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config
from lidc_baseline.p5_blackbox import validate_execution_config


SOURCE = Path("configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml")
RESOLVED = Path("configs/experiments/baseline_v2_reference_training_h200_warn_only.resolved.yaml")
DIGEST = Path("configs/experiments/baseline_v2_reference_training_h200_warn_only.sha256")


def test_common_execution_config_is_canonical_and_frozen() -> None:
    source = load_config(SOURCE)
    assert RESOLVED.read_bytes() == canonical_yaml(source)
    assert DIGEST.read_text(encoding="ascii") == f"{compute_config_sha256(source)}\n"


def test_reference_reported_and_project_choices_are_separated() -> None:
    config = load_config(SOURCE)
    reported = config["reference_reported"]
    project = config["project_preregistered"]

    assert config["scope"]["applies_to"] == [
        "blackbox",
        "standard_cbm",
        "cem",
        "gam",
    ]
    assert reported["optimizer"] == "Adam"
    assert reported["learning_rate"] == 1e-4
    assert reported["epochs"] == 80
    assert reported["batch_size"] == 16
    assert reported["scheduler"] == {
        "monitor": "validation_loss",
        "bad_epochs_before_decay": 4,
        "factor": 0.9,
    }
    assert project["statement"] == (
        "Baseline-v2 project pre-registered implementation choices, "
        "not exact hyperparameters reported by the reference paper."
    )
    assert config["execution_profile"] == {
        "profile_id": "baseline-v2-formal-h200-warn-only",
        "amendment_type": "execution_reproducibility_profile",
        "formal_gpu_model": "H200",
        "applies_to_formal_training": ["blackbox", "standard_cbm", "cem", "gam"],
        "supersedes_execution_profile": "configs/experiments/baseline_v2_reference_training_h200.yaml",
        "statement": "Baseline-v2 execution/reproducibility profile amendment. H200 remains the unified formal training GPU for P5-P8; CUDA operations without deterministic implementations are recorded as warnings rather than blocking backward.",
    }


def test_exact_project_training_choices_are_pre_registered() -> None:
    project = load_config(SOURCE)["project_preregistered"]
    assert project["optimizer"] == {
        "betas": [0.9, 0.999],
        "epsilon": 1e-7,
        "weight_decay": 0.0,
    }
    assert project["batching"] == {
        "micro_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "effective_batch_size": 16,
        "drop_last": False,
        "final_partial_batch_reduction": "sample_mean",
    }
    scheduler = project["scheduler"]
    assert scheduler["min_delta"] == 1e-4
    assert scheduler["bad_epoch_counter_reset_after_decay"] is True
    assert scheduler["cooldown"] == 0
    assert scheduler["min_learning_rate"] == 0.0
    assert scheduler["checkpoint_tolerance"] == 0.0
    assert scheduler["checkpoint_tie_break"] == "earlier_epoch"

    augmentation = project["augmentation"]
    assert augmentation["order"] == [
        "axial_rotation",
        "h_axis_flip",
        "w_axis_flip",
        "z_order_reversal",
    ]
    assert augmentation["excludes_model_name"] is True
    assert augmentation["axial_rotation"]["torch_grid_sample"] == {
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": False,
        "five_dimensional_behavior": "trilinear_sampling",
    }
    assert project["precision"] == {
        "floating_point": "FP32",
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
    }
    assert project["reproducibility"]["torch_use_deterministic_algorithms"] is True
    assert project["reproducibility"]["warn_only"] is True


def test_superseded_execution_profiles_cannot_drive_formal_p5_runs() -> None:
    from pytest import raises

    with raises(ValueError, match="H200_PROFILE_MISMATCH"):
        validate_execution_config("configs/experiments/baseline_v2_reference_training.yaml")
    with raises(ValueError, match="H200_PROFILE_MISMATCH"):
        validate_execution_config("configs/experiments/baseline_v2_reference_training_h200.yaml")
