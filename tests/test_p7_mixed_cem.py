from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import lidc_baseline.p7_mixed_cem as p7
from lidc_baseline.config import load_config
from lidc_baseline.p5_blackbox import validate_execution_config
from lidc_baseline.p6_standard_cbm import ConceptRecord
from lidc_baseline.p6_standard_cbm import CONCEPT_GROUP_ORDER, CONTINUOUS_CONCEPTS
from lidc_baseline.p7_mixed_cem import (
    MixedTypeCEM,
    apply_intervention_weights,
    batch_shared_intervention_mask,
    build_deterministic_cem_components,
    cem_losses,
    evaluate_test_once,
    run_cem_epoch,
    task_predictions_and_contributions,
    train_fold,
    validate_p7_execution_config,
)


class TinyEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(2, 3, 4), keepdim=True)
        return pooled.repeat(1, 1024, 2, 2, 2)


def targets(batch_size: int) -> dict[str, object]:
    concepts: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        if group == "internalStructure":
            concepts[group] = torch.tensor([[0.1, 0.2, 0.3, 0.4]]).repeat(batch_size, 1)
        elif group == "calcification":
            concepts[group] = torch.tensor(
                [[0.05, 0.1, 0.15, 0.2, 0.2, 0.3]]
            ).repeat(batch_size, 1)
        else:
            concepts[group] = torch.full((batch_size, 1), 0.35)
    return {
        "concepts": concepts,
        "malignancy": torch.full((batch_size, 1), 0.6),
    }


def build_model(fold_seed: int = 20260808) -> torch.nn.Module:
    components, _metadata = build_deterministic_cem_components(fold_seed)
    return MixedTypeCEM.build(TinyEncoder(), components)


def concept_records(tmp_path: Path, count: int) -> list[ConceptRecord]:
    result = []
    for index in range(count):
        roi_path = tmp_path / f"roi_{index}.npz"
        image = np.full((1, 64, 64, 64), index / max(count, 1), dtype=np.float32)
        np.savez(roi_path, image=image)
        result.append(
            ConceptRecord(
                nodule_uid=f"nodule-{index}",
                patient_key=f"patient-{index}",
                roi_path=roi_path,
                target_normalized=0.2 + 0.1 * index,
                target_1_to_5=1.8 + 0.4 * index,
                extreme_binary_eligible=index % 2 == 0,
                extreme_binary_label=0 if index % 2 == 0 else None,
                continuous_targets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
                internal_structure_target=(0.1, 0.2, 0.3, 0.4),
                calcification_target=(0.05, 0.1, 0.15, 0.2, 0.2, 0.3),
                valid_reader_counts=(2, 2, 2, 2, 2, 2, 2, 2),
                categorical_ties=(False, False),
            )
        )
    return result


def test_execution_config_enforces_mixed_cem_identity() -> None:
    config, digest = validate_p7_execution_config()
    assert config["method_declaration"]["label"] == (
        "A project-specific mixed-type extension of the original CEM."
    )
    assert digest == "60e84612eec0ce60b0d17284f6888ddea3627778ab39bcee4c0c6ee3b0c63a2c"


def test_dynamic_states_shapes_probabilities_and_shared_scorers() -> None:
    model = build_model()
    h_x = torch.stack((torch.zeros(1024), torch.ones(1024)))
    outputs = model.forward_from_features(h_x)
    assert model.continuous_scorer.in_features == 32
    assert model.categorical_scorer.in_features == 16
    assert outputs["flat_mixed_embedding"].shape == (2, 128)
    for group in CONCEPT_GROUP_ORDER:
        states = outputs["states"][group]
        probabilities = outputs["activated"][group]
        if group in CONTINUOUS_CONCEPTS:
            assert states.shape == (2, 2, 16)
            assert probabilities.shape == (2, 1)
            assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
        else:
            classes = 4 if group == "internalStructure" else 6
            assert states.shape == (2, classes, 16)
            assert probabilities.shape == (2, classes)
            assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
        assert not torch.equal(states[0], states[1])
    assert not any("state_table" in name for name, _parameter in model.named_parameters())


def test_fixed_probabilities_with_changed_features_change_dynamic_states() -> None:
    model = build_model()
    h_a = torch.zeros(1, 1024)
    h_b = torch.ones(1, 1024)
    a = model.states_and_probabilities(h_a)
    b = model.states_and_probabilities(h_b)
    fixed = OrderedDict(
        (group, a["activated"][group]) for group in CONCEPT_GROUP_ORDER
    )
    mixed_a = model.mix_states(a["states"], fixed)
    mixed_b = model.mix_states(b["states"], fixed)
    assert any(
        not torch.equal(mixed_a[group], mixed_b[group])
        for group in CONCEPT_GROUP_ORDER
    )


def test_joint_loss_is_task_plus_point_zero_one_equal_group_loss() -> None:
    model = build_model()
    outputs = model.forward_from_features(torch.randn(3, 1024))
    losses = cem_losses(outputs, targets(3))
    assert tuple(losses["group_losses"]) == CONCEPT_GROUP_ORDER
    expected_concept = torch.stack(tuple(losses["group_losses"].values())).mean()
    assert torch.allclose(losses["concept_loss"], expected_concept)
    assert torch.allclose(
        losses["total_loss"], losses["task_loss"] + 0.01 * losses["concept_loss"]
    )


def test_randint_intervention_mask_is_batch_shared_and_resume_deterministic() -> None:
    arguments = {
        "base_seed": 20260808,
        "fold_index": 2,
        "epoch_index": 7,
        "batch_index": 11,
    }
    first = batch_shared_intervention_mask(**arguments)
    second = batch_shared_intervention_mask(**arguments)
    changed = batch_shared_intervention_mask(**{**arguments, "batch_index": 12})
    assert first.dtype == torch.bool
    assert first.shape == (8,)
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)


def test_randint_intervention_decision_rates_match_preregistered_gates() -> None:
    masks = torch.stack(
        [
            batch_shared_intervention_mask(
                base_seed=20260808,
                fold_index=0,
                epoch_index=batch_index // 128,
                batch_index=batch_index,
            )
            for batch_index in range(4096)
        ]
    ).float()
    assert 0.24 <= float(masks.mean()) <= 0.26
    assert torch.all((masks.mean(dim=0) >= 0.23) & (masks.mean(dim=0) <= 0.27))


def test_intervention_replaces_weights_only_and_preserves_sample_states() -> None:
    model = build_model()
    h_x = torch.randn(2, 1024)
    generated = model.states_and_probabilities(h_x)
    mask = torch.tensor([True, False, True, False, True, False, True, False])
    concept_targets = targets(2)["concepts"]
    effective = apply_intervention_weights(generated["activated"], concept_targets, mask)
    for index, group in enumerate(CONCEPT_GROUP_ORDER):
        expected = concept_targets[group] if bool(mask[index]) else generated["activated"][group]
        assert effective[group] is expected
    intervened = model.forward_from_features(
        h_x,
        intervention_targets=concept_targets,
        intervention_mask=mask,
    )
    for group in CONCEPT_GROUP_ORDER:
        assert torch.equal(intervened["states"][group], generated["states"][group])
    report = task_predictions_and_contributions(model, intervened)
    assert report["normalized_reconstruction_max_abs_error"] <= 1e-6
    assert report["rating_reconstruction_max_abs_error"] <= 1e-6


def test_initialization_is_isolated_reproducible_and_fold_specific() -> None:
    torch.manual_seed(123)
    _ = torch.rand(9)
    _first, first = build_deterministic_cem_components(20260808)
    _ = torch.rand(17)
    _second, second = build_deterministic_cem_components(20260808)
    _third, third = build_deterministic_cem_components(20260809)
    assert first["combined_cem_initialization_sha256"] == second[
        "combined_cem_initialization_sha256"
    ]
    assert first["combined_cem_initialization_sha256"] != third[
        "combined_cem_initialization_sha256"
    ]
    assert first["state_generator_initialization_sha256"] == second[
        "state_generator_initialization_sha256"
    ]


def test_task_output_is_unconstrained_and_contributions_reconstruct() -> None:
    model = build_model()
    with torch.no_grad():
        model.task_head.weight.fill_(0.01)
        model.task_head.bias.fill_(1.25)
    outputs = model.forward_from_features(torch.randn(4, 1024))
    report = task_predictions_and_contributions(model, outputs)
    assert torch.any(report["malignancy_raw_score"] > 1.0)
    assert torch.equal(
        report["malignancy_raw_score"], report["malignancy_score_normalized"]
    )
    assert report["normalized_reconstruction_max_abs_error"] <= 1e-6
    assert report["rating_reconstruction_max_abs_error"] <= 1e-6


def test_invalid_partial_intervention_is_rejected() -> None:
    model = build_model()
    with pytest.raises(ValueError, match="P7_PARTIAL_INTERVENTION_ARGUMENTS"):
        model.forward_from_features(
            torch.zeros(1, 1024),
            intervention_targets=targets(1)["concepts"],
        )


def test_epoch_uses_all_samples_including_partial_batch(tmp_path: Path) -> None:
    model = build_model()
    records = concept_records(tmp_path, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_report = run_cem_epoch(
        model,
        records,
        torch.device("cpu"),
        optimizer=optimizer,
        base_seed=20260808,
        fold_index=0,
        epoch_index=0,
        batch_size=2,
        num_workers=0,
    )
    validation_report = run_cem_epoch(
        model,
        records,
        torch.device("cpu"),
        optimizer=None,
        base_seed=20260808,
        fold_index=0,
        epoch_index=0,
        batch_size=2,
        num_workers=0,
    )
    assert train_report["sample_count"] == 3
    assert train_report["batch_count"] == 2
    assert validation_report["sample_count"] == 3
    assert all(
        count == 0
        for count in validation_report["intervention_decision_counts"].values()
    )


def test_prediction_artifact_preserves_states_and_detects_tampering(
    tmp_path: Path,
) -> None:
    model = build_model()
    records = concept_records(tmp_path, 2)
    rows = p7._prediction_rows(
        model,
        records,
        torch.device("cpu"),
        batch_size=2,
        num_workers=0,
    )
    frame = pd.DataFrame(rows)
    frame["fold_index"] = 0
    p7._validate_test_predictions(frame, records, {"fold_index": 0}, model)
    tampered = frame.copy()
    tampered.loc[0, "subtlety_states"] = json.dumps([[0.0] * 16, [0.0] * 16])
    with pytest.raises(ValueError, match="P7_TEST_STATE_MIXTURE_MISMATCH"):
        p7._validate_test_predictions(tampered, records, {"fold_index": 0}, model)
    invalid_bool = frame.copy()
    invalid_bool["extreme_binary_eligible"] = invalid_bool[
        "extreme_binary_eligible"
    ].astype(object)
    invalid_bool.loc[0, "extreme_binary_eligible"] = "False"
    with pytest.raises(ValueError, match="P7_TEST_EXTREME_ELIGIBILITY_INVALID"):
        p7._validate_test_predictions(invalid_bool, records, {"fold_index": 0}, model)
    invalid_count = frame.copy()
    invalid_count["subtlety_valid_reader_count"] = invalid_count[
        "subtlety_valid_reader_count"
    ].astype(object)
    invalid_count.loc[0, "subtlety_valid_reader_count"] = 2.9
    with pytest.raises(ValueError, match="P7_TEST_VALID_READER_COUNT_INVALID"):
        p7._validate_test_predictions(invalid_count, records, {"fold_index": 0}, model)
    invalid_tie_type = frame.copy()
    invalid_tie_type["internalStructure_modal_tie"] = invalid_tie_type[
        "internalStructure_modal_tie"
    ].astype(object)
    invalid_tie_type.loc[0, "internalStructure_modal_tie"] = "False"
    with pytest.raises(ValueError, match="P7_TEST_TIE_FLAG_INVALID"):
        p7._validate_test_predictions(
            invalid_tie_type, records, {"fold_index": 0}, model
        )
    wrong_record_tie = [replace(records[0], categorical_ties=(True, False)), records[1]]
    with pytest.raises(ValueError, match="P7_RECORD_TIE_FLAG_SEMANTIC_MISMATCH"):
        p7._validate_test_predictions(
            frame, wrong_record_tie, {"fold_index": 0}, model
        )


def test_intervention_rate_gate_checks_decision_and_sample_weighted_rates() -> None:
    row: dict[str, object] = {"train_batch_count": 100, "train_sample_count": 1600}
    for group in CONCEPT_GROUP_ORDER:
        row[f"intervention_{group}_decisions"] = 25
        row[f"intervention_{group}_sample_weighted"] = 400
    report = p7._intervention_rates(pd.DataFrame([row]))
    assert report["overall_decision_rate"] == 0.25
    assert report["overall_sample_weighted_rate"] == 0.25
    invalid = pd.DataFrame([row]).copy()
    invalid.loc[0, "intervention_subtlety_decisions"] = 40
    with pytest.raises(ValueError, match="P7_.*INTERVENTION_DECISION_RATE"):
        p7._intervention_rates(invalid)


def test_history_runtime_gate_requires_h200_precision_and_both_partitions() -> None:
    split = {
        "partitions": {
            "train": {
                "summary": {"nodules": 2},
                "nodule_uids": ["train-a", "train-b"],
            },
            "validation": {
                "summary": {"nodules": 1},
                "nodule_uids": ["validation-a"],
            },
        }
    }
    train_hash = p7._partition_uid_sha256(split, "train")
    validation_hash = p7._partition_uid_sha256(split, "validation")
    history = pd.DataFrame(
        [
            {
                "epoch_index": epoch,
                "train_sample_count": 2,
                "validation_sample_count": 1,
                "train_nodule_set_sha256": train_hash,
                "validation_nodule_set_sha256": validation_hash,
            }
            for epoch in range(80)
        ]
    )
    provenance = {
        "fold_index": 0,
        "torch_use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
    }
    runtime = {
        **provenance,
        "device_type": "cuda",
        "gpu_name": "NVIDIA H200",
        "fp32": True,
        "amp_enabled": False,
        "bfloat16_enabled": False,
        "cuda_matmul_tf32_enabled": False,
        "cudnn_tf32_enabled": False,
        "epochs_total": 80,
        "peak_reserved_bytes": 123,
    }
    assert p7._validate_history_and_runtime(history, runtime, split, provenance) == (
        2,
        1,
    )
    cpu = {**runtime, "device_type": "cpu", "gpu_name": None}
    with pytest.raises(ValueError, match="P7_RUNTIME_H200_MISMATCH"):
        p7._validate_history_and_runtime(history, cpu, split, provenance)
    tf32 = {**runtime, "cuda_matmul_tf32_enabled": True}
    with pytest.raises(ValueError, match="P7_RUNTIME_PRECISION_POLICY_MISMATCH"):
        p7._validate_history_and_runtime(history, tf32, split, provenance)
    missing_validation = history.copy()
    missing_validation.loc[0, "validation_sample_count"] = 0
    with pytest.raises(ValueError, match="P7_HISTORY_VALIDATION_COVERAGE_MISMATCH"):
        p7._validate_history_and_runtime(
            missing_validation, runtime, split, provenance
        )


def test_training_resume_finishes_and_completed_run_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific = {
        "protocol": {"version": "Baseline-v2"},
        "reproducibility": {"base_seed": 20260808},
    }
    execution, execution_hash = validate_execution_config(
        Path("configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml")
    )
    split = {
        "fold_index": 0,
        "split_sha256": "1" * 64,
        "partitions": {
            "train": {
                "summary": {"nodules": 3},
                "nodule_uids": ["train-0", "train-1", "train-2"],
            },
            "validation": {
                "summary": {"nodules": 2},
                "nodule_uids": ["validation-0", "validation-1"],
            },
        },
    }
    monkeypatch.setattr(
        p7,
        "_load_sources",
        lambda *_args, **_kwargs: (
            scientific,
            execution,
            execution_hash,
            {},
            "2" * 64,
            split,
            pd.DataFrame(),
            pd.DataFrame(),
            Path("encoder.pt"),
        ),
    )

    class TinyTrainModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    monkeypatch.setattr(
        p7,
        "build_initialized_model",
        lambda *_args, **_kwargs: (
            TinyTrainModel(),
            {"fold_seed": 20260808, "initialization_sha256": "3" * 64},
        ),
    )
    monkeypatch.setattr(
        p7,
        "build_partition_concept_records",
        lambda _manifest, _index, _split, partition, _path: [
            object()
        ]
        * (3 if partition == "train" else 2),
    )

    def fake_epoch(
        _model: object,
        records: list[object],
        _device: object,
        *,
        optimizer: object | None,
        epoch_index: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        training = optimizer is not None
        count = len(records)
        partition = "train" if training else "validation"
        return {
            "task_loss": 1.0 / (epoch_index + 1),
            "concept_loss": 0.5 / (epoch_index + 1),
            "total_loss": 1.005 / (epoch_index + 1),
            "group_losses": OrderedDict(
                (group, 0.5 / (epoch_index + 1)) for group in CONCEPT_GROUP_ORDER
            ),
            "sample_count": count,
            "batch_count": 2 if training else 1,
            "nodule_set_sha256": p7._partition_uid_sha256(split, partition),
            "intervention_decision_counts": OrderedDict(
                (group, 1 if training else 0) for group in CONCEPT_GROUP_ORDER
            ),
            "intervention_sample_weighted_counts": OrderedDict(
                (group, 1 if training else 0) for group in CONCEPT_GROUP_ORDER
            ),
        }

    monkeypatch.setattr(p7, "run_cem_epoch", fake_epoch)
    monkeypatch.setattr(
        p7,
        "_runtime_environment",
        lambda _device: {
            "device_type": "cuda",
            "gpu_name": "NVIDIA H200",
            "fp32": True,
            "amp_enabled": False,
            "bfloat16_enabled": False,
            "cuda_matmul_tf32_enabled": False,
            "cudnn_tf32_enabled": False,
        },
    )
    common = {
        "scientific_config_path": Path("scientific.yaml"),
        "execution_config_path": Path("execution.yaml"),
        "p7_config_path": Path("p7.yaml"),
        "manifest_path": Path("manifest.parquet"),
        "roi_index_path": Path("roi_index.parquet"),
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": tmp_path,
    }
    interrupted = train_fold(
        **common, resume=False, _stop_after_epoch_for_test=2
    )
    assert interrupted["epoch_index"] == 2
    completed = train_fold(**common, resume=True)
    assert completed["epochs_completed"] == 80
    assert pd.read_csv(tmp_path / "fold_0" / "history.csv")["epoch_index"].tolist() == list(
        range(80)
    )
    monkeypatch.setattr(
        p7,
        "_validate_history_and_runtime",
        lambda *_args, **_kwargs: (3, 2),
    )
    reused = train_fold(**common, resume=True)
    assert reused["best_epoch_index"] == completed["best_epoch_index"]
    history_path = tmp_path / "fold_0" / "history.csv"
    history_path.write_text(history_path.read_text() + "tamper\n", encoding="utf-8")
    with pytest.raises(ValueError, match="P7_ARTIFACT_HASH_MISMATCH:history.csv"):
        train_fold(**common, resume=True)


def test_test_transaction_recovers_without_second_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fold_0"
    output.mkdir(parents=True)
    execution = load_config(
        "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
    )
    model = build_model()
    records = concept_records(tmp_path, 1)
    completion_path = output / "training_complete.json"
    completion = {
        "status": "TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "test_evaluated": False,
        "best_checkpoint_sha256": "a" * 64,
        "best_epoch_index": 7,
        "best_validation_total_loss": 0.2,
    }
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    context = {
        "execution": execution,
        "completion": completion,
        "completion_path": completion_path,
        "output": output,
        "provenance": {"fold_index": 0, "model": "mixed_type_cem"},
        "model": model,
        "manifest": pd.DataFrame(),
        "roi_index": pd.DataFrame(),
        "split": {},
    }
    monkeypatch.setattr(p7, "_trained_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        p7, "build_partition_concept_records", lambda *_args, **_kwargs: records
    )
    monkeypatch.setattr(p7, "EXPECTED_FOLD_TEST_COUNTS", {0: 1})
    inference_calls = 0

    def fake_predictions(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        nonlocal inference_calls
        inference_calls += 1
        return [{"nodule_uid": "nodule-0", "target_normalized": 0.2, "malignancy_raw_score": 0.3}]

    monkeypatch.setattr(p7, "_prediction_rows", fake_predictions)
    monkeypatch.setattr(p7, "_validate_test_predictions", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        p7,
        "regression_metrics",
        lambda _rows: {"samples": 1, "original_scale_mae": 0.4},
    )
    original_atomic_json = p7._atomic_json
    failed_once = False

    def fail_before_evaluation_seal(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed_once
        if path.name == "test_evaluation.json" and not failed_once:
            failed_once = True
            raise RuntimeError("simulated interruption")
        original_atomic_json(path, payload)

    monkeypatch.setattr(p7, "_atomic_json", fail_before_evaluation_seal)
    common = {
        "scientific_config_path": Path("scientific.yaml"),
        "execution_config_path": Path("execution.yaml"),
        "p7_config_path": Path("p7.yaml"),
        "manifest_path": Path("manifest.parquet"),
        "roi_index_path": Path("roi_index.parquet"),
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": tmp_path,
    }
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluate_test_once(**common)
    assert inference_calls == 1
    recovered = evaluate_test_once(**common)
    assert recovered["evaluation"]["status"] == "TEST_EVALUATED_ONCE"
    assert inference_calls == 1
    with pytest.raises(FileExistsError, match="P7_TEST_ALREADY_EVALUATED"):
        evaluate_test_once(**common)
    assert inference_calls == 1
