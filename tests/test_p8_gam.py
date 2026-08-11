from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import lidc_baseline.p8_gam_lifecycle as p8l
from lidc_baseline.config import load_config
from lidc_baseline.p5_blackbox import validate_execution_config
from lidc_baseline.p6_standard_cbm import (
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    CONTINUOUS_CONCEPTS,
    ConceptRecord,
)
from lidc_baseline.p8_gam import (
    EXPERTS_PER_GROUP,
    LearnedSoftmaxGAM,
    build_deterministic_gam_components,
    gam_losses,
    task_predictions_and_contributions,
    validate_p8_execution_config,
)


class IdentityEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


class TinyEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pooled = image.mean(dim=(2, 3, 4), keepdim=True)
        return pooled.repeat(1, 1024, 2, 2, 2)


def _model(fold_seed: int = 20260808) -> torch.nn.Module:
    components, _ = build_deterministic_gam_components(fold_seed)
    return LearnedSoftmaxGAM.build(IdentityEncoder(), components)


def _image_model(fold_seed: int = 20260808) -> torch.nn.Module:
    components, _ = build_deterministic_gam_components(fold_seed)
    return LearnedSoftmaxGAM.build(TinyEncoder(), components)


def _features(batch_size: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(101)
    return torch.randn(batch_size, 1024, generator=generator)


def _targets(outputs: dict[str, object]) -> dict[str, object]:
    batch_size = int(outputs["malignancy_raw_score"].shape[0])
    concepts: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group in CONCEPT_GROUP_ORDER:
        size = CONCEPT_OUTPUT_SIZES[group]
        if group in CONTINUOUS_CONCEPTS:
            concepts[group] = torch.full((batch_size, size), 0.4)
        else:
            concepts[group] = torch.full((batch_size, size), 1.0 / size)
    return {
        "malignancy": torch.full((batch_size, 1), 0.5),
        "concepts": concepts,
    }


def _concept_records(tmp_path: Path, count: int) -> list[ConceptRecord]:
    records = []
    for index in range(count):
        roi_path = tmp_path / f"roi_{index}.npz"
        image = np.full((1, 64, 64, 64), index / max(count, 1), dtype=np.float32)
        np.savez(roi_path, image=image)
        records.append(
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
    return records


def test_p8_execution_config_runtime_guard() -> None:
    config, digest = validate_p8_execution_config()
    assert config["phase"] == "P8"
    assert len(digest) == 64


def test_gam_has_eight_groups_and_five_independent_local_experts() -> None:
    model = _model()
    assert tuple(model.experts) == CONCEPT_GROUP_ORDER
    parameter_ids: set[int] = set()
    for group in CONCEPT_GROUP_ORDER:
        experts = model.experts[group]
        assert len(experts) == EXPERTS_PER_GROUP
        expected_input = CONCEPT_OUTPUT_SIZES[group]
        for expert in experts:
            assert isinstance(expert[0], torch.nn.Linear)
            assert expert[0].in_features == expected_input
            assert expert[0].out_features == 32
            assert isinstance(expert[1], torch.nn.ReLU)
            assert expert[2].in_features == 32
            assert expert[2].out_features == 16
            assert isinstance(expert[3], torch.nn.ReLU)
            assert expert[4].in_features == 16
            assert expert[4].out_features == 1
            for parameter in expert.parameters():
                assert id(parameter) not in parameter_ids
                parameter_ids.add(id(parameter))
    assert len(parameter_ids) == 40 * 6


def test_zero_alpha_logits_produce_exact_uniform_weights() -> None:
    model = _model()
    outputs = model.forward_from_features(_features())
    for group in CONCEPT_GROUP_ORDER:
        assert torch.equal(model.alpha_logits[group], torch.zeros(5))
        assert torch.equal(outputs["alpha_weights"][group], torch.full((5,), 0.2))
        assert outputs["alpha_weights"][group].requires_grad


def test_stage_a_structure_gate_proves_all_experts_are_independent_and_local() -> None:
    report = p8l._stage_a_structure_report(_model())
    assert report["status"] == "PASS"
    assert report["independent_experts"] == 40
    assert report["shared_expert_parameters"] == 0
    for group in CONCEPT_GROUP_ORDER:
        assert report["groups"][group]["experts"] == 5
        assert report["groups"][group]["concept_local_input_only"] is True
        assert report["groups"][group]["input_dimensions"] == [
            CONCEPT_OUTPUT_SIZES[group]
        ] * 5


@pytest.mark.parametrize("command", ["overfit-check", "preflight"])
def test_stage_a_cli_does_not_forward_lifecycle_output_root(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def stage_a_stub(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "PASS", "command": command}

    target = "overfit_check" if command == "overfit-check" else "preflight"
    monkeypatch.setattr(p8l, target, stage_a_stub)
    arguments = [
        command,
        "--fold",
        "0",
        "--output-root",
        str(tmp_path / "must-not-be-forwarded"),
        "--output",
        str(tmp_path / f"{command}.json"),
    ]
    assert p8l.main(arguments) == 0
    assert "output_root" not in captured
    assert captured["fold_index"] == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_task_path_is_group_local_and_uses_activated_predictions() -> None:
    model = _model()
    features = _features()
    baseline = model.forward_from_features(features)
    with torch.no_grad():
        model.concept_heads["subtlety"].bias.add_(10.0)
    changed = model.forward_from_features(features)
    assert not torch.equal(
        baseline["expert_outputs"]["subtlety"],
        changed["expert_outputs"]["subtlety"],
    )
    for group in CONCEPT_GROUP_ORDER:
        if group != "subtlety":
            assert torch.equal(
                baseline["expert_outputs"][group], changed["expert_outputs"][group]
            )
    expected = torch.cat(tuple(changed["activated"].values()), dim=1)
    assert torch.equal(changed["canonical_vector"], expected)


def test_alpha_receives_gradient_and_updates_for_every_group() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    initial = {group: model.alpha_logits[group].detach().clone() for group in CONCEPT_GROUP_ORDER}
    outputs = model.forward_from_features(_features(16))
    losses = gam_losses(outputs, _targets(outputs))
    losses["total_loss"].backward()
    for group in CONCEPT_GROUP_ORDER:
        gradient = model.alpha_logits[group].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0
    optimizer.step()
    for group in CONCEPT_GROUP_ORDER:
        assert not torch.equal(model.alpha_logits[group].detach(), initial[group])


def test_gam_loss_is_task_plus_equal_weight_concept_loss() -> None:
    model = _model()
    outputs = model.forward_from_features(_features(7))
    losses = gam_losses(outputs, _targets(outputs))
    assert torch.allclose(
        losses["concept_loss"],
        torch.stack(tuple(losses["group_losses"].values())).mean(),
    )
    assert torch.allclose(
        losses["total_loss"], losses["task_loss"] + losses["concept_loss"]
    )
    assert losses["batch_size"] == 7


def test_deterministic_initialization_is_order_isolated_and_fold_specific() -> None:
    first, first_metadata = build_deterministic_gam_components(20260808)
    torch.manual_seed(7)
    _ = torch.randn(500)
    second, second_metadata = build_deterministic_gam_components(20260808)
    _, other_metadata = build_deterministic_gam_components(20260809)
    assert first_metadata == second_metadata
    assert first_metadata["combined_gam_initialization_sha256"] != (
        other_metadata["combined_gam_initialization_sha256"]
    )
    assert first_metadata["subnetwork_initialization_sha256"] == (
        second_metadata["subnetwork_initialization_sha256"]
    )
    assert first_metadata["initial_alpha_logits_sha256"] == (
        second_metadata["initial_alpha_logits_sha256"]
    )
    assert first_metadata["initial_raw_bias_sha256"] == (
        second_metadata["initial_raw_bias_sha256"]
    )
    assert first["concept_heads"] is not second["concept_heads"]


def test_unbounded_output_and_contribution_reconstruction() -> None:
    model = _model()
    with torch.no_grad():
        model.global_raw_bias.fill_(2.0)
    outputs = model.forward_from_features(_features(5))
    result = task_predictions_and_contributions(model, outputs)
    assert torch.all(result["malignancy_raw_score"] > 1.0)
    assert torch.equal(
        result["malignancy_raw_score"], result["malignancy_score_normalized"]
    )
    assert torch.allclose(
        result["malignancy_score_1_to_5"],
        1.0 + 4.0 * result["malignancy_raw_score"],
    )
    assert result["normalized_reconstruction_max_abs_error"] <= 1e-6
    assert result["rating_reconstruction_max_abs_error"] <= 1e-6


def test_contribution_guard_rejects_tampering() -> None:
    model = _model()
    outputs = model.forward_from_features(_features(3))
    outputs["group_contributions"]["subtlety"] = (
        outputs["group_contributions"]["subtlety"] + 0.01
    )
    with pytest.raises(ValueError, match="P8_NORMALIZED_CONTRIBUTION"):
        task_predictions_and_contributions(model, outputs)


def test_epoch_uses_partial_batch_full_coverage_and_no_validation_augmentation(
    tmp_path: Path,
) -> None:
    model = _image_model()
    records = _concept_records(tmp_path, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    train = p8l.run_gam_epoch(
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
    validation = p8l.run_gam_epoch(
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
    assert train["sample_count"] == validation["sample_count"] == 3
    assert train["batch_count"] == validation["batch_count"] == 2
    assert all(value > 0.0 for value in train["alpha_gradient_l1"].values())
    assert all(value == 0.0 for value in validation["alpha_gradient_l1"].values())


def test_prediction_schema_reconstructs_experts_alpha_and_ties(tmp_path: Path) -> None:
    model = _image_model()
    records = _concept_records(tmp_path, 2)
    rows = p8l._prediction_rows(
        model, records, torch.device("cpu"), batch_size=2, num_workers=0
    )
    frame = pd.DataFrame(rows)
    frame["fold_index"] = 0
    p8l._validate_test_predictions(frame, records, {"fold_index": 0}, model)
    alpha_tamper = frame.copy()
    alpha_tamper.loc[0, "subtlety_alpha_weights"] = json.dumps([1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="P8_TEST_ALPHA_SOFTMAX_MISMATCH"):
        p8l._validate_test_predictions(
            alpha_tamper, records, {"fold_index": 0}, model
        )
    expert_tamper = frame.copy()
    expert_tamper.loc[0, "subtlety_expert_outputs"] = json.dumps([0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="P8_TEST_NUMERIC_RECONSTRUCTION_MISMATCH"):
        p8l._validate_test_predictions(
            expert_tamper, records, {"fold_index": 0}, model
        )
    roundoff = frame.copy()
    roundoff_values = json.loads(str(roundoff.loc[0, "subtlety_expert_outputs"]))
    roundoff_values[0] += 5e-7
    roundoff.loc[0, "subtlety_expert_outputs"] = json.dumps(roundoff_values)
    p8l._validate_test_predictions(roundoff, records, {"fold_index": 0}, model)
    extra_column = frame.copy()
    extra_column["unexpected_private_field"] = 1
    with pytest.raises(ValueError, match="P8_TEST_PREDICTION_SCHEMA_MISMATCH"):
        p8l._validate_test_predictions(
            extra_column, records, {"fold_index": 0}, model
        )
    with pytest.raises(ValueError, match="P8_TEST_PREDICTION_SCHEMA_MISMATCH"):
        p8l._validate_test_predictions(
            frame.drop(columns=["texture_expert_outputs"]),
            records,
            {"fold_index": 0},
            model,
        )
    invalid_bool = frame.copy()
    invalid_bool["internalStructure_modal_tie"] = invalid_bool[
        "internalStructure_modal_tie"
    ].astype(object)
    invalid_bool.loc[0, "internalStructure_modal_tie"] = "False"
    with pytest.raises(ValueError, match="P8_TEST_TIE_FLAG_INVALID"):
        p8l._validate_test_predictions(
            invalid_bool, records, {"fold_index": 0}, model
        )
    invalid_count = frame.copy()
    invalid_count["subtlety_valid_reader_count"] = invalid_count[
        "subtlety_valid_reader_count"
    ].astype(object)
    invalid_count.loc[0, "subtlety_valid_reader_count"] = 2.9
    with pytest.raises(ValueError, match="P8_TEST_VALID_READER_COUNT_INVALID"):
        p8l._validate_test_predictions(
            invalid_count, records, {"fold_index": 0}, model
        )


def test_history_runtime_requires_h200_precision_coverage_and_alpha_evidence() -> None:
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
    rows = []
    for epoch in range(80):
        row: dict[str, object] = {
            "epoch_index": epoch,
            "train_sample_count": 2,
            "validation_sample_count": 1,
            "train_nodule_set_sha256": p8l._partition_uid_sha256(split, "train"),
            "validation_nodule_set_sha256": p8l._partition_uid_sha256(
                split, "validation"
            ),
        }
        for group in CONCEPT_GROUP_ORDER:
            row[f"alpha_{group}_gradient_l1"] = 0.1
            row[f"alpha_{group}_logits"] = json.dumps([0.1, 0, 0, 0, 0])
            weights = torch.softmax(torch.tensor([0.1, 0, 0, 0, 0]), dim=0)
            row[f"alpha_{group}_weights"] = json.dumps(weights.tolist())
        rows.append(row)
    history = pd.DataFrame(rows)
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
        "peak_reserved_bytes": 1,
    }
    assert p8l._validate_history_and_runtime(history, runtime, split, provenance) == (
        2,
        1,
    )
    with pytest.raises(ValueError, match="P8_RUNTIME_H200_MISMATCH"):
        p8l._validate_history_and_runtime(
            history, {**runtime, "device_type": "cpu"}, split, provenance
        )
    with pytest.raises(ValueError, match="P8_RUNTIME_PRECISION_POLICY_MISMATCH"):
        p8l._validate_history_and_runtime(
            history,
            {**runtime, "cuda_matmul_tf32_enabled": True},
            split,
            provenance,
        )
    no_gradient = history.copy()
    no_gradient.loc[:, "alpha_subtlety_gradient_l1"] = 0.0
    with pytest.raises(ValueError, match="P8_ALPHA_GRADIENT_EVIDENCE_INVALID"):
        p8l._validate_history_and_runtime(no_gradient, runtime, split, provenance)
    invalid_softmax = history.copy()
    invalid_softmax.loc[0, "alpha_subtlety_weights"] = json.dumps([1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="P8_ALPHA_HISTORY_SOFTMAX_MISMATCH"):
        p8l._validate_history_and_runtime(
            invalid_softmax, runtime, split, provenance
        )


def test_checkpoint_metadata_binds_objectives_alpha_and_earlier_tie() -> None:
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    execution, _ = validate_execution_config(
        Path(
            "configs/experiments/"
            "baseline_v2_reference_training_h200_warn_only.yaml"
        )
    )
    scheduler = p8l._scheduler(optimizer, execution)
    outputs = model.forward_from_features(_features(3))
    losses = gam_losses(outputs, _targets(outputs))
    validation_report = {
        "task_loss": float(losses["task_loss"].detach()),
        "concept_loss": float(losses["concept_loss"].detach()),
        "total_loss": float(losses["total_loss"].detach()),
        "group_losses": OrderedDict(
            (group, float(value.detach()))
            for group, value in losses["group_losses"].items()
        ),
    }
    alpha = p8l._alpha_snapshot(model)
    history_row: dict[str, object] = {
        "validation_task_loss": validation_report["task_loss"],
        "validation_concept_loss": validation_report["concept_loss"],
        "validation_total_loss": validation_report["total_loss"],
    }
    for group in CONCEPT_GROUP_ORDER:
        history_row[f"validation_{group}_loss"] = validation_report[
            "group_losses"
        ][group]
        history_row[f"alpha_{group}_logits"] = json.dumps(alpha["logits"][group])
        history_row[f"alpha_{group}_weights"] = json.dumps(alpha["weights"][group])
    payload = p8l._checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch_index=4,
        validation_total_loss=validation_report["total_loss"],
        validation_report=validation_report,
        best_epoch_index=4,
        best_validation_total_loss=validation_report["total_loss"],
        provenance={"fold_index": 0},
        history=[history_row],
    )
    assert p8l._validate_checkpoint_metadata(payload, history_row) == alpha
    assert p8l.checkpoint_improves(0.2, 0.3) is True
    assert p8l.checkpoint_improves(0.2, 0.2) is False
    tampered = dict(payload)
    tampered_state = OrderedDict(payload["model_state_dict"])
    tampered_state["alpha_logits.subtlety"] = (
        tampered_state["alpha_logits.subtlety"].clone() + 0.01
    )
    tampered["model_state_dict"] = tampered_state
    with pytest.raises(ValueError, match="P8_CHECKPOINT_ALPHA_SNAPSHOT_MISMATCH"):
        p8l._validate_checkpoint_metadata(tampered, history_row)


def test_training_resume_completes_and_verified_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific = {
        "protocol": {"version": "Baseline-v2"},
        "reproducibility": {"base_seed": 20260808},
    }
    execution, execution_hash = validate_execution_config(
        Path(
            "configs/experiments/"
            "baseline_v2_reference_training_h200_warn_only.yaml"
        )
    )
    split = {
        "fold_index": 0,
        "split_sha256": "1" * 64,
        "partitions": {
            "train": {"summary": {"nodules": 3}, "nodule_uids": ["a", "b", "c"]},
            "validation": {"summary": {"nodules": 2}, "nodule_uids": ["d", "e"]},
        },
    }
    monkeypatch.setattr(
        p8l,
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
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.alpha_logits = torch.nn.ParameterDict(
                {
                    group: torch.nn.Parameter(torch.zeros(5))
                    for group in CONCEPT_GROUP_ORDER
                }
            )

    monkeypatch.setattr(
        p8l,
        "build_initialized_model",
        lambda *_args, **_kwargs: (
            TinyTrainModel(),
            {"fold_seed": 20260808, "initialization_sha256": "3" * 64},
        ),
    )
    monkeypatch.setattr(
        p8l,
        "build_partition_concept_records",
        lambda _manifest, _index, _split, partition, _path: [object()]
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
        partition = "train" if optimizer is not None else "validation"
        return {
            "task_loss": 1 / (epoch_index + 1),
            "concept_loss": 0.5 / (epoch_index + 1),
            "total_loss": 1.5 / (epoch_index + 1),
            "group_losses": OrderedDict(
                (group, 0.5 / (epoch_index + 1)) for group in CONCEPT_GROUP_ORDER
            ),
            "sample_count": len(records),
            "batch_count": 2 if optimizer is not None else 1,
            "nodule_set_sha256": p8l._partition_uid_sha256(split, partition),
            "alpha_gradient_l1": OrderedDict(
                (group, 0.1 if optimizer is not None else 0.0)
                for group in CONCEPT_GROUP_ORDER
            ),
        }

    monkeypatch.setattr(p8l, "run_gam_epoch", fake_epoch)
    monkeypatch.setattr(
        p8l,
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
        "p8_config_path": Path("p8.yaml"),
        "manifest_path": Path("manifest.parquet"),
        "roi_index_path": Path("roi.parquet"),
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": tmp_path,
    }
    interrupted = p8l.train_fold(
        **common, resume=False, _stop_after_epoch_for_test=2
    )
    assert interrupted["epoch_index"] == 2
    completed = p8l.train_fold(**common, resume=True)
    assert completed["epochs_completed"] == 80
    monkeypatch.setattr(
        p8l, "_validate_history_and_runtime", lambda *_args, **_kwargs: (3, 2)
    )
    reused = p8l.train_fold(**common, resume=True)
    assert reused["best_epoch_index"] == completed["best_epoch_index"]


def test_test_transaction_recovers_committed_predictions_without_second_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "fold_0"
    output.mkdir(parents=True)
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
        "execution": load_config(
            "configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml"
        ),
        "completion": completion,
        "completion_path": completion_path,
        "output": output,
        "provenance": {"fold_index": 0, "model": "p8"},
        "model": _image_model(),
        "manifest": pd.DataFrame(),
        "roi_index": pd.DataFrame(),
        "split": {},
    }

    def context_loader(**_kwargs: object) -> dict[str, object]:
        context["completion"] = json.loads(completion_path.read_text())
        return context

    monkeypatch.setattr(p8l, "_trained_context", context_loader)
    monkeypatch.setattr(
        p8l, "build_partition_concept_records", lambda *_args, **_kwargs: [object()]
    )
    calls = 0

    def fake_predictions(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return [{"nodule_uid": "x", "target_normalized": 0.2, "malignancy_raw_score": 0.3}]

    monkeypatch.setattr(p8l, "_prediction_rows", fake_predictions)
    monkeypatch.setattr(
        p8l,
        "_validate_test_predictions",
        lambda *_args, **_kwargs: {
            "numeric_reconstruction_schema": p8l.NUMERIC_SCHEMA,
            "numeric_reconstruction_maximum_absolute_error": 0.0,
            "numeric_reconstruction_maximum_allowed_absolute_error": 1e-6,
        },
    )
    monkeypatch.setattr(
        p8l, "regression_metrics", lambda _rows: {"samples": 1, "original_scale_mae": 0.4}
    )
    original = p8l._atomic_json
    failed = False

    def interrupt(path: Path, payload: dict[str, object]) -> None:
        nonlocal failed
        if (
            path == completion_path
            and (output / "test_evaluation.json").exists()
            and not failed
        ):
            failed = True
            raise RuntimeError("interrupt")
        original(path, payload)

    monkeypatch.setattr(p8l, "_atomic_json", interrupt)
    common = {
        "scientific_config_path": Path("scientific.yaml"),
        "execution_config_path": Path("execution.yaml"),
        "p8_config_path": Path("p8.yaml"),
        "manifest_path": Path("manifest.parquet"),
        "roi_index_path": Path("roi.parquet"),
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": tmp_path,
    }
    with pytest.raises(RuntimeError, match="interrupt"):
        p8l.evaluate_test_once(**common)
    assert calls == 1
    result = p8l.evaluate_test_once(**common)
    assert result["evaluation"]["test_transaction_count"] == 1
    assert result["recovered_without_inference"] is True
    assert calls == 1
    with pytest.raises(FileExistsError, match="P8_TEST_ALREADY_EVALUATED"):
        p8l.evaluate_test_once(**common)
    calls_before_corruption_check = calls
    for filename in (
        "test_evaluation.json",
        "test_claim.json",
        "test_predictions.parquet",
        "metrics.json",
    ):
        (output / filename).unlink()
    with pytest.raises(
        ValueError, match="P8_COMPLETION_CLAIMS_MISSING_EVALUATION"
    ):
        p8l.evaluate_test_once(**common)
    assert calls == calls_before_corruption_check


def test_verify_all_enforces_oof_counts_patients_and_zero_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold_counts = [479, 502, 539, 549, 564]
    patient_counts = [171, 180, 169, 174, 174]
    frames: dict[int, pd.DataFrame] = {}
    nodule_offset = 0
    patient_offset = 0
    for fold_index, (nodule_count, patient_count) in enumerate(
        zip(fold_counts, patient_counts, strict=True)
    ):
        patients = [
            f"patient-{patient_offset + index}" for index in range(patient_count)
        ]
        frames[fold_index] = pd.DataFrame(
            {
                "nodule_uid": [
                    f"nodule-{nodule_offset + index}" for index in range(nodule_count)
                ],
                "patient_key": [patients[index % patient_count] for index in range(nodule_count)],
                "fold_index": [fold_index] * nodule_count,
            }
        )
        nodule_offset += nodule_count
        patient_offset += patient_count
    monkeypatch.setattr(p8l, "verify_fold", lambda **kwargs: {"fold": kwargs["fold_index"]})

    def read_frame(path: Path) -> pd.DataFrame:
        fold_index = int(path.parent.name.removeprefix("fold_"))
        return frames[fold_index].copy()

    monkeypatch.setattr(p8l.pd, "read_parquet", read_frame)
    report = p8l.verify_all(
        scientific_config_path=Path("scientific.yaml"),
        execution_config_path=Path("execution.yaml"),
        p8_config_path=Path("p8.yaml"),
        manifest_path=Path("manifest.parquet"),
        roi_index_path=Path("roi.parquet"),
        output_root=tmp_path,
    )
    assert report["oof_nodules"] == 2633
    assert report["oof_patients"] == 868
    assert report["fold_test_counts"] == fold_counts
    original = frames[0].loc[0, "patient_key"]
    assert (frames[0]["patient_key"] == original).sum() > 1
    frames[0].loc[0, "patient_key"] = frames[1].loc[0, "patient_key"]
    with pytest.raises(ValueError, match="P8_OOF_PATIENT_LEAKAGE"):
        p8l.verify_all(
            scientific_config_path=Path("scientific.yaml"),
            execution_config_path=Path("execution.yaml"),
            p8_config_path=Path("p8.yaml"),
            manifest_path=Path("manifest.parquet"),
            roi_index_path=Path("roi.parquet"),
            output_root=tmp_path,
        )
