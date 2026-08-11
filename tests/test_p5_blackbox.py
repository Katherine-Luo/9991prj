from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lidc_baseline.config import load_config
from lidc_baseline.p4_prepare import (
    encoder_state_sha256,
    read_split,
    validate_encoder_artifact,
)
from lidc_baseline.p5_blackbox import (
    AUGMENTATION_SCHEMA_VERSION,
    BlackBoxRegressor,
    SampleRecord,
    ValidationMSEPlateau,
    _atomic_torch_save,
    _evaluate_test_once_locked,
    _loader,
    _load_checkpoint,
    _provenance,
    _test_row_provenance,
    _validate_test_prediction_frame,
    _optimizer,
    apply_training_augmentation,
    augmentation_material,
    augmentation_parameters,
    build_deterministic_head,
    build_initialized_model,
    capture_rng_state,
    checkpoint_payload,
    checkpoint_improves,
    epoch_uid_order,
    exclusive_fold_lifecycle_lock,
    head_initialization_seed,
    head_state_sha256,
    predict_records,
    regression_metrics,
    require_formal_gpu_for_cuda,
    configure_fp32_determinism,
    reproducibility_provenance,
    restore_rng_state,
    serialized_float_consistent,
    train_one_epoch,
    train_fold,
    validate_execution_config,
)


EXECUTION_CONFIG = Path("configs/experiments/baseline_v2_reference_training_h200_warn_only.yaml")


def _record(tmp_path: Path, uid: str, value: float = 0.25) -> SampleRecord:
    path = tmp_path / f"{uid}.npz"
    image = np.full((1, 64, 64, 64), value, dtype=np.float32)
    np.savez(path, image=image)
    return SampleRecord(
        nodule_uid=uid,
        patient_key=f"patient-key-{uid}",
        roi_path=path,
        target_normalized=value,
        target_1_to_5=1.0 + 4.0 * value,
        extreme_binary_eligible=False,
        extreme_binary_label=None,
    )


def test_execution_config_rejects_hash_or_policy_tampering(tmp_path: Path) -> None:
    config, observed = validate_execution_config(EXECUTION_CONFIG)
    assert observed == "66c925a7b43bf9fa312ceb850b43746a34d1808888667c39392eaef9e47495bb"

    copied = tmp_path / "execution.yaml"
    copied.write_text(EXECUTION_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    digest = tmp_path / "execution.sha256"
    digest.write_text(f"{observed}\n", encoding="ascii")
    copied.write_text(
        copied.read_text(encoding="utf-8").replace("  epochs: 80\n", "  epochs: 81\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_execution_config(copied, digest)

    changed = copy.deepcopy(config)
    changed["reference_reported"]["batch_size"] = 8
    from lidc_baseline.config import canonical_yaml, compute_config_sha256

    copied.write_bytes(canonical_yaml(changed))
    digest.write_text(f"{compute_config_sha256(changed)}\n", encoding="ascii")
    with pytest.raises(ValueError, match="REFERENCE_POLICY_MISMATCH"):
        validate_execution_config(copied, digest)


def test_warn_only_deterministic_policy_is_explicit_and_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    execution, _digest = validate_execution_config(EXECUTION_CONFIG)
    observed: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only=False: observed.append((enabled, warn_only)),
    )
    policy = configure_fp32_determinism(torch.device("cpu"), execution)
    assert policy == {
        "torch_use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
    }
    assert reproducibility_provenance(execution) == policy
    assert observed == [(True, True)]


def test_scheduler_decays_after_exactly_four_consecutive_bad_epochs() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = ValidationMSEPlateau(optimizer)
    assert scheduler.step(1.0) is False
    for _ in range(3):
        assert scheduler.step(1.0) is False
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert scheduler.step(1.0) is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(9e-5)
    assert scheduler.bad_epoch_counter == 0


def test_serialized_float_consistency_accepts_tiny_round_trip_difference() -> None:
    assert serialized_float_consistent(
        0.01997598138996362,
        0.0199759813899636,
    )


def test_serialized_float_consistency_rejects_real_objective_mismatch() -> None:
    assert not serialized_float_consistent(
        0.01997598138996362,
        0.01997698138996362,
    )


def test_scheduler_tolerance_is_independent_of_exact_checkpoint_rule() -> None:
    import torch

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    scheduler = ValidationMSEPlateau(optimizer)
    scheduler.step(0.5)
    assert scheduler.step(0.49995) is False
    assert scheduler.best == 0.5
    assert scheduler.bad_epoch_counter == 1
    assert checkpoint_improves(0.49995, 0.5) is True
    assert checkpoint_improves(0.5, 0.5) is False


def test_optimizer_is_adam_with_exact_zero_weight_decay() -> None:
    import torch

    config = load_config(EXECUTION_CONFIG)
    model = torch.nn.Linear(2, 1)
    optimizer = _optimizer(model, config)
    assert isinstance(optimizer, torch.optim.Adam)
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(1e-4)
    assert group["betas"] == pytest.approx((0.9, 0.999))
    assert group["eps"] == pytest.approx(1e-7)
    assert group["weight_decay"] == 0.0


def test_augmentation_material_has_only_approved_fields() -> None:
    first = augmentation_material(20260808, 2, 7, "abc")
    second = augmentation_material(20260808, 2, 7, "abc")
    assert first == second
    assert b"blackbox" not in first
    payload = json.loads(first.split(b"\0", 1)[1])
    assert payload == {
        "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
        "base_seed": 20260808,
        "epoch_index": 7,
        "fold_index": 2,
        "nodule_uid": "abc",
    }


def test_augmentation_parameters_are_deterministic_and_bounded() -> None:
    first = augmentation_parameters(20260808, 0, 1, "sample")
    assert first == augmentation_parameters(20260808, 0, 1, "sample")
    assert first != augmentation_parameters(20260808, 0, 2, "sample")
    assert -15.0 <= first["angle_degrees"] <= 15.0
    assert first["operation_order"] == (
        "axial_rotation",
        "h_axis_flip",
        "w_axis_flip",
        "z_order_reversal",
    )
    draws = [augmentation_parameters(20260808, 0, 0, f"n-{index}") for index in range(1000)]
    for field in ("rotate", "flip_h", "flip_w", "reverse_z"):
        rate = sum(bool(draw[field]) for draw in draws) / len(draws)
        assert 0.45 < rate < 0.55


def test_rotation_uses_bilinear_zero_padding_and_align_corners_false(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    image = torch.zeros((1, 64, 64, 64), dtype=torch.float32)
    observed: dict[str, object] = {}
    original = torch.nn.functional.grid_sample

    def spy(*args: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grid_sample", spy)
    result = apply_training_augmentation(
        image,
        {
            "rotate": True,
            "angle_degrees": 15.0,
            "flip_h": False,
            "flip_w": False,
            "reverse_z": False,
        },
    )
    assert result.shape == image.shape
    assert observed["mode"] == "bilinear"
    assert observed["padding_mode"] == "zeros"
    assert observed["align_corners"] is False


def test_flip_order_matches_h_w_then_z() -> None:
    import torch

    image = torch.arange(64**3, dtype=torch.float32).reshape(1, 64, 64, 64)
    result = apply_training_augmentation(
        image,
        {
            "rotate": False,
            "angle_degrees": 0.0,
            "flip_h": True,
            "flip_w": True,
            "reverse_z": True,
        },
    )
    assert torch.equal(result, torch.flip(image, dims=(-2, -1, -3)))


def test_epoch_order_is_deterministic_complete_and_epoch_specific() -> None:
    uids = [f"n-{index}" for index in range(31)]
    first = epoch_uid_order(uids, 20260808, 0, 0)
    assert first == epoch_uid_order(reversed(uids), 20260808, 0, 0)
    assert first != epoch_uid_order(uids, 20260808, 0, 1)
    assert sorted(first) == sorted(uids)


def test_head_seed_and_hash_are_isolated_repeatable_and_fold_specific() -> None:
    import torch

    torch.manual_seed(444)
    before = torch.get_rng_state().clone()
    first, first_seed, first_hash = build_deterministic_head(20260808)
    after = torch.get_rng_state()
    second, second_seed, second_hash = build_deterministic_head(20260808)
    third, third_seed, third_hash = build_deterministic_head(20260809)
    assert torch.equal(before, after)
    assert first_seed == second_seed == head_initialization_seed(20260808)
    assert first_hash == second_hash == head_state_sha256(first) == head_state_sha256(second)
    assert third_seed != first_seed
    assert third_hash != first_hash
    assert head_state_sha256(third) == third_hash


@pytest.mark.integration
def test_p4_encoder_hash_is_verified_before_blackbox_use() -> None:
    import torch

    scientific = load_config("configs/baseline_v2.yaml")
    split = read_split("artifacts/baseline_v2/splits/fold_0.json")
    artifact = Path("artifacts/baseline_v2/encoder_initializations/fold_0.pt")
    model, provenance = build_initialized_model(scientific, split, artifact)
    expected = validate_encoder_artifact(artifact, scientific, split)["metadata"]["encoder_state_sha256"]
    assert provenance["encoder_initialization_sha256"] == expected
    assert encoder_state_sha256(model.encoder.state_dict()) == expected
    assert isinstance(model.encoder.pool0, torch.nn.MaxPool3d)
    assert all(
        isinstance(getattr(model.encoder, name).pool, torch.nn.AvgPool3d)
        for name in ("transition1", "transition2", "transition3")
    )


def test_blackbox_output_is_unconstrained_linear_score() -> None:
    import torch

    class Encoder(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return torch.ones((image.shape[0], 1024, 1, 1, 1), dtype=image.dtype)

    head = torch.nn.Linear(1024, 1)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(2.0)
    model = BlackBoxRegressor.build(Encoder(), head)
    score = model(torch.zeros((2, 1, 64, 64, 64)))
    assert score.shape == (2, 1)
    assert score.tolist() == [[2.0], [2.0]]
    assert 1.0 + 4.0 * score[0, 0].item() == 9.0


def test_validation_prediction_has_no_augmentation_and_no_mask_dependency(tmp_path: Path) -> None:
    import torch

    record = _record(tmp_path, "only", value=0.25)

    class MeanModel(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)

    rows = predict_records(MeanModel(), [record], torch.device("cpu"), batch_size=16, num_workers=0)
    assert rows[0]["malignancy_raw_score"] == pytest.approx(0.25)
    assert rows[0]["malignancy_score_normalized"] == pytest.approx(0.25)
    assert rows[0]["malignancy_score_1_to_5"] == pytest.approx(2.0)


def test_drop_last_false_uses_every_training_nodule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch
    import lidc_baseline.p5_blackbox as module

    records = [_record(tmp_path, f"n-{index}", value=index / 100.0) for index in range(17)]
    monkeypatch.setattr(
        module,
        "augmentation_parameters",
        lambda *args, **kwargs: {
            "rotate": False,
            "angle_degrees": 0.0,
            "flip_h": False,
            "flip_w": False,
            "reverse_z": False,
        },
    )

    class MeanModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1) * self.scale

    model = MeanModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    report = train_one_epoch(
        model,
        records,
        optimizer,
        torch.device("cpu"),
        base_seed=20260808,
        fold_index=0,
        epoch_index=0,
        batch_size=16,
        num_workers=0,
    )
    assert report["sample_count"] == 17
    loader = _loader(list(range(17)), batch_size=16, num_workers=0)
    assert [len(batch) for batch in loader] == [16, 1]
    assert loader.drop_last is False


def test_regression_metrics_use_unclipped_predictions() -> None:
    predictions = [
        {"target_normalized": 0.0, "malignancy_raw_score": -0.5},
        {"target_normalized": 1.0, "malignancy_raw_score": 1.5},
        {"target_normalized": 0.5, "malignancy_raw_score": 0.5},
    ]
    result = regression_metrics(predictions)
    assert result["normalized_mae"] == pytest.approx(1.0 / 3.0)
    assert result["original_scale_mae"] == pytest.approx(4.0 / 3.0)
    assert result["prediction_below_0_rate"] == pytest.approx(1.0 / 3.0)
    assert result["prediction_above_1_rate"] == pytest.approx(1.0 / 3.0)


def test_checkpoint_resume_restores_model_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    import random
    import torch

    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-7)
    scheduler = ValidationMSEPlateau(optimizer)
    image = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.25]])
    loss = torch.nn.functional.mse_loss(model(image), target)
    loss.backward()
    optimizer.step()
    scheduler.step(0.5)
    scheduler.step(0.5)
    provenance = {"fold_index": 0, "execution_config_sha256": "config"}
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch_index=0,
        validation_mse=0.5,
        best_epoch_index=0,
        best_validation_mse=0.5,
        provenance=provenance,
    )
    payload["history"] = [{"epoch_index": 0}]
    path = tmp_path / "last.pt"
    _atomic_torch_save(path, payload)
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = float(torch.rand(1))

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-4, eps=1e-7)
    restored_scheduler = ValidationMSEPlateau(restored_optimizer)
    loaded = _load_checkpoint(path, provenance)
    restored_model.load_state_dict(loaded["model_state_dict"])
    restored_optimizer.load_state_dict(loaded["optimizer_state_dict"])
    restored_scheduler.load_state_dict(loaded["scheduler_state_dict"])
    restore_rng_state(loaded["rng_state"])
    assert restored_model.state_dict().keys() == model.state_dict().keys()
    assert all(
        torch.equal(restored_model.state_dict()[name], model.state_dict()[name])
        for name in model.state_dict()
    )
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert float(torch.rand(1)) == expected_torch
    assert loaded["history"] == [{"epoch_index": 0}]


def test_capture_restore_rng_is_exact() -> None:
    import random
    import torch

    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    restore_rng_state(state)
    observed = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert observed == expected


def test_fold_lifecycle_lock_blocks_second_writer(tmp_path: Path) -> None:
    lock = tmp_path / "fold_0" / ".p5_lifecycle.lock"
    with exclusive_fold_lifecycle_lock(lock):
        with pytest.raises(RuntimeError, match="ALREADY_RUNNING"):
            with exclusive_fold_lifecycle_lock(lock):
                pass


def test_fold_lifecycle_lock_blocks_actual_second_process(tmp_path: Path) -> None:
    lock = tmp_path / "fold_0" / ".p5_lifecycle.lock"
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from lidc_baseline.p5_blackbox import exclusive_fold_lifecycle_lock\n"
        "with exclusive_fold_lifecycle_lock(Path(sys.argv[1])):\n"
        " print('READY', flush=True)\n"
        " sys.stdin.readline()\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock)],
        cwd=Path.cwd(),
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        with pytest.raises(RuntimeError, match="ALREADY_RUNNING"):
            with exclusive_fold_lifecycle_lock(lock):
                pass
    finally:
        if process.stdin is not None:
            process.stdin.write("release\n")
            process.stdin.flush()
        process.wait(timeout=10)
    assert process.returncode == 0


def test_cuda_execution_requires_h200_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import lidc_baseline.p5_blackbox as module

    class Device:
        type = "cuda"

    class Cuda:
        @staticmethod
        def get_device_name(device: object) -> str:
            return "NVIDIA RTX 6000 Ada"

    class Torch:
        cuda = Cuda()

    monkeypatch.setattr(module, "_torch", lambda: Torch())
    execution = load_config(EXECUTION_CONFIG)
    with pytest.raises(RuntimeError, match="REQUIRES_NVIDIA_H200"):
        require_formal_gpu_for_cuda(Device(), execution)
    monkeypatch.setattr(Cuda, "get_device_name", staticmethod(lambda device: "NVIDIA H200"))
    require_formal_gpu_for_cuda(Device(), execution)


def test_partial_test_transaction_recovers_without_second_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    import lidc_baseline.p5_blackbox as module

    scientific = {"protocol": {"version": "Baseline-v2"}}
    execution = load_config(EXECUTION_CONFIG)
    execution_hash = "execution-hash"
    split = {"fold_index": 0, "split_sha256": "split-hash"}
    initialization = {
        "fold_seed": 20260808,
        "head_initialization_seed": 123,
        "head_initialization_sha256": "head-hash",
        "head_seed_derivation": "derivation",
        "encoder_initialization_sha256": "encoder-hash",
        "encoder_artifact_file_sha256": "encoder-file-hash",
    }
    records = [
        SampleRecord(
            nodule_uid=f"n-{index:03d}",
            patient_key=f"patient-{index:03d}",
            roi_path=tmp_path / "unused.npz",
            target_normalized=index / 478.0,
            target_1_to_5=1.0 + 4.0 * index / 478.0,
            extreme_binary_eligible=index % 2 == 0,
            extreme_binary_label=index % 2 if index % 2 == 0 else None,
        )
        for index in range(479)
    ]
    output_root = tmp_path / "runs"
    output = output_root / "fold_0"
    output.mkdir(parents=True)
    best_path = output / "best.pt"
    best_path.write_bytes(b"sealed-best")
    provenance = _provenance(scientific, execution_hash, split, initialization)
    completion = {
        **provenance,
        "status": "TRAINING_COMPLETE_TEST_NOT_EVALUATED",
        "test_evaluated": False,
        "best_checkpoint_sha256": module.sha256_file(best_path),
        "best_epoch_index": 2,
        "best_validation_mse": 0.1,
    }
    module._atomic_json(output / "training_complete.json", completion)

    monkeypatch.setattr(
        module,
        "_prepare_sources",
        lambda *args, **kwargs: (
            scientific,
            execution,
            execution_hash,
            split,
            object(),
            object(),
            Path("encoder.pt"),
        ),
    )
    monkeypatch.setattr(
        module,
        "_load_best_model",
        lambda *args, **kwargs: (torch.nn.Linear(1, 1), initialization, {"epoch_index": 2}),
    )
    monkeypatch.setattr(module, "build_partition_records", lambda *args, **kwargs: records)
    inference_calls = {"count": 0}

    def fake_predict(*args: object, **kwargs: object) -> list[dict[str, object]]:
        inference_calls["count"] += 1
        return [
            {
                "nodule_uid": record.nodule_uid,
                "patient_key": record.patient_key,
                "target_normalized": record.target_normalized,
                "target_1_to_5": record.target_1_to_5,
                "malignancy_raw_score": record.target_normalized,
                "malignancy_score_normalized": record.target_normalized,
                "malignancy_score_1_to_5": record.target_1_to_5,
                "extreme_binary_eligible": record.extreme_binary_eligible,
                "extreme_binary_label": record.extreme_binary_label,
            }
            for record in records
        ]

    monkeypatch.setattr(module, "predict_records", fake_predict)
    monkeypatch.setattr(module, "_prediction_plot", lambda rows, path: path.write_bytes(b"plot"))
    original_atomic_json = module._atomic_json
    failed_once = {"value": False}

    def fail_after_predictions(path: Path, payload: object) -> None:
        if path.name == "metrics.json" and not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("simulated-crash")
        original_atomic_json(path, payload)

    monkeypatch.setattr(module, "_atomic_json", fail_after_predictions)
    common = {
        "scientific_config_path": Path("scientific.yaml"),
        "execution_config_path": Path("execution.yaml"),
        "manifest_path": Path("manifest.parquet"),
        "roi_index_path": Path("roi.parquet"),
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
        "output_root": output_root,
    }
    with pytest.raises(RuntimeError, match="simulated-crash"):
        _evaluate_test_once_locked(**common)
    assert inference_calls["count"] == 1
    assert (output / "test_claim.json").is_file()
    assert (output / "test_predictions.parquet").is_file()
    assert not (output / "test_evaluation.json").exists()

    monkeypatch.setattr(module, "_atomic_json", original_atomic_json)
    completed = _evaluate_test_once_locked(**common)
    assert completed["evaluation"]["status"] == "TEST_EVALUATED_ONCE"
    assert inference_calls["count"] == 1
    partially_sealed = json.loads((output / "training_complete.json").read_text(encoding="utf-8"))
    partially_sealed["status"] = "TRAINING_COMPLETE_TEST_NOT_EVALUATED"
    partially_sealed["test_evaluated"] = False
    partially_sealed.pop("test_evaluation_sha256")
    original_atomic_json(output / "training_complete.json", partially_sealed)
    recovered = _evaluate_test_once_locked(**common)
    assert recovered["recovered_after_evaluation_seal"] is True
    assert inference_calls["count"] == 1
    with pytest.raises(FileExistsError, match="ALREADY_EVALUATED"):
        _evaluate_test_once_locked(**common)


def test_prediction_provenance_and_exact_uid_set_tampering_are_blocked(tmp_path: Path) -> None:
    record = _record(tmp_path, "expected", value=0.5)
    provenance = {"fold_index": 0, "model": "blackbox", "checkpoint_sha256": "best"}
    row = {
        "nodule_uid": record.nodule_uid,
        "patient_key": record.patient_key,
        "target_normalized": 0.5,
        "target_1_to_5": 3.0,
        "malignancy_raw_score": 0.5,
        "malignancy_score_normalized": 0.5,
        "malignancy_score_1_to_5": 3.0,
        **provenance,
    }
    import pandas as pd

    _validate_test_prediction_frame(pd.DataFrame([row]), [record], provenance)
    wrong_uid = dict(row, nodule_uid="replacement")
    with pytest.raises(ValueError, match="UID_SET_MISMATCH"):
        _validate_test_prediction_frame(pd.DataFrame([wrong_uid]), [record], provenance)
    wrong_provenance = dict(row, checkpoint_sha256="other")
    with pytest.raises(ValueError, match="PROVENANCE_MISMATCH"):
        _validate_test_prediction_frame(pd.DataFrame([wrong_provenance]), [record], provenance)


def test_interrupted_and_resumed_training_matches_uninterrupted_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch
    import lidc_baseline.p5_blackbox as module

    scientific = {
        "protocol": {"version": "Baseline-v2"},
        "reproducibility": {"base_seed": 20260808},
    }
    execution = copy.deepcopy(load_config(EXECUTION_CONFIG))
    execution["reference_reported"]["epochs"] = 4
    split = {
        "fold_index": 0,
        "split_sha256": "split",
        "partitions": {
            "train": {"summary": {"nodules": 2}},
            "validation": {"summary": {"nodules": 1}},
        },
    }
    initialization = {
        "fold_seed": 20260808,
        "head_initialization_seed": 123,
        "head_initialization_sha256": "head",
        "head_seed_derivation": "derivation",
        "encoder_initialization_sha256": "encoder",
        "encoder_artifact_file_sha256": "encoder-file",
    }

    def sources(*args: object, **kwargs: object) -> tuple[object, ...]:
        return scientific, execution, "execution", split, object(), object(), Path("encoder.pt")

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.1]]))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            return image[:, :1] * self.weight

    records = [object(), object()]
    monkeypatch.setattr(module, "_prepare_sources", sources)
    monkeypatch.setattr(module, "build_initialized_model", lambda *args: (TinyModel(), initialization))
    monkeypatch.setattr(module, "build_partition_records", lambda *args, **kwargs: records)

    def fake_train(
        model: TinyModel,
        records: object,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        **kwargs: object,
    ) -> dict[str, object]:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model.weight - 0.75).mean()
        loss.backward()
        optimizer.step()
        return {"mse": float(loss.detach()), "sample_count": 2, "nodule_set_sha256": "train-set"}

    def fake_predict(model: TinyModel, records: object, device: torch.device, **kwargs: object) -> list[dict[str, float]]:
        return [{"malignancy_raw_score": float(model.weight.detach()), "target_normalized": 0.75}]

    monkeypatch.setattr(module, "train_one_epoch", fake_train)
    monkeypatch.setattr(module, "predict_records", fake_predict)
    monkeypatch.setattr(module, "_training_plot", lambda rows, path: path.write_bytes(b"plot"))
    monkeypatch.setattr(module, "_runtime_environment", lambda device: {"device_type": "cpu"})
    common = {
        "scientific_config_path": Path("scientific"),
        "execution_config_path": Path("execution"),
        "manifest_path": Path("manifest"),
        "roi_index_path": Path("roi"),
        "fold_index": 0,
        "device_name": "cpu",
        "num_workers": 0,
    }
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    train_fold(**common, output_root=uninterrupted_root, resume=False)
    train_fold(
        **common,
        output_root=resumed_root,
        resume=False,
        _stop_after_epoch_for_test=1,
    )
    history_path = resumed_root / "fold_0" / "history.csv"
    history_path.write_text("corrupted partial history\n", encoding="utf-8")
    train_fold(**common, output_root=resumed_root, resume=True)

    first = torch.load(uninterrupted_root / "fold_0" / "last.pt", map_location="cpu", weights_only=False)
    second = torch.load(resumed_root / "fold_0" / "last.pt", map_location="cpu", weights_only=False)
    assert first["epoch_index"] == second["epoch_index"] == 3
    assert first["best_epoch_index"] == second["best_epoch_index"]
    assert first["best_validation_mse"] == second["best_validation_mse"]
    assert first["scheduler_state_dict"] == second["scheduler_state_dict"]
    assert first["optimizer_state_dict"] == second["optimizer_state_dict"]
    assert torch.equal(first["model_state_dict"]["weight"], second["model_state_dict"]["weight"])
    stable_columns = [
        "epoch_index",
        "train_mse",
        "validation_mse",
        "learning_rate_start",
        "learning_rate_end",
        "scheduler_decayed",
        "scheduler_best",
        "scheduler_bad_epoch_counter",
        "train_sample_count",
        "train_nodule_set_sha256",
    ]
    import pandas as pd

    left = pd.read_csv(uninterrupted_root / "fold_0" / "history.csv")[stable_columns]
    right = pd.read_csv(resumed_root / "fold_0" / "history.csv")[stable_columns]
    pd.testing.assert_frame_equal(left, right)
