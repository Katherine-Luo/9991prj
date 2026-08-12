from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from lidc_baseline import p9_spatial as spatial
from lidc_baseline import p9_spatial_lifecycle as lifecycle
from lidc_baseline.p4_prepare import sha256_file


def _approval(path: Path, stage_a_sha: str = "a" * 64) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "phase": "P9",
        "status": "USER_APPROVED_FORMAL_SPATIAL_EXECUTION",
        "jobs": 20,
        "models": list(lifecycle.MODEL_ORDER),
        "folds": list(range(5)),
        "stage_a_preflight_sha256": stage_a_sha,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _stage_a_report() -> dict[str, object]:
    models = []
    for model in lifecycle.MODEL_ORDER:
        models.append(
            {
                "model": model,
                "checkpoint_sha256": "c" * 64,
                "implementation_sha256": lifecycle.implementation_sha256(),
                "p4_encoder_initialization_sha256": "e" * 64,
                "model_semantic_sha256_before": "f" * 64,
                "model_semantic_sha256_after": "f" * 64,
                "projected_slowest_fold_hours": 1.0,
                "target_reports": [
                    {
                        "valid_map_count": 1,
                        "true_occlusion_batch_size_observed": 16,
                    }
                    for _ in spatial.target_specs(model)
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "partition": "validation",
        "test_read": False,
        "optimizer_or_parameter_update": False,
        "target_path_count": 28,
        "true_batch_16_occlusion_forward": True,
        "raw_fp32_shard_roundtrip": True,
        "saliency_voxel_count": spatial.SALIENCY_VOXELS,
        "matched_random_masks": spatial.RANDOM_MASKS,
        "actual_positive_error_increase_count": 1,
        "actual_negative_error_increase_count": 1,
        "peak_reserved_fraction": 0.1,
        "scratch_free_to_projected_peak_ratio": 2.0,
        "p9_spatial_approved": "0",
        "models": models,
        "runtime": {
            "device_type": "cuda",
            "gpu_name": "NVIDIA H200",
            "fp32": True,
            "amp": False,
            "bf16": False,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
        },
    }


def test_formal_approval_requires_environment_and_exact_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "approval.json"
    _approval(path)
    monkeypatch.delenv(spatial.SPATIAL_APPROVAL_ENV, raising=False)
    with pytest.raises(PermissionError, match="USER_APPROVAL_REQUIRED"):
        lifecycle.require_formal_approval_record(path)
    monkeypatch.setenv(spatial.SPATIAL_APPROVAL_ENV, "1")
    assert lifecycle.require_formal_approval_record(path)["jobs"] == 20
    payload = json.loads(path.read_text())
    payload["jobs"] = 19
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="RECORD_INVALID"):
        lifecycle.require_formal_approval_record(path)


def test_stage_a_validator_requires_actual_h200_occlusion_and_sign_evidence(
    tmp_path: Path,
) -> None:
    stage_a_path = tmp_path / "stage_a" / "preflight.json"
    lifecycle._atomic_json(stage_a_path, _stage_a_report())
    approval = {
        "stage_a_preflight_sha256": sha256_file(stage_a_path),
    }
    assert lifecycle._validated_stage_a_artifact(tmp_path, approval)["status"] == "PASS"
    tampered = _stage_a_report()
    tampered["models"][0]["target_reports"][0][
        "true_occlusion_batch_size_observed"
    ] = 0
    lifecycle._atomic_json(stage_a_path, tampered)
    approval["stage_a_preflight_sha256"] = sha256_file(stage_a_path)
    with pytest.raises(PermissionError, match="GATES_NOT_PASS"):
        lifecycle._validated_stage_a_artifact(tmp_path, approval)


def test_stage_a_roundtrip_binds_generated_raw_map_bytes() -> None:
    generated = [
        {
            "model": "blackbox",
            "fold_index": 0,
            "nodule_uid": "uid",
            "target": "malignancy",
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "implementation_sha256": "c" * 64,
            "map": np.ones(spatial.MAP_SHAPE, dtype=np.float32),
        }
    ]
    expected_hash = hashlib.sha256(
        generated[0]["map"].astype("<f4").tobytes()
    ).hexdigest()
    restored = [{**generated[0], "map_sha256": expected_hash}]
    lifecycle._validate_stage_a_roundtrip_identity(generated, restored)
    restored[0]["map_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="ROUNDTRIP_MISMATCH"):
        lifecycle._validate_stage_a_roundtrip_identity(generated, restored)


def test_target_layer_resolution_matches_frozen_path() -> None:
    conv2 = object()
    model = SimpleNamespace(
        encoder=SimpleNamespace(
            denseblock4=SimpleNamespace(
                denselayer16=SimpleNamespace(layers=SimpleNamespace(conv2=conv2))
            )
        )
    )
    assert lifecycle._resolve_target_layer(model) is conv2


def test_standard_cbm_forward_uses_real_p6_contribution_contract() -> None:
    class ConceptModel(torch.nn.Module):
        def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"canonical_vector": image.reshape(image.shape[0], 16)}

    task_head = torch.nn.Linear(16, 1)
    bundle = SimpleNamespace(
        model_name="standard_cbm",
        model=ConceptModel(),
        task_head=task_head,
    )
    image = (
        torch.arange(32, dtype=torch.float32).reshape(2, 1, 2, 2, 4) / 31.0
    )
    output = lifecycle._forward_bundle(bundle, image)
    assert output["malignancy_raw_score"].shape == (2, 1)
    output["malignancy_raw_score"].sum().backward()
    assert task_head.weight.grad is not None


def test_persisted_concept_logits_parse_real_json_schema_strictly() -> None:
    continuous = spatial.TargetSpec("subtlety", "continuous_concept", "logit")
    internal = spatial.TargetSpec(
        "internalStructure", "categorical_concept", "predicted"
    )
    calcification = spatial.TargetSpec(
        "calcification", "categorical_concept", "predicted"
    )
    assert lifecycle._persisted_target_score(
        {"subtlety_logits": json.dumps([0.125])}, continuous, None
    ) == pytest.approx(0.125)
    assert lifecycle._persisted_target_score(
        {"internalStructure_logits": json.dumps([0.1, 0.2, 0.3, 0.4])},
        internal,
        2,
    ) == pytest.approx(0.3)
    assert lifecycle._persisted_target_score(
        {"calcification_logits": json.dumps([0, 1, 2, 3, 4, 5])},
        calcification,
        5,
    ) == pytest.approx(5.0)
    for value in ("not-json", json.dumps([1.0, 2.0]), json.dumps([float("inf")])):
        with pytest.raises(ValueError, match="LOGIT"):
            lifecycle._persisted_target_score(
                {"subtlety_logits": value}, continuous, None
            )


def test_cli_wires_preflight_output_root_and_formal_approval_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def fake_preflight(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(lifecycle, "preflight", fake_preflight)
    output = tmp_path / "stage-a.json"
    root = tmp_path / "private"
    assert (
        spatial.main(
            [
                "preflight",
                "--fold",
                "0",
                "--output",
                str(output),
                "--p9-root",
                str(root),
            ]
        )
        == 0
    )
    assert observed["output_path"] == output
    assert observed["p9_root"] == root
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"

    def fake_run(**kwargs: object) -> dict[str, object]:
        observed.clear()
        observed.update(kwargs)
        return {"status": "PASS"}

    monkeypatch.setattr(lifecycle, "run_model_fold", fake_run)
    approval = tmp_path / "approval.json"
    assert (
        spatial.main(
            [
                "run",
                "--model",
                "blackbox",
                "--fold",
                "0",
                "--p9-root",
                str(root),
                "--approval-record",
                str(approval),
            ]
        )
        == 0
    )
    assert observed["approval_path"] == approval


def test_auxiliary_prediction_seal_reuses_exact_artifact_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(1, 1)
    bundle = lifecycle.FrozenModelBundle(
        "blackbox",
        0,
        model,
        None,
        {},
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path / "roi.parquet",
        "a" * 64,
        "b" * 64,
        tmp_path,
        {},
        "e" * 64,
    )
    records = [SimpleNamespace(nodule_uid="u1")]
    calls = {"count": 0}

    def predictions(*_args: object, **_kwargs: object) -> tuple[list[dict[str, object]], dict[str, float]]:
        calls["count"] += 1
        return [
            {
                "nodule_uid": "u1",
                "fold_index": 0,
                "model": "blackbox",
                "malignancy_raw_score": 0.1,
                "malignancy_score_1_to_5": 1.4,
                "target_normalized": 0.25,
                "target_1_to_5": 2.0,
            }
        ], {
            "normalized_reconstruction_max_abs_error": 0.0,
            "rating_reconstruction_max_abs_error": 0.0,
        }

    monkeypatch.setattr(lifecycle, "_prediction_rows", predictions)
    path = tmp_path / "validation.parquet"
    first = lifecycle._verify_or_write_auxiliary_predictions(
        bundle,
        records,
        device=torch.device("cpu"),
        output_path=path,
        partition="validation",
    )
    second = lifecycle._verify_or_write_auxiliary_predictions(
        bundle,
        records,
        device=torch.device("cpu"),
        output_path=path,
        partition="validation",
    )
    assert first == second
    assert calls["count"] == 1
    seal_path = path.with_suffix(".parquet.json")
    seal = json.loads(seal_path.read_text())
    seal["checkpoint_sha256"] = "c" * 64
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="REUSE_MISMATCH"):
        lifecycle._verify_or_write_auxiliary_predictions(
            bundle,
            records,
            device=torch.device("cpu"),
            output_path=path,
            partition="validation",
        )


def test_centering_requires_exact_train_membership_and_finite_matrix() -> None:
    bundle = SimpleNamespace(
        model_name="mixed_cem",
        fold_index=0,
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    rows = [
        {
            "nodule_uid": "u1",
            **{
                f"{group}_raw_contribution": float(index)
                for index, group in enumerate(lifecycle.CONCEPT_GROUP_ORDER)
            },
        }
    ]
    report = lifecycle._centering_from_contribution_rows(bundle, rows, ["u1"])
    report.update(
        normalized_reconstruction_max_abs_error=0.0,
        rating_reconstruction_max_abs_error=0.0,
    )
    lifecycle._validate_centering_report(report, bundle, ["u1"])
    assert report["sample_count"] == 1
    assert report["rating_group_means"]["subtlety"] == 0.0
    tampered = json.loads(json.dumps(report))
    tampered["rating_group_means"]["sphericity"] += 1e-4
    with pytest.raises(ValueError, match="GROUP_VALUE_MISMATCH"):
        lifecycle._validate_centering_report(tampered, bundle, ["u1"])
    with pytest.raises(ValueError, match="UID_MEMBERSHIP"):
        lifecycle._centering_from_contribution_rows(bundle, rows, ["other"])
    rows[0]["subtlety_raw_contribution"] = float("nan")
    with pytest.raises(ValueError, match="MATRIX_INVALID"):
        lifecycle._centering_from_contribution_rows(bundle, rows, ["u1"])


def test_faithfulness_validator_accepts_roundtrip_and_rejects_material_change() -> None:
    payload = spatial.build_faithfulness_record(
        score_original=0.5,
        target_normalized=1.0,
        saliency_score_occluded=0.25,
        random_scores_occluded=np.linspace(0.1, 0.9, 20),
    )
    lifecycle._validate_faithfulness_payload(payload)
    payload["random_error_increase_values"][0] += 1e-4
    with pytest.raises(ValueError, match="SUMMARY_MISMATCH"):
        lifecycle._validate_faithfulness_payload(payload)


def test_final_verifier_binds_sources_auxiliary_runtime_and_spatial_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "p9" / "spatial" / "blackbox" / "fold_0"
    output.mkdir(parents=True)
    checkpoint_sha = "c" * 64
    config_sha = "d" * 64
    model = torch.nn.Linear(1, 1)
    split = {
        "partitions": {
            "train": {"nodule_uids": ["train"]},
            "validation": {"nodule_uids": ["validation"]},
            "test": {"nodule_uids": ["test"]},
        }
    }
    bundle = lifecycle.FrozenModelBundle(
        "blackbox",
        0,
        model,
        None,
        split,
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path / "roi.parquet",
        checkpoint_sha,
        config_sha,
        tmp_path,
        {},
        "e" * 64,
    )
    records = {
        name: [SimpleNamespace(nodule_uid=uid)]
        for name, uid in (
            ("train", "train"),
            ("validation", "validation"),
            ("test", "test"),
        )
    }
    monkeypatch.setattr(lifecycle, "load_frozen_model_bundle", lambda *_a, **_k: bundle)
    monkeypatch.setattr(lifecycle, "partition_records", lambda _bundle, name: records[name])

    oof_path = tmp_path / "oof.parquet"
    pd.DataFrame([{"nodule_uid": "test", "fold_index": 0}]).to_parquet(oof_path)
    monkeypatch.setattr(lifecycle, "_oof_path", lambda _model: oof_path)

    stage_a_path = tmp_path / "p9" / "stage_a" / "preflight.json"
    lifecycle._atomic_json(stage_a_path, _stage_a_report())
    approval_path = tmp_path / "approval.json"
    approval = _approval(approval_path, sha256_file(stage_a_path))
    monkeypatch.setattr(lifecycle, "SPATIAL_APPROVAL_DEFAULT", approval_path)

    validation_path = output / "validation_predictions.parquet"
    pd.DataFrame(
        [
            {
                "nodule_uid": "validation",
                "fold_index": 0,
                "model": "blackbox",
                "malignancy_raw_score": 0.1,
                "malignancy_score_1_to_5": 1.4,
                "target_normalized": 0.25,
                "target_1_to_5": 2.0,
            }
        ]
    ).to_parquet(validation_path)
    validation_seal = {
        "schema_version": 1,
        "status": "P9_AUXILIARY_PREDICTIONS_COMMITTED",
        "partition": "validation",
        "model": "blackbox",
        "fold_index": 0,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "sample_count": 1,
        "nodule_uid_set_sha256": lifecycle._uid_set_sha256(["validation"]),
        "file": validation_path.name,
        "file_sha256": sha256_file(validation_path),
    }
    lifecycle._atomic_json(validation_path.with_suffix(".parquet.json"), validation_seal)

    faithfulness = spatial.build_faithfulness_record(
        score_original=0.5,
        target_normalized=1.0,
        saliency_score_occluded=0.25,
        random_scores_occluded=np.linspace(0.1, 0.9, 20),
    )
    shard_path = output / "shard_0000.parquet"
    shard_seal = spatial.write_map_shard(
        shard_path,
        [
            {
                "nodule_uid": "test",
                "model": "blackbox",
                "fold_index": 0,
                "target": "malignancy",
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "implementation_sha256": lifecycle.implementation_sha256(),
                "map": np.ones(spatial.MAP_SHAPE, dtype=np.float32),
                "faithfulness": faithfulness,
            }
        ],
    )
    completion = {
        "schema_version": 1,
        "status": "P9_SPATIAL_MODEL_FOLD_COMPLETE",
        "model": "blackbox",
        "fold_index": 0,
        "sample_count": 1,
        "target_count_per_sample": 1,
        "expected_map_records": 1,
        "shard_count": 1,
        "shard_file_sha256": {shard_path.name: shard_seal["file_sha256"]},
        "validation_predictions_file_sha256": sha256_file(validation_path),
        "validation_predictions_seal_sha256": sha256_file(
            validation_path.with_suffix(".parquet.json")
        ),
        "train_contribution_centering_sha256": None,
        "checkpoint_sha256": checkpoint_sha,
        "p4_encoder_initialization_sha256": "e" * 64,
        "implementation_sha256": lifecycle.implementation_sha256(),
        "source_oof_sha256": sha256_file(oof_path),
        "model_semantic_sha256_before": lifecycle.bundle_state_sha256(bundle),
        "model_semantic_sha256_after": lifecycle.bundle_state_sha256(bundle),
        "optimizer_or_parameter_update": False,
        "second_committed_test_evaluation": False,
        "approval_record_sha256": sha256_file(approval_path),
        "stage_a_preflight_sha256": approval["stage_a_preflight_sha256"],
        "peak_reserved_fraction": 0.1,
        "runtime": {
            "device_type": "cuda",
            "gpu_name": "NVIDIA H200",
            "fp32": True,
            "amp": False,
            "bf16": False,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "deterministic_algorithms": True,
            "deterministic_warn_only": True,
        },
    }
    lifecycle._atomic_json(output / "spatial_complete.json", completion)
    report = lifecycle.verify_model_fold(
        "blackbox",
        0,
        p9_root=tmp_path / "p9",
        approval_path=approval_path,
    )
    assert report["status"] == "PASS"
    assert report["valid_map_count"] == 1
