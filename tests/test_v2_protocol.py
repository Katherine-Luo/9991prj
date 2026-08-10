from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config
from lidc_baseline.regression import (
    extreme_binary_label,
    malignancy_stratum,
    normalize_malignancy_target,
    normalized_score_to_rating_scale,
)
from lidc_baseline.smoke import build_smoke_model, run_smoke


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1.0, 0.0), (3.0, 0.5), (5.0, 1.0)],
)
def test_v2_target_normalization_boundaries(rating: float, expected: float) -> None:
    assert normalize_malignancy_target(rating) == expected


@pytest.mark.parametrize("rating", [0.99, 5.01, float("nan")])
def test_v2_target_normalization_rejects_invalid_ratings(rating: float) -> None:
    with pytest.raises(ValueError):
        normalize_malignancy_target(rating)


def test_unconstrained_score_conversion_does_not_clip() -> None:
    assert normalized_score_to_rating_scale(-0.25) == 0.0
    assert normalized_score_to_rating_scale(1.25) == 6.0


@pytest.mark.parametrize(
    ("rating", "label", "stratum"),
    [
        (2.0, 0, "mean_le_2"),
        (2.5, None, "mean_gt_2_lt_3"),
        (3.0, None, "mean_eq_3"),
        (3.5, None, "mean_gt_3_lt_4"),
        (4.0, 1, "mean_ge_4"),
    ],
)
def test_extreme_labels_and_five_strata(
    rating: float,
    label: int | None,
    stratum: str,
) -> None:
    assert extreme_binary_label(rating) == label
    assert malignancy_stratum(rating) == stratum


def test_v2_config_registers_one_unbounded_linear_task_head() -> None:
    config = load_config("configs/baseline_v2.yaml")
    assert config["protocol"]["version"] == "Baseline-v2"
    assert config["task"]["head"] == {
        "type": "linear",
        "output_activation": "none",
        "output_constraint": "unbounded",
        "clipping": False,
    }
    assert config["task"]["outputs"] == [
        "malignancy_raw_score",
        "malignancy_score_normalized",
        "malignancy_score_1_to_5",
    ]
    assert config["concepts"]["malignancy_is_input_concept"] is False
    assert len(config["concepts"]["groups"]) == 8
    assert all(
        model.get("independent_binary_head") is False
        for model in config["models"].values()
        if isinstance(model, dict) and "independent_binary_head" in model
    )
    assert config["losses"]["task"]["implementation"] == "MSELoss"
    assert config["metrics"]["clipping"] is False
    assert config["secondary_extreme_evaluation"]["probability_metrics"] == []
    assert config["katana"]["gpu_request"]["gpu_model"] == "L40S"


def test_v2_resolved_config_and_digest_match_source() -> None:
    config = load_config("configs/baseline_v2.yaml")
    resolved = Path("configs/baseline_v2.resolved.yaml")
    digest = Path("configs/baseline_v2.sha256")
    expected = compute_config_sha256(config)
    assert resolved.read_bytes() == canonical_yaml(config)
    assert digest.read_text(encoding="ascii").strip() == expected
    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == expected
    assert resolved.stat().st_mode & 0o777 == 0o444
    assert digest.stat().st_mode & 0o777 == 0o444


def test_protocol_index_has_one_active_protocol() -> None:
    index = Path("docs/PROTOCOL_INDEX.md").read_text(encoding="utf-8")
    table_rows = [line for line in index.splitlines() if line.startswith("| Baseline-")]
    assert sum("`ACTIVE`" in row for row in table_rows) == 1
    assert "| Baseline-v2 | `ACTIVE`" in index
    assert "| Baseline-v1 | `SUPERSEDED`" in index


@pytest.mark.local_audit
def test_v2_tracked_smoke_audits_share_config_and_linear_objective() -> None:
    digest = Path("configs/baseline_v2.sha256").read_text(encoding="ascii").strip()
    reports = {
        device: json.loads(
            Path(f"artifacts/baseline_v2/audit/v2m/{device}.json").read_text(
                encoding="utf-8"
            )
        )
        for device in ("cpu", "mps", "cuda")
    }
    for device, report in reports.items():
        assert report["status"] == "PASS", device
        assert report["device_requested"] == device
        assert report["config_sha256"] == digest
        assert report["loss_name"] == "mse"
        assert report["task_output_activation"] == "none"
        assert report["task_output_constraint"] == "unbounded"
        assert report["task_output_clipping"] is False
        assert report["post_output_transform_applied"] is False
        assert report["output_shape"] == [1, 1]
    assert reports["cuda"]["versions"]["gpu_name"] == "NVIDIA L40S"


def test_densenet_smoke_output_remains_unconstrained() -> None:
    model = build_smoke_model("cpu")
    final = model.class_layers.out
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(2.0)
    output = model(torch.zeros(1, 1, 64, 64, 64))
    assert output.item() == 2.0


@pytest.mark.integration
def test_v2_cpu_smoke_uses_mse_and_linear_output(tmp_path: Path) -> None:
    report = run_smoke(
        device_name="cpu",
        output_path=tmp_path / "cpu.json",
        config_path=Path("configs/baseline_v2.yaml"),
    )
    assert report["protocol_version"] == "Baseline-v2"
    assert report["loss_name"] == "mse"
    assert report["task_type"] == "continuous_regression"
    assert report["task_output_activation"] == "none"
    assert report["task_output_constraint"] == "unbounded"
    assert report["task_output_clipping"] is False
    assert report["post_output_transform_applied"] is False
    assert report["output_shape"] == [1, 1]
    assert report["loss_finite"] is True
    assert report["gradients_nonzero"] is True
