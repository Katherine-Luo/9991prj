import hashlib
import importlib
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


def test_config_module_is_importable() -> None:
    assert importlib.util.find_spec("lidc_baseline.config") is not None


def require_callable(name: str) -> Callable[..., object]:
    module = importlib.import_module("lidc_baseline.config")
    value = getattr(module, name, None)
    assert callable(value), f"lidc_baseline.config.{name} must be callable"
    return value


def test_canonical_yaml_is_sorted_utf8_with_lf() -> None:
    canonical_yaml = require_callable("canonical_yaml")

    rendered = canonical_yaml({"beta": {"z": 2, "a": 1}, "alpha": [3, 4]})

    assert rendered == b"alpha:\n- 3\n- 4\nbeta:\n  a: 1\n  z: 2\n"


def test_config_sha256_uses_canonical_yaml_bytes() -> None:
    compute_config_sha256 = require_callable("compute_config_sha256")
    source = {"beta": {"z": 2, "a": 1}, "alpha": [3, 4]}
    expected = "379dbb2d2637c727476200ab9f37a669e0ca12c3a81a7b24c15349881ac1693a"

    assert compute_config_sha256(source) == expected
    assert expected == hashlib.sha256(
        b"alpha:\n- 3\n- 4\nbeta:\n  a: 1\n  z: 2\n"
    ).hexdigest()


def test_fold_seed_is_additive_and_rejects_invalid_fold() -> None:
    fold_seed = require_callable("fold_seed")

    assert [fold_seed(20260808, fold) for fold in range(5)] == [
        20260808,
        20260809,
        20260810,
        20260811,
        20260812,
    ]
    with pytest.raises(ValueError, match="fold_index"):
        fold_seed(20260808, -1)
    with pytest.raises(ValueError, match="fold_index"):
        fold_seed(20260808, 5)


def test_load_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    load_config = require_callable("load_config")
    source = tmp_path / "invalid.yaml"
    source.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(source)


def test_freeze_config_is_idempotent_and_refuses_different_overwrite(
    tmp_path: Path,
) -> None:
    freeze_config = require_callable("freeze_config")
    source = tmp_path / "source.yaml"
    resolved = tmp_path / "resolved.yaml"
    digest = tmp_path / "resolved.sha256"
    source.write_text("beta: 2\nalpha: 1\n", encoding="utf-8")

    first = freeze_config(source, resolved, digest)
    first_resolved = resolved.read_bytes()
    first_digest = digest.read_bytes()
    second = freeze_config(source, resolved, digest)

    assert first == second
    assert first_resolved == b"alpha: 1\nbeta: 2\n"
    assert first_digest == f"{first}\n".encode()
    assert resolved.stat().st_mode & 0o777 == 0o444
    assert digest.stat().st_mode & 0o777 == 0o444

    source.write_text("alpha: 999\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="different content"):
        freeze_config(source, resolved, digest)
    assert resolved.read_bytes() == first_resolved
    assert digest.read_bytes() == first_digest


def test_freeze_cli_creates_resolved_yaml_and_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    resolved = tmp_path / "resolved.yaml"
    digest = tmp_path / "resolved.sha256"
    source.write_text("beta: 2\nalpha: 1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lidc_baseline.config",
            "freeze",
            "--source",
            str(source),
            "--resolved",
            str(resolved),
            "--digest",
            str(digest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert resolved.read_bytes() == b"alpha: 1\nbeta: 2\n"
    assert digest.read_text(encoding="utf-8").strip() == hashlib.sha256(
        resolved.read_bytes()
    ).hexdigest()


def test_baseline_config_captures_frozen_protocol() -> None:
    load_config = require_callable("load_config")
    config = load_config(Path("configs/baseline_v1.yaml"))

    assert config["protocol"]["version"] == "Baseline-v1"
    assert config["protocol"]["task_name"] == (
        "Radiologist-assessed pulmonary nodule malignancy classification"
    )
    assert config["reproducibility"]["base_seed"] == 20260808
    assert config["reproducibility"]["fold_seed_rule"] == "base_seed + fold_index"
    assert config["cohort"]["primary_annotation_class"] == "nodule >=3 mm"
    assert config["cohort"]["reference_reconciliation"]["hard_gate"] is False
    assert config["cohort"]["audit_fields"]["per_concept_valid_reader_counts"] is True
    assert config["cohort"]["stable_nodule_uid"]["pylidc_sql_id_only"] is False
    assert config["labels"]["uncertain"]["excluded_from_primary"] is True
    assert len(config["concepts"]["groups"]) == 8
    assert config["roi"]["shape"] == [64, 64, 64]
    assert config["roi"]["image_interpolation"] == "trilinear"
    assert config["roi"]["mask_interpolation"] == "nearest"
    assert config["splits"]["outer_folds"] == 5
    assert config["splits"]["validation_fraction_of_development"] == 0.125
    assert config["encoder_initialization"]["shared_across_models"] is True
    assert config["models"]["cem"]["sample_conditioned_states"] is True
    assert config["models"]["gam"]["subnetworks_per_group"] == 5
    assert config["models"]["gam"]["ensemble_weighting"] == "learned_softmax"
    assert config["models"]["gam"]["group_contribution"] == "softmax_weighted_sum"
    for model in ("standard_cbm", "cem", "gam"):
        assert "concept_specific_gradcams" in config["models"][model]["output_scope"]
        assert "centered_group_contributions" in config["models"][model]["output_scope"]
    assert config["losses"]["concept"]["group_reduction"] == "mean_of_8_groups"
    assert config["thresholds"]["primary"] == "validation_youden_j"
    assert config["interventions"]["random_permutations_per_fold"] == 100
    assert config["statistics"]["bootstrap_replicates"] == 2000
    assert config["gradcam"]["blackbox_maps_per_sample"] == 1
    assert config["gradcam"]["concept_model_maps_per_sample"] == 9
    assert config["occlusion"]["voxel_count"] == 26215
    assert config["training"]["max_epochs"] == 80
