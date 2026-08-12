from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.p6_standard_cbm import (
    CATEGORICAL_CONCEPTS,
    CONCEPT_GROUP_ORDER,
    CONCEPT_OUTPUT_SIZES,
    CONTINUOUS_CONCEPTS,
)
from lidc_baseline.p9_audit import (
    _concept_scope,
    _faithfulness_aggregate,
    _intervention_group_contributions,
    _write_private_draws,
    bootstrap_results,
    intervention_results,
    spatial_results,
)
from lidc_baseline.p9_evaluation import MODEL_ORDER


def _distribution(size: int, index: int) -> list[float]:
    value = [0.0] * size
    value[index % size] = 1.0
    return value


def _concept_frame(model: str, count: int = 8) -> pd.DataFrame:
    rows = []
    for index in range(count):
        targets = {}
        row: dict[str, object] = {
            "fold_index": index % 5,
            "internalStructure_modal_tie": False,
            "calcification_modal_tie": False,
        }
        for group in CONCEPT_GROUP_ORDER:
            size = int(CONCEPT_OUTPUT_SIZES[group])
            if group in CONTINUOUS_CONCEPTS:
                target = [index / max(1, count - 1)]
                prediction = [min(1.0, target[0] + 0.05)]
            else:
                target = _distribution(size, index)
                prediction = [0.1 / (size - 1)] * size
                prediction[index % size] = 0.9
            targets[group] = target
            row[f"{group}_activated_prediction"] = json.dumps(prediction)
            if model != "standard_cbm":
                row[f"{group}_target"] = json.dumps(target)
        if model == "standard_cbm":
            row["concept_targets"] = json.dumps(targets)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "model", ["standard_cbm", "mixed_cem", "learned_softmax_gam"]
)
def test_concept_scope_reports_all_mixed_type_metrics(model: str) -> None:
    result = _concept_scope(_concept_frame(model), model)
    assert set(result) == set(CONCEPT_GROUP_ORDER)
    for group in CONTINUOUS_CONCEPTS:
        assert result[group]["sample_count"] == 8
        assert set(result[group]) >= {"mae", "rmse", "pearson", "spearman"}
    for group in CATEGORICAL_CONCEPTS:
        assert result[group]["soft_sample_count"] == 8
        assert result[group]["hard_sample_count"] == 8
        assert result[group]["true_tie_count"] == 0


def test_concept_scope_rejects_tie_flag_inconsistent_with_votes() -> None:
    frame = _concept_frame("mixed_cem")
    frame.loc[0, "internalStructure_modal_tie"] = True
    with pytest.raises(ValueError, match="TRUE_TIE_MISMATCH"):
        _concept_scope(frame, "mixed_cem")


def _task_frames() -> dict[str, pd.DataFrame]:
    patient = ["p0", "p1", "p2", "p3", "p4", "p5"]
    target_rating = np.asarray([1.0, 1.5, 2.0, 4.0, 4.5, 5.0])
    target = (target_rating - 1.0) / 4.0
    result = {}
    for model_index, model in enumerate(MODEL_ORDER):
        score = target + (model_index - 1.5) * 0.01
        result[model] = pd.DataFrame(
            {
                "nodule_uid": [f"n{i}" for i in range(6)],
                "patient_key": patient,
                "fold_index": [0, 1, 2, 3, 4, 0],
                "target_normalized": target,
                "target_1_to_5": target_rating,
                "malignancy_raw_score": score,
            }
        )
    return result


def test_patient_bootstrap_preserves_shared_draws_and_all_six_pairs() -> None:
    tracked, private = bootstrap_results(_task_frames(), base_seed=20260808)
    assert tracked["draws"] == 2000
    assert len(private["primary_draws"]) == 2000
    assert len(private["secondary_draws"]) == 2000
    assert len(tracked["paired_mae_A_minus_B"]) == 6
    assert len(tracked["paired_auroc_B_minus_A"]) == 6
    assert all(
        len(draw) == 6 for draw in private["primary_draws"][:10]
    )


def test_private_bootstrap_draws_are_preserved_individually(tmp_path: Path) -> None:
    path = tmp_path / "draws.parquet"
    digest = _write_private_draws(
        path,
        {
            "primary_draws": [np.asarray(["p0", "p1"])],
            "secondary_draws": [np.asarray(["p1", "p1"])],
        },
    )
    frame = pd.read_parquet(path)
    assert len(digest) == 64
    assert frame["kind"].tolist() == ["primary_draws", "secondary_draws"]
    assert json.loads(frame.iloc[1]["patient_keys_json"]) == ["p1", "p1"]


def test_standard_cbm_intervention_uses_gt_activated_group_and_frozen_task_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import lidc_baseline.p9_audit as audit

    task_head = torch.nn.Linear(16, 1)
    with torch.no_grad():
        task_head.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(1, -1) / 16)
        task_head.bias.zero_()
    bundle = SimpleNamespace(task_head=task_head)
    monkeypatch.setattr(
        audit, "load_frozen_model_bundle", lambda *_args, **_kwargs: bundle
    )
    frame = _concept_frame("standard_cbm", count=2)
    offset = 0
    expected = np.empty((2, 8), dtype=float)
    for group_index, group in enumerate(CONCEPT_GROUP_ORDER):
        size = int(CONCEPT_OUTPUT_SIZES[group])
        weights = task_head.weight.detach().numpy().reshape(-1)[offset : offset + size]
        targets = [json.loads(value)[group] for value in frame["concept_targets"]]
        predictions = [json.loads(value) for value in frame[f"{group}_activated_prediction"]]
        expected[:, group_index] = np.asarray(targets) @ weights
        frame[f"{group}_raw_contribution"] = np.asarray(predictions) @ weights
        offset += size
    predicted, intervened = _intervention_group_contributions(
        frame, "standard_cbm", 0
    )
    assert np.allclose(intervened, expected, atol=1e-7)
    assert predicted.shape == intervened.shape == (2, 8)


def test_intervention_curves_use_trapezoid_and_k_zero_exact_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lidc_baseline.p9_audit as audit

    frames = {}
    for model in MODEL_ORDER[1:]:
        frame = _concept_frame(model, count=10)
        frame["nodule_uid"] = [f"{model}-{index}" for index in range(10)]
        frame["patient_key"] = [f"patient-{index}" for index in range(10)]
        frame["fold_index"] = np.repeat(np.arange(5), 2)
        frame["target_1_to_5"] = np.tile([1.0, 5.0], 5)
        frame["target_normalized"] = np.tile([0.0, 1.0], 5)
        frame["malignancy_raw_score"] = np.tile([0.1, 0.9], 5)
        for group in CONCEPT_GROUP_ORDER:
            frame[f"{group}_raw_contribution"] = 0.0
        frames[model] = frame
    monkeypatch.setattr(
        audit,
        "_intervention_group_contributions",
        lambda frame, _model, _fold: (
            np.zeros((len(frame), 8)),
            np.zeros((len(frame), 8)),
        ),
    )
    result = intervention_results(frames, base_seed=20260808)
    for report in result.values():
        random = report["random_permutations"]
        assert random["pooled_original_scale_mae_mean"][0] == report[
            "baseline_original_scale_mae"
        ]
        assert random["pooled_auroc_mean"][0] == report["baseline_auroc"]
        assert random["iMAE"] == pytest.approx(
            report["baseline_original_scale_mae"]
        )
        assert random["iAUC"] == pytest.approx(report["baseline_auroc"])
        assert random["Delta_iMAE"] == pytest.approx(0.0)
        assert random["Delta_iAUC"] == pytest.approx(0.0)


def _faithfulness_payload(offset: float = 0.0) -> dict[str, object]:
    return {
        "saliency_output_sensitivity": 0.5 + offset,
        "saliency_error_increase": 0.25 + offset,
        "saliency_greater_than_random_mean_output_sensitivity": True,
        "saliency_greater_than_random_mean_error_increase": False,
        "saliency_minus_random_mean_output_sensitivity": 0.2 + offset,
        "saliency_minus_random_mean_error_increase": -0.1 + offset,
    }


def test_faithfulness_aggregate_reports_saliency_minus_random_distribution() -> None:
    rows = [_faithfulness_payload(0.0), _faithfulness_payload(0.1)]
    report = _faithfulness_aggregate(rows, "output_sensitivity")
    assert report is not None
    comparison = report["saliency_minus_matched_random_mean"]
    assert comparison["mean"] == pytest.approx(0.25)
    assert comparison["sd"] == pytest.approx(0.05)
    assert comparison["median"] == pytest.approx(0.25)
    assert report["saliency_greater_than_matched_random_mean_rate"] == 1.0
    assert _faithfulness_aggregate([], "error_increase") is None


def test_spatial_results_retains_all_undefined_canonical_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lidc_baseline.p9_audit as audit
    from lidc_baseline.p9_spatial import target_specs

    monkeypatch.setattr(audit, "verify_all", lambda **_kwargs: {"jobs": 20})

    def fake_read(path: Path) -> list[dict[str, object]]:
        model = path.parent.parent.name
        fold = int(path.parent.name.split("_")[1])
        rows = []
        for spec in target_specs(model):
            undefined = model == "standard_cbm" and fold == 0 and spec.name == "subtlety"
            rows.append(
                {
                    "model": model,
                    "fold_index": fold,
                    "target": spec.name,
                    "status": "undefined" if undefined else "valid",
                    "faithfulness_json": (
                        None
                        if undefined
                        else json.dumps(_faithfulness_payload())
                    ),
                }
            )
        return rows

    monkeypatch.setattr(audit, "read_and_verify_map_shard", fake_read)
    for model in MODEL_ORDER:
        for fold in range(5):
            path = tmp_path / "spatial" / model / f"fold_{fold}" / "shard_0000.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    result = spatial_results(tmp_path)
    target = result["models"]["standard_cbm"]["folds"][0]["targets"]["subtlety"]
    assert target["valid_map_count"] == 0
    assert target["undefined_map_count"] == 1
    assert target["output_sensitivity"] is None
    assert target["error_increase"] is None
    assert set(result["models"]["standard_cbm"]["pooled_targets"]) == {
        spec.name for spec in target_specs("standard_cbm")
    }
