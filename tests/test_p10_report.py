from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest

from lidc_baseline.p10_report import (
    CONFIG_RESOLVED_DEFAULT,
    CONFIG_SHA_DEFAULT,
    MODEL_ORDER,
    P9_AUDIT_ROOT_DEFAULT,
    P9_CONFIG_SHA256,
    _data_dictionary_rows,
    _source_field_path,
    _verify_p9_tracked_report_hashes,
    _wrap_text,
    assert_public_payload,
    build_markdown,
    build_report_data,
    execution_registry,
    export_tables,
    extract_numeric_tokens,
    sha256_file,
    validate_execution_config,
    verify_bilingual_markdown,
    verify_inputs,
    verify_public_outputs,
)


def test_p10_execution_config_is_canonical() -> None:
    payload = validate_execution_config(CONFIG_RESOLVED_DEFAULT, CONFIG_SHA_DEFAULT)
    assert payload["phase"] == "P10"


@pytest.mark.local_audit
def test_p10_verify_inputs_closes_frozen_p5_through_p9_boundary() -> None:
    result = verify_inputs(audit_root=P9_AUDIT_ROOT_DEFAULT)
    assert result["status"] == "PASS"
    assert result["unique_nodules"] == 2633
    assert result["unique_patients"] == 868
    assert result["fold_counts"] == [479, 502, 539, 549, 564]
    assert result["new_training"] is False
    assert result["new_test_inference"] is False
    assert result["new_scientific_jobs"] is False


@pytest.mark.local_audit
def test_execution_registry_preserves_scheduler_and_scientific_status() -> None:
    rows = execution_registry()
    assert len(rows) == 40
    assert {(row["phase"], row["model"], row["fold"]) for row in rows} == {
        (phase, model, fold)
        for phase, model in (
            ("P5", "Black-box"),
            ("P6", "Standard CBM"),
            ("P7", "Mixed-type CEM"),
            ("P8", "Learned-softmax GAM"),
            ("P9", "Black-box"),
            ("P9", "Standard CBM"),
            ("P9", "Mixed-type CEM"),
            ("P9", "Learned-softmax GAM"),
        )
        for fold in range(5)
    }
    assert all(row["scientific_status"] == "PASS" for row in rows)
    p8 = [row for row in rows if row["phase"] == "P8"]
    assert [row["exit_status"] for row in p8] == [1, -18, -18, 1, -18]
    assert [row["run_count"] for row in p8] == [1, 21, 21, 1, 21]
    assert all(row["test_transaction_count"] == 1 for row in p8)
    p9 = [row for row in rows if row["phase"] == "P9"]
    assert all(row["test_transaction_count"] == 0 for row in p9)
    assert all(row["config_sha256"] == P9_CONFIG_SHA256 for row in p9)


@pytest.mark.local_audit
def test_report_data_contains_all_registered_scientific_results() -> None:
    data = build_report_data()
    assert tuple(data["task"]["models"]) == MODEL_ORDER
    assert len(data["bootstrap"]["paired_mae_A_minus_B"]) == 6
    assert len(data["bootstrap"]["paired_auroc_B_minus_A"]) == 6
    assert data["bootstrap"]["draws"] == 2000
    assert data["gradcam_accounting"]["requested"] == 73724
    assert data["gradcam_accounting"]["valid"] == 66769
    assert data["gradcam_accounting"]["undefined"] == 6955
    assert data["gradcam_accounting"]["root_cause_conclusion"] == (
        "SYSTEMATIC_MODEL/TARGET_ISSUE"
    )
    assert [event["job_id"] for event in data["execution_events"]] == [
        8986164,
        8987452,
        8987554,
    ]
    assert data["execution_events"][1]["scientific_status"] == (
        "INVALIDATED_AGGREGATE_ATTEMPT"
    )
    assert set(data["concept"]) == set(MODEL_ORDER[1:])
    assert all(len(data["concept"][model]["pooled"]) == 8 for model in MODEL_ORDER[1:])
    assert [row["field"] for row in data["data_dictionary"]["en"]] == [
        row["field"] for row in data["data_dictionary"]["zh"]
    ]
    assert len(data["terminology"]) == 8
    assert data["contribution_centering"]["mixed_cem"]["most_negative"] is None
    assert data["contribution_centering"]["mixed_cem"]["smallest_signed"][0] == (
        "texture"
    )


@pytest.mark.local_audit
def test_shared_tables_are_machine_readable_and_complete(tmp_path: Path) -> None:
    data = build_report_data()
    paths = export_tables(data, tmp_path)
    assert {path.name for path in paths} == {
        "primary_secondary_metrics.csv",
        "paired_comparisons.csv",
        "concept_metrics.csv",
        "intervention_curves.csv",
        "centered_contributions.csv",
        "learned_gam_alpha.csv",
        "gradcam_accounting.csv",
        "spatial_faithfulness.csv",
        "execution_registry.csv",
    }
    comparisons = (tmp_path / "paired_comparisons.csv").read_text(encoding="utf-8")
    assert comparisons.count("MAE_A_minus_MAE_B") == 6
    assert comparisons.count("AUROC_B_minus_AUROC_A") == 6
    with (tmp_path / "primary_secondary_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        primary_rows = list(csv.DictReader(handle))
    required_ci_fields = {
        f"{metric}_ci_{bound}"
        for metric in (
            "mae",
            "rmse",
            "normalized_mae",
            "pearson",
            "spearman",
            "auroc",
            "auprc",
        )
        for bound in ("low", "high")
    }
    assert required_ci_fields <= set(primary_rows[0])
    assert {
        "prediction_min_1_to_5",
        "prediction_max_1_to_5",
        "below_one_rate",
        "above_five_rate",
    } <= set(primary_rows[0])
    assert all(row[field] for row in primary_rows for field in required_ci_fields)
    accounting = (tmp_path / "gradcam_accounting.csv").read_text(encoding="utf-8")
    assert "undefined_rate" in accounting
    with (tmp_path / "spatial_faithfulness.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        faithfulness_rows = list(csv.DictReader(handle))
    assert {row["scope"] for row in faithfulness_rows} == {
        "fold_target",
        "pooled_target",
        "pooled_model",
    }
    assert sum(row["scope"] == "pooled_model" for row in faithfulness_rows) == 8
    dictionary_fields = {row["field"] for row in _data_dictionary_rows("en")}
    for path in paths:
        columns = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert set(columns) <= dictionary_fields
    assert [row["field"] for row in _data_dictionary_rows("en")] == [
        row["field"] for row in _data_dictionary_rows("zh")
    ]


def _minimal_report_data() -> dict[str, object]:
    models = {}
    bootstrap = {}
    for index, model in enumerate(MODEL_ORDER):
        models[model] = {
            "pooled": {
                "original_scale_mae": 0.40 + index / 100,
                "original_scale_rmse": 0.50 + index / 100,
            },
            "pooled_secondary": {"auroc": 0.90 + index / 100, "auprc": 0.80 + index / 100},
        }
        bootstrap[model] = {
            "original_scale_mae": {"percentile_2_5": 0.30, "percentile_97_5": 0.60}
        }
    return {
        "task": {"models": models},
        "bootstrap": {"models": bootstrap},
        "scientific_conclusion_codes": [
            "GAM_LOWEST_POINT_ESTIMATE_MAE",
            "PAIRED_MAE_SUPPORTS_GAM_OVER_BLACKBOX_AND_CBM",
            "AUROC_DIFFERENCES_MOSTLY_UNCERTAIN",
            "INTERVENTION_BENEFIT_MODEL_DEPENDENT",
            "SALIENCY_NOT_UNIFORMLY_MORE_FAITHFUL_THAN_RANDOM",
            "SYSTEMATIC_MODEL_TARGET_ZERO_MAP_LIMITATION",
        ],
        "references": ["[1] Reference 2011.", "[2] Reference 2017."],
    }


@pytest.mark.parametrize("variant", ["short", "technical"])
@pytest.mark.local_audit
def test_bilingual_markdown_uses_identical_numeric_layer(variant: str) -> None:
    data = build_report_data()
    en = build_markdown(data, variant, "en")
    zh = build_markdown(data, variant, "zh")
    verify_bilingual_markdown(en, zh, variant)
    assert extract_numeric_tokens(en) == extract_numeric_tokens(zh)


@pytest.mark.local_audit
def test_bilingual_numeric_tamper_is_rejected() -> None:
    data = build_report_data()
    en = build_markdown(data, "short", "en")
    zh = build_markdown(data, "short", "zh").replace("2,633", "2,634", 1)
    with pytest.raises(ValueError, match="BILINGUAL_NUMERIC_TOKEN_MISMATCH"):
        verify_bilingual_markdown(en, zh, "short")


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"nodule_uid": "private"}, "nodule_uid"),
        ({"path": "/srv/scratch/private"}, "/srv/scratch/"),
        ({"approval": "spatial_execution_approval"}, "spatial_execution_approval"),
    ],
)
def test_public_privacy_guard_rejects_private_content(
    payload: dict[str, str], code: str
) -> None:
    with pytest.raises(ValueError, match="P10_PUBLIC_PRIVACY_VIOLATION"):
        assert_public_payload(payload)


def test_report_data_json_is_public_safe(tmp_path: Path) -> None:
    payload = {"models": ["Black-box"], "hash": "a" * 64}
    path = tmp_path / "report_data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert_public_payload(payload)


@pytest.mark.local_audit
def test_references_are_complete_numbered_ieee_primary_sources() -> None:
    references = build_report_data()["references"]
    assert len(references) == 6
    assert [reference[:3] for reference in references] == [
        f"[{index}]" for index in range(1, 7)
    ]
    assert all('"' in reference for reference in references)
    assert sum("doi:" in reference.lower() for reference in references) >= 4


def test_p9_summary_manifest_rejects_scientific_report_tamper(tmp_path: Path) -> None:
    names = (
        "task",
        "concept",
        "contribution_centering",
        "intervention",
        "bootstrap",
        "learned_alpha",
        "spatial",
        "integrity",
    )
    tracked = {}
    for name in names:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"status": "PASS", "name": name}), encoding="utf-8")
        tracked[path.name] = sha256_file(path)
    summary = {"tracked_report_sha256": tracked}
    _verify_p9_tracked_report_hashes(tmp_path, summary)
    (tmp_path / "task.json").write_text(
        json.dumps({"status": "PASS", "name": "task", "mae": 999}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="TRACKED_REPORT_SHA256_MISMATCH:task.json"):
        _verify_p9_tracked_report_hashes(tmp_path, summary)


def test_table_and_figure_sources_have_explicit_shared_data_paths() -> None:
    assert _source_field_path("figures/figure_10_spatial_faithfulness_zh.svg") == (
        "spatial.models.*.pooled_all_targets"
    )
    assert _source_field_path("tables/paired_comparisons.csv") == (
        "bootstrap.paired_mae_A_minus_B;bootstrap.paired_auroc_B_minus_A"
    )
    assert _source_field_path("technical_en.pdf") == "$"


def test_pdf_wrapper_breaks_long_slash_joined_scientific_tokens() -> None:
    token = "subtlety=0.1/" * 30
    lines = _wrap_text(f"weights {token}", 40)
    assert len(lines) > 2
    assert all(len(line) <= 40 for line in lines)
    assert "".join(lines).replace("weights", "", 1).replace(" ", "").strip() == token


@pytest.mark.local_audit
def test_generated_public_reports_close_bilingual_pdf_and_source_gates() -> None:
    result = verify_public_outputs()
    assert result["status"] == "PASS"
    assert result["numeric_language_parity"] is True
    assert result["pdf_numeric_language_parity"] is True
    assert result["chinese_fonts_embedded"] is True
    assert result["public_privacy"] == "PASS"


@pytest.mark.local_audit
def test_generated_chinese_figures_localize_explanatory_labels() -> None:
    figure_root = Path("reports/baseline_v2/p10/public/figures")
    expected = {
        "figure_01_cohort_flow_zh.svg": ("结节", "患者", "第0折"),
        "figure_06_intervention_curves_zh.svg": ("误差优先", "排列均值"),
        "figure_08_learned_alpha_zh.svg": ("第0折",),
        "figure_10_spatial_faithfulness_zh.svg": ("输出敏感度", "预测误差增量"),
    }
    for name, labels in expected.items():
        text = (figure_root / name).read_text(encoding="utf-8")
        assert all(label in text for label in labels)
