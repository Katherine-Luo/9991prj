from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from lidc_baseline.p10_catalogue_private import (
    _case_science,
    _load_cases,
    _mapped_map_slice,
    _parse_scalar,
)
from lidc_baseline.p10_catalogue import _approved_revision_hashes
from lidc_baseline.p10_catalogue_report import (
    PUBLIC_FIGURE_IDS,
    PUBLIC_TABLE_IDS,
    build_markdown_manuscript,
    build_manuscript_sections,
    build_public_table_rows,
    load_catalogue_context,
    MANUAL_VISUAL_REVIEWER,
    verify_bilingual_numeric_parity,
    _verify_manual_review_provenance,
    _contribution_profiles,
    _pdf_table_layout,
    export_public_tables,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pdf_table_layout_preserves_all_intervention_and_alpha_columns() -> None:
    intervention = [{
        "Model": "Mixed-type CEM",
        "Ordering": "error_first",
        "Baseline MAE": "0.1",
        "k=4 MAE": "0.1",
        "k=8 MAE": "0.1",
        "iMAE": "0.1",
        "Delta_iMAE": "0.1",
        "Baseline AUROC": "0.9",
        "iAUC": "0.9",
        "Delta_iAUC": "0.0",
    }]
    cols, widths = _pdf_table_layout("RPT-T17", intervention, 480.0)
    assert cols[-2:] == ["iAUC", "Delta_iAUC"]
    assert len(cols) == len(widths) == 10
    assert sum(widths) == pytest.approx(480.0)

    alpha = [{
        "Fold": 0,
        "Concept": "texture",
        "Expert 1": "0.2",
        "Expert 2": "0.2",
        "Expert 3": "0.2",
        "Expert 4": "0.2",
        "Expert 5": "0.2",
        "Min–max": "0.2–0.2",
        "Simplex": "1.0",
    }]
    alpha_cols, alpha_widths = _pdf_table_layout("RPT-T16", alpha, 480.0)
    assert alpha_cols[-1] == "Simplex"
    assert len(alpha_cols) == len(alpha_widths) == 9
    assert sum(alpha_widths) == pytest.approx(480.0)


def test_revision_config_binds_approved_catalogue_and_gates() -> None:
    context = load_catalogue_context(REPOSITORY_ROOT)
    assert context.registry_sha256 == "624fa259430d6c5709568f7507ec0a92421d669ad64071d935eece566283b3cf"
    assert len(context.items) == 2395
    assert context.config["approval_gates"] == {
        "results_catalogue_plan_approved": 1,
        "p10_report_plan_approved": 1,
        "generated_catalogue_approved": 1,
        "report_revision_authorized": 1,
    }
    boundaries = context.config["scientific_boundaries"]
    assert boundaries["p5_through_p9_access"] == "read_only"
    for key in ("model_forward", "training", "scientific_recomputation", "bootstrap_recomputation", "intervention_recomputation", "gradcam_recomputation", "p11"):
        assert boundaries[key] == "forbidden"


def test_revision_hash_binding_is_exact_and_rejects_unlisted_file(tmp_path: Path) -> None:
    public, private = _approved_revision_hashes(REPOSITORY_ROOT, Path("/Users/katherine/Desktop/lidc_data/lidc_baseline_private_archive/baseline_v2"))
    technical = (REPOSITORY_ROOT / "reports/baseline_v2/p10/public/technical_en.md").resolve()
    assert public[technical] == hashlib.sha256(technical.read_bytes()).hexdigest()
    appendix = Path("/Users/katherine/Desktop/lidc_data/lidc_baseline_private_archive/baseline_v2/p10_private_report/qualitative_appendix_en.md").resolve()
    assert private[appendix] == hashlib.sha256(appendix.read_bytes()).hexdigest()
    assert (REPOSITORY_ROOT / "reports/baseline_v2/p10/public/unlisted.txt").resolve() not in public


def test_revision_hash_binding_rejects_invalid_private_manual_qa_provenance(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    private_root = tmp_path / "private"
    public_root = repository / "reports/baseline_v2/p10/public"
    private_report_root = private_root / "p10_private_report"
    config_root = repository / "configs/experiments"
    public_root.mkdir(parents=True)
    private_report_root.mkdir(parents=True)
    config_root.mkdir(parents=True)
    (config_root / "baseline_v2_p10_report_revision.resolved.yaml").write_text(
        "approval_gates:\n  report_revision_authorized: 1\n", encoding="utf-8"
    )
    registry_sha = "624fa259430d6c5709568f7507ec0a92421d669ad64071d935eece566283b3cf"
    public_manifest = {
        "catalogue_registry_sha256": registry_sha,
        "reports": {
            "technical_en": {"markdown_sha256": "1", "pdf_sha256": "2"},
            "technical_zh": {"markdown_sha256": "3", "pdf_sha256": "4"},
        },
        "tables": {},
        "figures": {},
        "reverse_traceability_sha256": "5",
    }
    report_names = (
        "qualitative_appendix_en",
        "qualitative_appendix_zh",
        "technical_en_with_appendix",
        "technical_zh_with_appendix",
    )
    private_manifest = {
        "catalogue_registry_sha256": registry_sha,
        "reports": {name: {"sha256": str(index)} for index, name in enumerate(report_names)},
        "tables": {},
        "figures": {},
    }
    (public_root / "catalogue_report_manifest.json").write_text(
        json.dumps(public_manifest), encoding="utf-8"
    )
    (private_report_root / "catalogue_private_report_manifest.json").write_text(
        json.dumps(private_manifest), encoding="utf-8"
    )
    rendered = {
        name: {"rendered_pages": [{"page": 1, "png_sha256": str(index)}]}
        for index, name in enumerate(report_names)
    }
    private_qa = {
        "status": "PASS",
        "manual_visual_review": "PASS",
        "manual_reviewer": "Codex primary agent (visual inspection of contact sheets and original-resolution critical pages)",
        "manual_review_timestamp_utc": "2026-08-13T01:02:03+00:00",
        "rendered_page_manifest_sha256": hashlib.sha256(
            json.dumps(
                {name: row["rendered_pages"] for name, row in sorted(rendered.items())},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "pdfs": {
            name: {**rendered[name], "pdf_sha256": private_manifest["reports"][name]["sha256"]}
            for name in report_names
        },
    }
    qa_path = private_report_root / "private_visual_qa.json"
    qa_path.write_text(json.dumps(private_qa), encoding="utf-8")
    _approved_revision_hashes(repository, private_root)
    for key in (
        "manual_reviewer",
        "manual_review_timestamp_utc",
        "rendered_page_manifest_sha256",
    ):
        tampered = dict(private_qa)
        tampered[key] = "incorrect"
        qa_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="P10_PRIVATE_VISUAL_QA_"):
            _approved_revision_hashes(repository, private_root)


def test_public_table_and_figure_inventory_is_exact() -> None:
    context = load_catalogue_context(REPOSITORY_ROOT)
    tables = build_public_table_rows(context)
    assert set(tables) == set(PUBLIC_TABLE_IDS)
    assert len(PUBLIC_TABLE_IDS) == 18
    assert len(PUBLIC_FIGURE_IDS) == 14
    assert "RPT-F09A" in PUBLIC_FIGURE_IDS
    assert "RPT-F09B" in PUBLIC_FIGURE_IDS
    assert "RPT-F09" not in PUBLIC_FIGURE_IDS
    assert len(tables["RPT-T15"]) == 6
    assert {row["Role within model"] for row in tables["RPT-T15"]} == {
        "Largest pooled signed mean",
        "Smallest pooled signed mean",
    }
    for model in {row["Model"] for row in tables["RPT-T15"]}:
        model_rows = [row for row in tables["RPT-T15"] if row["Model"] == model]
        by_role = {row["Role within model"]: float(row["Pooled signed mean (rating points)"]) for row in model_rows}
        assert by_role["Largest pooled signed mean"] >= by_role["Smallest pooled signed mean"]
    assert all(row["Target"] == "malignancy" for row in tables["RPT-T14"])


def test_controlled_comparison_codes_are_not_exported_in_reader_tables(tmp_path: Path) -> None:
    export_public_tables(load_catalogue_context(REPOSITORY_ROOT), tmp_path)
    for table_id in ("RPT-T08", "RPT-T10"):
        text = (tmp_path / "tables_catalogue" / f"{table_id}.csv").read_text(encoding="utf-8")
        assert "controlled_conclusion_code" not in text
        assert "SUPPORTS_A" not in text
        assert "SUPPORTS_B" not in text
        assert "NO_SUPPORTED_DIFFERENCE_CI_CROSSES_ZERO" not in text


def test_correction_round_scientific_semantics_are_fail_closed() -> None:
    tables = build_public_table_rows(load_catalogue_context(REPOSITORY_ROOT))
    assert all(row["Cohort component"] != "Reference physical nodules" for row in tables["RPT-T02"])
    for table_id in ("RPT-T08", "RPT-T10"):
        for row in tables[table_id]:
            assert row["Sign convention"] == "Positive Δ favors B"
            if row["Crosses zero"]:
                assert row["Supported conclusion"] == "No supported difference"
                assert row["controlled_conclusion_code"] == "NO_SUPPORTED_DIFFERENCE_CI_CROSSES_ZERO"
            elif float(row[next(key for key in row if key.startswith("Delta-"))]) > 0:
                assert row["Supported conclusion"] == "Supports B"
                assert row["controlled_conclusion_code"] == "SUPPORTS_B"
            else:
                assert row["Supported conclusion"] == "Supports A"
                assert row["controlled_conclusion_code"] == "SUPPORTS_A"
    how = next(row for row in tables["RPT-T18"] if row["Layer"] == "HOW")
    assert "strong and consistent for CEM" in how["Main evidence"]
    assert "unfavorable overall for GAM" in how["Main evidence"]
    assert len(tables["RPT-T05"]) == 12
    assert all("Best epoch" not in row for row in tables["RPT-T05"])
    assert next(row for row in tables["RPT-T05"] if row["Setting"] == "Standard CBM objective")["source_artifact_id"] != next(row for row in tables["RPT-T05"] if row["Setting"] == "Mixed-type CEM objective")["source_artifact_id"]


def test_bilingual_manuscript_has_same_numbers_and_interleaved_story() -> None:
    context = load_catalogue_context(REPOSITORY_ROOT)
    tables = build_public_table_rows(context)
    english = build_markdown_manuscript(context, tables, "en")
    chinese = build_markdown_manuscript(context, tables, "zh")
    verify_bilingual_numeric_parity(english, chinese)
    assert "No supported difference means that the paired 95% CI crosses zero" in english
    assert "No supported difference 表示配对 95% CI 跨越零" in chinese
    assert "Table RPT-T15 summarizes selected persisted pooled signed means" in english
    assert "表 RPT-T15 汇总经选择且已持久化的 pooled signed mean" in chinese
    assert "SUPPORTS_A" not in english and "SUPPORTS_B" not in english
    assert "NO_SUPPORTED_DIFFERENCE_CI_CROSSES_ZERO" not in english
    assert "SUPPORTS_A" not in chinese and "SUPPORTS_B" not in chinese
    assert "NO_SUPPORTED_DIFFERENCE_CI_CROSSES_ZERO" not in chinese
    for label in ("Prediction", "WHERE", "WHAT", "WHY", "HOW"):
        assert english.index(label) >= 0
        assert chinese.index(label) >= 0
    assert english.index("### 6.1 Results — Prediction") < english.index("### 6.2 Results — WHERE")
    assert english.index("### 6.2 Results — WHERE") < english.index("### 6.3 Results — WHAT")
    assert english.index("### 6.3 Results — WHAT") < english.index("### 6.4 Results — WHY")
    assert english.index("### 6.4 Results — WHY") < english.index("### 6.5 Results — HOW")
    assert english.index("RPT-F04") < english.index("## 7. Discussion")
    for identifier in (*PUBLIC_TABLE_IDS, *PUBLIC_FIGURE_IDS):
        assert english.count(identifier) >= 2
        assert chinese.count(identifier) >= 2
    assert "Scientific conclusion codes" not in english
    assert "Full machine-readable rows" not in english
    assert "Author / 作者" not in english
    assert "作者 /" not in chinese
    assert english.count("RPT-T06. Evaluation protocol") == 1


def test_results_sections_have_substantive_explanation() -> None:
    sections = build_manuscript_sections(load_catalogue_context(REPOSITORY_ROOT))
    result_sections = [section for section in sections if section.section_id.startswith("SEC-RESULTS-")]
    assert [section.section_id for section in result_sections] == [
        "SEC-RESULTS-PREDICTION",
        "SEC-RESULTS-WHERE",
        "SEC-RESULTS-WHAT",
        "SEC-RESULTS-WHY",
        "SEC-RESULTS-HOW",
        "SEC-RESULTS-SYNTHESIS",
    ]
    for section in result_sections:
        assert len(section.paragraphs_en) >= 3
        assert len(section.paragraphs_en) == len(section.paragraphs_zh)
        assert sum(len(value.split()) for value in section.paragraphs_en) >= 120
    all_en="\n".join(paragraph for section in sections for paragraph in section.paragraphs_en)
    assert "Mixed-type CEM uniquely showed strong, consistent integrated MAE benefit" in all_en
    assert "integrated Delta_iMAE was negative overall" in all_en
    assert "Intervention improvements in CEM and GAM" not in all_en
    assert "Dumaev et al." in all_en


def test_private_case_selection_and_fa06_are_catalogue_locked() -> None:
    context = load_catalogue_context(REPOSITORY_ROOT)
    cases = _load_cases(context)
    assert len(cases) == 14
    assert [case["case_label"] for case in cases] == [f"CASE-{index:04d}" for index in range(1, 15)]
    fa06 = [case for case in cases if "RPT-FA06" in case["report_figure_ids"]]
    assert len(fa06) == 1
    assert fa06[0]["case_label"] == "CASE-0004"
    assert fa06[0]["catalogue_details"]["fa06_component_availability"] == {
        "how": "DATA_NOT_PERSISTED",
        "prediction": "RESULT_ALREADY_EXISTS",
        "what": "RESULT_ALREADY_EXISTS",
        "where": "RESULT_ALREADY_EXISTS",
        "why": "RESULT_ALREADY_EXISTS",
    }


def test_case_science_preserves_vote_distribution_and_no_hard_gt_substitution() -> None:
    oof = pd.DataFrame(
        [
            {
                "nodule_uid": "x",
                "malignancy_score_1_to_5": 3.25,
                **{f"{concept}_activated_prediction": json.dumps([0.25]) for concept in ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")},
                "internalStructure_activated_prediction": json.dumps([0.1, 0.7, 0.1, 0.1]),
                "calcification_activated_prediction": json.dumps([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]),
                **{f"{concept}_rating_point_contribution": 0.1 for concept in ("subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation", "spiculation", "texture")},
            }
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "nodule_uid": "x",
                "mean_malignancy": 3.0,
                **{f"{concept}_target": 0.5 for concept in ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")},
                "internalStructure_vote_distribution": np.array([0.0, 0.5, 0.5, 0.0]),
                "internalStructure_modal_class": np.nan,
                "calcification_vote_distribution": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "calcification_modal_class": 6.0,
            }
        ]
    )
    evidence = _case_science({"case_label": "CASE-TEST", "nodule_uid": "x", "model": "standard_cbm"}, oof, manifest)
    assert evidence["concepts"]["internalStructure"]["target_modal_label"] == "Reader tie"
    assert evidence["concepts"]["internalStructure"]["target_vote_distribution"] == [0.0, 0.5, 0.5, 0.0]
    assert evidence["concepts"]["calcification"]["target_modal_label"] == "Absent"


def test_mapped_map_slice_is_deterministic_and_fail_closed_scalar_parser() -> None:
    heatmap = np.arange(64 * 64 * 64, dtype=np.float32).reshape(64, 64, 64)
    selected = _mapped_map_slice(heatmap, ((10, 20), (1, 3), (2, 4)), 15)
    expected_index = round((15 - 10) / (20 - 10 - 1) * 63)
    assert np.array_equal(selected, heatmap[expected_index])
    assert _parse_scalar("[0.25]") == 0.25
    with pytest.raises(ValueError, match="P10_PRIVATE_SCALAR_INVALID"):
        _parse_scalar("[0.25, 0.75]")


def test_categorical_contribution_profiles_do_not_connect_category_order() -> None:
    figure = _contribution_profiles("en")
    try:
        for axis in (figure.axes[1], figure.axes[2]):
            assert not axis.lines or all(len(line.get_xdata()) <= 2 for line in axis.lines)
            assert len(axis.patches) > 0
    finally:
        plt.close(figure)


def test_manual_visual_review_provenance_is_fail_closed() -> None:
    pages = {"technical_en": [{"page": 1, "sha256": "a" * 64}]}
    rendered_sha = hashlib.sha256(
        json.dumps(pages, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    valid = {
        "manual_reviewer": MANUAL_VISUAL_REVIEWER,
        "manual_review_timestamp_utc": "2026-08-13T01:02:03+00:00",
        "rendered_page_manifest_sha256": rendered_sha,
        "pdfs": {"technical_en": {"rendered_pages": pages["technical_en"]}},
    }
    _verify_manual_review_provenance(valid)
    for key in (
        "manual_reviewer",
        "manual_review_timestamp_utc",
        "rendered_page_manifest_sha256",
    ):
        tampered = dict(valid)
        tampered[key] = "incorrect"
        with pytest.raises(ValueError, match="P10_VISUAL_QA_"):
            _verify_manual_review_provenance(tampered)
