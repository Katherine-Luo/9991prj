from __future__ import annotations

from pathlib import Path

from lidc_baseline.config import canonical_yaml, compute_config_sha256, load_config


SOURCE = Path("configs/experiments/baseline_v2_p10_report_archive.yaml")
RESOLVED = Path("configs/experiments/baseline_v2_p10_report_archive.resolved.yaml")
DIGEST = Path("configs/experiments/baseline_v2_p10_report_archive.sha256")


def _project() -> dict[str, object]:
    return load_config(SOURCE)["project_preregistered"]


def test_p10_execution_supplement_is_canonical_and_frozen() -> None:
    source = load_config(SOURCE)
    assert RESOLVED.read_bytes() == canonical_yaml(source)
    assert DIGEST.read_text(encoding="ascii") == f"{compute_config_sha256(source)}\n"


def test_p10_inputs_are_strictly_read_only_and_no_new_compute_is_allowed() -> None:
    frozen = _project()["frozen_inputs"]
    assert frozen["phases"] == ["P5", "P6", "P7", "P8", "P9"]
    assert frozen["access"] == "read_only"
    assert frozen["retraining"] == "forbidden"
    assert frozen["test_inference"] == "forbidden"
    assert frozen["second_committed_test_evaluation"] == "forbidden"
    assert frozen["artifact_rewrite"] == "forbidden"
    assert frozen["new_h200_jobs"] == "forbidden"
    assert frozen["new_cpu_scientific_jobs"] == "forbidden"
    assert frozen["p11"] == "forbidden"


def test_p10_archive_is_exact_read_only_and_mac_private() -> None:
    archive = _project()["private_archive"]
    assert archive["remote_whitelist"] == [
        "blackbox",
        "standard_cbm",
        "cem",
        "gam",
        "p9",
    ]
    assert archive["local_root"] == (
        "/Users/katherine/Desktop/lidc_data/"
        "lidc_baseline_private_archive/baseline_v2"
    )
    assert archive["local_free_space_ratio_minimum"] == 1.2
    assert archive["partial_transfer_resume"] is True
    assert archive["remote_delete"] == "forbidden"
    assert archive["local_delete"] == "forbidden"
    assert archive["remote_write"] == "forbidden"
    assert archive["completion_marker"] == "ARCHIVE_COMPLETE.json"
    assert archive["tracked_private_file_list"] == "forbidden"


def test_p10_public_report_variants_and_page_gates_are_frozen() -> None:
    public = _project()["public_outputs"]
    assert public["formats"] == ["markdown", "pdf"]
    assert public["variants"] == [
        {"id": "short_en", "language": "en", "length_pages": [8, 12]},
        {"id": "short_zh", "language": "zh", "length_pages": [8, 12]},
        {
            "id": "technical_en",
            "language": "en",
            "length_pages": [25, 35],
        },
        {
            "id": "technical_zh",
            "language": "zh",
            "length_pages": [25, 35],
        },
    ]
    assert public["references"]["style"] == "IEEE_numeric"
    assert public["references"]["numbering_identical_across_languages"] is True


def test_p10_bilingual_numeric_and_chinese_font_contract_is_explicit() -> None:
    bilingual = _project()["bilingual_contract"]
    assert bilingual["authoritative_numeric_source"] == (
        "single_structured_report_data_model"
    )
    assert bilingual["recompute_during_translation"] == "forbidden"
    assert bilingual["exact_numeric_token_parity"] is True
    assert bilingual["exact_table_cell_parity"] is True
    assert bilingual["ci_zero_crossing_parity"] is True
    assert bilingual["model_labels_preserved"] == [
        "Black-box",
        "Standard CBM",
        "Mixed-type CEM",
        "Learned-softmax GAM",
    ]
    assert bilingual["chinese_font"] == {
        "path": "/System/Library/Fonts/Supplemental/Songti.ttc",
        "regular_subfont_index": 6,
        "bold_subfont_index": 1,
        "embed": True,
        "sha256_recorded_in_audit": True,
    }
    assert bilingual["missing_glyphs"] == "forbidden"


def test_p10_scientific_content_and_limitations_are_complete() -> None:
    content = _project()["scientific_content"]
    assert content["patient_cluster_bootstrap_draws"] == 2000
    assert content["paired_mae_model_pairs"] == 6
    assert content["paired_auroc_model_pairs"] == 6
    assert content["intervention_k"] == list(range(9))
    assert content["gradcam_accounting"] == {
        "requested": 73724,
        "valid": 66769,
        "undefined": 6955,
    }
    assert content["matched_random_values_per_valid_map"] == 20
    limitations = set(content["mandatory_limitations"])
    assert "primary_scores_are_unclipped" in limitations
    assert "radiologist_assessment_not_pathology_confirmed_diagnosis" in limitations
    assert "not_a_clinical_diagnostic_system" in limitations
    assert "undefined_RCA_conclusion_SYSTEMATIC_MODEL_TARGET_ISSUE" in limitations


def test_p10_private_appendix_is_fixed_to_fourteen_existing_cases() -> None:
    appendix = _project()["private_qualitative_appendix"]
    assert appendix["languages"] == ["en", "zh"]
    assert appendix["tracked_in_git"] is False
    assert appendix["total_cases"] == 14
    assert appendix["task_cases_per_model"] == {
        "median_absolute_error_representative": 1,
        "maximum_absolute_error_failure": 1,
    }
    assert appendix["concept_cases_per_concept_model"] == {
        "maximum_positive_error_worsening": 1,
        "highest_undefined_rate_target_zero_map": 1,
    }
    assert appendix["model_forward"] == "forbidden"
    assert appendix["case_label_pattern"] == "CASE-####"
    assert appendix["same_cases_slices_maps_layout_and_numbers_across_languages"]


def test_p10_public_privacy_and_quality_gates_fail_closed() -> None:
    project = _project()
    privacy = project["privacy"]
    assert privacy["public_raw_medical_images"] == "forbidden"
    assert privacy["public_private_archive"] == "forbidden"
    assert privacy["github_lfs"] == "forbidden"
    assert privacy["exact_git_whitelist_required"] is True
    gates = project["quality_gates"]
    assert gates["p5_through_p9_immutable_before_after"] is True
    assert gates["every_pdf_page_rendered_and_visually_inspected"] is True
    assert gates["clipped_overlapped_or_missing_glyph_content"] == "forbidden"
    assert gates["final_phase_state_before_user_confirmation"] == (
        "AWAITING_USER_APPROVAL"
    )
