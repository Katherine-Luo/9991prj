from __future__ import annotations

import json
from pathlib import Path

import pytest

from lidc_baseline.p10_catalogue import (
    CATEGORY_FILE_NAMES,
    CONTROLLED_AVAILABILITY,
    P10_REPORT_PLAN_SHA256,
    PHASE_STATUS_SNAPSHOT_NAME,
    PUBLIC_MANIFEST_NAME,
    PUBLIC_REGISTRY_NAME,
    REQUIRED_REGISTRY_FIELDS,
    RESULTS_CATALOGUE_PLAN_SHA256,
    _full_ct_display_gate,
    _scan_public_privacy,
    validate_catalogue_config,
    verify_catalogue,
)


PUBLIC_ROOT = Path("docs/results")


def _registry() -> dict[str, object]:
    return json.loads((PUBLIC_ROOT / PUBLIC_REGISTRY_NAME).read_text(encoding="utf-8"))


def test_catalogue_config_binds_both_user_approved_plan_hashes() -> None:
    payload = validate_catalogue_config()
    assert payload["approved_plans"]["results_catalogue"]["sha256"] == RESULTS_CATALOGUE_PLAN_SHA256
    assert payload["approved_plans"]["catalogue_driven_bilingual_report"]["sha256"] == P10_REPORT_PLAN_SHA256
    assert payload["gates"] == {
        "results_catalogue_plan_approved": 1,
        "p10_report_plan_approved": 1,
        "catalogue_implementation_authorized": 1,
        "generated_catalogue_approved": 0,
        "report_revision_authorized": 0,
    }


def test_registry_has_cat_a_through_t_and_required_schema() -> None:
    registry = _registry()
    assert registry["status"] == "CATALOGUE_BUILT_PENDING_USER_APPROVAL"
    assert set(registry["category_counts"]) == {f"CAT-{letter}" for letter in CATEGORY_FILE_NAMES}
    identifiers = []
    for item in registry["items"]:
        assert set(item) == REQUIRED_REGISTRY_FIELDS
        assert item["availability_status"] in CONTROLLED_AVAILABILITY
        identifiers.append(item["catalogue_item_id"])
    assert len(identifiers) == len(set(identifiers)) == 2395
    phase_rows = [item for item in registry["items"] if item["entity_type"] == "phase_overview"]
    assert phase_rows
    assert {item["source_relative_path"] for item in phase_rows} == {
        f"docs/results/{PHASE_STATUS_SNAPSHOT_NAME}"
    }
    assert all(item["source_relative_path"] != "docs/PROJECT_STATUS.md" for item in phase_rows)


def test_scientific_catalogue_cardinalities_and_gradcam_identity() -> None:
    registry = _registry()
    counts = registry["category_counts"]
    assert counts | {
        "CAT-B": 24,
        "CAT-C": 4,
        "CAT-D": 6,
        "CAT-E": 4,
        "CAT-F": 6,
        "CAT-G": 18,
        "CAT-H": 6,
        "CAT-I": 6,
        "CAT-J": 24,
        "CAT-K": 40,
        "CAT-L": 169,
        "CAT-M": 28,
        "CAT-N": 33,
        "CAT-Q": 14,
    } == counts
    global_item = next(item for item in registry["items"] if item["catalogue_item_id"] == "RES-P9-GRADCAM-GLOBAL-ACCOUNTING")
    assert global_item["details"]["requested"] == 73724
    assert global_item["details"]["valid"] == 66769
    assert global_item["details"]["undefined"] == 6955


def test_qualitative_inventory_preserves_ta02_and_fa06_availability() -> None:
    registry = _registry()
    cases = [item for item in registry["items"] if item["category"] == "CAT-Q"]
    assert len(cases) == 14
    assert {item["details"]["case_label"] for item in cases} == {f"CASE-{index:04d}" for index in range(1, 15)}
    for item in cases:
        details = item["details"]
        assert details["full_ct_source_available"] is True
        assert details["series_and_slice_provenance_available"] is True
        assert details["roi_to_full_volume_mapping_available"] is True
        assert details["read_only_full_ct_renderable"] is True
        assert details["new_inference_required"] is False
        assert details["fa06_component_availability"]["how"] == "DATA_NOT_PERSISTED"
        assert "roi_bbox_dhw" in details
        assert "frozen_context_z_index" in details
        assert "gradcam_status_by_target" in details
    ta02 = next(item for item in registry["items"] if item["details"].get("planned_table_id") == "RPT-TA02")
    assert "full reader vote distribution" in ta02["details"]["categorical_target_semantics"]
    fa06 = next(item for item in registry["items"] if item["details"].get("planned_figure_id") == "RPT-FA06")
    assert fa06["availability_status"] == "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS"
    assert fa06["details"]["case_level_intervention_component"] == "DATA_NOT_PERSISTED"
    selected = [item for item in cases if item["details"]["fa06_selected"]]
    assert len(selected) == 1
    assert selected[0]["details"]["case_label"] == fa06["details"]["selected_case_label"]
    assert selected[0]["details"]["fa06_case_role"] == "integrated_explanation"
    assert "RPT-FA06" in selected[0]["report_figure_ids"]
    assert all(
        "RPT-FA06" not in item["report_figure_ids"]
        for item in cases
        if not item["details"]["fa06_selected"]
    )


@pytest.mark.parametrize(
    ("ct", "bbox", "roi_available", "z_index", "expected"),
    [
        ({"available": False}, [[1, 3], [2, 4], [2, 4]], True, 1, False),
        (
            {"available": True, "study_instance_uid": "wrong", "series_instance_uid": "series", "dicom_file_count": 10},
            [[1, 3], [2, 4], [2, 4]], True, 1, False,
        ),
        (
            {"available": True, "study_instance_uid": "study", "series_instance_uid": "series", "dicom_file_count": 10},
            [[1, 3], [2, 4], [2, 4]], True, 8, False,
        ),
        (
            {"available": True, "study_instance_uid": "study", "series_instance_uid": "series", "dicom_file_count": 10},
            [[1, 3], [2, 4], [2, 4]], False, 1, False,
        ),
        (
            {"available": True, "study_instance_uid": "study", "series_instance_uid": "series", "dicom_file_count": 10},
            [[1, 3], [2, 4], [2, 4]], True, 1, True,
        ),
    ],
)
def test_full_ct_renderability_requires_source_slice_and_roi_mapping(
    ct: dict[str, object], bbox: list[list[int]], roi_available: bool, z_index: int, expected: bool
) -> None:
    result = _full_ct_display_gate(
        ct=ct,
        bbox=bbox,
        roi_available=roi_available,
        z_index=z_index,
        expected_study_uid="study",
        expected_series_uid="series",
    )
    assert result["read_only_full_ct_renderable"] is expected


def test_missing_outputs_are_explicit_and_never_silently_recomputed() -> None:
    registry = _registry()
    gaps = [item for item in registry["items"] if item["category"] == "CAT-T"]
    states = {item["availability_status"] for item in gaps}
    assert {
        "DATA_NOT_PERSISTED",
        "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE",
        "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
    } <= states
    case_intervention = next(item for item in gaps if item["catalogue_item_id"].endswith("CASE-INTERVENTION"))
    assert case_intervention["new_inference_required"] is True
    assert "never recompute" in case_intervention["details"]["permitted_action"]


def test_mean_absolute_centered_contribution_remains_non_authoritative() -> None:
    registry = _registry()
    gap = next(
        item
        for item in registry["items"]
        if item["catalogue_item_id"] == "RES-P10-GAP-MEAN-ABS-CONTRIBUTION"
    )
    assert gap["availability_status"] == "DATA_NOT_PERSISTED"
    assert "transient analysis" in gap["details"]["permitted_action"]
    assert "must not be reused in RPT-T15" in gap["details"]["permitted_action"]
    contribution_rows = [item for item in registry["items"] if item["category"] == "CAT-J"]
    assert len(contribution_rows) == 24
    assert all(
        item["details"]["mean_absolute_contribution_status"] == "DATA_NOT_PERSISTED"
        and "non-authoritative for P10" in item["details"]["mean_absolute_contribution_note"]
        for item in contribution_rows
    )
    table = next(
        item
        for item in registry["items"]
        if item["details"].get("planned_table_id") == "RPT-T15"
    )
    assert table["details"]["mean_absolute_centered_contribution_policy"].startswith(
        "UNAVAILABLE_FROM_AUTHORITATIVE_FROZEN_SOURCE"
    )


def test_cat_t_reconciles_exact_planned_output_counts() -> None:
    registry = _registry()
    gaps = {
        item["catalogue_item_id"]: item
        for item in registry["items"]
        if item["category"] == "CAT-T"
    }
    assert gaps["RES-P10-GAP-PLANNED-TABLES"]["result_name"] == (
        "20 planned tables: RPT-T01-RPT-T18 + RPT-TA01 + RPT-TA02"
    )
    assert gaps["RES-P10-GAP-PLANNED-FIGURES"]["result_name"] == (
        "14 planned public figures: RPT-F01-RPT-F08 + RPT-F09A + RPT-F09B + RPT-F10-RPT-F13"
    )
    assert gaps["RES-P10-GAP-PRIVATE-FIGURES"]["result_name"] == (
        "6 planned private figures: RPT-FA01-RPT-FA06"
    )
    planned_tables = [
        item for item in registry["items"] if item["entity_type"] == "planned_scientific_table"
    ]
    planned_figures = [
        item for item in registry["items"] if item["entity_type"] == "planned_scientific_figure"
    ]
    assert len(planned_tables) == 20
    assert len([item for item in planned_figures if not item["details"]["planned_figure_id"].startswith("RPT-FA")]) == 14
    assert len([item for item in planned_figures if item["details"]["planned_figure_id"].startswith("RPT-FA")]) == 6


def test_old_figures_and_pdfs_are_legacy_presentation_only() -> None:
    registry = _registry()
    existing_figures = [
        item for item in registry["items"] if item["entity_type"] == "existing_public_figure"
    ]
    assert existing_figures
    for item in existing_figures:
        assert item["scientific_status"] == "LEGACY_PRESENTATION_ONLY"
        assert item["report_usage_status"] == "AUDIT_ONLY"
        assert item["report_table_ids"] == []
        assert item["report_figure_ids"] == []
        assert item["details"]["may_satisfy_planned_report_requirement"] is False
        assert item["details"]["authoritative_rewrite_input"] is False
    legacy_files = [
        item for item in registry["items"] if item["entity_type"] == "legacy_presentation_file"
    ]
    assert legacy_files
    assert any(item["source_relative_path"].endswith(".pdf") for item in legacy_files)
    assert any("/figures/" in item["source_relative_path"] for item in legacy_files)
    assert all(
        item["scientific_status"] == "LEGACY_PRESENTATION_ONLY"
        and item["report_usage_status"] == "AUDIT_ONLY"
        and item["details"]["may_satisfy_planned_report_requirement"] is False
        and item["details"]["authoritative_rewrite_input"] is False
        for item in legacy_files
    )
    private_legacy_files = [
        item
        for item in registry["items"]
        if item["entity_type"] == "legacy_private_presentation_file"
    ]
    assert private_legacy_files
    assert any(item["source_relative_path"].endswith(".pdf") for item in private_legacy_files)
    assert any(item["source_relative_path"].endswith(".png") for item in private_legacy_files)
    assert all(
        item["source_root_alias"] == "private-report://"
        and item["scientific_status"] == "LEGACY_PRESENTATION_ONLY"
        and item["report_usage_status"] == "AUDIT_ONLY"
        and item["report_table_ids"] == []
        and item["report_figure_ids"] == []
        and item["details"]["may_satisfy_planned_report_requirement"] is False
        and item["details"]["authoritative_rewrite_input"] is False
        for item in private_legacy_files
    )


def test_public_catalogue_rejects_private_tokens(tmp_path: Path) -> None:
    safe = tmp_path / "safe.csv"
    safe.write_text("case_label,root_alias\nCASE-0001,mac-archive://\n", encoding="utf-8")
    _scan_public_privacy(safe)
    unsafe = tmp_path / "unsafe.csv"
    unsafe.write_text("patient_key,/Users/example\n", encoding="utf-8")
    with pytest.raises(ValueError, match="P10_CATALOGUE_PUBLIC_PRIVACY_VIOLATION"):
        _scan_public_privacy(unsafe)


def test_manifest_records_pending_gate_and_no_report_or_scientific_rebuild() -> None:
    manifest = json.loads((PUBLIC_ROOT / PUBLIC_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "CATALOGUE_VERIFIED_PENDING_USER_APPROVAL"
    assert manifest["generated_catalogue_approved"] == 0
    assert manifest["report_revision_authorized"] == 0
    assert manifest["scientific_compute_performed"] is False
    assert manifest["reports_regenerated"] is False
    assert manifest["p11_started"] is False


@pytest.mark.local_audit
def test_full_public_private_catalogue_verifier_passes() -> None:
    result = verify_catalogue()
    assert result["status"] == "PASS"
    assert result["private"]["status"] == "PASS"
    assert result["gradcam_accounting"] == {"requested": 73724, "valid": 66769, "undefined": 6955}
    assert result["reports_regenerated"] is False
    assert result["scientific_compute_performed"] is False
    assert result["p11_started"] is False
