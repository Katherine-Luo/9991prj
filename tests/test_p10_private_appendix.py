from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.p10_private_appendix import (
    _appendix_markdown,
    _load_roi,
    _verify_selected_case_index,
    _verify_private_visual_qa,
    record_private_visual_qa,
    re_case_label,
    select_concept_cases,
    select_task_cases,
)


def _oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "nodule_uid": [f"uid-{index}" for index in range(8)],
            "patient_key": [f"patient-{index}" for index in range(8)],
            "fold_index": [0, 1, 2, 3, 4, 0, 1, 2],
            "malignancy_raw_score": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.7],
            "target_normalized": [0.0, 0.1, 0.5, 0.4, 0.7, 0.5, 0.9, 0.7],
        }
    )


def test_task_case_selection_is_stable_median_and_maximum() -> None:
    first = select_task_cases(_oof(), "blackbox")
    second = select_task_cases(_oof().sample(frac=1, random_state=7), "blackbox")
    assert first == second
    assert {row["role"] for row in first} == {
        "median_error_representative",
        "maximum_error_failure",
    }
    assert len({row["nodule_uid"] for row in first}) == 2


def test_each_model_keeps_its_own_registered_task_case_criteria() -> None:
    blackbox = select_task_cases(_oof(), "blackbox")
    standard_cbm = select_task_cases(_oof(), "standard_cbm")
    assert [row["nodule_uid"] for row in blackbox] == [
        row["nodule_uid"] for row in standard_cbm
    ]


def _map_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(8):
        status = "undefined" if index in {2, 3, 4} else "valid"
        target = "calcification" if index < 5 else "subtlety"
        saliency_error_increase = None
        if status == "valid":
            saliency_error_increase = 0.01 * (index + 1)
        rows.append(
            {
                "nodule_uid": f"uid-{index}",
                "model": "standard_cbm",
                "fold_index": index % 5,
                "target": target,
                "status": status,
                "saliency_error_increase": saliency_error_increase,
                "shard_relative_path": f"p9/spatial/standard_cbm/fold_{index % 5}/shard.parquet",
            }
        )
    return rows


def test_concept_case_selection_uses_positive_worsening_and_highest_zero_rate() -> None:
    cases = select_concept_cases(_map_rows(), "standard_cbm", _oof())
    positive, zero = cases
    assert positive["role"] == "maximum_positive_error_worsening"
    assert positive["saliency_error_increase"] == 0.08
    assert zero["role"] == "highest_undefined_rate_concept_zero_map"
    assert zero["target"] == "calcification"
    assert zero["highest_undefined_rate"] == 3 / 5


def test_case_labels_are_opaque() -> None:
    assert re_case_label("CASE-0001")
    assert not re_case_label("uid-0001")
    assert not re_case_label("CASE-001")


def test_private_case_index_rejects_selection_or_no_forward_tamper() -> None:
    expected = [
        {
            "case_label": f"CASE-{index:04d}",
            "model": "blackbox",
            "role": "median_error_representative",
            "nodule_uid": f"uid-{index}",
        }
        for index in range(1, 15)
    ]
    observed = [dict(case) for case in expected]
    _verify_selected_case_index({"model_forward": False}, observed, expected)
    observed[0]["nodule_uid"] = "replacement"
    with pytest.raises(ValueError, match="CASE_SELECTION_MISMATCH"):
        _verify_selected_case_index({"model_forward": False}, observed, expected)
    with pytest.raises(ValueError, match="MODEL_FORWARD_BOUNDARY_INVALID"):
        _verify_selected_case_index({"model_forward": True}, expected, expected)


def test_chinese_appendix_localizes_explanatory_case_labels() -> None:
    case = {
        "case_label": "CASE-0001",
        "model": "standard_cbm",
        "role": "maximum_error_failure",
        "fold_index": 2,
        "target": "malignancy",
        "map_status": "undefined",
        "slice_index": 31,
        "score_original": 0.812345,
        "target_normalized": 0.250000,
        "absolute_error": 0.562345,
    }
    chinese = _appendix_markdown([case], "zh")
    english = _appendix_markdown([case], "en")
    assert "案例类型：最大绝对误差失败案例" in chinese
    assert "图状态：未定义（post-ReLU 全零）" in chinese
    assert "原始分数：0.812345" in chinese
    assert "Original score: 0.812345" in english
    assert "- Model:" not in chinese
    assert "- Role: maximum absolute-error failure" in english
    assert "Map status: undefined (post-ReLU exact zero)" in english


def test_zero_map_appendix_preserves_highest_undefined_rate_in_both_languages() -> None:
    case = {
        "case_label": "CASE-0014",
        "model": "mixed_cem",
        "role": "highest_undefined_rate_concept_zero_map",
        "fold_index": 4,
        "target": "calcification",
        "map_status": "undefined",
        "slice_index": 29,
        "highest_undefined_rate": 0.2252,
    }
    chinese = _appendix_markdown([case], "zh")
    english = _appendix_markdown([case], "en")
    assert "最高概念未定义率: 0.225200" in chinese
    assert "Highest concept undefined rate: 0.225200" in english


def test_private_roi_loader_uses_image_channel_and_ignores_mask_metadata(
    tmp_path,
) -> None:
    uid = "opaque-uid"
    image = np.linspace(0, 1, 64**3, dtype=np.float32).reshape(1, 64, 64, 64)
    mask = np.zeros_like(image, dtype=np.uint8)
    np.savez_compressed(
        tmp_path / f"{uid}.npz",
        image=image,
        mask=mask,
        **{"metadata.json": b'{}'},
    )
    loaded = _load_roi(uid, tmp_path)
    assert loaded.shape == (64, 64, 64)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, image[0])


def test_private_visual_qa_binds_all_four_pdfs(tmp_path, monkeypatch) -> None:
    import lidc_baseline.p10_private_appendix as appendix

    root = tmp_path / "p10_private_report"
    root.mkdir()
    names = (
        "qualitative_appendix_en",
        "qualitative_appendix_zh",
        "technical_en_with_appendix",
        "technical_zh_with_appendix",
    )
    for name in names:
        (root / f"{name}.pdf").write_bytes(name.encode("utf-8"))
    monkeypatch.setattr(appendix, "_resolve_pdftoppm", lambda: "pdftoppm")
    monkeypatch.setattr(appendix, "_pdf_page_count", lambda path: 1)
    monkeypatch.setattr(
        appendix,
        "_rendered_pdf_page_rows",
        lambda path, renderer: [
            {
                "page": 1,
                "width": 100,
                "height": 100,
                "png_sha256": "a" * 64,
                "nonblank": True,
            }
        ],
    )
    monkeypatch.setattr(
        appendix, "_pdfplumber_page_text_gate", lambda path, pages: None
    )
    evidence = record_private_visual_qa(tmp_path, manual_review_pass=True)
    assert evidence["rendered_page_count"] == 4
    assert _verify_private_visual_qa(tmp_path)["status"] == "PASS"
    (root / "qualitative_appendix_en.pdf").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="VISUAL_QA_BINDING_INVALID"):
        _verify_private_visual_qa(tmp_path)
