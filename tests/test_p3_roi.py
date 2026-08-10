"""Tests for deterministic Phase 3 consensus ROI primitives."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lidc_baseline.p1_audit import _key
from lidc_baseline.p3_roi import (
    DicomSlice,
    HU_MIN,
    ROI_SHAPE,
    _qa_image,
    assert_deidentified_audit,
    apply_duplicate_policy,
    convert_pixels_to_hu,
    consensus_physical_volume_mm3,
    deterministic_npz_bytes,
    enable_pylidc_numpy_compatibility,
    map_pylidc_mask_to_dhw,
    pad_to_cube,
    padding_ratio_for_dimensions,
    pilot_statistics_source_fingerprint,
    require_pilot_confirmation,
    reuse_or_schedule_rebuild,
    reusable_roi,
    reusable_pilot_statistics,
    roi_source_fingerprint,
    scan_geometry_fingerprint,
    resize_roi,
    sort_dicom_slices,
    tight_bbox,
    update_private_failures,
    validate_roi_entry,
    validate_annotation_mapping,
    write_roi,
)


def _slice(sop: str, projection: float, *, orientation: tuple[float, ...] = (1, 0, 0, 0, 1, 0)) -> DicomSlice:
    return DicomSlice(Path(f"/{sop}.dcm"), sop, (0.0, 0.0, projection), orientation, (0.7, 0.7), 1.0, 0.0, projection)


def test_numpy_compatibility_adds_only_missing_aliases() -> None:
    class Missing:
        bool_ = np.bool_

    class Existing:
        int = "present"
        bool = "present"
        bool_ = np.bool_

    missing = Missing()
    assert enable_pylidc_numpy_compatibility(missing) == (True, True)
    assert missing.int is int
    assert missing.bool is np.bool_
    assert enable_pylidc_numpy_compatibility(Existing()) == (False, False)


def test_spatial_sort_ignores_filename_and_instance_conventions() -> None:
    records = [_slice("z-file", 10.0), _slice("a-file", -5.0), _slice("m-file", 2.0)]
    assert [item.sop_uid for item in sort_dicom_slices(records)] == ["a-file", "m-file", "z-file"]


def test_orientation_mismatch_blocks_volume() -> None:
    with pytest.raises(ValueError, match="DICOM_ORIENTATION_INCONSISTENT"):
        sort_dicom_slices([_slice("a", 0.0), _slice("b", 1.0, orientation=(1, 0, 0, 0, 0, 1))])


@pytest.mark.parametrize("readers,expected", [(1, 1), (2, 1), (3, 2), (4, 2)])
def test_fifty_percent_consensus_boundary(readers: int, expected: int) -> None:
    masks = np.zeros((readers, 2, 2, 1), dtype=np.uint8)
    masks[:expected, 0, 0, 0] = 1
    consensus = np.mean(masks, axis=0) >= 0.5
    assert consensus[0, 0, 0]
    if expected > 1:
        masks[: expected - 1, 1, 1, 0] = 1
        assert not (np.mean(masks, axis=0) >= 0.5)[1, 1, 0]


def test_physical_z_mapping_handles_reversed_projection_order() -> None:
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[1, 0, 0] = 1
    bbox = (slice(0, 2), slice(0, 2), slice(0, 2))
    mapped = map_pylidc_mask_to_dhw(mask, bbox, np.array([10.0, 20.0]), np.array([20.0, 10.0]), (2, 2, 2))
    assert mapped[1, 1, 0] == 1
    assert mapped[0].sum() == 0


def test_unapproved_duplicate_plane_is_blocking() -> None:
    with pytest.raises(ValueError, match="DICOM_UNAPPROVED_DUPLICATE_SLICE_PLANE"):
        apply_duplicate_policy([_slice("a", 1.0), _slice("b", 1.0)], None)


def test_duplicate_sop_uid_blocks_even_on_distinct_planes() -> None:
    with pytest.raises(ValueError, match="DICOM_DUPLICATE_SOP_UID"):
        apply_duplicate_policy([_slice("same", 1.0), _slice("same", 2.0)], None)


def test_exact_duplicate_requires_the_p1_hashed_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dataset:
        pixel_array = np.array([[1]], dtype=np.int16)

    monkeypatch.setattr("lidc_baseline.p3_roi.pydicom.dcmread", lambda path: Dataset())
    first, second = _slice("first", 1.0), _slice("second", 1.0)
    selection = {
        "retained_sop_key": _key("sop", "first"),
        "discarded_sop_keys": _key("sop", "second"),
        "selection_rule": "RETAIN_LEXICOGRAPHICALLY_SMALLEST_SOP_UID",
    }
    retained, applied = apply_duplicate_policy([second, first], selection)
    assert [item.sop_uid for item in retained] == ["first"]
    assert applied is True


def test_different_content_duplicate_blocks_even_with_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    class Dataset:
        def __init__(self, value: int) -> None:
            self.pixel_array = np.array([[value]], dtype=np.int16)

    monkeypatch.setattr("lidc_baseline.p3_roi.pydicom.dcmread", lambda path: Dataset(1 if "first" in str(path) else 2))
    selection = {"retained_sop_key": _key("sop", "first"), "discarded_sop_keys": _key("sop", "second"), "selection_rule": "P1"}
    with pytest.raises(ValueError, match="P1_EXACT_DUPLICATE_PIXEL_CONTENT_MISMATCH"):
        apply_duplicate_policy([_slice("first", 1.0), _slice("second", 1.0)], selection)


def test_rescale_slope_intercept_hu_conversion() -> None:
    pixels = np.array([[-1000, 100]], dtype=np.int16)
    assert np.array_equal(convert_pixels_to_hu(pixels, 2.0, -1024.0), np.array([[-3024.0, -824.0]], dtype=np.float32))
    assert consensus_physical_volume_mm3(10, (2.0, 0.5, 0.5)) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="CONSENSUS_VOLUME_SPACING_INVALID"):
        consensus_physical_volume_mm3(1, (1.0, 0.0, 1.0))


def test_tight_bbox_and_high_side_odd_padding() -> None:
    mask = np.zeros((4, 3, 2), dtype=np.uint8)
    mask[1:4, 1:3, 0:1] = 1
    bbox = tight_bbox(mask)
    assert [(item.start, item.stop) for item in bbox] == [(1, 4), (1, 3), (0, 1)]
    image = np.full((3, 2, 1), -500.0, dtype=np.float32)
    cube_image, cube_mask, padding = pad_to_cube(image, np.ones_like(image, dtype=np.uint8))
    assert cube_image.shape == cube_mask.shape == (3, 3, 3)
    assert padding == ((0, 0), (0, 1), (1, 1))
    assert cube_image[0, 2, 0] == HU_MIN
    assert padding_ratio_for_dimensions([3, 2, 1]) == pytest.approx(1.0 - (6 / 27))


def test_resize_uses_trilinear_without_align_corners_and_nearest_binary_mask() -> None:
    image = np.full((2, 2, 2), HU_MIN, dtype=np.float32)
    image[1] = 700.0
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[:, :, 1] = 1
    normalized, resized_mask = resize_roi(image, mask)
    expected = torch.nn.functional.interpolate(torch.from_numpy(image)[None, None], size=ROI_SHAPE, mode="trilinear", align_corners=False)[0, 0]
    expected = ((expected.clamp(-1000, 700) + 1000) / 1700).numpy()
    assert normalized.shape == (1, *ROI_SHAPE)
    assert np.array_equal(normalized[0], expected)
    assert normalized[0, 0, 0, 0] == 0.0
    assert resized_mask.dtype == np.uint8
    assert set(np.unique(resized_mask)) == {0, 1}


def test_deterministic_npz_and_safe_resume(tmp_path: Path) -> None:
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8)
    mask[0, 1, 2, 3] = 1
    metadata = {"config_sha256": "config", "nodule_uid": "uid"}
    assert deterministic_npz_bytes(image, mask, metadata) == deterministic_npz_bytes(image, mask, metadata)
    path = tmp_path / "uid.npz"
    first = write_roi(path, image, mask, metadata)
    second = write_roi(path, image, mask, metadata)
    assert first["status"] == "WRITTEN"
    assert second["status"] == "REUSED"
    changed = image.copy(); changed[0, 0, 0, 0] = 1.0
    with pytest.raises(FileExistsError, match="ROI_EXISTS_WITH_DIFFERENT_CONTENT"):
        write_roi(path, changed, mask, metadata)
    assert write_roi(path, changed, mask, metadata, overwrite=True)["status"] == "WRITTEN"


def test_metadata_serialization_is_canonical_json() -> None:
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8)
    first = deterministic_npz_bytes(image, mask, {"b": 1, "a": [2, 3]})
    second = deterministic_npz_bytes(image, mask, {"a": [2, 3], "b": 1})
    assert first == second


def test_tracked_audit_privacy_rejects_raw_identifiers_and_absolute_paths(tmp_path: Path) -> None:
    audit = tmp_path / "summary.json"
    audit.write_text(json.dumps({"count": 1}), encoding="utf-8")
    assert_deidentified_audit(audit, {"LIDC-IDRI-0001", "1.2.3"})
    audit.write_text("LIDC-IDRI-0001", encoding="utf-8")
    with pytest.raises(ValueError, match="RAW_IDENTIFIER"):
        assert_deidentified_audit(audit, {"LIDC-IDRI-0001"})
    audit.write_text("/Users/example", encoding="utf-8")
    with pytest.raises(ValueError, match="ABSOLUTE_PATH"):
        assert_deidentified_audit(audit, set())


def test_annotation_mapping_must_exactly_match_manifest(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.parquet"
    import pandas as pd
    pd.DataFrame({"nodule_uid": ["one", "one"], "pylidc_annotation_sql_id": [4, 2]}).to_parquet(mapping, index=False)
    validate_annotation_mapping([{"nodule_uid": "one", "pylidc_annotation_sql_ids": "[2,4]"}], mapping)
    with pytest.raises(ValueError, match="ANNOTATION_MAPPING_MANIFEST_MISMATCH"):
        validate_annotation_mapping([{"nodule_uid": "one", "pylidc_annotation_sql_ids": "[2]"}], mapping)


def test_full_build_requires_matching_user_pilot_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text('{"nodule_uids":["one"]}', encoding="utf-8")
    marker = tmp_path / "marker.json"
    monkeypatch.setattr("lidc_baseline.p3_roi._pilot_confirmation_path", lambda: marker)
    with pytest.raises(PermissionError, match="PILOT_QA_USER_CONFIRMATION_REQUIRED"):
        require_pilot_confirmation("config", selection)
    marker.write_text(json.dumps({"user_confirmation": True, "config_sha256": "config", "pilot_selection_sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(PermissionError, match="PILOT_QA_CONFIRMATION_MARKER_INVALID"):
        require_pilot_confirmation("config", selection)
    import hashlib
    marker.write_text(json.dumps({"user_confirmation": True, "config_sha256": "config", "pilot_selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest()}), encoding="utf-8")
    require_pilot_confirmation("config", selection)


def test_pilot_statistics_cache_reuses_only_matching_source_geometry() -> None:
    row = {"nodule_uid": "one", "canonical_xml_sha256": "xml", "annotation_source_fingerprints": "annotations", "source_dicom_sop_fingerprints": "sops-a", "series_instance_uid": "series"}
    class Scan:
        slice_zvals = np.array([1.0, 2.0])
        slice_spacing = 1.0
        pixel_spacing = 0.5
    first_geometry = scan_geometry_fingerprint(Scan())
    matching = {"nodule_uid": "one", "source_fingerprint": pilot_statistics_source_fingerprint(row, first_geometry), "physical_volume_mm3": 1.0}
    assert reusable_pilot_statistics([matching], [row], {"one": first_geometry}) == {"one": matching}
    changed_geometry = scan_geometry_fingerprint(type("Changed", (), {"slice_zvals": np.array([1.0, 3.0]), "slice_spacing": 2.0, "pixel_spacing": 0.5})())
    assert reusable_pilot_statistics([matching], [row], {"one": changed_geometry}) == {}
    missing = {"nodule_uid": "two", **{key: value for key, value in row.items() if key != "nodule_uid"}}
    resumed = reusable_pilot_statistics([matching], [row, missing], {"one": first_geometry, "two": first_geometry})
    assert set(resumed) == {"one"}


def test_roi_verifier_rejects_bad_status_uid_and_metadata_hash(tmp_path: Path) -> None:
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8); mask[0, 1, 1, 1] = 1
    path = tmp_path / "rois" / "uid.npz"
    outcome = write_roi(path, image, mask, {"config_sha256": "config", "nodule_uid": "uid"})
    row = {"status": "WRITTEN", "relative_roi_path": "rois/uid.npz", "roi_file_sha256": outcome["content_sha256"]}
    validate_roi_entry("uid", row, "config", tmp_path)
    with pytest.raises(ValueError, match="STATUS_INVALID"):
        validate_roi_entry("uid", {**row, "status": "FAILED"}, "config", tmp_path)
    with pytest.raises(ValueError, match="CONTENT_INVALID"):
        validate_roi_entry("other", row, "config", tmp_path)


def test_reusable_roi_requires_matching_manifest_provenance(tmp_path: Path) -> None:
    row = {
        "nodule_uid": "uid", "reader_count": 2, "canonical_xml_sha256": "xml",
        "annotation_source_fingerprints": "annotations", "source_dicom_sop_fingerprints": "sops",
    }
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8); mask[0, 1, 1, 1] = 1
    metadata = {
        "config_sha256": "config", "nodule_uid": "uid", "source_fingerprint": roi_source_fingerprint(row),
        "tight_bbox_dhw": [[0, 1], [0, 1], [0, 1]], "cube_edge_voxels": 1,
        "padding_dhw": [[0, 0], [0, 0], [0, 0]], "source_spacing_dhw_mm": [1.0, 1.0, 1.0],
        "pre_resize_mask_voxels": 1, "post_resize_mask_voxels": 1,
        "exact_duplicate_selection_applied": False,
    }
    write_roi(tmp_path / "rois" / "uid.npz", image, mask, metadata)
    index, reused_metadata = reusable_roi(row, "config", tmp_path / "rois") or (None, None)
    assert index is not None and index["status"] == "REUSED"
    assert reused_metadata is not None and reused_metadata["nodule_uid"] == "uid"
    changed = {**row, "source_dicom_sop_fingerprints": "different-sops"}
    with pytest.raises(FileExistsError, match="ROI_EXISTS_WITH_DIFFERENT_PROVENANCE"):
        reusable_roi(changed, "config", tmp_path / "rois")


def test_failed_reuse_is_persisted_in_private_failure_registry(tmp_path: Path) -> None:
    row = {
        "nodule_uid": "uid", "reader_count": 1, "canonical_xml_sha256": "xml",
        "annotation_source_fingerprints": "annotations", "source_dicom_sop_fingerprints": "expected",
    }
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8); mask[0, 1, 1, 1] = 1
    metadata = {
        "config_sha256": "config", "nodule_uid": "uid", "source_fingerprint": "wrong",
        "tight_bbox_dhw": [[0, 1], [0, 1], [0, 1]], "cube_edge_voxels": 1,
        "padding_dhw": [[0, 0], [0, 0], [0, 0]], "source_spacing_dhw_mm": [1.0, 1.0, 1.0],
        "pre_resize_mask_voxels": 1, "post_resize_mask_voxels": 1,
        "exact_duplicate_selection_applied": False,
    }
    write_roi(tmp_path / "rois" / "uid.npz", image, mask, metadata)
    with pytest.raises(FileExistsError, match="ROI_EXISTS_WITH_DIFFERENT_PROVENANCE"):
        reusable_roi(row, "config", tmp_path / "rois")
    failures = tmp_path / "roi_failures.parquet"
    failure = {"nodule_uid": "uid", "patient_id": "private", "series_instance_uid": "private", "reason": "FileExistsError:ROI_EXISTS_WITH_DIFFERENT_PROVENANCE"}
    update_private_failures(failures, ["uid"], [failure])
    assert pd.read_parquet(failures).to_dict(orient="records") == [failure]


def test_existing_roi_provenance_mismatch_requires_explicit_overwrite(tmp_path: Path) -> None:
    row = {
        "nodule_uid": "uid", "reader_count": 1, "canonical_xml_sha256": "xml",
        "annotation_source_fingerprints": "annotations", "source_dicom_sop_fingerprints": "expected",
    }
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8); mask[0, 1, 1, 1] = 1
    metadata = {
        "config_sha256": "config", "nodule_uid": "uid", "source_fingerprint": "wrong",
        "tight_bbox_dhw": [[0, 1], [0, 1], [0, 1]], "cube_edge_voxels": 1,
        "padding_dhw": [[0, 0], [0, 0], [0, 0]], "source_spacing_dhw_mm": [1.0, 1.0, 1.0],
        "pre_resize_mask_voxels": 1, "post_resize_mask_voxels": 1,
        "exact_duplicate_selection_applied": False,
    }
    write_roi(tmp_path / "rois" / "uid.npz", image, mask, metadata)
    with pytest.raises(FileExistsError, match="ROI_EXISTS_WITH_DIFFERENT_PROVENANCE"):
        reuse_or_schedule_rebuild(row, "config", tmp_path / "rois", overwrite=False)
    assert reuse_or_schedule_rebuild(row, "config", tmp_path / "rois", overwrite=True) is None


def test_corrupt_existing_roi_requires_explicit_overwrite(tmp_path: Path) -> None:
    row = {
        "nodule_uid": "uid", "reader_count": 1, "canonical_xml_sha256": "xml",
        "annotation_source_fingerprints": "annotations", "source_dicom_sop_fingerprints": "sops",
    }
    path = tmp_path / "rois" / "uid.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not-a-zip")
    with pytest.raises(zipfile.BadZipFile):
        reuse_or_schedule_rebuild(row, "config", tmp_path / "rois", overwrite=False)
    assert reuse_or_schedule_rebuild(row, "config", tmp_path / "rois", overwrite=True) is None


def test_qa_writer_handles_single_slice_source_crop(tmp_path: Path) -> None:
    source = np.zeros((1, 4, 4), dtype=np.float32)
    source_mask = np.zeros((1, 4, 4), dtype=np.uint8); source_mask[0, 1, 1] = 1
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8); mask[0, 0, 1, 1] = 1
    target = tmp_path / "qa.png"
    _qa_image(target, source, source_mask, image, mask, "deidentified")
    assert target.exists() and target.stat().st_size > 0


def test_qa_writer_keeps_visible_mask_fallback_when_contour_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from matplotlib.axes import Axes
    monkeypatch.setattr(Axes, "contour", lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("degenerate contour")))
    source = np.zeros((2, 4, 4), dtype=np.float32)
    source_mask = np.zeros((2, 4, 4), dtype=np.uint8); source_mask[0, 1, 1] = 1
    image = np.zeros((1, *ROI_SHAPE), dtype=np.float32)
    mask = np.zeros((1, *ROI_SHAPE), dtype=np.uint8); mask[0, 2, 2, 2] = 1
    target = tmp_path / "qa_fallback.png"
    _qa_image(target, source, source_mask, image, mask, "deidentified")
    assert target.exists() and target.stat().st_size > 0


def test_failure_registry_clears_successful_retry_but_keeps_unrelated_failures(tmp_path: Path) -> None:
    path = tmp_path / "failures.parquet"
    import pandas as pd
    pd.DataFrame([
        {"nodule_uid": "retry", "patient_id": "p", "series_instance_uid": "s", "reason": "old"},
        {"nodule_uid": "other", "patient_id": "p", "series_instance_uid": "s", "reason": "still"},
    ]).to_parquet(path, index=False)
    update_private_failures(path, ["retry"], [])
    assert pd.read_parquet(path)["nodule_uid"].tolist() == ["other"]
