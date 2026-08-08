"""Tests for the Phase 1 header-only XML/DICOM audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from lidc_baseline.p1_audit import AuditPaths, resolve_paths, run_audit
from lidc_baseline.p1_resolution import classify_duplicate_plane


def _write_xml(path: Path, root_name: str, study_uid: str, series_uid: str | None) -> None:
    series = f"<SeriesInstanceUid>{series_uid}</SeriesInstanceUid>" if series_uid else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<{root_name} xmlns=\"http://www.nih.gov\">
  <ResponseHeader><StudyInstanceUID>{study_uid}</StudyInstanceUID>{series}</ResponseHeader>
</{root_name}>
""",
        encoding="utf-8",
    )


def _write_dicom(
    path: Path,
    *,
    patient_id: str,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    modality: str = "CT",
    position: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    instance_number: int | None = 1,
    slice_thickness: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.PatientID = patient_id
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.SOPInstanceUID = sop_uid
    dataset.Modality = modality
    if modality == "CT":
        if position is not None:
            dataset.ImagePositionPatient = list(position)
        if orientation is not None:
            dataset.ImageOrientationPatient = list(orientation)
        dataset.PixelSpacing = [0.7, 0.7]
        dataset.SliceThickness = slice_thickness
        if instance_number is not None:
            dataset.InstanceNumber = instance_number
    pydicom.dcmwrite(path, dataset, write_like_original=False)


def _paths(tmp_path: Path) -> AuditPaths:
    raw = tmp_path / "raw"
    canonical = raw / "LIDC-XML-only" / "tcia-lidc-xml"
    dicom = raw / "manifest-1600709154662" / "LIDC-IDRI"
    canonical.mkdir(parents=True)
    dicom.mkdir(parents=True)
    return AuditPaths(raw, canonical, dicom, tmp_path / "audit")


def _valid_fixture(tmp_path: Path) -> tuple[AuditPaths, dict[str, str]]:
    paths = _paths(tmp_path)
    ids = {name: generate_uid() for name in ("study", "series", "sop0", "sop1", "dx", "cr")}
    _write_xml(paths.canonical_xml / "ct.xml", "LidcReadMessage", ids["study"], ids["series"])
    _write_xml(paths.canonical_xml / "cxr.xml", "IdriReadMessage", ids["study"], generate_uid())
    _write_xml(paths.dicom_root / "embedded.xml", "LidcReadMessage", ids["study"], ids["series"])
    patient_root = paths.dicom_root / "LIDC-IDRI-TEST" / "study" / "series"
    _write_dicom(patient_root / "z-last.dcm", patient_id="PATIENT-RAW", study_uid=ids["study"], series_uid=ids["series"], sop_uid=ids["sop1"], position=(0.0, 0.0, 1.0), instance_number=1)
    _write_dicom(patient_root / "a-first.dcm", patient_id="PATIENT-RAW", study_uid=ids["study"], series_uid=ids["series"], sop_uid=ids["sop0"], position=(0.0, 0.0, 0.0), instance_number=99)
    _write_dicom(patient_root / "dx.dcm", patient_id="PATIENT-RAW", study_uid=ids["study"], series_uid=generate_uid(), sop_uid=ids["dx"], modality="DX")
    _write_dicom(patient_root / "cr.dcm", patient_id="PATIENT-RAW", study_uid=ids["study"], series_uid=generate_uid(), sop_uid=ids["cr"], modality="CR")
    return paths, ids


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_resolve_paths_uses_fixed_layout_and_rejects_missing_roots(tmp_path: Path) -> None:
    paths, _ = _valid_fixture(tmp_path)
    resolved = resolve_paths(paths.raw_data, paths.output)
    assert resolved.canonical_xml == paths.canonical_xml.resolve()
    assert resolved.dicom_root == paths.dicom_root.resolve()
    with pytest.raises(FileNotFoundError, match="raw_data"):
        resolve_paths(tmp_path / "missing", paths.output)


def test_audit_separates_canonical_ct_cxr_and_embedded_xml(tmp_path: Path) -> None:
    paths, _ = _valid_fixture(tmp_path)
    summary = run_audit(paths)
    assert summary["counts"]["canonical_xml_files"] == 2
    assert summary["counts"]["canonical_xml_roots"] == {"IdriReadMessage": 1, "LidcReadMessage": 1}
    assert summary["counts"]["canonical_ct_xml"] == 1
    assert summary["counts"]["canonical_cxr_xml"] == 1
    assert summary["counts"]["download_tree_xml_files"] == 1
    assert summary["counts"]["download_tree_xml_roots"] == {"LidcReadMessage": 1}
    assert summary["counts"]["ct_series"] == 1
    assert summary["counts"]["dicom_modalities"] == {"CR": 1, "CT": 2, "DX": 1}
    assert summary["reference_reconciliation"]["ct_series"]["hard_gate"] is False
    assert summary["issues"]["by_severity"] == {}
    series = _read_csv(paths.output / "series_audit.csv")
    assert len(series) == 1
    assert series[0]["mapping_status"] == "MAPPED"
    assert series[0]["volume_status"] == "DETERMINISTIC"


def test_audit_uses_spatial_projection_not_file_name_or_instance_number(tmp_path: Path) -> None:
    paths, _ = _valid_fixture(tmp_path)
    run_audit(paths, retain_instance_detail=True)
    detail = pd_read_parquet(paths.output / "local" / "instances.parquet")
    assert detail["projection_mm"].tolist() == [0.0, 1.0]
    assert detail["instance_number"].tolist() == ["99", "1"]


def pd_read_parquet(path: Path):
    """Keep pandas import local to make test fixture dependencies explicit."""
    import pandas as pd

    return pd.read_parquet(path)


def test_audit_records_mapping_and_geometry_exceptions(tmp_path: Path) -> None:
    paths, ids = _valid_fixture(tmp_path)
    _write_xml(paths.canonical_xml / "unmatched.xml", "LidcReadMessage", ids["study"], generate_uid())
    _write_xml(paths.canonical_xml / "missing-series.xml", "LidcReadMessage", ids["study"], None)
    _write_xml(paths.canonical_xml / "study-conflict.xml", "LidcReadMessage", generate_uid(), ids["series"])
    (paths.canonical_xml / "duplicate-content.xml").write_bytes((paths.canonical_xml / "ct.xml").read_bytes())
    (paths.canonical_xml / "broken.xml").write_text("<LidcReadMessage>", encoding="utf-8")
    (paths.dicom_root / "broken.dcm").write_bytes(b"not-a-dicom")
    _write_dicom(
        paths.dicom_root / "LIDC-IDRI-TEST" / "study" / "series" / "duplicate-plane.dcm",
        patient_id="PATIENT-RAW",
        study_uid=ids["study"],
        series_uid=ids["series"],
        sop_uid=generate_uid(),
        position=(0.0, 0.0, 0.0),
        instance_number=2,
    )
    _write_dicom(
        paths.dicom_root / "LIDC-IDRI-TEST" / "study" / "series" / "orientation.dcm",
        patient_id="PATIENT-RAW",
        study_uid=ids["study"],
        series_uid=ids["series"],
        sop_uid=generate_uid(),
        position=(0.0, 0.0, 3.0),
        orientation=(0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        instance_number=3,
        slice_thickness=2.0,
    )
    _write_dicom(
        paths.dicom_root / "LIDC-IDRI-TEST" / "study" / "series" / "missing-position.dcm",
        patient_id="PATIENT-RAW",
        study_uid=ids["study"],
        series_uid=ids["series"],
        sop_uid=generate_uid(),
        position=None,
        instance_number=None,
    )
    summary = run_audit(paths)
    assert summary["issues"]["by_code"]["XML_CT_SERIES_UNMATCHED"] == 1
    assert summary["issues"]["by_code"]["XML_SERIES_UID_INVALID"] == 1
    assert summary["issues"]["by_code"]["XML_DICOM_STUDY_UID_CONFLICT"] == 1
    assert summary["issues"]["by_code"]["DUPLICATE_CANONICAL_XML_CONTENT"] == 2
    assert summary["issues"]["by_code"]["XML_PARSE_ERROR"] == 1
    assert summary["issues"]["by_code"]["DICOM_PARSE_ERROR"] == 1
    assert summary["issues"]["by_code"]["DUPLICATE_SLICE_PLANE"] == 1
    assert summary["issues"]["by_code"]["ORIENTATION_INCONSISTENT"] == 1
    assert summary["issues"]["by_code"]["IMAGE_POSITION_INVALID"] == 1
    assert summary["issues"]["by_code"]["INSTANCE_NUMBER_MISSING"] == 1
    assert summary["issues"]["by_code"]["SLICE_THICKNESS_INCONSISTENT"] == 1
    series = _read_csv(paths.output / "series_audit.csv")
    assert series[0]["volume_status"] == "BLOCKING_EXCEPTION"


def test_audit_detects_nonuniform_spacing_and_suspected_gap(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    study_uid, series_uid = generate_uid(), generate_uid()
    _write_xml(paths.canonical_xml / "ct.xml", "LidcReadMessage", study_uid, series_uid)
    series_root = paths.dicom_root / "LIDC-IDRI-TEST" / "study" / "series"
    for index, z_position in enumerate((0.0, 1.0, 2.0, 5.0), start=1):
        _write_dicom(
            series_root / f"slice-{index}.dcm",
            patient_id="PATIENT-RAW",
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=generate_uid(),
            position=(0.0, 0.0, z_position),
            instance_number=index,
        )
    summary = run_audit(paths)
    assert summary["issues"]["by_code"]["SPACING_NONUNIFORM"] == 1
    assert summary["issues"]["by_code"]["SUSPECTED_SLICE_GAP"] == 1


def test_audit_is_byte_deterministic_and_reports_are_deidentified(tmp_path: Path) -> None:
    paths, ids = _valid_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    run_audit(AuditPaths(paths.raw_data, paths.canonical_xml, paths.dicom_root, first))
    run_audit(AuditPaths(paths.raw_data, paths.canonical_xml, paths.dicom_root, second))
    for name in ("summary.json", "series_audit.csv", "anomalies.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    tracked_text = "\n".join((first / name).read_text(encoding="utf-8") for name in ("summary.json", "series_audit.csv", "anomalies.csv"))
    assert "PATIENT-RAW" not in tracked_text
    assert ids["study"] not in tracked_text
    assert str(paths.raw_data) not in tracked_text
    assert json.loads((first / "summary.json").read_text(encoding="utf-8"))["privacy"].startswith("Tracked reports")


def test_duplicate_plane_resolution_only_deduplicates_identical_pixel_bytes() -> None:
    exact = classify_duplicate_plane([("2.25.1", b"identical"), ("2.25.2", b"identical")])
    assert exact == {
        "classification": "EXACT_DUPLICATE_IMAGE_CONTENT",
        "action": "RETAIN_LEXICOGRAPHICALLY_SMALLEST_SOP_UID",
        "retained_sop_uid": "2.25.1",
        "discarded_sop_uids": ["2.25.2"],
    }
    different = classify_duplicate_plane([("2.25.1", b"first"), ("2.25.2", b"second")])
    assert different["classification"] == "DIFFERENT_IMAGE_CONTENT"
    assert different["action"] == "EXCLUDE_ENTIRE_SERIES"
    assert different["retained_sop_uid"] is None
