"""Apply the user-approved Phase 1 duplicate-slice resolution policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as element_tree
from collections import defaultdict
from pathlib import Path
from typing import Any

import pydicom

from lidc_baseline import __version__
from lidc_baseline.audit import write_json
from lidc_baseline.p1_audit import (
    DUPLICATE_PROJECTION_TOLERANCE_MM,
    _key,
    _local_name,
    _number_sequence,
    _unit_cross,
    _write_csv,
    resolve_paths,
)


def classify_duplicate_plane(images: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Classify one duplicate spatial plane without modifying source images."""
    hashes = {sop_uid: hashlib.sha256(pixel_bytes).hexdigest() for sop_uid, pixel_bytes in images}
    if len(set(hashes.values())) == 1:
        retained = min(hashes)
        return {
            "classification": "EXACT_DUPLICATE_IMAGE_CONTENT",
            "action": "RETAIN_LEXICOGRAPHICALLY_SMALLEST_SOP_UID",
            "retained_sop_uid": retained,
            "discarded_sop_uids": sorted(uid for uid in hashes if uid != retained),
        }
    return {
        "classification": "DIFFERENT_IMAGE_CONTENT",
        "action": "EXCLUDE_ENTIRE_SERIES",
        "retained_sop_uid": None,
        "discarded_sop_uids": [],
    }


def _target_series_keys(anomalies_path: Path) -> set[str]:
    with anomalies_path.open(newline="", encoding="utf-8") as stream:
        return {
            row["entity_key"]
            for row in csv.DictReader(stream)
            if row["code"] == "DUPLICATE_SLICE_PLANE"
        }


def _target_series_records(dicom_root: Path, target_keys: set[str]) -> dict[str, list[dict[str, Any]]]:
    tags = ["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "ImagePositionPatient", "ImageOrientationPatient"]
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(dicom_root.rglob("*.dcm")):
        dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        series_uid = str(getattr(dataset, "SeriesInstanceUID", ""))
        series_key = _key("series", series_uid)
        if series_key not in target_keys:
            continue
        series[series_key].append(
            {
                "path": path,
                "series_uid": series_uid,
                "patient_id": str(getattr(dataset, "PatientID", "")),
                "study_uid": str(getattr(dataset, "StudyInstanceUID", "")),
                "sop_uid": str(getattr(dataset, "SOPInstanceUID", "")),
                "position": _number_sequence(getattr(dataset, "ImagePositionPatient", None), 3),
                "orientation": _number_sequence(getattr(dataset, "ImageOrientationPatient", None), 6),
            }
        )
    return series


def _duplicate_groups(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    reference = next(record["orientation"] for record in records if record["orientation"] is not None)
    normal = _unit_cross(reference[:3], reference[3:])
    if normal is None:
        raise ValueError("Duplicate-plane series has degenerate orientation")
    projected = []
    for record in records:
        if record["position"] is None:
            continue
        projection = sum(left * right for left, right in zip(record["position"], normal, strict=True))
        projected.append((projection, record["sop_uid"], record))
    projected.sort(key=lambda item: (item[0], item[1]))
    groups: list[list[tuple[float, str, dict[str, Any]]]] = []
    for item in projected:
        if not groups or item[0] - groups[-1][0][0] > DUPLICATE_PROJECTION_TOLERANCE_MM:
            groups.append([item])
        else:
            groups[-1].append(item)
    return [[record for _, _, record in group] for group in groups if len(group) > 1]


def _reader_annotation_counts(canonical_xml: Path, excluded_series_uids: set[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for path in sorted(canonical_xml.rglob("*.xml")):
        try:
            root = element_tree.parse(path).getroot()
        except element_tree.ParseError:
            continue
        if _local_name(root.tag) != "LidcReadMessage":
            continue
        series_uids = {
            (element.text or "").strip()
            for element in root.iter()
            if _local_name(element.tag) == "SeriesInstanceUid" and (element.text or "").strip()
        }
        if len(series_uids) != 1 or next(iter(series_uids)) not in excluded_series_uids:
            continue
        counts[next(iter(series_uids))] += sum(1 for element in root.iter() if _local_name(element.tag) == "unblindedReadNodule")
    return counts


def run_resolution(raw_data: str | Path, audit_output: str | Path) -> dict[str, Any]:
    """Produce approved eligibility metadata while preserving every raw DICOM file."""
    paths = resolve_paths(raw_data, audit_output)
    target_keys = _target_series_keys(paths.output / "anomalies.csv")
    target_series = _target_series_records(paths.dicom_root, target_keys)
    if set(target_series) != target_keys:
        missing = len(target_keys - set(target_series))
        raise RuntimeError(f"Could not locate {missing} duplicate-plane series from the audit report")

    resolutions: list[dict[str, Any]] = []
    excluded_uids: set[str] = set()
    patients: set[str] = set()
    exact_selection_rows: list[dict[str, str]] = []
    for series_key in sorted(target_series):
        records = target_series[series_key]
        plane_decisions = []
        for group in _duplicate_groups(records):
            images = []
            for record in group:
                dataset = pydicom.dcmread(record["path"], stop_before_pixels=False)
                images.append((record["sop_uid"], bytes(dataset.PixelData)))
            decision = classify_duplicate_plane(images)
            plane_decisions.append(decision)
            if decision["classification"] == "EXACT_DUPLICATE_IMAGE_CONTENT":
                exact_selection_rows.append(
                    {
                        "series_key": series_key,
                        "retained_sop_key": _key("sop", decision["retained_sop_uid"]),
                        "discarded_sop_keys": ";".join(_key("sop", uid) for uid in decision["discarded_sop_uids"]),
                        "selection_rule": decision["action"],
                    }
                )
        has_different = any(item["classification"] == "DIFFERENT_IMAGE_CONTENT" for item in plane_decisions)
        decision = "EXCLUDED_DIFFERENT_DUPLICATE_PLANE" if has_different else "ELIGIBLE_EXACT_DUPLICATE_DEDUPLICATED"
        series_uid = records[0]["series_uid"]
        if has_different:
            excluded_uids.add(series_uid)
            patients.add(records[0]["patient_id"])
        resolutions.append(
            {
                "series_key": series_key,
                "decision": decision,
                "exact_duplicate_plane_groups": sum(item["classification"] == "EXACT_DUPLICATE_IMAGE_CONTENT" for item in plane_decisions),
                "different_duplicate_plane_groups": sum(item["classification"] == "DIFFERENT_IMAGE_CONTENT" for item in plane_decisions),
                "raw_dicom_modified": "false",
            }
        )

    annotations = _reader_annotation_counts(paths.canonical_xml, excluded_uids)
    total_ct_series = json.loads((paths.output / "summary.json").read_text(encoding="utf-8"))["counts"]["ct_series"]
    summary = {
        "audit": "P1 approved duplicate-plane resolution",
        "program_version": __version__,
        "policy": {
            "exact_duplicate": "Retain lexicographically smallest SOP UID only when constructing a derived CT volume; never modify raw DICOM.",
            "different_image_content": "Exclude the complete CT series from Baseline-v1; never select one duplicate slice automatically.",
        },
        "counts": {
            "duplicate_plane_series_reviewed": len(resolutions),
            "exact_duplicate_series_eligible": sum(row["decision"] == "ELIGIBLE_EXACT_DUPLICATE_DEDUPLICATED" for row in resolutions),
            "different_image_content_series_excluded": len(excluded_uids),
            "patients_with_excluded_series": len(patients),
            "affected_nodule_ge_3mm_reader_annotations": sum(annotations.values()),
            "physical_nodule_count_available_in_p1": False,
            "ct_series_eligible_for_downstream_pipeline": total_ct_series - len(excluded_uids),
            "raw_dicom_files_modified": 0,
        },
        "excluded_series_keys": sorted(_key("series", uid) for uid in excluded_uids),
    }
    resolutions.sort(key=lambda row: row["series_key"])
    exact_selection_rows.sort(key=lambda row: (row["series_key"], row["retained_sop_key"]))
    write_json(paths.output / "duplicate_plane_resolution.json", summary)
    _write_csv(
        paths.output / "duplicate_plane_resolution.csv",
        resolutions,
        ["series_key", "decision", "exact_duplicate_plane_groups", "different_duplicate_plane_groups", "raw_dicom_modified"],
    )
    _write_csv(
        paths.output / "derived_volume_selection.csv",
        exact_selection_rows,
        ["series_key", "retained_sop_key", "discarded_sop_keys", "selection_rule"],
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--audit-output", required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(run_resolution(arguments.raw_data, arguments.audit_output)["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
