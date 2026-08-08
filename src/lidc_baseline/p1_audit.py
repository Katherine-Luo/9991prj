"""Deterministic, header-only audit for LIDC-IDRI XML and DICOM sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import xml.etree.ElementTree as element_tree
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pydicom

from lidc_baseline import __version__
from lidc_baseline.audit import write_json


ORIENTATION_TOLERANCE = 1e-5
DUPLICATE_PROJECTION_TOLERANCE_MM = 1e-4
SPACING_ABSOLUTE_TOLERANCE_MM = 0.1
SPACING_RELATIVE_TOLERANCE = 0.10
GAP_MULTIPLIER = 1.5
REFERENCE_INVENTORY = {
    "patient_directories": 1010,
    "ct_series": 1018,
    "ct_instances": 243958,
    "dx_instances": 513,
    "cr_instances": 56,
    "canonical_xml_files": 1319,
}


@dataclass(frozen=True)
class AuditPaths:
    """Resolved source and output locations for the P1 audit."""

    raw_data: Path
    canonical_xml: Path
    dicom_root: Path
    output: Path


@dataclass
class Issue:
    """A non-sensitive audit finding."""

    severity: str
    code: str
    entity_type: str
    entity_key: str
    detail: str

    def as_row(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "entity_type": self.entity_type,
            "entity_key": self.entity_key,
            "detail": self.detail,
        }


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def _digest(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _key(kind: str, value: str | None) -> str:
    return _digest(f"{kind}:{value or ''}")


def _relative_key(path: Path, root: Path) -> str:
    return _key("relative-path", path.relative_to(root).as_posix())


def _text_values(root: element_tree.Element, name: str) -> list[str]:
    return [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element.tag) == name and (element.text or "").strip()
    ]


def _single_value(values: Iterable[str]) -> tuple[str | None, bool]:
    unique = sorted(set(values))
    return (unique[0], len(unique) == 1) if unique else (None, False)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write deterministic CSV atomically without persisting raw input paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def resolve_paths(
    raw_data: str | Path,
    output: str | Path,
    canonical_xml: str | Path | None = None,
    dicom_root: str | Path | None = None,
) -> AuditPaths:
    """Resolve fixed LIDC source layouts, failing rather than guessing."""
    raw = Path(raw_data).expanduser().resolve()
    canonical = (
        Path(canonical_xml).expanduser().resolve()
        if canonical_xml is not None
        else raw / "LIDC-XML-only" / "tcia-lidc-xml"
    )
    dicom = (
        Path(dicom_root).expanduser().resolve()
        if dicom_root is not None
        else raw / "manifest-1600709154662" / "LIDC-IDRI"
    )
    for label, path in (("raw_data", raw), ("canonical_xml", canonical), ("dicom_root", dicom)):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return AuditPaths(raw, canonical, dicom, Path(output).expanduser().resolve())


def _canonical_xml_records(paths: AuditPaths, issues: list[Issue]) -> tuple[list[dict[str, Any]], Counter[str], str]:
    records: list[dict[str, Any]] = []
    root_counts: Counter[str] = Counter()
    tree_items: list[str] = []
    for path in sorted(paths.canonical_xml.rglob("*.xml")):
        relative = path.relative_to(paths.canonical_xml).as_posix()
        content = path.read_bytes()
        xml_hash = _digest(content)
        tree_items.append(f"{relative}\t{xml_hash}")
        file_key = _key("canonical-xml-source", f"{relative}\t{xml_hash}")
        try:
            root = element_tree.fromstring(content)
        except element_tree.ParseError:
            issues.append(Issue("BLOCKING", "XML_PARSE_ERROR", "canonical_xml", file_key, "XML parsing failed"))
            records.append({"xml_key": file_key, "content_key": _key("xml-content", xml_hash), "root_type": "UNPARSEABLE", "series_uid": None, "study_uid": None})
            continue
        root_type = _local_name(root.tag)
        root_counts[root_type] += 1
        series_uid, series_is_single = _single_value(_text_values(root, "SeriesInstanceUid"))
        study_uid, study_is_single = _single_value(_text_values(root, "StudyInstanceUID"))
        record = {
            "xml_key": file_key,
            "content_key": _key("xml-content", xml_hash),
            "root_type": root_type,
            "series_uid": series_uid,
            "study_uid": study_uid,
        }
        records.append(record)
        if root_type != "LidcReadMessage":
            continue
        if not series_is_single:
            issues.append(Issue("BLOCKING", "XML_SERIES_UID_INVALID", "canonical_xml", file_key, "Expected exactly one SeriesInstanceUid"))
        if not study_is_single:
            issues.append(Issue("BLOCKING", "XML_STUDY_UID_INVALID", "canonical_xml", file_key, "Expected exactly one StudyInstanceUID"))
    return records, root_counts, _digest("\n".join(tree_items))


def _download_tree_xml_inventory(paths: AuditPaths, issues: list[Issue]) -> tuple[Counter[str], int]:
    """Count embedded XML without permitting it to enter canonical mapping."""
    root_counts: Counter[str] = Counter()
    total = 0
    for path in sorted(paths.dicom_root.rglob("*.xml")):
        total += 1
        source_key = _relative_key(path, paths.dicom_root)
        try:
            root_counts[_local_name(element_tree.parse(path).getroot().tag)] += 1
        except element_tree.ParseError:
            root_counts["UNPARSEABLE"] += 1
            issues.append(Issue("WARNING", "EMBEDDED_XML_PARSE_ERROR", "embedded_xml", source_key, "DICOM-tree XML parsing failed; it was excluded from canonical mapping"))
    return root_counts, total


def _number_sequence(value: Any, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != length or not all(math.isfinite(item) for item in result):
        return None
    return result


def _header_value(dataset: Any, name: str) -> str | None:
    value = getattr(dataset, name, None)
    return None if value in (None, "") else str(value)


def _dicom_records(paths: AuditPaths, issues: list[Issue]) -> tuple[list[dict[str, Any]], Counter[str], str]:
    tags = [
        "PatientID",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Modality",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PixelSpacing",
        "SliceThickness",
        "InstanceNumber",
    ]
    records: list[dict[str, Any]] = []
    modalities: Counter[str] = Counter()
    inventory_items: list[str] = []
    for path in sorted(paths.dicom_root.rglob("*.dcm")):
        relative = path.relative_to(paths.dicom_root).as_posix()
        path_key = _relative_key(path, paths.dicom_root)
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        except Exception as error:  # pydicom exposes several parse exception types.
            issues.append(Issue("BLOCKING", "DICOM_PARSE_ERROR", "dicom_file", path_key, type(error).__name__))
            inventory_items.append(f"{relative}\tUNPARSEABLE")
            continue
        modality = _header_value(dataset, "Modality") or "MISSING"
        modalities[modality] += 1
        record = {
            "path_key": path_key,
            "patient_id": _header_value(dataset, "PatientID"),
            "study_uid": _header_value(dataset, "StudyInstanceUID"),
            "series_uid": _header_value(dataset, "SeriesInstanceUID"),
            "sop_uid": _header_value(dataset, "SOPInstanceUID"),
            "modality": modality,
            "position": _number_sequence(getattr(dataset, "ImagePositionPatient", None), 3),
            "orientation": _number_sequence(getattr(dataset, "ImageOrientationPatient", None), 6),
            "pixel_spacing": _number_sequence(getattr(dataset, "PixelSpacing", None), 2),
            "slice_thickness": _header_value(dataset, "SliceThickness"),
            "instance_number": _header_value(dataset, "InstanceNumber"),
        }
        records.append(record)
        inventory_items.append(
            "\t".join(str(record[name] or "") for name in ("patient_id", "study_uid", "series_uid", "sop_uid", "modality"))
        )
    return records, modalities, _digest("\n".join(inventory_items))


def _add_issue(issues: list[Issue], severity: str, code: str, series_uid: str, detail: str) -> None:
    issues.append(Issue(severity, code, "ct_series", _key("series", series_uid), detail))


def _unit_cross(row: tuple[float, float, float], column: tuple[float, float, float]) -> tuple[float, float, float] | None:
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    magnitude = math.sqrt(sum(value * value for value in normal))
    if magnitude <= ORIENTATION_TOLERANCE:
        return None
    return tuple(value / magnitude for value in normal)


def _near(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def _series_row(series_uid: str, records: list[dict[str, Any]], xml_count: int, issues: list[Issue]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one CT series and return a safe series row and optional slice rows."""
    before = len(issues)
    patient_ids = {record["patient_id"] for record in records if record["patient_id"]}
    study_uids = {record["study_uid"] for record in records if record["study_uid"]}
    sop_uids = [record["sop_uid"] for record in records]
    if len(patient_ids) != 1:
        _add_issue(issues, "BLOCKING", "PATIENT_ID_CONFLICT", series_uid, "Expected one non-empty PatientID")
    if len(study_uids) != 1:
        _add_issue(issues, "BLOCKING", "STUDY_UID_CONFLICT", series_uid, "Expected one non-empty StudyInstanceUID")
    if any(not value for value in sop_uids):
        _add_issue(issues, "BLOCKING", "SOP_UID_MISSING", series_uid, "At least one SOPInstanceUID is missing")
    elif len(set(sop_uids)) != len(sop_uids):
        _add_issue(issues, "BLOCKING", "DUPLICATE_SOP_UID", series_uid, "Duplicate SOPInstanceUID within CT series")
    instance_numbers: list[int] = []
    missing_instance = False
    invalid_instance = False
    for record in records:
        if record["instance_number"] is None:
            missing_instance = True
            continue
        try:
            number = int(record["instance_number"])
            if str(number) != str(record["instance_number"]).strip() or number <= 0:
                invalid_instance = True
            else:
                instance_numbers.append(number)
        except ValueError:
            invalid_instance = True
    if missing_instance:
        _add_issue(issues, "WARNING", "INSTANCE_NUMBER_MISSING", series_uid, "InstanceNumber is audited but not used for sorting")
    if invalid_instance:
        _add_issue(issues, "WARNING", "INSTANCE_NUMBER_INVALID", series_uid, "InstanceNumber is invalid and is not used for sorting")
    if len(set(instance_numbers)) != len(instance_numbers):
        _add_issue(issues, "WARNING", "INSTANCE_NUMBER_DUPLICATE", series_uid, "Duplicate InstanceNumber is audited but not used for sorting")

    positions = [record["position"] for record in records]
    orientations = [record["orientation"] for record in records]
    pixel_spacings = [record["pixel_spacing"] for record in records]
    if any(value is None for value in positions):
        _add_issue(issues, "BLOCKING", "IMAGE_POSITION_INVALID", series_uid, "ImagePositionPatient must contain three finite values")
    if any(value is None for value in orientations):
        _add_issue(issues, "BLOCKING", "IMAGE_ORIENTATION_INVALID", series_uid, "ImageOrientationPatient must contain six finite values")
    if any(value is None or any(item <= 0 for item in value) for value in pixel_spacings):
        _add_issue(issues, "BLOCKING", "PIXEL_SPACING_INVALID", series_uid, "PixelSpacing must contain two positive finite values")

    projections: list[tuple[float, str, dict[str, Any]]] = []
    reference = next((orientation for orientation in orientations if orientation is not None), None)
    if reference is not None:
        row, column = reference[:3], reference[3:]
        normal = _unit_cross(row, column)
        if normal is None:
            _add_issue(issues, "BLOCKING", "IMAGE_ORIENTATION_DEGENERATE", series_uid, "Direction cosine cross-product is zero")
        else:
            for record, orientation, position in zip(records, orientations, positions, strict=True):
                if orientation is not None and any(not _near(left, right, ORIENTATION_TOLERANCE) for left, right in zip(reference, orientation, strict=True)):
                    _add_issue(issues, "BLOCKING", "ORIENTATION_INCONSISTENT", series_uid, "ImageOrientationPatient differs from reference")
                    break
            for record, position in zip(records, positions, strict=True):
                if position is None:
                    continue
                projection = sum(left * right for left, right in zip(position, normal, strict=True))
                projections.append((projection, record["sop_uid"] or "", record))
    projections.sort(key=lambda item: (item[0], item[1], item[2]["path_key"]))
    spacings: list[float] = []
    if projections:
        for left, right in zip(projections, projections[1:], strict=False):
            delta = right[0] - left[0]
            if delta <= DUPLICATE_PROJECTION_TOLERANCE_MM:
                _add_issue(issues, "BLOCKING", "DUPLICATE_SLICE_PLANE", series_uid, "Two slices share the same spatial projection")
            else:
                spacings.append(delta)
        if spacings:
            median_spacing = statistics.median(spacings)
            tolerance = max(SPACING_ABSOLUTE_TOLERANCE_MM, median_spacing * SPACING_RELATIVE_TOLERANCE)
            if any(abs(value - median_spacing) > tolerance for value in spacings):
                _add_issue(issues, "WARNING", "SPACING_NONUNIFORM", series_uid, "Adjacent spatial spacings vary beyond the fixed tolerance")
            if any(value > GAP_MULTIPLIER * median_spacing + DUPLICATE_PROJECTION_TOLERANCE_MM for value in spacings):
                _add_issue(issues, "WARNING", "SUSPECTED_SLICE_GAP", series_uid, "Adjacent spatial projection gap exceeds fixed threshold")
    else:
        median_spacing = None

    thicknesses: list[float] = []
    for record in records:
        try:
            thickness = float(record["slice_thickness"]) if record["slice_thickness"] is not None else None
        except ValueError:
            thickness = None
        if thickness is None or not math.isfinite(thickness) or thickness <= 0:
            _add_issue(issues, "BLOCKING", "SLICE_THICKNESS_INVALID", series_uid, "SliceThickness must be positive and finite")
        else:
            thicknesses.append(thickness)
    if thicknesses and max(thicknesses) - min(thicknesses) > 1e-3:
        _add_issue(issues, "WARNING", "SLICE_THICKNESS_INCONSISTENT", series_uid, "SliceThickness varies within CT series")

    series_issues = issues[before:]
    blocking = any(issue.severity == "BLOCKING" for issue in series_issues)
    sorted_rows = [
        {
            "series_key": _key("series", series_uid),
            "sop_key": _key("sop", record["sop_uid"]),
            "projection_mm": projection,
            "instance_number": record["instance_number"],
        }
        for projection, _, record in projections
    ]
    return (
        {
            "patient_key": _key("patient", next(iter(patient_ids), None)),
            "study_key": _key("study", next(iter(study_uids), None)),
            "series_key": _key("series", series_uid),
            "ct_instance_count": len(records),
            "canonical_xml_count": xml_count,
            "mapping_status": "MAPPED" if xml_count else "UNMAPPED",
            "volume_status": "BLOCKING_EXCEPTION" if blocking else "DETERMINISTIC",
            "median_spatial_spacing_mm": "" if median_spacing is None else f"{median_spacing:.6f}",
            "issue_codes": ";".join(sorted({issue.code for issue in series_issues})),
        },
        sorted_rows,
    )


def run_audit(paths: AuditPaths, retain_instance_detail: bool = False) -> dict[str, Any]:
    """Run P1 audit and write deterministic, de-identified output artifacts."""
    issues: list[Issue] = []
    xml_records, xml_roots, xml_fingerprint = _canonical_xml_records(paths, issues)
    embedded_xml_roots, embedded_xml_count = _download_tree_xml_inventory(paths, issues)
    dicom_records, modalities, dicom_fingerprint = _dicom_records(paths, issues)
    ct_by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dicom_records:
        if record["modality"] != "CT":
            continue
        if record["series_uid"] is None:
            issues.append(Issue("BLOCKING", "CT_SERIES_UID_MISSING", "dicom_file", record["path_key"], "CT file lacks SeriesInstanceUID"))
            continue
        ct_by_series[record["series_uid"]].append(record)
    xml_by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    canonical_ct_xml = [record for record in xml_records if record["root_type"] == "LidcReadMessage"]
    duplicate_contents = Counter(record["content_key"] for record in canonical_ct_xml)
    for record in canonical_ct_xml:
        if duplicate_contents[record["content_key"]] > 1:
            issues.append(Issue("BLOCKING", "DUPLICATE_CANONICAL_XML_CONTENT", "canonical_xml", record["xml_key"], "Canonical XML content is duplicated across source files"))
    for record in canonical_ct_xml:
        if record["series_uid"]:
            xml_by_series[record["series_uid"]].append(record)
            if record["series_uid"] not in ct_by_series:
                issues.append(Issue("BLOCKING", "XML_CT_SERIES_UNMATCHED", "canonical_xml", record["xml_key"], "Canonical CT XML has no matching CT series"))
            elif record["study_uid"] not in {item["study_uid"] for item in ct_by_series[record["series_uid"]]}:
                issues.append(Issue("BLOCKING", "XML_DICOM_STUDY_UID_CONFLICT", "canonical_xml", record["xml_key"], "Canonical XML StudyInstanceUID disagrees with mapped CT series"))
    for series_uid in ct_by_series:
        if not xml_by_series[series_uid]:
            _add_issue(issues, "BLOCKING", "CT_SERIES_WITHOUT_CANONICAL_XML", series_uid, "CT series has no canonical CT XML")

    series_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    for series_uid in sorted(ct_by_series):
        row, rows = _series_row(series_uid, ct_by_series[series_uid], len(xml_by_series[series_uid]), issues)
        series_rows.append(row)
        instance_rows.extend(rows)

    issues.sort(key=lambda item: (item.severity, item.code, item.entity_type, item.entity_key, item.detail))
    series_rows.sort(key=lambda row: row["series_key"])
    counts = {
        "patient_directories": sum(1 for path in paths.dicom_root.iterdir() if path.is_dir()),
        "canonical_xml_files": len(xml_records),
        "canonical_xml_roots": dict(sorted(xml_roots.items())),
        "canonical_ct_xml": len(canonical_ct_xml),
        "canonical_cxr_xml": xml_roots["IdriReadMessage"],
        "download_tree_xml_files": embedded_xml_count,
        "download_tree_xml_roots": dict(sorted(embedded_xml_roots.items())),
        "dicom_files_parsed": len(dicom_records),
        "dicom_modalities": dict(sorted(modalities.items())),
        "ct_series": len(ct_by_series),
        "ct_series_mapped_to_canonical_xml": sum(bool(xml_by_series[uid]) for uid in ct_by_series),
        "canonical_ct_xml_mapped_to_ct_series": sum(record["series_uid"] in ct_by_series for record in canonical_ct_xml if record["series_uid"]),
    }
    observed_inventory = {
        "patient_directories": counts["patient_directories"],
        "ct_series": counts["ct_series"],
        "ct_instances": modalities["CT"],
        "dx_instances": modalities["DX"],
        "cr_instances": modalities["CR"],
        "canonical_xml_files": counts["canonical_xml_files"],
    }
    summary = {
        "audit": "P1 DICOM/XML audit",
        "program_version": __version__,
        "input_fingerprints": {
            "canonical_xml_tree_sha256": xml_fingerprint,
            "dicom_header_inventory_sha256": dicom_fingerprint,
        },
        "counts": counts,
        "reference_reconciliation": {
            name: {
                "reference": reference,
                "observed": observed_inventory[name],
                "difference": observed_inventory[name] - reference,
                "hard_gate": False,
            }
            for name, reference in REFERENCE_INVENTORY.items()
        },
        "issues": {
            "total": len(issues),
            "by_severity": dict(sorted(Counter(issue.severity for issue in issues).items())),
            "by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
        },
        "geometry_rules": {
            "orientation_tolerance": ORIENTATION_TOLERANCE,
            "duplicate_projection_tolerance_mm": DUPLICATE_PROJECTION_TOLERANCE_MM,
            "spacing_absolute_tolerance_mm": SPACING_ABSOLUTE_TOLERANCE_MM,
            "spacing_relative_tolerance": SPACING_RELATIVE_TOLERANCE,
            "gap_multiplier": GAP_MULTIPLIER,
            "sort_rule": "dot(ImagePositionPatient, normalized(cross(row_cosine,column_cosine))), then SOP UID",
        },
        "privacy": "Tracked reports use SHA-256-derived keys and omit raw identifiers and absolute paths.",
    }
    write_json(paths.output / "summary.json", summary)
    _write_csv(
        paths.output / "series_audit.csv",
        series_rows,
        ["patient_key", "study_key", "series_key", "ct_instance_count", "canonical_xml_count", "mapping_status", "volume_status", "median_spatial_spacing_mm", "issue_codes"],
    )
    _write_csv(paths.output / "anomalies.csv", [issue.as_row() for issue in issues], ["severity", "code", "entity_type", "entity_key", "detail"])
    if retain_instance_detail:
        local = paths.output / "local"
        local.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(instance_rows).sort_values(["series_key", "projection_mm", "sop_key"], kind="stable").to_parquet(local / "instances.parquet", index=False)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True, help="LIDC raw-data root")
    parser.add_argument("--output", required=True, help="P1 audit output directory")
    parser.add_argument("--canonical-xml-dir", help="Explicit canonical XML root override")
    parser.add_argument("--dicom-root", help="Explicit DICOM root override")
    parser.add_argument("--retain-instance-detail", action="store_true", help="Write ignored local/instances.parquet")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the P1 audit command."""
    arguments = _parser().parse_args(argv)
    paths = resolve_paths(arguments.raw_data, arguments.output, arguments.canonical_xml_dir, arguments.dicom_root)
    summary = run_audit(paths, retain_instance_detail=arguments.retain_instance_detail)
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
