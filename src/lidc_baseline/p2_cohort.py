"""Build the Phase 2 physical-nodule cohort from canonical XML and pylidc."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as element_tree
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from lidc_baseline import __version__
from lidc_baseline.audit import write_json
from lidc_baseline.p1_audit import _key, _local_name, _write_csv, resolve_paths


CONTINUOUS_CONCEPTS = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
CATEGORICAL_CONCEPTS = {"internalStructure": (1, 2, 3, 4), "calcification": (1, 2, 3, 4, 5, 6)}
TARGETS = ("malignancy", *CONTINUOUS_CONCEPTS, *CATEGORICAL_CONCEPTS)
RATING_RANGES = {"malignancy": (1, 5), **{name: (1, 5) for name in CONTINUOUS_CONCEPTS}}
CLUSTER_PARAMETERS = {"metric": "min", "tol": None, "factor": 0.9, "min_tol": 0.1, "verbose": False}
PRIVATE_MANIFEST_REQUIRED_COLUMNS = (
    "nodule_uid",
    "patient_id",
    "study_instance_uid",
    "series_instance_uid",
    "canonical_xml_sha256",
    "annotation_class",
    "annotation_source_fingerprints",
    "source_dicom_sop_fingerprints",
    "reader_count",
    "has_at_least_3_readers",
    "missing_required_target_fields",
    "clustering_initial_tolerance_mm",
    "clustering_effective_tolerance_mm",
    "cohort_status",
    "computed_axial_diameter_mm_max",
    "computed_strict_gt_3mm",
    *(f"{target}_valid_reader_count" for target in TARGETS),
    *(f"{target}_raw_ratings" for target in TARGETS),
    *(f"{target}_target" for target in CONTINUOUS_CONCEPTS),
    *(f"{target}_vote_distribution" for target in CATEGORICAL_CONCEPTS),
    *(f"{target}_modal_tie" for target in CATEGORICAL_CONCEPTS),
)


@dataclass(frozen=True)
class SourceAnnotation:
    """One canonical XML unblinded reader annotation and its stable provenance."""

    patient_id: str
    study_uid: str
    series_uid: str
    xml_sha256: str
    xml_relative_path: str
    session_index: int
    nodule_id: str
    annotation_class: str
    characteristics: dict[str, int | None]
    geometry_signature: str
    matching_geometry_signature: str
    sop_fingerprint: str
    has_required_spatial_source: bool
    source_fingerprint: str


@dataclass(frozen=True)
class CanonicalDocument:
    """The representative canonical XML document for one CT series."""

    patient_id: str
    study_uid: str
    series_uid: str
    xml_sha256: str
    relative_path: str
    annotations: tuple[SourceAnnotation, ...]
    class_counts: dict[str, int]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def enable_pylidc_numpy_compatibility(np_module: Any = np) -> bool:
    """Restore only the removed alias required by pylidc 0.2.3 at runtime."""
    if hasattr(np_module, "int"):
        return False
    np_module.int = int
    return True


def cluster_annotations(scan: Any) -> list[list[Any]]:
    """Invoke the pre-registered pylidc default clustering parameters."""
    return scan.cluster_annotations(**CLUSTER_PARAMETERS)


def _cluster_id_sets(clusters: list[list[Any]]) -> list[tuple[int, ...]]:
    return sorted(tuple(sorted(int(annotation.id) for annotation in cluster)) for cluster in clusters)


def cluster_annotations_with_effective_tolerance(scan: Any) -> tuple[list[list[Any]], float]:
    """Run pylidc defaults and capture the tolerance that produced its clusters."""
    initial_tolerance = float(scan.slice_thickness) if CLUSTER_PARAMETERS["tol"] is None else float(CLUSTER_PARAMETERS["tol"])
    if len(scan.annotations) < 2:
        return cluster_annotations(scan), initial_tolerance
    parameters = {**CLUSTER_PARAMETERS, "return_distance_matrix": True}
    clusters, distance_matrix = scan.cluster_annotations(**parameters)
    effective_tolerance = initial_tolerance
    _, component_ids = connected_components(distance_matrix <= effective_tolerance, directed=False)
    component_counts = [(component_ids == component_id).sum() for component_id in np.unique(component_ids)]
    while any(count > 4 for count in component_counts):
        candidate_tolerance = effective_tolerance * float(CLUSTER_PARAMETERS["factor"])
        if candidate_tolerance < float(CLUSTER_PARAMETERS["min_tol"]):
            break
        effective_tolerance = candidate_tolerance
        _, component_ids = connected_components(distance_matrix <= effective_tolerance, directed=False)
        component_counts = [(component_ids == component_id).sum() for component_id in np.unique(component_ids)]
    expected_clusters = [
        tuple(sorted(int(scan.annotations[index].id) for index, component_id in enumerate(component_ids) if component_id == current_id))
        for current_id in np.unique(component_ids)
    ]
    if sorted(expected_clusters) != _cluster_id_sets(clusters):
        raise RuntimeError("PYLIDC_EFFECTIVE_TOLERANCE_RECONSTRUCTION_FAILED")
    return clusters, effective_tolerance


def _text(element: element_tree.Element, name: str) -> str | None:
    for child in element.iter():
        if _local_name(child.tag) == name and (child.text or "").strip():
            return (child.text or "").strip()
    return None


def _integer(value: str | None, allowed: Iterable[int]) -> int | None:
    try:
        numeric = int(value) if value is not None else None
    except ValueError:
        return None
    return numeric if numeric in allowed else None


def _float_text(value: str | None) -> float:
    try:
        numeric = float(value) if value is not None else float("nan")
    except ValueError:
        numeric = float("nan")
    return numeric


def _source_geometry_signatures(nodule: element_tree.Element) -> tuple[str, str, str, bool]:
    rois = []
    matching_rois = []
    all_rois_complete = True
    for roi in (item for item in nodule.iter() if _local_name(item.tag) == "roi"):
        z = _float_text(_text(roi, "imageZposition"))
        inclusion = (_text(roi, "inclusion") or "").upper() == "TRUE"
        points = []
        for edge in (item for item in roi if _local_name(item.tag) == "edgeMap"):
            x, y = _integer(_text(edge, "xCoord"), range(-1000000, 1000001)), _integer(_text(edge, "yCoord"), range(-1000000, 1000001))
            if x is not None and y is not None:
                points.append((x, y))
        sop_uid = _text(roi, "imageSOP_UID")
        if not math.isfinite(z) or not sop_uid or not points:
            all_rois_complete = False
        matching_rois.append((None if not math.isfinite(z) else round(z, 4), inclusion, sorted(points)))
        rois.append((None if not math.isfinite(z) else round(z, 4), sop_uid, inclusion, sorted(points)))
    matching_payload = sorted(matching_rois, key=repr)
    provenance_payload = sorted(rois, key=repr)
    sop_fingerprint = _sha256(_canonical_json(sorted(sop_uid for _, sop_uid, _, _ in rois if sop_uid)))
    return _sha256(_canonical_json(provenance_payload)), _sha256(_canonical_json(matching_payload)), sop_fingerprint, bool(rois) and all_rois_complete


def _pylidc_geometry_signature(annotation: Any) -> str:
    rois = []
    for contour in annotation.contours:
        points = []
        for line in str(contour.coords).splitlines():
            left, right = line.split(",")
            points.append((int(left), int(right)))
        rois.append((round(float(contour.image_z_position), 4), bool(contour.inclusion), sorted(points)))
    return _sha256(_canonical_json(sorted(rois, key=repr)))


def _characteristics(nodule: element_tree.Element) -> dict[str, int | None]:
    characteristic = next((item for item in nodule if _local_name(item.tag) == "characteristics"), None)
    if characteristic is None:
        return {target: None for target in TARGETS}
    values: dict[str, int | None] = {}
    for target in TARGETS:
        if target in CATEGORICAL_CONCEPTS:
            values[target] = _integer(_text(characteristic, target), CATEGORICAL_CONCEPTS[target])
        else:
            values[target] = _integer(_text(characteristic, target), range(RATING_RANGES[target][0], RATING_RANGES[target][1] + 1))
    return values


def _session_payload(session: element_tree.Element) -> list[Any]:
    """Canonicalize one reading session to identify duplicated canonical XML files."""
    def visit(element: element_tree.Element) -> Any:
        return [_local_name(element.tag), (element.text or "").strip(), sorted((key.rsplit("}", 1)[-1], value) for key, value in element.attrib.items()), [visit(child) for child in element]]

    return visit(session)


def parse_canonical_document(path: Path, root: Path) -> CanonicalDocument:
    """Parse one canonical CT XML source without relying on pylidc SQL identifiers."""
    payload = path.read_bytes()
    xml = element_tree.fromstring(payload)
    if _local_name(xml.tag) != "LidcReadMessage":
        raise ValueError("Canonical source is not a LidcReadMessage")
    response = next((item for item in xml if _local_name(item.tag) == "ResponseHeader"), None)
    if response is None:
        raise ValueError("Canonical source has no ResponseHeader")
    study_uid, series_uid = _text(response, "StudyInstanceUID"), _text(response, "SeriesInstanceUid")
    if not study_uid or not series_uid:
        raise ValueError("Canonical source has missing study or series UID")
    patient_id = path.parent.name  # Replaced by pylidc scan patient ID before emitting a manifest.
    xml_sha = _sha256(payload)
    relative_path = path.relative_to(root).as_posix()
    annotations: list[SourceAnnotation] = []
    class_counts: Counter[str] = Counter()
    for session_index, session in enumerate(item for item in xml.iter() if _local_name(item.tag) == "readingSession"):
        for nodule in session:
            node_class = _local_name(nodule.tag)
            if node_class == "unblindedReadNodule":
                class_counts["nodule >=3 mm"] += 1
            elif node_class == "smallNodule":
                class_counts["nodule <3 mm"] += 1
                continue
            elif node_class == "nonNodule":
                class_counts["non-nodule"] += 1
                continue
            else:
                continue
            nodule_id = _text(nodule, "noduleID")
            if not nodule_id:
                raise ValueError("Unblinded reader annotation has no noduleID")
            values = _characteristics(nodule)
            geometry_signature, matching_geometry_signature, sop_fingerprint, has_required_spatial_source = _source_geometry_signatures(nodule)
            fingerprint_payload = {
                "xml_sha256": xml_sha,
                "session_index": session_index,
                "nodule_id": nodule_id,
                "characteristics": values,
                "geometry_signature": geometry_signature,
                "sop_fingerprint": sop_fingerprint,
            }
            annotations.append(
                SourceAnnotation(
                    patient_id=patient_id,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    xml_sha256=xml_sha,
                    xml_relative_path=relative_path,
                    session_index=session_index,
                    nodule_id=nodule_id,
                    annotation_class="nodule >=3 mm",
                    characteristics=values,
                    geometry_signature=geometry_signature,
                    matching_geometry_signature=matching_geometry_signature,
                    sop_fingerprint=sop_fingerprint,
                    has_required_spatial_source=has_required_spatial_source,
                    source_fingerprint=_sha256(_canonical_json(fingerprint_payload)),
                )
            )
    return CanonicalDocument(patient_id, study_uid, series_uid, xml_sha, relative_path, tuple(annotations), dict(class_counts))


def _document_session_fingerprint(path: Path) -> str:
    root = element_tree.fromstring(path.read_bytes())
    payload = [_session_payload(session) for session in root.iter() if _local_name(session.tag) == "readingSession"]
    return _sha256(_canonical_json(payload))


def _canonical_ct_series_uid(path: Path) -> str | None:
    """Read only the response header needed to index a canonical XML file."""
    root_type: str | None = None
    try:
        for event, element in element_tree.iterparse(path, events=("start", "end")):
            if event == "start" and root_type is None:
                root_type = _local_name(element.tag)
                if root_type != "LidcReadMessage":
                    return None
            if event == "end" and _local_name(element.tag) == "ResponseHeader":
                return _text(element, "SeriesInstanceUid")
    except element_tree.ParseError:
        return None
    return None


def index_canonical_document_paths(canonical_root: Path) -> tuple[dict[str, Path], list[dict[str, str]], int]:
    """Index source paths without retaining all XML contour structures in memory."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    issues: list[dict[str, str]] = []
    for path in sorted(canonical_root.rglob("*.xml")):
        series_uid = _canonical_ct_series_uid(path)
        if series_uid is None:
            continue
        grouped[series_uid].append(path)
    selected: dict[str, Path] = {}
    redundant = 0
    for series_uid, choices in grouped.items():
        session_fingerprints = {_document_session_fingerprint(path) for path in choices} if len(choices) > 1 else {"single"}
        if len(session_fingerprints) != 1:
            issues.append({"series_key": _key("series", series_uid), "reason": "CANONICAL_XML_SERIES_CONTENT_CONFLICT"})
            continue
        selected[series_uid] = min(choices, key=lambda path: path.relative_to(canonical_root).as_posix())
        redundant += len(choices) - 1
    return selected, issues, redundant


def load_canonical_documents(canonical_root: Path) -> tuple[dict[str, CanonicalDocument], list[dict[str, str]], int]:
    """Choose one deterministic source when canonical XML copies have identical readings."""
    grouped: dict[str, list[tuple[CanonicalDocument, Path]]] = defaultdict(list)
    issues: list[dict[str, str]] = []
    for path in sorted(canonical_root.rglob("*.xml")):
        try:
            document = parse_canonical_document(path, canonical_root)
        except (element_tree.ParseError, ValueError) as error:
            if str(error) == "Canonical source is not a LidcReadMessage":
                continue
            issues.append({"series_key": _key("canonical-xml-path", path.relative_to(canonical_root).as_posix()), "reason": f"CANONICAL_XML_PARSE_OR_SCHEMA_ERROR:{type(error).__name__}"})
            continue
        grouped[document.series_uid].append((document, path))
    selected: dict[str, CanonicalDocument] = {}
    redundant = 0
    for series_uid, choices in grouped.items():
        session_fingerprints = {_document_session_fingerprint(item[1]) for item in choices} if len(choices) > 1 else {"single"}
        if len(session_fingerprints) != 1:
            issues.append({"series_key": _key("series", series_uid), "reason": "CANONICAL_XML_SERIES_CONTENT_CONFLICT"})
            continue
        winner = min((item[0] for item in choices), key=lambda item: item.relative_path)
        selected[series_uid] = winner
        redundant += len(choices) - 1
    return selected, issues, redundant


def match_source_annotation(pylidc_annotation: Any, candidates: Iterable[SourceAnnotation]) -> SourceAnnotation:
    """Require a unique XML source match; use contours to resolve repeated nodule IDs."""
    nodule_id = str(pylidc_annotation._nodule_id)
    same_id = [candidate for candidate in candidates if candidate.nodule_id == nodule_id]
    if len(same_id) == 1:
        return same_id[0]
    geometry = _pylidc_geometry_signature(pylidc_annotation)
    exact = [candidate for candidate in same_id if candidate.matching_geometry_signature == geometry]
    if len(exact) != 1:
        raise ValueError(f"SOURCE_ANNOTATION_MATCH_{'AMBIGUOUS' if len(exact) > 1 else 'MISSING'}")
    return exact[0]


def split_spatially_usable_sources(sources: Iterable[SourceAnnotation]) -> tuple[list[SourceAnnotation], list[SourceAnnotation]]:
    """Separate source annotations that can be traced through XML ROI/SOP/contours."""
    usable, excluded = [], []
    for source in sources:
        (usable if source.has_required_spatial_source else excluded).append(source)
    return usable, excluded


def stable_nodule_uid(patient_id: str, study_uid: str, series_uid: str, xml_sha256: str, annotation_fingerprints: Iterable[str]) -> str:
    """Derive the canonical physical-nodule ID exclusively from stable source data."""
    payload = {
        "schema": "baseline-v1-source-nodule-uid",
        "patient_id": patient_id,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "canonical_xml_sha256": xml_sha256,
        "sorted_annotation_fingerprints": sorted(annotation_fingerprints),
    }
    return _sha256(_canonical_json(payload))


def _normalize(value: int, target: str) -> float:
    lower, upper = RATING_RANGES[target]
    return (value - lower) / (upper - lower)


def aggregate_cluster(sources: list[SourceAnnotation], diameters_mm: list[float]) -> dict[str, Any]:
    """Aggregate all permitted Baseline-v1 reader targets for one physical cluster."""
    output: dict[str, Any] = {
        "reader_count": len(sources),
        "has_at_least_3_readers": len(sources) >= 3,
        "annotation_fingerprints": sorted(source.source_fingerprint for source in sources),
        "source_nodule_identifiers": sorted(source.nodule_id for source in sources),
        "source_dicom_sop_fingerprints": sorted(source.sop_fingerprint for source in sources),
    }
    all_targets_valid = True
    missing_required_target_fields = []
    for target in TARGETS:
        ratings = [source.characteristics[target] for source in sources if source.characteristics[target] is not None]
        output[f"{target}_raw_ratings"] = ratings
        output[f"{target}_valid_reader_count"] = len(ratings)
        if not ratings:
            all_targets_valid = False
            missing_required_target_fields.append(target)
        if target in CONTINUOUS_CONCEPTS:
            normalized = [_normalize(value, target) for value in ratings]
            output[f"{target}_normalized_ratings"] = normalized
            output[f"{target}_target"] = statistics.fmean(normalized) if normalized else None
            output[f"{target}_reader_std"] = statistics.pstdev(normalized) if len(normalized) > 1 else 0.0 if normalized else None
        elif target == "malignancy":
            mean = statistics.fmean(ratings) if ratings else None
            output["mean_malignancy"] = mean
            output["malignancy_reader_std"] = statistics.pstdev(ratings) if len(ratings) > 1 else 0.0 if ratings else None
            output["malignancy_label"] = 0 if mean is not None and mean <= 2 else 1 if mean is not None and mean >= 4 else None
            output["malignancy_status"] = "BENIGN" if output["malignancy_label"] == 0 else "MALIGNANT" if output["malignancy_label"] == 1 else "UNCERTAIN" if mean is not None else "MISSING"
        else:
            classes = CATEGORICAL_CONCEPTS[target]
            if ratings:
                distribution = [ratings.count(value) / len(ratings) for value in classes]
                modes = [value for value, probability in zip(classes, distribution, strict=True) if probability == max(distribution)]
                output[f"{target}_vote_distribution"] = distribution
                output[f"{target}_modal_tie"] = len(modes) > 1
                output[f"{target}_modal_class"] = modes[0] if len(modes) == 1 else None
            else:
                output[f"{target}_vote_distribution"] = None
                output[f"{target}_modal_tie"] = None
                output[f"{target}_modal_class"] = None
    finite_diameters = [float(value) for value in diameters_mm if math.isfinite(float(value)) and float(value) >= 0]
    output["computed_axial_diameter_mm_max"] = max(finite_diameters) if finite_diameters else None
    output["computed_strict_gt_3mm"] = bool(output["computed_axial_diameter_mm_max"] is not None and output["computed_axial_diameter_mm_max"] > 3.0)
    output["all_required_targets_valid"] = all_targets_valid
    output["missing_required_target_fields"] = missing_required_target_fields
    return output


def _private_manifest_row(scan: Any, sources: list[SourceAnnotation], cluster: list[Any], effective_tolerance_mm: float) -> dict[str, Any]:
    aggregate = aggregate_cluster(sources, [float(annotation.diameter) for annotation in cluster])
    uid = stable_nodule_uid(scan.patient_id, scan.study_instance_uid, scan.series_instance_uid, sources[0].xml_sha256, aggregate["annotation_fingerprints"])
    row: dict[str, Any] = {
        "nodule_uid": uid,
        "patient_id": scan.patient_id,
        "study_instance_uid": scan.study_instance_uid,
        "series_instance_uid": scan.series_instance_uid,
        "scan_sql_id": int(scan.id),
        "pylidc_annotation_sql_ids": _canonical_json(sorted(int(annotation.id) for annotation in cluster)),
        "canonical_xml_sha256": sources[0].xml_sha256,
        "xml_relative_path": sources[0].xml_relative_path,
        "annotation_source_fingerprints": _canonical_json(aggregate["annotation_fingerprints"]),
        "source_nodule_identifiers": _canonical_json(aggregate["source_nodule_identifiers"]),
        "annotation_class": "nodule >=3 mm",
        "source_dicom_sop_fingerprints": _canonical_json(aggregate["source_dicom_sop_fingerprints"]),
        "clustering_initial_tolerance_mm": float(scan.slice_thickness) if CLUSTER_PARAMETERS["tol"] is None else float(CLUSTER_PARAMETERS["tol"]),
        "clustering_effective_tolerance_mm": effective_tolerance_mm,
        **aggregate,
    }
    row["cohort_status"] = "EXCLUDED_MISSING_REQUIRED_TARGET" if not aggregate["all_required_targets_valid"] else "PRIMARY_BINARY" if aggregate["malignancy_label"] is not None else "EXCLUDED_UNCERTAIN_MALIGNANCY"
    return row


def annotation_to_cluster_mapping_rows(nodule_uid: str, scan: Any, series_uid: str, sources: list[SourceAnnotation], cluster: list[Any]) -> list[dict[str, Any]]:
    """Create deterministically sorted, local-only reader-annotation mapping records."""
    if len(sources) != len(cluster):
        raise ValueError("SOURCE_CLUSTER_CARDINALITY_MISMATCH")
    rows = [
        {
            "nodule_uid": nodule_uid,
            "patient_id": scan.patient_id,
            "series_instance_uid": series_uid,
            "source_annotation_fingerprint": source.source_fingerprint,
            "source_nodule_identifier": source.nodule_id,
            "reading_session_index": source.session_index,
            "pylidc_annotation_sql_id": int(annotation.id),
        }
        for source, annotation in zip(sources, cluster, strict=True)
    ]
    return sorted(rows, key=lambda row: (row["source_annotation_fingerprint"], row["pylidc_annotation_sql_id"]))


def _load_excluded_series(p1_audit: Path) -> set[str]:
    payload = json.loads((p1_audit / "duplicate_plane_resolution.json").read_text(encoding="utf-8"))
    return set(payload["excluded_series_keys"])


def cluster_has_supported_reader_count(cluster: list[Any]) -> bool:
    """Accept the LIDC reader-count range supported by Baseline-v1."""
    return 1 <= len(cluster) <= 4


def validate_private_manifest(frame: pd.DataFrame) -> None:
    """Reject incomplete or invalid local manifest rows before persistence."""
    missing = sorted(set(PRIVATE_MANIFEST_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"PRIVATE_MANIFEST_REQUIRED_COLUMNS_MISSING:{','.join(missing)}")
    if frame["nodule_uid"].isna().any() or frame["nodule_uid"].duplicated().any():
        raise ValueError("PRIVATE_MANIFEST_NODULE_UID_NOT_UNIQUE")
    if not frame["annotation_class"].eq("nodule >=3 mm").all():
        raise ValueError("PRIVATE_MANIFEST_INVALID_ANNOTATION_CLASS")
    for target in CONTINUOUS_CONCEPTS:
        for value in frame[f"{target}_target"].dropna():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"PRIVATE_MANIFEST_CONTINUOUS_TARGET_OUT_OF_RANGE:{target}")
    for target, classes in CATEGORICAL_CONCEPTS.items():
        for distribution in frame[f"{target}_vote_distribution"].dropna():
            if len(distribution) != len(classes) or not math.isclose(sum(float(value) for value in distribution), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"PRIVATE_MANIFEST_INVALID_CATEGORICAL_DISTRIBUTION:{target}")


def _write_private_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("nodule_uid")
    validate_private_manifest(frame)
    frame.to_parquet(path, index=False)


def run_cohort(raw_data: str | Path, p1_audit: str | Path, manifest: str | Path, annotation_mapping: str | Path, audit_output: str | Path) -> dict[str, Any]:
    """Create a local private cohort manifest and deidentified P2 audit reports."""
    paths = resolve_paths(raw_data, audit_output)
    p1_output = Path(p1_audit).expanduser().resolve()
    excluded_keys = _load_excluded_series(p1_output)
    compatibility_enabled = enable_pylidc_numpy_compatibility()
    import pylidc as pl

    available_series = {series_uid for (series_uid,) in pl.query(pl.Scan.series_instance_uid).all()}
    document_paths, xml_issues, redundant_xml = index_canonical_document_paths(paths.canonical_xml)
    rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, str]] = list(xml_issues)
    tolerance_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reader_counts: Counter[str] = Counter()
    for series_uid in sorted(document_paths):
        document = parse_canonical_document(document_paths[series_uid], paths.canonical_xml)
        series_key = _key("series", series_uid)
        counts["source_unblinded_reader_annotations"] += document.class_counts.get("nodule >=3 mm", 0)
        counts["source_small_nodule_annotations"] += document.class_counts.get("nodule <3 mm", 0)
        counts["source_non_nodule_annotations"] += document.class_counts.get("non-nodule", 0)
        if series_key in excluded_keys:
            counts["p1_excluded_series"] += 1
            exclusion_rows.append({"series_key": series_key, "reason": "P1_EXCLUDED_DIFFERENT_DUPLICATE_PLANE"})
            continue
        if series_uid not in available_series:
            counts["missing_pylidc_scan"] += 1
            exclusion_rows.append({"series_key": series_key, "reason": "PYLIDC_SCAN_MISSING"})
            continue
        scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_uid).one()
        usable_sources, spatially_invalid_sources = split_spatially_usable_sources(document.annotations)
        if spatially_invalid_sources:
            counts["source_annotations_missing_spatial_source"] += len(spatially_invalid_sources)
            exclusion_rows.append({"series_key": series_key, "reason": "SOURCE_ANNOTATION_MISSING_SPATIAL_SOURCE", "annotation_count": str(len(spatially_invalid_sources))})
        matched_sources: dict[int, SourceAnnotation] = {}
        for annotation in scan.annotations:
            try:
                matched_sources[int(annotation.id)] = match_source_annotation(annotation, usable_sources)
            except ValueError as error:
                counts["pylidc_annotations_unmatched_source"] += 1
                exclusion_rows.append({"series_key": series_key, "reason": str(error), "annotation_count": "1"})
        source_without_pylidc = [source for source in usable_sources if source.source_fingerprint not in {item.source_fingerprint for item in matched_sources.values()}]
        if source_without_pylidc:
            counts["source_annotations_not_in_pylidc"] += len(source_without_pylidc)
            exclusion_rows.append({"series_key": series_key, "reason": "SOURCE_ANNOTATION_NOT_IN_PYLIDC", "annotation_count": str(len(source_without_pylidc))})
        try:
            clusters, effective_tolerance = cluster_annotations_with_effective_tolerance(scan)
        except Exception as error:
            counts["clustering_error_series"] += 1
            exclusion_rows.append({"series_key": series_key, "reason": f"CLUSTERING_ERROR:{type(error).__name__}"})
            scan._sa_instance_state.session.expunge_all()
            continue
        counts["eligible_series_clustered"] += 1
        tolerance_rows.append(
            {
                "series_key": series_key,
                "initial_tolerance_mm": float(scan.slice_thickness) if CLUSTER_PARAMETERS["tol"] is None else float(CLUSTER_PARAMETERS["tol"]),
                "effective_tolerance_mm": effective_tolerance,
                "pylidc_annotation_count": len(scan.annotations),
                "cluster_count": len(clusters),
            }
        )
        for cluster_index, cluster in enumerate(clusters):
            if not cluster_has_supported_reader_count(cluster):
                counts["clusters_excluded_more_than_4_readers"] += 1
                exclusion_rows.append({"series_key": series_key, "reason": "CLUSTER_MORE_THAN_4_READERS"})
                continue
            try:
                sources = [matched_sources[int(annotation.id)] for annotation in cluster]
            except KeyError:
                counts["clusters_excluded_source_mapping"] += 1
                exclusion_rows.append({"series_key": series_key, "reason": "SOURCE_ANNOTATION_MATCH_MISSING"})
                continue
            if len({source.source_fingerprint for source in sources}) != len(sources):
                counts["clusters_excluded_source_mapping"] += 1
                exclusion_rows.append({"series_key": series_key, "reason": "SOURCE_ANNOTATION_REUSED"})
                continue
            row = _private_manifest_row(scan, sources, cluster, effective_tolerance)
            rows.append(row)
            counts["physical_clusters"] += 1
            reader_counts[str(row["reader_count"])] += 1
            counts[row["cohort_status"]] += 1
            if row["cohort_status"] == "EXCLUDED_MISSING_REQUIRED_TARGET":
                exclusion_rows.append(
                    {
                        "series_key": series_key,
                        "reason": f"MISSING_REQUIRED_TARGET:{','.join(row['missing_required_target_fields'])}",
                        "annotation_count": str(row["reader_count"]),
                    }
                )
            mapping_rows.extend(annotation_to_cluster_mapping_rows(row["nodule_uid"], scan, series_uid, sources, cluster))
        scan._sa_instance_state.session.expunge_all()
    if not rows:
        raise RuntimeError("P2 cohort produced no physical clusters")
    _write_private_parquet(Path(manifest).expanduser().resolve(), rows)
    mapping_path = Path(annotation_mapping).expanduser().resolve()
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mapping_rows).sort_values(["nodule_uid", "source_annotation_fingerprint"]).to_parquet(mapping_path, index=False)
    binary = [row for row in rows if row["cohort_status"] == "PRIMARY_BINARY"]
    binary_at_least_3_readers = [row for row in binary if row["has_at_least_3_readers"]]
    effective_tolerances = [row["effective_tolerance_mm"] for row in tolerance_rows]
    summary = {
        "audit": "P2 physical nodule cohort",
        "program_version": __version__,
        "pylidc": {"version": pl.__version__, "numpy_version": np.__version__, "np_int_compatibility_enabled": compatibility_enabled, "cluster_parameters": CLUSTER_PARAMETERS},
        "privacy": "Tracked P2 reports contain only SHA-256 derived identifiers; the full manifest and annotation mapping are local-only.",
        "clustering_tolerance": {
            "configured_tol": CLUSTER_PARAMETERS["tol"],
            "tol_none_resolution": "scan.slice_thickness",
            "series_metadata_file": "clustering_tolerances.csv",
            "effective_tolerance_min_mm": min(effective_tolerances),
            "effective_tolerance_max_mm": max(effective_tolerances),
        },
        "counts": {**dict(sorted(counts.items())), "physical_clusters_total": len(rows), "binary_primary_cohort": len(binary), "binary_primary_cohort_at_least_3_readers": len(binary_at_least_3_readers), "patients_total": len({row["patient_id"] for row in rows}), "patients_binary": len({row["patient_id"] for row in binary}), "canonical_ct_xml_series": len(document_paths), "canonical_xml_redundant_copies": redundant_xml},
        "reference_reconciliation": {"nodules": 2651, "patients": 875, "hard_gate": False, "observed_binary_nodules": len(binary), "observed_binary_patients": len({row["patient_id"] for row in binary})},
        "reader_count_distribution": dict(sorted(reader_counts.items(), key=lambda item: int(item[0]))),
    }
    write_json(paths.output / "summary.json", summary)
    reconciliation = [
        {"stage": "canonical_ct_xml_series", "count": len(document_paths)},
        {"stage": "p1_eligible_series_clustered", "count": counts["eligible_series_clustered"]},
        {"stage": "source_unblinded_reader_annotations", "count": counts["source_unblinded_reader_annotations"]},
        {"stage": "source_annotations_not_in_pylidc", "count": counts["source_annotations_not_in_pylidc"]},
        {"stage": "physical_clusters", "count": len(rows)},
        {"stage": "primary_binary_cohort", "count": len(binary)},
        {"stage": "primary_binary_cohort_at_least_3_readers", "count": len(binary_at_least_3_readers)},
        {"stage": "uncertain_malignancy", "count": counts["EXCLUDED_UNCERTAIN_MALIGNANCY"]},
        {"stage": "missing_required_target", "count": counts["EXCLUDED_MISSING_REQUIRED_TARGET"]},
    ]
    _write_csv(paths.output / "reconciliation.csv", reconciliation, ["stage", "count"])
    _write_csv(paths.output / "exclusions.csv", sorted(exclusion_rows, key=lambda row: (row["reason"], row["series_key"])), ["series_key", "reason", "annotation_count"])
    _write_csv(paths.output / "clustering_tolerances.csv", sorted(tolerance_rows, key=lambda row: row["series_key"]), ["series_key", "initial_tolerance_mm", "effective_tolerance_mm", "pylidc_annotation_count", "cluster_count"])
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--p1-audit", default="artifacts/audit/p1")
    parser.add_argument("--manifest", default="artifacts/manifests/nodules.parquet")
    parser.add_argument("--annotation-mapping", default="artifacts/manifests/annotation_mapping.parquet")
    parser.add_argument("--audit-output", default="artifacts/audit/p2")
    arguments = parser.parse_args(argv)
    summary = run_cohort(arguments.raw_data, arguments.p1_audit, arguments.manifest, arguments.annotation_mapping, arguments.audit_output)
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
