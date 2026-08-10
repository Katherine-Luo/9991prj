"""Build deterministic Baseline-v2 consensus masks, 3D ROIs, and QA evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as functional

from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p1_audit import (
    DUPLICATE_PROJECTION_TOLERANCE_MM,
    ORIENTATION_TOLERANCE,
    _key,
    _number_sequence,
)


ROI_SHAPE = (64, 64, 64)
HU_MIN, HU_MAX = -1000.0, 700.0
PILOT_SIZE = 41


@dataclass(frozen=True)
class DicomSlice:
    """One CT slice with only geometry and source path information."""

    path: Path
    sop_uid: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float, float, float]
    pixel_spacing: tuple[float, float]
    slope: float
    intercept: float
    projection: float


@dataclass(frozen=True)
class Volume:
    """A D,H,W CT volume and immutable geometry provenance."""

    hu: np.ndarray
    slices: tuple[DicomSlice, ...]
    spacing_dhw: tuple[float, float, float]
    exact_duplicate_applied: bool


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def enable_pylidc_numpy_compatibility(np_module: Any = np) -> tuple[bool, bool]:
    """Restore only removed aliases needed by pylidc, without warning probes."""
    def exists(name: str) -> bool:
        return name in vars(np_module) or any(name in vars(parent) for parent in type(np_module).__mro__)

    int_added = False
    bool_added = False
    if not exists("int"):
        np_module.int = int
        int_added = True
    if not exists("bool"):
        np_module.bool = np_module.bool_
        bool_added = True
    return int_added, bool_added


def _normal(orientation: Sequence[float]) -> np.ndarray:
    row = np.asarray(orientation[:3], dtype=np.float64)
    column = np.asarray(orientation[3:], dtype=np.float64)
    normal = np.cross(row, column)
    magnitude = float(np.linalg.norm(normal))
    if not math.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError("DICOM_INVALID_ORIENTATION")
    return normal / magnitude


def _header_slice(path: Path) -> DicomSlice:
    fields = [
        "SOPInstanceUID", "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing",
        "RescaleSlope", "RescaleIntercept", "Modality",
    ]
    dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=fields)
    if str(getattr(dataset, "Modality", "")) != "CT":
        raise ValueError("DICOM_NON_CT_IN_SERIES")
    sop_uid = str(getattr(dataset, "SOPInstanceUID", "") or "")
    position = _number_sequence(getattr(dataset, "ImagePositionPatient", None), 3)
    orientation = _number_sequence(getattr(dataset, "ImageOrientationPatient", None), 6)
    pixel_spacing = _number_sequence(getattr(dataset, "PixelSpacing", None), 2)
    if not sop_uid or position is None or orientation is None or pixel_spacing is None:
        raise ValueError("DICOM_MISSING_OR_INVALID_GEOMETRY")
    if any(item <= 0 for item in pixel_spacing):
        raise ValueError("DICOM_NON_POSITIVE_PIXEL_SPACING")
    normal = _normal(orientation)
    slope, intercept = float(getattr(dataset, "RescaleSlope", 1.0)), float(getattr(dataset, "RescaleIntercept", 0.0))
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("DICOM_INVALID_RESCALE")
    return DicomSlice(path, sop_uid, position, orientation, pixel_spacing, slope, intercept, float(np.dot(position, normal)))


def sort_dicom_slices(records: Iterable[DicomSlice]) -> list[DicomSlice]:
    """Sort by spatial projection with SOP UID only as deterministic tiebreaker."""
    items = list(records)
    if not items:
        raise ValueError("DICOM_SERIES_EMPTY")
    reference = np.asarray(items[0].orientation, dtype=np.float64)
    for item in items:
        if not np.allclose(item.orientation, reference, rtol=0.0, atol=ORIENTATION_TOLERANCE):
            raise ValueError("DICOM_ORIENTATION_INCONSISTENT")
    return sorted(items, key=lambda item: (item.projection, item.sop_uid))


def _selection_for_series(p1_audit: Path, series_uid: str) -> dict[str, str] | None:
    series_key = _key("series", series_uid)
    path = p1_audit / "derived_volume_selection.csv"
    if not path.exists():
        raise FileNotFoundError(f"P1 derived volume selection is missing: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["series_key"] == series_key]
    if len(rows) > 1:
        raise ValueError("P1_EXACT_DUPLICATE_SELECTION_AMBIGUOUS")
    return rows[0] if rows else None


def apply_duplicate_policy(records: Iterable[DicomSlice], selection: dict[str, str] | None) -> tuple[list[DicomSlice], bool]:
    """Apply only the P1-approved exact-duplicate selection, after pixel equality."""
    ordered = sort_dicom_slices(records)
    if len({item.sop_uid for item in ordered}) != len(ordered):
        raise ValueError("DICOM_DUPLICATE_SOP_UID")
    retained: list[DicomSlice] = []
    applied = False
    index = 0
    while index < len(ordered):
        group = [ordered[index]]
        index += 1
        while index < len(ordered) and abs(ordered[index].projection - group[0].projection) <= DUPLICATE_PROJECTION_TOLERANCE_MM:
            group.append(ordered[index])
            index += 1
        if len(group) == 1:
            retained.extend(group)
            continue
        if selection is None:
            raise ValueError("DICOM_UNAPPROVED_DUPLICATE_SLICE_PLANE")
        expected = selection["retained_sop_key"]
        chosen = [item for item in group if _key("sop", item.sop_uid) == expected]
        discarded = sorted(_key("sop", item.sop_uid) for item in group if item not in chosen)
        expected_discarded = sorted(filter(None, selection["discarded_sop_keys"].split(";")))
        if len(chosen) != 1 or discarded != expected_discarded:
            raise ValueError("P1_EXACT_DUPLICATE_SELECTION_MISMATCH")
        reference = pydicom.dcmread(chosen[0].path).pixel_array
        for item in group:
            if item is chosen[0]:
                continue
            if not np.array_equal(reference, pydicom.dcmread(item.path).pixel_array):
                raise ValueError("P1_EXACT_DUPLICATE_PIXEL_CONTENT_MISMATCH")
        retained.extend(chosen)
        applied = True
    return retained, applied


def _series_files(raw_data: Path, patient_id: str, study_uid: str, series_uid: str) -> list[Path]:
    """Find a CT series by verified DICOM IDs, never by directory/file names."""
    patient_root = raw_data / "manifest-1600709154662" / "LIDC-IDRI" / patient_id
    if not patient_root.is_dir():
        raise FileNotFoundError("DICOM_PATIENT_DIRECTORY_MISSING")
    matched: list[Path] = []
    for candidate in sorted(patient_root.rglob("*.dcm")):
        try:
            header = pydicom.dcmread(candidate, stop_before_pixels=True, specific_tags=["PatientID", "StudyInstanceUID", "SeriesInstanceUID", "Modality"])
        except Exception as error:
            raise ValueError(f"DICOM_PARSE_ERROR:{type(error).__name__}") from error
        if str(getattr(header, "StudyInstanceUID", "")) == study_uid and str(getattr(header, "SeriesInstanceUID", "")) == series_uid:
            if str(getattr(header, "PatientID", "")) != patient_id:
                raise ValueError("DICOM_PATIENT_UID_CONFLICT")
            if str(getattr(header, "Modality", "")) != "CT":
                raise ValueError("DICOM_MATCHED_SERIES_NOT_CT")
            matched.append(candidate)
    if not matched:
        raise FileNotFoundError("DICOM_SERIES_NOT_FOUND_BY_UID")
    return matched


def load_volume(raw_data: Path, row: dict[str, Any], p1_audit: Path) -> Volume:
    """Load one eligible series using P1 geometry rules and HU conversion."""
    files = _series_files(raw_data, str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))
    selection = _selection_for_series(p1_audit, str(row["series_instance_uid"]))
    slices, exact_applied = apply_duplicate_policy((_header_slice(path) for path in files), selection)
    projections = np.asarray([item.projection for item in slices], dtype=np.float64)
    if len(projections) < 2:
        raise ValueError("DICOM_SERIES_TOO_FEW_SLICES")
    spacings = np.abs(np.diff(projections))
    if np.any(spacings <= DUPLICATE_PROJECTION_TOLERANCE_MM):
        raise ValueError("DICOM_DUPLICATE_SLICE_PLANE_REMAINS")
    median_spacing = float(np.median(spacings))
    if median_spacing <= 0.0:
        raise ValueError("DICOM_NON_POSITIVE_SLICE_SPACING")
    arrays: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for item in slices:
        dataset = pydicom.dcmread(item.path)
        pixels = dataset.pixel_array.astype(np.float32, copy=False)
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in pixels.shape)
        elif tuple(int(value) for value in pixels.shape) != expected_shape:
            raise ValueError("DICOM_ROWS_COLUMNS_INCONSISTENT")
        arrays.append(convert_pixels_to_hu(pixels, item.slope, item.intercept))
    hu = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    return Volume(hu, tuple(slices), (median_spacing, slices[0].pixel_spacing[0], slices[0].pixel_spacing[1]), exact_applied)


def convert_pixels_to_hu(pixels: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Convert DICOM stored pixels to HU without integer truncation."""
    return pixels.astype(np.float32, copy=False) * float(slope) + float(intercept)


def consensus_mask_dhw(annotations: Sequence[Any], scan: Any, volume: Volume) -> np.ndarray:
    """Map pylidc's I,J,K consensus mask into projection-sorted D,H,W coordinates."""
    from pylidc.utils import consensus

    mask_ijk, bbox = consensus(list(annotations), clevel=0.5, ret_masks=False)
    if not bool(mask_ijk.any()):
        raise ValueError("CONSENSUS_MASK_EMPTY")
    return map_pylidc_mask_to_dhw(mask_ijk, bbox, np.asarray(scan.slice_zvals, dtype=np.float64), np.asarray([item.position[2] for item in volume.slices], dtype=np.float64), volume.hu.shape)


def map_pylidc_mask_to_dhw(mask_ijk: np.ndarray, bbox: tuple[slice, slice, slice], scan_z: np.ndarray, dicom_z_in_projection_order: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    """Map I,J,K consensus voxels by physical z into D,H,W volume order.

    `D` may increase or decrease with physical z because it is projection-sorted;
    matching each physical z separately deliberately handles the reversed case.
    """
    output = np.zeros(volume_shape, dtype=np.uint8)
    dicom_z = np.asarray(dicom_z_in_projection_order, dtype=np.float64)
    for local_k, scan_k in enumerate(range(bbox[2].start, bbox[2].stop)):
        if scan_k >= len(scan_z):
            raise ValueError("PYLIDC_MASK_SLICE_INDEX_INVALID")
        distances = np.abs(dicom_z - float(scan_z[scan_k]))
        candidates = np.flatnonzero(distances <= 1e-3)
        if len(candidates) != 1:
            raise ValueError("PYLIDC_DICOM_Z_MAPPING_AMBIGUOUS")
        output[int(candidates[0]), bbox[0], bbox[1]] = mask_ijk[:, :, local_k]
    if not bool(output.any()):
        raise ValueError("CONSENSUS_MASK_DHW_EMPTY")
    return output


def tight_bbox(mask: np.ndarray) -> tuple[slice, slice, slice]:
    """Return the non-zero tight bbox in D,H,W order."""
    if mask.ndim != 3 or not bool(mask.any()):
        raise ValueError("TIGHT_BBOX_REQUIRES_NONEMPTY_3D_MASK")
    coordinates = np.argwhere(mask)
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0) + 1
    return tuple(slice(int(start), int(stop)) for start, stop in zip(lower, upper, strict=True))  # type: ignore[return-value]


def pad_to_cube(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    """Pad a cropped D,H,W image/mask to a cube, putting odd voxel at high side."""
    if image.shape != mask.shape or image.ndim != 3:
        raise ValueError("CROP_IMAGE_MASK_SHAPE_MISMATCH")
    edge = max(image.shape)
    padding = tuple(((edge - size) // 2, edge - size - (edge - size) // 2) for size in image.shape)
    return np.pad(image, padding, constant_values=HU_MIN), np.pad(mask, padding, constant_values=0), padding


def resize_roi(image_hu: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply fixed P3 interpolation, clip, and normalization in D,H,W order."""
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_hu)).float()[None, None]
    mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None, None]
    resized_image = functional.interpolate(image_tensor, size=ROI_SHAPE, mode="trilinear", align_corners=False)[0, 0]
    resized_mask = functional.interpolate(mask_tensor, size=ROI_SHAPE, mode="nearest")[0, 0]
    normalized = ((resized_image.clamp(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).to(torch.float32).numpy()
    result_mask = resized_mask.to(torch.uint8).numpy()
    if set(np.unique(result_mask).tolist()) - {0, 1}:
        raise ValueError("ROI_MASK_NOT_BINARY_AFTER_NEAREST_RESIZE")
    return normalized[None], result_mask[None]


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def deterministic_npz_bytes(image: np.ndarray, mask: np.ndarray, metadata: dict[str, Any]) -> bytes:
    """Serialize ROI files with stable member order, timestamps, and compression."""
    stream = io.BytesIO()
    metadata_bytes = _canonical_json(metadata).encode("utf-8")
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for name, content in (("image.npy", _npy_bytes(image)), ("mask.npy", _npy_bytes(mask)), ("metadata.json", metadata_bytes)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return stream.getvalue()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metadata(row: dict[str, Any], config_hash: str, volume: Volume, bbox: tuple[slice, slice, slice], padding: tuple[tuple[int, int], ...], pre_mask_voxels: int, post_mask_voxels: int) -> dict[str, Any]:
    source_fingerprint = roi_source_fingerprint(row)
    bbox_dimensions = [item.stop - item.start for item in bbox]
    cube_edge = int(max(bbox_dimensions))
    return {
        "schema_version": 1,
        "protocol": "Baseline-v2",
        "config_sha256": config_hash,
        "nodule_uid": str(row["nodule_uid"]),
        "source_fingerprint": source_fingerprint,
        "annotation_source_fingerprints": str(row["annotation_source_fingerprints"]),
        "source_dicom_sop_fingerprints": str(row["source_dicom_sop_fingerprints"]),
        "axis_order": "D,H,W",
        "source_spacing_dhw_mm": list(volume.spacing_dhw),
        "tight_bbox_dhw": [[item.start, item.stop] for item in bbox],
        "cube_edge_voxels": cube_edge,
        "padding_dhw": [list(item) for item in padding],
        "padding_ratio": 1.0 - (math.prod(bbox_dimensions) / float(cube_edge ** 3)),
        "image_padding_hu": HU_MIN,
        "mask_padding_value": 0,
        "image_interpolation": "trilinear",
        "image_align_corners": False,
        "mask_interpolation": "nearest",
        "image_clip_hu": [HU_MIN, HU_MAX],
        "normalization": "(HU + 1000) / 1700",
        "pre_resize_mask_voxels": pre_mask_voxels,
        "post_resize_mask_voxels": post_mask_voxels,
        "exact_duplicate_selection_applied": volume.exact_duplicate_applied,
    }


def roi_source_fingerprint(row: dict[str, Any]) -> str:
    """Bind a reusable ROI to its immutable manifest-level source provenance."""
    return _sha256(_canonical_json({
        "canonical_xml_sha256": str(row["canonical_xml_sha256"]),
        "annotation_source_fingerprints": str(row["annotation_source_fingerprints"]),
        "source_dicom_sop_fingerprints": str(row["source_dicom_sop_fingerprints"]),
    }))


def write_roi(path: Path, image: np.ndarray, mask: np.ndarray, metadata: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    """Write or safely reuse a deterministic private ROI without silent overwrite."""
    if image.shape != (1, *ROI_SHAPE) or image.dtype != np.float32 or mask.shape != (1, *ROI_SHAPE) or mask.dtype != np.uint8:
        raise ValueError("ROI_INTERFACE_INVALID")
    unsigned = {key: value for key, value in metadata.items() if key != "roi_content_sha256"}
    metadata = {**unsigned, "roi_content_sha256": _sha256(_canonical_json(unsigned) + _sha256(image.tobytes()) + _sha256(mask.tobytes()))}
    content = deterministic_npz_bytes(image, mask, metadata)
    content_hash = _sha256(content)
    if path.exists():
        existing = path.read_bytes()
        if existing == content:
            return {"status": "REUSED", "content_sha256": content_hash, "metadata": metadata}
        if not overwrite:
            raise FileExistsError(f"ROI_EXISTS_WITH_DIFFERENT_CONTENT:{path.name}")
    _atomic_bytes(path, content)
    return {"status": "WRITTEN", "content_sha256": content_hash, "metadata": metadata}


def validate_roi_entry(uid: str, row: dict[str, Any], config_hash: str, root: Path) -> None:
    """Validate one private ROI's index, bytes, arrays, and provenance metadata."""
    if row.get("status") not in {"WRITTEN", "REUSED"} or not row.get("relative_roi_path") or not row.get("roi_file_sha256"):
        raise ValueError("ROI_VERIFY_INDEX_STATUS_INVALID")
    path = root / str(row["relative_roi_path"])
    if not path.exists() or _sha256(path.read_bytes()) != str(row["roi_file_sha256"]):
        raise ValueError("ROI_VERIFY_FILE_SHA256_MISMATCH")
    with zipfile.ZipFile(path) as archive:
        image = np.load(io.BytesIO(archive.read("image.npy")), allow_pickle=False)
        mask = np.load(io.BytesIO(archive.read("mask.npy")), allow_pickle=False)
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    if image.shape != (1, *ROI_SHAPE) or image.dtype != np.float32 or mask.shape != (1, *ROI_SHAPE) or mask.dtype != np.uint8:
        raise ValueError("ROI_VERIFY_INTERFACE_INVALID")
    if not bool(mask.any()) or set(np.unique(mask).tolist()) - {0, 1} or metadata.get("config_sha256") != config_hash or metadata.get("nodule_uid") != uid:
        raise ValueError("ROI_VERIFY_CONTENT_INVALID")
    unsigned = {key: value for key, value in metadata.items() if key != "roi_content_sha256"}
    expected = _sha256(_canonical_json(unsigned) + _sha256(image.tobytes()) + _sha256(mask.tobytes()))
    if metadata.get("roi_content_sha256") != expected:
        raise ValueError("ROI_VERIFY_METADATA_CONTENT_HASH_MISMATCH")


def _index_from_metadata(uid: str, metadata: dict[str, Any], file_hash: str, status: str, reader_count: int) -> dict[str, Any]:
    """Create an auditable ROI-index entry from authoritative private metadata."""
    return {
        "nodule_uid": uid, "status": status, "relative_roi_path": f"rois/{uid}.npz",
        "roi_file_sha256": file_hash, "source_fingerprint": metadata["source_fingerprint"],
        "reader_count": reader_count, "bbox_dhw": _canonical_json(metadata["tight_bbox_dhw"]),
        "cube_edge_voxels": metadata["cube_edge_voxels"], "padding_dhw": _canonical_json(metadata["padding_dhw"]),
        "spacing_dhw_mm": _canonical_json(metadata["source_spacing_dhw_mm"]), "pre_resize_mask_voxels": metadata["pre_resize_mask_voxels"],
        "post_resize_mask_voxels": metadata["post_resize_mask_voxels"], "exact_duplicate_selection_applied": metadata["exact_duplicate_selection_applied"],
    }


def reusable_roi(row: dict[str, Any], config_hash: str, roi_root: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Reuse only a fully verified ROI whose manifest provenance is unchanged."""
    uid = str(row["nodule_uid"])
    path = roi_root / f"{uid}.npz"
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as archive:
        image = np.load(io.BytesIO(archive.read("image.npy")), allow_pickle=False)
        mask = np.load(io.BytesIO(archive.read("mask.npy")), allow_pickle=False)
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    if metadata.get("config_sha256") != config_hash or metadata.get("nodule_uid") != uid or metadata.get("source_fingerprint") != roi_source_fingerprint(row):
        raise FileExistsError(f"ROI_EXISTS_WITH_DIFFERENT_PROVENANCE:{path.name}")
    unsigned = {key: value for key, value in metadata.items() if key != "roi_content_sha256"}
    expected_content = _sha256(_canonical_json(unsigned) + _sha256(image.tobytes()) + _sha256(mask.tobytes()))
    if image.shape != (1, *ROI_SHAPE) or image.dtype != np.float32 or mask.shape != (1, *ROI_SHAPE) or mask.dtype != np.uint8 or not bool(mask.any()) or set(np.unique(mask).tolist()) - {0, 1} or metadata.get("roi_content_sha256") != expected_content:
        raise ValueError("ROI_REUSE_VALIDATION_FAILED")
    return _index_from_metadata(uid, metadata, _sha256(path.read_bytes()), "REUSED", int(row["reader_count"])), metadata


ROI_REUSE_VALIDATION_ERRORS = (FileExistsError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile, OSError, EOFError)


def reuse_or_schedule_rebuild(row: dict[str, Any], config_hash: str, roi_root: Path, overwrite: bool) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Default to a hard provenance failure; explicit overwrite schedules a rebuild."""
    try:
        return reusable_roi(row, config_hash, roi_root)
    except ROI_REUSE_VALIDATION_ERRORS:
        if overwrite:
            return None
        raise


def _consensus_for_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    enable_pylidc_numpy_compatibility()
    import pylidc as pl

    identifiers = sorted({int(item) for row in rows for item in json.loads(str(row["pylidc_annotation_sql_ids"]))})
    # Eager-load once: SQLite parameter limits make one large IN query unsafe.
    annotations = pl.query(pl.Annotation).all()
    lookup = {int(annotation.id): annotation for annotation in annotations}
    grouped: dict[str, list[Any]] = {}
    scans: dict[str, Any] = {}
    for row in rows:
        ids = [int(item) for item in json.loads(str(row["pylidc_annotation_sql_ids"]))]
        selected = [lookup[item] for item in ids]
        if len(selected) != len(ids):
            raise ValueError("PYLIDC_ANNOTATION_MAPPING_INCOMPLETE")
        grouped[str(row["nodule_uid"])] = selected
        scans[str(row["nodule_uid"])] = selected[0].scan
    return grouped, scans


def validate_annotation_mapping(rows: list[dict[str, Any]], mapping_path: Path) -> None:
    """Ensure local source-to-SQL mapping exactly agrees with manifest clusters."""
    mapping = pd.read_parquet(mapping_path)
    expected = {str(row["nodule_uid"]): sorted(int(value) for value in json.loads(str(row["pylidc_annotation_sql_ids"]))) for row in rows}
    mapping = mapping.loc[mapping["nodule_uid"].astype(str).isin(expected)]
    actual = {str(uid): sorted(int(value) for value in group["pylidc_annotation_sql_id"].tolist()) for uid, group in mapping.groupby("nodule_uid", sort=False)}
    if set(actual) != set(expected) or any(actual[uid] != expected[uid] for uid in expected):
        raise ValueError("ANNOTATION_MAPPING_MANIFEST_MISMATCH")


def _pilot_uids(rows: list[dict[str, Any]], statistics: dict[str, dict[str, Any]], seed: int, exact_uid: str | None) -> list[str]:
    """Select 41 deterministic QA samples according to the approved strata."""
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    by_readers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_readers[int(row["reader_count"])].append(row)
    for count in range(1, 5):
        options = sorted(by_readers[count], key=lambda item: str(item["nodule_uid"]))
        picks = rng.choice(len(options), size=min(6, len(options)), replace=False) if options else []
        selected.extend(str(options[int(index)]["nodule_uid"]) for index in sorted(picks))
    ordered_volume = sorted(rows, key=lambda row: (float(statistics[str(row["nodule_uid"])]["physical_volume_mm3"]), str(row["nodule_uid"])))
    selected.extend(str(row["nodule_uid"]) for row in ordered_volume[:8])
    selected.extend(str(row["nodule_uid"]) for row in ordered_volume[-8:])
    if exact_uid is not None:
        selected.append(exact_uid)
    unique: list[str] = []
    for uid in selected:
        if uid not in unique:
            unique.append(uid)
    fallback = sorted(rows, key=lambda row: (-float(statistics[str(row["nodule_uid"])]["padding_ratio"]), str(row["nodule_uid"])))
    for row in fallback:
        if len(unique) >= PILOT_SIZE:
            break
        uid = str(row["nodule_uid"])
        if uid not in unique:
            unique.append(uid)
    if len(unique) != PILOT_SIZE:
        raise ValueError("PILOT_SELECTION_SIZE_MISMATCH")
    return unique


def padding_ratio_for_dimensions(dimensions: Sequence[int]) -> float:
    edge = max(dimensions)
    return 1.0 - (math.prod(dimensions) / float(edge ** 3))


def consensus_physical_volume_mm3(voxels: int, spacing_dhw: Sequence[float]) -> float:
    """Calculate consensus physical volume from its voxel count and D,H,W spacing."""
    if len(spacing_dhw) != 3 or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in spacing_dhw):
        raise ValueError("CONSENSUS_VOLUME_SPACING_INVALID")
    return float(voxels) * math.prod(float(value) for value in spacing_dhw)


def scan_geometry_fingerprint(scan: Any) -> str:
    """Fingerprint geometry values on which mask mapping and physical volume depend."""
    return _sha256(_canonical_json({
        "slice_zvals": [float(value) for value in scan.slice_zvals],
        "slice_spacing": abs(float(scan.slice_spacing)),
        "pixel_spacing": float(scan.pixel_spacing),
    }))


def pilot_statistics_source_fingerprint(row: dict[str, Any], geometry_fingerprint: str) -> str:
    """Bind resumable pilot statistics to annotation and DICOM geometry provenance."""
    return _sha256(_canonical_json({
        "canonical_xml_sha256": str(row["canonical_xml_sha256"]),
        "annotation_source_fingerprints": str(row["annotation_source_fingerprints"]),
        "source_dicom_sop_fingerprints": str(row["source_dicom_sop_fingerprints"]),
        "series_instance_uid": str(row["series_instance_uid"]),
        "dicom_geometry_fingerprint": geometry_fingerprint,
    }))


def reusable_pilot_statistics(cached: Iterable[dict[str, Any]], rows: Iterable[dict[str, Any]], geometry_fingerprints: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Return only cache rows whose source provenance matches the current manifest."""
    expected = {str(row["nodule_uid"]): pilot_statistics_source_fingerprint(row, geometry_fingerprints[str(row["nodule_uid"])]) for row in rows}
    return {str(row["nodule_uid"]): row for row in cached if str(row.get("source_fingerprint", "")) == expected.get(str(row["nodule_uid"]))}


def _write_index(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows).sort_values("nodule_uid")
    _atomic_parquet(path, frame)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically replace a private parquet file with a unique sibling temporary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_private_failures(path: Path, attempted_uids: Iterable[str], rows: list[dict[str, Any]]) -> None:
    """Replace failure state for attempted nodules while retaining unrelated failures."""
    attempted = set(attempted_uids)
    existing = pd.read_parquet(path).to_dict(orient="records") if path.exists() else []
    combined = [row for row in existing if str(row["nodule_uid"]) not in attempted] + rows
    _atomic_parquet(path, pd.DataFrame(combined, columns=["nodule_uid", "patient_id", "series_instance_uid", "reason"]).sort_values("nodule_uid"))


def assert_deidentified_audit(path: Path, forbidden_values: Iterable[str]) -> None:
    """Reject tracked audit text containing raw IDs or absolute source paths."""
    content = path.read_text(encoding="utf-8")
    if "/Users/" in content or "/private/" in content:
        raise ValueError("TRACKED_AUDIT_CONTAINS_ABSOLUTE_PATH")
    if any(value and value in content for value in forbidden_values):
        raise ValueError("TRACKED_AUDIT_CONTAINS_RAW_IDENTIFIER")


def _qa_image(path: Path, source_hu: np.ndarray, source_mask: np.ndarray, image: np.ndarray, mask: np.ndarray, label: str) -> None:
    """Create a deidentified source/final three-plane overlay panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    def largest_slice(binary: np.ndarray, axis: int) -> int:
        areas = binary.sum(axis=tuple(index for index in range(3) if index != axis))
        return int(np.argmax(areas))

    fig, axes = pyplot.subplots(2, 3, figsize=(11, 7))
    def overlay(axis: Any, binary: np.ndarray) -> None:
        if binary.shape[0] >= 2 and binary.shape[1] >= 2 and bool(binary.any()):
            try:
                axis.contour(binary, levels=[0.5], colors="lime", linewidths=0.8)
            except TypeError:
                # Preserve visible spatial evidence when contour topology is degenerate.
                overlay_mask = np.ma.masked_where(binary == 0, binary)
                axis.imshow(overlay_mask, cmap="summer", alpha=0.65, vmin=0, vmax=1, interpolation="nearest")

    for column, axis in enumerate(range(3)):
        source_index = largest_slice(source_mask, axis)
        final_index = largest_slice(mask[0], axis)
        source_plane = np.take(source_hu, source_index, axis=axis)
        source_binary = np.take(source_mask, source_index, axis=axis)
        final_plane = np.take(image[0], final_index, axis=axis)
        final_binary = np.take(mask[0], final_index, axis=axis)
        axes[0, column].imshow(source_plane, cmap="gray", vmin=HU_MIN, vmax=HU_MAX)
        overlay(axes[0, column], source_binary)
        axes[1, column].imshow(final_plane, cmap="gray", vmin=0.0, vmax=1.0)
        overlay(axes[1, column], final_binary)
        axes[0, column].set_title(("D", "H", "W")[axis])
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle(label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, metadata={"Creator": "lidc_baseline"})
    pyplot.close(fig)


def _exact_primary_uid(rows: list[dict[str, Any]], p1_audit: Path) -> str | None:
    with (p1_audit / "derived_volume_selection.csv").open(encoding="utf-8", newline="") as stream:
        keys = {row["series_key"] for row in csv.DictReader(stream)}
    matched = [str(row["nodule_uid"]) for row in rows if _key("series", str(row["series_instance_uid"])) in keys]
    if len(matched) != 1:
        raise ValueError("P1_EXACT_DUPLICATE_PRIMARY_NODULE_EXPECTATION_FAILED")
    return matched[0]


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    rows = frame.loc[frame["primary_regression_eligible"].astype(bool)].to_dict(orient="records")
    if len(rows) != 2633:
        raise ValueError(f"P3_PRIMARY_COHORT_COUNT_MISMATCH:{len(rows)}")
    return sorted(rows, key=lambda row: str(row["nodule_uid"]))


def _pilot_confirmation_path() -> Path:
    return Path("reports/baseline_v2/p3_qa/pilot_confirmation.json")


def _selection_hash(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError("PILOT_SELECTION_MISSING")
    return _sha256(path.read_bytes())


def require_pilot_confirmation(config_hash: str, selection_path: Path) -> None:
    """Block a full build unless the user-confirmed private pilot marker matches."""
    path = _pilot_confirmation_path()
    if not path.exists():
        raise PermissionError("PILOT_QA_USER_CONFIRMATION_REQUIRED")
    marker = json.loads(path.read_text(encoding="utf-8"))
    if marker.get("user_confirmation") is not True or marker.get("config_sha256") != config_hash or marker.get("pilot_selection_sha256") != _selection_hash(selection_path):
        raise PermissionError("PILOT_QA_CONFIRMATION_MARKER_INVALID")


def confirm_pilot(config_path: str | Path, confirmation_note: str) -> dict[str, Any]:
    """Record a private, explicit user confirmation before a full ROI build."""
    if not confirmation_note.strip():
        raise ValueError("PILOT_CONFIRMATION_NOTE_REQUIRED")
    config_hash = compute_config_sha256(load_config(config_path))
    selection_path = Path("reports/baseline_v2/p3_qa/pilot_selection.json")
    payload = {"user_confirmation": True, "confirmation_note": confirmation_note.strip(), "config_sha256": config_hash, "pilot_selection_sha256": _selection_hash(selection_path)}
    write_json(_pilot_confirmation_path(), payload)
    return payload


def _precompute_pilot_statistics(rows: list[dict[str, Any]], grouped: dict[str, list[Any]], scans: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Cache resumable local consensus volumes/bboxes for deterministic pilot selection."""
    from pylidc.utils import consensus

    cache_path = Path("artifacts/baseline_v2/manifests/pilot_consensus_statistics.parquet")
    cached = pd.read_parquet(cache_path).to_dict(orient="records") if cache_path.exists() else []
    geometries = {str(row["nodule_uid"]): scan_geometry_fingerprint(scans[str(row["nodule_uid"])]) for row in rows}
    expected_source = {str(row["nodule_uid"]): pilot_statistics_source_fingerprint(row, geometries[str(row["nodule_uid"])]) for row in rows}
    result = reusable_pilot_statistics(cached, rows, geometries)
    changed = False
    for row in rows:
        uid = str(row["nodule_uid"])
        if uid in result:
            continue
        mask, _ = consensus(grouped[uid], clevel=0.5, ret_masks=False)
        if not bool(mask.any()):
            raise ValueError(f"CONSENSUS_MASK_EMPTY:{uid}")
        bbox = tight_bbox(np.transpose(mask.astype(np.uint8), (2, 0, 1)))
        dimensions = [item.stop - item.start for item in bbox]
        spacing = (abs(float(scans[uid].slice_spacing)), float(scans[uid].pixel_spacing), float(scans[uid].pixel_spacing))
        voxels = int(mask.sum())
        result[uid] = {"nodule_uid": uid, "source_fingerprint": expected_source[uid], "mask_voxels": voxels, "physical_volume_mm3": consensus_physical_volume_mm3(voxels, spacing), "bbox_dimensions_dhw": _canonical_json(dimensions), "padding_ratio": padding_ratio_for_dimensions(dimensions)}
        changed = True
        if len(result) % 20 == 0:
            _write_index(cache_path, list(result.values()))
    if changed:
        _write_index(cache_path, list(result.values()))
    if len(result) != len(rows):
        raise ValueError("PILOT_CONSENSUS_STATISTICS_INCOMPLETE")
    return result


def _process_one(row: dict[str, Any], annotations: list[Any], scan: Any, raw_data: Path, p1_audit: Path, config_hash: str, roi_root: Path, qa_root: Path | None, overwrite: bool, volume: Volume | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    volume = volume if volume is not None else load_volume(raw_data, row, p1_audit)
    consensus = consensus_mask_dhw(annotations, scan, volume)
    bbox = tight_bbox(consensus)
    source_image, source_mask = volume.hu[bbox], consensus[bbox]
    cube_image, cube_mask, padding = pad_to_cube(source_image, source_mask)
    image, mask = resize_roi(cube_image, cube_mask)
    metadata = _metadata(row, config_hash, volume, bbox, padding, int(consensus.sum()), int(mask.sum()))
    written = write_roi(roi_root / f"{row['nodule_uid']}.npz", image, mask, metadata, overwrite=overwrite)
    index = _index_from_metadata(str(row["nodule_uid"]), metadata, written["content_sha256"], written["status"], int(row["reader_count"]))
    if qa_root is not None:
        label = f"nodule={str(row['nodule_uid'])[:12]} readers={row['reader_count']} bbox={tuple(item.stop-item.start for item in bbox)} cube={metadata['cube_edge_voxels']} spacing={tuple(round(item, 3) for item in volume.spacing_dhw)}"
        _qa_image(qa_root / f"{row['nodule_uid']}.png", source_image, source_mask, image, mask, label)
    return index, metadata


def build(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    config_hash = compute_config_sha256(config)
    raw_data, manifest, mapping, p1_audit = (Path(value).expanduser().resolve() for value in (arguments.raw_data, arguments.manifest, arguments.annotation_mapping, arguments.p1_audit))
    if not mapping.exists():
        raise FileNotFoundError(f"Annotation mapping is missing: {mapping}")
    rows = _manifest_rows(manifest)
    validate_annotation_mapping(rows, mapping)
    grouped, scans = _consensus_for_rows(rows)
    exact_uid = _exact_primary_uid(rows, p1_audit)
    selection_path = Path("reports/baseline_v2/p3_qa/pilot_selection.json")
    if arguments.scope == "pilot":
        statistics = _precompute_pilot_statistics(rows, grouped, scans)
        selected = _pilot_uids(rows, statistics, int(config["reproducibility"]["base_seed"]), exact_uid)
        write_json(selection_path, {"scope": "pilot", "nodule_uids": selected, "seed": int(config["reproducibility"]["base_seed"]), "count": len(selected)})
        targets = [row for row in rows if str(row["nodule_uid"]) in set(selected)]
    else:
        require_pilot_confirmation(config_hash, selection_path)
        targets = rows
    roi_root = Path(config["paths"]["roi_directory"])
    index_path = Path("artifacts/baseline_v2/manifests/roi_index.parquet")
    existing_rows = pd.read_parquet(index_path).to_dict(orient="records") if index_path.exists() else []
    index_by_uid = {str(row["nodule_uid"]): row for row in existing_rows}
    qa_root = Path("reports/baseline_v2/p3_qa/pilot") if arguments.scope == "pilot" else None
    metadata_by_uid: dict[str, dict[str, Any]] = {}
    failure_rows: list[dict[str, Any]] = []
    reuse_failure_rows: list[dict[str, Any]] = []
    failures_path = Path("artifacts/baseline_v2/manifests/roi_failures.parquet")
    pending_by_series: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    def record_failure(row: dict[str, Any], error: Exception) -> dict[str, Any]:
        uid = str(row["nodule_uid"])
        failure = {"nodule_uid": uid, "patient_id": str(row["patient_id"]), "series_instance_uid": str(row["series_instance_uid"]), "reason": f"{type(error).__name__}:{error}"}
        index_by_uid[uid] = {"nodule_uid": uid, "status": "FAILED", "relative_roi_path": None, "roi_file_sha256": None, "source_fingerprint": None, "reader_count": int(row["reader_count"]), "bbox_dhw": None, "cube_edge_voxels": None, "padding_dhw": None, "spacing_dhw_mm": None, "pre_resize_mask_voxels": None, "post_resize_mask_voxels": None, "exact_duplicate_selection_applied": None}
        return failure

    for row in targets:
        uid = str(row["nodule_uid"])
        try:
            existing = reuse_or_schedule_rebuild(row, config_hash, roi_root, arguments.overwrite)
            if existing is not None:
                index_by_uid[uid], metadata_by_uid[uid] = existing
                continue
        except Exception as error:
            failure = record_failure(row, error)
            failure_rows.append(failure)
            reuse_failure_rows.append(failure)
            continue
        pending_by_series[(str(row["patient_id"]), str(row["study_instance_uid"]), str(row["series_instance_uid"]))].append(row)

    # Persist every CT-series unit. A stopped full build is therefore resumable
    # without trusting partial index state or recomputing verified private ROIs.
    for pending in (pending_by_series[key] for key in sorted(pending_by_series)):
        attempted = [str(row["nodule_uid"]) for row in pending]
        series_failures: list[dict[str, Any]] = []
        try:
            volume = load_volume(raw_data, pending[0], p1_audit)
        except Exception as error:
            series_failures.extend(record_failure(row, error) for row in pending)
        else:
            for row in pending:
                try:
                    index, metadata = _process_one(row, grouped[str(row["nodule_uid"])], scans[str(row["nodule_uid"])], raw_data, p1_audit, config_hash, roi_root, qa_root, arguments.overwrite, volume=volume)
                    index_by_uid[index["nodule_uid"]] = index
                    metadata_by_uid[index["nodule_uid"]] = metadata
                except Exception as error:
                    series_failures.append(record_failure(row, error))
        failure_rows.extend(series_failures)
        _write_index(index_path, list(index_by_uid.values()))
        update_private_failures(failures_path, attempted, series_failures)

    if reuse_failure_rows:
        update_private_failures(failures_path, (row["nodule_uid"] for row in reuse_failure_rows), reuse_failure_rows)
    # Reused ROIs also clear stale failure records from previous interrupted runs.
    reused_uids = [uid for uid, item in index_by_uid.items() if item.get("status") == "REUSED"]
    if reused_uids:
        update_private_failures(failures_path, reused_uids, [])
    metadata_items = [metadata_by_uid[str(row["nodule_uid"])] for row in targets if str(row["nodule_uid"]) in metadata_by_uid]
    summary = {
        "scope": arguments.scope, "input_primary_nodules": len(rows), "processed_nodules": len(targets), "successful_nodules": len(metadata_items),
        "nonempty_masks": sum(item["pre_resize_mask_voxels"] > 0 for item in metadata_items), "failures": len(failure_rows), "config_sha256": config_hash,
        "exact_duplicate_policy_applied": sum(bool(item["exact_duplicate_selection_applied"]) for item in metadata_items),
        "raw_dicom_files_modified": 0, "private_outputs": {"roi_directory": "artifacts/baseline_v2/rois", "roi_index": "artifacts/baseline_v2/manifests/roi_index.parquet"},
    }
    if arguments.scope == "full" and not failure_rows:
        audit_root = Path("artifacts/baseline_v2/audit/p3")
        roi_files = list(roi_root.glob("*.npz"))
        summary.update({
            "ct_series_count": len({str(row["series_instance_uid"]) for row in rows}),
            "reader_count_distribution": dict(sorted(Counter(int(row["reader_count"]) for row in rows).items())),
            "roi_file_count": len(roi_files), "roi_total_bytes": sum(path.stat().st_size for path in roi_files),
            "fingerprints": {"manifest": _sha256(manifest.read_bytes()), "roi_index": _sha256(index_path.read_bytes()), "config": config_hash},
            "qa": {"pilot_sample_count": PILOT_SIZE, "human_confirmation": "CONFIRMED"},
        })
        write_json(audit_root / "summary.json", summary)
        distributions = pd.DataFrame(metadata_items)
        rows_out = []
        for name in ("pre_resize_mask_voxels", "post_resize_mask_voxels", "cube_edge_voxels", "padding_ratio"):
            values = distributions[name].astype(float)
            rows_out.extend({"metric": name, "quantile": label, "value": float(values.quantile(value))} for label, value in (("p00", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p100", 1)))
        for axis in range(3):
            values = pd.Series([item["tight_bbox_dhw"][axis][1] - item["tight_bbox_dhw"][axis][0] for item in metadata_items], dtype=float)
            rows_out.extend({"metric": f"bbox_axis_{axis}_voxels", "quantile": label, "value": float(values.quantile(value))} for label, value in (("p00", 0), ("p25", .25), ("p50", .5), ("p75", .75), ("p100", 1)))
        for name, count in sorted(Counter(int(row["reader_count"]) for row in rows).items()):
            rows_out.append({"metric": "reader_count", "quantile": str(name), "value": float(count)})
        with (audit_root / "distributions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "quantile", "value"], lineterminator="\n")
            writer.writeheader(); writer.writerows(rows_out)
        with (audit_root / "reconciliation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["stage", "count"], lineterminator="\n")
            writer.writeheader(); writer.writerows([{"stage": "primary_regression_input", "count": len(rows)}, {"stage": "roi_success", "count": len(metadata_items)}])
        raw_ids = {str(row["patient_id"]) for row in rows} | {str(row["series_instance_uid"]) for row in rows} | {str(row["nodule_uid"]) for row in rows}
        for artifact in (audit_root / "summary.json", audit_root / "distributions.csv", audit_root / "reconciliation.csv"):
            assert_deidentified_audit(artifact, raw_ids)
    if failure_rows:
        raise RuntimeError(f"P3_ROI_BUILD_FAILED:{len(failure_rows)}")
    return summary


def verify(arguments: argparse.Namespace) -> dict[str, Any]:
    config = load_config(arguments.config)
    config_hash = compute_config_sha256(config)
    index_path = Path("artifacts/baseline_v2/manifests/roi_index.parquet")
    if not index_path.exists():
        raise FileNotFoundError("ROI_INDEX_MISSING")
    frame = pd.read_parquet(index_path)
    primary = _manifest_rows(Path(config["paths"]["manifest"]))
    if arguments.scope == "pilot":
        selection = json.loads(Path("reports/baseline_v2/p3_qa/pilot_selection.json").read_text(encoding="utf-8"))
        expected_uids = set(selection["nodule_uids"])
        if len(expected_uids) != PILOT_SIZE:
            raise ValueError("PILOT_SELECTION_VERIFY_INVALID")
    else:
        expected_uids = {str(row["nodule_uid"]) for row in primary}
        require_pilot_confirmation(config_hash, Path("reports/baseline_v2/p3_qa/pilot_selection.json"))
    indexed = {str(row["nodule_uid"]): row for row in frame.to_dict(orient="records")}
    if arguments.scope == "full" and set(indexed) != expected_uids:
        raise ValueError("ROI_INDEX_UID_SET_MISMATCH")
    if not expected_uids.issubset(indexed):
        raise ValueError("ROI_INDEX_INCOMPLETE_OR_FAILED")
    checked = 0
    for uid in sorted(expected_uids):
        validate_roi_entry(uid, indexed[uid], config_hash, Path(config["paths"]["roi_directory"]).parent)
        checked += 1
    return {"scope": arguments.scope, "verified_rois": checked, "config_sha256": config_hash}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("build", "verify", "confirm-pilot"):
        command = commands.add_parser(name)
        command.add_argument("--config", default="configs/baseline_v2.yaml")
        if name != "confirm-pilot":
            command.add_argument("--scope", choices=("pilot", "full"), required=True)
        if name == "build":
            command.add_argument("--raw-data", required=True)
            command.add_argument("--manifest", required=True)
            command.add_argument("--annotation-mapping", required=True)
            command.add_argument("--p1-audit", required=True)
            command.add_argument("--overwrite", action="store_true")
        if name == "confirm-pilot":
            command.add_argument("--confirmation-note", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    summary = build(arguments) if arguments.command == "build" else verify(arguments) if arguments.command == "verify" else confirm_pilot(arguments.config, arguments.confirmation_note)
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
