"""Unit tests for Phase 2 source-derived cohort logic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.p2_cohort import (
    CLUSTER_PARAMETERS,
    PRIVATE_MANIFEST_REQUIRED_COLUMNS,
    SourceAnnotation,
    _load_excluded_series,
    _private_manifest_row,
    aggregate_cluster,
    annotation_to_cluster_mapping_rows,
    cluster_has_supported_reader_count,
    cluster_annotations,
    cluster_annotations_with_effective_tolerance,
    enable_pylidc_numpy_compatibility,
    match_source_annotation,
    parse_canonical_document,
    split_spatially_usable_sources,
    stable_nodule_uid,
    validate_private_manifest,
)


class _Contour:
    def __init__(self, z: float, inclusion: bool, coords: str) -> None:
        self.image_z_position = z
        self.inclusion = inclusion
        self.coords = coords


class _Annotation:
    def __init__(self, nodule_id: str, contours: list[_Contour], *, annotation_id: int = 1, diameter: float = 3.1) -> None:
        self._nodule_id = nodule_id
        self.contours = contours
        self.id = annotation_id
        self.diameter = diameter


class _ManifestScan:
    id = 7
    patient_id = "patient"
    study_instance_uid = "study"
    series_instance_uid = "series"
    slice_thickness = 2.0


class _ToleranceScan:
    slice_thickness = 2.0

    def __init__(self) -> None:
        self.annotations = [_Annotation(f"nodule-{index}", [], annotation_id=index) for index in range(5)]

    def cluster_annotations(self, **kwargs):
        assert kwargs["metric"] == "min"
        assert kwargs["tol"] is None
        assert kwargs["factor"] == 0.9
        assert kwargs["min_tol"] == 0.1
        assert kwargs["verbose"] is False
        clusters = [[annotation] for annotation in self.annotations]
        if kwargs.get("return_distance_matrix"):
            distances = np.full((5, 5), 1.9)
            np.fill_diagonal(distances, 0.0)
            return clusters, distances
        return clusters


class _Scan:
    def __init__(self) -> None:
        self.arguments = None

    def cluster_annotations(self, **kwargs):
        self.arguments = kwargs
        return [["annotation"]]


def _source(fingerprint: str, *, nodule_id: str = "nodule-1", characteristics: dict[str, int | None] | None = None, geometry: str = "geometry") -> SourceAnnotation:
    values = characteristics or {
        "malignancy": 4,
        "subtlety": 3,
        "sphericity": 3,
        "margin": 3,
        "lobulation": 3,
        "spiculation": 3,
        "texture": 3,
        "internalStructure": 1,
        "calcification": 2,
    }
    return SourceAnnotation(
        patient_id="patient",
        study_uid="study",
        series_uid="series",
        xml_sha256="xml-hash",
        xml_relative_path="source.xml",
        session_index=0,
        nodule_id=nodule_id,
        annotation_class="nodule >=3 mm",
        characteristics=values,
        geometry_signature=geometry,
        matching_geometry_signature=geometry,
        sop_fingerprint="sop-fingerprint",
        has_required_spatial_source=True,
        source_fingerprint=fingerprint,
    )


def test_numpy_compatibility_adapter_only_sets_missing_alias() -> None:
    class Missing:
        pass

    class Existing:
        int = "existing"

    missing = Missing()
    assert enable_pylidc_numpy_compatibility(missing) is True
    assert missing.int is int
    assert enable_pylidc_numpy_compatibility(Existing()) is False


def test_clustering_uses_fixed_pylidc_defaults() -> None:
    scan = _Scan()
    assert cluster_annotations(scan) == [["annotation"]]
    assert scan.arguments == CLUSTER_PARAMETERS


def test_effective_clustering_tolerance_records_pylidc_default_reduction() -> None:
    clusters, tolerance = cluster_annotations_with_effective_tolerance(_ToleranceScan())
    assert tolerance == pytest.approx(1.8)
    assert [len(cluster) for cluster in clusters] == [1, 1, 1, 1, 1]


def test_parse_source_annotation_records_all_targets_and_geometry(tmp_path: Path) -> None:
    root = tmp_path / "xml"
    path = root / "patient" / "source.xml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """<LidcReadMessage xmlns=\"urn:test\"><ResponseHeader><StudyInstanceUID>study</StudyInstanceUID><SeriesInstanceUid>series</SeriesInstanceUid></ResponseHeader><readingSession><unblindedReadNodule><noduleID>shared</noduleID><characteristics><subtlety>3</subtlety><internalStructure>1</internalStructure><calcification>2</calcification><sphericity>3</sphericity><margin>3</margin><lobulation>3</lobulation><spiculation>3</spiculation><texture>3</texture><malignancy>4</malignancy></characteristics><roi><imageSOP_UID>sop</imageSOP_UID><imageZposition>1.0</imageZposition><inclusion>TRUE</inclusion><edgeMap><xCoord>2</xCoord><yCoord>4</yCoord></edgeMap></roi></unblindedReadNodule><smallNodule><noduleID>small</noduleID></smallNodule><nonNodule><nonNoduleID>non</nonNoduleID></nonNodule></readingSession></LidcReadMessage>""",
        encoding="utf-8",
    )
    document = parse_canonical_document(path, root)
    assert document.class_counts == {"nodule >=3 mm": 1, "nodule <3 mm": 1, "non-nodule": 1}
    assert document.annotations[0].characteristics["malignancy"] == 4
    assert document.annotations[0].annotation_class == "nodule >=3 mm"
    assert document.annotations[0].has_required_spatial_source is True
    assert len(document.annotations[0].source_fingerprint) == 64


def test_source_characteristics_and_sop_changes_change_source_fingerprint(tmp_path: Path) -> None:
    root = tmp_path / "xml"
    path = root / "patient" / "source.xml"
    path.parent.mkdir(parents=True)
    template = """<LidcReadMessage><ResponseHeader><StudyInstanceUID>study</StudyInstanceUID><SeriesInstanceUid>series</SeriesInstanceUid></ResponseHeader><readingSession><unblindedReadNodule><noduleID>id</noduleID><characteristics><subtlety>{subtlety}</subtlety><internalStructure>1</internalStructure><calcification>1</calcification><sphericity>1</sphericity><margin>1</margin><lobulation>1</lobulation><spiculation>1</spiculation><texture>1</texture><malignancy>1</malignancy></characteristics><roi><imageSOP_UID>{sop_uid}</imageSOP_UID><imageZposition>1</imageZposition><inclusion>TRUE</inclusion><edgeMap><xCoord>1</xCoord><yCoord>1</yCoord></edgeMap></roi></unblindedReadNodule></readingSession></LidcReadMessage>"""
    path.write_text(template.format(subtlety=1, sop_uid="sop-a"), encoding="utf-8")
    first = parse_canonical_document(path, root).annotations[0].source_fingerprint
    path.write_text(template.format(subtlety=2, sop_uid="sop-a"), encoding="utf-8")
    second = parse_canonical_document(path, root).annotations[0].source_fingerprint
    path.write_text(template.format(subtlety=2, sop_uid="sop-b"), encoding="utf-8")
    third = parse_canonical_document(path, root).annotations[0].source_fingerprint
    assert first != second
    assert second != third


def test_missing_spatial_source_is_explicitly_excluded_from_matching(tmp_path: Path) -> None:
    root = tmp_path / "xml"
    path = root / "patient" / "source.xml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """<LidcReadMessage><ResponseHeader><StudyInstanceUID>study</StudyInstanceUID><SeriesInstanceUid>series</SeriesInstanceUid></ResponseHeader><readingSession><unblindedReadNodule><noduleID>id</noduleID><characteristics><subtlety>1</subtlety><internalStructure>1</internalStructure><calcification>1</calcification><sphericity>1</sphericity><margin>1</margin><lobulation>1</lobulation><spiculation>1</spiculation><texture>1</texture><malignancy>1</malignancy></characteristics></unblindedReadNodule></readingSession></LidcReadMessage>""",
        encoding="utf-8",
    )
    source = parse_canonical_document(path, root).annotations[0]
    usable, excluded = split_spatially_usable_sources([source])
    assert usable == []
    assert excluded == [source]


def test_repeated_nodule_id_requires_contour_match(monkeypatch: pytest.MonkeyPatch) -> None:
    annotation = _Annotation("shared", [_Contour(1.0, True, "2,4")])
    first, second = _source("a", nodule_id="shared", geometry="first"), _source("b", nodule_id="shared", geometry="second")
    monkeypatch.setattr("lidc_baseline.p2_cohort._pylidc_geometry_signature", lambda _: "second")
    assert match_source_annotation(annotation, [first, second]) == second
    monkeypatch.setattr("lidc_baseline.p2_cohort._pylidc_geometry_signature", lambda _: "missing")
    with pytest.raises(ValueError, match="SOURCE_ANNOTATION_MATCH_MISSING"):
        match_source_annotation(annotation, [first, second])


def test_repeated_nodule_id_rejects_ambiguous_contour_match(monkeypatch: pytest.MonkeyPatch) -> None:
    annotation = _Annotation("shared", [_Contour(1.0, True, "2,4")])
    first, second = _source("a", nodule_id="shared", geometry="same"), _source("b", nodule_id="shared", geometry="same")
    monkeypatch.setattr("lidc_baseline.p2_cohort._pylidc_geometry_signature", lambda _: "same")
    with pytest.raises(ValueError, match="SOURCE_ANNOTATION_MATCH_AMBIGUOUS"):
        match_source_annotation(annotation, [first, second])


def test_unique_nodule_id_does_not_require_contour_serialization_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    annotation = _Annotation("unique", [_Contour(1.0, True, "2,4")])
    source = _source("source", nodule_id="unique", geometry="xml-geometry")
    monkeypatch.setattr("lidc_baseline.p2_cohort._pylidc_geometry_signature", lambda _: "pylidc-geometry")
    assert match_source_annotation(annotation, [source]) == source


def test_stable_uid_is_sql_id_independent_and_source_sensitive() -> None:
    first = stable_nodule_uid("patient", "study", "series", "xml", ["fingerprint-a", "fingerprint-b"])
    reordered = stable_nodule_uid("patient", "study", "series", "xml", ["fingerprint-b", "fingerprint-a"])
    changed = stable_nodule_uid("patient", "study", "series", "xml", ["fingerprint-a", "fingerprint-c"])
    assert first == reordered
    assert first != changed


def test_annotation_to_cluster_mapping_is_deterministic_under_input_reordering() -> None:
    scan = _ManifestScan()
    first = _source("fingerprint-a", nodule_id="nodule-a")
    second = _source("fingerprint-b", nodule_id="nodule-b")
    first_annotation = _Annotation("nodule-a", [], annotation_id=20)
    second_annotation = _Annotation("nodule-b", [], annotation_id=10)
    expected = annotation_to_cluster_mapping_rows("uid", scan, "series", [first, second], [first_annotation, second_annotation])
    reordered = annotation_to_cluster_mapping_rows("uid", scan, "series", [second, first], [second_annotation, first_annotation])
    assert expected == reordered


def test_aggregate_cluster_uses_soft_votes_boundaries_and_strict_diameter() -> None:
    first = _source("a", characteristics={
        "malignancy": 2,
        "subtlety": 1,
        "sphericity": 5,
        "margin": 3,
        "lobulation": 3,
        "spiculation": 3,
        "texture": 3,
        "internalStructure": 1,
        "calcification": 1,
    })
    second = _source("b", characteristics={
        "malignancy": 2,
        "subtlety": 5,
        "sphericity": 1,
        "margin": 3,
        "lobulation": 3,
        "spiculation": 3,
        "texture": 3,
        "internalStructure": 2,
        "calcification": 2,
    })
    aggregate = aggregate_cluster([first, second], [3.0, 2.5])
    assert aggregate["malignancy_label"] == 0
    assert aggregate["subtlety_target"] == pytest.approx(0.5)
    assert aggregate["internalStructure_vote_distribution"] == [0.5, 0.5, 0.0, 0.0]
    assert aggregate["internalStructure_modal_tie"] is True
    assert aggregate["computed_strict_gt_3mm"] is False
    assert aggregate["all_required_targets_valid"] is True


@pytest.mark.parametrize(
    ("rating", "expected_label", "expected_status"),
    [(2, 0, "BENIGN"), (3, None, "UNCERTAIN"), (4, 1, "MALIGNANT")],
)
def test_malignancy_boundaries_are_aggregated_exactly(rating: int, expected_label: int | None, expected_status: str) -> None:
    values = _source("a").characteristics.copy()
    values["malignancy"] = rating
    aggregate = aggregate_cluster([_source("a", characteristics=values)], [3.1])
    assert aggregate["malignancy_label"] == expected_label
    assert aggregate["malignancy_status"] == expected_status


def test_aggregate_cluster_excludes_missing_required_target() -> None:
    values = _source("a").characteristics.copy()
    values["texture"] = None
    aggregate = aggregate_cluster([_source("a", characteristics=values)], [3.1])
    assert aggregate["all_required_targets_valid"] is False
    assert aggregate["missing_required_target_fields"] == ["texture"]
    assert aggregate["computed_strict_gt_3mm"] is True


def test_missing_categorical_target_is_null_and_non_null_soft_targets_sum_to_one() -> None:
    values = _source("a").characteristics.copy()
    values["internalStructure"] = None
    aggregate = aggregate_cluster([_source("a", characteristics=values)], [3.1])
    assert aggregate["internalStructure_vote_distribution"] is None
    assert aggregate["internalStructure_modal_tie"] is None
    assert aggregate["missing_required_target_fields"] == ["internalStructure"]
    assert sum(aggregate["calcification_vote_distribution"]) == pytest.approx(1.0)


def test_reader_count_policy_rejects_more_than_four_readers() -> None:
    assert cluster_has_supported_reader_count([object()]) is True
    assert cluster_has_supported_reader_count([object()] * 4) is True
    assert cluster_has_supported_reader_count([object()] * 5) is False


def test_p1_excluded_series_are_loaded_from_resolution_output(tmp_path: Path) -> None:
    (tmp_path / "duplicate_plane_resolution.json").write_text(json.dumps({"excluded_series_keys": ["series-a", "series-b"]}), encoding="utf-8")
    assert _load_excluded_series(tmp_path) == {"series-a", "series-b"}


def test_private_manifest_schema_and_soft_target_validation() -> None:
    source = _source("a")
    annotation = _Annotation("nodule-1", [_Contour(1.0, True, "2,4")])
    frame = pd.DataFrame([_private_manifest_row(_ManifestScan(), [source], [annotation], 1.8)])
    validate_private_manifest(frame)
    assert set(PRIVATE_MANIFEST_REQUIRED_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "clustering_initial_tolerance_mm"] == 2.0
    assert frame.loc[0, "clustering_effective_tolerance_mm"] == 1.8
    broken = frame.drop(columns=["annotation_class"])
    with pytest.raises(ValueError, match="PRIVATE_MANIFEST_REQUIRED_COLUMNS_MISSING"):
        validate_private_manifest(broken)
