from __future__ import annotations

import json
from pathlib import Path

import pytest

from lidc_baseline.p10_audit import (
    P10_CATALOGUE_PUBLIC_FILES,
    P10_DEVELOPMENT_FILES,
    _source_manifest_sha256,
    validate_git_candidates,
    verify_audit,
)


def test_generated_pdf_and_svg_git_attributes_are_whitespace_safe() -> None:
    assert ".gitattributes" in P10_DEVELOPMENT_FILES
    attributes = Path(".gitattributes").read_text(encoding="utf-8")
    assert "reports/baseline_v2/p10/public/**/*.pdf binary" in attributes
    assert "reports/baseline_v2/p10/public/**/*.svg binary" in attributes


def test_catalogue_git_whitelist_is_exact_and_excludes_private_outputs() -> None:
    assert "docs/results/results_catalogue_registry.json" in P10_CATALOGUE_PUBLIC_FILES
    assert "docs/results/catalogue_tables/CAT_Q_qualitative_cases.csv" in P10_CATALOGUE_PUBLIC_FILES
    assert "docs/results/catalogue_tables/CAT_T_gaps.csv" in P10_CATALOGUE_PUBLIC_FILES
    assert all(not path.endswith(".xlsx") for path in P10_CATALOGUE_PUBLIC_FILES)
    assert all("p10_private_report" not in path for path in P10_CATALOGUE_PUBLIC_FILES)


def test_source_manifest_is_order_invariant_and_hash_bound() -> None:
    first = {"b.json": "b" * 64, "a.json": "a" * 64}
    second = {"a.json": "a" * 64, "b.json": "b" * 64}
    assert _source_manifest_sha256(first) == _source_manifest_sha256(second)
    second["b.json"] = "c" * 64
    assert _source_manifest_sha256(first) != _source_manifest_sha256(second)


def test_git_candidate_gate_rejects_unexpected_and_oversized_files() -> None:
    allowed = {"reports/report.pdf", "src/lidc_baseline/p10_report.py"}
    assert validate_git_candidates(
        {"reports/report.pdf": 2000}, allowed, maximum_bytes=3000
    )["status"] == "PASS"
    with pytest.raises(ValueError, match="NOT_WHITELISTED"):
        validate_git_candidates({"private/checkpoint.pt": 1}, allowed)
    with pytest.raises(ValueError, match="OVERSIZED"):
        validate_git_candidates(
            {"reports/report.pdf": 3001}, allowed, maximum_bytes=3000
        )


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_tracked_archive_evidence_never_contains_private_file_list(tmp_path: Path) -> None:
    from lidc_baseline.p10_report import sha256_file

    report = {"status": "PASS"}
    archive = {
        "status": "PASS",
        "file_count": 10,
        "total_bytes": 100,
        "manifest_sha256": "a" * 64,
    }
    integrity = {"status": "PASS"}
    _write(tmp_path / "report.json", report)
    _write(tmp_path / "archive.json", archive)
    _write(tmp_path / "integrity.json", integrity)
    summary = {
        "status": "PASS",
        "reports": {
            "report": sha256_file(tmp_path / "report.json"),
            "archive": sha256_file(tmp_path / "archive.json"),
            "integrity": sha256_file(tmp_path / "integrity.json"),
        },
    }
    _write(tmp_path / "summary.json", summary)
    assert verify_audit(tmp_path)["status"] == "PASS"
    archive["files"] = [{"relative_path": "private/predictions.parquet"}]
    _write(tmp_path / "archive.json", archive)
    summary["reports"]["archive"] = sha256_file(tmp_path / "archive.json")
    _write(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match="TRACKED_PRIVATE_FILE_LIST_FORBIDDEN"):
        verify_audit(tmp_path)


def test_audit_hash_tamper_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "report.json", {"status": "PASS"})
    _write(tmp_path / "archive.json", {"status": "PASS"})
    _write(tmp_path / "integrity.json", {"status": "PASS"})
    _write(
        tmp_path / "summary.json",
        {
            "status": "PASS",
            "reports": {"report": "0" * 64, "archive": "0" * 64, "integrity": "0" * 64},
        },
    )
    with pytest.raises(ValueError, match="AUDIT_HASH_MISMATCH"):
        verify_audit(tmp_path)
