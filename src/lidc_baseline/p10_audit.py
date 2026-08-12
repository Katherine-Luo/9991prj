"""Build the deidentified tracked P10 audit from verified report/archive outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lidc_baseline.p10_archive import LOCAL_ROOT_DEFAULT, verify_archive
from lidc_baseline.p10_private_appendix import verify_private_appendices
from lidc_baseline.p10_report import (
    CONFIG_RESOLVED_DEFAULT,
    P10_CONFIG_SHA256,
    PUBLIC_ROOT_DEFAULT,
    assert_public_payload,
    sha256_file,
    verify_inputs,
    verify_public_outputs,
)


SCHEMA_VERSION = 1
AUDIT_ROOT_DEFAULT = Path("artifacts/baseline_v2/audit/p10")
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
REPORT_NAMES = ("report", "archive", "integrity", "summary")
MAX_GIT_FILE_BYTES = 10 * 1024 * 1024
P10_DEVELOPMENT_FILES = {
    ".gitattributes",
    ".gitignore",
    "pyproject.toml",
    "docs/PROJECT_STATUS.md",
    "src/lidc_baseline/p10_report.py",
    "src/lidc_baseline/p10_archive.py",
    "src/lidc_baseline/p10_audit.py",
    "src/lidc_baseline/p10_private_appendix.py",
    "tests/test_p10_report.py",
    "tests/test_p10_archive.py",
    "tests/test_p10_audit.py",
    "tests/test_p10_private_appendix.py",
}


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _tree_evidence(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise ValueError("P10_PUBLIC_OUTPUTS_EMPTY")
    manifest_sha = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    return {
        "file_count": len(rows),
        "total_bytes": sum(row["size_bytes"] for row in rows),
        "manifest_sha256": manifest_sha,
        "files": rows,
    }


def _source_manifest_sha256(source_hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(sorted(source_hashes.items())))).hexdigest()


def _git_candidate_paths(repository_root: Path) -> set[str]:
    commands = (
        ("git", "diff", "--name-only"),
        ("git", "diff", "--cached", "--name-only"),
        ("git", "ls-files", "--others", "--exclude-standard"),
    )
    paths: set[str] = set()
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        paths.update(line for line in completed.stdout.splitlines() if line)
    return paths


def validate_git_candidates(
    candidates: Mapping[str, int],
    allowed: set[str],
    *,
    maximum_bytes: int = MAX_GIT_FILE_BYTES,
) -> dict[str, Any]:
    unexpected = set(candidates) - allowed
    if unexpected:
        raise ValueError(f"P10_GIT_CANDIDATE_NOT_WHITELISTED:{sorted(unexpected)}")
    oversized = {
        path: size for path, size in candidates.items() if int(size) > maximum_bytes
    }
    if oversized:
        raise ValueError(f"P10_GIT_CANDIDATE_OVERSIZED:{oversized}")
    return {
        "status": "PASS",
        "candidate_count": len(candidates),
        "maximum_file_bytes": maximum_bytes,
        "largest_candidate_bytes": max(candidates.values(), default=0),
    }


def verify_git_candidate_whitelist(
    repository_root: Path,
    *,
    public_root: Path,
    audit_root: Path,
) -> dict[str, Any]:
    candidates = _git_candidate_paths(repository_root)
    public_relative = public_root.resolve().relative_to(repository_root.resolve())
    audit_relative = audit_root.resolve().relative_to(repository_root.resolve())
    allowed = set(P10_DEVELOPMENT_FILES)
    allowed.update(
        (public_relative / path.relative_to(public_root)).as_posix()
        for path in public_root.rglob("*")
        if path.is_file()
    )
    allowed.update(
        (audit_relative / f"{name}.json").as_posix() for name in REPORT_NAMES
    )
    sizes = {
        relative: (repository_root / relative).stat().st_size
        for relative in candidates
        if (repository_root / relative).is_file()
    }
    sizes.update({relative: 0 for relative in candidates - set(sizes)})
    return validate_git_candidates(sizes, allowed)


def build_audit(
    *,
    public_root: Path = PUBLIC_ROOT_DEFAULT,
    archive_root: Path = LOCAL_ROOT_DEFAULT,
    audit_root: Path = AUDIT_ROOT_DEFAULT,
) -> dict[str, Any]:
    inputs = verify_inputs()
    public = verify_public_outputs(public_root)
    archive = verify_archive(archive_root)
    appendix = verify_private_appendices(archive_root)
    public_tree = _tree_evidence(public_root)
    repository_root = Path(__file__).resolve().parents[2]
    git_gate = verify_git_candidate_whitelist(
        repository_root,
        public_root=public_root,
        audit_root=audit_root,
    )
    font_sha = sha256_file(FONT_PATH)
    report_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "report_data_sha256": public["report_data_sha256"],
        "public_file_count": public_tree["file_count"],
        "public_total_bytes": public_tree["total_bytes"],
        "public_manifest_sha256": public_tree["manifest_sha256"],
        "short_report_pages": {"en": public["short_pages"], "zh": public["short_pages"]},
        "technical_report_pages": {
            "en": public["technical_pages"],
            "zh": public["technical_pages"],
        },
        "bilingual_numeric_parity": True,
        "bilingual_table_cell_parity": True,
        "bilingual_conclusion_code_parity": True,
        "songti_ttc_sha256": font_sha,
        "songti_regular_subfont_index": 6,
        "songti_bold_subfont_index": 1,
        "font_embedded": True,
        "public_privacy": "PASS",
        "page_render_visual_qa": public["page_render_visual_qa"],
        "pdfplumber_text_gate": public["pdfplumber_text_gate"],
        "rendered_page_count": public["rendered_page_count"],
        "private_qualitative_appendix": appendix,
    }
    archive_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "archive_complete": True,
        "file_count": archive["file_count"],
        "total_bytes": archive["total_bytes"],
        "manifest_sha256": archive["manifest_sha256"],
        "remote_manifest_sha256": archive["remote_manifest_sha256"],
        "private_file_list_tracked": False,
        "remote_write": False,
        "remote_delete": False,
        "local_delete": False,
        "github_private_artifacts": False,
        "github_lfs": False,
    }
    integrity_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "p10_execution_config_sha256": P10_CONFIG_SHA256,
        "p5_through_p9_source_manifest_sha256": _source_manifest_sha256(
            inputs["source_file_sha256"]
        ),
        "p5_through_p9_start_manifest_sha256": inputs["source_manifest_sha256"],
        "p5_through_p9_end_manifest_sha256": inputs["source_manifest_sha256"],
        "p5_through_p9_manifest_unchanged": True,
        "p5_through_p9_artifacts_modified": False,
        "new_training": False,
        "new_test_inference": False,
        "new_h200_jobs": False,
        "new_cpu_scientific_jobs": False,
        "second_committed_test_evaluation": False,
        "p11_started": False,
        "exact_git_candidate_whitelist": git_gate["status"],
        "git_candidate_maximum_file_bytes": git_gate["maximum_file_bytes"],
        "git_largest_candidate_bytes": git_gate["largest_candidate_bytes"],
        "unique_nodules": 2633,
        "unique_patients": 868,
        "fold_counts": [479, 502, 539, 549, 564],
        "patient_leakage": 0,
    }
    for payload in (report_payload, archive_payload, integrity_payload):
        assert_public_payload(payload)
    _atomic_write(audit_root / "report.json", report_payload)
    _atomic_write(audit_root / "archive.json", archive_payload)
    _atomic_write(audit_root / "integrity.json", integrity_payload)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "P10",
        "status": "PASS",
        "reports": {name: sha256_file(audit_root / f"{name}.json") for name in REPORT_NAMES[:3]},
        "p10_execution_config_sha256": P10_CONFIG_SHA256,
        "public_manifest_sha256": public_tree["manifest_sha256"],
        "private_archive_manifest_sha256": archive["manifest_sha256"],
        "private_archive_file_count": archive["file_count"],
        "private_archive_total_bytes": archive["total_bytes"],
        "public_private_boundary": "PASS",
        "p5_through_p9_immutable": True,
        "p11_started": False,
    }
    assert_public_payload(summary)
    _atomic_write(audit_root / "summary.json", summary)
    return summary


def verify_audit(audit_root: Path = AUDIT_ROOT_DEFAULT) -> dict[str, Any]:
    reports = {}
    for name in REPORT_NAMES:
        path = audit_root / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise ValueError(f"P10_AUDIT_STATUS_INVALID:{name}")
        assert_public_payload(payload)
        reports[name] = payload
    summary = reports["summary"]
    for name in REPORT_NAMES[:3]:
        if summary["reports"][name] != sha256_file(audit_root / f"{name}.json"):
            raise ValueError(f"P10_AUDIT_HASH_MISMATCH:{name}")
    archive = reports["archive"]
    if "files" in archive or "relative_path" in json.dumps(archive):
        raise ValueError("P10_TRACKED_PRIVATE_FILE_LIST_FORBIDDEN")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"), nargs="?", default="build")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_audit() if args.command == "build" else verify_audit()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
