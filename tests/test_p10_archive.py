from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lidc_baseline.p10_archive import (
    COMPLETE_NAME,
    LOCAL_ROOT_DEFAULT,
    MANIFEST_NAME,
    WHITELIST,
    build_local_manifest,
    check_free_space,
    manifest_sha256,
    rsync_command,
    verify_archive,
    write_archive_completion,
)


def _archive_tree(root: Path) -> list[dict[str, object]]:
    for index, directory in enumerate(WHITELIST):
        path = root / directory / "nested" / f"file-{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]) * (index + 1))
    return build_local_manifest(root)


def test_rsync_is_resumable_read_only_and_never_deletes() -> None:
    command = rsync_command("p9")
    assert "--partial" in command
    assert "--append" in command
    assert not any("delete" in argument for argument in command)
    assert command[-2].endswith("/runs/baseline_v2/p9/")
    assert command[-1].endswith("/baseline_v2/p9/")


def test_rsync_rejects_non_whitelisted_scope() -> None:
    with pytest.raises(ValueError, match="DIRECTORY_NOT_WHITELISTED"):
        rsync_command("rois")


def test_free_space_gate_requires_120_percent() -> None:
    usage = lambda _path: SimpleNamespace(total=500, used=380, free=120)
    report = check_free_space(Path("/private/tmp/archive"), 100, disk_usage=usage)
    assert report["required_free_bytes"] == 120
    assert report["observed_ratio"] == 1.2
    with pytest.raises(ValueError, match="INSUFFICIENT_SPACE"):
        check_free_space(
            Path("/private/tmp/archive"),
            101,
            disk_usage=usage,
        )


def test_archive_completion_is_atomic_and_file_level_bound(tmp_path: Path) -> None:
    rows = _archive_tree(tmp_path)
    completion = write_archive_completion(
        tmp_path,
        rows,
        remote_manifest_sha256=manifest_sha256(rows),
    )
    assert completion["file_count"] == 5
    assert (tmp_path / COMPLETE_NAME).stat().st_mode & 0o777 == 0o600
    assert (tmp_path / MANIFEST_NAME).stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".*.tmp"))


def test_archive_rejects_remote_local_hash_mismatch(tmp_path: Path) -> None:
    rows = _archive_tree(tmp_path)
    with pytest.raises(ValueError, match="REMOTE_LOCAL_MANIFEST_MISMATCH"):
        write_archive_completion(tmp_path, rows, remote_manifest_sha256="0" * 64)


def test_archive_verify_missing_extra_and_tampered_files(tmp_path: Path) -> None:
    rows = _archive_tree(tmp_path)
    write_archive_completion(tmp_path, rows, remote_manifest_sha256=manifest_sha256(rows))
    # The production root is intentionally fixed.  Rebind only this pure verifier test.
    import lidc_baseline.p10_archive as archive

    original = archive.validate_archive_roots
    archive.validate_archive_roots = lambda *_args, **_kwargs: None
    try:
        assert verify_archive(tmp_path)["status"] == "ARCHIVE_COMPLETE"
        target = tmp_path / rows[0]["relative_path"]
        target.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="TAMPER_OR_COVERAGE_MISMATCH"):
            verify_archive(tmp_path)
        target.write_bytes(b"\x00")
        (tmp_path / "unexpected").mkdir()
        with pytest.raises(ValueError, match="EXTRA_TOP_LEVEL"):
            verify_archive(tmp_path)
    finally:
        archive.validate_archive_roots = original


def test_archive_manifest_contains_no_absolute_paths(tmp_path: Path) -> None:
    rows = _archive_tree(tmp_path)
    text = json.dumps(rows)
    assert str(tmp_path) not in text
    assert all(not str(row["relative_path"]).startswith("/") for row in rows)


@pytest.mark.parametrize(
    "relative_path",
    (
        "p9/raw_scan.dcm",
        "blackbox/source.xml",
        "gam/.git/config",
    ),
)
def test_archive_manifest_rejects_raw_inputs_and_git_metadata(
    tmp_path: Path, relative_path: str
) -> None:
    for directory in WHITELIST:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    forbidden = tmp_path / relative_path
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_bytes(b"private")
    with pytest.raises(ValueError, match="FORBIDDEN_CONTENT"):
        build_local_manifest(tmp_path)
