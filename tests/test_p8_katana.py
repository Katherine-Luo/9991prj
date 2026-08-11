from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lidc_baseline.audit import write_json
from lidc_baseline.p4_prepare import sha256_file
from lidc_baseline.p8_katana import (
    P8_DELTA_FILES,
    _manifest_digest,
    read_transfer_manifest,
    transfer_file_list,
    verify_transfer_manifest,
)


def _payload(root: Path, relative_paths: list[str]) -> dict:
    entries = [
        {
            "relative_path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(relative_paths)
    ]
    payload = {
        "schema_version": 1,
        "protocol_version": "Baseline-v2",
        "scientific_config_sha256": "scientific",
        "p8_execution_config_sha256": "p8-execution",
        "p7_base_transfer_manifest_sha256": "base",
        "p7_base_transfer_manifest_file_sha256": "base-file",
        "total_files": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    return payload


def _materialize(root: Path, paths: list[str]) -> None:
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def test_p8_delta_manifest_verifies_exact_files_and_hashes(tmp_path: Path) -> None:
    paths = list(P8_DELTA_FILES)
    _materialize(tmp_path, paths)
    payload = _payload(tmp_path, paths)
    relative_manifest = Path(
        "artifacts/baseline_v2/manifests/p8_stage_a_transfer_manifest.json"
    )
    manifest = tmp_path / relative_manifest
    write_json(manifest, payload)
    result = verify_transfer_manifest(tmp_path, relative_manifest)
    assert result["status"] == "PASS"
    assert result["verified_files"] == 10
    assert read_transfer_manifest(manifest) == payload
    assert transfer_file_list(tmp_path, relative_manifest) == sorted(
        paths + [relative_manifest.as_posix()]
    )
    (tmp_path / paths[0]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SIZE_MISMATCH|HASH_MISMATCH"):
        verify_transfer_manifest(tmp_path, relative_manifest)


def test_p8_delta_manifest_rejects_missing_extra_unsafe_and_duplicate(
    tmp_path: Path,
) -> None:
    paths = list(P8_DELTA_FILES)
    _materialize(tmp_path, paths)
    missing = _payload(tmp_path, paths[:-1])
    write_json(tmp_path / "missing.json", missing)
    with pytest.raises(ValueError, match="FILE_SET_MISMATCH"):
        verify_transfer_manifest(tmp_path, Path("missing.json"))
    extra_path = tmp_path / "unexpected.txt"
    extra_path.write_text("unexpected", encoding="utf-8")
    extra = _payload(tmp_path, paths + ["unexpected.txt"])
    write_json(tmp_path / "extra.json", extra)
    with pytest.raises(ValueError, match="FILE_SET_MISMATCH"):
        verify_transfer_manifest(tmp_path, Path("extra.json"))
    unsafe = _payload(tmp_path, paths)
    unsafe["files"][0]["relative_path"] = "../outside"
    unsafe["transfer_manifest_sha256"] = _manifest_digest(unsafe)
    write_json(tmp_path / "unsafe.json", unsafe)
    with pytest.raises(ValueError, match="UNSAFE_PATH"):
        verify_transfer_manifest(tmp_path, Path("unsafe.json"))
    duplicate = _payload(tmp_path, paths)
    duplicate["files"].append(dict(duplicate["files"][0]))
    duplicate["total_files"] += 1
    duplicate["total_bytes"] += int(duplicate["files"][0]["bytes"])
    duplicate["transfer_manifest_sha256"] = _manifest_digest(duplicate)
    write_json(tmp_path / "duplicate.json", duplicate)
    with pytest.raises(ValueError, match="DUPLICATE_PATH"):
        verify_transfer_manifest(tmp_path, Path("duplicate.json"))


def test_p8_delta_contains_only_p8_stage_formal_and_audit_files() -> None:
    assert set(P8_DELTA_FILES) == {
        "configs/experiments/baseline_v2_p8_gam_h200.yaml",
        "configs/experiments/baseline_v2_p8_gam_h200.resolved.yaml",
        "configs/experiments/baseline_v2_p8_gam_h200.sha256",
        "scripts/katana/p8_fold.pbs",
        "scripts/katana/p8_oof.pbs",
        "scripts/katana/p8_stage_a.pbs",
        "src/lidc_baseline/p8_audit.py",
        "src/lidc_baseline/p8_gam.py",
        "src/lidc_baseline/p8_gam_lifecycle.py",
        "src/lidc_baseline/p8_katana.py",
    }
    forbidden = ("lidc_data", "DICOM", "XML", "runs/", "reports/", ".git", "rois/")
    assert not any(
        token in relative for relative in P8_DELTA_FILES for token in forbidden
    )


def test_p8_scripts_enforce_stage_hardware_scope_and_formal_gate() -> None:
    paths = (
        Path("scripts/katana/sync_p8_stage_a.sh"),
        Path("scripts/katana/p8_stage_a.pbs"),
        Path("scripts/katana/p8_fold.pbs"),
        Path("scripts/katana/p8_oof.pbs"),
    )
    for path in paths:
        result = subprocess.run(
            ["bash", "-n", str(path)], check=False, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    stage = paths[1].read_text(encoding="utf-8")
    assert "gpu_model=H200" in stage
    assert "overfit-check" in stage and "preflight" in stage
    assert "p8_gam train" not in stage and "evaluate-test" not in stage
    formal = paths[2].read_text(encoding="utf-8")
    assert "gpu_model=H200" in formal
    assert "P8_FORMAL_APPROVED" in formal
    assert "p8_gam train" in formal and "--resume" in formal
    assert "evaluate-test" in formal and "p8_gam verify" in formal
    assert "test_transaction_sealed" in formal
    assert "TRAINING_COMPLETE_TEST_EVALUATED" in formal
    assert 'completion.get("test_transaction_count") == 1' in formal
    assert 'test_evaluation.json" ]]' not in formal
    oof = paths[3].read_text(encoding="utf-8")
    assert "p8_audit build-oof" in oof
    assert "p8_gam train" not in oof and "evaluate-test" not in oof
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert "lidc_data" not in content and "DICOM" not in content


def test_sync_p8_uses_kdm_and_manifest_whitelist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = source / (
        "artifacts/baseline_v2/manifests/p8_stage_a_transfer_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    key = tmp_path / "key"
    key.write_text("test", encoding="utf-8")
    captured_ssh = tmp_path / "ssh.txt"
    captured_rsync = tmp_path / "rsync.txt"
    captured_files = tmp_path / "files.txt"
    fake_ssh = tmp_path / "ssh"
    fake_rsync = tmp_path / "rsync"
    fake_python = tmp_path / "python"
    fake_ssh.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURED_SSH\"\n",
        encoding="utf-8",
    )
    fake_rsync.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURED_RSYNC\"\n"
        "for argument in \"$@\"; do\n"
        " case \"$argument\" in --files-from=*) cp \"${argument#*=}\" \"$CAPTURED_FILES\";; esac\n"
        "done\n",
        encoding="utf-8",
    )
    fake_python.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' "
        "'artifacts/baseline_v2/manifests/p8_stage_a_transfer_manifest.json' "
        "'configs/experiments/baseline_v2_p8_gam_h200.yaml' "
        "'scripts/katana/p8_stage_a.pbs' "
        "'src/lidc_baseline/p8_gam.py'\n",
        encoding="utf-8",
    )
    for executable in (fake_ssh, fake_rsync, fake_python):
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/katana/sync_p8_stage_a.sh", str(source)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "KATANA_SSH_KEY": str(key),
            "SSH_BIN": str(fake_ssh),
            "RSYNC_BIN": str(fake_rsync),
            "PYTHON_BIN": str(fake_python),
            "CAPTURED_SSH": str(captured_ssh),
            "CAPTURED_RSYNC": str(captured_rsync),
            "CAPTURED_FILES": str(captured_files),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "z5448417@kdm.restech.unsw.edu.au" in captured_ssh.read_text(
        encoding="utf-8"
    )
    assert "--relative" in captured_rsync.read_text(encoding="utf-8")
    transferred = captured_files.read_text(encoding="utf-8")
    assert "p8_gam.py" in transferred
    assert "p8_stage_a.pbs" in transferred
    assert "p8_stage_a_transfer_manifest.json" in transferred
    assert ".DS_Store" not in transferred and "runs/" not in transferred
