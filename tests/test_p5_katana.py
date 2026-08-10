from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lidc_baseline.audit import write_json
from lidc_baseline.p4_prepare import canonical_json_bytes, sha256_bytes, sha256_file
from lidc_baseline.p5_katana import (
    P5_DELTA_FILES,
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
        "execution_config_sha256": "execution",
        "p4_base_transfer_manifest_sha256": "base",
        "p4_base_transfer_manifest_file_sha256": "base-file",
        "total_files": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    return payload


def test_p5_delta_manifest_verifies_exact_files_and_hashes(tmp_path: Path) -> None:
    paths = ["src/lidc_baseline/p5_blackbox.py", "scripts/katana/p5_stage_a.pbs"]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    payload = _payload(tmp_path, paths)
    manifest = tmp_path / "artifacts/baseline_v2/manifests/p5_stage_a_transfer_manifest.json"
    write_json(manifest, payload)
    relative_manifest = Path("artifacts/baseline_v2/manifests/p5_stage_a_transfer_manifest.json")

    result = verify_transfer_manifest(tmp_path, relative_manifest)
    assert result["status"] == "PASS"
    assert result["verified_files"] == 2
    assert read_transfer_manifest(manifest) == payload
    assert transfer_file_list(tmp_path, relative_manifest) == sorted(paths + [relative_manifest.as_posix()])

    (tmp_path / paths[0]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SIZE_MISMATCH|HASH_MISMATCH"):
        verify_transfer_manifest(tmp_path, relative_manifest)


def test_p5_delta_manifest_rejects_unsafe_or_duplicate_paths(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"x")
    payload = _payload(tmp_path, ["file"])
    payload["files"][0]["relative_path"] = "../file"
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    manifest = tmp_path / "unsafe.json"
    write_json(manifest, payload)
    with pytest.raises(ValueError, match="UNSAFE_PATH"):
        verify_transfer_manifest(tmp_path, Path("unsafe.json"))

    payload = _payload(tmp_path, ["file"])
    payload["files"].append(dict(payload["files"][0]))
    payload["total_files"] = 2
    payload["total_bytes"] = 2
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    duplicate = tmp_path / "duplicate.json"
    write_json(duplicate, payload)
    with pytest.raises(ValueError, match="DUPLICATE_PATH"):
        verify_transfer_manifest(tmp_path, Path("duplicate.json"))


def test_p5_delta_is_code_only_and_contains_required_stage_a_files() -> None:
    assert set(P5_DELTA_FILES) == {
        "configs/experiments/baseline_v2_reference_training_h200.yaml",
        "configs/experiments/baseline_v2_reference_training_h200.resolved.yaml",
        "configs/experiments/baseline_v2_reference_training_h200.sha256",
        "scripts/katana/p5_fold.pbs",
        "scripts/katana/p5_stage_a.pbs",
        "src/lidc_baseline/p5_blackbox.py",
        "src/lidc_baseline/p5_katana.py",
    }
    forbidden = ("lidc_data", "DICOM", "XML", "runs/", "reports/", ".git", "rois/")
    assert not any(token in relative for relative in P5_DELTA_FILES for token in forbidden)


def test_p5_katana_scripts_are_valid_and_enforce_stage_gates() -> None:
    paths = (
        Path("scripts/katana/sync_p5_stage_a.sh"),
        Path("scripts/katana/p5_stage_a.pbs"),
        Path("scripts/katana/p5_fold.pbs"),
    )
    for path in paths:
        result = subprocess.run(["bash", "-n", str(path)], check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    stage = paths[1].read_text(encoding="utf-8")
    assert "gpu_model=H200" in stage
    assert "overfit-check" in stage
    assert "preflight" in stage
    assert "p5_blackbox train" not in stage
    formal = paths[2].read_text(encoding="utf-8")
    assert "gpu_model=H200" in formal
    assert "P5_STAGE_B_APPROVED" in formal
    assert "p5_blackbox train" in formal
    assert "evaluate-test" in formal
    assert "p5_blackbox verify" in formal
    assert "--resume" in formal
    assert "training_complete.json" in formal
    assert "test_evaluation.json" in formal
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert "lidc_data" not in content
        assert "DICOM" not in content


def test_sync_p5_uses_kdm_and_manifest_whitelist(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = source / "artifacts/baseline_v2/manifests/p5_stage_a_transfer_manifest.json"
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
    fake_ssh.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURED_SSH\"\n", encoding="utf-8")
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
        "'artifacts/baseline_v2/manifests/p5_stage_a_transfer_manifest.json' "
        "'configs/experiments/baseline_v2_reference_training_h200.yaml' "
        "'scripts/katana/p5_stage_a.pbs' "
        "'src/lidc_baseline/p5_blackbox.py'\n",
        encoding="utf-8",
    )
    for executable in (fake_ssh, fake_rsync, fake_python):
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/katana/sync_p5_stage_a.sh", str(source)],
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
    assert "z5448417@kdm.restech.unsw.edu.au" in captured_ssh.read_text(encoding="utf-8")
    rsync = captured_rsync.read_text(encoding="utf-8")
    assert "--relative" in rsync
    assert "--files-from=" in rsync
    transferred = captured_files.read_text(encoding="utf-8")
    assert "p5_blackbox.py" in transferred
    assert "p5_stage_a.pbs" in transferred
    assert "p5_stage_a_transfer_manifest.json" in transferred
    assert ".DS_Store" not in transferred
    assert "runs/" not in transferred
