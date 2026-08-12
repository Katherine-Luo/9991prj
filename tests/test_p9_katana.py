from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lidc_baseline.audit import write_json
from lidc_baseline.p4_prepare import sha256_file
from lidc_baseline.p9_katana import (
    P9_DELTA_FILES,
    _manifest_digest,
    read_transfer_manifest,
    transfer_file_list,
    verify_transfer_manifest,
)


def _materialize(root: Path, paths: list[str]) -> None:
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def _payload(root: Path, paths: list[str]) -> dict:
    entries = [
        {
            "relative_path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(paths)
    ]
    payload = {
        "schema_version": 1,
        "protocol_version": "Baseline-v2",
        "scientific_config_sha256": "scientific",
        "p9_execution_config_sha256": "p9",
        "p8_base_transfer_manifest_sha256": "base",
        "p8_base_transfer_manifest_file_sha256": "base-file",
        "total_files": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "files": entries,
    }
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    return payload


def test_p9_delta_manifest_requires_exact_whitelist_and_hashes(tmp_path: Path) -> None:
    paths = list(P9_DELTA_FILES)
    _materialize(tmp_path, paths)
    relative = Path("artifacts/baseline_v2/manifests/p9_stage_a_transfer_manifest.json")
    write_json(tmp_path / relative, _payload(tmp_path, paths))
    result = verify_transfer_manifest(tmp_path, relative)
    assert result["status"] == "PASS"
    assert result["verified_files"] == len(paths) == 11
    assert read_transfer_manifest(tmp_path / relative)["total_files"] == 11
    assert transfer_file_list(tmp_path, relative) == sorted(
        paths + [relative.as_posix()]
    )
    (tmp_path / paths[0]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SIZE_MISMATCH|HASH_MISMATCH"):
        verify_transfer_manifest(tmp_path, relative)


def test_p9_delta_manifest_rejects_missing_extra_unsafe_and_duplicate(tmp_path: Path) -> None:
    paths = list(P9_DELTA_FILES)
    _materialize(tmp_path, paths)
    write_json(tmp_path / "missing.json", _payload(tmp_path, paths[:-1]))
    with pytest.raises(ValueError, match="FILE_SET_MISMATCH"):
        verify_transfer_manifest(tmp_path, Path("missing.json"))
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    write_json(tmp_path / "extra.json", _payload(tmp_path, paths + ["extra.txt"]))
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
    duplicate["total_bytes"] += duplicate["files"][0]["bytes"]
    duplicate["transfer_manifest_sha256"] = _manifest_digest(duplicate)
    write_json(tmp_path / "duplicate.json", duplicate)
    with pytest.raises(ValueError, match="DUPLICATE_PATH"):
        verify_transfer_manifest(tmp_path, Path("duplicate.json"))


def test_p9_delta_contains_only_p9_code_config_and_job_templates() -> None:
    assert set(P9_DELTA_FILES) == {
        "configs/experiments/baseline_v2_p9_evaluation_h200.yaml",
        "configs/experiments/baseline_v2_p9_evaluation_h200.resolved.yaml",
        "configs/experiments/baseline_v2_p9_evaluation_h200.sha256",
        "scripts/katana/p9_aggregate.pbs",
        "scripts/katana/p9_spatial.pbs",
        "scripts/katana/p9_stage_a.pbs",
        "src/lidc_baseline/p9_audit.py",
        "src/lidc_baseline/p9_evaluation.py",
        "src/lidc_baseline/p9_katana.py",
        "src/lidc_baseline/p9_spatial.py",
        "src/lidc_baseline/p9_spatial_lifecycle.py",
    }
    forbidden = ("lidc_data", "DICOM", "XML", "runs/", "reports/", ".git", "rois/")
    assert not any(
        token in relative for relative in P9_DELTA_FILES for token in forbidden
    )


def test_p9_scripts_enforce_h200_scope_approval_and_no_training_or_test() -> None:
    paths = (
        Path("scripts/katana/sync_p9_stage_a.sh"),
        Path("scripts/katana/p9_stage_a.pbs"),
        Path("scripts/katana/p9_spatial.pbs"),
        Path("scripts/katana/p9_aggregate.pbs"),
    )
    for path in paths:
        checked = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True, check=False
        )
        assert checked.returncode == 0, checked.stderr
    stage = paths[1].read_text(encoding="utf-8")
    assert "gpu_model=H200" in stage and "#PBS -q csegpu12" in stage
    assert "P9_SPATIAL_APPROVED=0" in stage and "p9_spatial preflight" in stage
    formal = paths[2].read_text(encoding="utf-8")
    assert "gpu_model=H200" in formal and "walltime=11:00:00" in formal
    assert "#PBS -q csegpu12" in formal
    assert '${P9_SPATIAL_APPROVED:-0}' in formal
    assert '!= "1"' in formal
    assert "p9_spatial run" in formal and "--resume" in formal
    aggregate = paths[3].read_text(encoding="utf-8")
    assert "p9_audit build" in aggregate and "p9_spatial verify" in aggregate
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert " train " not in content
        assert "evaluate-test" not in content
        assert "p10" not in content.lower()


def test_twenty_formal_model_fold_mappings_are_unique() -> None:
    mappings = [(model, fold) for model in (
        "blackbox", "standard_cbm", "mixed_cem", "learned_softmax_gam"
    ) for fold in range(5)]
    assert len(mappings) == len(set(mappings)) == 20


def test_sync_p9_uses_kdm_and_exact_manifest_list(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = source / "artifacts/baseline_v2/manifests/p9_stage_a_transfer_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    key = tmp_path / "key"
    key.write_text("key", encoding="utf-8")
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
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURED_RSYNC\"\n"
        "for argument in \"$@\"; do case \"$argument\" in --files-from=*) cp \"${argument#*=}\" \"$CAPTURED_FILES\";; esac; done\n",
        encoding="utf-8",
    )
    fake_python.write_text(
        "#!/bin/bash\nprintf '%s\\n' "
        "'artifacts/baseline_v2/manifests/p9_stage_a_transfer_manifest.json' "
        "'src/lidc_baseline/p9_audit.py' 'scripts/katana/p9_stage_a.pbs'\n",
        encoding="utf-8",
    )
    for executable in (fake_ssh, fake_rsync, fake_python):
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/katana/sync_p9_stage_a.sh", str(source)],
        capture_output=True,
        text=True,
        check=False,
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
    assert "z5448417@kdm.restech.unsw.edu.au" in captured_ssh.read_text()
    assert "--files-from=" in captured_rsync.read_text()
    transferred = captured_files.read_text()
    assert "p9_audit.py" in transferred
    assert "p9_stage_a.pbs" in transferred
    assert "p9_stage_a_transfer_manifest.json" in transferred
    assert "runs/" not in transferred
