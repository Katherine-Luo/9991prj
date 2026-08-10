from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, load_config
from lidc_baseline.p4_katana import (
    _manifest_digest,
    build_transfer_manifest,
    load_roi_sample,
    read_transfer_manifest,
    transfer_file_list,
    verify_transfer_manifest,
    write_aggregate_audit,
)
from lidc_baseline.p4_prepare import canonical_json_bytes, sha256_file


def _transfer_payload(root: Path, relative_paths: list[str]) -> dict:
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
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "roi_index_sha256": "index",
        "primary_nodules": 2,
        "primary_patients": 2,
        "roi_set_sha256": "roi-set",
        "total_files": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "categories": {"roi_files": 1},
        "files": entries,
    }
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    return payload


def test_transfer_manifest_verifies_each_file_and_detects_mutation(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src/module.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "roi.npz").write_bytes(b"roi")
    payload = _transfer_payload(tmp_path, ["src/module.py", "roi.npz"])
    write_json(tmp_path / "transfer.json", payload)

    verified = verify_transfer_manifest(tmp_path, Path("transfer.json"))
    assert verified["status"] == "PASS"
    assert verified["verified_files"] == 2
    assert len(verified["transfer_manifest_file_sha256"]) == 64
    assert read_transfer_manifest(tmp_path / "transfer.json") == payload
    assert transfer_file_list(tmp_path, Path("transfer.json")) == ["roi.npz", "src/module.py", "transfer.json"]

    (tmp_path / "roi.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="TRANSFER_FILE_SIZE_MISMATCH|TRANSFER_FILE_HASH_MISMATCH"):
        verify_transfer_manifest(tmp_path, Path("transfer.json"))


def test_transfer_manifest_rejects_hash_tampering_and_unsafe_paths(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"x")
    payload = _transfer_payload(tmp_path, ["file"])
    payload["total_bytes"] = 99
    write_json(tmp_path / "tampered.json", payload)
    with pytest.raises(ValueError, match="TRANSFER_MANIFEST_HASH_MISMATCH"):
        read_transfer_manifest(tmp_path / "tampered.json")

    payload = _transfer_payload(tmp_path, ["file"])
    payload["files"][0]["relative_path"] = "../file"
    payload["transfer_manifest_sha256"] = _manifest_digest(payload)
    write_json(tmp_path / "unsafe.json", payload)
    with pytest.raises(ValueError, match="TRANSFER_MANIFEST_UNSAFE_PATH"):
        verify_transfer_manifest(tmp_path, Path("unsafe.json"))


def test_build_transfer_manifest_is_deterministic_and_private(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = []
    for relative, content in (("src/a.py", b"a"), ("artifacts/baseline_v2/rois/private-uid.npz", b"b")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        files.append(path)
    config = copy.deepcopy(load_config("configs/baseline_v2.yaml"))
    monkeypatch.setattr("lidc_baseline.p4_katana.load_config", lambda _path: config)
    monkeypatch.setattr("lidc_baseline.p4_katana.transfer_files", lambda _root, _config: files)
    monkeypatch.setattr(
        "lidc_baseline.p4_katana.verify_p4",
        lambda *_args: {
            "primary_nodules": 2633,
            "primary_patients": 868,
            "roi_integrity": {"roi_set_sha256": "roi-set"},
        },
    )
    manifest = tmp_path / "manifest.parquet"
    roi_index = tmp_path / "roi_index.parquet"
    manifest.write_bytes(b"manifest")
    roi_index.write_bytes(b"index")
    output = Path("artifacts/baseline_v2/manifests/p4_transfer_manifest.json")

    first = build_transfer_manifest(tmp_path, Path("config.yaml"), Path("manifest.parquet"), Path("roi_index.parquet"), output)
    first_bytes = (tmp_path / output).read_bytes()
    second = build_transfer_manifest(tmp_path, Path("config.yaml"), Path("manifest.parquet"), Path("roi_index.parquet"), output)
    assert first == second
    assert (tmp_path / output).read_bytes() == first_bytes
    assert first["categories"]["roi_files"] == 1
    assert not any(Path(entry["relative_path"]).is_absolute() for entry in first["files"])


def test_real_roi_loader_enforces_shape_dtype_and_binary_mask(tmp_path: Path) -> None:
    path = tmp_path / "roi.npz"
    image = np.zeros((1, 64, 64, 64), dtype=np.float32)
    mask = np.zeros((1, 64, 64, 64), dtype=np.uint8)
    mask[:, 32, 32, 32] = 1
    np.savez(path, image=image, mask=mask)
    loaded_image, loaded_mask = load_roi_sample(path)
    assert loaded_image.dtype == np.float32
    assert loaded_mask.dtype == np.uint8

    np.savez(path, image=image.astype(np.float64), mask=mask)
    with pytest.raises(ValueError, match="IMAGE_INTERFACE"):
        load_roi_sample(path)
    np.savez(path, image=image, mask=np.zeros_like(mask))
    with pytest.raises(ValueError, match="MASK_VALUE"):
        load_roi_sample(path)


def test_sync_p4_uses_kdm_and_only_explicit_workset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = source / "artifacts/baseline_v2/manifests/p4_transfer_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    key = tmp_path / "key"
    key.write_text("test", encoding="utf-8")
    captured_ssh = tmp_path / "ssh.txt"
    captured_rsync = tmp_path / "rsync.txt"
    fake_ssh = tmp_path / "ssh"
    fake_rsync = tmp_path / "rsync"
    fake_python = tmp_path / "python"
    captured_files = tmp_path / "files.txt"
    fake_ssh.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURED_SSH\"\n", encoding="utf-8")
    fake_rsync.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURED_RSYNC\"\n"
        "for argument in \"$@\"; do\n"
        "  case \"$argument\" in --files-from=*) cp \"${argument#*=}\" \"$CAPTURED_FILES\";; esac\n"
        "done\n",
        encoding="utf-8",
    )
    fake_python.write_text(
        "#!/bin/bash\n"
        "printf '%s\\n' "
        "'artifacts/baseline_v2/encoder_initializations/fold_4.pt' "
        "'artifacts/baseline_v2/manifests/nodules.parquet' "
        "'artifacts/baseline_v2/manifests/p4_transfer_manifest.json' "
        "'artifacts/baseline_v2/rois/example.npz' "
        "'artifacts/baseline_v2/splits/fold_4.json' "
        "'src/lidc_baseline/p4_katana.py'\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    fake_rsync.chmod(0o755)
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/katana/sync_p4.sh", str(source)],
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
    ssh_arguments = captured_ssh.read_text(encoding="utf-8")
    rsync_arguments = captured_rsync.read_text(encoding="utf-8")
    transferred_files = captured_files.read_text(encoding="utf-8")
    assert "z5448417@kdm.restech.unsw.edu.au" in ssh_arguments
    assert "/srv/scratch/z5448417/lidc-baseline-v2" in ssh_arguments
    assert "--relative" in rsync_arguments
    assert "--files-from=" in rsync_arguments
    assert "--exclude=.DS_Store" in rsync_arguments
    assert "artifacts/baseline_v2/rois/example.npz" in transferred_files
    assert "artifacts/baseline_v2/manifests/nodules.parquet" in transferred_files
    assert "artifacts/baseline_v2/splits/fold_4.json" in transferred_files
    assert "artifacts/baseline_v2/encoder_initializations/fold_4.pt" in transferred_files
    assert "src/lidc_baseline/p4_katana.py" in transferred_files
    for forbidden in (".git", "lidc_data", "DICOM", "reports/", "runs/"):
        assert forbidden not in rsync_arguments + transferred_files
    assert ".DS_Store" not in transferred_files


def test_p4_katana_scripts_are_l40s_no_training_and_valid_bash() -> None:
    for path in (Path("scripts/katana/sync_p4.sh"), Path("scripts/katana/p4_cuda_smoke.pbs")):
        result = subprocess.run(["bash", "-n", str(path)], check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    script = Path("scripts/katana/p4_cuda_smoke.pbs").read_text(encoding="utf-8")
    assert "gpu_model=L40S" in script
    assert "lidc_baseline.p4_katana cuda-smoke" in script
    assert "optimizer" not in script
    assert "backward" not in script
    assert "lidc_data" not in script


def test_aggregate_audit_is_deidentified(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = copy.deepcopy(load_config("configs/baseline_v2.yaml"))
    monkeypatch.setattr("lidc_baseline.p4_katana.load_config", lambda _path: config)
    local = {
        "primary_nodules": 2,
        "primary_patients": 2,
        "roi_integrity": {"roi_files": 2, "roi_total_bytes": 10, "roi_set_sha256": "roi-set"},
        "folds": [{"fold_index": index} for index in range(5)],
    }
    monkeypatch.setattr("lidc_baseline.p4_katana.verify_p4", lambda *_args: local)
    split = {
        "split_sha256": "split",
        "partitions": {
            name: {"summary": {"nodules": 1, "patients": 1, "extremes": {"low": 1, "high": 1}}}
            for name in ("train", "validation", "test")
        },
    }
    monkeypatch.setattr("lidc_baseline.p4_katana.read_split", lambda _path: split)
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame({
        "nodule_uid": ["private-nodule-a", "private-nodule-b"],
        "patient_id": ["private-patient-a", "private-patient-b"],
        "series_instance_uid": ["1.2.3", "1.2.4"],
    }).to_parquet(manifest, index=False)
    roi_index = tmp_path / "roi_index.parquet"
    roi_index.write_bytes(b"index")
    remote = {
        "status": "PASS",
        "config_sha256": compute_config_sha256(config),
        "primary_nodules": 2,
        "primary_patients": 2,
        "roi_integrity": local["roi_integrity"],
        "transfer": {
            "transfer_manifest_sha256": "transfer",
            "transfer_manifest_file_sha256": "pending",
            "verified_files": 20,
            "verified_bytes": 30,
        },
        "runtime": {"gpu_name": "NVIDIA L40S", "pbs_job_id": "123"},
        "training_operations": {"optimizer_created": False, "backward_called": False, "parameter_update": False},
        "folds": [
            {
                "fold_index": index,
                "encoder_state_sha256": f"state-{index}",
                "encoder_file_sha256": f"file-{index}",
                "consumer_count": 4,
                "consumer_hashes_equal": True,
                "cuda_forward_finite": True,
            }
            for index in range(5)
        ],
    }
    remote_path = tmp_path / "remote.json"
    local_transfer = _transfer_payload(tmp_path, [])
    local_transfer["transfer_manifest_sha256"] = "transfer"
    local_transfer["transfer_manifest_sha256"] = _manifest_digest(local_transfer)
    transfer_path = tmp_path / "artifacts/baseline_v2/manifests/p4_transfer_manifest.json"
    write_json(transfer_path, local_transfer)
    remote["transfer"]["transfer_manifest_sha256"] = local_transfer["transfer_manifest_sha256"]
    remote["transfer"]["transfer_manifest_file_sha256"] = sha256_file(transfer_path)
    remote_path.write_bytes(canonical_json_bytes(remote))
    output = Path("audit")
    result = write_aggregate_audit(
        tmp_path,
        Path("config.yaml"),
        Path("manifest.parquet"),
        Path("roi_index.parquet"),
        Path("remote.json"),
        output,
    )
    assert result["status"] == "PASS"
    combined = "".join(path.read_text(encoding="utf-8") for path in sorted((tmp_path / output).iterdir()))
    for forbidden in ("private-nodule", "private-patient", "1.2.3", str(tmp_path)):
        assert forbidden not in combined


@pytest.mark.local_audit
def test_tracked_p4_audit_is_complete_and_deidentified() -> None:
    root = Path("artifacts/baseline_v2/audit/p4")
    expected = {
        "summary.json",
        "folds.csv",
        "initializations.csv",
        "katana_cuda.json",
        "katana_job.json",
    }
    assert {path.name for path in root.iterdir() if path.is_file()} == expected
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    job = json.loads((root / "katana_job.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["primary_nodules"] == 2633
    assert summary["primary_patients"] == 868
    assert summary["patient_leakage"] == 0
    assert summary["oof_nodule_coverage"] == 2633
    assert summary["katana_cuda_status"] == "PASS"
    assert summary["training_operations"] == {
        "backward_called": False,
        "optimizer_created": False,
        "parameter_update": False,
    }
    assert job["exit_status"] == 0
    assert job["requested_resources"]["gpu_model"] == "L40S"
    assert job["training_performed"] is False
    combined = "\n".join((root / name).read_text(encoding="utf-8") for name in sorted(expected))
    for forbidden in (
        "/Users/",
        "/private/",
        "/srv/scratch/",
        "LIDC-IDRI-",
        "nodule_uid",
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
    ):
        assert forbidden not in combined
