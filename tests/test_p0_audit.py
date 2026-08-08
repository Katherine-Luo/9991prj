import hashlib
import json
from pathlib import Path

import pytest


AUDIT_DIRECTORY = Path("artifacts/audit/p0")
pytestmark = pytest.mark.local_audit


def load_report(name: str) -> dict[str, object]:
    return json.loads((AUDIT_DIRECTORY / name).read_text(encoding="utf-8"))


def test_frozen_config_digest_matches_resolved_bytes() -> None:
    resolved = Path("configs/baseline_v1.resolved.yaml")
    digest = Path("configs/baseline_v1.sha256").read_text(encoding="ascii").strip()

    assert hashlib.sha256(resolved.read_bytes()).hexdigest() == digest
    assert resolved.stat().st_mode & 0o222 == 0


def test_three_device_reports_pass_with_the_frozen_config() -> None:
    digest = Path("configs/baseline_v1.sha256").read_text(encoding="ascii").strip()
    reports = {device: load_report(f"{device}.json") for device in ("cpu", "mps", "cuda")}

    for device, report in reports.items():
        assert report["status"] == "PASS"
        assert report["device_requested"] == device
        assert report["config_sha256"] == digest
        assert report["input_shape"] == [1, 1, 64, 64, 64]
        assert report["target_shape"] == [1, 1]
        assert report["output_shape"] == [1, 1]
        assert report["loss_finite"] is True
        assert report["gradients_finite"] is True
        assert report["gradients_nonzero"] is True
        assert report["versions"]["pylidc"] == "0.2.3"
        assert report["versions"]["setuptools"] == "80.10.2"

    mps_versions = reports["mps"]["versions"]
    assert mps_versions["mps_cpu_fallback_enabled"] is True
    assert mps_versions["mps_fallback_operators"] == ["aten::max_pool3d_with_indices"]
    cuda_versions = reports["cuda"]["versions"]
    assert cuda_versions["cublas_workspace_config"] == ":4096:8"
    assert cuda_versions["cuda_runtime"] == "12.1"


def test_katana_storage_and_environment_gates_pass() -> None:
    storage = load_report("katana-storage-preflight.json")
    environment = load_report("katana-linux-environment.json")

    assert storage["status"] == "PASS"
    assert storage["remaining_space_pass"] is True
    assert storage["p0_workset_pass"] is True
    assert environment["status"] == "PASS"
    assert environment["pip_check"] == "PASS"
    assert environment["pylidc_import"] == "PASS"
    assert environment["setuptools"] == "80.10.2"
    assert environment["cuda_repeat_byte_identical"] is True
