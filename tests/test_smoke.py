import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from packaging.version import Version


def test_smoke_module_is_importable() -> None:
    assert importlib.util.find_spec("lidc_baseline.smoke") is not None


def require_callable(name: str) -> Callable[..., object]:
    module = importlib.import_module("lidc_baseline.smoke")
    value = getattr(module, name, None)
    assert callable(value), f"lidc_baseline.smoke.{name} must be callable"
    return value


def test_synthetic_batch_is_reproducible_with_registered_shapes() -> None:
    make_synthetic_batch = require_callable("make_synthetic_batch")

    first_image, first_target = make_synthetic_batch(20260808)
    second_image, second_target = make_synthetic_batch(20260808)

    assert first_image.shape == (1, 1, 64, 64, 64)
    assert first_target.shape == (1, 1)
    assert first_image.dtype == torch.float32
    assert first_target.dtype == torch.float32
    assert torch.equal(first_image, second_image)
    assert torch.equal(first_target, second_target)
    assert first_image.min().item() >= 0.0
    assert first_image.max().item() <= 1.0


def test_synthetic_batch_changes_when_seed_changes() -> None:
    make_synthetic_batch = require_callable("make_synthetic_batch")

    first_image, _ = make_synthetic_batch(20260808)
    second_image, _ = make_synthetic_batch(20260809)

    assert not torch.equal(first_image, second_image)


def test_smoke_model_uses_deterministic_global_mean_pool() -> None:
    build_smoke_model = require_callable("build_smoke_model")

    model = build_smoke_model("cuda")
    pool = model.class_layers.pool
    fixture = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)

    assert type(model).__name__ == "DenseNet121"
    assert type(pool).__name__ == "DeterministicGlobalAveragePool3d"
    assert torch.equal(pool(fixture), fixture.mean(dim=(2, 3, 4), keepdim=True))


def test_smoke_model_replaces_transition_avg_pool_with_equivalent_mean() -> None:
    build_smoke_model = require_callable("build_smoke_model")

    model = build_smoke_model("cuda")
    pool = model.features.transition1.pool
    fixture = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 1, 2, 4, 6)
    expected = torch.nn.AvgPool3d(kernel_size=2, stride=2)(fixture)

    assert not any(isinstance(module, torch.nn.AvgPool3d) for module in model.modules())
    assert type(pool).__name__ == "DeterministicNonOverlappingAveragePool3d"
    assert torch.equal(pool(fixture), expected)


def test_smoke_model_replaces_initial_max_pool_with_equivalent_unfold() -> None:
    build_smoke_model = require_callable("build_smoke_model")

    model = build_smoke_model("cuda")
    pool = model.features.pool0
    fixture = torch.arange(5 * 6 * 7, dtype=torch.float32).reshape(1, 1, 5, 6, 7)
    expected = torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1)(fixture)

    assert not any(isinstance(module, torch.nn.MaxPool3d) for module in model.modules())
    assert type(pool).__name__ == "DeterministicPaddedMaxPool3d"
    assert torch.equal(pool(fixture), expected)


def test_smoke_model_preserves_monai_pooling_outside_cuda() -> None:
    build_smoke_model = require_callable("build_smoke_model")

    model = build_smoke_model("mps")

    assert isinstance(model.features.pool0, torch.nn.MaxPool3d)
    assert isinstance(model.features.transition1.pool, torch.nn.AvgPool3d)
    assert isinstance(model.class_layers.pool, torch.nn.AdaptiveAvgPool3d)


def test_unavailable_accelerator_raises_without_cpu_fallback(monkeypatch) -> None:
    resolve_device = require_callable("resolve_device")
    module = importlib.import_module("lidc_baseline.smoke")
    error_type = getattr(module, "DeviceUnavailableError", RuntimeError)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(error_type, match="cuda"):
        resolve_device("cuda")


def test_mps_resolution_requires_built_and_available_backend(monkeypatch) -> None:
    resolve_device = require_callable("resolve_device")
    module = importlib.import_module("lidc_baseline.smoke")
    error_type = getattr(module, "DeviceUnavailableError", RuntimeError)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(error_type, match="mps"):
        resolve_device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("mps") == torch.device("mps")


def test_environment_versions_include_required_packages() -> None:
    environment_versions = require_callable("environment_versions")

    versions = environment_versions(torch.device("cpu"))

    assert versions["python"].startswith("3.11.")
    assert Version(versions["torch"]).base_version == "2.5.1"
    assert versions["monai"] == "1.4.0"
    assert versions["pylidc"] == "0.2.3"
    assert versions["setuptools"] == "80.10.2"
    assert versions["cuda_runtime"] is None
    assert versions["cuda_driver"] is None
    assert versions["gpu_name"] is None
    assert versions["cublas_workspace_config"] is None


def test_mps_environment_records_no_cpu_operator_fallback() -> None:
    environment_versions = require_callable("environment_versions")

    versions = environment_versions(torch.device("mps"))

    assert versions["mps_cpu_fallback_enabled"] is False
    assert versions["mps_fallback_operators"] == []


def test_mps_backend_configuration_enables_only_registered_fallback(monkeypatch) -> None:
    configure_backend_environment = require_callable("configure_backend_environment")
    environment_versions = require_callable("environment_versions")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    configure_backend_environment("mps")
    versions = environment_versions(torch.device("mps"))

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert versions["mps_cpu_fallback_enabled"] is True
    assert versions["mps_fallback_operators"] == ["aten::max_pool3d_with_indices"]


def test_smoke_module_sets_backend_environment_before_torch_execution() -> None:
    environment = dict(os.environ)
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    environment.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; import lidc_baseline.smoke; "
                "print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')); "
                "print(os.environ.get('CUBLAS_WORKSPACE_CONFIG'))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines() == ["None", ":4096:8"]


def test_cli_rejects_raw_data_argument_without_running_model(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lidc_baseline.smoke",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "report.json"),
            "--data-root",
            "/Users/katherine/Desktop/lidc_data",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --data-root" in result.stderr


@pytest.mark.integration
def test_cpu_smoke_runs_real_densenet_and_writes_audit_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_smoke = require_callable("run_smoke")
    output = tmp_path / "cpu.json"
    raw_root = Path("/Users/katherine/Desktop/lidc_data").resolve()
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        resolved = path.resolve()
        if resolved == raw_root or raw_root in resolved.parents:
            raise AssertionError(f"Smoke test accessed raw data: {resolved}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    report = run_smoke(
        device_name="cpu",
        output_path=output,
        config_path=Path("configs/baseline_v1.yaml"),
    )

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert report == stored
    assert stored["status"] == "PASS"
    assert stored["device_requested"] == "cpu"
    assert stored["device_resolved"] == "cpu"
    assert stored["seed"] == 20260808
    assert stored["input_shape"] == [1, 1, 64, 64, 64]
    assert stored["target_shape"] == [1, 1]
    assert stored["output_shape"] == [1, 1]
    assert stored["loss_finite"] is True
    assert stored["gradients_finite"] is True
    assert stored["gradients_nonzero"] is True
    assert len(stored["config_sha256"]) == 64
