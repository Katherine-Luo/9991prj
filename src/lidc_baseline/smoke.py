"""Cross-device DenseNet smoke test."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import platform
import random
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from lidc_baseline.audit import write_json
from lidc_baseline.config import compute_config_sha256, load_config

if TYPE_CHECKING:
    from torch import Tensor


class DeviceUnavailableError(RuntimeError):
    """Raised when the explicitly requested device is unavailable."""


def make_synthetic_batch(seed: int, target_value: float = 1.0) -> tuple[Tensor, Tensor]:
    """Load the registered synthetic fixture through a DataLoader."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.rand(
        (1, 1, 64, 64, 64),
        dtype=torch.float32,
        generator=generator,
    )
    target = torch.tensor([[target_value]], dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(image, target),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return next(iter(loader))


def configure_backend_environment(device_name: str) -> None:
    """Configure required backend behavior before PyTorch is imported."""
    if device_name == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def build_smoke_model(device_name: str) -> Any:
    """Build the registered DenseNet with a deterministic global mean pool."""
    import torch
    from monai.networks.nets import DenseNet121

    class DeterministicGlobalAveragePool3d(torch.nn.Module):
        def forward(self, inputs: Tensor) -> Tensor:
            return inputs.mean(dim=(2, 3, 4), keepdim=True)

    class DeterministicNonOverlappingAveragePool3d(torch.nn.Module):
        def forward(self, inputs: Tensor) -> Tensor:
            windows = inputs.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)
            return windows.mean(dim=(-3, -2, -1))

    class DeterministicPaddedMaxPool3d(torch.nn.Module):
        def forward(self, inputs: Tensor) -> Tensor:
            padded = torch.nn.functional.pad(
                inputs,
                (1, 1, 1, 1, 1, 1),
                mode="constant",
                value=float("-inf"),
            )
            windows = padded.unfold(2, 3, 2).unfold(3, 3, 2).unfold(4, 3, 2)
            return windows.amax(dim=(-3, -2, -1))

    model = DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
    )
    if device_name != "cuda":
        return model

    initial_pool = model.features.pool0
    if not isinstance(initial_pool, torch.nn.MaxPool3d):
        raise RuntimeError(f"Unexpected initial pool: {initial_pool!r}")
    if (
        initial_pool.kernel_size != 3
        or initial_pool.stride != 2
        or initial_pool.padding != 1
        or initial_pool.dilation != 1
        or initial_pool.ceil_mode
    ):
        raise RuntimeError(f"Unsupported initial pool settings: {initial_pool!r}")
    model.features.pool0 = DeterministicPaddedMaxPool3d()
    for transition_name in ("transition1", "transition2", "transition3"):
        transition = getattr(model.features, transition_name)
        pool = transition.pool
        if not isinstance(pool, torch.nn.AvgPool3d):
            raise RuntimeError(f"Unexpected {transition_name} pool: {pool!r}")
        if pool.kernel_size != 2 or pool.stride != 2 or pool.padding != 0:
            raise RuntimeError(f"Unsupported {transition_name} pool settings: {pool!r}")
        transition.pool = DeterministicNonOverlappingAveragePool3d()
    model.class_layers.pool = DeterministicGlobalAveragePool3d()
    return model


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for the smoke execution."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(device_name: str) -> Any:
    """Resolve an explicitly requested device without fallback."""
    import torch

    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise DeviceUnavailableError("Requested mps device is unavailable")
        return torch.device("mps")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise DeviceUnavailableError("Requested cuda device is unavailable")
        return torch.device("cuda:0")
    raise ValueError(f"Unsupported device: {device_name}")


def _cuda_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.strip().splitlines()
    return first_line[0].strip() if first_line else None


def environment_versions(device: Any) -> dict[str, Any]:
    """Collect the required environment versions for an audit report."""
    import torch

    is_cuda = device.type == "cuda"
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "monai": importlib.metadata.version("monai"),
        "pylidc": importlib.metadata.version("pylidc"),
        "setuptools": importlib.metadata.version("setuptools"),
        "cuda_runtime": torch.version.cuda if is_cuda else None,
        "cuda_driver": _cuda_driver_version() if is_cuda else None,
        "gpu_name": torch.cuda.get_device_name(device) if is_cuda else None,
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if is_cuda else None
        ),
        "mps_cpu_fallback_enabled": (
            device.type == "mps"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
        ),
        "mps_fallback_operators": (
            ["aten::max_pool3d_with_indices"]
            if device.type == "mps"
            and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
            else []
        ),
    }


def run_smoke(
    device_name: str,
    output_path: str | Path,
    config_path: str | Path = Path("configs/baseline_v1.yaml"),
) -> dict[str, Any]:
    """Run a real 3D DenseNet forward/backward smoke test."""
    configure_backend_environment(device_name)
    import torch

    config = load_config(config_path)
    seed = int(config["reproducibility"]["base_seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    is_regression = config.get("task", {}).get("type") == "continuous_regression"
    target_value = 0.5 if is_regression else 1.0
    image, target = make_synthetic_batch(seed, target_value=target_value)
    image = image.to(device)
    target = target.to(device)

    model = build_smoke_model(device_name).to(device)
    model.train()
    output = model(image)
    if is_regression:
        head = config["task"]["head"]
        if head != {
            "type": "linear",
            "output_activation": "none",
            "output_constraint": "unbounded",
            "clipping": False,
        }:
            raise ValueError(f"Unsupported Baseline-v2 task head: {head!r}")
        loss_function = torch.nn.MSELoss()
        loss_name = "mse"
    else:
        loss_function = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.ones(1, dtype=torch.float32, device=device)
        )
        loss_name = "weighted_bce"
    loss = loss_function(output, target)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    loss_value = float(loss.detach().cpu().item())
    gradients_finite = bool(
        gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients)
    )
    gradients_nonzero = bool(
        gradients and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": config["protocol"]["version"],
        "status": "PASS",
        "device_requested": device_name,
        "device_resolved": str(device),
        "seed": seed,
        "input_shape": list(image.shape),
        "target_shape": list(target.shape),
        "output_shape": list(output.shape),
        "output_values": [float(value) for value in output.detach().cpu().reshape(-1)],
        "loss_name": loss_name,
        "task_type": config.get("task", {}).get("type", "binary_classification"),
        "task_output_activation": config.get("task", {}).get("head", {}).get(
            "output_activation", "none"
        ),
        "task_output_constraint": config.get("task", {}).get("head", {}).get(
            "output_constraint", "unbounded"
        ),
        "task_output_clipping": config.get("task", {}).get("head", {}).get(
            "clipping", False
        ),
        "post_output_transform_applied": False,
        "loss": loss_value,
        "loss_finite": math.isfinite(loss_value),
        "gradients_finite": gradients_finite,
        "gradients_nonzero": gradients_nonzero,
        "config_sha256": compute_config_sha256(config),
        "versions": environment_versions(device),
    }
    if not report["loss_finite"] or not gradients_finite or not gradients_nonzero:
        raise RuntimeError("Smoke test produced non-finite or zero gradients")
    write_json(Path(output_path), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_v1.yaml"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke command-line interface."""
    arguments = _parser().parse_args(argv)
    try:
        run_smoke(arguments.device, arguments.output, arguments.config)
    except DeviceUnavailableError as error:
        print(str(error), file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
