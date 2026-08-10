import json
import os
import subprocess
import sys
from pathlib import Path


def test_preflight_reports_pass_and_blocked_storage_states(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    workset = scratch / "lidc-baseline-p0"
    workset.mkdir(parents=True)
    (workset / "fixture.bin").write_bytes(b"x" * 128)
    output = tmp_path / "preflight.json"
    command = [
        sys.executable,
        "-m",
        "lidc_baseline.katana",
        "preflight",
        "--scratch",
        str(scratch),
        "--quota-bytes",
        "1024",
        "--min-remaining-bytes",
        "512",
        "--max-workset-bytes",
        "256",
        "--workset",
        str(workset),
        "--output",
        str(output),
    ]

    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    passed_report = json.loads(output.read_text(encoding="utf-8"))

    assert passed.returncode == 0, passed.stderr
    assert passed_report["status"] == "PASS"
    assert passed_report["scratch_used_bytes"] == 128
    assert passed_report["scratch_remaining_bytes"] == 896
    assert passed_report["p0_workset_bytes"] == 128

    (scratch / "other.bin").write_bytes(b"y" * 600)
    blocked = subprocess.run(command, check=False, capture_output=True, text=True)
    blocked_report = json.loads(output.read_text(encoding="utf-8"))

    assert blocked.returncode == 1
    assert blocked_report["status"] == "BLOCKED"
    assert blocked_report["remaining_space_pass"] is False
    assert blocked_report["p0_workset_pass"] is True


def test_sync_script_invokes_kdm_rsync_with_required_exclusions(tmp_path: Path) -> None:
    fake_rsync = tmp_path / "fake-rsync"
    captured = tmp_path / "captured.txt"
    key = tmp_path / "katana-key"
    source = tmp_path / "source"
    source.mkdir()
    key.write_text("test-only-key", encoding="utf-8")
    fake_rsync.write_text(
        "#!/bin/bash\nprintf '%s\\n' \"$@\" > \"$CAPTURED_ARGS\"\n",
        encoding="utf-8",
    )
    fake_rsync.chmod(0o755)
    environment = {
        **os.environ,
        "RSYNC_BIN": str(fake_rsync),
        "CAPTURED_ARGS": str(captured),
        "KATANA_SSH_KEY": str(key),
    }

    result = subprocess.run(
        ["bash", "scripts/katana/sync_code.sh", str(source)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    arguments = captured.read_text(encoding="utf-8").splitlines()

    assert result.returncode == 0, result.stderr
    assert "-avhP" in arguments
    assert "--exclude=.git" in arguments
    assert "--exclude=.DS_Store" in arguments
    assert "--exclude=.pytest_cache" in arguments
    assert "--exclude=__pycache__" in arguments
    assert "--exclude=*.egg-info" in arguments
    assert "--exclude=artifacts" in arguments
    assert "--exclude=runs" in arguments
    assert "--exclude=reports/baseline_v1" in arguments
    assert "--exclude=reports/baseline_v2" in arguments
    assert "--exclude=lidc_data" in arguments
    assert arguments[-2] == f"{source}/"
    assert arguments[-1] == "z5448417@kdm.restech.unsw.edu.au:lidc_baseline/"
    assert "BatchMode=yes" in " ".join(arguments)


def test_katana_shell_scripts_have_valid_bash_syntax() -> None:
    for script in (
        Path("scripts/katana/sync_code.sh"),
        Path("scripts/katana/bootstrap_cuda_env.sh"),
        Path("scripts/katana/cuda_smoke.pbs"),
        Path("scripts/katana/v2_cuda_smoke.pbs"),
    ):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{script}: {result.stderr}"


def test_cuda_batch_uses_the_synchronized_source_tree() -> None:
    script = Path("scripts/katana/cuda_smoke.pbs").read_text(encoding="utf-8")

    assert 'export PYTHONPATH="$code_directory/src' in script


def test_v2_cuda_batch_pins_compatible_l40s_and_linear_regression_config() -> None:
    script = Path("scripts/katana/v2_cuda_smoke.pbs").read_text(encoding="utf-8")

    assert "gpu_model=L40S" in script
    assert "--config configs/baseline_v2.yaml" in script
    assert 'export PYTHONPATH="$code_directory/src' in script


def test_cuda_bootstrap_keeps_installer_outside_conda_prefix() -> None:
    script = Path("scripts/katana/bootstrap_cuda_env.sh").read_text(encoding="utf-8")

    assert 'bootstrap_directory="${KATANA_P0_DIR' in script
    assert 'installer="$bootstrap_directory/' in script
    assert 'mkdir -p "$bootstrap_directory"' in script
    assert 'mkdir -p "$conda_root"' not in script


def test_cuda_bootstrap_avoids_anaconda_default_channel_tos() -> None:
    script = Path("scripts/katana/bootstrap_cuda_env.sh").read_text(encoding="utf-8")

    assert "--override-channels" in script
    assert "--channel conda-forge" in script
    assert "setuptools=80.10.2" in script
    assert "import pylidc" in script
