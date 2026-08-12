"""Read-only, resumable Katana-to-Mac private archive for P10."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from lidc_baseline.p10_report import CONFIG_RESOLVED_DEFAULT, validate_execution_config


SCHEMA_VERSION = 1
REMOTE_ACCOUNT_DEFAULT = "z5448417"
SSH_KEY_DEFAULT = Path("/Users/katherine/.ssh/id_ed25519_katana_lidc")
LOCAL_ROOT_DEFAULT = Path(
    "/Users/katherine/Desktop/lidc_data/lidc_baseline_private_archive/baseline_v2"
)
REMOTE_ROOT_DEFAULT = Path("/srv/scratch/z5448417/lidc-baseline-v2/runs/baseline_v2")
REMOTE_HOST_DEFAULT = "kdm.restech.unsw.edu.au"
WHITELIST = ("blackbox", "standard_cbm", "cem", "gam", "p9")
MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
COMPLETE_NAME = "ARCHIVE_COMPLETE.json"
PRIVATE_REPORT_DIRECTORY = "p10_private_report"
BANNED_ARCHIVE_SUFFIXES = {".dcm", ".xml"}
BANNED_ARCHIVE_PARTS = {".git"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_relative_archive_path(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] not in WHITELIST
        or any(part in BANNED_ARCHIVE_PARTS for part in path.parts)
        or path.suffix.lower() in BANNED_ARCHIVE_SUFFIXES
    ):
        raise ValueError(f"P10_ARCHIVE_FORBIDDEN_CONTENT:{relative}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Any, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)


def validate_archive_roots(
    local_root: Path,
    remote_root: Path = REMOTE_ROOT_DEFAULT,
    whitelist: Sequence[str] = WHITELIST,
) -> None:
    if local_root != LOCAL_ROOT_DEFAULT:
        raise ValueError("P10_ARCHIVE_LOCAL_ROOT_INVALID")
    if remote_root != REMOTE_ROOT_DEFAULT:
        raise ValueError("P10_ARCHIVE_REMOTE_ROOT_INVALID")
    if tuple(whitelist) != WHITELIST or len(set(whitelist)) != len(WHITELIST):
        raise ValueError("P10_ARCHIVE_WHITELIST_INVALID")
    if any("/" in name or name in {".", "..", ".git"} for name in whitelist):
        raise ValueError("P10_ARCHIVE_WHITELIST_UNSAFE")


def check_free_space(
    local_root: Path,
    expected_bytes: int,
    *,
    minimum_ratio: float = 1.2,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    if expected_bytes <= 0 or not math_is_finite_positive(minimum_ratio):
        raise ValueError("P10_ARCHIVE_SIZE_ESTIMATE_INVALID")
    local_root.parent.mkdir(parents=True, exist_ok=True)
    usage = disk_usage(local_root.parent)
    required = math_ceil(expected_bytes * minimum_ratio)
    if int(usage.free) < required:
        raise ValueError(
            f"P10_ARCHIVE_INSUFFICIENT_SPACE:{usage.free}:{required}:{expected_bytes}"
        )
    return {
        "expected_bytes": int(expected_bytes),
        "minimum_ratio": float(minimum_ratio),
        "required_free_bytes": required,
        "observed_free_bytes": int(usage.free),
        "observed_ratio": int(usage.free) / expected_bytes,
    }


def math_is_finite_positive(value: float) -> bool:
    return value > 0 and value != float("inf") and value == value


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1


def build_local_manifest(
    local_root: Path,
    *,
    whitelist: Sequence[str] = WHITELIST,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in whitelist:
        root = local_root / directory
        if not root.is_dir():
            raise ValueError(f"P10_ARCHIVE_DIRECTORY_MISSING:{directory}")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"P10_ARCHIVE_SYMLINK_FORBIDDEN:{path}")
            if not path.is_file():
                continue
            relative = path.relative_to(local_root).as_posix()
            _validate_relative_archive_path(relative)
            rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise ValueError("P10_ARCHIVE_EMPTY")
    return sorted(rows, key=lambda row: row["relative_path"])


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["relative_path"])
    return hashlib.sha256(_canonical_json_bytes(ordered)).hexdigest()


def _remote_manifest_script(
    remote_root: Path = REMOTE_ROOT_DEFAULT,
    whitelist: Sequence[str] = WHITELIST,
) -> str:
    root_literal = repr(str(remote_root))
    names_literal = repr(list(whitelist))
    return """import hashlib, json, pathlib
root = pathlib.Path(__ROOT__)
names = __NAMES__
rows = []
for name in names:
    for path in sorted((root / name).rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        rows.append({
            'relative_path': str(path.relative_to(root)),
            'size_bytes': path.stat().st_size,
            'sha256': digest.hexdigest(),
        })
print(json.dumps(sorted(rows, key=lambda row: row['relative_path']), sort_keys=True, separators=(',', ':')))
""".replace("__ROOT__", root_literal).replace("__NAMES__", names_literal)


def fetch_remote_manifest(
    *,
    account: str = REMOTE_ACCOUNT_DEFAULT,
    host: str = REMOTE_HOST_DEFAULT,
    ssh_key: Path = SSH_KEY_DEFAULT,
    remote_root: Path = REMOTE_ROOT_DEFAULT,
    whitelist: Sequence[str] = WHITELIST,
) -> list[dict[str, Any]]:
    validate_archive_roots(LOCAL_ROOT_DEFAULT, remote_root, whitelist)
    command = [
        "ssh",
        "-i",
        str(ssh_key),
        "-o",
        "BatchMode=yes",
        f"{account}@{host}",
        "python3",
        "-",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        input=_remote_manifest_script(remote_root, whitelist),
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or not payload:
        raise ValueError("P10_REMOTE_ARCHIVE_MANIFEST_INVALID")
    rows = sorted(payload, key=lambda row: row["relative_path"])
    for row in rows:
        if set(row) != {"relative_path", "size_bytes", "sha256"}:
            raise ValueError("P10_REMOTE_ARCHIVE_MANIFEST_SCHEMA_INVALID")
        _validate_relative_archive_path(str(row["relative_path"]))
        if not any(row["relative_path"].startswith(f"{name}/") for name in whitelist):
            raise ValueError("P10_REMOTE_ARCHIVE_MANIFEST_SCOPE_INVALID")
        if int(row["size_bytes"]) < 0 or len(str(row["sha256"])) != 64:
            raise ValueError("P10_REMOTE_ARCHIVE_MANIFEST_VALUE_INVALID")
    return rows


def rsync_command(
    directory: str,
    *,
    account: str = REMOTE_ACCOUNT_DEFAULT,
    host: str = REMOTE_HOST_DEFAULT,
    ssh_key: Path = SSH_KEY_DEFAULT,
    remote_root: Path = REMOTE_ROOT_DEFAULT,
    local_root: Path = LOCAL_ROOT_DEFAULT,
) -> list[str]:
    if directory not in WHITELIST:
        raise ValueError("P10_ARCHIVE_DIRECTORY_NOT_WHITELISTED")
    return [
        "rsync",
        "--archive",
        "--partial",
        "--append",
        "--human-readable",
        "--progress",
        "--stats",
        "-e",
        f"ssh -i {ssh_key} -o BatchMode=yes",
        f"{account}@{host}:{remote_root}/{directory}/",
        f"{local_root}/{directory}/",
    ]


def _run_rsync(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def write_archive_completion(
    local_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    remote_manifest_sha256: str,
) -> dict[str, Any]:
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["relative_path"])
    digest = manifest_sha256(ordered)
    if digest != remote_manifest_sha256:
        raise ValueError("P10_ARCHIVE_REMOTE_LOCAL_MANIFEST_MISMATCH")
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "files": ordered,
    }
    manifest_path = local_root / MANIFEST_NAME
    _atomic_write_json(manifest_path, manifest_payload)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "ARCHIVE_COMPLETE",
        "file_count": len(ordered),
        "total_bytes": sum(int(row["size_bytes"]) for row in ordered),
        "manifest_sha256": digest,
        "remote_manifest_sha256": remote_manifest_sha256,
        "remote_whitelist": list(WHITELIST),
        "remote_write": False,
        "remote_delete": False,
        "local_delete": False,
    }
    _atomic_write_json(local_root / COMPLETE_NAME, completion)
    return completion


def sync_archive(
    *,
    local_root: Path = LOCAL_ROOT_DEFAULT,
    remote_root: Path = REMOTE_ROOT_DEFAULT,
    host: str = REMOTE_HOST_DEFAULT,
    account: str = REMOTE_ACCOUNT_DEFAULT,
    ssh_key: Path = SSH_KEY_DEFAULT,
    manifest_fetcher: Callable[..., list[dict[str, Any]]] = fetch_remote_manifest,
    transfer_runner: Callable[[Sequence[str]], None] = _run_rsync,
) -> dict[str, Any]:
    validate_execution_config(CONFIG_RESOLVED_DEFAULT)
    validate_archive_roots(local_root, remote_root)
    remote_rows = manifest_fetcher(
        account=account,
        host=host,
        ssh_key=ssh_key,
        remote_root=remote_root,
        whitelist=WHITELIST,
    )
    expected_bytes = sum(int(row["size_bytes"]) for row in remote_rows)
    space = check_free_space(local_root, expected_bytes)
    local_root.mkdir(parents=True, exist_ok=True)
    for directory in WHITELIST:
        (local_root / directory).mkdir(parents=True, exist_ok=True)
        transfer_runner(
            rsync_command(
                directory,
                account=account,
                host=host,
                ssh_key=ssh_key,
                remote_root=remote_root,
                local_root=local_root,
            )
        )
    local_rows = build_local_manifest(local_root)
    if local_rows != remote_rows:
        raise ValueError("P10_ARCHIVE_FILE_LEVEL_VERIFICATION_FAILED")
    completion = write_archive_completion(
        local_root,
        local_rows,
        remote_manifest_sha256=manifest_sha256(remote_rows),
    )
    return {**completion, "free_space_gate": space}


def verify_archive(local_root: Path = LOCAL_ROOT_DEFAULT) -> dict[str, Any]:
    validate_archive_roots(local_root)
    allowed_top_level = set(WHITELIST) | {
        MANIFEST_NAME,
        COMPLETE_NAME,
        PRIVATE_REPORT_DIRECTORY,
    }
    extra = {path.name for path in local_root.iterdir()} - allowed_top_level
    if extra:
        raise ValueError(f"P10_ARCHIVE_EXTRA_TOP_LEVEL:{sorted(extra)}")
    completion = json.loads((local_root / COMPLETE_NAME).read_text(encoding="utf-8"))
    manifest = json.loads((local_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if completion.get("status") != "ARCHIVE_COMPLETE" or manifest.get("status") != "PASS":
        raise ValueError("P10_ARCHIVE_COMPLETION_STATUS_INVALID")
    expected_rows = manifest.get("files")
    if not isinstance(expected_rows, list):
        raise ValueError("P10_ARCHIVE_MANIFEST_SCHEMA_INVALID")
    actual_rows = build_local_manifest(local_root)
    if actual_rows != expected_rows:
        raise ValueError("P10_ARCHIVE_TAMPER_OR_COVERAGE_MISMATCH")
    digest = manifest_sha256(actual_rows)
    if (
        digest != completion.get("manifest_sha256")
        or digest != completion.get("remote_manifest_sha256")
        or len(actual_rows) != completion.get("file_count")
        or sum(row["size_bytes"] for row in actual_rows) != completion.get("total_bytes")
    ):
        raise ValueError("P10_ARCHIVE_COMPLETION_BINDING_INVALID")
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync")
    subparsers.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = sync_archive() if args.command == "sync" else verify_archive()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
