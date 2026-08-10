"""Versioned baseline configuration utilities."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration whose document root is a mapping."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return dict(loaded)


def canonical_yaml(config: Mapping[str, Any]) -> bytes:
    """Serialize a configuration deterministically as UTF-8 YAML."""
    rendered = yaml.safe_dump(
        dict(config),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
        width=4096,
    )
    return rendered.replace("\r\n", "\n").encode("utf-8")


def compute_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of canonical configuration bytes."""
    return hashlib.sha256(canonical_yaml(config)).hexdigest()


def fold_seed(base_seed: int, fold_index: int, outer_folds: int = 5) -> int:
    """Derive the pre-registered additive seed for one outer fold."""
    if fold_index < 0 or fold_index >= outer_folds:
        raise ValueError(
            f"fold_index must be in [0, {outer_folds - 1}], got {fold_index}"
        )
    return base_seed + fold_index


def _validate_existing(path: Path, expected: bytes) -> None:
    if path.exists() and path.read_bytes() != expected:
        raise FileExistsError(f"Refusing to overwrite different content: {path}")


def _write_read_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
        )
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    path.chmod(0o444)


def freeze_config(
    source: str | Path,
    resolved: str | Path,
    digest: str | Path,
) -> str:
    """Freeze canonical YAML and its digest without destructive overwrite."""
    config = load_config(source)
    resolved_bytes = canonical_yaml(config)
    sha256 = hashlib.sha256(resolved_bytes).hexdigest()
    digest_bytes = f"{sha256}\n".encode("ascii")
    resolved_path = Path(resolved)
    digest_path = Path(digest)

    _validate_existing(resolved_path, resolved_bytes)
    _validate_existing(digest_path, digest_bytes)
    _write_read_only(resolved_path, resolved_bytes)
    _write_read_only(digest_path, digest_bytes)
    return sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze", help="Freeze a canonical snapshot")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--resolved", type=Path, required=True)
    freeze.add_argument("--digest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configuration command-line interface."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        sha256 = freeze_config(
            arguments.source,
            arguments.resolved,
            arguments.digest,
        )
        print(sha256)
        return 0
    raise AssertionError(f"Unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
