import importlib
import tomllib
from pathlib import Path

import yaml


EXPECTED_RUNTIME_DEPENDENCIES = {
    "torch": "2.5.1",
    "monai": "1.4.0",
    "pylidc": "0.2.3",
    "pydicom": "2.4.4",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "pyarrow": "18.1.0",
    "scikit-learn": "1.5.2",
    "scipy": "1.14.1",
    "PyYAML": "6.0.2",
    "setuptools": "80.10.2",
}


def test_project_runtime_dependencies_match_registered_versions() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = dict(item.split("==", maxsplit=1) for item in project["dependencies"])

    assert dependencies == EXPECTED_RUNTIME_DEPENDENCIES


def test_platform_environment_definitions_use_python_311_and_project_extras() -> None:
    for definition in (
        Path("environment/macos-arm64.yml"),
        Path("environment/katana-cuda.yml"),
    ):
        environment = yaml.safe_load(definition.read_text(encoding="utf-8"))
        dependencies = environment["dependencies"]
        python_requirement = next(
            item for item in dependencies if isinstance(item, str) and item.startswith("python=")
        )
        pip_section = next(item["pip"] for item in dependencies if isinstance(item, dict))

        assert python_requirement.startswith("python=3.11")
        assert environment["channels"] == ["conda-forge"]
        assert "setuptools=80.10.2" in dependencies
        assert "-e .[dev]" in pip_section


def test_pylidc_is_importable_in_the_runtime_environment() -> None:
    pylidc = importlib.import_module("pylidc")

    assert pylidc.__version__ == "0.2.3"


def test_tracked_environment_evidence_has_no_temporary_checkout_path() -> None:
    evidence_files = list(Path("environment").rglob("*"))
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in evidence_files if path.is_file()
    )

    assert "/private/tmp/lidc-baseline-p0" not in text
    assert "/Users/katherine/Desktop/lidc_data" not in text
