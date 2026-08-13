"""Build the read-only P10 Results & Artifacts Master Catalogue.

The Catalogue is an index over already frozen P5--P10 evidence.  It never
invokes a model, recomputes a scientific result, or modifies a report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from lidc_baseline.p10_archive import manifest_sha256
from lidc_baseline.p10_report import (
    CONCEPT_ORDER,
    FOLD_COUNTS,
    MODEL_LABELS,
    MODEL_ORDER,
    sha256_file,
    verify_inputs,
)


SCHEMA_VERSION = 1
CATALOGUE_CONFIG_DEFAULT = Path(
    "configs/experiments/baseline_v2_p10_catalogue.yaml"
)
CATALOGUE_CONFIG_RESOLVED_DEFAULT = Path(
    "configs/experiments/baseline_v2_p10_catalogue.resolved.yaml"
)
CATALOGUE_CONFIG_SHA_DEFAULT = Path(
    "configs/experiments/baseline_v2_p10_catalogue.sha256"
)
CATALOGUE_CONFIG_SHA256 = (
    "2c50282f1c85a769a99a8040e8099e5adcbe8b9447742571e9da4dfba093bdd9"
)
RESULTS_CATALOGUE_PLAN_SHA256 = (
    "eb3aa9110fc06acd8cd9e2a375b64bcfa5a60d768ef6f37f16e1c693576c5c93"
)
P10_REPORT_PLAN_SHA256 = (
    "be33c07d566914d40bd50fba65b347118e706940a64495782cf0245f7914629c"
)
PUBLIC_ROOT_DEFAULT = Path("docs/results")
REPORT_DATA_DEFAULT = Path("reports/baseline_v2/p10/public/report_data.json")
PRIVATE_ARCHIVE_ROOT_DEFAULT = Path(
    "/Users/katherine/Desktop/lidc_data/lidc_baseline_private_archive/baseline_v2"
)
PRIVATE_OVERLAY_ROOT_DEFAULT = (
    PRIVATE_ARCHIVE_ROOT_DEFAULT / "p10_private_report"
)
RAW_DATA_ROOT_DEFAULT = Path("/Users/katherine/Desktop/lidc_data")
PRIVATE_CASE_INDEX_RELATIVE = Path("p10_private_report/private_case_index.json")
ARCHIVE_MANIFEST_NAME = "ARCHIVE_MANIFEST.json"
ARCHIVE_COMPLETE_NAME = "ARCHIVE_COMPLETE.json"
PUBLIC_REGISTRY_NAME = "results_catalogue_registry.json"
PUBLIC_MANIFEST_NAME = "catalogue_manifest.json"
PHASE_STATUS_SNAPSHOT_NAME = "catalogue_phase_status_snapshot.json"
PRIVATE_COMPLETE_NAME = "PRIVATE_CATALOGUE_COMPLETE.json"
PRIVATE_LOCATIONS_NAME = "results_catalogue_private_locations.csv"
PRIVATE_MASTER_XLSX = "RESULTS_MASTER_CATALOGUE.xlsx"
PRIVATE_HUMAN_XLSX = "RESULTS_ARTIFACTS_MASTER_TABLE.xlsx"

CONTINUOUS_CONCEPTS = (
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
CATEGORICAL_CONCEPTS = ("internalStructure", "calcification")
CONCEPT_MODELS = ("standard_cbm", "mixed_cem", "learned_softmax_gam")
TARGETS_BY_MODEL = {
    "blackbox": ("malignancy",),
    "standard_cbm": ("malignancy", *CONCEPT_ORDER),
    "mixed_cem": ("malignancy", *CONCEPT_ORDER),
    "learned_softmax_gam": ("malignancy", *CONCEPT_ORDER),
}
PHASE_BY_MODEL = {
    "blackbox": "P5",
    "standard_cbm": "P6",
    "mixed_cem": "P7",
    "learned_softmax_gam": "P8",
}
PRIVATE_DIR_BY_MODEL = {
    "blackbox": "blackbox",
    "standard_cbm": "standard_cbm",
    "mixed_cem": "cem",
    "learned_softmax_gam": "gam",
}
OOF_RELATIVE_BY_MODEL = {
    "blackbox": "p9/canonical_oof/blackbox_oof_predictions.parquet",
    "standard_cbm": "p9/canonical_oof/standard_cbm_oof_predictions.parquet",
    "mixed_cem": "p9/canonical_oof/cem_oof_predictions.parquet",
    "learned_softmax_gam": "p9/canonical_oof/gam_oof_predictions.parquet",
}
CONTROLLED_AVAILABILITY = {
    "RESULT_ALREADY_EXISTS",
    "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
    "DATA_NOT_PERSISTED",
    "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE",
}
CONTROLLED_USAGE = {
    "USED_MAIN_TEXT",
    "USED_APPENDIX",
    "USED_PRIVATE_APPENDIX",
    "AUDIT_ONLY",
    "INTENTIONALLY_OMITTED_WITH_REASON",
}
CONTROLLED_INTEGRITY = {"VERIFIED", "MISSING", "HASH_MISMATCH", "NOT_APPLICABLE"}
PUBLIC_FORBIDDEN_VALUES = ("/Users/", "/srv/scratch/", "patient_key", "nodule_uid")
REQUIRED_REGISTRY_FIELDS = {
    "catalogue_item_id",
    "entity_type",
    "phase",
    "model",
    "fold",
    "concept_or_target",
    "result_name",
    "scientific_question",
    "scientific_status",
    "availability_status",
    "report_usage_status",
    "source_artifact_id",
    "source_root_alias",
    "source_relative_path",
    "source_field_path",
    "source_sha256",
    "row_or_sample_count",
    "privacy_class",
    "new_inference_required",
    "report_section_id",
    "report_table_ids",
    "report_figure_ids",
    "omission_reason",
    "approval_reference",
    "integrity_status",
    "category",
    "details",
}
CATEGORY_FILE_NAMES = {
    "A": "CAT_A_phase_overview.csv",
    "B": "CAT_B_training_results.csv",
    "C": "CAT_C_primary_results.csv",
    "D": "CAT_D_paired_primary.csv",
    "E": "CAT_E_secondary_results.csv",
    "F": "CAT_F_paired_secondary.csv",
    "G": "CAT_G_continuous_concepts.csv",
    "H": "CAT_H_categorical_concepts.csv",
    "I": "CAT_I_interventions.csv",
    "J": "CAT_J_contributions.csv",
    "K": "CAT_K_gam_alpha.csv",
    "L": "CAT_L_gradcam.csv",
    "M": "CAT_M_undefined_rca.csv",
    "N": "CAT_N_spatial_faithfulness.csv",
    "O": "CAT_O_tables.csv",
    "P": "CAT_P_figures.csv",
    "Q": "CAT_Q_qualitative_cases.csv",
    "R": "CAT_R_storage.csv",
    "S": "CAT_S_report_evidence.csv",
    "T": "CAT_T_gaps.csv",
}


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P10_CATALOGUE_JSON_OBJECT_REQUIRED:{path}")
    return payload


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any, *, mode: int = 0o644) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(payload), mode=mode)


def _stringify(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _stringify(row.get(name)) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str] | None = None,
    mode: int = 0o644,
) -> None:
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    _atomic_write_bytes(path, _csv_bytes(rows, fieldnames), mode=mode)


def _source_hash(repository_root: Path, relative: str) -> str:
    path = repository_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"P10_CATALOGUE_SOURCE_MISSING:{relative}")
    return sha256_file(path)


def _file_tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".DS_Store"
    }


def validate_catalogue_config(
    *,
    repository_root: Path = Path("."),
    resolved_path: Path = CATALOGUE_CONFIG_RESOLVED_DEFAULT,
    sha_path: Path = CATALOGUE_CONFIG_SHA_DEFAULT,
) -> dict[str, Any]:
    resolved = repository_root / resolved_path
    recorded = (repository_root / sha_path).read_text(encoding="utf-8").split()[0]
    actual = sha256_file(resolved)
    if actual != recorded or actual != CATALOGUE_CONFIG_SHA256:
        raise ValueError("P10_CATALOGUE_CONFIG_SHA256_MISMATCH")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    plans = payload["approved_plans"]
    expected = {
        "results_catalogue": RESULTS_CATALOGUE_PLAN_SHA256,
        "catalogue_driven_bilingual_report": P10_REPORT_PLAN_SHA256,
    }
    for name, digest in expected.items():
        plan = plans[name]
        path = repository_root / plan["path"]
        if plan.get("approved") is not True or plan.get("sha256") != digest:
            raise ValueError(f"P10_CATALOGUE_PLAN_APPROVAL_INVALID:{name}")
        if sha256_file(path) != digest:
            raise ValueError(f"P10_CATALOGUE_PLAN_SHA256_MISMATCH:{name}")
    gates = payload["gates"]
    if (
        gates.get("results_catalogue_plan_approved") != 1
        or gates.get("p10_report_plan_approved") != 1
        or gates.get("catalogue_implementation_authorized") != 1
        or gates.get("generated_catalogue_approved") != 0
        or gates.get("report_revision_authorized") != 0
    ):
        raise ValueError("P10_CATALOGUE_GATE_INVALID")
    if any(value != "forbidden" for key, value in payload["boundaries"].items() if key != "source_access"):
        raise ValueError("P10_CATALOGUE_BOUNDARY_INVALID")
    if payload["boundaries"].get("source_access") != "read_only":
        raise ValueError("P10_CATALOGUE_SOURCE_ACCESS_INVALID")
    return payload


def _archive_sources(private_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    complete = _read_json(private_root / ARCHIVE_COMPLETE_NAME)
    manifest = _read_json(private_root / ARCHIVE_MANIFEST_NAME)
    rows = manifest.get("files")
    if complete.get("status") != "ARCHIVE_COMPLETE" or manifest.get("status") != "PASS":
        raise ValueError("P10_CATALOGUE_ARCHIVE_STATUS_INVALID")
    if not isinstance(rows, list) or len(rows) != complete.get("file_count"):
        raise ValueError("P10_CATALOGUE_ARCHIVE_MANIFEST_INVALID")
    if manifest_sha256(rows) != complete.get("manifest_sha256"):
        raise ValueError("P10_CATALOGUE_ARCHIVE_MANIFEST_SHA256_MISMATCH")
    by_path = {str(row["relative_path"]): dict(row) for row in rows}
    if len(by_path) != len(rows):
        raise ValueError("P10_CATALOGUE_ARCHIVE_DUPLICATE_PATH")
    return complete, by_path


def _load_verified_sources(
    repository_root: Path,
    private_root: Path,
    report_data_path: Path,
) -> dict[str, Any]:
    inputs = verify_inputs(repository_root=repository_root)
    report_path = repository_root / report_data_path
    report = _read_json(report_path)
    if report.get("status") != "PASS":
        raise ValueError("P10_CATALOGUE_REPORT_DATA_STATUS_INVALID")
    verification = report.get("input_verification", {})
    if verification.get("source_manifest_sha256") != inputs["source_manifest_sha256"]:
        raise ValueError("P10_CATALOGUE_REPORT_DATA_SOURCE_BINDING_INVALID")
    complete, archive_by_path = _archive_sources(private_root)
    case_index_path = private_root / PRIVATE_CASE_INDEX_RELATIVE
    case_index = _read_json(case_index_path)
    if case_index.get("status") != "PRIVATE_CASE_INDEX" or case_index.get("model_forward") is not False:
        raise ValueError("P10_CATALOGUE_PRIVATE_CASE_INDEX_INVALID")
    if len(case_index.get("cases", ())) != 14:
        raise ValueError("P10_CATALOGUE_PRIVATE_CASE_COUNT_INVALID")
    return {
        "inputs": inputs,
        "report": report,
        "report_data_sha256": sha256_file(report_path),
        "archive_complete": complete,
        "archive_by_path": archive_by_path,
        "case_index": case_index,
        "case_index_sha256": sha256_file(case_index_path),
    }


def _item(
    *,
    category: str,
    catalogue_item_id: str,
    entity_type: str,
    phase: str,
    result_name: str,
    scientific_question: str,
    source_artifact_id: str,
    source_root_alias: str,
    source_relative_path: str,
    source_field_path: str,
    source_sha256: str,
    model: str | None = None,
    fold: int | None = None,
    concept_or_target: str | None = None,
    row_or_sample_count: int | None = None,
    privacy_class: str = "PUBLIC_DEIDENTIFIED",
    report_section_id: str = "SEC-REPRODUCIBILITY",
    report_table_ids: Sequence[str] = (),
    report_figure_ids: Sequence[str] = (),
    availability_status: str = "RESULT_ALREADY_EXISTS",
    report_usage_status: str = "USED_APPENDIX",
    scientific_status: str = "PASS",
    integrity_status: str = "VERIFIED",
    new_inference_required: bool = False,
    omission_reason: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "catalogue_item_id": catalogue_item_id,
        "entity_type": entity_type,
        "phase": phase,
        "model": model,
        "fold": fold,
        "concept_or_target": concept_or_target,
        "result_name": result_name,
        "scientific_question": scientific_question,
        "scientific_status": scientific_status,
        "availability_status": availability_status,
        "report_usage_status": report_usage_status,
        "source_artifact_id": source_artifact_id,
        "source_root_alias": source_root_alias,
        "source_relative_path": source_relative_path,
        "source_field_path": source_field_path,
        "source_sha256": source_sha256,
        "row_or_sample_count": row_or_sample_count,
        "privacy_class": privacy_class,
        "new_inference_required": bool(new_inference_required),
        "report_section_id": report_section_id,
        "report_table_ids": list(report_table_ids),
        "report_figure_ids": list(report_figure_ids),
        "omission_reason": omission_reason,
        "approval_reference": f"RESULTS_CATALOGUE_PLAN:{RESULTS_CATALOGUE_PLAN_SHA256}",
        "integrity_status": integrity_status,
        "category": f"CAT-{category}",
        "details": dict(details or {}),
    }


def _phase_status_snapshot() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "snapshot_kind": "P10_RESULTS_CATALOGUE_GATE_3",
        "phase_statuses": {
            **{f"P{phase}": "COMPLETED" for phase in range(10)},
            "P10": "IN_PROGRESS_CATALOGUE_VERIFIED_PENDING_USER_APPROVAL",
        },
        "p9_delivery_status": "DELIVERED",
        "generated_catalogue_approved": 0,
        "report_revision_authorized": 0,
        "p11_started": False,
        "evidence_policy": (
            "This immutable Catalogue snapshot indexes phase state; full completion evidence is "
            "bound separately in CAT-B through CAT-R and the P5-P9 source manifest."
        ),
    }


def _phase_rows(repository_root: Path) -> list[dict[str, Any]]:
    snapshot_path = f"docs/results/{PHASE_STATUS_SNAPSHOT_NAME}"
    snapshot_sha = hashlib.sha256(_canonical_json_bytes(_phase_status_snapshot())).hexdigest()
    purposes = {
        "P0": "Environment and deterministic regression smoke",
        "P1": "Canonical DICOM/XML mapping and geometry audit",
        "P2": "Physical-nodule cohort and target materialization",
        "P3": "Consensus masks and deterministic 64-cubed ROI",
        "P4": "Patient-grouped folds and shared encoder initializations",
        "P5": "Black-box regression",
        "P6": "Sequential Standard CBM regression",
        "P7": "Mixed-type CEM regression",
        "P8": "Learned-softmax GAM regression",
        "P9": "Unified evaluation, intervention, and spatial explanation",
        "P10": "Catalogue, bilingual reporting, and private archive",
    }
    rows = []
    for phase, purpose in purposes.items():
        rows.append(
            _item(
                category="A",
                catalogue_item_id=f"RES-{phase}-PHASE-OVERVIEW",
                entity_type="phase_overview",
                phase=phase,
                result_name=purpose,
                scientific_question="What was completed in this protocol phase?",
                source_artifact_id="ART-CATALOGUE-PHASE-STATUS-SNAPSHOT",
                source_root_alias="repo://",
                source_relative_path=snapshot_path,
                source_field_path=f"phase_statuses.{phase}",
                source_sha256=snapshot_sha,
                report_usage_status="AUDIT_ONLY" if phase in {"P0", "P1", "P2", "P3", "P4"} else "USED_APPENDIX",
                details={"component": "phase_summary", "purpose": purpose},
            )
        )
    for component in (
        "task_evaluation",
        "bootstrap",
        "concept_fidelity",
        "intervention",
        "contribution_centering",
        "gradcam",
        "occlusion_faithfulness",
        "undefined_map_rca",
    ):
        path = f"artifacts/baseline_v2/audit/p9/{'spatial' if component in {'gradcam','occlusion_faithfulness','undefined_map_rca'} else ('task' if component == 'task_evaluation' else component if component != 'concept_fidelity' else 'concept')}.json"
        rows.append(
            _item(
                category="A",
                catalogue_item_id=f"RES-P9-{component.upper().replace('_','-')}",
                entity_type="phase_component",
                phase="P9",
                result_name=component.replace("_", " ").title(),
                scientific_question="Which P9 evaluation component produced this evidence?",
                source_artifact_id=f"ART-P9-{Path(path).stem.upper()}",
                source_root_alias="repo://",
                source_relative_path=path,
                source_field_path="$",
                source_sha256=_source_hash(repository_root, path),
                report_usage_status="USED_MAIN_TEXT",
                report_section_id="SEC-RESULTS",
                details={"component": component, "scientific_status": "PASS"},
            )
        )
    return rows


def _best_fields(phase: str, fold: Mapping[str, Any]) -> tuple[str, str, str]:
    if phase == "P6":
        epoch = f"concept={fold['concept_best_epoch_index']};task={fold['task_best_epoch_index']}"
        objective = (
            f"concept_loss={fold['concept_best_validation_loss']};"
            f"task_mse={fold['task_best_validation_mse']}"
        )
        checkpoint = "concept_stage/best.pt+task_stage/best.pt"
    elif phase == "P5":
        epoch = str(fold["best_epoch_index"])
        objective = f"task_mse={fold['best_validation_mse']}"
        checkpoint = "best.pt"
    else:
        epoch = str(fold["best_epoch_index"])
        objective = f"total_loss={fold['best_validation_total_loss']}"
        checkpoint = "best.pt"
    return epoch, objective, checkpoint


def _training_rows(
    repository_root: Path,
    report: Mapping[str, Any],
    archive_by_path: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_models = report["task"]["models"]
    for model in MODEL_ORDER:
        phase = PHASE_BY_MODEL[model]
        private_dir = PRIVATE_DIR_BY_MODEL[model]
        audit_phase = phase.lower()
        for fold_index in range(5):
            audit_relative = f"artifacts/baseline_v2/audit/{audit_phase}/fold_{fold_index}.json"
            audit = _read_json(repository_root / audit_relative)
            epoch, objective, checkpoint = _best_fields(phase, audit)
            metric = task_models[model]["folds"][fold_index]["task"]
            test_count = int(audit.get("test_inference_transactions", audit.get("test_transaction_count", 1)))
            rows.append(
                _item(
                    category="B",
                    catalogue_item_id=f"RES-{phase}-{model.upper()}-FOLD{fold_index}-TRAIN-TEST",
                    entity_type="model_fold_run",
                    phase=phase,
                    model=model,
                    fold=fold_index,
                    result_name="Frozen training, best checkpoint, and exactly-once test evaluation",
                    scientific_question="What did this model-fold run produce?",
                    source_artifact_id=f"ART-{phase}-FOLD{fold_index}-AUDIT",
                    source_root_alias="repo://",
                    source_relative_path=audit_relative,
                    source_field_path="$",
                    source_sha256=sha256_file(repository_root / audit_relative),
                    row_or_sample_count=int(audit["test_samples"]),
                    privacy_class="PRIVATE_SOURCE_PUBLIC_AUDIT",
                    report_section_id="SEC-EXPERIMENTAL-SETUP",
                    report_table_ids=("RPT-T05",),
                    details={
                        "n_test": int(audit["test_samples"]),
                        "best_epoch": epoch,
                        "best_validation_objective": objective,
                        "test_mae": metric["original_scale_mae"],
                        "test_rmse": metric["original_scale_rmse"],
                        "pearson": metric["pearson"],
                        "spearman": metric["spearman"],
                        "checkpoint": f"mac-archive://{private_dir}/fold_{fold_index}/{checkpoint}",
                        "predictions": f"mac-archive://{private_dir}/fold_{fold_index}/test_predictions.parquet",
                        "metrics": f"mac-archive://{private_dir}/fold_{fold_index}/metrics.json",
                        "evaluation": f"mac-archive://{private_dir}/fold_{fold_index}/test_evaluation.json",
                        "test_transaction_count": test_count,
                        "scheduler_terminal_status_distinct_from_scientific_status": True,
                        "scientific_status": audit["status"],
                    },
                )
            )
        oof_relative = OOF_RELATIVE_BY_MODEL[model]
        archive_row = archive_by_path[oof_relative]
        rows.append(
            _item(
                category="B",
                catalogue_item_id=f"RES-P9-OOF-{model.upper()}",
                entity_type="pooled_oof",
                phase="P9",
                model=model,
                result_name="Canonical pooled OOF prediction set",
                scientific_question="Does the model have exactly one frozen OOF prediction per nodule?",
                source_artifact_id=f"ART-P9-OOF-{model.upper()}",
                source_root_alias="mac-archive://",
                source_relative_path=oof_relative,
                source_field_path="$",
                source_sha256=str(archive_row["sha256"]),
                row_or_sample_count=2633,
                privacy_class="PRIVATE_RESTRICTED",
                report_section_id="SEC-RESULTS-PREDICTION",
                report_table_ids=("RPT-T07", "RPT-T09"),
                report_figure_ids=("RPT-F04", "RPT-F06"),
                details={"unique_nodules": 2633, "unique_patients": 868, "exactly_once": True},
            )
        )
    return rows


def _task_rows(repository_root: Path, report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], ...]:
    task_path = "artifacts/baseline_v2/audit/p9/task.json"
    bootstrap_path = "artifacts/baseline_v2/audit/p9/bootstrap.json"
    task_sha = _source_hash(repository_root, task_path)
    bootstrap_sha = _source_hash(repository_root, bootstrap_path)
    task_models = report["task"]["models"]
    bootstrap = report["bootstrap"]
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    ranking = sorted(MODEL_ORDER, key=lambda name: task_models[name]["pooled"]["original_scale_mae"])
    for model in MODEL_ORDER:
        pooled = task_models[model]["pooled"]
        ci = bootstrap["models"][model]
        primary.append(
            _item(
                category="C", catalogue_item_id=f"RES-P9-PRIMARY-{model.upper()}",
                entity_type="pooled_primary_result", phase="P9", model=model,
                result_name="Pooled primary regression metrics and patient-bootstrap intervals",
                scientific_question="How accurately does the model predict radiologist-assessed malignancy?",
                source_artifact_id="ART-P9-TASK", source_root_alias="repo://",
                source_relative_path=task_path, source_field_path=f"models.{model}.pooled",
                source_sha256=task_sha, row_or_sample_count=2633,
                report_section_id="SEC-RESULTS-PREDICTION", report_table_ids=("RPT-T07",),
                report_figure_ids=("RPT-F04",), report_usage_status="USED_MAIN_TEXT",
                details={
                    **pooled,
                    "bootstrap_draws": 2000,
                    "bootstrap_intervals": ci,
                    "mae_rank": ranking.index(model) + 1,
                    "unclipped": True,
                    "unique_patients": 868,
                },
            )
        )
        sec = task_models[model]["pooled_secondary"]
        secondary.append(
            _item(
                category="E", catalogue_item_id=f"RES-P9-SECONDARY-{model.upper()}",
                entity_type="pooled_secondary_result", phase="P9", model=model,
                result_name="Extreme-subset discrimination and validation-threshold metrics",
                scientific_question="How well does the same continuous score separate low and high extremes?",
                source_artifact_id="ART-P9-TASK", source_root_alias="repo://",
                source_relative_path=task_path, source_field_path=f"models.{model}.pooled_secondary",
                source_sha256=task_sha, row_or_sample_count=1073,
                report_section_id="SEC-RESULTS-PREDICTION", report_table_ids=("RPT-T09",),
                report_figure_ids=("RPT-F06",), report_usage_status="USED_MAIN_TEXT",
                details={
                    **sec,
                    "unique_patients": 578,
                    "bootstrap_draws": 2000,
                    "auroc_interval": ci["auroc"],
                    "auprc_interval": ci["auprc"],
                    "threshold_rule": "fold-specific validation extreme subset only; largest finite Youden-J threshold",
                    "fixed_normalized_0_5_use": "sensitivity only",
                },
            )
        )
    paired_primary: list[dict[str, Any]] = []
    paired_secondary: list[dict[str, Any]] = []
    for pair, value in sorted(bootstrap["paired_mae_A_minus_B"].items()):
        a, b = pair.split("__")
        crosses = float(value["percentile_2_5"]) <= 0 <= float(value["percentile_97_5"])
        paired_primary.append(
            _item(
                category="D", catalogue_item_id=f"RES-P9-PAIRED-MAE-{a.upper()}-VS-{b.upper()}",
                entity_type="paired_primary_comparison", phase="P9",
                result_name="Paired patient-bootstrap Delta-MAE", scientific_question="Which model has lower MAE under shared patient draws?",
                source_artifact_id="ART-P9-BOOTSTRAP", source_root_alias="repo://",
                source_relative_path=bootstrap_path, source_field_path=f"paired_mae_A_minus_B.{pair}",
                source_sha256=bootstrap_sha, row_or_sample_count=2000,
                report_section_id="SEC-RESULTS-PREDICTION", report_table_ids=("RPT-T08",),
                report_figure_ids=("RPT-F05",), report_usage_status="USED_MAIN_TEXT",
                details={"model_a": a, "model_b": b, **value, "ci_crosses_zero": crosses,
                         "sign_convention": "MAE_A - MAE_B; positive supports B"},
            )
        )
    for pair, value in sorted(bootstrap["paired_auroc_B_minus_A"].items()):
        a, b = pair.split("__")
        crosses = float(value["percentile_2_5"]) <= 0 <= float(value["percentile_97_5"])
        paired_secondary.append(
            _item(
                category="F", catalogue_item_id=f"RES-P9-PAIRED-AUROC-{a.upper()}-VS-{b.upper()}",
                entity_type="paired_secondary_comparison", phase="P9",
                result_name="Paired patient-bootstrap Delta-AUROC", scientific_question="Which model has higher AUROC under shared patient draws?",
                source_artifact_id="ART-P9-BOOTSTRAP", source_root_alias="repo://",
                source_relative_path=bootstrap_path, source_field_path=f"paired_auroc_B_minus_A.{pair}",
                source_sha256=bootstrap_sha, row_or_sample_count=2000,
                report_section_id="SEC-RESULTS-PREDICTION", report_table_ids=("RPT-T10",),
                report_figure_ids=("RPT-F06",), report_usage_status="USED_MAIN_TEXT",
                details={"model_a": a, "model_b": b, **value, "ci_crosses_zero": crosses,
                         "sign_convention": "AUROC_B - AUROC_A; positive supports B"},
            )
        )
    return primary, paired_primary, secondary, paired_secondary


def _concept_rows(repository_root: Path, report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = "artifacts/baseline_v2/audit/p9/concept.json"
    digest = _source_hash(repository_root, path)
    continuous: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    for model in CONCEPT_MODELS:
        pooled = report["concept"][model]["pooled"]
        for concept in CONTINUOUS_CONCEPTS:
            continuous.append(
                _item(
                    category="G", catalogue_item_id=f"RES-P9-CONCEPT-{model.upper()}-{concept.upper()}",
                    entity_type="continuous_concept_result", phase="P9", model=model,
                    concept_or_target=concept, result_name="Pooled continuous concept fidelity",
                    scientific_question="How faithfully is this reader concept predicted?",
                    source_artifact_id="ART-P9-CONCEPT", source_root_alias="repo://",
                    source_relative_path=path, source_field_path=f"{model}.pooled.{concept}",
                    source_sha256=digest, row_or_sample_count=int(pooled[concept]["sample_count"]),
                    report_section_id="SEC-RESULTS-WHAT", report_table_ids=("RPT-T11",),
                    report_figure_ids=("RPT-F09A",), report_usage_status="USED_MAIN_TEXT",
                    details=pooled[concept],
                )
            )
        for concept in CATEGORICAL_CONCEPTS:
            categorical.append(
                _item(
                    category="H", catalogue_item_id=f"RES-P9-CONCEPT-{model.upper()}-{concept.upper()}",
                    entity_type="categorical_concept_result", phase="P9", model=model,
                    concept_or_target=concept, result_name="Pooled categorical concept fidelity",
                    scientific_question="How faithfully is the full reader-vote target predicted?",
                    source_artifact_id="ART-P9-CONCEPT", source_root_alias="repo://",
                    source_relative_path=path, source_field_path=f"{model}.pooled.{concept}",
                    source_sha256=digest, row_or_sample_count=int(pooled[concept]["soft_sample_count"]),
                    report_section_id="SEC-RESULTS-WHAT", report_table_ids=("RPT-T12",),
                    report_figure_ids=("RPT-F09B",), report_usage_status="USED_MAIN_TEXT",
                    details={**pooled[concept], "ground_truth_semantics": "full reader vote distribution; modal label display-only"},
                )
            )
    return continuous, categorical


def _intervention_rows(repository_root: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = "artifacts/baseline_v2/audit/p9/intervention.json"
    digest = _source_hash(repository_root, path)
    rows = []
    for model in CONCEPT_MODELS:
        payload = report["intervention"][model]
        for ordering, key in (("random_permutation", "random_permutations"), ("error_first", "error_first")):
            curve = payload[key]
            rows.append(
                _item(
                    category="I", catalogue_item_id=f"RES-P9-INTERVENTION-{model.upper()}-{ordering.upper()}",
                    entity_type="intervention_curve", phase="P9", model=model,
                    result_name=f"k=0..8 intervention ({ordering})",
                    scientific_question="How do concept corrections change task error and extreme discrimination?",
                    source_artifact_id="ART-P9-INTERVENTION", source_root_alias="repo://",
                    source_relative_path=path, source_field_path=f"{model}.{key}",
                    source_sha256=digest, row_or_sample_count=9,
                    report_section_id="SEC-RESULTS-HOW", report_table_ids=("RPT-T17",),
                    report_figure_ids=("RPT-F12",), report_usage_status="USED_MAIN_TEXT",
                    details={
                        "ordering": ordering,
                        "k": curve["k"],
                        "mae_curve": curve.get("pooled_original_scale_mae_mean", curve.get("pooled_original_scale_mae")),
                        "auroc_curve": curve.get("pooled_auroc_mean", curve.get("pooled_auroc")),
                        "iMAE": curve["iMAE"], "Delta_iMAE": curve["Delta_iMAE"],
                        "iAUC": curve["iAUC"], "Delta_iAUC": curve["Delta_iAUC"],
                        "positive_delta_means_improvement": True,
                    },
                )
            )
    return rows


def _contribution_rows(repository_root: Path, report: Mapping[str, Any], archive_by_path: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    path = "artifacts/baseline_v2/audit/p9/contribution_centering.json"
    digest = _source_hash(repository_root, path)
    rows = []
    for model in CONCEPT_MODELS:
        payload = report["contribution_centering"][model]
        oof_relative = OOF_RELATIVE_BY_MODEL[model]
        for concept in CONCEPT_ORDER:
            fold_constants = [fold["train_group_means_rating_point_units"][concept] for fold in payload["folds"]]
            rows.append(
                _item(
                    category="J", catalogue_item_id=f"RES-P9-CONTRIBUTION-{model.upper()}-{concept.upper()}",
                    entity_type="centered_contribution", phase="P9", model=model,
                    concept_or_target=concept, result_name="Train-centered OOF concept contribution",
                    scientific_question="Which concepts contribute positively or negatively to the frozen task score?",
                    source_artifact_id="ART-P9-CONTRIBUTION-CENTERING", source_root_alias="repo://",
                    source_relative_path=path, source_field_path=f"{model}", source_sha256=digest,
                    row_or_sample_count=2633, report_section_id="SEC-RESULTS-WHY",
                    report_table_ids=("RPT-T15",), report_figure_ids=("RPT-F10",),
                    report_usage_status="USED_MAIN_TEXT",
                    details={
                        "train_fold_centering_constants_rating_points": fold_constants,
                        "pooled_signed_mean_rating_points": payload["mean_rating_point_contribution_by_concept"][concept],
                        "centering_constant_is_importance": False,
                        "centered_oof_point_source": f"mac-archive://{oof_relative}",
                        "centered_oof_point_source_sha256": archive_by_path[oof_relative]["sha256"],
                        "empirical_oof_profile_renderable": True,
                        "case_level_contribution_bar_renderable": True,
                        "mean_absolute_contribution_status": "DATA_NOT_PERSISTED",
                        "mean_absolute_contribution_note": (
                            "No authoritative frozen P9 aggregate stores model-by-concept mean "
                            "absolute centered contribution. Per-sample contributions and train-fold "
                            "centering constants exist, but deriving the aggregate would be a prohibited "
                            "new scientific computation at the Catalogue gate. Any earlier narrative "
                            "summary is non-authoritative for P10."
                        ),
                    },
                )
            )
    return rows


def _alpha_rows(repository_root: Path, report: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = "artifacts/baseline_v2/audit/p9/learned_alpha.json"
    digest = _source_hash(repository_root, path)
    rows = []
    for fold in report["learned_alpha"]["folds"]:
        fold_index = int(fold["fold_index"])
        for concept in CONCEPT_ORDER:
            group = fold["groups"][concept]
            weights = [float(value) for value in group["weights"]]
            rows.append(
                _item(
                    category="K", catalogue_item_id=f"RES-P9-GAM-ALPHA-FOLD{fold_index}-{concept.upper()}",
                    entity_type="learned_alpha", phase="P9", model="learned_softmax_gam",
                    fold=fold_index, concept_or_target=concept, result_name="Fold-level learned expert mixture weights",
                    scientific_question="Did the local-expert mixture depart from uniform initialization?",
                    source_artifact_id="ART-P9-LEARNED-ALPHA", source_root_alias="repo://",
                    source_relative_path=path, source_field_path=f"folds[{fold_index}].groups.{concept}",
                    source_sha256=digest, row_or_sample_count=5, report_section_id="SEC-RESULTS-WHY",
                    report_table_ids=("RPT-T16",), report_figure_ids=("RPT-F11",),
                    report_usage_status="USED_MAIN_TEXT",
                    details={
                        "initial_weights": [0.2] * 5,
                        "final_weights": weights,
                        "logits": group["logits"],
                        "minimum_weight": min(weights), "maximum_weight": max(weights),
                        "simplex_sum": sum(weights),
                        "simplex_verified": abs(sum(weights) - 1.0) <= 1e-6,
                        "gradient_l1_at_best_epoch": group["gradient_l1_at_best_epoch"],
                    },
                )
            )
    return rows


def _spatial_rows(
    repository_root: Path, report: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    path = "artifacts/baseline_v2/audit/p9/spatial.json"
    digest = _source_hash(repository_root, path)
    detail: list[dict[str, Any]] = []
    rca: list[dict[str, Any]] = []
    faithfulness: list[dict[str, Any]] = []
    spatial_models = report["spatial"]["models"]
    for model in MODEL_ORDER:
        model_payload = spatial_models[model]
        for fold in model_payload["folds"]:
            fold_index = int(fold["fold_index"])
            for target in TARGETS_BY_MODEL[model]:
                payload = fold["targets"][target]
                valid = int(payload["valid_map_count"])
                undefined = int(payload["undefined_map_count"])
                requested = valid + undefined
                detail.append(
                    _item(
                        category="L",
                        catalogue_item_id=f"RES-P9-GRADCAM-{model.upper()}-FOLD{fold_index}-{target.upper()}",
                        entity_type="gradcam_model_fold_target",
                        phase="P9",
                        model=model,
                        fold=fold_index,
                        concept_or_target=target,
                        result_name="Raw FP32 Grad-CAM and occlusion inventory",
                        scientific_question="Where is this frozen target spatially sensitive?",
                        source_artifact_id="ART-P9-SPATIAL",
                        source_root_alias="repo://",
                        source_relative_path=path,
                        source_field_path=f"models.{model}.folds[{fold_index}].targets.{target}",
                        source_sha256=digest,
                        row_or_sample_count=requested,
                        privacy_class="PRIVATE_SOURCE_PUBLIC_AGGREGATE",
                        report_section_id="SEC-RESULTS-WHERE",
                        report_table_ids=("RPT-T13",),
                        report_figure_ids=("RPT-F07",),
                        report_usage_status="USED_MAIN_TEXT",
                        details={
                            "requested_map_count": requested,
                            "valid_map_count": valid,
                            "undefined_map_count": undefined,
                            "undefined_rate": undefined / requested if requested else 0.0,
                            "raw_map_dtype": "FP32",
                            "raw_map_normalized": False,
                            "raw_map_storage": f"mac-archive://p9/spatial/{model}/fold_{fold_index}/shards",
                            "occlusion_evidence_available": valid > 0,
                            "matched_random_masks_per_valid_map": 20,
                            "full_ct_case_feasibility": "REGISTERED_SEPARATELY_IN_CAT_Q",
                            "display_only_overlay_normalization": True,
                            "quantitative_faithfulness_uses_raw_fp32": True,
                        },
                    )
                )
        for target in TARGETS_BY_MODEL[model]:
            payload = model_payload["pooled_targets"][target]
            valid = int(payload["valid_map_count"])
            undefined = int(payload["undefined_map_count"])
            requested = valid + undefined
            detail.append(
                _item(
                    category="L",
                    catalogue_item_id=f"RES-P9-GRADCAM-{model.upper()}-POOLED-{target.upper()}",
                    entity_type="gradcam_model_target_pooled",
                    phase="P9",
                    model=model,
                    concept_or_target=target,
                    result_name="Pooled Grad-CAM accounting",
                    scientific_question="How often is this target map valid or post-ReLU all-zero?",
                    source_artifact_id="ART-P9-SPATIAL",
                    source_root_alias="repo://",
                    source_relative_path=path,
                    source_field_path=f"models.{model}.pooled_targets.{target}",
                    source_sha256=digest,
                    row_or_sample_count=requested,
                    privacy_class="PUBLIC_DEIDENTIFIED",
                    report_section_id="SEC-RESULTS-WHERE",
                    report_table_ids=("RPT-T13",),
                    report_figure_ids=("RPT-F07",),
                    report_usage_status="USED_MAIN_TEXT",
                    details={
                        "requested_map_count": requested,
                        "valid_map_count": valid,
                        "undefined_map_count": undefined,
                        "undefined_rate": undefined / requested if requested else 0.0,
                        "confirmed_undefined_semantics": "exactly all-zero post-ReLU Grad-CAM",
                    },
                )
            )
            rca.append(
                _item(
                    category="M",
                    catalogue_item_id=f"RES-P9-UNDEFINED-RCA-{model.upper()}-{target.upper()}",
                    entity_type="undefined_map_rca",
                    phase="P9",
                    model=model,
                    concept_or_target=target,
                    result_name="Undefined Grad-CAM root-cause classification",
                    scientific_question="Is the all-zero map expected, systematic, numeric, or an implementation error?",
                    source_artifact_id="ART-P9-SPATIAL",
                    source_root_alias="repo://",
                    source_relative_path=path,
                    source_field_path=f"models.{model}.pooled_targets.{target}",
                    source_sha256=digest,
                    row_or_sample_count=undefined,
                    report_section_id="SEC-LIMITATIONS",
                    report_table_ids=("RPT-T13",),
                    report_figure_ids=("RPT-F07", "RPT-FA05"),
                    report_usage_status="USED_MAIN_TEXT",
                    details={
                        "requested_map_count": requested,
                        "undefined_map_count": undefined,
                        "undefined_rate": undefined / requested if requested else 0.0,
                        "confirmed_observation": "post-ReLU Grad-CAM exactly all-zero",
                        "pre_relu_cam_statistics": "DATA_NOT_PERSISTED",
                        "gradient_norm": "DATA_NOT_PERSISTED",
                        "activation_norm": "DATA_NOT_PERSISTED",
                        "channel_weight_norm": "DATA_NOT_PERSISTED",
                        "numeric_underflow_nan_inf_evidence": "NOT_OBSERVED",
                        "implementation_or_loading_error_evidence": "NOT_OBSERVED",
                        "exact_mechanism": "UNRESOLVED_FROM_FROZEN_ARTIFACTS",
                        "root_cause_conclusion": report["gradcam_accounting"]["root_cause_conclusion"],
                    },
                )
            )
            faithfulness.append(
                _item(
                    category="N",
                    catalogue_item_id=f"RES-P9-FAITHFULNESS-{model.upper()}-{target.upper()}",
                    entity_type="spatial_faithfulness_model_target",
                    phase="P9",
                    model=model,
                    concept_or_target=target,
                    result_name="Matched-random spatial faithfulness",
                    scientific_question="Does saliency occlusion differ from matched random occlusion?",
                    source_artifact_id="ART-P9-SPATIAL",
                    source_root_alias="repo://",
                    source_relative_path=path,
                    source_field_path=f"models.{model}.pooled_targets.{target}",
                    source_sha256=digest,
                    row_or_sample_count=valid,
                    report_section_id="SEC-RESULTS-WHERE",
                    report_table_ids=("RPT-T14",),
                    report_figure_ids=("RPT-F08",),
                    report_usage_status="USED_MAIN_TEXT",
                    details={
                        "valid_map_count": valid,
                        "matched_random_values_per_valid_map": 20,
                        "output_sensitivity": payload.get("output_sensitivity"),
                        "error_increase": payload.get("error_increase"),
                        "interpretation": {
                            "output_sensitivity": "absolute output movement only",
                            "error_increase": "positive means prediction error worsened",
                        },
                    },
                )
            )
        faithfulness.append(
            _item(
                category="N",
                catalogue_item_id=f"RES-P9-FAITHFULNESS-{model.upper()}-POOLED",
                entity_type="spatial_faithfulness_model_pooled",
                phase="P9",
                model=model,
                result_name="All-target pooled spatial faithfulness",
                scientific_question="What is the model-wide saliency versus random pattern?",
                source_artifact_id="ART-P9-SPATIAL",
                source_root_alias="repo://",
                source_relative_path=path,
                source_field_path=f"models.{model}.pooled_all_targets",
                source_sha256=digest,
                row_or_sample_count=sum(int(v["valid_map_count"]) for v in model_payload["pooled_targets"].values()),
                report_section_id="SEC-RESULTS-WHERE",
                report_table_ids=("RPT-T14",),
                report_figure_ids=("RPT-F08",),
                report_usage_status="USED_MAIN_TEXT",
                details=model_payload["pooled_all_targets"],
            )
        )
    accounting = report["gradcam_accounting"]
    detail.append(
        _item(
            category="L",
            catalogue_item_id="RES-P9-GRADCAM-GLOBAL-ACCOUNTING",
            entity_type="gradcam_global_accounting",
            phase="P9",
            result_name="Global Grad-CAM accounting identity",
            scientific_question="Are all requested maps accounted for exactly?",
            source_artifact_id="ART-P9-SPATIAL",
            source_root_alias="repo://",
            source_relative_path=path,
            source_field_path="gradcam_accounting",
            source_sha256=digest,
            row_or_sample_count=int(accounting["requested"]),
            report_section_id="SEC-RESULTS-WHERE",
            report_table_ids=("RPT-T13",),
            report_figure_ids=("RPT-F07",),
            report_usage_status="USED_MAIN_TEXT",
            details=dict(accounting),
        )
    )
    faithfulness.append(
        _item(
            category="N",
            catalogue_item_id="RES-P9-FAITHFULNESS-GLOBAL",
            entity_type="spatial_faithfulness_global",
            phase="P9",
            result_name="Global faithfulness evidence availability",
            scientific_question="Which global faithfulness summaries were persisted without recomputation?",
            source_artifact_id="ART-P9-SPATIAL",
            source_root_alias="repo://",
            source_relative_path=path,
            source_field_path="models",
            source_sha256=digest,
            row_or_sample_count=int(accounting["valid"]),
            report_section_id="SEC-RESULTS-WHERE",
            report_table_ids=("RPT-T14",),
            report_figure_ids=("RPT-F08",),
            report_usage_status="USED_APPENDIX",
            availability_status="DATA_NOT_PERSISTED",
            omission_reason="A cross-model micro-average is not persisted and is not recomputed at Catalogue stage.",
            details={
                "model_target_aggregates_available": True,
                "model_pooled_aggregates_available": True,
                "cross_model_micro_average": "DATA_NOT_PERSISTED",
            },
        )
    )
    return detail, rca, faithfulness


PLANNED_TABLES = {
    "RPT-T01": ("Related-work comparison", "SEC-RELATED-WORK", "A"),
    "RPT-T02": ("Frozen cohort flow", "SEC-DATASET", "A"),
    "RPT-T03": ("Target and concept definitions", "SEC-DATASET", "G"),
    "RPT-T04": ("Four-model architecture comparison", "SEC-METHODS", "A"),
    "RPT-T05": ("Frozen training configuration", "SEC-EXPERIMENTAL-SETUP", "B"),
    "RPT-T06": ("Evaluation protocol", "SEC-EXPERIMENTAL-SETUP", "A"),
    "RPT-T07": ("Primary regression", "SEC-RESULTS-PREDICTION", "C"),
    "RPT-T08": ("Six paired Delta-MAE comparisons", "SEC-RESULTS-PREDICTION", "D"),
    "RPT-T09": ("Extreme-task performance", "SEC-RESULTS-PREDICTION", "E"),
    "RPT-T10": ("Six paired Delta-AUROC comparisons", "SEC-RESULTS-PREDICTION", "F"),
    "RPT-T11": ("Continuous concept metrics", "SEC-RESULTS-WHAT", "G"),
    "RPT-T12": ("Categorical concept metrics", "SEC-RESULTS-WHAT", "H"),
    "RPT-T13": ("Grad-CAM accounting", "SEC-RESULTS-WHERE", "L"),
    "RPT-T14": ("Spatial faithfulness", "SEC-RESULTS-WHERE", "N"),
    "RPT-T15": ("Centered contribution summary", "SEC-RESULTS-WHY", "J"),
    "RPT-T16": ("Fold-level learned GAM alpha", "SEC-RESULTS-WHY", "K"),
    "RPT-T17": ("Intervention summary", "SEC-RESULTS-HOW", "I"),
    "RPT-T18": ("WHERE-WHAT-WHY-HOW synthesis", "SEC-RESULTS-SYNTHESIS", "S"),
    "RPT-TA01": ("Frozen 14-case index", "SEC-PRIVATE-APPENDIX", "Q"),
    "RPT-TA02": ("Case-level concept and malignancy predictions", "SEC-PRIVATE-APPENDIX", "Q"),
}

PLANNED_FIGURES = {
    "RPT-F01": ("End-to-end evidence pipeline", "SEC-INTRODUCTION", "A"),
    "RPT-F02": ("Cohort, preprocessing, and five-fold flow", "SEC-DATASET", "A"),
    "RPT-F03": ("Four architectures and interpretability interfaces", "SEC-METHODS", "A"),
    "RPT-F04": ("Four-model MAE bootstrap intervals", "SEC-RESULTS-PREDICTION", "C"),
    "RPT-F05": ("Paired Delta-MAE forest plot", "SEC-RESULTS-PREDICTION", "D"),
    "RPT-F06": ("Extreme AUROC/AUPRC and paired Delta-AUROC", "SEC-RESULTS-PREDICTION", "E,F"),
    "RPT-F07": ("Undefined Grad-CAM rate heatmap", "SEC-RESULTS-WHERE", "L,M"),
    "RPT-F08": ("Spatial faithfulness dual panel", "SEC-RESULTS-WHERE", "N"),
    "RPT-F09A": ("Continuous concept fidelity", "SEC-RESULTS-WHAT", "G"),
    "RPT-F09B": ("Categorical concept fidelity", "SEC-RESULTS-WHAT", "H"),
    "RPT-F10": ("Empirical OOF contribution profiles", "SEC-RESULTS-WHY", "J"),
    "RPT-F11": ("GAM alpha heatmap", "SEC-RESULTS-WHY", "K"),
    "RPT-F12": ("Intervention curves", "SEC-RESULTS-HOW", "I"),
    "RPT-F13": ("Integrated evidence synthesis", "SEC-RESULTS-SYNTHESIS", "S"),
    "RPT-FA01": ("Representative-case comparison", "SEC-PRIVATE-APPENDIX", "Q"),
    "RPT-FA02": ("Maximum-error failure comparison", "SEC-PRIVATE-APPENDIX", "Q"),
    "RPT-FA03": ("Concept contribution explanation", "SEC-PRIVATE-APPENDIX", "Q"),
    "RPT-FA04": ("Intervention-worsening cases", "SEC-PRIVATE-APPENDIX", "Q"),
    "RPT-FA05": ("Undefined zero-map limitation", "SEC-PRIVATE-APPENDIX", "Q"),
    "RPT-FA06": ("Integrated Prediction-WHERE-WHAT-WHY-HOW case explanation", "SEC-PRIVATE-APPENDIX", "Q"),
}


def _table_figure_rows(
    repository_root: Path, *, fa06_selected_case_label: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plan_path = "docs/results/P10_CATALOGUE_DRIVEN_BILINGUAL_REPORT_PLAN.md"
    plan_sha = _source_hash(repository_root, plan_path)
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    for identifier, (title, section, category) in PLANNED_TABLES.items():
        private = identifier.startswith("RPT-TA")
        tables.append(
            _item(
                category="O", catalogue_item_id=f"RES-P10-TABLE-{identifier}",
                entity_type="planned_scientific_table", phase="P10", result_name=title,
                scientific_question="Which frozen evidence supports this planned report table?",
                source_artifact_id="ART-P10-REPORT-PLAN", source_root_alias="repo://",
                source_relative_path=plan_path, source_field_path=identifier, source_sha256=plan_sha,
                privacy_class="PRIVATE_RESTRICTED" if private else "PUBLIC_DEIDENTIFIED",
                report_section_id=section, report_table_ids=(identifier,),
                report_usage_status="USED_PRIVATE_APPENDIX" if private else "USED_MAIN_TEXT",
                availability_status="VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
                details={
                    "planned_table_id": identifier, "evidence_categories": category,
                    "render_status": "NOT_RENDERED_AT_CATALOGUE_GATE",
                    "new_inference_required": False,
                    "mean_absolute_centered_contribution_policy": (
                        "UNAVAILABLE_FROM_AUTHORITATIVE_FROZEN_SOURCE; do not use transient narrative "
                        "values or recompute from per-sample contributions"
                        if identifier == "RPT-T15" else None
                    ),
                    "categorical_target_semantics": (
                        "full reader vote distribution retained; modal label display-only"
                        if identifier == "RPT-TA02" else None
                    ),
                },
            )
        )
    existing_tables = repository_root / "reports/baseline_v2/p10/public/tables"
    for path in sorted(existing_tables.glob("*.csv")):
        rel = path.relative_to(repository_root).as_posix()
        tables.append(
            _item(
                category="O", catalogue_item_id=f"RES-P10-EXISTING-TABLE-{path.stem.upper()}",
                entity_type="existing_machine_readable_table", phase="P10",
                result_name=f"Existing table: {path.name}",
                scientific_question="Which current machine-readable table already exists?",
                source_artifact_id=f"ART-P10-TABLE-{path.stem.upper()}", source_root_alias="repo://",
                source_relative_path=rel, source_field_path="$", source_sha256=sha256_file(path),
                row_or_sample_count=max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1),
                report_section_id="SEC-PUBLIC-REPRODUCIBILITY", report_usage_status="USED_APPENDIX",
                details={"existing": True, "revision_needed": True, "modified_by_catalogue": False},
            )
        )
    for identifier, (title, section, category) in PLANNED_FIGURES.items():
        private = identifier.startswith("RPT-FA")
        how_missing = identifier == "RPT-FA06"
        figures.append(
            _item(
                category="P", catalogue_item_id=f"RES-P10-FIGURE-{identifier}",
                entity_type="planned_scientific_figure", phase="P10", result_name=title,
                scientific_question="Which frozen evidence supports this planned report figure?",
                source_artifact_id="ART-P10-REPORT-PLAN", source_root_alias="repo://",
                source_relative_path=plan_path, source_field_path=identifier, source_sha256=plan_sha,
                privacy_class="PRIVATE_RESTRICTED" if private else "PUBLIC_DEIDENTIFIED",
                report_section_id=section, report_figure_ids=(identifier,),
                report_usage_status="USED_PRIVATE_APPENDIX" if private else "USED_MAIN_TEXT",
                availability_status="VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
                details={
                    "planned_figure_id": identifier, "evidence_categories": category,
                    "render_status": "NOT_RENDERED_AT_CATALOGUE_GATE",
                    "quantitative_summary_requires_ct": False if not private else None,
                    "case_level_intervention_component": "DATA_NOT_PERSISTED" if how_missing else "NOT_APPLICABLE",
                    "component_availability": (
                        {
                            "prediction": "RESULT_ALREADY_EXISTS",
                            "where": "RESULT_ALREADY_EXISTS",
                            "what": "RESULT_ALREADY_EXISTS",
                            "why": "RESULT_ALREADY_EXISTS",
                            "how": "DATA_NOT_PERSISTED",
                        }
                        if how_missing else None
                    ),
                    "selected_case_label": fa06_selected_case_label if how_missing else None,
                    "missing_component_policy": (
                        "Render the integrated figure with an explicit DATA_NOT_PERSISTED HOW placeholder."
                        if how_missing else None
                    ),
                    "new_inference_required": False,
                },
            )
        )
    existing_figures = repository_root / "reports/baseline_v2/p10/public/figures"
    for path in sorted(existing_figures.glob("*.*")):
        if path.suffix.lower() not in {".png", ".svg"}:
            continue
        rel = path.relative_to(repository_root).as_posix()
        figures.append(
            _item(
                category="P", catalogue_item_id=f"RES-P10-EXISTING-FIGURE-{path.stem.upper()}-{path.suffix[1:].upper()}",
                entity_type="existing_public_figure", phase="P10", result_name=f"Existing figure: {path.name}",
                scientific_question="Which current public figure asset already exists?",
                source_artifact_id=f"ART-P10-FIGURE-{path.stem.upper()}", source_root_alias="repo://",
                source_relative_path=rel, source_field_path="$", source_sha256=sha256_file(path),
                scientific_status="LEGACY_PRESENTATION_ONLY",
                report_section_id="SEC-PUBLIC-REPRODUCIBILITY", report_usage_status="AUDIT_ONLY",
                details={
                    "existing": True,
                    "presentation_class": "LEGACY_PRESENTATION_ONLY",
                    "revision_needed": True,
                    "modified_by_catalogue": False,
                    "may_satisfy_planned_report_requirement": False,
                    "authoritative_rewrite_input": False,
                },
            )
        )
    return tables, figures


def _json_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError("P10_CATALOGUE_VECTOR_REQUIRED")
    result = [float(item) for item in value]
    if not result or not np.isfinite(result).all():
        raise ValueError("P10_CATALOGUE_VECTOR_INVALID")
    return result


def _selected_oof_rows(
    private_root: Path, cases: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    wanted: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        wanted[str(case["model"])].add(str(case["nodule_uid"]))
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for model, uids in wanted.items():
        table = pq.read_table(private_root / OOF_RELATIVE_BY_MODEL[model])
        for row in table.to_pylist():
            uid = str(row["nodule_uid"])
            if uid in uids:
                output[(model, uid)] = row
    if len(output) != sum(len(values) for values in wanted.values()):
        raise ValueError("P10_CATALOGUE_QUALITATIVE_OOF_ROW_MISSING")
    return output


def _selected_spatial_status(
    private_root: Path, cases: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    wanted: dict[tuple[str, int], set[str]] = defaultdict(set)
    for case in cases:
        wanted[(str(case["model"]), int(case["fold_index"]))].add(str(case["nodule_uid"]))
    output: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for (model, fold), uids in wanted.items():
        directory = private_root / "p9" / "spatial" / model / f"fold_{fold}"
        for shard in sorted(directory.glob("shard_*.parquet")):
            table = pq.read_table(shard, columns=["nodule_uid", "target", "status", "map_sha256"])
            for row in table.to_pylist():
                uid = str(row["nodule_uid"])
                if uid in uids:
                    output[(model, fold, uid, str(row["target"]))] = {
                        "status": str(row["status"]),
                        "map_sha256": str(row["map_sha256"]),
                        "shard_relative_path": shard.relative_to(private_root).as_posix(),
                    }
    return output


def _exact_ct_series(
    raw_data_root: Path,
    patient_id: str,
    study_uid: str,
    series_uid: str,
) -> dict[str, Any]:
    patient_root = raw_data_root / "manifest-1600709154662" / "LIDC-IDRI" / patient_id
    if not patient_root.is_dir():
        return {"available": False, "reason": "ORIGINAL_FROZEN_CT_SOURCE_MISSING"}
    matching: list[Path] = []
    for candidate in patient_root.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() != ".dcm":
            continue
        try:
            with candidate.open("rb") as handle:
                header = handle.read(512 * 1024)
        except OSError:
            continue
        # DICOM UI values are persisted as exact ASCII strings in the frozen source.
        # This is a read-only provenance check; pixels are never decoded.
        if study_uid.encode("ascii") in header and series_uid.encode("ascii") in header:
            matching.append(candidate)
    if not matching:
        return {"available": False, "reason": "EXACT_SERIES_NOT_FOUND"}
    directories = {path.parent for path in matching}
    if len(directories) != 1:
        return {"available": False, "reason": "EXACT_SERIES_DIRECTORY_AMBIGUOUS"}
    directory = next(iter(directories))
    return {
        "available": True,
        "source_directory": str(directory),
        "dicom_file_count": len(matching),
        "series_instance_uid": series_uid,
        "study_instance_uid": study_uid,
    }


def _full_ct_display_gate(
    *,
    ct: Mapping[str, Any],
    bbox: Any,
    roi_available: bool,
    z_index: int,
    expected_study_uid: str,
    expected_series_uid: str,
) -> dict[str, bool]:
    """Fail closed unless source, exact slice provenance, and ROI mapping all exist."""

    source_available = bool(ct.get("available"))
    exact_series = bool(
        source_available
        and str(ct.get("study_instance_uid", "")) == expected_study_uid
        and str(ct.get("series_instance_uid", "")) == expected_series_uid
        and int(ct.get("dicom_file_count", 0)) > 0
    )
    bbox_valid = bool(
        isinstance(bbox, list)
        and len(bbox) == 3
        and all(
            isinstance(axis, list)
            and len(axis) == 2
            and all(isinstance(value, int) for value in axis)
            and 0 <= axis[0] < axis[1]
            for axis in bbox
        )
    )
    slice_provenance = bool(
        exact_series
        and isinstance(z_index, int)
        and 0 <= z_index < int(ct.get("dicom_file_count", 0))
        and bbox_valid
        and bbox[0][0] <= z_index < bbox[0][1]
    )
    roi_mapping = bool(roi_available and bbox_valid and slice_provenance)
    return {
        "full_ct_source_available": source_available,
        "series_and_slice_provenance_available": slice_provenance,
        "roi_to_full_volume_mapping_available": roi_mapping,
        "read_only_full_ct_renderable": bool(source_available and slice_provenance and roi_mapping),
    }


def _case_scientific_values(model: str, row: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {
        "malignancy_prediction_1_to_5": float(row["malignancy_score_1_to_5"]),
        "malignancy_radiologist_mean_target_1_to_5": float(row["target_1_to_5"]),
        "concepts": {},
    }
    if model == "blackbox":
        return values
    targets = json.loads(row["concept_targets"]) if model == "standard_cbm" else None
    for concept in CONCEPT_ORDER:
        prediction = _json_vector(row[f"{concept}_activated_prediction"])
        if model == "standard_cbm":
            target = targets[concept]
            target_vector = [float(target)] if not isinstance(target, list) else [float(x) for x in target]
            contribution_key = f"{concept}_rating_point_contribution"
        else:
            target_vector = _json_vector(row[f"{concept}_target"])
            contribution_key = f"{concept}_rating_contribution"
        categorical = concept in CATEGORICAL_CONCEPTS
        values["concepts"][concept] = {
            "prediction": prediction,
            "reader_target": target_vector,
            "target_semantics": "full_reader_vote_distribution" if categorical else "radiologist_mean_normalized",
            "prediction_modal_class_display_only": int(np.argmax(prediction)) + 1 if categorical else None,
            "target_modal_class_display_only": int(np.argmax(target_vector)) + 1 if categorical else None,
            "signed_contribution_rating_points": float(row[contribution_key]),
        }
    return values


def _qualitative_rows(
    repository_root: Path,
    private_root: Path,
    raw_data_root: Path,
    case_index: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases = list(case_index["cases"])
    oof = _selected_oof_rows(private_root, cases)
    spatial = _selected_spatial_status(private_root, cases)
    nodules_table = pq.read_table(repository_root / "artifacts/baseline_v2/manifests/nodules.parquet")
    nodules = {str(row["nodule_uid"]): row for row in nodules_table.to_pylist()}
    roi_table = pq.read_table(repository_root / "artifacts/baseline_v2/manifests/roi_index.parquet")
    rois = {str(row["nodule_uid"]): row for row in roi_table.to_pylist()}
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    role_map = {
        "median_error_representative": "representative",
        "maximum_error_failure": "failure",
        "maximum_positive_error_worsening": "intervention_worsening",
        "highest_undefined_rate_concept_zero_map": "undefined_zero_map",
    }
    private_index_sha = sha256_file(private_root / PRIVATE_CASE_INDEX_RELATIVE)
    for case in sorted(cases, key=lambda value: str(value["case_label"])):
        uid = str(case["nodule_uid"])
        model = str(case["model"])
        fold = int(case["fold_index"])
        label = str(case["case_label"])
        nodule = nodules[uid]
        roi = rois[uid]
        bbox = json.loads(roi["bbox_dhw"])
        z_index = (int(bbox[0][0]) + int(bbox[0][1]) - 1) // 2
        ct = _exact_ct_series(
            raw_data_root,
            str(nodule["patient_id"]),
            str(nodule["study_instance_uid"]),
            str(nodule["series_instance_uid"]),
        )
        roi_path = repository_root / "artifacts/baseline_v2" / str(roi["relative_roi_path"])
        roi_available = roi_path.is_file() and sha256_file(roi_path) == str(roi["roi_file_sha256"])
        ct_gate = _full_ct_display_gate(
            ct=ct,
            bbox=bbox,
            roi_available=roi_available,
            z_index=z_index,
            expected_study_uid=str(nodule["study_instance_uid"]),
            expected_series_uid=str(nodule["series_instance_uid"]),
        )
        full_reprojection = ct_gate["read_only_full_ct_renderable"]
        target_status: dict[str, str] = {}
        target_sources: dict[str, str] = {}
        for target in TARGETS_BY_MODEL[model]:
            record = spatial.get((model, fold, uid, target))
            target_status[target] = str(record["status"]) if record else "DATA_NOT_PERSISTED"
            target_sources[target] = str(record["shard_relative_path"]) if record else ""
        science = _case_scientific_values(model, oof[(model, uid)])
        role = role_map.get(str(case["role"]), str(case["role"]))
        concept_evidence = model != "blackbox"
        required_targets = ("malignancy", "spiculation", "margin", "texture")
        support = {
            "original_full_axial_ct_slice": "FULL_CT_CONTEXT_AVAILABLE" if full_reprojection else "NOT_RENDERABLE_FROM_FROZEN_DATA",
            "full_ct_with_roi_box": "FULL_CT_CONTEXT_AVAILABLE" if full_reprojection else "FULL_SLICE_REPROJECTION_NOT_AVAILABLE_FROM_FROZEN_DATA",
            "zoomed_roi_crop": "ROI_ONLY_AVAILABLE" if roi_available else "NOT_RENDERABLE_FROM_FROZEN_DATA",
            "roi_plus_gradcam_overlay": "GRADCAM_AVAILABLE" if target_status.get(str(case["target"])) == "valid" else "GRADCAM_UNDEFINED_ZERO_MAP",
            "full_slice_gradcam_reprojection": "FULL_SLICE_REPROJECTION_AVAILABLE" if full_reprojection else "FULL_SLICE_REPROJECTION_NOT_AVAILABLE_FROM_FROZEN_DATA",
            "malignancy_gradcam": "GRADCAM_AVAILABLE" if target_status.get("malignancy") == "valid" else "GRADCAM_UNDEFINED_ZERO_MAP",
            "spiculation_gradcam": target_status.get("spiculation", "NOT_RENDERABLE_FROM_FROZEN_DATA"),
            "margin_gradcam": target_status.get("margin", "NOT_RENDERABLE_FROM_FROZEN_DATA"),
            "texture_gradcam": target_status.get("texture", "NOT_RENDERABLE_FROM_FROZEN_DATA"),
            "case_prediction_gt_table": "RESULT_ALREADY_EXISTS" if concept_evidence else "NOT_APPLICABLE",
            "case_centered_contribution_bars": "RESULT_ALREADY_EXISTS" if concept_evidence else "NOT_APPLICABLE",
            "undefined_zero_map_panel": "GRADCAM_UNDEFINED_ZERO_MAP" if "undefined" in target_status.values() else "NOT_APPLICABLE",
            "case_level_intervention_before_after": "DATA_NOT_PERSISTED",
            "integrated_prediction_where_what_why_how": "PENDING_DETERMINISTIC_FA06_SELECTION",
        }
        public_details = {
            "case_label": label,
            "case_role": role,
            "selected_target": str(case["target"]),
            "map_status": str(case["map_status"]),
            "malignancy_prediction_1_to_5": science["malignancy_prediction_1_to_5"],
            "malignancy_target_1_to_5": science["malignancy_radiologist_mean_target_1_to_5"],
            **ct_gate,
            "frozen_context_z_index": z_index,
            "roi_bbox_dhw": bbox,
            "zoomed_roi_available": roi_available,
            "gradcam_status_by_target": target_status,
            "display_windowing_policy": "lung_window_from_frozen_CT; display_only",
            "overlay_normalization_policy": "display-only per-map normalization; raw FP32 unchanged",
            "caption_warning_required": "ROI is the 64-cubed model input, not a full CT slice; faithfulness used unnormalized raw FP32 maps.",
            "ta02_available": concept_evidence,
            "fa06_component_availability": {
                "prediction": "RESULT_ALREADY_EXISTS",
                "where": "RESULT_ALREADY_EXISTS",
                "what": "RESULT_ALREADY_EXISTS" if concept_evidence else "NOT_APPLICABLE",
                "why": "RESULT_ALREADY_EXISTS" if concept_evidence else "NOT_APPLICABLE",
                "how": "DATA_NOT_PERSISTED",
            },
            "fourteen_component_support": support,
            "new_inference_required": False,
        }
        figures = ["RPT-FA01" if role == "representative" else "RPT-FA02" if role == "failure" else "RPT-FA04" if role == "intervention_worsening" else "RPT-FA05"]
        if concept_evidence:
            figures.append("RPT-FA03")
        public_rows.append(
            _item(
                category="Q", catalogue_item_id=f"RES-P10-QUALITATIVE-{label}",
                entity_type="qualitative_case", phase="P10", model=model, fold=fold,
                concept_or_target=str(case["target"]), result_name=f"Frozen qualitative case {label}",
                scientific_question="Which frozen assets support a paper-style deidentified case explanation?",
                source_artifact_id="ART-P10-PRIVATE-CASE-INDEX", source_root_alias="private-report://",
                source_relative_path="private_case_index.json", source_field_path=f"cases.{label}",
                source_sha256=private_index_sha, row_or_sample_count=1, privacy_class="PRIVATE_RESTRICTED",
                report_section_id="SEC-PRIVATE-APPENDIX", report_table_ids=("RPT-TA01", "RPT-TA02"),
                report_figure_ids=tuple(figures), report_usage_status="USED_PRIVATE_APPENDIX",
                details=public_details,
            )
        )
        private_rows.append(
            {
                "case_label": label,
                "model": model,
                "fold": fold,
                "role": role,
                "restricted_nodule_identifier": uid,
                "restricted_patient_identifier": str(case["patient_key"]),
                "source_patient_id": str(nodule["patient_id"]),
                "study_instance_uid": str(nodule["study_instance_uid"]),
                "series_instance_uid": str(nodule["series_instance_uid"]),
                "full_ct_source_directory": ct.get("source_directory", ""),
                "full_ct_dicom_file_count": ct.get("dicom_file_count", 0),
                "context_z_index": z_index,
                "roi_bbox_dhw": json.dumps(bbox, separators=(",", ":")),
                "roi_source_path": str(roi_path),
                "roi_source_sha256": str(roi["roi_file_sha256"]),
                "spatial_shards_by_target": json.dumps(target_sources, sort_keys=True, separators=(",", ":")),
                "scientific_values": json.dumps(science, sort_keys=True, separators=(",", ":")),
                "case_level_intervention_evidence_available": False,
                "model_forward": False,
            }
        )
    fa06_candidates = sorted(
        (
            item
            for item in public_rows
            if item["model"] != "blackbox"
            and item["details"]["read_only_full_ct_renderable"]
            and all(
                status == "valid"
                for status in item["details"]["gradcam_status_by_target"].values()
            )
            and item["details"]["fa06_component_availability"]["what"] == "RESULT_ALREADY_EXISTS"
            and item["details"]["fa06_component_availability"]["why"] == "RESULT_ALREADY_EXISTS"
        ),
        key=lambda item: item["details"]["case_label"],
    )
    if not fa06_candidates:
        raise ValueError("P10_CATALOGUE_FA06_CASE_NOT_AVAILABLE")
    selected_fa06_label = fa06_candidates[0]["details"]["case_label"]
    for item in public_rows:
        selected = item["details"]["case_label"] == selected_fa06_label
        item["details"]["fa06_selected"] = selected
        item["details"]["fa06_case_role"] = "integrated_explanation" if selected else "NOT_SELECTED"
        item["details"]["fa06_selection_rule"] = (
            "lowest CASE label among concept-model cases with full CT provenance/mapping, "
            "frozen WHAT/WHY values, and valid Grad-CAM for every registered model target"
        )
        item["details"]["fourteen_component_support"]["integrated_prediction_where_what_why_how"] = (
            "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS_WITH_HOW_PLACEHOLDER"
            if selected else "NOT_SELECTED_FOR_FA06"
        )
        if selected:
            item["report_figure_ids"] = [*item["report_figure_ids"], "RPT-FA06"]
    for row in private_rows:
        selected = row["case_label"] == selected_fa06_label
        row["fa06_selected"] = selected
        row["fa06_case_role"] = "integrated_explanation" if selected else "NOT_SELECTED"
    return public_rows, private_rows


def _storage_rows(
    repository_root: Path,
    private_root: Path,
    archive_by_path: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    private_locations: list[dict[str, Any]] = []
    groups = {
        "blackbox": "P5 Black-box",
        "standard_cbm": "P6 Standard CBM",
        "cem": "P7 Mixed-type CEM",
        "gam": "P8 Learned-softmax GAM",
        "p9": "P9 unified evaluation and spatial evidence",
    }
    for prefix, title in groups.items():
        members = [row for path, row in archive_by_path.items() if path == prefix or path.startswith(f"{prefix}/")]
        rows.append(
            _item(
                category="R", catalogue_item_id=f"ART-STORAGE-GROUP-{prefix.upper()}",
                entity_type="storage_group", phase="P9" if prefix == "p9" else PHASE_BY_MODEL[{"blackbox":"blackbox","standard_cbm":"standard_cbm","cem":"mixed_cem","gam":"learned_softmax_gam"}[prefix]],
                result_name=title, scientific_question="Where is this complete frozen result family stored?",
                source_artifact_id=f"ART-ARCHIVE-GROUP-{prefix.upper()}", source_root_alias="mac-archive://",
                source_relative_path=prefix, source_field_path="$", source_sha256=hashlib.sha256(_canonical_json_bytes(members)).hexdigest(),
                row_or_sample_count=len(members), privacy_class="PRIVATE_RESTRICTED",
                report_section_id="SEC-PUBLIC-REPRODUCIBILITY", report_usage_status="AUDIT_ONLY",
                details={"file_count": len(members), "total_bytes": sum(int(row["size_bytes"]) for row in members), "human_master_include": True},
            )
        )
    for index, (relative, archive_row) in enumerate(sorted(archive_by_path.items()), start=1):
        first = relative.split("/", 1)[0]
        phase = "P9" if first == "p9" else PHASE_BY_MODEL.get({"blackbox":"blackbox","standard_cbm":"standard_cbm","cem":"mixed_cem","gam":"learned_softmax_gam"}.get(first, ""), "P10")
        rows.append(
            _item(
                category="R", catalogue_item_id=f"ART-STORAGE-ARCHIVE-{index:04d}",
                entity_type="archived_file", phase=phase, result_name=f"Archived file: {Path(relative).name}",
                scientific_question="Where is this verified private file stored?",
                source_artifact_id=f"ART-ARCHIVE-FILE-{index:04d}", source_root_alias="mac-archive://",
                source_relative_path=relative, source_field_path="$", source_sha256=str(archive_row["sha256"]),
                row_or_sample_count=None, privacy_class="PRIVATE_RESTRICTED",
                report_section_id="SEC-PUBLIC-REPRODUCIBILITY", report_usage_status="AUDIT_ONLY",
                details={"byte_size": int(archive_row["size_bytes"]), "human_master_include": False},
            )
        )
        remote = f"/srv/scratch/z5448417/lidc-baseline-v2/runs/baseline_v2/{relative}"
        private_locations.append(
            {
                "catalogue_item_id": f"ART-STORAGE-ARCHIVE-{index:04d}",
                "root_alias": "mac-archive://",
                "relative_path": relative,
                "mac_absolute_path": str(private_root / relative),
                "katana_absolute_path": remote,
                "byte_size": int(archive_row["size_bytes"]),
                "sha256": str(archive_row["sha256"]),
                "privacy_class": "PRIVATE_RESTRICTED",
            }
        )
    tracked_roots = (
        repository_root / "artifacts/baseline_v2/audit",
        repository_root / "reports/baseline_v2/p10/public",
        repository_root / "docs/results",
        repository_root / "configs/experiments",
    )
    tracked_files: set[Path] = set()
    for root in tracked_roots:
        if root.is_dir():
            tracked_files.update(
                path for path in root.rglob("*")
                if path.is_file()
                and path.name != ".DS_Store"
                and not (
                    repository_root / "docs/results" in path.parents
                    and (
                        "catalogue_tables" in path.parts
                        or path.name.startswith("results_catalogue_")
                        or path.name.startswith("results_master_catalogue")
                        or path.name.startswith("RESULTS_MASTER_CATALOGUE")
                        or path.name.startswith("RESULTS_ARTIFACTS_MASTER_TABLE")
                        or path.name in {
                            "catalogue_manifest.json", "artifacts_inventory.csv",
                            "tables_inventory.csv", "figures_inventory.csv",
                            "qualitative_case_inventory.csv", "missing_incomplete_outputs.csv",
                            "report_evidence_map.csv", "public_private_storage_map.csv",
                            "catalogue_to_report_plan.csv", PHASE_STATUS_SNAPSHOT_NAME,
                        }
                    )
                )
            )
    for path in sorted(tracked_files):
        relative = path.relative_to(repository_root).as_posix()
        digest = sha256_file(path)
        item_id = f"ART-STORAGE-REPO-{hashlib.sha256(relative.encode()).hexdigest()[:12].upper()}"
        legacy_presentation = (
            relative.startswith("reports/baseline_v2/p10/public/")
            and (
                relative.endswith(".pdf")
                or "/figures/" in relative
            )
        )
        rows.append(
            _item(
                category="R", catalogue_item_id=item_id,
                entity_type="legacy_presentation_file" if legacy_presentation else "tracked_repository_file",
                phase="P10",
                result_name=f"Repository evidence: {path.name}", scientific_question="Where is this tracked public evidence stored?",
                source_artifact_id=item_id, source_root_alias="repo://", source_relative_path=relative,
                source_field_path="$", source_sha256=digest, privacy_class="PUBLIC_DEIDENTIFIED",
                scientific_status="LEGACY_PRESENTATION_ONLY" if legacy_presentation else "PASS",
                report_section_id="SEC-PUBLIC-REPRODUCIBILITY", report_usage_status="AUDIT_ONLY",
                details={
                    "byte_size": path.stat().st_size,
                    "human_master_include": relative.startswith("artifacts/baseline_v2/audit/p9/"),
                    "presentation_class": "LEGACY_PRESENTATION_ONLY" if legacy_presentation else "NOT_APPLICABLE",
                    "may_satisfy_planned_report_requirement": False if legacy_presentation else None,
                    "authoritative_rewrite_input": False if legacy_presentation else None,
                },
            )
        )
        private_locations.append(
            {
                "catalogue_item_id": item_id, "root_alias": "repo://", "relative_path": relative,
                "mac_absolute_path": str(path), "katana_absolute_path": "NOT_APPLICABLE",
                "byte_size": path.stat().st_size, "sha256": digest, "privacy_class": "PUBLIC_DEIDENTIFIED",
            }
        )
    private_report = private_root / "p10_private_report"
    for path in sorted(private_report.rglob("*")):
        if (
            not path.is_file()
            or "results_catalogue" in path.parts
            or "xlsx_qa_previews" in path.parts
            or path.name == ".DS_Store"
            or path.name in {
                PRIVATE_MASTER_XLSX, PRIVATE_HUMAN_XLSX, PRIVATE_LOCATIONS_NAME,
                PRIVATE_COMPLETE_NAME, "PRIVATE_XLSX_QA.json",
                "qualitative_case_private_overlay.csv",
                f"{PRIVATE_MASTER_XLSX}.inspect.ndjson",
                f"{PRIVATE_HUMAN_XLSX}.inspect.ndjson",
            }
        ):
            continue
        relative = path.relative_to(private_report).as_posix()
        digest = sha256_file(path)
        item_id = f"ART-STORAGE-PRIVATE-REPORT-{hashlib.sha256(relative.encode()).hexdigest()[:12].upper()}"
        legacy_presentation = path.suffix.lower() in {".pdf", ".png", ".svg"}
        rows.append(
            _item(
                category="R", catalogue_item_id=item_id,
                entity_type="legacy_private_presentation_file" if legacy_presentation else "private_report_file",
                phase="P10",
                result_name=f"Private report asset: {path.name}", scientific_question="Where is this private appendix asset stored?",
                source_artifact_id=item_id, source_root_alias="private-report://", source_relative_path=relative,
                source_field_path="$", source_sha256=digest, privacy_class="PRIVATE_RESTRICTED",
                scientific_status="LEGACY_PRESENTATION_ONLY" if legacy_presentation else "PASS",
                report_section_id="SEC-PRIVATE-APPENDIX",
                report_usage_status="AUDIT_ONLY" if legacy_presentation else "USED_PRIVATE_APPENDIX",
                details={
                    "byte_size": path.stat().st_size,
                    "human_master_include": path.suffix.lower() in {".pdf", ".md", ".json"},
                    "presentation_class": "LEGACY_PRESENTATION_ONLY" if legacy_presentation else "NOT_APPLICABLE",
                    "may_satisfy_planned_report_requirement": False if legacy_presentation else None,
                    "authoritative_rewrite_input": False if legacy_presentation else None,
                },
            )
        )
        private_locations.append(
            {
                "catalogue_item_id": item_id, "root_alias": "private-report://", "relative_path": relative,
                "mac_absolute_path": str(path), "katana_absolute_path": "NOT_APPLICABLE",
                "byte_size": path.stat().st_size, "sha256": digest, "privacy_class": "PRIVATE_RESTRICTED",
            }
        )
    return rows, private_locations


def _evidence_rows(repository_root: Path) -> list[dict[str, Any]]:
    plan_path = "docs/results/P10_CATALOGUE_DRIVEN_BILINGUAL_REPORT_PLAN.md"
    digest = _source_hash(repository_root, plan_path)
    specifications = {
        "SEC-DATASET": ("Dataset and preprocessing", ["RES-P0-PHASE-OVERVIEW", "RES-P2-PHASE-OVERVIEW", "RES-P3-PHASE-OVERVIEW", "RES-P4-PHASE-OVERVIEW"], ["RPT-T02", "RPT-T03"], ["RPT-F02"]),
        "SEC-METHODS": ("Model design", ["RES-P5-PHASE-OVERVIEW", "RES-P6-PHASE-OVERVIEW", "RES-P7-PHASE-OVERVIEW", "RES-P8-PHASE-OVERVIEW"], ["RPT-T04"], ["RPT-F03"]),
        "SEC-RESULTS-PREDICTION": ("Prediction", ["CAT-C", "CAT-D", "CAT-E", "CAT-F"], ["RPT-T07", "RPT-T08", "RPT-T09", "RPT-T10"], ["RPT-F04", "RPT-F05", "RPT-F06"]),
        "SEC-RESULTS-WHERE": ("WHERE", ["CAT-L", "CAT-M", "CAT-N"], ["RPT-T13", "RPT-T14"], ["RPT-F07", "RPT-F08"]),
        "SEC-RESULTS-WHAT": ("WHAT", ["CAT-G", "CAT-H"], ["RPT-T11", "RPT-T12"], ["RPT-F09A", "RPT-F09B"]),
        "SEC-RESULTS-WHY": ("WHY", ["CAT-J", "CAT-K"], ["RPT-T15", "RPT-T16"], ["RPT-F10", "RPT-F11"]),
        "SEC-RESULTS-HOW": ("HOW", ["CAT-I"], ["RPT-T17"], ["RPT-F12"]),
        "SEC-RESULTS-SYNTHESIS": ("Integrated interpretation", ["CAT-C", "CAT-L", "CAT-G", "CAT-J", "CAT-I"], ["RPT-T18"], ["RPT-F13"]),
        "SEC-DISCUSSION": ("Discussion", ["CAT-C", "CAT-D", "CAT-E", "CAT-F", "CAT-M", "CAT-I"], [], []),
        "SEC-LIMITATIONS": ("Limitations", ["CAT-M", "CAT-T"], [], ["RPT-FA05"]),
        "SEC-PUBLIC-REPRODUCIBILITY": ("Reproducibility", ["CAT-A", "CAT-B", "CAT-R"], ["RPT-T05", "RPT-T06"], []),
    }
    rows = []
    for index, (section, (title, result_ids, table_ids, figure_ids)) in enumerate(specifications.items(), start=1):
        rows.append(
            _item(
                category="S", catalogue_item_id=f"RES-P10-REPORT-EVIDENCE-{index:02d}",
                entity_type="report_evidence_map", phase="P10", result_name=title,
                scientific_question="Which registered evidence must this report section consume?",
                source_artifact_id="ART-P10-REPORT-PLAN", source_root_alias="repo://",
                source_relative_path=plan_path, source_field_path=section, source_sha256=digest,
                report_section_id=section, report_table_ids=tuple(table_ids), report_figure_ids=tuple(figure_ids),
                report_usage_status="USED_MAIN_TEXT" if not section.endswith("REPRODUCIBILITY") else "USED_APPENDIX",
                details={"required_result_or_category_ids": result_ids, "required_table_ids": table_ids, "required_figure_ids": figure_ids, "catalogue_driven_required": True},
            )
        )
    return rows


def _gap_rows(repository_root: Path) -> list[dict[str, Any]]:
    plan_path = "docs/results/RESULTS_CATALOGUE_PLAN.md"
    digest = _source_hash(repository_root, plan_path)
    specifications = [
        (
            "MEAN-ABS-CONTRIBUTION",
            "Mean absolute centered contribution summary",
            "DATA_NOT_PERSISTED",
            "No authoritative frozen P9 aggregate persists model-by-concept mean absolute centered "
            "contribution. Earlier narrative values were transient analysis, are not authoritative P10 "
            "sources, and must not be reused in RPT-T15. Do not recompute from frozen per-sample values.",
        ),
        ("UNDEFINED-MECHANISM", "Pre-ReLU CAM and gradient decomposition", "DATA_NOT_PERSISTED", "Report the mechanism boundary; do not run a new forward pass."),
        ("CASE-INTERVENTION", "Case-level intervention before/after evidence for RPT-FA06", "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE", "Mark HOW unavailable; never recompute for presentation."),
        (
            "PLANNED-TABLES",
            "20 planned tables: RPT-T01-RPT-T18 + RPT-TA01 + RPT-TA02",
            "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
            "Render only after generated Catalogue approval.",
        ),
        (
            "PLANNED-FIGURES",
            "14 planned public figures: RPT-F01-RPT-F08 + RPT-F09A + RPT-F09B + RPT-F10-RPT-F13",
            "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
            "Render only after generated Catalogue approval.",
        ),
        (
            "PRIVATE-FIGURES",
            "6 planned private figures: RPT-FA01-RPT-FA06",
            "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS",
            "Render supported components only after Catalogue approval.",
        ),
        ("FULL-CT-PANELS", "Full CT context, ROI box, and deterministic reprojection panels", "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS", "Use existing exact source/mapping only; no inference."),
    ]
    return [
        _item(
            category="T", catalogue_item_id=f"RES-P10-GAP-{identifier}", entity_type="missing_or_incomplete_output",
            phase="P10", result_name=title, scientific_question="What is missing and what action is permitted?",
            source_artifact_id="ART-P10-CATALOGUE-PLAN", source_root_alias="repo://", source_relative_path=plan_path,
            source_field_path=f"CAT-T.{identifier}", source_sha256=digest,
            report_section_id="SEC-LIMITATIONS", report_usage_status="USED_APPENDIX",
            availability_status=status, scientific_status="CLASSIFIED", integrity_status="NOT_APPLICABLE",
            new_inference_required=status == "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE",
            omission_reason=action if status in {"DATA_NOT_PERSISTED", "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE"} else None,
            details={"report_blocking_at_catalogue_gate": False, "permitted_action": action},
        )
        for identifier, title, status, action in specifications
    ]


REGISTRY_FIELD_ORDER = (
    "catalogue_item_id", "category", "entity_type", "phase", "model", "fold",
    "concept_or_target", "result_name", "scientific_question", "scientific_status",
    "availability_status", "report_usage_status", "source_artifact_id",
    "source_root_alias", "source_relative_path", "source_field_path", "source_sha256",
    "row_or_sample_count", "privacy_class", "new_inference_required",
    "report_section_id", "report_table_ids", "report_figure_ids", "omission_reason",
    "approval_reference", "integrity_status", "details",
)


def build_registry(
    *,
    repository_root: Path = Path("."),
    private_root: Path = PRIVATE_ARCHIVE_ROOT_DEFAULT,
    raw_data_root: Path = RAW_DATA_ROOT_DEFAULT,
    report_data_path: Path = REPORT_DATA_DEFAULT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = validate_catalogue_config(repository_root=repository_root)
    sources = _load_verified_sources(repository_root, private_root, report_data_path)
    report = sources["report"]
    categories: dict[str, list[dict[str, Any]]] = {letter: [] for letter in CATEGORY_FILE_NAMES}
    categories["A"] = _phase_rows(repository_root)
    categories["B"] = _training_rows(repository_root, report, sources["archive_by_path"])
    categories["C"], categories["D"], categories["E"], categories["F"] = _task_rows(repository_root, report)
    categories["G"], categories["H"] = _concept_rows(repository_root, report)
    categories["I"] = _intervention_rows(repository_root, report)
    categories["J"] = _contribution_rows(repository_root, report, sources["archive_by_path"])
    categories["K"] = _alpha_rows(repository_root, report)
    categories["L"], categories["M"], categories["N"] = _spatial_rows(repository_root, report)
    categories["Q"], private_case_rows = _qualitative_rows(
        repository_root, private_root, raw_data_root, sources["case_index"]
    )
    fa06_selected_case_label = next(
        item["details"]["case_label"]
        for item in categories["Q"]
        if item["details"]["fa06_selected"]
    )
    categories["O"], categories["P"] = _table_figure_rows(
        repository_root, fa06_selected_case_label=fa06_selected_case_label
    )
    categories["R"], private_locations = _storage_rows(
        repository_root, private_root, sources["archive_by_path"]
    )
    categories["S"] = _evidence_rows(repository_root)
    categories["T"] = _gap_rows(repository_root)
    items = [item for letter in CATEGORY_FILE_NAMES for item in categories[letter]]
    identifiers = [item["catalogue_item_id"] for item in items]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(key for key, count in Counter(identifiers).items() if count > 1)
        raise ValueError(f"P10_CATALOGUE_DUPLICATE_ITEM_ID:{duplicates[:5]}")
    for item in items:
        if set(item) != REQUIRED_REGISTRY_FIELDS:
            raise ValueError(f"P10_CATALOGUE_REGISTRY_SCHEMA_INVALID:{item.get('catalogue_item_id')}")
        if item["availability_status"] not in CONTROLLED_AVAILABILITY:
            raise ValueError(f"P10_CATALOGUE_AVAILABILITY_INVALID:{item['catalogue_item_id']}")
        if item["report_usage_status"] not in CONTROLLED_USAGE:
            raise ValueError(f"P10_CATALOGUE_USAGE_INVALID:{item['catalogue_item_id']}")
        if item["integrity_status"] not in CONTROLLED_INTEGRITY:
            raise ValueError(f"P10_CATALOGUE_INTEGRITY_INVALID:{item['catalogue_item_id']}")
    expected = config["expected_cardinalities"]
    checks = {
        "training_model_fold_rows": sum(item["entity_type"] == "model_fold_run" for item in categories["B"]),
        "pooled_oof_rows": sum(item["entity_type"] == "pooled_oof" for item in categories["B"]),
        "primary_models": len(categories["C"]), "paired_primary": len(categories["D"]),
        "secondary_models": len(categories["E"]), "paired_secondary": len(categories["F"]),
        "continuous_concepts": len(categories["G"]), "categorical_concepts": len(categories["H"]),
        "interventions": len(categories["I"]), "contributions": len(categories["J"]),
        "learned_alpha": len(categories["K"]),
        "gradcam_detail": sum(item["entity_type"] == "gradcam_model_fold_target" for item in categories["L"]),
        "gradcam_pooled": sum(item["entity_type"] == "gradcam_model_target_pooled" for item in categories["L"]),
        "gradcam_global": sum(item["entity_type"] == "gradcam_global_accounting" for item in categories["L"]),
        "undefined_rca": len(categories["M"]), "qualitative_cases": len(categories["Q"]),
    }
    for name, value in checks.items():
        if int(expected[name]) != value:
            raise ValueError(f"P10_CATALOGUE_CARDINALITY_INVALID:{name}:{value}")
    global_accounting = next(item for item in categories["L"] if item["entity_type"] == "gradcam_global_accounting")["details"]
    for key in ("requested", "valid", "undefined"):
        if int(global_accounting[key]) != int(expected[f"gradcam_{key}"]):
            raise ValueError(f"P10_CATALOGUE_GRADCAM_ACCOUNTING_INVALID:{key}")
    registry = {
        "schema_version": SCHEMA_VERSION,
        "status": "CATALOGUE_BUILT_PENDING_USER_APPROVAL",
        "phase": "P10",
        "approved_plan_sha256": {
            "results_catalogue": RESULTS_CATALOGUE_PLAN_SHA256,
            "catalogue_driven_bilingual_report": P10_REPORT_PLAN_SHA256,
        },
        "gates": {
            "results_catalogue_plan_approved": 1,
            "p10_report_plan_approved": 1,
            "generated_catalogue_approved": 0,
            "report_revision_authorized": 0,
        },
        "scientific_boundaries": {
            "model_forward": False, "training": False, "test_inference": False,
            "scientific_recomputation": False, "report_regeneration": False, "p11_started": False,
        },
        "source_bindings": {
            "p5_p9_source_manifest_sha256": sources["inputs"]["source_manifest_sha256"],
            "report_data_sha256": sources["report_data_sha256"],
            "private_archive_manifest_sha256": sources["archive_complete"]["manifest_sha256"],
            "private_archive_file_count": sources["archive_complete"]["file_count"],
            "private_archive_total_bytes": sources["archive_complete"]["total_bytes"],
            "private_case_index_sha256": sources["case_index_sha256"],
        },
        "category_counts": {f"CAT-{letter}": len(categories[letter]) for letter in categories},
        "items": items,
    }
    return registry, private_case_rows, private_locations


def _human_rows(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        if item["entity_type"] in {"archived_file", "tracked_repository_file"} and not item["details"].get("human_master_include"):
            continue
        tables = item["report_table_ids"]
        figures = item["report_figure_ids"]
        rows.append(
            {
                "Catalogue Item ID": item["catalogue_item_id"],
                "Phase": item["phase"], "Model": item["model"] or "ALL/NA", "Fold": item["fold"] if item["fold"] is not None else "ALL/NA",
                "Result / Artifact type": item["entity_type"], "Scientific content": item["result_name"],
                "Exists?": item["availability_status"] == "RESULT_ALREADY_EXISTS",
                "Frozen source": f"{item['source_root_alias']}{item['source_relative_path']}",
                "Public / Private": item["privacy_class"],
                "Report placement": {
                    "USED_MAIN_TEXT": "Main report", "USED_APPENDIX": "Appendix",
                    "USED_PRIVATE_APPENDIX": "Private appendix", "AUDIT_ONLY": "Audit-only",
                    "INTENTIONALLY_OMITTED_WITH_REASON": "No",
                }[item["report_usage_status"]],
                "Table renderable?": bool(tables) and item["availability_status"] not in {"DATA_NOT_PERSISTED", "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE"},
                "Figure renderable?": bool(figures) and item["availability_status"] not in {"DATA_NOT_PERSISTED", "WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE"},
                "Table IDs": ", ".join(tables), "Figure IDs": ", ".join(figures),
                "Existing visualization?": item["entity_type"].startswith("existing_"),
                "Visualization status": item["availability_status"],
                "New inference required?": item["new_inference_required"],
                "Assigned report section": item["report_section_id"],
                "Integrity status": item["integrity_status"],
                "Notes": item["omission_reason"] or "",
            }
        )
    return rows


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    def clean(value: Any) -> str:
        text = str(_stringify(value) if value is not None else "")
        return text.replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(clean(row.get(column, "")) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def _catalogue_markdown(registry: Mapping[str, Any], human: Sequence[Mapping[str, Any]]) -> str:
    category_counts = registry["category_counts"]
    overview = [
        {"Output family": "Model-fold training runs", "Count": 20, "Main file": "CAT_B_training_results.csv", "Completeness": "PASS"},
        {"Output family": "Canonical OOF sets", "Count": 4, "Main file": "CAT_B_training_results.csv", "Completeness": "PASS"},
        {"Output family": "Primary/secondary/paired results", "Count": 20, "Main file": "CAT_C...CAT_F", "Completeness": "PASS"},
        {"Output family": "Concept/intervention/contribution/alpha", "Count": sum(category_counts[key] for key in ("CAT-G", "CAT-H", "CAT-I", "CAT-J", "CAT-K")), "Main file": "CAT_G...CAT_K", "Completeness": "PASS"},
        {"Output family": "Grad-CAM maps", "Count": 73724, "Main file": "CAT_L_gradcam.csv", "Completeness": "66,769 valid + 6,955 undefined"},
        {"Output family": "Tables / figures", "Count": category_counts["CAT-O"] + category_counts["CAT-P"], "Main file": "tables_inventory.csv / figures_inventory.csv", "Completeness": "REGISTERED"},
        {"Output family": "Frozen qualitative cases", "Count": 14, "Main file": "qualitative_case_inventory.csv", "Completeness": "PASS"},
        {"Output family": "Private archive", "Count": registry["source_bindings"]["private_archive_file_count"], "Main file": "CAT_R_storage.csv", "Completeness": "PASS"},
    ]
    lines = [
        "# Results & Artifacts Master Catalogue",
        "",
        "Status: `CATALOGUE_BUILT_PENDING_USER_APPROVAL`. This catalogue is read-only and did not regenerate reports or scientific results.",
        "",
        "## Master index",
        "",
        _markdown_table(overview, ("Output family", "Count", "Main file", "Completeness")),
        "",
        "## Human-readable Results & Artifacts Master Table",
        "",
        "The complete deterministic view is [RESULTS_ARTIFACTS_MASTER_TABLE.md](RESULTS_ARTIFACTS_MASTER_TABLE.md).",
        "",
        _markdown_table(human[:40], ("Catalogue Item ID", "Phase", "Model", "Result / Artifact type", "Scientific content", "Visualization status")),
        "",
        "The preview above shows the first 40 rows; the linked master table contains the full human-readable ledger.",
    ]
    for letter, filename in CATEGORY_FILE_NAMES.items():
        lines.extend(["", f"## CAT-{letter}", "", f"Complete machine-readable table: [{filename}](catalogue_tables/{filename}).", "", f"Registered rows: **{category_counts[f'CAT-{letter}']}**."])
    lines.extend([
        "", "## Scientific-compute boundary", "",
        "Items marked `DATA_NOT_PERSISTED` or `WOULD_REQUIRE_NEW_SCIENTIFIC_COMPUTE` remain unavailable. They did not trigger inference, intervention recomputation, training, or any P11 work.", "",
    ])
    return "\n".join(lines)


def _human_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = (
        "Catalogue Item ID", "Phase", "Model", "Fold", "Result / Artifact type", "Scientific content",
        "Exists?", "Frozen source", "Public / Private", "Report placement", "Table renderable?",
        "Figure renderable?", "Existing visualization?", "Visualization status", "New inference required?",
        "Assigned report section", "Integrity status", "Table IDs", "Figure IDs", "Notes",
    )
    return "# Results & Artifacts Master Table\n\n" + _markdown_table(rows, columns) + "\n"


def build_public_catalogue(
    *, repository_root: Path = Path("."), public_root: Path = PUBLIC_ROOT_DEFAULT,
    private_root: Path = PRIVATE_ARCHIVE_ROOT_DEFAULT, raw_data_root: Path = RAW_DATA_ROOT_DEFAULT,
) -> dict[str, Any]:
    report_tree_before = _file_tree_hashes(repository_root / "reports/baseline_v2/p10")
    output_root = repository_root / public_root
    snapshot_path = output_root / PHASE_STATUS_SNAPSHOT_NAME
    if snapshot_path.is_file():
        if _read_json(snapshot_path) != _phase_status_snapshot():
            raise ValueError("P10_CATALOGUE_PHASE_STATUS_SNAPSHOT_TAMPER")
    else:
        _atomic_write_json(snapshot_path, _phase_status_snapshot())
    registry, private_cases, private_locations = build_registry(
        repository_root=repository_root, private_root=private_root, raw_data_root=raw_data_root
    )
    items = registry["items"]
    human = _human_rows(items)
    by_category = {letter: [item for item in items if item["category"] == f"CAT-{letter}"] for letter in CATEGORY_FILE_NAMES}
    _atomic_write_json(output_root / PUBLIC_REGISTRY_NAME, registry)
    _atomic_write_csv(output_root / "results_master_catalogue.csv", items, fieldnames=REGISTRY_FIELD_ORDER)
    _atomic_write_csv(output_root / "RESULTS_ARTIFACTS_MASTER_TABLE.csv", human)
    _atomic_write_bytes(output_root / "RESULTS_MASTER_CATALOGUE.md", _catalogue_markdown(registry, human).encode("utf-8"))
    _atomic_write_bytes(output_root / "RESULTS_ARTIFACTS_MASTER_TABLE.md", _human_markdown(human).encode("utf-8"))
    for letter, filename in CATEGORY_FILE_NAMES.items():
        _atomic_write_csv(output_root / "catalogue_tables" / filename, by_category[letter], fieldnames=REGISTRY_FIELD_ORDER)
    named_views = {
        "artifacts_inventory.csv": by_category["R"], "tables_inventory.csv": by_category["O"],
        "figures_inventory.csv": by_category["P"], "qualitative_case_inventory.csv": by_category["Q"],
        "missing_incomplete_outputs.csv": by_category["T"], "report_evidence_map.csv": by_category["S"],
        "public_private_storage_map.csv": by_category["R"],
        "catalogue_to_report_plan.csv": by_category["S"] + by_category["O"] + by_category["P"],
    }
    for filename, rows in named_views.items():
        _atomic_write_csv(output_root / filename, rows, fieldnames=REGISTRY_FIELD_ORDER)
    report_tree_after = _file_tree_hashes(repository_root / "reports/baseline_v2/p10")
    if report_tree_after != report_tree_before:
        raise ValueError("P10_CATALOGUE_REPORT_ARTIFACT_MODIFIED")
    output_files = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {PUBLIC_MANIFEST_NAME} and path.name not in {
            "RESULTS_CATALOGUE_PLAN.md", "P10_CATALOGUE_DRIVEN_BILINGUAL_REPORT_PLAN.md"
        }:
            relative = path.relative_to(output_root).as_posix()
            output_files[relative] = {"sha256": sha256_file(path), "byte_size": path.stat().st_size}
    canonical_registry_sha = hashlib.sha256(_canonical_json_bytes(registry)).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "CATALOGUE_VERIFIED_PENDING_USER_APPROVAL",
        "generated_catalogue_approved": 0, "report_revision_authorized": 0,
        "approved_plan_sha256": registry["approved_plan_sha256"],
        "source_bindings": registry["source_bindings"],
        "canonical_registry_sha256": canonical_registry_sha,
        "registry_item_count": len(items), "human_master_row_count": len(human),
        "category_counts": registry["category_counts"], "output_files": output_files,
        "report_tree_sha256": hashlib.sha256(_canonical_json_bytes(report_tree_after)).hexdigest(),
        "private_overlay_expected": True,
        "scientific_compute_performed": False, "reports_regenerated": False, "p11_started": False,
    }
    _atomic_write_json(output_root / PUBLIC_MANIFEST_NAME, manifest)
    return {
        "registry": registry, "manifest": manifest, "private_cases": private_cases,
        "private_locations": private_locations, "public_root": output_root,
    }


def _workbook_sheet(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    headers = list(rows[0].keys()) if rows else ["Status"]
    values = [dict(row) for row in rows] if rows else [{"Status": "NO_ROWS"}]
    return {"name": name[:31], "headers": headers, "rows": values}


def export_private_catalogue(
    build: Mapping[str, Any],
    *,
    repository_root: Path = Path("."),
    private_root: Path = PRIVATE_ARCHIVE_ROOT_DEFAULT,
    private_overlay_root: Path = PRIVATE_OVERLAY_ROOT_DEFAULT,
    node_binary: Path = Path("/Users/katherine/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"),
    node_modules: Path = Path("/Users/katherine/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules"),
) -> dict[str, Any]:
    private_overlay_root.mkdir(parents=True, exist_ok=True)
    os.chmod(private_overlay_root, 0o700)
    locations = list(build["private_locations"])
    private_cases = list(build["private_cases"])
    _atomic_write_csv(private_overlay_root / PRIVATE_LOCATIONS_NAME, locations, mode=0o600)
    _atomic_write_csv(private_overlay_root / "qualitative_case_private_overlay.csv", private_cases, mode=0o600)
    registry = build["registry"]
    items = registry["items"]
    human = _human_rows(items)
    readme_rows = [
        {"Field": "Status", "Value": "CATALOGUE_BUILT_PENDING_USER_APPROVAL"},
        {"Field": "Scientific compute", "Value": "NONE"},
        {"Field": "Reports regenerated", "Value": "NO"},
        {"Field": "Registry item count", "Value": len(items)},
        {"Field": "Private archive file count", "Value": registry["source_bindings"]["private_archive_file_count"]},
        {"Field": "Private archive total bytes", "Value": registry["source_bindings"]["private_archive_total_bytes"]},
        {"Field": "Catalogue plan SHA-256", "Value": RESULTS_CATALOGUE_PLAN_SHA256},
        {"Field": "Report plan SHA-256", "Value": P10_REPORT_PLAN_SHA256},
    ]
    category_sheets = [
        _workbook_sheet(f"CAT-{letter}", [item for item in items if item["category"] == f"CAT-{letter}"])
        for letter in CATEGORY_FILE_NAMES
    ]
    payload = {
        "workbooks": [
            {
                "output": PRIVATE_MASTER_XLSX,
                "sheets": [_workbook_sheet("README", readme_rows), _workbook_sheet("Master", human), *category_sheets],
            },
            {
                "output": PRIVATE_HUMAN_XLSX,
                "sheets": [
                    _workbook_sheet("README", readme_rows), _workbook_sheet("Master", human),
                    _workbook_sheet("Private Locations", locations), _workbook_sheet("Private Cases", private_cases),
                ],
            },
        ]
    }
    qa_dir = private_overlay_root / "xlsx_qa_previews"
    if qa_dir.exists():
        shutil.rmtree(qa_dir)
    qa_dir.mkdir(parents=True)
    os.chmod(qa_dir, 0o700)
    with tempfile.TemporaryDirectory(prefix="p10_catalogue_xlsx_", dir="/private/tmp") as temporary:
        work = Path(temporary)
        payload_path = work / "payload.json"
        _atomic_write_json(payload_path, payload, mode=0o600)
        (work / "node_modules").symlink_to(node_modules, target_is_directory=True)
        builder = work / "p10_catalogue_xlsx.mjs"
        shutil.copy2(repository_root / "scripts/p10_catalogue_xlsx.mjs", builder)
        subprocess.run(
            [str(node_binary), str(builder), str(payload_path), str(private_overlay_root), str(qa_dir)],
            cwd=work,
            check=True,
            env={**os.environ, "NODE_PATH": str(node_modules)},
        )
    for filename in (PRIVATE_MASTER_XLSX, PRIVATE_HUMAN_XLSX, "PRIVATE_XLSX_QA.json"):
        os.chmod(private_overlay_root / filename, 0o600)
    for inspection in private_overlay_root.glob("*.inspect.ndjson"):
        os.chmod(inspection, 0o600)
    for preview in qa_dir.glob("*.png"):
        os.chmod(preview, 0o600)
    qa = _read_json(private_overlay_root / "PRIVATE_XLSX_QA.json")
    if qa.get("status") != "PASS" or len(qa.get("workbooks", ())) != 2:
        raise ValueError("P10_CATALOGUE_PRIVATE_XLSX_QA_FAILED")
    expected_previews = sum(int(workbook["sheet_count"]) for workbook in qa["workbooks"])
    actual_previews = len(list(qa_dir.glob("*.png")))
    if actual_previews != expected_previews:
        raise ValueError("P10_CATALOGUE_PRIVATE_XLSX_RENDER_COUNT_INVALID")
    complete = {
        "schema_version": SCHEMA_VERSION, "status": "PRIVATE_CATALOGUE_COMPLETE",
        "public_registry_sha256": sha256_file(build["public_root"] / PUBLIC_REGISTRY_NAME),
        "public_manifest_sha256": sha256_file(build["public_root"] / PUBLIC_MANIFEST_NAME),
        "private_locations_sha256": sha256_file(private_overlay_root / PRIVATE_LOCATIONS_NAME),
        "private_case_overlay_sha256": sha256_file(private_overlay_root / "qualitative_case_private_overlay.csv"),
        "master_xlsx_sha256": sha256_file(private_overlay_root / PRIVATE_MASTER_XLSX),
        "human_xlsx_sha256": sha256_file(private_overlay_root / PRIVATE_HUMAN_XLSX),
        "xlsx_qa_sha256": sha256_file(private_overlay_root / "PRIVATE_XLSX_QA.json"),
        "registry_item_count": len(items), "private_location_row_count": len(locations),
        "private_case_row_count": len(private_cases), "rendered_sheet_preview_count": actual_previews,
        "scientific_compute_performed": False, "report_regeneration_performed": False,
        "generated_catalogue_approved": 0, "report_revision_authorized": 0,
    }
    _atomic_write_json(private_overlay_root / PRIVATE_COMPLETE_NAME, complete, mode=0o600)
    return complete


def _scan_public_privacy(path: Path) -> None:
    data = path.read_bytes()
    for token in PUBLIC_FORBIDDEN_VALUES:
        if token.encode("utf-8") in data:
            raise ValueError(f"P10_CATALOGUE_PUBLIC_PRIVACY_VIOLATION:{path.name}:{token}")


def _approved_revision_hashes(
    repository_root: Path, private_root: Path
) -> tuple[dict[Path, str], dict[Path, str]]:
    """Return only paths explicitly sealed by the approved revision manifests."""
    config_path = repository_root / "configs/experiments/baseline_v2_p10_report_revision.resolved.yaml"
    public_root = repository_root / "reports/baseline_v2/p10/public"
    public_manifest_path = public_root / "catalogue_report_manifest.json"
    private_report_root = private_root / "p10_private_report"
    private_manifest_path = private_report_root / "catalogue_private_report_manifest.json"
    if not (config_path.is_file() and public_manifest_path.is_file() and private_manifest_path.is_file()):
        return {}, {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("approval_gates", {}).get("report_revision_authorized") != 1:
        return {}, {}
    public_manifest = _read_json(public_manifest_path)
    private_manifest = _read_json(private_manifest_path)
    expected_registry = "624fa259430d6c5709568f7507ec0a92421d669ad64071d935eece566283b3cf"
    if public_manifest.get("catalogue_registry_sha256") != expected_registry or private_manifest.get("catalogue_registry_sha256") != expected_registry:
        raise ValueError("P10_REPORT_REVISION_CATALOGUE_BINDING_INVALID")

    public: dict[Path, str] = {}
    for language in ("en", "zh"):
        report = public_manifest["reports"][f"technical_{language}"]
        public[(public_root / f"technical_{language}.md").resolve()] = report["markdown_sha256"]
        public[(public_root / f"technical_{language}.pdf").resolve()] = report["pdf_sha256"]
    for evidence in public_manifest["tables"].values():
        public[(public_root / evidence["path"]).resolve()] = evidence["sha256"]
    for figure in public_manifest["figures"].values():
        for evidence in figure.values():
            public[(public_root / evidence["path"]).resolve()] = evidence["sha256"]
    public[(public_root / "reverse_traceability.csv").resolve()] = public_manifest["reverse_traceability_sha256"]
    audit_root = repository_root / "artifacts/baseline_v2/audit/p10"
    summary_path = audit_root / "summary.json"
    if summary_path.is_file():
        audit_summary = _read_json(summary_path)
        for name, digest in audit_summary.get("reports", {}).items():
            public[(audit_root / f"{name}.json").resolve()] = str(digest)
        public[summary_path.resolve()] = sha256_file(summary_path)

    private: dict[Path, str] = {}
    report_names = {
        "qualitative_appendix_en": "qualitative_appendix_en.pdf",
        "qualitative_appendix_zh": "qualitative_appendix_zh.pdf",
        "technical_en_with_appendix": "technical_en_with_appendix.pdf",
        "technical_zh_with_appendix": "technical_zh_with_appendix.pdf",
    }
    for key, filename in report_names.items():
        evidence = private_manifest["reports"][key]
        private[(private_report_root / filename).resolve()] = evidence["sha256"]
        if "markdown_sha256" in evidence:
            private[(private_report_root / filename.replace(".pdf", ".md")).resolve()] = evidence["markdown_sha256"]
    for evidence in private_manifest["tables"].values():
        private[(private_report_root / evidence["path"]).resolve()] = evidence["sha256"]
    for by_language in private_manifest["figures"].values():
        for evidence in by_language.values():
            private[(private_report_root / evidence["path"]).resolve()] = evidence["sha256"]
    private_qa_path = private_report_root / "private_visual_qa.json"
    if private_qa_path.is_file():
        from lidc_baseline.p10_private_appendix import (
            _verify_private_manual_review_provenance,
        )

        private_qa = _read_json(private_qa_path)
        _verify_private_manual_review_provenance(private_qa)
        expected_names = set(report_names)
        if (
            private_qa.get("status") != "PASS"
            or private_qa.get("manual_visual_review") != "PASS"
            or set(private_qa.get("pdfs", {})) != expected_names
            or any(
                private_qa["pdfs"][name].get("pdf_sha256")
                != private_manifest["reports"][name]["sha256"]
                for name in expected_names
            )
        ):
            raise ValueError("P10_PRIVATE_REVISION_VISUAL_QA_INVALID")
        private[private_qa_path.resolve()] = sha256_file(private_qa_path)
    return public, private


def verify_catalogue(
    *, repository_root: Path = Path("."), public_root: Path = PUBLIC_ROOT_DEFAULT,
    private_root: Path = PRIVATE_ARCHIVE_ROOT_DEFAULT,
    private_overlay_root: Path = PRIVATE_OVERLAY_ROOT_DEFAULT,
    require_private: bool = True,
) -> dict[str, Any]:
    validate_catalogue_config(repository_root=repository_root)
    inputs = verify_inputs(repository_root=repository_root)
    root = repository_root / public_root
    registry = _read_json(root / PUBLIC_REGISTRY_NAME)
    manifest = _read_json(root / PUBLIC_MANIFEST_NAME)
    if registry.get("status") != "CATALOGUE_BUILT_PENDING_USER_APPROVAL":
        raise ValueError("P10_CATALOGUE_REGISTRY_STATUS_INVALID")
    if manifest.get("status") != "CATALOGUE_VERIFIED_PENDING_USER_APPROVAL":
        raise ValueError("P10_CATALOGUE_MANIFEST_STATUS_INVALID")
    if registry["gates"].get("generated_catalogue_approved") != 0 or registry["gates"].get("report_revision_authorized") != 0:
        raise ValueError("P10_CATALOGUE_DOWNSTREAM_GATE_BYPASS")
    if registry["source_bindings"]["p5_p9_source_manifest_sha256"] != inputs["source_manifest_sha256"]:
        raise ValueError("P10_CATALOGUE_P5_P9_SOURCE_BINDING_INVALID")
    if manifest["canonical_registry_sha256"] != hashlib.sha256(_canonical_json_bytes(registry)).hexdigest():
        raise ValueError("P10_CATALOGUE_REGISTRY_SHA256_INVALID")
    items = registry.get("items", [])
    identifiers: set[str] = set()
    revision_public, revision_private = _approved_revision_hashes(repository_root, private_root)
    for item in items:
        if set(item) != REQUIRED_REGISTRY_FIELDS:
            raise ValueError("P10_CATALOGUE_REGISTRY_SCHEMA_INVALID")
        if item["catalogue_item_id"] in identifiers:
            raise ValueError("P10_CATALOGUE_DUPLICATE_ITEM_ID")
        identifiers.add(item["catalogue_item_id"])
        if item["availability_status"] not in CONTROLLED_AVAILABILITY:
            raise ValueError("P10_CATALOGUE_AVAILABILITY_INVALID")
        if item["report_usage_status"] not in CONTROLLED_USAGE:
            raise ValueError("P10_CATALOGUE_USAGE_INVALID")
        if item["integrity_status"] not in CONTROLLED_INTEGRITY:
            raise ValueError("P10_CATALOGUE_INTEGRITY_INVALID")
    if len(items) != int(manifest["registry_item_count"]):
        raise ValueError("P10_CATALOGUE_ITEM_COUNT_INVALID")
    global_row = next(item for item in items if item["catalogue_item_id"] == "RES-P9-GRADCAM-GLOBAL-ACCOUNTING")
    if global_row["details"]["requested"] != 73724 or global_row["details"]["valid"] != 66769 or global_row["details"]["undefined"] != 6955:
        raise ValueError("P10_CATALOGUE_GRADCAM_ACCOUNTING_INVALID")
    if not any(item["details"].get("planned_table_id") == "RPT-TA02" for item in items):
        raise ValueError("P10_CATALOGUE_TA02_MISSING")
    fa06 = next((item for item in items if item["details"].get("planned_figure_id") == "RPT-FA06"), None)
    if (
        fa06 is None
        or fa06["availability_status"] != "VISUALIZATION_NOT_YET_RENDERED_BUT_FROZEN_DATA_EXISTS"
        or fa06["details"].get("case_level_intervention_component") != "DATA_NOT_PERSISTED"
    ):
        raise ValueError("P10_CATALOGUE_FA06_GATE_INVALID")
    qualitative = [item for item in items if item["category"] == "CAT-Q"]
    selected_fa06 = [item for item in qualitative if item["details"].get("fa06_selected")]
    if (
        len(selected_fa06) != 1
        or selected_fa06[0]["details"]["case_label"] != fa06["details"].get("selected_case_label")
        or selected_fa06[0]["details"].get("fa06_case_role") != "integrated_explanation"
        or "RPT-FA06" not in selected_fa06[0]["report_figure_ids"]
        or any("RPT-FA06" in item["report_figure_ids"] for item in qualitative if item is not selected_fa06[0])
    ):
        raise ValueError("P10_CATALOGUE_FA06_SELECTION_INVALID")
    for item in qualitative:
        details = item["details"]
        renderable = bool(
            details.get("full_ct_source_available")
            and details.get("series_and_slice_provenance_available")
            and details.get("roi_to_full_volume_mapping_available")
        )
        if details.get("read_only_full_ct_renderable") is not renderable:
            raise ValueError("P10_CATALOGUE_FULL_CT_GATE_INVALID")
    complete, archive_by_path = _archive_sources(private_root)
    if complete["manifest_sha256"] != registry["source_bindings"]["private_archive_manifest_sha256"]:
        raise ValueError("P10_CATALOGUE_ARCHIVE_BINDING_INVALID")
    for item in items:
        if item["integrity_status"] != "VERIFIED" or item["entity_type"] == "storage_group":
            continue
        alias = item["source_root_alias"]
        relative = item["source_relative_path"]
        expected = item["source_sha256"]
        if alias == "repo://":
            path = repository_root / relative
            current = sha256_file(path) if path.is_file() else None
            replacement = revision_public.get(path.resolve())
            if current != expected and current != replacement:
                raise ValueError(f"P10_CATALOGUE_REPO_SOURCE_INVALID:{item['catalogue_item_id']}")
        elif alias == "mac-archive://":
            row = archive_by_path.get(relative)
            if row is None or str(row["sha256"]) != expected:
                raise ValueError(f"P10_CATALOGUE_ARCHIVE_SOURCE_INVALID:{item['catalogue_item_id']}")
        elif alias == "private-report://":
            path = private_root / "p10_private_report" / relative
            current = sha256_file(path) if path.is_file() else None
            replacement = revision_private.get(path.resolve())
            if current != expected and current != replacement:
                raise ValueError(f"P10_CATALOGUE_PRIVATE_REPORT_SOURCE_INVALID:{item['catalogue_item_id']}")
        elif alias != "katana-run://":
            raise ValueError(f"P10_CATALOGUE_SOURCE_ALIAS_INVALID:{alias}")
    for filename, evidence in manifest["output_files"].items():
        path = root / filename
        if not path.is_file() or sha256_file(path) != evidence["sha256"] or path.stat().st_size != int(evidence["byte_size"]):
            raise ValueError(f"P10_CATALOGUE_OUTPUT_TAMPER:{filename}")
    public_files = [root / PUBLIC_REGISTRY_NAME, root / PUBLIC_MANIFEST_NAME]
    public_files.extend(root / name for name in manifest["output_files"])
    for path in public_files:
        _scan_public_privacy(path)
    # The frozen report-tree hash describes the legacy presentation at Catalogue
    # approval. Once the separate report-revision gate is explicitly authorized,
    # the new presentation tree is instead bound and verified by its own manifest.
    if not revision_public:
        report_tree = _file_tree_hashes(repository_root / "reports/baseline_v2/p10")
        if hashlib.sha256(_canonical_json_bytes(report_tree)).hexdigest() != manifest["report_tree_sha256"]:
            raise ValueError("P10_CATALOGUE_REPORT_TREE_MODIFIED")
    private_status: dict[str, Any] = {"required": require_private, "status": "NOT_REQUIRED"}
    if require_private:
        complete_path = private_overlay_root / PRIVATE_COMPLETE_NAME
        private_complete = _read_json(complete_path)
        if private_complete.get("status") != "PRIVATE_CATALOGUE_COMPLETE":
            raise ValueError("P10_CATALOGUE_PRIVATE_COMPLETE_INVALID")
        expected_private = {
            PRIVATE_LOCATIONS_NAME: private_complete["private_locations_sha256"],
            "qualitative_case_private_overlay.csv": private_complete["private_case_overlay_sha256"],
            PRIVATE_MASTER_XLSX: private_complete["master_xlsx_sha256"],
            PRIVATE_HUMAN_XLSX: private_complete["human_xlsx_sha256"],
            "PRIVATE_XLSX_QA.json": private_complete["xlsx_qa_sha256"],
        }
        for filename, digest in expected_private.items():
            path = private_overlay_root / filename
            if not path.is_file() or sha256_file(path) != digest or (path.stat().st_mode & 0o777) != 0o600:
                raise ValueError(f"P10_CATALOGUE_PRIVATE_OUTPUT_INVALID:{filename}")
        qa = _read_json(private_overlay_root / "PRIVATE_XLSX_QA.json")
        if qa.get("status") != "PASS" or sum(w["sheet_count"] for w in qa["workbooks"]) != private_complete["rendered_sheet_preview_count"]:
            raise ValueError("P10_CATALOGUE_PRIVATE_QA_INVALID")
        private_status = {"required": True, "status": "PASS", "manifest_sha256": sha256_file(complete_path)}
    return {
        "schema_version": SCHEMA_VERSION, "status": "PASS", "registry_item_count": len(items),
        "category_counts": registry["category_counts"], "public_manifest_sha256": sha256_file(root / PUBLIC_MANIFEST_NAME),
        "public_registry_sha256": sha256_file(root / PUBLIC_REGISTRY_NAME), "private": private_status,
        "gradcam_accounting": {"requested": 73724, "valid": 66769, "undefined": 6955},
        "reports_regenerated": False, "scientific_compute_performed": False, "p11_started": False,
    }


def build_all_catalogue_outputs(
    *, repository_root: Path = Path("."), private_root: Path = PRIVATE_ARCHIVE_ROOT_DEFAULT,
    raw_data_root: Path = RAW_DATA_ROOT_DEFAULT,
) -> dict[str, Any]:
    build = build_public_catalogue(
        repository_root=repository_root, private_root=private_root, raw_data_root=raw_data_root
    )
    export_private_catalogue(build, repository_root=repository_root, private_root=private_root)
    return verify_catalogue(repository_root=repository_root, private_root=private_root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-inputs")
    subparsers.add_parser("verify-report-inputs")
    subparsers.add_parser("build")
    subparsers.add_parser("export-private")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--public-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(".").resolve()
    if args.command in {"verify-inputs", "verify-report-inputs"}:
        payload = _load_verified_sources(root, PRIVATE_ARCHIVE_ROOT_DEFAULT, REPORT_DATA_DEFAULT)
        result = {
            "status": "PASS", "approved_plan_sha256": {
                "results_catalogue": RESULTS_CATALOGUE_PLAN_SHA256,
                "catalogue_driven_bilingual_report": P10_REPORT_PLAN_SHA256,
            },
            "p5_p9_source_manifest_sha256": payload["inputs"]["source_manifest_sha256"],
            "archive_manifest_sha256": payload["archive_complete"]["manifest_sha256"],
            "archive_file_count": payload["archive_complete"]["file_count"],
            "archive_total_bytes": payload["archive_complete"]["total_bytes"],
            "generated_catalogue_approved": 0, "report_revision_authorized": 0,
        }
    elif args.command == "build":
        result = build_all_catalogue_outputs(repository_root=root)
    elif args.command == "export-private":
        registry, private_cases, private_locations = build_registry(repository_root=root)
        result = export_private_catalogue(
            {
                "registry": registry,
                "private_cases": private_cases,
                "private_locations": private_locations,
                "public_root": root / PUBLIC_ROOT_DEFAULT,
            },
            repository_root=root,
        )
    else:
        result = verify_catalogue(repository_root=root, require_private=not args.public_only)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
