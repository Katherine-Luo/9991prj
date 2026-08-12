"""Build the read-only P10 bilingual final-report deliverables.

The authoritative numeric layer is assembled exclusively from tracked,
deidentified P5--P9 audit evidence.  Rendering never invokes a model forward.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
CONFIG_DEFAULT = Path("configs/experiments/baseline_v2_p10_report_archive.yaml")
CONFIG_RESOLVED_DEFAULT = Path(
    "configs/experiments/baseline_v2_p10_report_archive.resolved.yaml"
)
CONFIG_SHA_DEFAULT = Path(
    "configs/experiments/baseline_v2_p10_report_archive.sha256"
)
P9_AUDIT_ROOT_DEFAULT = Path("artifacts/baseline_v2/audit/p9")
PUBLIC_ROOT_DEFAULT = Path("reports/baseline_v2/p10/public")
PRIVATE_ARCHIVE_COMPLETE_DEFAULT = Path(
    "/Users/katherine/Desktop/lidc_data/lidc_baseline_private_archive/baseline_v2/ARCHIVE_COMPLETE.json"
)
P10_CONFIG_SHA256 = "09a6e99c78c5aed9f75a2d054b90235b5a8d065f61b8d702e599d0102feaae6a"
P9_CONFIG_SHA256 = "a52d559cb241e8e5cb0f834f41fc171dca63ba2fcfda2164f251e48f4dfc4906"
P9_SUMMARY_SHA256 = "16626aa6e6a8fe711fd66766145aad2d4646c8dfd22cc0d926f90558d2af2294"
P5_P9_SOURCE_MANIFEST_SHA256 = (
    "7f2b569480e5f044e45bcd2b3295e1a72836ce67bfa136f9f46363926d6fd9af"
)
MODEL_ORDER = (
    "blackbox",
    "standard_cbm",
    "mixed_cem",
    "learned_softmax_gam",
)
MODEL_LABELS = {
    "blackbox": "Black-box",
    "standard_cbm": "Standard CBM",
    "mixed_cem": "Mixed-type CEM",
    "learned_softmax_gam": "Learned-softmax GAM",
}
CONCEPT_ORDER = (
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
P9_REPORT_NAMES = (
    "task",
    "concept",
    "contribution_centering",
    "intervention",
    "bootstrap",
    "learned_alpha",
    "spatial",
    "integrity",
    "summary",
)
FOLD_COUNTS = (479, 502, 539, 549, 564)
PUBLIC_FORBIDDEN_TOKENS = (
    "nodule_uid",
    "patient_key",
    "/srv/scratch/",
    "/Users/",
    "spatial_execution_approval",
    "checkpoint.pt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"P10_JSON_OBJECT_REQUIRED:{path}")
    return payload


def validate_execution_config(
    resolved_path: Path = CONFIG_RESOLVED_DEFAULT,
    sha_path: Path = CONFIG_SHA_DEFAULT,
) -> dict[str, Any]:
    expected = sha_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(resolved_path)
    if expected != actual or actual != P10_CONFIG_SHA256:
        raise ValueError("P10_EXECUTION_CONFIG_SHA256_MISMATCH")
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    project = payload["project_preregistered"]
    frozen = project["frozen_inputs"]
    if frozen["access"] != "read_only" or frozen["artifact_rewrite"] != "forbidden":
        raise ValueError("P10_FROZEN_INPUT_BOUNDARY_INVALID")
    if any(
        frozen[field] != "forbidden"
        for field in (
            "retraining",
            "test_inference",
            "second_committed_test_evaluation",
            "new_h200_jobs",
            "new_cpu_scientific_jobs",
            "p11",
        )
    ):
        raise ValueError("P10_NO_NEW_COMPUTE_BOUNDARY_INVALID")
    if tuple(frozen["fold_test_counts"]) != FOLD_COUNTS:
        raise ValueError("P10_FOLD_COUNTS_INVALID")
    if tuple(frozen["model_order"]) != MODEL_ORDER:
        raise ValueError("P10_MODEL_ORDER_INVALID")
    return payload


def verify_inputs(
    *,
    audit_root: Path = P9_AUDIT_ROOT_DEFAULT,
    config_resolved: Path = CONFIG_RESOLVED_DEFAULT,
    config_sha: Path = CONFIG_SHA_DEFAULT,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    """Fail closed on frozen report sources and return aggregate evidence."""
    validate_execution_config(config_resolved, config_sha)
    if sha256_file(audit_root / "summary.json") != P9_SUMMARY_SHA256:
        raise ValueError("P10_P9_SUMMARY_SHA256_MISMATCH")
    reports: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name in P9_REPORT_NAMES:
        path = audit_root / f"{name}.json"
        payload = _read_json(path)
        if payload.get("status") != "PASS":
            raise ValueError(f"P10_P9_REPORT_STATUS_INVALID:{name}")
        reports[name] = payload
        hashes[name] = sha256_file(path)
    integrity = reports["integrity"]
    summary = reports["summary"]
    _verify_p9_tracked_report_hashes(audit_root, summary)
    if (
        integrity.get("unique_nodules") != 2633
        or integrity.get("unique_patients") != 868
        or tuple(integrity.get("fold_counts", ())) != FOLD_COUNTS
        or integrity.get("patient_leakage") != 0
    ):
        raise ValueError("P10_P9_OOF_INTEGRITY_INVALID")
    if (
        summary.get("oof_nodules") != 2633
        or summary.get("oof_patients") != 868
        or tuple(summary.get("fold_counts", ())) != FOLD_COUNTS
        or summary.get("patient_leakage") != 0
        or summary.get("p5_through_p8_artifacts_modified") is not False
        or summary.get("second_committed_test_evaluation") is not False
        or summary.get("p10_started") is not False
    ):
        raise ValueError("P10_P9_SUMMARY_INVARIANT_INVALID")
    source_hashes: dict[str, str] = {}
    for phase in ("p5", "p6", "p7", "p8"):
        root = repository_root / "artifacts" / "baseline_v2" / "audit" / phase
        phase_summary = _read_json(root / "summary.json")
        if phase_summary.get("status") != "PASS":
            raise ValueError(f"P10_SOURCE_PHASE_STATUS_INVALID:{phase}")
        if phase_summary.get("folds") != 5:
            raise ValueError(f"P10_SOURCE_PHASE_FOLD_COUNT_INVALID:{phase}")
        if phase_summary.get("test_evaluated_once_all_folds") is not True:
            raise ValueError(f"P10_SOURCE_TEST_ONCE_INVALID:{phase}")
        if tuple(phase_summary.get("fold_test_counts", ())) != FOLD_COUNTS:
            raise ValueError(f"P10_SOURCE_FOLD_COUNTS_INVALID:{phase}")
        for path in sorted(root.glob("*.json")):
            relative = path.relative_to(repository_root).as_posix()
            source_hashes[relative] = sha256_file(path)
    for name, digest in hashes.items():
        source_hashes[(audit_root / f"{name}.json").as_posix()] = digest
    source_manifest_sha256 = hashlib.sha256(
        (
            json.dumps(
                dict(sorted(source_hashes.items())),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if source_manifest_sha256 != P5_P9_SOURCE_MANIFEST_SHA256:
        raise ValueError("P10_P5_P9_SOURCE_MANIFEST_SHA256_MISMATCH")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "p10_execution_config_sha256": P10_CONFIG_SHA256,
        "p9_summary_sha256": P9_SUMMARY_SHA256,
        "p9_report_sha256": hashes,
        "source_file_sha256": source_hashes,
        "source_manifest_sha256": source_manifest_sha256,
        "unique_nodules": 2633,
        "unique_patients": 868,
        "fold_counts": list(FOLD_COUNTS),
        "patient_leakage": 0,
        "new_training": False,
        "new_test_inference": False,
        "new_scientific_jobs": False,
        "p11_started": False,
    }


def _verify_p9_tracked_report_hashes(
    audit_root: Path, summary: Mapping[str, Any]
) -> None:
    tracked = summary.get("tracked_report_sha256")
    expected_names = {f"{name}.json" for name in P9_REPORT_NAMES if name != "summary"}
    if not isinstance(tracked, Mapping) or set(tracked) != expected_names:
        raise ValueError("P10_P9_TRACKED_REPORT_MANIFEST_INVALID")
    for filename in sorted(expected_names):
        path = audit_root / filename
        if not path.is_file() or sha256_file(path) != tracked[filename]:
            raise ValueError(f"P10_P9_TRACKED_REPORT_SHA256_MISMATCH:{filename}")


def _checkpoint_field(phase: str, fold: Mapping[str, Any]) -> str:
    if phase == "p6":
        return "+".join(
            (fold["concept_best_checkpoint_sha256"], fold["task_best_checkpoint_sha256"])
        )
    return str(fold["best_checkpoint_sha256"])


def execution_registry(repository_root: Path = Path(".")) -> list[dict[str, Any]]:
    """Return immutable P5--P9 scheduler/scientific provenance rows."""
    training = {
        "p5": {
            "model": "blackbox",
            "jobs": (8965243, 8965994, 8965995, 8965996, 8965997),
            "queue": "csegpu48",
            "memory_gb": 46,
            "walltime": "48:00:00",
            "exit": (1, 0, 0, 0, 0),
            "runs": (1, 1, 1, 1, 1),
            "note": (
                "Verifier false failure; existing-artifact verifier recovery PASS",
                "Direct formal job Exit0; final verifier PASS",
                "Direct formal job Exit0; final verifier PASS",
                "Direct formal job Exit0; final verifier PASS",
                "Direct formal job Exit0; final verifier PASS",
            ),
        },
        "p6": {
            "model": "standard_cbm",
            "jobs": (8969575, 8969576, 8969577, 8969578, 8969579),
            "queue": "csegpu48",
            "memory_gb": 64,
            "walltime": "48:00:00",
            "exit": (0, 0, 0, 0, 0),
            "runs": (1, 1, 1, 1, 1),
            "note": ("Direct formal verifier PASS",) * 5,
        },
        "p7": {
            "model": "mixed_cem",
            "jobs": (8974425, 8974429, 8974427, 8974428, 8974426),
            "queue": "csegpu100",
            "memory_gb": 64,
            "walltime": "96:00:00",
            "exit": (0, 0, 0, 0, 1),
            "runs": (1, 1, 1, 1, 1),
            "note": (
                "Direct formal job Exit0; final verifier PASS",
                "Direct formal job Exit0; final verifier PASS",
                "Direct formal job Exit0; final verifier PASS",
                "Direct formal job Exit0; final verifier PASS",
                "Invalidated precommit test attempt; controlled recovery 8976532 PASS",
            ),
        },
        "p8": {
            "model": "learned_softmax_gam",
            "jobs": (8979874, 8979876, 8979877, 8979873, 8979875),
            "queue": "csegpu12",
            "memory_gb": 64,
            "walltime": "11:00:00",
            "exit": (1, -18, -18, 1, -18),
            "runs": (1, 21, 21, 1, 21),
            "note": (
                "Verifier false failure Exit1; CPU verifier 8983016 PASS",
                "PBS automatic rerun/hold run_count21 Exit-18; test transaction guard held at 1; CPU verifier 8983016 PASS",
                "PBS automatic rerun/hold run_count21 Exit-18; test transaction guard held at 1; CPU verifier 8983016 PASS",
                "Verifier false failure Exit1; CPU verifier 8983016 PASS",
                "PBS automatic rerun/hold run_count21 Exit-18; test transaction guard held at 1; CPU verifier 8983016 PASS",
            ),
        },
    }
    rows: list[dict[str, Any]] = []
    for phase, spec in training.items():
        for fold_index in range(5):
            fold = _read_json(
                repository_root
                / "artifacts"
                / "baseline_v2"
                / "audit"
                / phase
                / f"fold_{fold_index}.json"
            )
            test_transactions = fold.get(
                "test_inference_transactions",
                fold.get("valid_committed_test_evaluations", 1),
            )
            rows.append(
                {
                    "phase": phase.upper(),
                    "model": MODEL_LABELS[spec["model"]],
                    "fold": fold_index,
                    "job_id": spec["jobs"][fold_index],
                    "queue": spec["queue"],
                    "gpu_model": "NVIDIA H200",
                    "ncpus": 8,
                    "memory_gb": spec["memory_gb"],
                    "walltime": spec["walltime"],
                    "run_count": spec["runs"][fold_index],
                    "exit_status": spec["exit"][fold_index],
                    "scientific_status": "PASS",
                    "config_sha256": fold.get(
                        f"{phase}_execution_config_sha256",
                        fold.get("execution_config_sha256"),
                    ),
                    "split_sha256": fold["split_sha256"],
                    "encoder_sha256": fold["encoder_initialization_sha256"],
                    "checkpoint_sha256": _checkpoint_field(phase, fold),
                    "test_transaction_count": int(test_transactions),
                    "verifier_or_recovery_evidence": spec["note"][fold_index],
                }
            )
    p9_jobs = range(8986218, 8986238)
    job = iter(p9_jobs)
    for model in MODEL_ORDER:
        for fold_index in range(5):
            p9_fold = _read_json(
                repository_root
                / "artifacts"
                / "baseline_v2"
                / "audit"
                / ("p5" if model == "blackbox" else {"standard_cbm": "p6", "mixed_cem": "p7", "learned_softmax_gam": "p8"}[model])
                / f"fold_{fold_index}.json"
            )
            rows.append(
                {
                    "phase": "P9",
                    "model": MODEL_LABELS[model],
                    "fold": fold_index,
                    "job_id": next(job),
                    "queue": "csegpu12",
                    "gpu_model": "NVIDIA H200",
                    "ncpus": 8,
                    "memory_gb": 64,
                    "walltime": "11:00:00",
                    "run_count": 1,
                    "exit_status": 0,
                    "scientific_status": "PASS",
                    "config_sha256": P9_CONFIG_SHA256,
                    "split_sha256": p9_fold["split_sha256"],
                    "encoder_sha256": p9_fold["encoder_initialization_sha256"],
                    "checkpoint_sha256": _checkpoint_field(
                        {"blackbox": "p5", "standard_cbm": "p6", "mixed_cem": "p7", "learned_softmax_gam": "p8"}[model],
                        p9_fold,
                    ),
                    "test_transaction_count": 0,
                    "verifier_or_recovery_evidence": "Spatial-only strict verifier PASS; no test evaluation",
                }
            )
    if len(rows) != 40:
        raise AssertionError("P10_EXECUTION_REGISTRY_CARDINALITY_INVALID")
    return rows


def _crosses_zero(interval: Mapping[str, Any]) -> bool:
    return float(interval["percentile_2_5"]) <= 0.0 <= float(
        interval["percentile_97_5"]
    )


def _pooled_map_counts(model_spatial: Mapping[str, Any]) -> tuple[int, int]:
    valid = sum(
        int(target["valid_map_count"])
        for target in model_spatial["pooled_targets"].values()
    )
    undefined = sum(
        int(target["undefined_map_count"])
        for target in model_spatial["pooled_targets"].values()
    )
    return valid, undefined


def _contribution_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODEL_ORDER[1:]:
        values: dict[str, list[float]] = defaultdict(list)
        for fold in payload[model]["folds"]:
            for concept, value in fold["train_group_means_rating_point_units"].items():
                values[concept].append(float(value))
        means = {concept: sum(items) / len(items) for concept, items in values.items()}
        negatives = [item for item in means.items() if item[1] < 0]
        result[model] = {
            "mean_rating_point_contribution_by_concept": means,
            "most_positive": max(means.items(), key=lambda item: (item[1], item[0])),
            "most_negative": (
                min(negatives, key=lambda item: (item[1], item[0]))
                if negatives
                else None
            ),
            "smallest_signed": min(means.items(), key=lambda item: (item[1], item[0])),
            "folds": payload[model]["folds"],
        }
    return result


def build_report_data(
    *,
    audit_root: Path = P9_AUDIT_ROOT_DEFAULT,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    verified = verify_inputs(audit_root=audit_root, repository_root=repository_root)
    p9 = {name: _read_json(audit_root / f"{name}.json") for name in P9_REPORT_NAMES}
    bootstrap = p9["bootstrap"]
    paired_mae = {
        key: {**value, "ci_crosses_zero": _crosses_zero(value)}
        for key, value in bootstrap["paired_mae_A_minus_B"].items()
    }
    paired_auroc = {
        key: {**value, "ci_crosses_zero": _crosses_zero(value)}
        for key, value in bootstrap["paired_auroc_B_minus_A"].items()
    }
    counts = [
        _pooled_map_counts(p9["spatial"]["models"][model]) for model in MODEL_ORDER
    ]
    map_valid = sum(valid for valid, _ in counts)
    map_undefined = sum(undefined for _, undefined in counts)
    map_requested = map_valid + map_undefined
    if (map_requested, map_valid, map_undefined) != (73724, 66769, 6955):
        raise ValueError("P10_SPATIAL_ACCOUNTING_INVALID")
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scientific_conclusion_codes": [
            "GAM_LOWEST_POINT_ESTIMATE_MAE",
            "PAIRED_MAE_SUPPORTS_GAM_OVER_BLACKBOX_AND_CBM",
            "AUROC_DIFFERENCES_MOSTLY_UNCERTAIN",
            "INTERVENTION_BENEFIT_MODEL_DEPENDENT",
            "SALIENCY_NOT_UNIFORMLY_MORE_FAITHFUL_THAN_RANDOM",
            "SYSTEMATIC_MODEL_TARGET_ZERO_MAP_LIMITATION",
        ],
        "cohort": {
            "unique_nodules": 2633,
            "unique_patients": 868,
            "fold_counts": list(FOLD_COUNTS),
            "patient_leakage": 0,
        },
        "task": {
            **{key: value for key, value in p9["task"].items() if key != "models"},
            "models": {model: p9["task"]["models"][model] for model in MODEL_ORDER},
        },
        "bootstrap": {
            "draws": bootstrap["draws"],
            "models": {model: bootstrap["models"][model] for model in MODEL_ORDER},
            "paired_mae_A_minus_B": paired_mae,
            "paired_auroc_B_minus_A": paired_auroc,
            "primary_draw_sha256": bootstrap["primary_draw_sha256"],
            "secondary_draw_sha256": bootstrap["secondary_draw_sha256"],
        },
        "concept": {model: p9["concept"][model] for model in MODEL_ORDER[1:]},
        "contribution_centering": _contribution_summary(p9["contribution_centering"]),
        "intervention": p9["intervention"],
        "learned_alpha": p9["learned_alpha"],
        "spatial": p9["spatial"],
        "gradcam_accounting": {
            "requested": map_requested,
            "valid": map_valid,
            "undefined": map_undefined,
            "undefined_rate": map_undefined / map_requested,
            "root_cause_conclusion": "SYSTEMATIC_MODEL/TARGET_ISSUE",
            "confirmed_observation": "post-ReLU Grad-CAM map is exactly all-zero",
            "mechanism_limit": (
                "The persisted artifacts do not contain complete pre-ReLU CAM or gradient "
                "decomposition, and no new forward pass is permitted."
            ),
        },
        "execution_registry": execution_registry(repository_root),
        "execution_events": [
            {
                "phase": "P9",
                "event": "spatial_stage_a",
                "job_id": 8986164,
                "resource": "NVIDIA H200",
                "exit_status": 0,
                "scientific_status": "PASS",
                "note": "Fold-0 validation-only preflight; no test access or parameter update",
            },
            {
                "phase": "P9",
                "event": "aggregate_invalidated_attempt",
                "job_id": 8987452,
                "resource": "CPU-only",
                "exit_status": 1,
                "scientific_status": "INVALIDATED_AGGREGATE_ATTEMPT",
                "note": "Validation verifier implementation bug; 20 spatial artifacts remained valid",
            },
            {
                "phase": "P9",
                "event": "aggregate_verifier_recovery",
                "job_id": 8987554,
                "resource": "CPU-only",
                "exit_status": 0,
                "scientific_status": "PASS",
                "note": "Existing-artifact-only aggregate recovery and final audit PASS",
            },
        ],
        "input_verification": verified,
        "limitations": [
            "Primary regression scores are unclipped.",
            "LIDC malignancy is a radiologist assessment, not pathology-confirmed diagnosis.",
            "This research system is not a clinical diagnostic product.",
            "Mixed-type CEM is a project-specific extension of the original CEM.",
            "Learned-softmax GAM is the preregistered local-expert design.",
            "The exact pre-ReLU/gradient cause of every zero map cannot be recovered from persisted artifacts.",
        ],
        "references": [
            '[1] S. G. Armato III et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans," Med. Phys., vol. 38, no. 2, pp. 915-931, 2011, doi: 10.1118/1.3528204.',
            '[2] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4700-4708, 2017, doi: 10.1109/CVPR.2017.243.',
            '[3] P. W. Koh et al., "Concept Bottleneck Models," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 119, pp. 5338-5348, 2020.',
            '[4] M. Espinosa Zarlenga et al., "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off," in Adv. Neural Inf. Process. Syst., vol. 35, pp. 21400-21413, 2022.',
            '[5] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 618-626, 2017, doi: 10.1109/ICCV.2017.74.',
            '[6] B. Efron, "Bootstrap Methods: Another Look at the Jackknife," Ann. Stat., vol. 7, no. 1, pp. 1-26, 1979, doi: 10.1214/aos/1176344552.',
        ],
        "data_dictionary": {
            "en": _data_dictionary_rows("en"),
            "zh": _data_dictionary_rows("zh"),
        },
        "terminology": _terminology_rows(),
    }
    if PRIVATE_ARCHIVE_COMPLETE_DEFAULT.is_file():
        archive = _read_json(PRIVATE_ARCHIVE_COMPLETE_DEFAULT)
        if archive.get("status") != "ARCHIVE_COMPLETE":
            raise ValueError("P10_PRIVATE_ARCHIVE_COMPLETION_STATUS_INVALID")
        output["private_archive"] = {
            "status": "PASS",
            "file_count": int(archive["file_count"]),
            "total_bytes": int(archive["total_bytes"]),
            "manifest_sha256": str(archive["manifest_sha256"]),
            "remote_manifest_sha256": str(archive["remote_manifest_sha256"]),
            "remote_write": bool(archive["remote_write"]),
            "remote_delete": bool(archive["remote_delete"]),
        }
    assert_public_payload(output)
    return output


def assert_public_payload(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    lowered = serialized.lower()
    for token in PUBLIC_FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            raise ValueError(f"P10_PUBLIC_PRIVACY_VIOLATION:{token}")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
    temporary.replace(path)


def export_tables(data: Mapping[str, Any], table_root: Path) -> list[Path]:
    paths: list[Path] = []
    primary_rows = []
    for model in MODEL_ORDER:
        pooled = data["task"]["models"][model]["pooled"]
        secondary = data["task"]["models"][model]["pooled_secondary"]
        ci = data["bootstrap"]["models"][model]
        primary_rows.append(
            {
                "model": MODEL_LABELS[model],
                "mae": pooled["original_scale_mae"],
                "mae_ci_low": ci["original_scale_mae"]["percentile_2_5"],
                "mae_ci_high": ci["original_scale_mae"]["percentile_97_5"],
                "rmse": pooled["original_scale_rmse"],
                "rmse_ci_low": ci["original_scale_rmse"]["percentile_2_5"],
                "rmse_ci_high": ci["original_scale_rmse"]["percentile_97_5"],
                "normalized_mae": pooled["normalized_mae"],
                "normalized_mae_ci_low": ci["normalized_mae"]["percentile_2_5"],
                "normalized_mae_ci_high": ci["normalized_mae"]["percentile_97_5"],
                "pearson": pooled["pearson"],
                "pearson_ci_low": ci["pearson"]["percentile_2_5"],
                "pearson_ci_high": ci["pearson"]["percentile_97_5"],
                "spearman": pooled["spearman"],
                "spearman_ci_low": ci["spearman"]["percentile_2_5"],
                "spearman_ci_high": ci["spearman"]["percentile_97_5"],
                "prediction_min_1_to_5": pooled["prediction_range_1_to_5"][0],
                "prediction_max_1_to_5": pooled["prediction_range_1_to_5"][1],
                "below_one_rate": pooled["below_one_rate"],
                "above_five_rate": pooled["above_five_rate"],
                "auroc": secondary["auroc"],
                "auroc_ci_low": ci["auroc"]["percentile_2_5"],
                "auroc_ci_high": ci["auroc"]["percentile_97_5"],
                "auprc": secondary["auprc"],
                "auprc_ci_low": ci["auprc"]["percentile_2_5"],
                "auprc_ci_high": ci["auprc"]["percentile_97_5"],
            }
        )
    path = table_root / "primary_secondary_metrics.csv"
    _write_csv(path, tuple(primary_rows[0]), primary_rows)
    paths.append(path)
    comparison_rows = []
    for metric_key, label in (
        ("paired_mae_A_minus_B", "MAE_A_minus_MAE_B"),
        ("paired_auroc_B_minus_A", "AUROC_B_minus_AUROC_A"),
    ):
        for pair, interval in data["bootstrap"][metric_key].items():
            model_a, model_b = pair.split("__")
            comparison_rows.append(
                {
                    "metric": label,
                    "model_A": MODEL_LABELS[model_a],
                    "model_B": MODEL_LABELS[model_b],
                    "estimate": interval["estimate_mean"],
                    "ci_low": interval["percentile_2_5"],
                    "ci_high": interval["percentile_97_5"],
                    "ci_crosses_zero": interval["ci_crosses_zero"],
                }
            )
    path = table_root / "paired_comparisons.csv"
    _write_csv(path, tuple(comparison_rows[0]), comparison_rows)
    paths.append(path)
    concept_rows = []
    for model in MODEL_ORDER[1:]:
        for concept in CONCEPT_ORDER:
            metrics = data["concept"][model]["pooled"][concept]
            for metric, value in metrics.items():
                concept_rows.append(
                    {
                        "model": MODEL_LABELS[model],
                        "concept": concept,
                        "metric": metric,
                        "value": value,
                    }
                )
    path = table_root / "concept_metrics.csv"
    _write_csv(path, tuple(concept_rows[0]), concept_rows)
    paths.append(path)
    intervention_rows = []
    for model in MODEL_ORDER[1:]:
        report = data["intervention"][model]
        random = report["random_permutations"]
        error_first = report["error_first"]
        for index, k in enumerate(random["k"]):
            intervention_rows.extend(
                (
                    {
                        "model": MODEL_LABELS[model],
                        "strategy": "random_permutations",
                        "k": k,
                        "mae": random["pooled_original_scale_mae_mean"][index],
                        "mae_sd": random["pooled_original_scale_mae_sd"][index],
                        "auroc": random["pooled_auroc_mean"][index],
                        "auroc_sd": random["pooled_auroc_sd"][index],
                    },
                    {
                        "model": MODEL_LABELS[model],
                        "strategy": "error_first",
                        "k": k,
                        "mae": error_first["pooled_original_scale_mae"][index],
                        "mae_sd": None,
                        "auroc": error_first["pooled_auroc"][index],
                        "auroc_sd": None,
                    },
                )
            )
    path = table_root / "intervention_curves.csv"
    _write_csv(path, tuple(intervention_rows[0]), intervention_rows)
    paths.append(path)
    contribution_rows = []
    for model in MODEL_ORDER[1:]:
        summary = data["contribution_centering"][model]
        for concept, value in summary["mean_rating_point_contribution_by_concept"].items():
            contribution_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "concept": concept,
                    "mean_train_center_rating_points": value,
                }
            )
    path = table_root / "centered_contributions.csv"
    _write_csv(path, tuple(contribution_rows[0]), contribution_rows)
    paths.append(path)
    alpha_rows = []
    for fold in data["learned_alpha"]["folds"]:
        for concept, values in fold["groups"].items():
            for expert, weight in enumerate(values["weights"]):
                alpha_rows.append(
                    {
                        "fold": fold["fold_index"],
                        "concept": concept,
                        "expert": expert,
                        "weight": weight,
                        "logit": values["logits"][expert],
                        "gradient_l1": values["gradient_l1_at_best_epoch"],
                    }
                )
    path = table_root / "learned_gam_alpha.csv"
    _write_csv(path, tuple(alpha_rows[0]), alpha_rows)
    paths.append(path)
    spatial_rows = []
    faithfulness_rows = []

    def append_faithfulness(
        *, model: str, scope: str, fold: int | None, target: str, values: Mapping[str, Any]
    ) -> None:
        for quantity in ("output_sensitivity", "error_increase"):
            metrics = values.get(quantity)
            faithfulness_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "scope": scope,
                    "fold": fold,
                    "target": target,
                    "quantity": quantity,
                    "sample_count": 0 if metrics is None else metrics["sample_count"],
                    "saliency_mean": None if metrics is None else metrics["mean"],
                    "saliency_sd": None if metrics is None else metrics["sd"],
                    "saliency_median": None if metrics is None else metrics["median"],
                    "saliency_minus_random_mean": (
                        None
                        if metrics is None
                        else metrics["saliency_minus_matched_random_mean"]["mean"]
                    ),
                    "saliency_greater_than_random_rate": (
                        None
                        if metrics is None
                        else metrics["saliency_greater_than_matched_random_mean_rate"]
                    ),
                }
            )

    for model in MODEL_ORDER:
        model_spatial = data["spatial"]["models"][model]
        for fold in model_spatial["folds"]:
            for target, values in fold["targets"].items():
                total = values["valid_map_count"] + values["undefined_map_count"]
                spatial_rows.append(
                    {
                        "model": MODEL_LABELS[model],
                        "fold": fold["fold_index"],
                        "target": target,
                        "requested_maps": total,
                        "valid_maps": values["valid_map_count"],
                        "undefined_maps": values["undefined_map_count"],
                        "undefined_rate": values["undefined_map_count"] / total,
                    }
                )
                append_faithfulness(
                    model=model,
                    scope="fold_target",
                    fold=fold["fold_index"],
                    target=target,
                    values=values,
                )
        for target, values in model_spatial["pooled_targets"].items():
            append_faithfulness(
                model=model,
                scope="pooled_target",
                fold=None,
                target=target,
                values=values,
            )
        append_faithfulness(
            model=model,
            scope="pooled_model",
            fold=None,
            target="ALL",
            values=model_spatial["pooled_all_targets"],
        )
    path = table_root / "gradcam_accounting.csv"
    _write_csv(path, tuple(spatial_rows[0]), spatial_rows)
    paths.append(path)
    path = table_root / "spatial_faithfulness.csv"
    _write_csv(path, tuple(faithfulness_rows[0]), faithfulness_rows)
    paths.append(path)
    path = table_root / "execution_registry.csv"
    _write_csv(path, tuple(data["execution_registry"][0]), data["execution_registry"])
    paths.append(path)
    return paths


def _model_palette() -> dict[str, str]:
    return {
        "blackbox": "#264653",
        "standard_cbm": "#2a9d8f",
        "mixed_cem": "#e9c46a",
        "learned_softmax_gam": "#e76f51",
    }


def build_figures(data: Mapping[str, Any], root: Path, language: str) -> list[Path]:
    """Render ten aggregate-only figures as both PNG and SVG."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text

    if language not in {"en", "zh"}:
        raise ValueError("P10_LANGUAGE_INVALID")
    plt.rcParams["axes.unicode_minus"] = False
    root.mkdir(parents=True, exist_ok=True)
    font = None
    if language == "zh":
        font = FontProperties(fname="/System/Library/Fonts/Supplemental/Songti.ttc")
    palette = _model_palette()
    labels = [MODEL_LABELS[model] for model in MODEL_ORDER]
    paths: list[Path] = []

    def save(fig: Any, name: str) -> None:
        if font is not None:
            for value in fig.findobj(match=Text):
                value.set_fontproperties(font)
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = root / f"{name}_{language}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight", metadata={"Creator": "P10"})
            paths.append(path)
        plt.close(fig)

    title = lambda en, zh: zh if language == "zh" else en
    title_kw = {"fontproperties": font} if font else {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = [2633, 868, *FOLD_COUNTS]
    names = (
        ["结节", "患者", "第0折", "第1折", "第2折", "第3折", "第4折"]
        if language == "zh"
        else ["Nodules", "Patients", "F0", "F1", "F2", "F3", "F4"]
    )
    ax.bar(names, counts, color="#2a9d8f")
    ax.set_title(title("Cohort and fold flow", "队列与五折分布"), **title_kw)
    ax.set_ylabel(title("Count", "数量"), **title_kw)
    save(fig, "figure_01_cohort_flow")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    means = [data["task"]["models"][m]["pooled"]["original_scale_mae"] for m in MODEL_ORDER]
    lows = [data["bootstrap"]["models"][m]["original_scale_mae"]["percentile_2_5"] for m in MODEL_ORDER]
    highs = [data["bootstrap"]["models"][m]["original_scale_mae"]["percentile_97_5"] for m in MODEL_ORDER]
    positions = np.arange(4)
    ax.errorbar(positions, means, yerr=[np.asarray(means)-lows, np.asarray(highs)-means], fmt="o", color="#264653", capsize=5)
    ax.set_xticks(positions, labels, rotation=15)
    ax.set_ylabel("MAE")
    ax.set_title(title("Primary regression with 95% CI", "主要回归表现与95%置信区间"), **title_kw)
    save(fig, "figure_02_primary_performance")

    fig, ax = plt.subplots(figsize=(9, 5))
    pairs = list(data["bootstrap"]["paired_mae_A_minus_B"])
    vals = [data["bootstrap"]["paired_mae_A_minus_B"][p]["estimate_mean"] for p in pairs]
    low = [data["bootstrap"]["paired_mae_A_minus_B"][p]["percentile_2_5"] for p in pairs]
    high = [data["bootstrap"]["paired_mae_A_minus_B"][p]["percentile_97_5"] for p in pairs]
    y = np.arange(len(pairs))
    ax.errorbar(vals, y, xerr=[np.asarray(vals)-low, np.asarray(high)-vals], fmt="o", capsize=4)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(y, [_pair_axis_label(p) for p in pairs])
    ax.set_title(title("Paired ΔMAE", "配对ΔMAE"), **title_kw)
    save(fig, "figure_03_paired_mae")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    auc = [data["task"]["models"][model]["pooled_secondary"]["auroc"] for model in MODEL_ORDER]
    auc_low = [data["bootstrap"]["models"][model]["auroc"]["percentile_2_5"] for model in MODEL_ORDER]
    auc_high = [data["bootstrap"]["models"][model]["auroc"]["percentile_97_5"] for model in MODEL_ORDER]
    positions = np.arange(len(MODEL_ORDER))
    axes[0].errorbar(
        positions,
        auc,
        yerr=[np.asarray(auc) - auc_low, np.asarray(auc_high) - auc],
        fmt="o",
        capsize=4,
    )
    axes[0].set_xticks(positions, labels, rotation=20)
    axes[0].set_ylabel("AUROC")
    axes[0].set_title(title("Extreme-task AUROC", "极端任务AUROC"), **title_kw)
    paired_auc = data["bootstrap"]["paired_auroc_B_minus_A"]
    pairs = list(paired_auc)
    vals = [paired_auc[p]["estimate_mean"] for p in pairs]
    low = [paired_auc[p]["percentile_2_5"] for p in pairs]
    high = [paired_auc[p]["percentile_97_5"] for p in pairs]
    y = np.arange(len(pairs))
    axes[1].errorbar(vals, y, xerr=[np.asarray(vals)-low, np.asarray(high)-vals], fmt="o", capsize=4)
    axes[1].axvline(0, color="#555555", linewidth=1)
    axes[1].set_yticks(y, [_pair_axis_label(p) for p in pairs])
    axes[1].set_title(title("Paired ΔAUROC", "配对ΔAUROC"), **title_kw)
    fig.suptitle(title("Secondary performance and paired comparison", "次要任务表现与配对比较"), **title_kw)
    save(fig, "figure_04_secondary")

    fig, ax = plt.subplots(figsize=(9, 5))
    matrix = []
    for model in MODEL_ORDER[1:]:
        row = []
        for concept in CONCEPT_ORDER:
            metrics = data["concept"][model]["pooled"][concept]
            row.append(metrics.get("mae", metrics.get("soft_cross_entropy")))
        matrix.append(row)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(8), CONCEPT_ORDER, rotation=35, ha="right")
    ax.set_yticks(range(3), labels[1:])
    ax.set_title(title("Concept error metrics", "概念误差指标"), **title_kw)
    fig.colorbar(im, ax=ax)
    save(fig, "figure_05_concept_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model in MODEL_ORDER[1:]:
        report = data["intervention"][model]["error_first"]
        random = data["intervention"][model]["random_permutations"]
        label = MODEL_LABELS[model]
        error_first_label = title("error-first", "误差优先")
        permutation_label = title("permutation mean", "排列均值")
        axes[0].plot(report["k"], report["pooled_original_scale_mae"], marker="o", label=f"{label} {error_first_label}")
        axes[0].plot(random["k"], random["pooled_original_scale_mae_mean"], linestyle="--", label=f"{label} {permutation_label}")
        axes[1].plot(report["k"], report["pooled_auroc"], marker="o", label=f"{label} {error_first_label}")
        axes[1].plot(random["k"], random["pooled_auroc_mean"], linestyle="--", label=f"{label} {permutation_label}")
    axes[0].set_title("MAE")
    axes[1].set_title("AUROC")
    for ax in axes:
        ax.set_xlabel("k")
        ax.legend(fontsize=5.5)
    fig.suptitle(title("Concept-intervention curves", "概念干预曲线"), **title_kw)
    save(fig, "figure_06_intervention_curves")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER[1:], strict=True):
        means = data["contribution_centering"][model]["mean_rating_point_contribution_by_concept"]
        ax.barh(CONCEPT_ORDER, [means[c] for c in CONCEPT_ORDER], color=palette[model])
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_title(MODEL_LABELS[model])
    fig.suptitle(title("Train-fold contribution centers", "训练折概念贡献中心"), **title_kw)
    save(fig, "figure_07_centered_contributions")

    fig, ax = plt.subplots(figsize=(10, 5))
    alpha = np.zeros((5, 8))
    for fold in data["learned_alpha"]["folds"]:
        for column, concept in enumerate(CONCEPT_ORDER):
            alpha[fold["fold_index"], column] = max(fold["groups"][concept]["weights"])
    im = ax.imshow(alpha, vmin=0.19, vmax=max(0.21, float(alpha.max())), aspect="auto", cmap="magma")
    ax.set_xticks(range(8), CONCEPT_ORDER, rotation=35, ha="right")
    ax.set_yticks(
        range(5),
        [f"第{i}折" for i in range(5)] if language == "zh" else [f"Fold {i}" for i in range(5)],
    )
    ax.set_title(title("Maximum learned expert weight", "各组最大学习专家权重"), **title_kw)
    fig.colorbar(im, ax=ax)
    save(fig, "figure_08_learned_alpha")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates = []
    for model in MODEL_ORDER:
        valid, undefined = _pooled_map_counts(data["spatial"]["models"][model])
        rates.append(undefined / (valid + undefined))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=[palette[m] for m in MODEL_ORDER])
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel(title("Undefined rate", "未定义比例"), **title_kw)
    ax.set_title(title("Post-ReLU zero Grad-CAM maps", "ReLU后全零Grad-CAM图"), **title_kw)
    save(fig, "figure_09_gradcam_undefined")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, quantity in zip(axes, ("output_sensitivity", "error_increase"), strict=True):
        values = []
        for model in MODEL_ORDER:
            values.append(data["spatial"]["models"][model]["pooled_all_targets"][quantity]["saliency_minus_matched_random_mean"]["mean"])
        axis.bar(labels, values, color=[palette[m] for m in MODEL_ORDER])
        axis.axhline(0, color="#555555", linewidth=1)
        axis.tick_params(axis="x", rotation=15)
        quantity_label = {
            "output_sensitivity": title("output sensitivity", "输出敏感度"),
            "error_increase": title("prediction error increase", "预测误差增量"),
        }[quantity]
        axis.set_title(quantity_label, **title_kw)
    fig.suptitle(title("Saliency minus matched-random mean", "显著区域减匹配随机均值"), **title_kw)
    save(fig, "figure_10_spatial_faithfulness")
    return paths


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _scientific_capsule(data: Mapping[str, Any], language: str) -> list[str]:
    lines = (
        [
            "2,633 nodules; 868 patients; folds 479/502/539/549/564; patient leakage 0.",
            "Grad-CAM: 73,724 requested = 66,769 valid + 6,955 undefined post-ReLU zero maps.",
            "Bootstrap: 2,000 patient-cluster draws.",
        ]
        if language == "en"
        else [
            "2,633个结节；868名患者；五折计数479/502/539/549/564；患者泄漏0。",
            "Grad-CAM：73,724张请求图 = 66,769张有效图 + 6,955张ReLU后全零未定义图。",
            "Bootstrap：2,000次患者聚类抽样。",
        ]
    )
    for model in MODEL_ORDER:
        pooled = data["task"]["models"][model]["pooled"]
        secondary = data["task"]["models"][model]["pooled_secondary"]
        ci = data["bootstrap"]["models"][model]["original_scale_mae"]
        lines.append(
            f"{MODEL_LABELS[model]}: MAE {_format_float(pooled['original_scale_mae'])} "
            f"[{_format_float(ci['percentile_2_5'])}, {_format_float(ci['percentile_97_5'])}]; "
            f"RMSE {_format_float(pooled['original_scale_rmse'])}; "
            f"AUROC {_format_float(secondary['auroc'])}; AUPRC {_format_float(secondary['auprc'])}."
        )
    return lines


def _model_from_pair(name: str) -> tuple[str, str]:
    first, second = name.split("__", maxsplit=1)
    return MODEL_LABELS[first], MODEL_LABELS[second]


def _pair_axis_label(name: str) -> str:
    first, second = name.split("__", maxsplit=1)
    return f"{MODEL_LABELS[first]} vs {MODEL_LABELS[second]}"


def _curve(values: Sequence[float]) -> str:
    return "/".join(_format_float(float(value)) for value in values)


def _section_bilingual_lines(
    data: Mapping[str, Any], section: str
) -> list[tuple[str, str]]:
    """Return section-specific prose sourced from the shared numeric data model."""
    lines: list[tuple[str, str]] = []
    if section in {"Abstract", "Executive Summary"}:
        return list(
            zip(
                _scientific_capsule(data, "en"),
                _scientific_capsule(data, "zh"),
                strict=True,
            )
        )
    if section in {"Introduction", "Clinical and Scientific Context"}:
        return [
            (
                "LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis [1].",
                "LIDC恶性度是放射科医师评估，并非病理确诊 [1]。",
            ),
            (
                "The system is a research benchmark and is not a clinical diagnostic product.",
                "本系统是研究基准，并非临床诊断产品。",
            ),
            (
                "The models used DenseNet-121 [2]; CBM and CEM terminology follows [3], [4], while Mixed-type CEM and Learned-softmax GAM denote the preregistered project-specific designs.",
                "各模型使用DenseNet-121 [2]；CBM与CEM术语沿用 [3]、[4]，Mixed-type CEM与Learned-softmax GAM则指本项目预注册的特定设计。",
            ),
        ]
    if section in {"Methods", "Unified Evaluation Methods"}:
        return [
            (
                "Primary scores were used without clipping; 2,000 shared patient-cluster bootstrap draws produced percentile 95% CIs [6].",
                "主要分数未经截断；2,000次共享患者聚类Bootstrap抽样生成百分位95%置信区间 [6]。",
            ),
            (
                "Secondary Youden-J thresholds used only fold-specific validation samples with malignancy <= 2 or >= 4.",
                "次要任务Youden-J阈值仅使用各折验证集中恶性度 <= 2或 >= 4的样本。",
            ),
            (
                "Each valid Grad-CAM map [5] used 26,215 saliency voxels and 20 matched random masks.",
                "每张有效Grad-CAM图 [5] 使用26,215个显著体素和20个匹配随机遮罩。",
            ),
            (
                "Paired signs are DeltaMAE=MAE_A-MAE_B and DeltaAUROC=AUROC_B-AUROC_A; positive values favor model B.",
                "配对符号定义为DeltaMAE=MAE_A-MAE_B、DeltaAUROC=AUROC_B-AUROC_A；正值表示模型B更优。",
            ),
            (
                "Intervention signs are Delta_iMAE=baseline_MAE-iMAE and Delta_iAUC=iAUC-baseline_AUROC; positive values denote improvement.",
                "干预符号定义为Delta_iMAE=baseline_MAE-iMAE、Delta_iAUC=iAUC-baseline_AUROC；正值表示改善。",
            ),
        ]
    if section in {"Cohort and Integrity", "Cohort Construction", "Patient-grouped Five-fold Protocol"}:
        return [
                (
                    "The canonical OOF cohort contained 2,633 unique nodules from 868 patients.",
                    "标准OOF队列包含2,633个唯一结节，来自868名患者。",
                ),
            (
                "Outer test fold counts were 479/502/539/549/564; patient leakage was 0.",
                "外层测试折计数为479/502/539/549/564；患者泄漏为0。",
            ),
            (
                "All 4 models used identical targets and fold-specific outer test membership.",
                "全部4个模型使用相同目标值与逐折一致的外层测试成员。",
            ),
        ]
    if section == "Model Architectures":
        return [
            ("P5: Black-box DenseNet-121 regression [2].", "P5：Black-box DenseNet-121回归 [2]。"),
            ("P6: Standard CBM with 8 activated concept groups [3].", "P6：具有8个激活概念组的Standard CBM [3]。"),
            ("P7: project-specific Mixed-type CEM with 8 concept groups [4].", "P7：具有8个概念组的项目特定Mixed-type CEM [4]。"),
            ("P8: Learned-softmax GAM with 8 groups and 5 local experts per group.", "P8：具有8组、每组5个局部专家的Learned-softmax GAM。"),
        ]
    if section in {"Primary and Secondary Results", "Primary Regression Results"}:
        for model in MODEL_ORDER:
            pooled = data["task"]["models"][model]["pooled"]
            ci = data["bootstrap"]["models"][model]
            label = MODEL_LABELS[model]
            numeric = (
                f"MAE {_format_float(pooled['original_scale_mae'])} "
                f"[95% CI {_format_float(ci['original_scale_mae']['percentile_2_5'])}, {_format_float(ci['original_scale_mae']['percentile_97_5'])}]; "
                f"RMSE {_format_float(pooled['original_scale_rmse'])} "
                f"[95% CI {_format_float(ci['original_scale_rmse']['percentile_2_5'])}, {_format_float(ci['original_scale_rmse']['percentile_97_5'])}]; "
                f"normalized MAE {_format_float(pooled['normalized_mae'])} "
                f"[95% CI {_format_float(ci['normalized_mae']['percentile_2_5'])}, {_format_float(ci['normalized_mae']['percentile_97_5'])}]; "
                f"Pearson {_format_float(pooled['pearson'])} "
                f"[95% CI {_format_float(ci['pearson']['percentile_2_5'])}, {_format_float(ci['pearson']['percentile_97_5'])}]; "
                f"Spearman {_format_float(pooled['spearman'])} "
                f"[95% CI {_format_float(ci['spearman']['percentile_2_5'])}, {_format_float(ci['spearman']['percentile_97_5'])}]; "
                f"unclipped 1-5 prediction range [{_format_float(pooled['prediction_range_1_to_5'][0])}, {_format_float(pooled['prediction_range_1_to_5'][1])}]; "
                f"below 1 rate {_format_float(pooled['below_one_rate'])}; above 5 rate {_format_float(pooled['above_five_rate'])}."
            )
            lines.append((f"{label}: {numeric}", f"{label}：{numeric}"))
        for pair, value in data["bootstrap"]["paired_mae_A_minus_B"].items():
            first, second = _model_from_pair(pair)
            crosses_en = "yes" if value["ci_crosses_zero"] else "no"
            crosses_zh = "是" if value["ci_crosses_zero"] else "否"
            numeric = (
                f"DeltaMAE {first} - {second} = {_format_float(value['estimate_mean'])} "
                f"[95% CI {_format_float(value['percentile_2_5'])}, {_format_float(value['percentile_97_5'])}]"
            )
            lines.append((f"{numeric}; crosses zero: {crosses_en}.", f"{numeric}；跨零：{crosses_zh}。"))
        if section == "Primary Regression Results":
            return lines
    if section in {"Secondary Extreme-task Results", "Primary and Secondary Results"}:
        for model in MODEL_ORDER:
            value = data["task"]["models"][model]["pooled_secondary"]
            ci_auc = data["bootstrap"]["models"][model]["auroc"]
            ci_pr = data["bootstrap"]["models"][model]["auprc"]
            numeric = (
                f"AUROC {_format_float(value['auroc'])} [95% CI {_format_float(ci_auc['percentile_2_5'])}, {_format_float(ci_auc['percentile_97_5'])}]; "
                f"AUPRC {_format_float(value['auprc'])} [95% CI {_format_float(ci_pr['percentile_2_5'])}, {_format_float(ci_pr['percentile_97_5'])}]."
            )
            lines.append((f"{MODEL_LABELS[model]}: {numeric}", f"{MODEL_LABELS[model]}：{numeric}"))
        for pair, value in data["bootstrap"]["paired_auroc_B_minus_A"].items():
            first, second = _model_from_pair(pair)
            crosses_en = "yes" if value["ci_crosses_zero"] else "no"
            crosses_zh = "是" if value["ci_crosses_zero"] else "否"
            numeric = (
                f"DeltaAUROC {second} - {first} = {_format_float(value['estimate_mean'])} "
                f"[95% CI {_format_float(value['percentile_2_5'])}, {_format_float(value['percentile_97_5'])}]"
            )
            lines.append((f"{numeric}; crosses zero: {crosses_en}.", f"{numeric}；跨零：{crosses_zh}。"))
        return lines
    if section in {"Concept and Intervention Results", "Concept Prediction Results"}:
        for model in MODEL_ORDER[1:]:
            for concept in CONCEPT_ORDER:
                value = data["concept"][model]["pooled"][concept]
                if "mae" in value:
                    numeric = (
                        f"MAE {_format_float(value['mae'])}; RMSE {_format_float(value['rmse'])}; "
                        f"Pearson {_format_float(value['pearson'])}; Spearman {_format_float(value['spearman'])}; N {int(value['sample_count'])}."
                    )
                else:
                    numeric = (
                        f"soft CE {_format_float(value['soft_cross_entropy'])}; Brier {_format_float(value['multiclass_brier'])}; "
                        f"macro-F1 {_format_float(value['hard_modal_macro_f1'])}; soft N {int(value['soft_sample_count'])}; "
                        f"hard N {int(value['hard_sample_count'])}; ties {int(value['true_tie_count'])}."
                    )
                label = f"{MODEL_LABELS[model]} / {concept}"
                lines.append((f"{label}: {numeric}", f"{label}：{numeric}"))
        if section == "Concept Prediction Results":
            return lines
    if section == "Contribution Centering":
        for model in MODEL_ORDER[1:]:
            value = data["contribution_centering"][model]
            contributions = "/".join(
                f"{concept}={_format_float(value['mean_rating_point_contribution_by_concept'][concept])}"
                for concept in CONCEPT_ORDER
            )
            positive = value["most_positive"]
            negative = value["most_negative"]
            if negative is None:
                smallest = value["smallest_signed"]
                negative_en = (
                    f"no negative pooled mean; smallest positive {smallest[0]}="
                    f"{_format_float(smallest[1])}"
                )
                negative_zh = (
                    f"无负向汇总均值；最小正贡献{smallest[0]}="
                    f"{_format_float(smallest[1])}"
                )
            else:
                negative_en = (
                    f"most negative {negative[0]}={_format_float(negative[1])}"
                )
                negative_zh = (
                    f"最大负向{negative[0]}={_format_float(negative[1])}"
                )
            lines.append(
                (
                    f"{MODEL_LABELS[model]} centered rating contributions: {contributions}; most positive {positive[0]}={_format_float(positive[1])}; {negative_en}.",
                    f"{MODEL_LABELS[model]}中心化评分贡献：{contributions}；最大正向{positive[0]}={_format_float(positive[1])}；{negative_zh}。",
                )
            )
        return lines
    if section in {"Concept Intervention", "Concept and Intervention Results"}:
        for model in MODEL_ORDER[1:]:
            random = data["intervention"][model]["random_permutations"]
            value = data["intervention"][model]["error_first"]
            label = MODEL_LABELS[model]
            lines.extend(
                (
                    (
                        f"{label} 100-permutation mean k=0/1/2/3/4/5/6/7/8 MAE: {_curve(random['pooled_original_scale_mae_mean'])}.",
                        f"{label} 100次排列均值k=0/1/2/3/4/5/6/7/8 MAE：{_curve(random['pooled_original_scale_mae_mean'])}。",
                    ),
                    (
                        f"{label} permutation iMAE {_format_float(random['iMAE'])}; Delta_iMAE {_format_float(random['Delta_iMAE'])} (positive denotes improvement).",
                        f"{label}排列iMAE {_format_float(random['iMAE'])}；Delta_iMAE {_format_float(random['Delta_iMAE'])}（正值表示改善）。",
                    ),
                    (
                        f"{label} 100-permutation mean k=0/1/2/3/4/5/6/7/8 AUROC: {_curve(random['pooled_auroc_mean'])}.",
                        f"{label} 100次排列均值k=0/1/2/3/4/5/6/7/8 AUROC：{_curve(random['pooled_auroc_mean'])}。",
                    ),
                    (
                        f"{label} permutation iAUC {_format_float(random['iAUC'])}; Delta_iAUC {_format_float(random['Delta_iAUC'])} (positive denotes improvement).",
                        f"{label}排列iAUC {_format_float(random['iAUC'])}；Delta_iAUC {_format_float(random['Delta_iAUC'])}（正值表示改善）。",
                    ),
                    (
                        f"{label} error-first k=0/1/2/3/4/5/6/7/8 MAE: {_curve(value['pooled_original_scale_mae'])}.",
                        f"{label}误差优先k=0/1/2/3/4/5/6/7/8 MAE：{_curve(value['pooled_original_scale_mae'])}。",
                    ),
                    (
                        f"{label} iMAE {_format_float(value['iMAE'])}; Delta_iMAE {_format_float(value['Delta_iMAE'])} (positive denotes improvement).",
                        f"{label} iMAE {_format_float(value['iMAE'])}；Delta_iMAE {_format_float(value['Delta_iMAE'])}（正值表示改善）。",
                    ),
                    (
                        f"{label} error-first k=0/1/2/3/4/5/6/7/8 AUROC: {_curve(value['pooled_auroc'])}.",
                        f"{label}误差优先k=0/1/2/3/4/5/6/7/8 AUROC：{_curve(value['pooled_auroc'])}。",
                    ),
                    (
                        f"{label} iAUC {_format_float(value['iAUC'])}; Delta_iAUC {_format_float(value['Delta_iAUC'])} (positive denotes improvement).",
                        f"{label} iAUC {_format_float(value['iAUC'])}；Delta_iAUC {_format_float(value['Delta_iAUC'])}（正值表示改善）。",
                    ),
                )
            )
        return lines
    if section == "Learned GAM Alpha":
        for fold in data["learned_alpha"]["folds"]:
            weights = "/".join(
                f"{concept}={_format_float(max(fold['groups'][concept]['weights']))}"
                for concept in CONCEPT_ORDER
            )
            lines.append((f"Fold {fold['fold_index']} maximum expert weights: {weights}.", f"Fold {fold['fold_index']}各组最大专家权重：{weights}。"))
        return lines
    if section in {"Grad-CAM Methods"}:
        return [
            ("Grad-CAM used spatial-mean gradients, weighted activations, ReLU, and trilinear upsampling to 64^3 [5].", "Grad-CAM使用空间均值梯度、加权激活、ReLU及三线性上采样至64^3 [5]。"),
            ("Raw FP32 maps were stored without normalization; all-zero post-ReLU maps were explicitly undefined.", "原始FP32图未经归一化保存；ReLU后全零图被明确记为未定义。"),
            ("Occlusion preserved output_sensitivity and error_increase separately for saliency and 20 matched-random masks.", "遮挡分析分别保存显著区域与20个匹配随机遮罩的output_sensitivity和error_increase。"),
        ]
    if section in {"Spatial Explanation Results", "Grad-CAM Accounting"}:
        account = data["gradcam_accounting"]
        lines.append(
            (
                f"Requested {account['requested']} = valid {account['valid']} + undefined {account['undefined']}; undefined rate {_format_float(account['undefined_rate'])}.",
                f"请求图{account['requested']} = 有效图{account['valid']} + 未定义图{account['undefined']}；未定义率{_format_float(account['undefined_rate'])}。",
            )
        )
        for model in MODEL_ORDER:
            model_payload = data["spatial"]["models"][model]
            valid, undefined = _pooled_map_counts(model_payload)
            rate = undefined / (valid + undefined)
            lines.append((f"{MODEL_LABELS[model]}: valid {valid}; undefined {undefined}; rate {_format_float(rate)}.", f"{MODEL_LABELS[model]}：有效{valid}；未定义{undefined}；比例{_format_float(rate)}。"))
            for fold in model_payload["folds"]:
                fold_valid = sum(
                    int(target["valid_map_count"])
                    for target in fold["targets"].values()
                )
                fold_undefined = sum(
                    int(target["undefined_map_count"])
                    for target in fold["targets"].values()
                )
                fold_rate = fold_undefined / (fold_valid + fold_undefined)
                lines.append((f"{MODEL_LABELS[model]} fold {fold['fold_index']}: valid {fold_valid}; undefined {fold_undefined}; rate {_format_float(fold_rate)}.", f"{MODEL_LABELS[model]} fold {fold['fold_index']}：有效{fold_valid}；未定义{fold_undefined}；比例{_format_float(fold_rate)}。"))
        lines.append(("The complete model x fold x target/concept breakdown is retained in Table 7 (gradcam_accounting.csv).", "完整model x fold x target/concept明细保存在表7（gradcam_accounting.csv）。"))
        if section == "Grad-CAM Accounting":
            return lines
    if section in {"Spatial Faithfulness", "Spatial Explanation Results"}:
        for model in MODEL_ORDER:
            pooled = data["spatial"]["models"][model]["pooled_all_targets"]
            for quantity in ("output_sensitivity", "error_increase"):
                value = pooled[quantity]
                delta = value["saliency_minus_matched_random_mean"]
                numeric = (
                    f"saliency mean {_format_float(value['mean'])}; saliency-random mean {_format_float(delta['mean'])}; "
                    f"95% range [{_format_float(delta['percentile_2_5'])}, {_format_float(delta['percentile_97_5'])}]; "
                    f"saliency > random mean rate {_format_float(value['saliency_greater_than_matched_random_mean_rate'])}."
                )
                lines.append((f"{MODEL_LABELS[model]} {quantity}: {numeric}", f"{MODEL_LABELS[model]} {quantity}：{numeric}"))
        lines.append(
            (
                "Complete fold-target, pooled-target, and pooled-model results are retained in Table 8 (spatial_faithfulness.csv).",
                "完整折-目标、汇总目标及汇总模型结果保存在表8（spatial_faithfulness.csv）。",
            )
        )
        return lines
    if section in {"Training and Test Governance", "Execution Provenance"}:
        provenance = [
            ("The execution registry contains 40 immutable model/fold records spanning P5-P9.", "执行登记包含覆盖P5-P9的40条不可变model/fold记录。"),
            ("P5-P8 each retained exactly 1 valid committed test evaluation per fold; P9 created 0 additional test evaluations.", "P5-P8每折均仅保留1次有效提交测试评估；P9新增测试评估为0。"),
            ("Verifier recoveries are distinguished from scientific execution and do not change persisted predictions or metrics.", "Verifier恢复与科学执行明确区分，且不改变已保存预测或指标。"),
        ]
        if section == "Execution Provenance":
            for event in data["execution_events"]:
                numeric = (
                    f"P9 {event['event']}: job {event['job_id']}; {event['resource']}; "
                    f"Exit_status {event['exit_status']}; scientific status {event['scientific_status']}."
                )
                chinese = (
                    f"P9 {event['event']}：任务{event['job_id']}；{event['resource']}；"
                    f"Exit_status {event['exit_status']}；科学状态{event['scientific_status']}。"
                )
                provenance.append((numeric, chinese))
        return provenance
    if section in {"Reproducibility", "Storage and Reproducibility"}:
        reproducibility = [
            ("All public values, CIs, tables, and figures derive from 1 shared report_data.json model.", "全部公开数值、置信区间、表格与图均来自1个共享report_data.json数据模型。"),
            ("GitHub excludes checkpoints, private predictions, raw Grad-CAM maps, CT/ROI volumes, UIDs, and patient keys.", "GitHub排除检查点、私有预测、原始Grad-CAM图、CT/ROI体数据、UID及patient key。"),
            ("The private archive is verified file-by-file with SHA-256 and is stored only on the Mac.", "私有备份按文件逐一使用SHA-256验证，且仅保存在Mac。"),
        ]
        if "private_archive" in data:
            archive = data["private_archive"]
            reproducibility.append(
                (
                    f"Verified private archive: {archive['file_count']} files; {archive['total_bytes']} bytes; manifest {archive['manifest_sha256']}.",
                    f"已验证私有备份：{archive['file_count']}个文件；{archive['total_bytes']}字节；manifest {archive['manifest_sha256']}。",
                )
            )
        table_names = (
            "primary_secondary_metrics.csv",
            "paired_comparisons.csv",
            "concept_metrics.csv",
            "intervention_curves.csv",
            "centered_contributions.csv",
            "learned_gam_alpha.csv",
            "gradcam_accounting.csv",
            "spatial_faithfulness.csv",
            "execution_registry.csv",
        )
        reproducibility.extend(
            (
                f"Table {index}: {name}.",
                f"表{index}：{name}。",
            )
            for index, name in enumerate(table_names, start=1)
        )
        return reproducibility
    if section in {"Discussion and Limitations", "Negative Findings"}:
        findings = [
            ("Learned-softmax GAM had the lowest pooled MAE point estimate, but not every paired CI excluded 0.", "Learned-softmax GAM的pooled MAE点估计最低，但并非所有配对置信区间都排除0。"),
            ("Intervention benefit was model-dependent; positive improvement was not uniform across k=0-8.", "干预收益依赖模型；在k=0-8范围内并非始终呈正向改善。"),
            ("Saliency was not uniformly more faithful than matched random masks for either faithfulness quantity.", "对于两种faithfulness量，显著区域并非始终优于匹配随机遮罩。"),
            ("Unclipped Black-box scores extended below 1 and above 5; Standard CBM and Mixed-type CEM also produced a small fraction below 1, whereas Learned-softmax GAM stayed within the rating range.", "未经截断的Black-box分数同时低于1并高于5；Standard CBM与Mixed-type CEM也有少量分数低于1，而Learned-softmax GAM保持在评分范围内。"),
        ]
        if section == "Negative Findings":
            return findings
        return findings + [
            ("All 6,955 undefined maps were confirmed post-ReLU zeros, but their full pre-ReLU/gradient decomposition was not persisted.", "全部6,955张未定义图均确认是ReLU后全零，但未保存其完整ReLU前/梯度分解。"),
            ("The concentration of zero maps is classified SYSTEMATIC_MODEL/TARGET_ISSUE rather than a proven implementation bug.", "全零图的集中分布被归类为SYSTEMATIC_MODEL/TARGET_ISSUE，而非已证实的实现缺陷。"),
        ]
    if section == "Limitations":
        return [
            ("LIDC malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, limiting claims about clinical cancer detection.", "LIDC恶性度是放射科医师评估而非病理确诊，因此不能据此声称具备临床癌症检测效用。"),
            ("The study evaluates one patient-grouped five-fold cohort and does not establish external-site generalization.", "本研究评估的是一个按患者分组的五折队列，尚未证明对外部中心的泛化能力。"),
            ("The exact pre-ReLU CAM, gradient, activation, and channel-weight decomposition was not persisted, so the 6,955 zero maps cannot be mechanistically resolved without a prohibited new forward pass.", "现有产物未保存完整的ReLU前CAM、梯度、激活及通道权重分解；若不执行被禁止的新前向计算，无法对6,955张全零图作机制层面的精确归因。"),
            ("Occlusion on normalized-zero voxels is a registered perturbation test, not a causal explanation of malignancy.", "将体素置为归一化零值的遮挡实验是预注册扰动检验，并非恶性度的因果解释。"),
            ("This research benchmark is not a clinical diagnostic product.", "本研究基准不是临床诊断产品。"),
        ]
    if section == "Conclusions":
        return [
            ("Learned-softmax GAM achieved the strongest primary point estimate; uncertainty and spatial limitations remain material.", "Learned-softmax GAM取得最佳主要任务点估计；不确定性与空间解释局限仍然重要。"),
            ("The findings support research comparison only and do not establish clinical diagnostic utility.", "这些发现仅支持研究比较，并不确立临床诊断效用。"),
        ]
    if section == "References":
        return [(reference, reference) for reference in data["references"]]
    return list(
        zip(
            _scientific_capsule(data, "en"),
            _scientific_capsule(data, "zh"),
            strict=True,
        )
    )


SECTIONS = {
    "short": (
        "Abstract",
        "Introduction",
        "Methods",
        "Cohort and Integrity",
        "Primary and Secondary Results",
        "Concept Prediction Results",
        "Concept Intervention",
        "Grad-CAM Accounting",
        "Spatial Faithfulness",
        "Discussion and Limitations",
        "Reproducibility",
        "References",
    ),
    "technical": (
        "Executive Summary",
        "Clinical and Scientific Context",
        "Cohort Construction",
        "Patient-grouped Five-fold Protocol",
        "Model Architectures",
        "Training and Test Governance",
        "Unified Evaluation Methods",
        "Primary Regression Results",
        "Secondary Extreme-task Results",
        "Concept Prediction Results",
        "Contribution Centering",
        "Concept Intervention",
        "Learned GAM Alpha",
        "Grad-CAM Methods",
        "Grad-CAM Accounting",
        "Spatial Faithfulness",
        "Execution Provenance",
        "Storage and Reproducibility",
        "Negative Findings",
        "Limitations",
        "Conclusions",
        "References",
    ),
}
ZH_SECTIONS = {
    "Abstract": "摘要",
    "Introduction": "引言",
    "Methods": "方法",
    "Cohort and Integrity": "队列与完整性",
    "Primary and Secondary Results": "主要与次要结果",
    "Concept and Intervention Results": "概念与干预结果",
    "Spatial Explanation Results": "空间解释结果",
    "Discussion and Limitations": "讨论与局限",
    "Reproducibility": "可复现性",
    "References": "参考文献",
    "Executive Summary": "执行摘要",
    "Clinical and Scientific Context": "临床与科学背景",
    "Cohort Construction": "队列构建",
    "Patient-grouped Five-fold Protocol": "患者分组五折协议",
    "Model Architectures": "模型架构",
    "Training and Test Governance": "训练与测试治理",
    "Unified Evaluation Methods": "统一评估方法",
    "Primary Regression Results": "主要回归结果",
    "Secondary Extreme-task Results": "次要极端任务结果",
    "Concept Prediction Results": "概念预测结果",
    "Contribution Centering": "贡献中心化",
    "Concept Intervention": "概念干预",
    "Learned GAM Alpha": "GAM学习权重",
    "Grad-CAM Methods": "Grad-CAM方法",
    "Grad-CAM Accounting": "Grad-CAM计数",
    "Spatial Faithfulness": "空间忠实度",
    "Execution Provenance": "执行溯源",
    "Storage and Reproducibility": "存储与可复现性",
    "Negative Findings": "负面发现",
    "Limitations": "局限",
    "Conclusions": "结论",
}
SHORT_SECTION_FIGURES = {
    "Cohort and Integrity": 1,
    "Primary and Secondary Results": 2,
    "Concept Prediction Results": 5,
    "Concept Intervention": 6,
    "Grad-CAM Accounting": 9,
    "Spatial Faithfulness": 10,
}


def build_markdown(data: Mapping[str, Any], variant: str, language: str) -> str:
    if variant not in SECTIONS or language not in {"en", "zh"}:
        raise ValueError("P10_REPORT_VARIANT_OR_LANGUAGE_INVALID")
    title = (
        "Baseline-v2: Unified Evaluation of Interpretable 3D Models for LIDC-IDRI"
        if language == "en"
        else "Baseline-v2：LIDC-IDRI可解释三维模型的统一评估"
    )
    output = [f"# {title}", "", "**REPORT-DATA-SHA256:** `PENDING_RENDER_HASH`", ""]
    body_en = (
        "All results were reconstructed read-only from frozen P5–P9 evidence. "
        "Primary predictions were not clipped. LIDC malignancy is a radiologist "
        "assessment rather than pathology-confirmed diagnosis, and this system is not "
        "a clinical diagnostic product."
    )
    body_zh = (
        "所有结果均由冻结的P5–P9证据只读重建。主要预测分数未经截断。LIDC恶性度是"
        "放射科医师评估，并非病理确诊；本系统不是临床诊断产品。"
    )
    for index, section in enumerate(SECTIONS[variant], start=1):
        heading = ZH_SECTIONS[section] if language == "zh" else section
        output.extend((f"## {index}. {heading}", "", body_zh if language == "zh" else body_en, ""))
        column = 1 if language == "zh" else 0
        output.extend(
            f"- {pair[column]}" for pair in _section_bilingual_lines(data, section)
        )
        output.append("")
    output.extend(("## Scientific conclusion codes", ""))
    output.extend(f"- `{code}`" for code in data["scientific_conclusion_codes"])
    output.append("")
    return "\n".join(output)


def extract_numeric_tokens(text: str) -> list[str]:
    cleaned = re.sub(r"[0-9a-f]{64}", "", text, flags=re.IGNORECASE)
    return re.findall(
        r"(?<![A-Za-z0-9])[+-]?(?:\d+(?:,\d{3})*|\d*\.\d+)(?:[eE][+-]?\d+)?(?![A-Za-z])",
        cleaned,
    )


def verify_bilingual_markdown(en_text: str, zh_text: str, variant: str) -> None:
    if extract_numeric_tokens(en_text) != extract_numeric_tokens(zh_text):
        raise ValueError(f"P10_BILINGUAL_NUMERIC_TOKEN_MISMATCH:{variant}")
    for code in (
        "GAM_LOWEST_POINT_ESTIMATE_MAE",
        "PAIRED_MAE_SUPPORTS_GAM_OVER_BLACKBOX_AND_CBM",
        "AUROC_DIFFERENCES_MOSTLY_UNCERTAIN",
        "INTERVENTION_BENEFIT_MODEL_DEPENDENT",
        "SALIENCY_NOT_UNIFORMLY_MORE_FAITHFUL_THAN_RANDOM",
        "SYSTEMATIC_MODEL_TARGET_ZERO_MAP_LIMITATION",
    ):
        if en_text.count(code) != 1 or zh_text.count(code) != 1:
            raise ValueError(f"P10_BILINGUAL_CONCLUSION_CODE_MISMATCH:{code}")
    if len(SECTIONS[variant]) != sum(1 for line in en_text.splitlines() if line.startswith("## ") and "Scientific conclusion" not in line):
        raise ValueError("P10_EN_SECTION_COUNT_INVALID")
    if len(SECTIONS[variant]) != sum(1 for line in zh_text.splitlines() if line.startswith("## ") and "Scientific conclusion" not in line):
        raise ValueError("P10_ZH_SECTION_COUNT_INVALID")


def _wrap_text(text: str, width: int) -> list[str]:
    # Concept vectors are slash-delimited. Make each separator a legal wrap point so
    # long scientific tokens never split a decimal number in extracted PDF text.
    wrapped_text = text.replace("/", "/ ")
    for punctuation in ("；", "。", "，", "："):
        wrapped_text = wrapped_text.replace(punctuation, f"{punctuation} ")
    words = wrapped_text.split(" ") if " " in wrapped_text else list(wrapped_text)
    normalized_words: list[str] = []
    for word in words:
        if len(word) <= width:
            normalized_words.append(word)
        else:
            normalized_words.extend(
                word[start : start + width] for start in range(0, len(word), width)
            )
    lines: list[str] = []
    current = ""
    for word in normalized_words:
        separator = " " if current and " " in wrapped_text else ""
        candidate = current + separator + word
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_pdf(
    data: Mapping[str, Any],
    *,
    variant: str,
    language: str,
    destination: Path,
    figure_root: Path,
    report_data_sha256: str,
) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
    except ImportError as error:
        raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED") from error
    if language == "zh":
        font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
        pdfmetrics.registerFont(TTFont("P10Songti", font_path, subfontIndex=6))
        pdfmetrics.registerFont(TTFont("P10SongtiBold", font_path, subfontIndex=1))
        regular, bold = "P10Songti", "P10SongtiBold"
    else:
        regular, bold = "Helvetica", "Helvetica-Bold"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    page_width, page_height = A4
    doc = canvas.Canvas(
        str(temporary),
        pagesize=A4,
        pageCompression=1,
        initialFontName=regular,
        initialFontSize=9,
    )
    doc.setTitle(
        "Baseline-v2 Final Report" if language == "en" else "Baseline-v2最终报告"
    )
    doc.setAuthor("LIDC-IDRI Baseline-v2")
    doc.setSubject("Read-only P10 bilingual scientific report")
    doc.setCreator("lidc_baseline.p10_report")
    sections = list(SECTIONS[variant])
    page_plan: list[tuple[str, Any]] = [("section", section) for section in sections]
    if variant == "technical":
        page_plan.extend(("figure", figure_number) for figure_number in range(1, 11))
    page_target = len(page_plan)
    for page_index, (page_kind, page_value) in enumerate(page_plan):
        if page_kind == "section":
            section = str(page_value)
            heading = ZH_SECTIONS[section] if language == "zh" else section
        else:
            figure_number = int(page_value)
            heading = (
                f"图 {figure_number}：聚合科学结果"
                if language == "zh"
                else f"Figure {figure_number}: Aggregate scientific results"
            )
        doc.setFillColorRGB(0.12, 0.25, 0.29)
        doc.rect(0, page_height - 55, page_width, 55, fill=1, stroke=0)
        doc.setFillColorRGB(1, 1, 1)
        doc.setFont(bold, 16)
        doc.drawString(42, page_height - 36, f"{page_index + 1}. {heading}")
        doc.setFillColorRGB(0.1, 0.1, 0.1)
        if page_kind == "section":
            y = page_height - 82
            intro = (
                "This page summarizes frozen, deidentified P5–P9 evidence. No new model forward or scientific job was used."
                if language == "en"
                else "本页汇总冻结且脱敏的P5–P9证据；未运行新的模型前向或科学作业。"
            )
            for line in _wrap_text(intro, 88 if language == "en" else 48):
                doc.setFont(regular, 9)
                doc.drawString(44, y, line)
                y -= 13
            y -= 5
            section_lines = _section_bilingual_lines(data, section)
            if section == "References":
                section_lines = [
                    *section_lines,
                    *(
                        (f"Scientific conclusion code: {code}.", f"科学结论代码：{code}。")
                        for code in data["scientific_conclusion_codes"]
                    ),
                ]
            content_font = 7.8 if len(section_lines) > 20 else 8.5
            content_leading = 9.2 if len(section_lines) > 20 else 11
            wrap_width = 118 if language == "en" else 68
            doc.setFont(regular, content_font)
            for pair in section_lines:
                line = pair[1 if language == "zh" else 0]
                for wrapped in _wrap_text(line, wrap_width):
                    doc.drawString(50, y, wrapped)
                    y -= content_leading
                y -= 2
            if variant == "short":
                figure_number = SHORT_SECTION_FIGURES.get(section)
                candidates = (
                    []
                    if figure_number is None
                    else sorted(
                        figure_root.glob(
                            f"figure_{figure_number:02d}_*_{language}.png"
                        )
                    )
                )
                maximum_wrapped_lines = max(
                    sum(len(_wrap_text(pair[column], width)) for pair in section_lines)
                    for column, width in ((0, 118), (1, 68))
                )
                if candidates and maximum_wrapped_lines <= 22:
                    doc.drawImage(
                        str(candidates[0]),
                        58,
                        100,
                        width=page_width - 116,
                        height=220,
                        preserveAspectRatio=True,
                        anchor="c",
                    )
                    doc.setFont(regular, 7)
                    caption = (
                        f"Figure {figure_number}"
                        if language == "en"
                        else f"图 {figure_number}"
                    )
                    doc.drawCentredString(page_width / 2, 88, caption)
        else:
            candidates = sorted(
                figure_root.glob(f"figure_{figure_number:02d}_*_{language}.png")
            )
            if len(candidates) != 1:
                raise ValueError(f"P10_REPORT_FIGURE_MISSING:{figure_number}:{language}")
            doc.drawImage(
                str(candidates[0]),
                42,
                120,
                width=page_width - 84,
                height=page_height - 220,
                preserveAspectRatio=True,
                anchor="c",
            )
            doc.setFont(regular, 9)
            caption = (
                f"Figure {figure_number}. Values originate from the shared report data model."
                if language == "en"
                else f"图 {figure_number}。数值均来自共享报告数据模型。"
            )
            doc.drawCentredString(page_width / 2, 95, caption)
        doc.setStrokeColorRGB(0.8, 0.8, 0.8)
        doc.line(42, 45, page_width - 42, 45)
        doc.setFont(regular, 7)
        doc.drawString(42, 31, f"REPORT-DATA-SHA256 {report_data_sha256}")
        doc.drawRightString(page_width - 42, 31, f"{page_index + 1}/{page_target}")
        doc.showPage()
    doc.save()
    temporary.replace(destination)


def _data_dictionary_rows(language: str) -> list[dict[str, str]]:
    definitions = {
        "model": ("Frozen model identifier", "冻结模型标识"),
        "concept": ("Canonical concept-group name", "规范概念组名称"),
        "target": ("Grad-CAM or prediction target name", "Grad-CAM或预测目标名称"),
        "scope": ("Fold-target, pooled-target, or pooled-model aggregation level", "折-目标、汇总目标或汇总模型的聚合层级"),
        "fold": ("Outer patient-grouped fold index", "患者分组外层折索引"),
        "mae": ("Mean absolute error on the 1–5 scale", "1–5量表平均绝对误差"),
        "mae_sd": ("Across-permutation MAE standard deviation; blank for error-first", "排列间MAE标准差；误差优先策略为空"),
        "mae_ci_low": ("2.5th percentile patient-bootstrap MAE", "患者Bootstrap MAE第2.5百分位"),
        "mae_ci_high": ("97.5th percentile patient-bootstrap MAE", "患者Bootstrap MAE第97.5百分位"),
        "rmse": ("Root mean squared error on the 1–5 scale", "1–5量表均方根误差"),
        "rmse_ci_low": ("2.5th percentile patient-bootstrap RMSE", "患者Bootstrap RMSE第2.5百分位"),
        "rmse_ci_high": ("97.5th percentile patient-bootstrap RMSE", "患者Bootstrap RMSE第97.5百分位"),
        "normalized_mae": ("Mean absolute error on the normalized 0–1 scale", "归一化0–1量表平均绝对误差"),
        "normalized_mae_ci_low": ("2.5th percentile patient-bootstrap normalized MAE", "患者Bootstrap归一化MAE第2.5百分位"),
        "normalized_mae_ci_high": ("97.5th percentile patient-bootstrap normalized MAE", "患者Bootstrap归一化MAE第97.5百分位"),
        "pearson": ("Pearson correlation coefficient", "Pearson相关系数"),
        "pearson_ci_low": ("2.5th percentile patient-bootstrap Pearson correlation", "患者Bootstrap Pearson相关系数第2.5百分位"),
        "pearson_ci_high": ("97.5th percentile patient-bootstrap Pearson correlation", "患者Bootstrap Pearson相关系数第97.5百分位"),
        "spearman": ("Spearman rank correlation coefficient", "Spearman秩相关系数"),
        "spearman_ci_low": ("2.5th percentile patient-bootstrap Spearman correlation", "患者Bootstrap Spearman相关系数第2.5百分位"),
        "spearman_ci_high": ("97.5th percentile patient-bootstrap Spearman correlation", "患者Bootstrap Spearman相关系数第97.5百分位"),
        "prediction_min_1_to_5": ("Minimum unclipped prediction on the original 1-5 rating scale", "原始1-5评分量表上未经截断预测的最小值"),
        "prediction_max_1_to_5": ("Maximum unclipped prediction on the original 1-5 rating scale", "原始1-5评分量表上未经截断预测的最大值"),
        "below_one_rate": ("Fraction of unclipped predictions below rating 1", "未经截断预测低于评分1的比例"),
        "above_five_rate": ("Fraction of unclipped predictions above rating 5", "未经截断预测高于评分5的比例"),
        "auroc": ("Area under the ROC curve on extreme cases", "极端病例ROC曲线下面积"),
        "auroc_sd": ("Across-permutation AUROC standard deviation; blank for error-first", "排列间AUROC标准差；误差优先策略为空"),
        "auroc_ci_low": ("2.5th percentile patient-bootstrap AUROC", "患者Bootstrap AUROC第2.5百分位"),
        "auroc_ci_high": ("97.5th percentile patient-bootstrap AUROC", "患者Bootstrap AUROC第97.5百分位"),
        "auprc": ("Area under the precision-recall curve on extreme cases", "极端病例精确率-召回率曲线下面积"),
        "auprc_ci_low": ("2.5th percentile patient-bootstrap AUPRC", "患者Bootstrap AUPRC第2.5百分位"),
        "auprc_ci_high": ("97.5th percentile patient-bootstrap AUPRC", "患者Bootstrap AUPRC第97.5百分位"),
        "metric": ("Registered metric or paired-comparison quantity", "预注册指标或配对比较量"),
        "value": ("Metric value in the units defined by metric", "按metric字段定义单位表示的指标值"),
        "model_A": ("First model in the registered signed comparison", "预注册带符号比较中的第一个模型"),
        "model_B": ("Second model in the registered signed comparison", "预注册带符号比较中的第二个模型"),
        "estimate": ("Observed signed paired-model difference", "观测到的带符号模型配对差异"),
        "ci_low": ("2.5th percentile of the paired bootstrap difference", "配对Bootstrap差异第2.5百分位"),
        "ci_high": ("97.5th percentile of the paired bootstrap difference", "配对Bootstrap差异第97.5百分位"),
        "undefined_rate": ("Post-ReLU all-zero map fraction", "ReLU后全零图比例"),
        "ci_crosses_zero": ("Whether the paired percentile CI includes zero", "配对百分位置信区间是否包含零"),
        "mean_train_center_rating_points": ("Fold-train mean concept contribution on the 1–5 rating scale", "折内训练集概念贡献在1–5评分量表上的均值"),
        "strategy": ("Registered concept-intervention ordering strategy", "预注册概念干预排序策略"),
        "k": ("Number of intervened concept groups", "已干预概念组数量"),
        "expert": ("Local GAM expert index within a concept group", "概念组内局部GAM专家索引"),
        "weight": ("Learned softmax expert weight", "学习得到的softmax专家权重"),
        "logit": ("Learned softmax expert logit", "学习得到的softmax专家logit"),
        "gradient_l1": ("Recorded L1 norm of the alpha-logit gradient", "已记录的alpha-logit梯度L1范数"),
        "requested_maps": ("Requested Grad-CAM maps", "请求生成的Grad-CAM图数量"),
        "valid_maps": ("Persisted nonzero post-ReLU Grad-CAM maps", "已保存的非零ReLU后Grad-CAM图数量"),
        "undefined_maps": ("Explicitly recorded post-ReLU all-zero maps", "明确记录的ReLU后全零图数量"),
        "quantity": ("Faithfulness quantity: output_sensitivity or error_increase", "faithfulness量：output_sensitivity或error_increase"),
        "sample_count": ("Valid maps contributing to the aggregate", "纳入聚合的有效图数量"),
        "saliency_mean": ("Mean saliency-mask faithfulness value", "显著区域遮罩faithfulness均值"),
        "saliency_sd": ("Standard deviation of saliency-mask faithfulness", "显著区域遮罩faithfulness标准差"),
        "saliency_median": ("Median saliency-mask faithfulness value", "显著区域遮罩faithfulness中位数"),
        "saliency_minus_random_mean": ("Mean paired saliency value minus matched-random mean", "显著区域值减匹配随机均值的配对均值"),
        "saliency_greater_than_random_rate": ("Fraction where saliency exceeds its matched-random mean", "显著区域值高于其匹配随机均值的比例"),
        "phase": ("Baseline-v2 execution phase", "Baseline-v2执行阶段"),
        "job_id": ("Katana PBS job identifier", "Katana PBS任务标识"),
        "queue": ("Katana PBS queue", "Katana PBS队列"),
        "gpu_model": ("Allocated GPU model", "分配的GPU型号"),
        "ncpus": ("Allocated CPU-core count", "分配的CPU核心数"),
        "memory_gb": ("Requested memory in GB", "申请内存（GB）"),
        "walltime": ("Requested PBS walltime", "申请的PBS walltime"),
        "run_count": ("PBS execution count", "PBS执行次数"),
        "exit_status": ("Recorded scheduler exit status", "记录的调度器退出状态"),
        "scientific_status": ("Scientific artifact/verifier status independent of scheduler status", "独立于调度器状态的科学产物/验证器状态"),
        "config_sha256": ("Frozen execution-config SHA-256", "冻结执行配置SHA-256"),
        "split_sha256": ("Frozen fold split SHA-256", "冻结折划分SHA-256"),
        "encoder_sha256": ("P4 encoder-initialization SHA-256", "P4编码器初始化SHA-256"),
        "checkpoint_sha256": ("Best-checkpoint SHA-256, composite where applicable", "最佳checkpoint SHA-256（适用时为组合哈希）"),
        "test_transaction_count": ("Valid committed test-evaluation transaction count", "有效已提交测试评估事务计数"),
        "verifier_or_recovery_evidence": ("Scheduler/verifier recovery interpretation", "调度器/验证器恢复说明"),
    }
    return [
        {"field": field, "definition": definitions[field][1 if language == "zh" else 0]}
        for field in definitions
    ]


def _terminology_rows() -> list[dict[str, str]]:
    return [
        {
            "canonical_term": "Black-box",
            "english": "Black-box",
            "chinese": "Black-box",
            "translation_policy": "Preserve registered model name",
        },
        {
            "canonical_term": "Standard CBM",
            "english": "Standard CBM",
            "chinese": "Standard CBM",
            "translation_policy": "Preserve registered model name",
        },
        {
            "canonical_term": "Mixed-type CEM",
            "english": "Mixed-type CEM",
            "chinese": "Mixed-type CEM",
            "translation_policy": "Preserve project-specific model name",
        },
        {
            "canonical_term": "Learned-softmax GAM",
            "english": "Learned-softmax GAM",
            "chinese": "Learned-softmax GAM",
            "translation_policy": "Preserve preregistered model name",
        },
        {
            "canonical_term": "output_sensitivity",
            "english": "output sensitivity",
            "chinese": "输出敏感度",
            "translation_policy": "Keep variable name in code and tables",
        },
        {
            "canonical_term": "error_increase",
            "english": "prediction error increase",
            "chinese": "预测误差增量",
            "translation_policy": "Keep variable name in code and tables",
        },
        {
            "canonical_term": "undefined Grad-CAM map",
            "english": "undefined post-ReLU all-zero Grad-CAM map",
            "chinese": "ReLU后全零的未定义Grad-CAM图",
            "translation_policy": "Do not infer an unobserved gradient mechanism",
        },
        {
            "canonical_term": "radiologist assessment",
            "english": "radiologist assessment",
            "chinese": "放射科医师评估",
            "translation_policy": "Never describe as pathology-confirmed diagnosis",
        },
    ]


def build_public_outputs(
    *,
    public_root: Path = PUBLIC_ROOT_DEFAULT,
    audit_root: Path = P9_AUDIT_ROOT_DEFAULT,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    data = build_report_data(audit_root=audit_root, repository_root=repository_root)
    public_root.mkdir(parents=True, exist_ok=True)
    data_path = public_root / "report_data.json"
    data_text = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    data_path.write_text(data_text, encoding="utf-8")
    data_sha = sha256_file(data_path)
    table_paths = export_tables(data, public_root / "tables")
    for language in ("en", "zh"):
        dictionary = public_root / "tables" / f"data_dictionary_{language}.csv"
        rows = data["data_dictionary"][language]
        _write_csv(dictionary, ("field", "definition"), rows)
        table_paths.append(dictionary)
    terminology = public_root / "tables" / "bilingual_terminology.csv"
    terminology_rows = data["terminology"]
    _write_csv(terminology, tuple(terminology_rows[0]), terminology_rows)
    table_paths.append(terminology)
    figure_paths = []
    for language in ("en", "zh"):
        figure_paths.extend(build_figures(data, public_root / "figures", language))
    report_paths: list[Path] = []
    for variant in ("short", "technical"):
        markdown: dict[str, str] = {}
        for language in ("en", "zh"):
            value = build_markdown(data, variant, language).replace(
                "PENDING_RENDER_HASH", data_sha
            )
            markdown[language] = value
            path = public_root / f"{variant}_{language}.md"
            path.write_text(value, encoding="utf-8")
            report_paths.append(path)
        verify_bilingual_markdown(markdown["en"], markdown["zh"], variant)
        for language in ("en", "zh"):
            path = public_root / f"{variant}_{language}.pdf"
            render_pdf(
                data,
                variant=variant,
                language=language,
                destination=path,
                figure_root=public_root / "figures",
                report_data_sha256=data_sha,
            )
            report_paths.append(path)
    visual_qa_path = public_root / "visual_qa.json"
    record_visual_qa(
        public_root, manual_review_pass=False, refresh_source_index=False
    )
    source_rows = []
    for path in sorted(
        [data_path, *table_paths, *figure_paths, *report_paths, visual_qa_path]
    ):
        relative = path.relative_to(public_root).as_posix()
        source_rows.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(path),
                "source_json": "report_data.json",
                "source_json_sha256": data_sha,
                "source_field_path": _source_field_path(relative),
            }
        )
    source_path = public_root / "table_figure_sources.csv"
    _write_csv(source_path, tuple(source_rows[0]), source_rows)
    assert_public_payload(
        {path.relative_to(public_root).as_posix(): path.read_text(encoding="utf-8", errors="ignore") for path in public_root.rglob("*") if path.is_file() and path.suffix in {".json", ".csv", ".md", ".svg"}}
    )
    return {
        "status": "PASS",
        "report_data_sha256": data_sha,
        "reports": [path.as_posix() for path in report_paths],
        "tables": [path.as_posix() for path in table_paths],
        "figures": [path.as_posix() for path in figure_paths],
        "source_index": source_path.as_posix(),
    }


def _source_field_path(relative_path: str) -> str:
    """Map every public table/figure to its exact shared-data source fields."""
    figure_fields = {
        "figure_01_cohort_flow": "cohort",
        "figure_02_primary_performance": "task.models;bootstrap.models",
        "figure_03_paired_mae": "bootstrap.paired_mae_A_minus_B",
        "figure_04_secondary": "task.models.*.pooled_secondary;bootstrap.models",
        "figure_05_concept_metrics": "concept",
        "figure_06_intervention_curves": "intervention",
        "figure_07_centered_contributions": "contribution_centering",
        "figure_08_learned_alpha": "learned_alpha",
        "figure_09_gradcam_undefined": "gradcam_accounting;spatial.models",
        "figure_10_spatial_faithfulness": "spatial.models.*.pooled_all_targets",
    }
    table_fields = {
        "primary_secondary_metrics.csv": "task.models;bootstrap.models",
        "paired_comparisons.csv": "bootstrap.paired_mae_A_minus_B;bootstrap.paired_auroc_B_minus_A",
        "concept_metrics.csv": "concept",
        "intervention_curves.csv": "intervention",
        "centered_contributions.csv": "contribution_centering",
        "learned_gam_alpha.csv": "learned_alpha",
        "gradcam_accounting.csv": "gradcam_accounting;spatial.models.*.folds.*.targets",
        "spatial_faithfulness.csv": "spatial.models.*.pooled_targets;spatial.models.*.pooled_all_targets",
        "execution_registry.csv": "execution_registry",
        "data_dictionary_en.csv": "data_dictionary.en",
        "data_dictionary_zh.csv": "data_dictionary.zh",
        "bilingual_terminology.csv": "terminology",
    }
    name = Path(relative_path).name
    if relative_path.startswith("figures/"):
        stem = re.sub(r"_(en|zh)$", "", Path(name).stem)
        if stem not in figure_fields:
            raise ValueError(f"P10_FIGURE_SOURCE_FIELD_UNKNOWN:{name}")
        return figure_fields[stem]
    if relative_path.startswith("tables/"):
        if name not in table_fields:
            raise ValueError(f"P10_TABLE_SOURCE_FIELD_UNKNOWN:{name}")
        return table_fields[name]
    return "$"


def _pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED") from error
    return len(PdfReader(str(path)).pages)


def _resolve_pdftoppm() -> str:
    candidates = (
        os.environ.get("P10_PDFTOPPM"),
        shutil.which("pdftoppm"),
        "/Users/katherine/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("P10_PDFTOPPM_REQUIRED")


def _rendered_pdf_page_rows(path: Path, renderer: str) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED") from error
    with tempfile.TemporaryDirectory(prefix="p10-pdf-render-") as directory:
        prefix = Path(directory) / "page"
        subprocess.run(
            [renderer, "-png", "-r", "80", str(path), str(prefix)],
            check=True,
            capture_output=True,
        )
        images = sorted(
            Path(directory).glob("page-*.png"),
            key=lambda item: int(item.stem.rsplit("-", 1)[1]),
        )
        rows = []
        for page_index, image_path in enumerate(images, start=1):
            with Image.open(image_path) as image:
                gray = image.convert("L")
                minimum, maximum = gray.getextrema()
                rows.append(
                    {
                        "page": page_index,
                        "width": int(image.width),
                        "height": int(image.height),
                        "png_sha256": sha256_file(image_path),
                        "nonblank": bool(minimum != maximum),
                    }
                )
        return rows


def _pdfplumber_page_text_gate(path: Path, expected_pages: int) -> None:
    try:
        import pdfplumber
    except ImportError:
        bundled = Path(
            os.environ.get(
                "P10_PDFPLUMBER_PYTHON",
                str(
                    Path.home()
                    / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
                ),
            )
        )
        if not bundled.is_file():
            raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED")
        script = (
            "import pdfplumber,sys; p=sys.argv[1]; n=int(sys.argv[2]); "
            "d=pdfplumber.open(p); "
            "assert len(d.pages)==n; "
            "assert all((x.extract_text() or '').strip() and '□' not in (x.extract_text() or '') "
            "and '\\ufffd' not in (x.extract_text() or '') for x in d.pages); d.close()"
        )
        completed = subprocess.run(
            [str(bundled), "-c", script, str(path), str(expected_pages)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError(f"P10_PDFPLUMBER_TEXT_INVALID:{path.name}")
        return
    with pdfplumber.open(path) as document:
        if len(document.pages) != expected_pages:
            raise ValueError(f"P10_PDFPLUMBER_PAGE_COUNT_INVALID:{path.name}")
        for page_index, page in enumerate(document.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip() or "□" in text or "\ufffd" in text:
                raise ValueError(
                    f"P10_PDFPLUMBER_TEXT_INVALID:{path.name}:{page_index}"
                )


def _refresh_visual_qa_source_index(public_root: Path, evidence_path: Path) -> None:
    source_index = public_root / "table_figure_sources.csv"
    if not source_index.is_file():
        return
    with source_index.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    relative = evidence_path.relative_to(public_root).as_posix()
    matching = [row for row in rows if row["relative_path"] == relative]
    if len(matching) != 1:
        raise ValueError("P10_VISUAL_QA_SOURCE_INDEX_MISSING")
    matching[0]["sha256"] = sha256_file(evidence_path)
    matching[0]["source_json_sha256"] = sha256_file(public_root / "report_data.json")
    _write_csv(source_index, tuple(rows[0]), rows)


def record_visual_qa(
    public_root: Path = PUBLIC_ROOT_DEFAULT,
    *,
    manual_review_pass: bool,
    refresh_source_index: bool = True,
) -> dict[str, Any]:
    renderer = _resolve_pdftoppm()
    version = subprocess.run(
        [renderer, "-v"], check=True, capture_output=True, text=True
    )
    version_text = (version.stderr or version.stdout).splitlines()[0].strip()
    pdfs: dict[str, Any] = {}
    total_pages = 0
    for name in ("short_en", "short_zh", "technical_en", "technical_zh"):
        path = public_root / f"{name}.pdf"
        page_count = _pdf_page_count(path)
        page_rows = _rendered_pdf_page_rows(path, renderer)
        if len(page_rows) != page_count or not all(row["nonblank"] for row in page_rows):
            raise ValueError(f"P10_PDF_RENDER_COVERAGE_INVALID:{name}")
        _pdfplumber_page_text_gate(path, page_count)
        total_pages += page_count
        pdfs[name] = {
            "pdf_sha256": sha256_file(path),
            "page_count": page_count,
            "rendered_pages": page_rows,
        }
    if total_pages != 88:
        raise ValueError(f"P10_PDF_RENDER_TOTAL_INVALID:{total_pages}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if manual_review_pass else "PENDING_MANUAL_REVIEW",
        "renderer": "pdftoppm",
        "renderer_version": version_text,
        "render_dpi": 80,
        "rendered_page_count": total_pages,
        "pdfplumber_text_gate": "PASS",
        "manual_visual_review": "PASS" if manual_review_pass else "PENDING",
        "manual_checklist": {
            "clipping": "PASS" if manual_review_pass else "PENDING",
            "overlap": "PASS" if manual_review_pass else "PENDING",
            "fonts_and_missing_glyphs": "PASS" if manual_review_pass else "PENDING",
            "legends_and_tables": "PASS" if manual_review_pass else "PENDING",
        },
        "pdfs": pdfs,
    }
    evidence_path = public_root / "visual_qa.json"
    evidence_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if refresh_source_index:
        _refresh_visual_qa_source_index(public_root, evidence_path)
    return payload


def _verify_visual_qa(public_root: Path) -> dict[str, Any]:
    evidence = _read_json(public_root / "visual_qa.json")
    if (
        evidence.get("status") != "PASS"
        or evidence.get("manual_visual_review") != "PASS"
        or evidence.get("pdfplumber_text_gate") != "PASS"
        or evidence.get("rendered_page_count") != 88
        or any(value != "PASS" for value in evidence.get("manual_checklist", {}).values())
    ):
        raise ValueError("P10_VISUAL_QA_NOT_APPROVED")
    renderer = _resolve_pdftoppm()
    expected_names = {"short_en", "short_zh", "technical_en", "technical_zh"}
    if set(evidence.get("pdfs", {})) != expected_names:
        raise ValueError("P10_VISUAL_QA_PDF_SET_INVALID")
    for name in sorted(expected_names):
        path = public_root / f"{name}.pdf"
        expected = evidence["pdfs"][name]
        page_count = _pdf_page_count(path)
        _pdfplumber_page_text_gate(path, page_count)
        if (
            expected.get("pdf_sha256") != sha256_file(path)
            or expected.get("page_count") != page_count
            or expected.get("rendered_pages") != _rendered_pdf_page_rows(path, renderer)
        ):
            raise ValueError(f"P10_VISUAL_QA_BINDING_INVALID:{name}")
    return evidence


def _pdf_text_and_font_evidence(path: Path) -> tuple[str, list[dict[str, Any]]]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED") from error
    reader = PdfReader(str(path))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    fonts: dict[str, dict[str, Any]] = {}
    for page in reader.pages:
        resources = page.get("/Resources", {}).get_object()
        content = page.get_contents().get_data()
        for key, reference in resources.get("/Font", {}).items():
            token = f"{key} ".encode("ascii")
            if token not in content:
                continue
            font = reference.get_object()
            descriptor_reference = font.get("/FontDescriptor")
            descriptor = (
                descriptor_reference.get_object() if descriptor_reference else {}
            )
            base_name = str(font.get("/BaseFont", ""))
            fonts[base_name] = {
                "base_font": base_name,
                "embedded": any(
                    name in descriptor for name in ("/FontFile", "/FontFile2", "/FontFile3")
                ),
                "to_unicode": "/ToUnicode" in font,
            }
    return text, sorted(fonts.values(), key=lambda row: row["base_font"])


def _verify_chinese_pdf_fonts(path: Path) -> None:
    text, fonts = _pdf_text_and_font_evidence(path)
    if not fonts or "□" in text or "\ufffd" in text:
        raise ValueError(f"P10_CHINESE_PDF_TEXT_OR_FONT_INVALID:{path.name}")
    for font in fonts:
        if (
            "STSongti-SC" not in font["base_font"]
            or font["embedded"] is not True
            or font["to_unicode"] is not True
        ):
            raise ValueError(f"P10_CHINESE_FONT_NOT_EMBEDDED:{path.name}:{font}")


def _verify_source_index(public_root: Path) -> None:
    source_index = public_root / "table_figure_sources.csv"
    with source_index.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 62:
        raise ValueError(f"P10_PUBLIC_SOURCE_INDEX_CARDINALITY_INVALID:{len(rows)}")
    indexed = set()
    report_data_sha256 = sha256_file(public_root / "report_data.json")
    for row in rows:
        relative = row.get("relative_path", "")
        if not relative or relative in indexed:
            raise ValueError("P10_PUBLIC_SOURCE_INDEX_DUPLICATE_OR_EMPTY")
        indexed.add(relative)
        path = public_root / relative
        if (
            not path.is_file()
            or sha256_file(path) != row.get("sha256")
            or row.get("source_json") != "report_data.json"
            or row.get("source_json_sha256") != report_data_sha256
            or not row.get("source_field_path")
            or row.get("source_field_path") == "multiple; see data dictionaries"
        ):
            raise ValueError(f"P10_PUBLIC_SOURCE_INDEX_BINDING_INVALID:{relative}")
    actual = {
        path.relative_to(public_root).as_posix()
        for path in public_root.rglob("*")
        if path.is_file() and path != source_index
    }
    if actual != indexed:
        raise ValueError("P10_PUBLIC_OUTPUT_EXACT_INVENTORY_INVALID")


def verify_public_outputs(public_root: Path = PUBLIC_ROOT_DEFAULT) -> dict[str, Any]:
    data_path = public_root / "report_data.json"
    data = _read_json(data_path)
    assert_public_payload(data)
    pdf_text: dict[tuple[str, str], str] = {}
    for variant, bounds in (("short", (8, 12)), ("technical", (25, 35))):
        en = (public_root / f"{variant}_en.md").read_text(encoding="utf-8")
        zh = (public_root / f"{variant}_zh.md").read_text(encoding="utf-8")
        verify_bilingual_markdown(en, zh, variant)
        for language in ("en", "zh"):
            pdf = public_root / f"{variant}_{language}.pdf"
            pages = _pdf_page_count(pdf)
            if not bounds[0] <= pages <= bounds[1]:
                raise ValueError(f"P10_PDF_PAGE_COUNT_INVALID:{variant}:{language}:{pages}")
            text, _ = _pdf_text_and_font_evidence(pdf)
            assert_public_payload(text)
            if not text.strip() or "REPORT-DATA-SHA256" not in text:
                raise ValueError(f"P10_PDF_TEXT_EXTRACTION_INVALID:{variant}:{language}")
            pdf_text[(variant, language)] = text
            if language == "zh":
                _verify_chinese_pdf_fonts(pdf)
        if extract_numeric_tokens(pdf_text[(variant, "en")]) != extract_numeric_tokens(
            pdf_text[(variant, "zh")]
        ):
            raise ValueError(f"P10_BILINGUAL_PDF_NUMERIC_TOKEN_MISMATCH:{variant}")
        for code in data["scientific_conclusion_codes"]:
            if (
                pdf_text[(variant, "en")].count(code) != 1
                or pdf_text[(variant, "zh")].count(code) != 1
            ):
                raise ValueError(
                    f"P10_BILINGUAL_PDF_CONCLUSION_CODE_MISMATCH:{variant}:{code}"
                )
    if len(list((public_root / "figures").glob("*.png"))) != 20:
        raise ValueError("P10_PUBLIC_PNG_COUNT_INVALID")
    if len(list((public_root / "figures").glob("*.svg"))) != 20:
        raise ValueError("P10_PUBLIC_SVG_COUNT_INVALID")
    if len(list((public_root / "tables").glob("*.csv"))) < 10:
        raise ValueError("P10_PUBLIC_TABLE_COUNT_INVALID")
    with (public_root / "tables" / "data_dictionary_en.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        dictionary_en = list(csv.DictReader(handle))
    with (public_root / "tables" / "data_dictionary_zh.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        dictionary_zh = list(csv.DictReader(handle))
    if [row["field"] for row in dictionary_en] != [
        row["field"] for row in dictionary_zh
    ]:
        raise ValueError("P10_BILINGUAL_DATA_DICTIONARY_FIELD_MISMATCH")
    _verify_source_index(public_root)
    visual_qa = _verify_visual_qa(public_root)
    return {
        "status": "PASS",
        "report_data_sha256": sha256_file(data_path),
        "short_pages": _pdf_page_count(public_root / "short_en.pdf"),
        "technical_pages": _pdf_page_count(public_root / "technical_en.pdf"),
        "numeric_language_parity": True,
        "pdf_numeric_language_parity": True,
        "chinese_fonts_embedded": True,
        "public_privacy": "PASS",
        "page_render_visual_qa": visual_qa["status"],
        "pdfplumber_text_gate": visual_qa["pdfplumber_text_gate"],
        "rendered_page_count": visual_qa["rendered_page_count"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_inputs_parser = subparsers.add_parser("verify-inputs")
    verify_inputs_parser.add_argument("--audit-root", type=Path, default=P9_AUDIT_ROOT_DEFAULT)
    build = subparsers.add_parser("build")
    build.add_argument("--variant", choices=("short", "technical"), required=False)
    build.add_argument("--language", choices=("en", "zh"), required=False)
    build.add_argument("--output-root", type=Path, default=PUBLIC_ROOT_DEFAULT)
    appendix = subparsers.add_parser("build-private-appendix")
    appendix.add_argument("--language", choices=("en", "zh"), required=True)
    appendix.add_argument("--archive-root", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--scope", choices=("all",), default="all")
    verify.add_argument("--output-root", type=Path, default=PUBLIC_ROOT_DEFAULT)
    visual = subparsers.add_parser("record-visual-qa")
    visual.add_argument("--output-root", type=Path, default=PUBLIC_ROOT_DEFAULT)
    visual.add_argument("--manual-review-pass", action="store_true")
    private_visual = subparsers.add_parser("record-private-visual-qa")
    private_visual.add_argument("--archive-root", type=Path)
    private_visual.add_argument("--manual-review-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify-inputs":
        result = verify_inputs(audit_root=args.audit_root)
    elif args.command == "build":
        # The authoritative data layer is shared; building once emits all four reports.
        result = build_public_outputs(public_root=args.output_root)
    elif args.command == "build-private-appendix":
        from lidc_baseline.p10_private_appendix import build_private_appendix

        result = build_private_appendix(language=args.language, archive_root=args.archive_root)
    elif args.command == "record-visual-qa":
        result = record_visual_qa(
            args.output_root, manual_review_pass=args.manual_review_pass
        )
    elif args.command == "record-private-visual-qa":
        from lidc_baseline.p10_archive import LOCAL_ROOT_DEFAULT
        from lidc_baseline.p10_private_appendix import record_private_visual_qa

        result = record_private_visual_qa(
            args.archive_root or LOCAL_ROOT_DEFAULT,
            manual_review_pass=args.manual_review_pass,
        )
    else:
        result = verify_public_outputs(args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
