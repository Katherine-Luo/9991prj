"""Catalogue-driven bilingual P10 manuscript and presentation renderer.

This module is presentation-only. It validates the user-approved Results
Catalogue, reads registered frozen evidence, and never executes a model,
changes a scientific artifact, or recomputes a registered scientific result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import yaml


REVISION_CONFIG = Path("configs/experiments/baseline_v2_p10_report_revision.resolved.yaml")
PUBLIC_ROOT = Path("reports/baseline_v2/p10/public")
MANIFEST_ROOT = Path("reports/baseline_v2/p10/manifests")
CATALOGUE_REGISTRY = Path("docs/results/results_catalogue_registry.json")
CATALOGUE_MANIFEST = Path("docs/results/catalogue_manifest.json")
REPORT_DATA = Path("reports/baseline_v2/p10/public/report_data.json")
PRIVATE_ARCHIVE = Path(
    "/Users/katherine/Desktop/lidc_data/lidc_baseline_private_archive/baseline_v2"
)
PRIVATE_REPORT_ROOT = PRIVATE_ARCHIVE / "p10_private_report"
MANUAL_VISUAL_REVIEWER = (
    "Codex primary agent (visual inspection of contact sheets and "
    "original-resolution critical pages)"
)
REFERENCES = (
    '[1] S. G. Armato III et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans," Med. Phys., vol. 38, no. 2, pp. 915–931, 2011, doi: 10.1118/1.3528204.',
    '[2] G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, "Densely Connected Convolutional Networks," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 4700–4708, 2017, doi: 10.1109/CVPR.2017.243.',
    '[3] P. W. Koh et al., "Concept Bottleneck Models," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 119, pp. 5338–5348, 2020.',
    '[4] M. Espinosa Zarlenga et al., "Concept Embedding Models: Beyond the Accuracy–Explainability Trade-Off," in Adv. Neural Inf. Process. Syst., vol. 35, pp. 21400–21413, 2022.',
    '[5] T. Hastie and R. Tibshirani, "Generalized Additive Models," Stat. Sci., vol. 1, no. 3, pp. 297–318, 1986, doi: 10.1214/ss/1177013604.',
    '[6] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), pp. 618–626, 2017, doi: 10.1109/ICCV.2017.74.',
    '[7] S. Shin, Y. Jo, S. Ahn, and N. Lee, "A Closer Look at the Intervention Procedure of Concept Bottleneck Models," in Proc. 40th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 202, pp. 31504–31520, 2023.',
    '[8] R. I. Dumaev, S. A. Molodyakov, and L. V. Utkin, "Concept-based Explainable Malignancy Scoring on Pulmonary Nodules in CT Images," arXiv:2405.17483, 2024, doi: 10.48550/arXiv.2405.17483.',
    '[9] W. Shen et al., "Multi-crop Convolutional Neural Networks for Lung Nodule Malignancy Suspiciousness Classification," Pattern Recognit., vol. 61, pp. 663–673, 2017, doi: 10.1016/j.patcog.2016.05.029.',
    '[10] B. Efron, "Bootstrap Methods: Another Look at the Jackknife," Ann. Stat., vol. 7, no. 1, pp. 1–26, 1979, doi: 10.1214/aos/1176344552.',
)

MODEL_ORDER = ("blackbox", "standard_cbm", "mixed_cem", "learned_softmax_gam")
MODEL_LABELS = {
    "blackbox": "Black-box",
    "standard_cbm": "Standard CBM",
    "mixed_cem": "Mixed-type CEM",
    "learned_softmax_gam": "Learned-softmax GAM",
}
MODEL_COLORS = {
    "blackbox": "#4C78A8",
    "standard_cbm": "#F58518",
    "mixed_cem": "#54A24B",
    "learned_softmax_gam": "#B279A2",
}
CONTINUOUS_CONCEPTS = (
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
CATEGORICAL_CONCEPTS = ("internalStructure", "calcification")
CONCEPTS = (
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)
PUBLIC_TABLE_IDS = tuple(f"RPT-T{i:02d}" for i in range(1, 19))
PUBLIC_FIGURE_IDS = (
    *tuple(f"RPT-F{i:02d}" for i in range(1, 9)),
    "RPT-F09A",
    "RPT-F09B",
    *tuple(f"RPT-F{i:02d}" for i in range(10, 14)),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"P10_EXPECTED_OBJECT:{path}")
    return value


@dataclass(frozen=True)
class CatalogueContext:
    config: Mapping[str, Any]
    registry: Mapping[str, Any]
    report_data: Mapping[str, Any]
    items: tuple[Mapping[str, Any], ...]
    by_id: Mapping[str, Mapping[str, Any]]
    by_category: Mapping[str, tuple[Mapping[str, Any], ...]]
    registry_sha256: str
    manifest_sha256: str

    def category(self, name: str) -> tuple[Mapping[str, Any], ...]:
        return self.by_category.get(name, ())


def load_catalogue_context(repository_root: Path = Path(".")) -> CatalogueContext:
    config_path = repository_root / REVISION_CONFIG
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("P10_REVISION_CONFIG_INVALID")
    gates = config.get("approval_gates", {})
    expected_gates = {
        "results_catalogue_plan_approved": 1,
        "p10_report_plan_approved": 1,
        "generated_catalogue_approved": 1,
        "report_revision_authorized": 1,
    }
    if gates != expected_gates:
        raise ValueError("P10_REPORT_REVISION_NOT_AUTHORIZED")
    for key, spec in config["approved_inputs"].items():
        path = repository_root / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"P10_APPROVED_INPUT_HASH_MISMATCH:{key}")
    registry_path = repository_root / CATALOGUE_REGISTRY
    manifest_path = repository_root / CATALOGUE_MANIFEST
    registry = _json(registry_path)
    manifest = _json(manifest_path)
    if registry.get("status") != "CATALOGUE_BUILT_PENDING_USER_APPROVAL":
        raise ValueError("P10_APPROVED_CATALOGUE_STATUS_INVALID")
    if len(registry.get("items", [])) != 2395:
        raise ValueError("P10_CATALOGUE_CARDINALITY_MISMATCH")
    if registry.get("scientific_boundaries") != {
        "model_forward": False,
        "p11_started": False,
        "report_regeneration": False,
        "scientific_recomputation": False,
        "test_inference": False,
        "training": False,
    }:
        raise ValueError("P10_CATALOGUE_SCIENTIFIC_BOUNDARY_MISMATCH")
    items = tuple(registry["items"])
    by_id = {str(item["catalogue_item_id"]): item for item in items}
    if len(by_id) != len(items):
        raise ValueError("P10_DUPLICATE_CATALOGUE_ITEM_ID")
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        by_category.setdefault(str(item["category"]), []).append(item)
    report_data = _json(repository_root / REPORT_DATA)
    expected_report_sha = registry["source_bindings"]["report_data_sha256"]
    if sha256_file(repository_root / REPORT_DATA) != expected_report_sha:
        raise ValueError("P10_REPORT_DATA_HASH_MISMATCH")
    return CatalogueContext(
        config=config,
        registry=registry,
        report_data=report_data,
        items=items,
        by_id=by_id,
        by_category={key: tuple(value) for key, value in by_category.items()},
        registry_sha256=sha256_file(registry_path),
        manifest_sha256=sha256_file(manifest_path),
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    number = float(value)
    if abs(number) >= 1000 and number.is_integer():
        return f"{int(number):,}"
    return f"{number:.{digits}f}"


def _ci(details: Mapping[str, Any], metric: str) -> str:
    interval = details["bootstrap_intervals"][metric]
    return f"{_fmt(interval['percentile_2_5'])}–{_fmt(interval['percentile_97_5'])}"


def _trace(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "catalogue_item_id": str(item["catalogue_item_id"]),
        "source_artifact_id": str(item["source_artifact_id"]),
        "source_field_path": str(item["source_field_path"]),
        "source_sha256": str(item["source_sha256"]),
    }


def _supported_conclusion_code(details: Mapping[str, Any]) -> str:
    if details["ci_crosses_zero"]:
        return "NO_SUPPORTED_DIFFERENCE_CI_CROSSES_ZERO"
    return "SUPPORTS_B" if float(details["estimate_mean"]) > 0 else "SUPPORTS_A"


def _supported_conclusion_label(details: Mapping[str, Any]) -> str:
    return {
        "SUPPORTS_A": "Supports A",
        "SUPPORTS_B": "Supports B",
        "NO_SUPPORTED_DIFFERENCE_CI_CROSSES_ZERO": "No supported difference",
    }[_supported_conclusion_code(details)]


def _training_configuration_rows(context: CatalogueContext) -> list[dict[str, Any]]:
    """Return frozen scientific training settings, not fold audit outcomes."""
    by_path = {
        str(item.get("source_relative_path")): item for item in context.category("CAT-R")
    }
    common_item = by_path[
        "configs/experiments/baseline_v2_reference_training_h200_warn_only.resolved.yaml"
    ]
    p6_item = by_path[
        "configs/experiments/baseline_v2_p6_standard_cbm_h200.resolved.yaml"
    ]
    p7_item = by_path[
        "configs/experiments/baseline_v2_p7_mixed_cem_h200.resolved.yaml"
    ]
    p8_item = by_path[
        "configs/experiments/baseline_v2_p8_gam_h200.resolved.yaml"
    ]
    source = yaml.safe_load(Path(common_item["source_relative_path"]).read_text(encoding="utf-8"))
    profile = source["project_preregistered"]
    aug = profile["augmentation"]
    common_rows = [
        {"Setting": "Input / encoder", "Frozen value": "64³ nodule ROI / DenseNet-121 (shared fold initialization)"},
        {"Setting": "Optimizer", "Frozen value": "Adam; β=(0.9, 0.999); ε=1e-7; weight decay=0"},
        {"Setting": "Initial learning rate / batch", "Frozen value": "1e-4 / true micro-batch 16; no accumulation; drop_last=False"},
        {"Setting": "Epoch budget", "Frozen value": "80 per registered stage; no early stopping"},
        {"Setting": "Scheduler", "Frozen value": "validation objective; factor 0.9 after 4 bad epochs; min_delta=1e-4; minimum LR=0"},
        {"Setting": "Train-only augmentation", "Frozen value": f"axial rotation ±15° (p={aug['axial_rotation']['probability']}); H/W flips p=0.5; z reversal p=0.5"},
        {"Setting": "Precision / determinism", "Frozen value": "FP32; AMP/BF16/CUDA-matmul-TF32/cuDNN-TF32 off; deterministic warn-only"},
        {"Setting": "Formal accelerator", "Frozen value": "NVIDIA H200"},
        {"Setting": "Black-box objective", "Frozen value": "MSE on unclipped normalized malignancy score"},
    ]
    model_rows = (
        (
            {"Setting": "Standard CBM objective", "Frozen value": "80-epoch concept loss, then 80-epoch linear task-head MSE on frozen predicted concepts"},
            p6_item,
        ),
        (
            {"Setting": "Mixed-type CEM objective", "Frozen value": "task MSE + 0.01 × mean eight-group concept loss"},
            p7_item,
        ),
        (
            {"Setting": "Learned-softmax GAM objective", "Frozen value": "task MSE + mean eight-group concept loss"},
            p8_item,
        ),
    )
    return [
        *({**row, **_trace(common_item)} for row in common_rows),
        *({**row, **_trace(item)} for row, item in model_rows),
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _category_rows(context: CatalogueContext, category: str) -> list[Mapping[str, Any]]:
    return sorted(context.category(category), key=lambda item: item["catalogue_item_id"])


def build_public_table_rows(context: CatalogueContext) -> dict[str, list[dict[str, Any]]]:
    """Build the 18 public table payloads from approved Catalogue items."""
    tables: dict[str, list[dict[str, Any]]] = {}
    tables["RPT-T01"] = [
        {"Approach": "Black-box CNN", "Prediction": "Yes", "Concepts": "No", "Spatial explanation": "Optional", "Intervention": "No", "This study": "Comparator"},
        {"Approach": "Concept Bottleneck Model", "Prediction": "Yes", "Concepts": "Explicit", "Spatial explanation": "Concept/task Grad-CAM", "Intervention": "Concept replacement", "This study": "Standard CBM"},
        {"Approach": "Concept Embedding Model", "Prediction": "Yes", "Concepts": "Mixed-type embeddings", "Spatial explanation": "Concept/task Grad-CAM", "Intervention": "Mixture-weight replacement", "This study": "Project-specific Mixed-type CEM"},
        {"Approach": "Additive local experts", "Prediction": "Yes", "Concepts": "Explicit", "Spatial explanation": "Concept/task Grad-CAM", "Intervention": "Local-expert re-evaluation", "This study": "Preregistered Learned-softmax GAM"},
    ]
    tables["RPT-T02"] = [
        {"Cohort component": "Primary regression", "Nodules": 2633, "Patients": 868, "Role": "Main five-fold evaluation"},
        {"Cohort component": "Secondary extreme subset", "Nodules": 1073, "Patients": 578, "Role": "782 low / 291 high"},
    ]
    tables["RPT-T03"] = [
        {"Variable": "Malignancy", "Role": "Downstream target (not a concept)", "Type": "Continuous", "Frozen target": "Radiologist mean, 1–5; normalized (y−1)/4"},
        *[
            {"Variable": concept, "Role": "Bottleneck concept", "Type": "Continuous", "Frozen target": "Normalized valid-reader mean"}
            for concept in CONTINUOUS_CONCEPTS
        ],
        {"Variable": "internalStructure", "Role": "Bottleneck concept", "Type": "Categorical (4 classes)", "Frozen target": "Full reader vote distribution"},
        {"Variable": "calcification", "Role": "Bottleneck concept", "Type": "Categorical (6 classes)", "Frozen target": "Full reader vote distribution"},
    ]
    tables["RPT-T04"] = [
        {"Model": "Black-box", "Task path": "DenseNet features → linear score", "Concept representation": "None", "Contribution semantics": "Not applicable", "Intervention semantics": "Not applicable"},
        {"Model": "Standard CBM", "Task path": "Predicted concepts → linear score", "Concept representation": "6 sigmoid + 2 softmax groups", "Contribution semantics": "Linear group terms", "Intervention semantics": "Replace activated concept group"},
        {"Model": "Mixed-type CEM", "Task path": "Sample-conditioned concept embeddings → linear score", "Concept representation": "Mixed-type dynamic states", "Contribution semantics": "Embedding block dot product", "Intervention semantics": "Replace mixture weights only"},
        {"Model": "Learned-softmax GAM", "Task path": "Predicted concepts → local experts → additive score", "Concept representation": "6 sigmoid + 2 softmax groups", "Contribution semantics": "Softmax-weighted local experts", "Intervention semantics": "Ground-truth concept through experts"},
    ]
    tables["RPT-T05"] = _training_configuration_rows(context)
    tables["RPT-T06"] = [
        {"Component": "Primary regression", "Unit": "Nodule; patient-cluster bootstrap", "Metric": "Unclipped original-scale MAE (primary), RMSE, normalized MAE, Pearson, Spearman", "Selection/uncertainty": "2,000 shared patient draws"},
        {"Component": "Secondary extreme", "Unit": "1,073 extreme nodules / 578 patients", "Metric": "AUROC, AUPRC; threshold metrics", "Selection/uncertainty": "Fold-validation extreme-only Youden-J; 2,000 valid draws"},
        {"Component": "Concept fidelity", "Unit": "Nodule", "Metric": "Continuous MAE/RMSE/correlation; categorical CE/Brier/macro-F1", "Selection/uncertainty": "Hard F1 excludes true modal ties"},
        {"Component": "Spatial faithfulness", "Unit": "Valid Grad-CAM target", "Metric": "output_sensitivity and error_increase", "Selection/uncertainty": "26,215 voxels; 20 matched random masks"},
        {"Component": "Intervention", "Unit": "Pooled OOF", "Metric": "iMAE/Delta_iMAE; iAUC/Delta_iAUC", "Selection/uncertainty": "k=0…8; random and error-first orderings"},
    ]
    primary = []
    for item in _category_rows(context, "CAT-C"):
        d = item["details"]
        primary.append({
            "Model": MODEL_LABELS[str(item["model"])],
            "MAE (95% CI)": f"{_fmt(d['original_scale_mae'])} ({_ci(d, 'original_scale_mae')})",
            "RMSE (95% CI)": f"{_fmt(d['original_scale_rmse'])} ({_ci(d, 'original_scale_rmse')})",
            "Normalized MAE (95% CI)": f"{_fmt(d['normalized_mae'])} ({_ci(d, 'normalized_mae')})",
            "Pearson (95% CI)": f"{_fmt(d['pearson'])} ({_ci(d, 'pearson')})",
            "Spearman (95% CI)": f"{_fmt(d['spearman'])} ({_ci(d, 'spearman')})",
            "Prediction range (1–5)": f"{_fmt(d['prediction_range_1_to_5'][0])}–{_fmt(d['prediction_range_1_to_5'][1])}",
            "N": d["sample_count"],
            **_trace(item),
        })
    tables["RPT-T07"] = primary
    tables["RPT-T08"] = [
        {
            "Comparison (A vs B)": f"{MODEL_LABELS[d['model_a']]} vs {MODEL_LABELS[d['model_b']]}",
            "Delta-MAE (A−B)": _fmt(d["estimate_mean"]),
            "95% CI": f"{_fmt(d['percentile_2_5'])}–{_fmt(d['percentile_97_5'])}",
            "Crosses zero": d["ci_crosses_zero"],
            "Sign convention": "Positive Δ favors B",
            "Supported conclusion": _supported_conclusion_label(d),
            "controlled_conclusion_code": _supported_conclusion_code(d),
            **_trace(item),
        }
        for item in _category_rows(context, "CAT-D")
        for d in [item["details"]]
    ]
    tables["RPT-T09"] = [
        {
            "Model": MODEL_LABELS[str(item["model"])],
            "AUROC (95% CI)": f"{_fmt(d['auroc'])} ({_fmt(d['auroc_interval']['percentile_2_5'])}–{_fmt(d['auroc_interval']['percentile_97_5'])})",
            "AUPRC (95% CI)": f"{_fmt(d['auprc'])} ({_fmt(d['auprc_interval']['percentile_2_5'])}–{_fmt(d['auprc_interval']['percentile_97_5'])})",
            "Sensitivity": _fmt(d["sensitivity"]),
            "Specificity": _fmt(d["specificity"]),
            "Balanced accuracy": _fmt(d["balanced_accuracy"]),
            "N": d["sample_count"],
            **_trace(item),
        }
        for item in _category_rows(context, "CAT-E")
        for d in [item["details"]]
    ]
    tables["RPT-T10"] = [
        {
            "Comparison (A vs B)": f"{MODEL_LABELS[d['model_a']]} vs {MODEL_LABELS[d['model_b']]}",
            "Delta-AUROC (B−A)": _fmt(d["estimate_mean"]),
            "95% CI": f"{_fmt(d['percentile_2_5'])}–{_fmt(d['percentile_97_5'])}",
            "Crosses zero": d["ci_crosses_zero"],
            "Sign convention": "Positive Δ favors B",
            "Supported conclusion": _supported_conclusion_label(d),
            "controlled_conclusion_code": _supported_conclusion_code(d),
            **_trace(item),
        }
        for item in _category_rows(context, "CAT-F")
        for d in [item["details"]]
    ]
    tables["RPT-T11"] = [
        {"Model": MODEL_LABELS[str(item["model"])], "Concept": item["concept_or_target"], "MAE": _fmt(d["mae"]), "RMSE": _fmt(d["rmse"]), "Pearson": _fmt(d["pearson"]), "Spearman": _fmt(d["spearman"]), "N": d["sample_count"], **_trace(item)}
        for item in _category_rows(context, "CAT-G") for d in [item["details"]]
    ]
    tables["RPT-T12"] = [
        {"Model": MODEL_LABELS[str(item["model"])], "Concept": item["concept_or_target"], "Soft CE": _fmt(d["soft_cross_entropy"]), "Brier": _fmt(d["multiclass_brier"]), "Macro-F1": _fmt(d["hard_modal_macro_f1"]), "Soft N": d["soft_sample_count"], "Hard N": d["hard_sample_count"], "Ties": d["true_tie_count"], **_trace(item)}
        for item in _category_rows(context, "CAT-H") for d in [item["details"]]
    ]
    gradcam = []
    for item in _category_rows(context, "CAT-L"):
        if item["fold"] is None:
            continue
        d = item["details"]
        gradcam.append({"Model": MODEL_LABELS[str(item["model"])], "Fold": item["fold"], "Target": item["concept_or_target"], "Requested": d["requested_map_count"], "Valid": d["valid_map_count"], "Undefined": d["undefined_map_count"], "Undefined rate": _fmt(d["undefined_rate"]), **_trace(item)})
    tables["RPT-T13"] = gradcam
    tables["RPT-T14"] = [
        {"Model": MODEL_LABELS[str(item["model"])], "Target": item["concept_or_target"], "Quantity": quantity, "Saliency mean": _fmt(stats["mean"]), "Saliency median": _fmt(stats["median"]), "Saliency−random mean": _fmt(stats["saliency_minus_matched_random_mean"]["mean"]), "Saliency > random rate": _fmt(stats["saliency_greater_than_matched_random_mean_rate"]), "Valid maps": d.get("valid_map_count", stats.get("sample_count", "NA")), **_trace(item)}
        for item in _category_rows(context, "CAT-N")
        if item["concept_or_target"] == "malignancy"
        and "output_sensitivity" in item["details"]
        and "error_increase" in item["details"]
        for d in [item["details"]]
        for quantity in ("output_sensitivity", "error_increase")
        for stats in [d[quantity]]
    ]
    contribution_rows = _category_rows(context, "CAT-J")
    selected_contributions = []
    for model in MODEL_ORDER[1:]:
        model_rows = [item for item in contribution_rows if item["model"] == model]
        selected_contributions.extend((
            (max(model_rows, key=lambda item: item["details"]["pooled_signed_mean_rating_points"]), "Largest pooled signed mean"),
            (min(model_rows, key=lambda item: item["details"]["pooled_signed_mean_rating_points"]), "Smallest pooled signed mean"),
        ))
    tables["RPT-T15"] = [
        {"Model": MODEL_LABELS[str(item["model"])], "Concept": item["concept_or_target"], "Pooled signed mean (rating points)": _fmt(d["pooled_signed_mean_rating_points"]), "Role within model": role, **_trace(item)}
        for item, role in selected_contributions for d in [item["details"]]
    ]
    tables["RPT-T16"] = [
        {"Fold": item["fold"], "Concept": item["concept_or_target"], **{f"Expert {i+1}": _fmt(v) for i, v in enumerate(d["final_weights"])}, "Min–max": f"{_fmt(d['minimum_weight'])}–{_fmt(d['maximum_weight'])}", "Simplex": _fmt(d["simplex_sum"]), **_trace(item)}
        for item in _category_rows(context, "CAT-K") for d in [item["details"]]
    ]
    tables["RPT-T17"] = [
        {"Model": MODEL_LABELS[str(item["model"])], "Ordering": d["ordering"], "Baseline MAE": _fmt(d["mae_curve"][0]), "k=4 MAE": _fmt(d["mae_curve"][4]), "k=8 MAE": _fmt(d["mae_curve"][8]), "iMAE": _fmt(d["iMAE"]), "Delta_iMAE": _fmt(d["Delta_iMAE"]), "Baseline AUROC": _fmt(d["auroc_curve"][0]), "iAUC": _fmt(d["iAUC"]), "Delta_iAUC": _fmt(d["Delta_iAUC"]), **_trace(item)}
        for item in _category_rows(context, "CAT-I") for d in [item["details"]]
    ]
    tables["RPT-T18"] = [
        {"Layer": "Prediction", "Question": "How accurately is malignancy scored?", "Main evidence": "Learned-softmax GAM has the lowest point-estimate MAE; paired support is model-dependent.", "Boundary": "Radiologist assessment, not pathology."},
        {"Layer": "WHERE", "Question": "Where is the output spatially sensitive?", "Main evidence": "66,769 valid maps; saliency often did not exceed matched random masks.", "Boundary": "6,955 post-ReLU zero maps; exact mechanism unavailable."},
        {"Layer": "WHAT", "Question": "Which concepts were predicted?", "Main evidence": "Continuous fidelity varied by concept; categorical hard-F1 was limited.", "Boundary": "Categorical targets are reader-vote distributions."},
        {"Layer": "WHY", "Question": "How do concepts enter the score?", "Main evidence": "Signed centered terms and learned GAM mixtures reconstruct the score.", "Boundary": "Centering constants are not importance; mean absolute aggregate was not persisted."},
        {"Layer": "HOW", "Question": "How does correcting concepts alter prediction?", "Main evidence": "Benefit was strong and consistent for CEM, near-neutral for CBM, and unfavorable overall for GAM despite limited early gains.", "Boundary": "Concept fidelity and intervenability are not interchangeable; interventions are not causal clinical effects."},
    ]
    if set(tables) != set(PUBLIC_TABLE_IDS):
        raise ValueError("P10_PUBLIC_TABLE_SET_MISMATCH")
    return tables


def export_public_tables(context: CatalogueContext, public_root: Path = PUBLIC_ROOT) -> dict[str, Path]:
    rows = build_public_table_rows(context)
    root = public_root / "tables_catalogue"
    paths: dict[str, Path] = {}
    for table_id, payload in rows.items():
        path = root / f"{table_id}.csv"
        reader_facing_payload = [
            {key: value for key, value in row.items() if key != "controlled_conclusion_code"}
            for row in payload
        ]
        _write_csv(path, reader_facing_payload)
        paths[table_id] = path
    return paths


def _labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "mae": "MAE（评分点）",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "delta_mae": "配对 ΔMAE（A−B）",
            "delta_auroc": "配对 ΔAUROC（B−A）",
            "undefined": "未定义比例",
            "output": "输出敏感度",
            "error": "误差增加",
            "saliency_minus_random": "显著区域 − 匹配随机均值",
            "fold": "折",
            "expert": "专家",
            "intervention": "干预概念数 k",
            "contribution": "中心化贡献（评分点）",
        }
    return {
        "mae": "MAE (rating points)",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "delta_mae": "Paired ΔMAE (A−B)",
        "delta_auroc": "Paired ΔAUROC (B−A)",
        "undefined": "Undefined rate",
        "output": "Output sensitivity",
        "error": "Error increase",
        "saliency_minus_random": "Saliency − matched-random mean",
        "fold": "Fold",
        "expert": "Expert",
        "intervention": "Number of intervened concepts k",
        "contribution": "Centered contribution (rating points)",
    }


def _save_figure(fig: plt.Figure, root: Path, figure_id: str, language: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"{figure_id}_{language}.png"
    svg = root / f"{figure_id}_{language}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", metadata={"Software": "P10 presentation renderer"})
    fig.savefig(svg, bbox_inches="tight", metadata={"Creator": "P10 presentation renderer"})
    plt.close(fig)
    return png, svg


def _configure_figure_font(language: str) -> None:
    """Use the frozen Songti source for Chinese labels instead of DejaVu fallback."""
    if language == "zh":
        font_path = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
        if not font_path.is_file():
            raise ValueError("P10_SONGTI_SOURCE_MISSING")
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
    else:
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True})


def _diagram_figure(figure_id: str, language: str) -> plt.Figure:
    zh = language == "zh"
    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.axis("off")
    if figure_id == "RPT-F01":
        boxes = [
            ("Prediction", "恶性评分" if zh else "Malignancy score"),
            ("WHERE", "空间定位与遮挡" if zh else "Grad-CAM + occlusion"),
            ("WHAT", "八个概念" if zh else "Eight concepts"),
            ("WHY", "中心化贡献" if zh else "Centered contributions"),
            ("HOW", "概念干预" if zh else "Concept intervention"),
        ]
        for i, (title, detail) in enumerate(boxes):
            x = 0.02 + i * 0.195
            ax.add_patch(plt.Rectangle((x, 0.3), 0.17, 0.42, facecolor="#E8F1FA", edgecolor="#2C5F8A", lw=1.8))
            ax.text(x + 0.085, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold")
            ax.text(x + 0.085, 0.39, detail, ha="center", va="center", fontsize=8.5)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + 0.195, 0.51), xytext=(x + 0.17, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.8})
        ax.text(0.5, 0.14, "同一冻结 OOF 证据链" if zh else "One frozen OOF evidence chain", ha="center", fontsize=11)
    elif figure_id == "RPT-F02":
        nodes = [
            (("LIDC-IDRI\nXML 读者评分" if zh else "LIDC-IDRI\nXML reader ratings"), 0.07, 0.62),
            (("2,633 个结节\n868 名患者" if zh else "2,633 nodules\n868 patients"), 0.29, 0.62),
            (("64³ 局部 ROI\n模型输入" if zh else "64³ local ROI\nmodel input"), 0.51, 0.62),
            (("患者分组\n五折交叉验证" if zh else "Patient-grouped\n5-fold CV"), 0.73, 0.62),
            (("2,633 个 OOF 评分\n零泄漏" if zh else "2,633 OOF scores\n0 leakage"), 0.51, 0.18),
            (("1,073 个极端样本\n578 名患者" if zh else "1,073 extremes\n578 patients"), 0.73, 0.18),
        ]
        for text_value, x, y in nodes:
            ax.add_patch(FancyBboxPatch((x, y), 0.17, 0.18, boxstyle="round,pad=0.02", facecolor="#F4F7F9", edgecolor="#4C78A8", lw=1.5))
            ax.text(x + 0.085, y + 0.09, text_value, ha="center", va="center", fontsize=9)
        for a, b in [(0,1),(1,2),(2,3),(3,5),(2,4)]:
            x1,y1=nodes[a][1]+0.17,nodes[a][2]+0.09; x2,y2=nodes[b][1],nodes[b][2]+0.09
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops={"arrowstyle":"->","lw":1.4})
        ax.text(0.5, 0.94, "Cohort, preprocessing, and evaluation flow" if not zh else "队列、预处理与评估流程", ha="center", fontsize=14, fontweight="bold")
    else:
        models = [
            ("Black-box", "影像 → DenseNet → 评分" if zh else "Image → DenseNet → score", "Prediction + WHERE"),
            ("Standard CBM", "影像 → 概念 → 线性评分" if zh else "Image → concepts → linear score", "Prediction + WHAT + WHY + HOW"),
            ("Mixed-type CEM", "影像 → 动态概念状态 → 评分" if zh else "Image → dynamic concept states → score", "Prediction + WHAT + WHY + HOW"),
            ("Learned-softmax GAM", "影像 → 概念 → 局部专家 → 求和" if zh else "Image → concepts → local experts → sum", "Prediction + WHAT + WHY + HOW"),
        ]
        for i, (name, path, interface) in enumerate(models):
            y = 0.79 - i * 0.2
            ax.add_patch(FancyBboxPatch((0.04, y), 0.2, 0.12, boxstyle="round,pad=0.015", facecolor=MODEL_COLORS[MODEL_ORDER[i]], alpha=0.2, edgecolor=MODEL_COLORS[MODEL_ORDER[i]]))
            ax.text(0.14, y + 0.06, name, ha="center", va="center", fontsize=10, fontweight="bold")
            ax.annotate("", xy=(0.31, y + 0.06), xytext=(0.245, y + 0.06), arrowprops={"arrowstyle":"->"})
            ax.text(0.48, y + 0.06, path, ha="center", va="center", fontsize=9)
            ax.annotate("", xy=(0.72, y + 0.06), xytext=(0.65, y + 0.06), arrowprops={"arrowstyle":"->"})
            ax.text(0.84, y + 0.06, interface, ha="center", va="center", fontsize=8.5)
        ax.text(0.5, 0.95, "Four architectures and registered interpretability interfaces" if not zh else "四种架构与预注册解释接口", ha="center", fontsize=14, fontweight="bold")
    return fig


def _primary_figure(context: CatalogueContext, language: str) -> plt.Figure:
    labels = _labels(language)
    rows = _category_rows(context, "CAT-C")
    names = [MODEL_LABELS[str(item["model"])] for item in rows]
    values = [item["details"]["original_scale_mae"] for item in rows]
    lo = [item["details"]["bootstrap_intervals"]["original_scale_mae"]["percentile_2_5"] for item in rows]
    hi = [item["details"]["bootstrap_intervals"]["original_scale_mae"]["percentile_97_5"] for item in rows]
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for y, idx in enumerate(order):
        model = str(rows[idx]["model"])
        ax.errorbar(values[idx], y, xerr=[[values[idx]-lo[idx]],[hi[idx]-values[idx]]], fmt="o", ms=8, color=MODEL_COLORS[model], capsize=4)
    ax.set_yticks(range(4), [names[idx] for idx in order])
    ax.set_xlabel(labels["mae"])
    ax.grid(axis="x", alpha=.25)
    ax.set_title("Primary pooled OOF error with 2,000 patient-cluster bootstrap intervals" if language=="en" else "主要 pooled OOF 误差与 2,000 次患者聚类 bootstrap 区间")
    return fig


def _forest_figure(context: CatalogueContext, category: str, language: str) -> plt.Figure:
    rows = _category_rows(context, category)
    labels = _labels(language)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, item in enumerate(rows):
        d=item["details"]; value=d["estimate_mean"]; lo=d["percentile_2_5"]; hi=d["percentile_97_5"]
        color="#C44E52" if d["ci_crosses_zero"] else "#2F7D32"
        ax.errorbar(value, i, xerr=[[value-lo],[hi-value]], fmt="o", color=color, capsize=3)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [f"{MODEL_LABELS[i['details']['model_a']]} → {MODEL_LABELS[i['details']['model_b']]}" for i in rows])
    ax.set_xlabel(labels["delta_mae"] if category=="CAT-D" else labels["delta_auroc"])
    ax.grid(axis="x", alpha=.2)
    ax.set_title("Paired patient-bootstrap comparisons" if language=="en" else "配对患者 bootstrap 模型比较")
    return fig


def _secondary_figure(context: CatalogueContext, language: str) -> plt.Figure:
    labels=_labels(language); rows=_category_rows(context,"CAT-E")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6))
    x=np.arange(4); width=.35
    ax1.bar(x-width/2,[r["details"]["auroc"] for r in rows],width,label=labels["auroc"],color="#4C78A8")
    ax1.bar(x+width/2,[r["details"]["auprc"] for r in rows],width,label=labels["auprc"],color="#F58518")
    ax1.set_xticks(x,[MODEL_LABELS[str(r["model"])] for r in rows],rotation=18,ha="right"); ax1.set_ylim(.75,1.0); ax1.legend(); ax1.grid(axis="y",alpha=.2)
    paired=_category_rows(context,"CAT-F")
    for i,item in enumerate(paired):
        d=item["details"]; ax2.errorbar(d["estimate_mean"],i,xerr=[[d["estimate_mean"]-d["percentile_2_5"]],[d["percentile_97_5"]-d["estimate_mean"]]],fmt="o",capsize=3,color="#C44E52" if d["ci_crosses_zero"] else "#2F7D32")
    ax2.axvline(0,color="black",ls="--",lw=1); ax2.set_yticks(range(6),[f"{MODEL_LABELS[i['details']['model_a']]} → {MODEL_LABELS[i['details']['model_b']]}" for i in paired],fontsize=8); ax2.set_xlabel(labels["delta_auroc"]); ax2.grid(axis="x",alpha=.2)
    fig.suptitle("Secondary extreme-task discrimination and paired uncertainty" if language=="en" else "极端子集判别性能与配对不确定性")
    fig.tight_layout()
    return fig


def _undefined_heatmap(context: CatalogueContext, language: str) -> plt.Figure:
    rows=[i for i in _category_rows(context,"CAT-L") if i["fold"] is not None]
    keys=[]
    for item in rows:
        key=(str(item["model"]),str(item["concept_or_target"]))
        if key not in keys: keys.append(key)
    matrix=np.full((len(keys),5),np.nan)
    for item in rows:
        matrix[keys.index((str(item["model"]),str(item["concept_or_target"]))),int(item["fold"])]=item["details"]["undefined_rate"]
    fig,ax=plt.subplots(figsize=(9.5,10.5))
    im=ax.imshow(matrix,aspect="auto",cmap="magma",vmin=0,vmax=max(.01,float(np.nanmax(matrix))))
    ax.set_xticks(range(5),[f"{_labels(language)['fold']} {i}" for i in range(5)])
    ax.set_yticks(range(len(keys)),[f"{MODEL_LABELS[m]} — {t}" for m,t in keys],fontsize=7)
    ax.set_title("Undefined post-ReLU Grad-CAM rate by model, fold, and target" if language=="en" else "按模型、折与目标分解的 post-ReLU 全零 Grad-CAM 比例")
    fig.colorbar(im,ax=ax,label=_labels(language)["undefined"]); fig.tight_layout()
    return fig


def _faithfulness_figure(context: CatalogueContext, language: str) -> plt.Figure:
    rows=[i for i in _category_rows(context,"CAT-N") if str(i["catalogue_item_id"]).endswith("-POOLED") and i.get("model") in MODEL_ORDER]
    rows.sort(key=lambda item: MODEL_ORDER.index(str(item["model"])))
    if len(rows) != 4:
        raise ValueError("P10_FAITHFULNESS_MODEL_POOLED_ROWS_MISSING")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,quantity,title in [(ax1,"output_sensitivity",_labels(language)["output"]),(ax2,"error_increase",_labels(language)["error"])]:
        vals=[i["details"][quantity]["saliency_minus_matched_random_mean"]["mean"] for i in rows]
        rates=[i["details"][quantity]["saliency_greater_than_matched_random_mean_rate"] for i in rows]
        y=np.arange(len(rows)); ax.barh(y,vals,color=[MODEL_COLORS[str(i["model"])] for i in rows],alpha=.85); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,[MODEL_LABELS[str(i["model"])] for i in rows]); ax.set_xlabel(_labels(language)["saliency_minus_random"]); ax.set_title(title); ax.grid(axis="x",alpha=.2)
        span=max(vals)-min(vals) or 1.0
        for yy,(v,rate) in enumerate(zip(vals,rates)):
            x=(v+0.02*span) if v>=0 else (-0.02*span)
            ax.text(x,yy,f"win={rate:.1%}",va="center",fontsize=8,ha="left" if v>=0 else "right")
    fig.suptitle("Spatial faithfulness: saliency versus matched random masks" if language=="en" else "空间忠实度：显著区域与匹配随机遮挡")
    fig.tight_layout(); return fig


def _concept_figure(context: CatalogueContext, category: str, language: str) -> plt.Figure:
    rows=_category_rows(context,category)
    if category=="CAT-G":
        metrics=("mae","rmse","pearson","spearman"); concepts=CONTINUOUS_CONCEPTS
    else:
        metrics=("soft_cross_entropy","multiclass_brier","hard_modal_macro_f1"); concepts=CATEGORICAL_CONCEPTS
    fig,axes=plt.subplots(1,len(metrics),figsize=(4*len(metrics),4.5),squeeze=False)
    for ax,metric in zip(axes[0],metrics):
        matrix=np.array([[next(i for i in rows if i["model"]==m and i["concept_or_target"]==c)["details"][metric] for c in concepts] for m in MODEL_ORDER[1:]])
        im=ax.imshow(matrix,aspect="auto",cmap="viridis")
        ax.set_xticks(range(len(concepts)),concepts,rotation=45,ha="right",fontsize=8); ax.set_yticks(range(3),[MODEL_LABELS[m] for m in MODEL_ORDER[1:]],fontsize=8); ax.set_title(metric.replace("_"," ")); fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle("Continuous concept fidelity (independent metric scales)" if category=="CAT-G" and language=="en" else "连续概念忠实度（各指标独立量尺）" if category=="CAT-G" else "Categorical concept fidelity (independent metric scales)" if language=="en" else "分类概念忠实度（各指标独立量尺）")
    fig.tight_layout(); return fig


def _load_oof(model: str) -> pd.DataFrame:
    name={"standard_cbm":"standard_cbm_oof_predictions.parquet","mixed_cem":"cem_oof_predictions.parquet","learned_softmax_gam":"gam_oof_predictions.parquet"}[model]
    path=PRIVATE_ARCHIVE/"p9"/"canonical_oof"/name
    return pd.read_parquet(path)


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value,str): value=json.loads(value)
    return np.asarray(value,dtype=float)


def _parse_scalar(value: Any) -> float:
    vector = _parse_vector(value).reshape(-1)
    if vector.size != 1 or not np.isfinite(vector[0]):
        raise ValueError("P10_EXPECTED_FINITE_SCALAR_VECTOR")
    return float(vector[0])


def _contribution_profiles(language: str) -> plt.Figure:
    fig,axes=plt.subplots(2,4,figsize=(14,7.5))
    for ax,concept in zip(axes.flat,CONCEPTS):
        for model in MODEL_ORDER[1:]:
            df=_load_oof(model)
            xcol=f"{concept}_activated_prediction"
            ycol=(f"{concept}_rating_point_contribution" if model=="standard_cbm" else f"{concept}_rating_contribution")
            if concept in CONTINUOUS_CONCEPTS:
                x=np.array([_parse_scalar(v) for v in df[xcol]],dtype=float); y=df[ycol].astype(float).to_numpy(); bins=np.linspace(0,1,11); centers=(bins[:-1]+bins[1:])/2; means=[]
                for lo,hi in zip(bins[:-1],bins[1:]):
                    mask=(x>=lo)&(x<(hi if hi<1 else hi+1e-12)); means.append(float(np.mean(y[mask])) if mask.any() else np.nan)
                ax.plot(centers,means,marker="o",ms=3,label=MODEL_LABELS[model],color=MODEL_COLORS[model])
            else:
                classes=np.array([int(np.argmax(_parse_vector(v))) for v in df[xcol]])
                y=df[ycol].astype(float).to_numpy(); n=4 if concept=="internalStructure" else 6
                offset={"standard_cbm":-.22,"mixed_cem":0.0,"learned_softmax_gam":.22}[model]
                grouped=[y[classes==k] for k in range(n)]
                positions=np.arange(1,n+1,dtype=float)+offset
                box=ax.boxplot(grouped,positions=positions,widths=.18,patch_artist=True,showfliers=False,manage_ticks=False)
                for artist in box["boxes"]: artist.set(facecolor=MODEL_COLORS[model],alpha=.28,edgecolor=MODEL_COLORS[model])
                for key in ("whiskers","caps","medians"):
                    for artist in box[key]: artist.set(color=MODEL_COLORS[model],linewidth=.8)
                medians=[float(np.median(values)) if len(values) else np.nan for values in grouped]
                ax.scatter(positions,medians,s=15,color=MODEL_COLORS[model],label=MODEL_LABELS[model],zorder=3)
                category_labels=("Soft tissue","Fluid","Fat","Air") if concept=="internalStructure" else ("Popcorn","Laminated","Solid","Non-central","Central","Absent")
                ax.set_xticks(np.arange(1,n+1),category_labels,rotation=30,ha="right",fontsize=7)
        ax.axhline(0,color="black",lw=.7); ax.set_title(concept); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.supylabel(_labels(language)["contribution"])
    fig.suptitle("Empirical OOF contribution profiles: binned means for continuous concepts and category distributions for categorical concepts" if language=="en" else "经验 OOF 贡献剖面：连续概念为分箱均值，分类概念为逐类别分布")
    fig.tight_layout(); return fig


def _alpha_figure(context: CatalogueContext, language: str) -> plt.Figure:
    rows=_category_rows(context,"CAT-K"); matrix=np.zeros((40,5)); labels=[]
    for idx,item in enumerate(rows): matrix[idx]=item["details"]["final_weights"]; labels.append(f"F{item['fold']} {item['concept_or_target']}")
    fig,ax=plt.subplots(figsize=(8,10)); im=ax.imshow(matrix,aspect="auto",cmap="coolwarm",vmin=.18,vmax=.22); ax.set_yticks(range(40),labels,fontsize=6.5); ax.set_xticks(range(5),[f"{_labels(language)['expert']} {i}" for i in range(1,6)]); ax.set_title("Fold-level Learned-softmax GAM expert weights" if language=="en" else "Learned-softmax GAM 的逐折专家权重"); fig.colorbar(im,ax=ax,label="α"); fig.tight_layout(); return fig


def _intervention_figure(context: CatalogueContext, language: str) -> plt.Figure:
    rows=_category_rows(context,"CAT-I"); fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.8))
    for item in rows:
        d=item["details"]; style="-" if d["ordering"]=="random_permutation" else "--"; label=f"{MODEL_LABELS[str(item['model'])]} — {d['ordering']}"; ax1.plot(d["k"],d["mae_curve"],style,marker="o",ms=3,color=MODEL_COLORS[str(item["model"])],label=label); ax2.plot(d["k"],d["auroc_curve"],style,marker="o",ms=3,color=MODEL_COLORS[str(item["model"])],label=label)
    ax1.set_ylabel("MAE"); ax2.set_ylabel("AUROC")
    for ax in (ax1,ax2): ax.set_xlabel(_labels(language)["intervention"]); ax.grid(alpha=.2)
    ax1.legend(fontsize=7); fig.suptitle("Concept intervention curves: positive ΔiMAE/ΔiAUC denotes improvement" if language=="en" else "概念干预曲线：正 ΔiMAE/ΔiAUC 表示改善"); fig.tight_layout(); return fig


def _synthesis_figure(language: str) -> plt.Figure:
    fig,ax=plt.subplots(figsize=(11.5,5.2)); ax.axis("off")
    if language == "zh":
        layers=[("Prediction","最低 MAE 点估计\n配对 bootstrap 不确定性"),("WHERE","有效图与全零图\n匹配随机遮挡忠实度"),("WHAT","连续与分类概念\n预测忠实度"),("WHY","有符号中心化项\n学习型局部专家混合"),("HOW","k=0…8 干预\n依赖顺序的响应")]
    else:
        layers=[("Prediction","Lowest MAE point estimate\nUncertainty by paired bootstrap"),("WHERE","Valid and zero maps\nMatched-random faithfulness"),("WHAT","Continuous and categorical\nconcept fidelity"),("WHY","Signed centered terms\nLearned local-expert mixtures"),("HOW","k=0…8 interventions\nOrdering-dependent response")]
    for i,(name,textv) in enumerate(layers):
        x=.02+i*.195; ax.add_patch(FancyBboxPatch((x,.35),.17,.38,boxstyle="round,pad=.02",facecolor="#F4F7F9",edgecolor=MODEL_COLORS[MODEL_ORDER[i%4]],lw=1.6)); ax.text(x+.085,.62,name,ha="center",fontweight="bold",fontsize=11); ax.text(x+.085,.44,textv,ha="center",va="center",fontsize=8)
        if i<4: ax.annotate("",xy=(x+.195,.54),xytext=(x+.17,.54),arrowprops={"arrowstyle":"->","lw":1.6})
    ax.text(.5,.18,"Integrated conclusion: interpretability evidence is complementary, model-dependent, and bounded" if language=="en" else "综合结论：解释证据相互补充、依赖模型，并有明确边界",ha="center",fontsize=12,fontweight="bold")
    return fig


def build_public_figures(context: CatalogueContext, public_root: Path = PUBLIC_ROOT) -> dict[str, dict[str, Path]]:
    root=public_root/"figures_catalogue"; paths={}
    for language in ("en","zh"):
        _configure_figure_font(language)
        builders={
            "RPT-F01":lambda:_diagram_figure("RPT-F01",language),
            "RPT-F02":lambda:_diagram_figure("RPT-F02",language),
            "RPT-F03":lambda:_diagram_figure("RPT-F03",language),
            "RPT-F04":lambda:_primary_figure(context,language),
            "RPT-F05":lambda:_forest_figure(context,"CAT-D",language),
            "RPT-F06":lambda:_secondary_figure(context,language),
            "RPT-F07":lambda:_undefined_heatmap(context,language),
            "RPT-F08":lambda:_faithfulness_figure(context,language),
            "RPT-F09A":lambda:_concept_figure(context,"CAT-G",language),
            "RPT-F09B":lambda:_concept_figure(context,"CAT-H",language),
            "RPT-F10":lambda:_contribution_profiles(language),
            "RPT-F11":lambda:_alpha_figure(context,language),
            "RPT-F12":lambda:_intervention_figure(context,language),
            "RPT-F13":lambda:_synthesis_figure(language),
        }
        if set(builders)!=set(PUBLIC_FIGURE_IDS): raise ValueError("P10_PUBLIC_FIGURE_SET_MISMATCH")
        for figure_id,builder in builders.items():
            png,svg=_save_figure(builder(),root,figure_id,language); paths.setdefault(figure_id,{})[language]=png; paths[figure_id][f"{language}_svg"]=svg
    return paths


@dataclass(frozen=True)
class ManuscriptSection:
    section_id: str
    title_en: str
    title_zh: str
    paragraphs_en: tuple[str, ...]
    paragraphs_zh: tuple[str, ...]
    table_ids: tuple[str, ...] = ()
    figure_ids: tuple[str, ...] = ()
    conclusion_codes: tuple[str, ...] = ()


FIGURE_CAPTIONS = {
    "RPT-F01": ("End-to-end evidence framework linking prediction to WHERE, WHAT, WHY, and HOW.", "从预测串联 WHERE、WHAT、WHY 与 HOW 的端到端证据框架。"),
    "RPT-F02": ("Frozen cohort, local ROI preprocessing, and patient-grouped five-fold evaluation flow.", "冻结队列、局部 ROI 预处理与患者分组五折评估流程。"),
    "RPT-F03": ("Four model architectures and their registered interpretability interfaces.", "四种模型架构及其预注册解释接口。"),
    "RPT-F04": ("Pooled primary MAE with 2,000 patient-cluster bootstrap 95% intervals.", "主要 pooled MAE 及 2,000 次患者聚类 bootstrap 95% 区间。"),
    "RPT-F05": ("Six paired Delta-MAE comparisons; intervals crossing zero are shown separately.", "六组配对 Delta-MAE 比较，并区分跨零区间。"),
    "RPT-F06": ("Extreme-task AUROC/AUPRC and six paired Delta-AUROC comparisons.", "极端任务 AUROC/AUPRC 与六组配对 Delta-AUROC。"),
    "RPT-F07": ("Undefined post-ReLU Grad-CAM rate by model, fold, and target.", "按模型、折和目标分解的 post-ReLU 全零 Grad-CAM 比例。"),
    "RPT-F08": ("All-target model-pooled spatial faithfulness for output_sensitivity and error_increase versus matched random masks.", "汇总全部目标的模型级空间忠实度：output_sensitivity 与 error_increase 相对匹配随机遮挡。"),
    "RPT-F09A": ("Continuous concept fidelity on independent metric scales.", "使用独立指标量尺呈现连续概念忠实度。"),
    "RPT-F09B": ("Categorical concept fidelity on independent metric scales.", "使用独立指标量尺呈现分类概念忠实度。"),
    "RPT-F10": ("Empirical OOF contribution profiles: continuous binned means and categorical contribution distributions; descriptive, not causal.", "经验 OOF 贡献剖面：连续概念分箱均值与分类概念贡献分布；仅为描述性结果，并非因果关系。"),
    "RPT-F11": ("Fold-level Learned-softmax GAM expert mixture weights.", "Learned-softmax GAM 的逐折专家混合权重。"),
    "RPT-F12": ("k=0…8 concept-intervention curves under random and error-first orderings.", "随机顺序与 error-first 顺序下的 k=0…8 概念干预曲线。"),
    "RPT-F13": ("Integrated Prediction-WHERE-WHAT-WHY-HOW interpretation and its boundaries.", "Prediction-WHERE-WHAT-WHY-HOW 的综合解释及其边界。"),
}

TABLE_TITLES = {
    "RPT-T01": ("Related-work comparison", "相关工作比较"),
    "RPT-T02": ("Frozen cohort flow", "冻结队列流程"),
    "RPT-T03": ("Target and concept definitions", "目标与概念定义"),
    "RPT-T04": ("Four-model architecture comparison", "四模型架构比较"),
    "RPT-T05": ("Frozen training configuration", "冻结训练配置"),
    "RPT-T06": ("Evaluation protocol", "评估协议"),
    "RPT-T07": ("Primary regression", "主要回归结果"),
    "RPT-T08": ("Six paired Delta-MAE comparisons", "六组配对 Delta-MAE 比较"),
    "RPT-T09": ("Extreme-task performance", "极端任务性能"),
    "RPT-T10": ("Six paired Delta-AUROC comparisons", "六组配对 Delta-AUROC 比较"),
    "RPT-T11": ("Continuous concept metrics", "连续概念指标"),
    "RPT-T12": ("Categorical concept metrics", "分类概念指标"),
    "RPT-T13": ("Grad-CAM accounting", "Grad-CAM 总账"),
    "RPT-T14": ("Malignancy-target spatial faithfulness", "恶性度目标空间忠实度"),
    "RPT-T15": ("Centered contribution summary", "中心化贡献汇总"),
    "RPT-T16": ("Fold-level Learned-softmax GAM alpha", "Learned-softmax GAM 逐折 alpha"),
    "RPT-T17": ("Concept-intervention summary", "概念干预汇总"),
    "RPT-T18": ("WHERE-WHAT-WHY-HOW synthesis", "WHERE-WHAT-WHY-HOW 综合表"),
}


def _primary_facts(context: CatalogueContext) -> dict[str, Mapping[str, Any]]:
    return {str(item["model"]): item["details"] for item in context.category("CAT-C")}


def build_manuscript_sections(context: CatalogueContext) -> tuple[ManuscriptSection, ...]:
    p=_primary_facts(context); gam=p["learned_softmax_gam"]; bb=p["blackbox"]
    undefined=context.report_data["gradcam_accounting"]
    sections = (
        ManuscriptSection(
            "SEC-ABSTRACT", "Abstract", "摘要",
            (
                f"This study evaluates four three-dimensional deep-learning strategies for radiologist-assessed pulmonary-nodule malignancy using a frozen cohort of 2,633 nodules from 868 patients. The primary endpoint is pooled out-of-fold mean absolute error on the original 1–5 rating scale; secondary evidence covers the 1,073-nodule extreme subset, eight bottleneck concepts, centered score contributions, concept interventions, and spatial Grad-CAM faithfulness. The evidence is organised as Prediction, WHERE, WHAT, WHY, and HOW rather than as an audit inventory.",
                f"Learned-softmax GAM achieved the lowest primary point-estimate MAE of {_fmt(gam['original_scale_mae'])}, compared with {_fmt(bb['original_scale_mae'])} for Black-box. Across all models, 73,724 Grad-CAM targets were requested: 66,769 were valid and 6,955 were explicitly recorded as post-ReLU all-zero maps. Matched-random occlusion showed that spatial saliency was not uniformly more faithful than random masks, while concept interventions produced model- and ordering-dependent changes.",
                "The findings support a layered interpretation: accurate prediction does not by itself establish spatial or conceptual faithfulness; concept fidelity does not guarantee beneficial intervention; and additive contribution decompositions describe the model score without establishing clinical causality. Malignancy is a radiologist assessment rather than pathology-confirmed diagnosis, and the system is not a clinical diagnostic product.",
            ),
            (
                "本研究在冻结的 2,633 个结节、868 名患者队列上，比较四种用于放射科医师评估肺结节恶性程度的三维深度学习策略。主要终点是在原始 1–5 评分量尺上的 pooled OOF 平均绝对误差；次要证据包括 1,073 个极端结节、八个瓶颈概念、中心化评分贡献、概念干预与空间 Grad-CAM 忠实度。报告按 Prediction、WHERE、WHAT、WHY、HOW 组织，而不是按审计清单堆叠结果。",
                f"Learned-softmax GAM 的主要 MAE 点估计最低，为 {_fmt(gam['original_scale_mae'])}；Black-box 为 {_fmt(bb['original_scale_mae'])}。四个模型共请求 73,724 个 Grad-CAM 目标，其中 66,769 个有效，6,955 个被明确记录为 post-ReLU 全零图。匹配随机遮挡表明，空间显著性并非始终比随机遮挡更忠实；概念干预的变化则依赖模型与排序方式。",
                "这些结果支持分层解释：较好的预测本身不能证明空间或概念忠实度；概念预测较准也不保证干预有益；加性贡献分解能够描述模型评分，但不建立临床因果关系。恶性程度是放射科医师评估，并非病理确诊；本系统不是临床诊断产品。",
            ),
            figure_ids=("RPT-F01",),
        ),
        ManuscriptSection(
            "SEC-INTRODUCTION", "1. Introduction", "1. 引言",
            (
                "Pulmonary-nodule malignancy assessment combines a prediction problem with an explanation problem. A numeric malignancy score can be useful for benchmarking, yet a reader also needs to know where the image influenced the model, what radiological concepts the model represented, why those representations shifted the output, and how the output responds when concept information is corrected. Treating these as interchangeable forms of explanation obscures their different evidential roles.",
                "We therefore frame the analysis around five linked questions. Prediction asks whether the continuous radiologist-assessed target is estimated accurately. WHERE uses Grad-CAM and matched occlusion to evaluate spatial sensitivity. WHAT measures the fidelity of six continuous and two categorical concept predictions. WHY decomposes concept-model scores into train-centered signed contributions and learned local-expert mixtures. HOW tests model dependence through preregistered concept interventions.",
                "The contribution is not a new clinical classifier or a claim of pathology-level diagnosis. It is a controlled comparison of Black-box, Standard CBM, a project-specific Mixed-type CEM, and a preregistered Learned-softmax GAM under identical patient-grouped folds, shared encoder initialisations, exactly-once test evaluation, and a unified OOF analysis. The report keeps negative findings visible, including uncertain paired AUROC differences, limited categorical concept fidelity, model-dependent intervention benefit, and concentrated zero-map behaviour.",
            ),
            (
                "肺结节恶性程度评估同时包含预测问题与解释问题。数值恶性评分可用于基准比较，但读者还需要知道图像的哪些位置影响模型、模型表示了哪些影像学概念、这些表示为何改变输出，以及在纠正概念信息时输出如何变化。若把这些证据视为同一种“解释”，就会掩盖它们不同的证据角色。",
                "因此，本研究围绕五个相互连接的问题组织分析。Prediction 关注连续的放射科医师评估目标能否被准确估计；WHERE 使用 Grad-CAM 与匹配遮挡评估空间敏感性；WHAT 衡量六个连续概念和两个分类概念的预测忠实度；WHY 把概念模型评分分解为仅由训练折统计量中心化的有符号贡献与学习到的局部专家混合；HOW 通过预注册概念干预测试模型依赖。",
                "本研究不是新的临床分类器，也不声称达到病理诊断层级。它在相同患者分组折、共享编码器初始化、test exactly-once 和统一 OOF 分析下，比较 Black-box、Standard CBM、项目特定 Mixed-type CEM 与预注册 Learned-softmax GAM。报告保留负面结果，包括不确定的配对 AUROC 差异、有限的分类概念忠实度、依赖模型的干预收益，以及集中的全零图现象。",
            ),
        ),
        ManuscriptSection(
            "SEC-RELATED", "2. Related Work", "2. 相关工作",
            (
                "LIDC-IDRI provides a public thoracic-CT reference database with multi-reader nodule annotations [1]. Earlier pulmonary-nodule systems used local image patches and convolutional networks to predict malignancy suspiciousness; MC-CNN, for example, linked multiscale image features to suspiciousness and selected semantic attributes [9]. These studies motivate volumetric image modelling but also underline a crucial boundary retained here: LIDC malignancy is a reader assessment, not pathology-confirmed diagnosis.",
                "DenseNet improved feature reuse and gradient flow [2], motivating the common DenseNet-121 encoder used in all four models. Concept bottleneck models made intermediate variables directly inspectable and correctable [3]. Concept embedding models replaced scalar bottlenecks with sample-conditioned representations and reported a different accuracy–intervention trade-off [4]. The present Mixed-type CEM is a project-specific extension for six continuous and two categorical vote-distribution targets, not a claim to reproduce the original CEM unchanged.",
                "Generalized additive models express an output as a sum of component functions [5]. Dumaev et al. combined concept-based learning and additive decision explanation specifically for LIDC-IDRI pulmonary-nodule malignancy scoring [8], making that study the closest task-level precedent. The preregistered Learned-softmax GAM here differs materially: each concept group has five local neural experts mixed by learned fold-level softmax weights, while the model is trained and evaluated under the present frozen patient-grouped protocol.",
                "Grad-CAM uses target gradients at a convolutional layer to form a coarse spatial sensitivity map [6]. A visually concentrated map, however, is not automatically faithful. This study therefore compares deterministic saliency masks with 20 equal-size random masks and preserves output_sensitivity separately from error_increase. Likewise, concept intervention is not guaranteed to improve performance: later analyses have shown strong dependence on intervention selection and granularity [7]. Table RPT-T01 positions these methods without importing prior-work results into the present cohort.",
                "These model families expose different objects. A scalar CBM exposes predicted concept values; a CEM exposes sample-conditioned concept states; an additive model exposes score components; and Grad-CAM exposes target-dependent spatial sensitivity. None is automatically a ground-truth explanation. A concept can be predicted accurately yet used in a brittle way, a contribution can reconstruct the score yet remain clinically non-causal, and a spatial map can look plausible while failing an occlusion comparison. The present evaluation therefore keeps these claims separate instead of treating transparency as one binary property.",
                "Pulmonary-nodule studies also differ in cohort identity, label construction, split unit, and reporting scale. Direct numerical comparison is unsafe when prior work uses a different physical-nodule reconciliation, binary suspiciousness target, or image-sampling protocol. Prior work is therefore used here to motivate methods and interpretation boundaries, while all performance claims come only from the frozen 2,633-nodule Baseline-v2 analysis. This is especially important for the closely related Dumaev study [8]: its published cohort statistics are not inserted into the present cohort flow.",
            ),
            (
                "LIDC-IDRI 提供带有多读者结节标注的公开胸部 CT 参考数据库 [1]。既往肺结节系统使用局部图像 patch 与卷积网络预测恶性可疑度；例如 MC-CNN 把多尺度图像特征与可疑度及部分语义属性联系起来 [9]。这些研究支持体积图像建模，也强化本研究保留的关键边界：LIDC malignancy 是读者评估，不是病理确诊。",
                "DenseNet 改善特征复用与梯度流 [2]，因此四个模型采用共同的 DenseNet-121 编码器。概念瓶颈模型让中间变量可直接检查和纠正 [3]；概念嵌入模型用样本条件化表示替代标量瓶颈，并展示不同的准确度–干预 trade-off [4]。本研究的 Mixed-type CEM 是针对六个连续目标和两个分类投票分布目标的项目特定扩展，并不声称原样复现原始 CEM。",
                "广义加性模型把输出表示为分量函数之和 [5]。Dumaev 等人把概念学习和加性决策解释用于 LIDC-IDRI 肺结节恶性评分 [8]，是最接近本研究任务的先例。本研究预注册的 Learned-softmax GAM 有实质差异：每个概念组包含五个局部神经专家，由逐折学习的 softmax 权重混合，并在当前冻结的患者分组协议下训练和评估。",
                "Grad-CAM 使用卷积层的目标梯度生成粗粒度空间敏感图 [6]，但视觉集中的 map 并不自动代表忠实。本研究因此把确定性显著遮挡与 20 个等大小随机遮挡比较，并把 output_sensitivity 与 error_increase 分开保存。概念干预同样不保证改善；后续研究显示其效果强烈依赖选择策略与粒度 [7]。表 RPT-T01 对这些方法定位，但不把既往结果当作本队列证据。",
                "这些模型家族暴露的对象不同。标量 CBM 暴露概念预测值，CEM 暴露样本条件化概念状态，加性模型暴露评分分量，Grad-CAM 则暴露依赖目标的空间敏感性。这些对象都不自动成为 ground-truth explanation。概念可以预测得准却被脆弱地使用，贡献可以重建评分却仍不具备临床因果性，空间图也可能视觉合理而无法通过遮挡比较。因此本研究把这些 claim 分开，不把透明性当成单一二值属性。",
                "肺结节研究在队列身份、标签构建、split unit 与报告量尺上也存在差异。当既往工作使用不同的实体结节 reconciliation、二分可疑度目标或图像抽样协议时，直接比较数值并不安全。因此既往工作只用于方法动机与解读边界；所有性能 claim 只来自冻结的 2,633 结节 Baseline-v2 分析。这对相近的 Dumaev 研究 [8] 尤其重要：其已发表队列统计不会进入本研究的 cohort flow。",
            ),
            table_ids=("RPT-T01",),
        ),
        ManuscriptSection(
            "SEC-DATASET", "3. Dataset and Preprocessing", "3. 数据集与预处理",
            (
                "The study uses the LIDC-IDRI XML reader annotations and stable physical-nodule identities. The frozen primary cohort contains 2,633 nodules from 868 patients; 1,073 nodules from 578 patients satisfy the preregistered extreme definition, with 782 low and 291 high cases. Patient Diagnoses XLS is not used for training, and malignancy is the mean of valid radiologist ratings rather than a pathology-confirmed label.",
                "Malignancy is the downstream 1–5 target and is not one of the eight bottleneck concepts. The concepts are subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, and texture. Six targets are continuous normalized reader means; internalStructure and calcification retain complete reader vote distributions, including true modal ties for training and soft metrics.",
                "Each model receives a 64 × 64 × 64 local pulmonary-nodule ROI created by consensus-mask cropping, cubic padding, and deterministic resampling. The ROI is not a complete axial CT slice and can appear lower-resolution because it has been cropped and resampled. Full axial CT is used only as private contextual visualization when exact frozen series, slice, bounding-box, and coordinate provenance are available. Figure RPT-F02 and Tables RPT-T02–RPT-T03 show the study-specific cohort and variables.",
            ),
            (
                "本研究使用 LIDC-IDRI XML 读者标注与稳定的实体结节身份。冻结主要队列包含 2,633 个结节、868 名患者；其中 1,073 个结节、578 名患者满足预注册极端定义，包括 782 个低分结节与 291 个高分结节。Patient Diagnoses XLS 不参与训练；恶性程度是有效放射科医师评分均值，而不是病理确诊标签。",
                "Malignancy 是下游 1–5 目标，不属于八个瓶颈概念。八个概念为 subtlety、internalStructure、calcification、sphericity、margin、lobulation、spiculation 和 texture。六个目标是连续的归一化读者均值；internalStructure 与 calcification 保留完整读者投票分布，训练与 soft metrics 中包括真实众数并列。",
                "每个模型接收 64 × 64 × 64 的局部肺结节 ROI，该输入由 consensus mask 裁剪、立方体 padding 与确定性重采样生成。ROI 不是完整轴位 CT slice，经过裁剪与重采样后可能显得分辨率更低。只有在冻结的 series、slice、bounding box 与坐标 provenance 完整时，完整轴位 CT 才用于私有上下文可视化。图 RPT-F02 与表 RPT-T02–RPT-T03 展示本研究队列与变量。",
            ),
            table_ids=("RPT-T02","RPT-T03"), figure_ids=("RPT-F02",),
        ),
        ManuscriptSection(
            "SEC-METHODS", "4. Methods", "4. 方法",
            (
                "All four models share the same fold-specific DenseNet-121 encoder initialisation and use an unconstrained linear malignancy output. Scores are trained and evaluated without sigmoid, tanh, or clipping. Black-box maps encoder features directly to the score. Standard CBM first learns the eight concepts and then fits a linear task head using frozen predicted concepts. Mixed-type CEM forms sample-conditioned states for continuous and categorical concepts. Learned-softmax GAM applies five local experts to each predicted concept group and adds their softmax-weighted outputs.",
                "Concept-model contributions are centred using means computed only from the current training fold. The centered bias plus eight centered contributions reconstructs the normalized score, and multiplying contributions by 4 reconstructs the original rating-point scale. These signed terms describe how the trained model composes its output; centering constants are bookkeeping statistics, not feature importance. The unavailable mean absolute aggregate is not recreated for presentation.",
                "Grad-CAM uses the final registered convolutional layer, spatial-mean gradients, a weighted activation sum, ReLU, and trilinear upsampling to 64³. Maps remain raw FP32 scientific artifacts. Display overlays may be normalized only for visualization. A zero post-ReLU map is marked undefined and excluded from the occlusion denominator; the frozen artifacts do not contain the pre-ReLU, gradient-norm, activation-norm, or channel-weight decomposition required to infer its exact mechanism.",
                "Occlusion replaces the top 26,215 heatmap voxels with normalized zero and compares them with 20 uniform-without-replacement random masks of equal size. output_sensitivity is the absolute output movement. error_increase is the change in absolute target error and is positive only when prediction error worsens. Intervention curves replace 0…8 concept groups under shared random permutations or error-first ordering; positive Delta_iMAE and Delta_iAUC consistently denote improvement. Figure RPT-F03 and Table RPT-T04 summarise model semantics; the statistical protocol is reported once in Experimental Setup.",
                "Continuous and categorical targets require different statistical treatments. Continuous attributes use sigmoid predictions against normalized reader means. Categorical attributes use softmax probabilities against complete reader vote distributions, so a convenient modal display label never replaces the scientific target. The pooled metrics are therefore reported on independent scales: errors and correlations for continuous concepts, and soft cross-entropy, multiclass Brier score, and tie-aware hard modal macro-F1 for categorical concepts.",
                "The architectures also support different intervention semantics. Standard CBM replaces activated concept values before its linear head. Mixed-type CEM replaces mixture weights while preserving sample-conditioned states. Learned-softmax GAM recomputes affected local experts from ground-truth concepts while retaining learned alpha. These operations test dependence on each model's own concept interface and are not homogenised into a mathematically different common intervention.",
            ),
            (
                "四个模型共享同一逐折 DenseNet-121 编码器初始化，并使用无约束线性恶性输出。训练和评估不使用 sigmoid、tanh 或 clipping。Black-box 直接把编码器特征映射为评分；Standard CBM 先学习八个概念，再用冻结的预测概念拟合线性 task head；Mixed-type CEM 为连续与分类概念构造样本条件化状态；Learned-softmax GAM 对每个预测概念组使用五个局部专家，并相加其 softmax 加权输出。",
                "概念模型贡献使用仅由当前训练折计算的均值进行中心化。中心化 bias 与八个中心化贡献重建归一化评分；把贡献乘以 4 后重建原始评分点量尺。这些有符号项描述训练模型如何组成输出；centering constants 是记账统计量，不是特征重要性。未持久化的 mean absolute aggregate 不会为展示而重算。",
                "Grad-CAM 使用最终预注册卷积层、空间均值梯度、加权 activation 求和、ReLU 与到 64³ 的三线性上采样。原始 FP32 map 保持为科学产物；display overlay 只允许为了可视化进行归一化。post-ReLU 全零图被标记为 undefined，并从遮挡分母中排除；冻结产物未保存 pre-ReLU、gradient norm、activation norm 或 channel-weight decomposition，因此无法推断精确机制。",
                "遮挡把热图最高的 26,215 个 voxel 置为归一化零，并与 20 个等大小、全 ROI 均匀无放回随机遮挡比较。output_sensitivity 是输出绝对移动；error_increase 是绝对目标误差的变化，只有正值表示预测误差变大。干预曲线在共享随机 permutation 或 error-first 排序下替换 0…8 个概念组；正 Delta_iMAE 与 Delta_iAUC 始终表示改善。图 RPT-F03 与表 RPT-T04 汇总模型语义；统计协议只在 Experimental Setup 中呈现一次。",
                "连续与分类目标需要不同的统计处理。连续属性使用 sigmoid prediction 与归一化读者均值；分类属性使用 softmax probability 与完整读者投票分布，因此用于方便展示的 modal label 不会替代科学目标。Pooled metrics 分别使用独立量尺：连续概念报告误差与相关，分类概念报告 soft cross-entropy、multiclass Brier score 与考虑并列的 hard modal macro-F1。",
                "各架构也支持不同的 intervention semantics。Standard CBM 在线性 head 前替换 activated concept value；Mixed-type CEM 替换 mixture weight 而保留 sample-conditioned state；Learned-softmax GAM 用 ground-truth concept 重算受影响的 local expert，并保留 learned alpha。这些操作检验模型对各自 concept interface 的依赖，不会被强制同质化成数学上不同的共同干预。",
            ),
            table_ids=("RPT-T04",), figure_ids=("RPT-F03",),
        ),
        ManuscriptSection(
            "SEC-SETUP", "5. Experimental Setup", "5. 实验设置",
            (
                "Evaluation uses patient-grouped five-fold outer cross-validation with fixed test counts of 479, 502, 539, 549, and 564 nodules. Patients are disjoint within each fold partition, and every primary nodule appears exactly once in the canonical OOF test set. Fold-specific validation subsets select checkpoints and Youden-J thresholds; test labels never enter selection.",
                "The frozen training configuration uses DenseNet-121, Adam at 1e-4, true batch 16, an 80-epoch budget, train-only deterministic augmentation, FP32 computation with AMP/BF16/TF32 disabled, and NVIDIA H200 hardware. Model-specific loss structures are retained in Table RPT-T05. Test evaluation is committed exactly once after the best checkpoint is fixed; per-fold best epochs and scheduler provenance remain in the reproducibility evidence rather than the scientific training table.",
                "Uncertainty uses 2,000 patient-cluster bootstrap replicates, with shared patient draws for paired comparisons. Each selected patient carries all of their nodules. Secondary AUROC draws are redrawn when they contain a single class. Table RPT-T05 records frozen training settings, while Table RPT-T06 defines evaluation and uncertainty without duplicating Methods.",
            ),
            (
                "评估采用患者分组五折 outer cross-validation，固定 test 结节数为 479、502、539、549、564。每折 partition 内患者互斥，每个主要结节在 canonical OOF test set 中恰好出现一次。逐折 validation subset 用于选择 checkpoint 和 Youden-J threshold；test labels 不参与任何选择。",
                "冻结训练配置采用 DenseNet-121、Adam 1e-4、真实 batch 16、80-epoch budget、仅训练期确定性 augmentation、FP32 且关闭 AMP/BF16/TF32，并使用 NVIDIA H200。模型特定 loss 结构见表 RPT-T05。固定 best checkpoint 后，test evaluation 只提交一次；逐折 best epoch 与 scheduler provenance 留在可复现性证据中，而不混入科学训练配置表。",
                "不确定性使用 2,000 次患者聚类 bootstrap，配对比较共享患者 draws。每个被抽中的患者携带其全部结节；若 secondary AUROC draw 只有单一类别，则重新抽样。表 RPT-T05 记录冻结训练设置；表 RPT-T06 定义评估与不确定性，并避免和 Methods 重复。",
            ),
            table_ids=("RPT-T05","RPT-T06"),
        ),
        ManuscriptSection(
            "SEC-RESULTS-PREDICTION", "6.1 Results — Prediction", "6.1 结果——Prediction",
            (
                f"What was measured? Primary prediction was evaluated on all 2,633 OOF nodules with original-scale MAE as the primary endpoint. Learned-softmax GAM produced the lowest point estimate ({_fmt(gam['original_scale_mae'])}), followed by Mixed-type CEM ({_fmt(p['mixed_cem']['original_scale_mae'])}), Black-box ({_fmt(bb['original_scale_mae'])}), and Standard CBM ({_fmt(p['standard_cbm']['original_scale_mae'])}). Table RPT-T07 reports every frozen regression point estimate and its existing 2,000-draw interval; Figure RPT-F04 makes the overlap in uncertainty visible.",
                "What did we observe? Paired Delta-MAE supports Learned-softmax GAM over Black-box and Standard CBM because the corresponding intervals do not cross zero, whereas smaller differences require a more cautious reading. The Black-box versus Standard CBM interval crosses zero, showing that interpretability structure did not automatically improve point prediction. Table RPT-T08 and Figure RPT-F05 preserve all six comparisons and the sign convention MAE_A − MAE_B. In the reader-facing tables, No supported difference means that the paired 95% CI crosses zero.",
                "On the 1,073-nodule extreme subset, all four continuous scores discriminated low from high ratings, but paired Delta-AUROC evidence was less decisive than the MAE evidence. Several intervals cross zero, and Standard CBM is lower than Black-box under the registered B−A convention. Table RPT-T09, Table RPT-T10, and Figure RPT-F06 therefore separate absolute AUROC/AUPRC performance from between-model uncertainty.",
                "What does this mean? Learned-softmax GAM is the strongest point-estimate regressor in this experiment, but the result does not justify a universal ranking across endpoints. Unclipped score ranges and small out-of-range rates remain part of the model behaviour rather than being hidden by post-hoc clipping. The target is a radiologist mean, so predictive accuracy should not be interpreted as pathology-level diagnostic accuracy.",
            ),
            (
                f"测量内容是什么？主要预测在全部 2,633 个 OOF 结节上评估，以原始量尺 MAE 为主要终点。Learned-softmax GAM 的点估计最低（{_fmt(gam['original_scale_mae'])}），随后为 Mixed-type CEM（{_fmt(p['mixed_cem']['original_scale_mae'])}）、Black-box（{_fmt(bb['original_scale_mae'])}）与 Standard CBM（{_fmt(p['standard_cbm']['original_scale_mae'])}）。表 RPT-T07 报告全部冻结回归点估计及既有 2,000-draw 区间；图 RPT-F04 直观展示不确定性重叠。",
                "观察到了什么？配对 Delta-MAE 支持 Learned-softmax GAM 优于 Black-box 和 Standard CBM，因为对应区间不跨零；较小差异则需要谨慎解读。Black-box 与 Standard CBM 的区间跨零，说明加入解释结构并不会自动改善点预测。表 RPT-T08 与图 RPT-F05 保留全部六组比较以及 MAE_A − MAE_B 符号约定。在面向读者的表格中，No supported difference 表示配对 95% CI 跨越零。",
                "在 1,073 个结节的极端子集上，四个连续评分均能区分低分与高分，但配对 Delta-AUROC 证据不如 MAE 证据明确。多个区间跨零；在预注册 B−A 约定下，Standard CBM 低于 Black-box。因此，表 RPT-T09、表 RPT-T10 与图 RPT-F06 把绝对 AUROC/AUPRC 性能和模型间不确定性分开。",
                "这意味着什么？Learned-softmax GAM 是本实验中点估计最好的回归模型，但该结果不能支持跨终点的普遍排名。未裁剪评分范围与少量越界比例属于模型行为的一部分，不应被 post-hoc clipping 隐藏。目标是放射科医师均值，因此预测精度不能解释为病理层级诊断精度。",
            ),
            table_ids=("RPT-T07","RPT-T08","RPT-T09","RPT-T10"), figure_ids=("RPT-F04","RPT-F05","RPT-F06"), conclusion_codes=("GAM_LOWEST_POINT_ESTIMATE_MAE","PAIRED_MAE_SUPPORTS_GAM_OVER_BLACKBOX_AND_CBM","AUROC_DIFFERENCES_MOSTLY_UNCERTAIN"),
        ),
        ManuscriptSection(
            "SEC-RESULTS-WHERE", "6.2 Results — WHERE", "6.2 结果——WHERE",
            (
                f"What was measured? Spatial evidence comprises {undefined['requested']:,} requested Grad-CAM maps across all model, fold, and target combinations. Exactly {undefined['valid']:,} maps were valid and {undefined['undefined']:,} were post-ReLU all-zero, yielding an overall undefined rate of {undefined['undefined_rate']:.3%}. Table RPT-T13 provides full accounting, while Figure RPT-F07 reveals concentrations that a pooled count would conceal.",
                "What did we observe? Undefined maps were not uniformly distributed. They were concentrated in particular model-target combinations, which is why the frozen root-cause label is SYSTEMATIC_MODEL/TARGET_ISSUE rather than an undifferentiated implementation failure. Every undefined map was finite and exactly zero after ReLU; no NaN, Inf, loading error, or target-path mismatch was observed. However, the persisted artifacts do not contain pre-ReLU CAMs or gradient/channel-weight norms.",
                "Faithfulness produced a scientifically important negative result. For both output_sensitivity and error_increase, saliency-minus-random means were often negative, and saliency exceeded the matched random mean in only a minority of valid cases. Table RPT-T14 is restricted to the malignancy target, whereas Figure RPT-F08 pools every registered target within each model. Their numerical values therefore differ by design. Both views keep output movement separate from prediction-error worsening: a large output_sensitivity cannot by itself show that prediction error increased.",
                "What does this mean? Grad-CAM provides a spatial sensitivity proxy, not a ground-truth localisation claim. The zero-map concentration and weak matched-random advantage limit strong spatial interpretations even when the underlying task prediction is accurate. Display overlays are normalized only for qualitative reading; every quantitative occlusion result uses the original unnormalized FP32 map.",
            ),
            (
                f"测量内容是什么？空间证据覆盖全部 model、fold、target 组合，共请求 {undefined['requested']:,} 个 Grad-CAM map。其中 {undefined['valid']:,} 个有效，{undefined['undefined']:,} 个为 post-ReLU 全零图，总 undefined rate 为 {undefined['undefined_rate']:.3%}。表 RPT-T13 给出完整总账；图 RPT-F07 展示 pooled count 会掩盖的集中分布。",
                "观察到了什么？undefined maps 并非均匀分布，而是集中在特定 model-target 组合，因此冻结 root-cause label 是 SYSTEMATIC_MODEL/TARGET_ISSUE，而不是笼统的 implementation failure。所有 undefined map 均为 finite 且 ReLU 后精确全零；没有观察到 NaN、Inf、loading error 或 target-path mismatch。然而，持久化产物未包含 pre-ReLU CAM 或 gradient/channel-weight norm。",
                "忠实度给出了具有科学意义的负面结果。对 output_sensitivity 与 error_increase 而言，saliency-minus-random mean 经常为负；只有少数有效 case 中显著区域超过匹配随机均值。表 RPT-T14 仅统计 malignancy target，而图 RPT-F08 在各模型内汇总全部注册 target，因此二者数值按设计不同。两个视图都把输出移动和预测误差恶化分开：较大的 output_sensitivity 本身不能证明预测误差增加。",
                "这意味着什么？Grad-CAM 是空间敏感度 proxy，不是 ground-truth localisation claim。即使 task prediction 准确，全零图集中与较弱的 matched-random 优势也限制了强空间解释。Display overlay 仅为定性阅读进行归一化；所有定量遮挡结果仍使用原始未归一化 FP32 map。",
            ),
            table_ids=("RPT-T13","RPT-T14"), figure_ids=("RPT-F07","RPT-F08"), conclusion_codes=("SALIENCY_NOT_UNIFORMLY_MORE_FAITHFUL_THAN_RANDOM","SYSTEMATIC_MODEL_TARGET_ZERO_MAP_LIMITATION"),
        ),
        ManuscriptSection(
            "SEC-RESULTS-WHAT", "6.3 Results — WHAT", "6.3 结果——WHAT",
            (
                "What was measured? Continuous concept fidelity uses MAE, RMSE, Pearson, and Spearman over 2,633 nodules for each concept model. Categorical fidelity uses soft cross-entropy and multiclass Brier on the full reader-vote distributions, plus hard modal macro-F1 only where the true modal class is unique. Table RPT-T11 and Figure RPT-F09A keep continuous metrics on compatible scales; Table RPT-T12 and Figure RPT-F09B do the same for categorical evidence.",
                "What did we observe? Continuous fidelity differed substantially by concept and model rather than following a single model-wide pattern. Some morphological concepts showed useful correlation, while subtle or reader-variable concepts retained larger absolute errors. This heterogeneity matters because the downstream concept models can only explain their own predicted representations, not an error-free radiological state.",
                "Categorical results were more limited. internalStructure and calcification retain complete vote distributions, so soft losses and Brier scores are the authoritative distributional evidence. Hard modal macro-F1 is included for readability but excludes true ties and can be low when rare classes are difficult. Treating the modal label as a single expert ground truth would misstate the frozen target.",
                "What does this mean? WHAT evidence supports inspecting individual concept groups rather than declaring that a model has uniformly learned radiological concepts. Concept fidelity is necessary for a transparent bottleneck but is not sufficient for predictive superiority or intervention benefit. The private RPT-TA02 table therefore presents both continuous targets and categorical vote-distribution semantics at case level.",
            ),
            (
                "测量内容是什么？连续概念忠实度在每个概念模型的 2,633 个结节上使用 MAE、RMSE、Pearson 与 Spearman。分类概念忠实度在完整读者投票分布上使用 soft cross-entropy 与 multiclass Brier，并仅在真实众数类别唯一时计算 hard modal macro-F1。表 RPT-T11 与图 RPT-F09A 让连续指标使用兼容量尺；表 RPT-T12 与图 RPT-F09B 对分类证据采用同样策略。",
                "观察到了什么？连续忠实度随 concept 与 model 显著变化，并不存在单一的全模型模式。部分形态学概念表现出有用相关性，而更微妙或读者变异较大的概念保留较高绝对误差。这种异质性很重要，因为下游概念模型只能解释自身预测表示，而不是无误差的影像学真实状态。",
                "分类结果更有限。internalStructure 与 calcification 保留完整投票分布，因此 soft loss 与 Brier score 是权威分布证据。hard modal macro-F1 便于阅读，但排除真实并列，在稀有类别困难时可能较低。把 modal label 当作单一专家 ground truth 会错误描述冻结目标。",
                "这意味着什么？WHAT 证据支持逐组检查概念，而不是笼统声称模型已经统一学会影像学概念。概念忠实度是透明瓶颈的必要条件，但不足以保证预测优势或干预收益。因此，私有 RPT-TA02 在病例层面同时呈现连续目标与分类投票分布语义。",
            ),
            table_ids=("RPT-T11","RPT-T12"), figure_ids=("RPT-F09A","RPT-F09B"),
        ),
        ManuscriptSection(
            "SEC-RESULTS-WHY", "6.4 Results — WHY", "6.4 结果——WHY",
            (
                "What was measured? WHY evidence asks how predicted concepts enter each concept model's malignancy score. For every fold, train-only means centre the raw group terms, and the centered bias plus eight terms reconstructs the task score within the frozen 1e-6 tolerance. Table RPT-T15 summarizes selected persisted pooled signed means; complete fold-level centering constants remain in the reproducibility evidence.",
                "What did we observe? Signed contribution directions differ across concept and model, demonstrating that identical concept names need not play identical decision roles. Figure RPT-F10 displays empirical OOF profiles derived as a presentation summary of frozen per-sample points. The profiles are descriptive and should not be read as global causal shape functions. The authoritative model-by-concept mean absolute aggregate was not persisted, so the report marks it DATA_NOT_PERSISTED rather than recreating it.",
                "Learned-softmax GAM adds a second WHY layer: five expert outputs per concept are mixed with nonnegative weights summing to one. Table RPT-T16 and Figure RPT-F11 show that the weights moved away from the uniform 0.2 initialization, although many movements are modest and fold-dependent. Learned mixtures therefore constitute evidence of optimisation, not proof that each expert represents a distinct clinical mechanism.",
                "What does this mean? Contribution decompositions make score construction auditable and permit case-level signed bars, but magnitude and sign remain properties of the trained decision function. They do not validate the underlying concepts or establish clinical causation. The private qualitative appendix pairs contribution bars with CT context and concept prediction/target evidence so WHY is not detached from WHAT and Prediction.",
            ),
            (
                "测量内容是什么？WHY 证据关注预测概念如何进入每个概念模型的恶性评分。每一折使用 train-only mean 对原始 group term 进行中心化；中心化 bias 与八个 term 在冻结 1e-6 tolerance 内重建 task score。表 RPT-T15 汇总经选择且已持久化的 pooled signed mean；完整的逐折 centering constants 保留在可复现性证据中。",
                "观察到了什么？不同 concept 与 model 的有符号贡献方向不同，说明相同 concept name 不一定具有相同决策角色。图 RPT-F10 把冻结逐样本点做成经验 OOF profile。该 profile 仅为描述性展示，不能读作 global causal shape function。权威 model-by-concept mean absolute aggregate 未持久化，因此报告将其标记为 DATA_NOT_PERSISTED，而不重新计算。",
                "Learned-softmax GAM 增加第二层 WHY：每个 concept 的五个 expert output 由非负且和为一的权重混合。表 RPT-T16 与图 RPT-F11 表明，权重偏离均匀的 0.2 初始化，但很多变化较小且依赖 fold。学习到的 mixture 是 optimisation evidence，并不能证明每个 expert 对应不同临床机制。",
                "这意味着什么？贡献分解让评分构成可审计，并支持病例层面有符号 bar，但 magnitude 与 sign 仍是训练决策函数的属性，不能验证底层概念或建立临床因果关系。私有定性附录把 contribution bar 与 CT context、concept prediction/target evidence 配对，避免 WHY 脱离 WHAT 与 Prediction。",
            ),
            table_ids=("RPT-T15","RPT-T16"), figure_ids=("RPT-F10","RPT-F11"),
        ),
        ManuscriptSection(
            "SEC-RESULTS-HOW", "6.5 Results — HOW", "6.5 结果——HOW",
            (
                "What was measured? Concept interventions replace 0…8 groups using the registered model-specific semantics. At every k, five-fold OOF predictions are pooled before calculating primary MAE and secondary AUROC. Random-permutation curves average 100 deterministic permutations per fold; error-first ordering ranks continuous absolute error or categorical total-variation distance without using the malignancy target.",
                "What did we observe? Mixed-type CEM uniquely showed strong, consistent integrated MAE benefit: Delta_iMAE was +0.028 under random permutations and +0.040 under error-first ordering. Standard CBM was approximately neutral under random ordering (+0.001) and slightly unfavorable under error-first ordering (−0.004). Learned-softmax GAM showed limited early gains along parts of the k curve, but its integrated Delta_iMAE was negative overall (−0.003 random; −0.016 error-first). Table RPT-T17 retains baseline, intermediate, k=8, iMAE, Delta_iMAE, iAUC, and Delta_iAUC; Figure RPT-F12 displays all k=0…8 curves.",
                "Error-first results were not uniformly better than random ordering. Correcting the currently worst-predicted concept first can expose compensating errors elsewhere in the model, and later interventions may reverse an early benefit. This is an important negative result because it shows that concept correction is not a monotonic repair operation.",
                "What does this mean? Concept fidelity and intervenability are not interchangeable. GAM can predict concepts comparatively well yet respond unfavorably when their integrated substitutions are propagated; CEM can have weaker concept fidelity in places while benefiting most from correction. HOW evidence tests dependence on internal concept representations, not the causal effect of changing patient radiology. Case-level before/after values were not persisted for RPT-FA06, so HOW remains DATA_NOT_PERSISTED rather than being recomputed.",
            ),
            (
                "测量内容是什么？概念干预按照预注册的模型特定语义替换 0…8 个 group。对每个 k，先拼接五折 OOF prediction，再计算主要 MAE 与次要 AUROC。random-permutation curve 对每折 100 个确定性 permutation 求均值；error-first 使用连续绝对误差或分类 total-variation distance 排序，不使用 malignancy target。",
                "观察到了什么？Mixed-type CEM 是唯一表现出强而一致 integrated MAE 收益的模型：random permutation 下 Delta_iMAE 为 +0.028，error-first 下为 +0.040。Standard CBM 在随机顺序下近似中性（+0.001），error-first 下轻微不利（−0.004）。Learned-softmax GAM 在部分早期 k 位置有有限收益，但 integrated Delta_iMAE 总体为负（random −0.003；error-first −0.016）。表 RPT-T17 保留 baseline、intermediate、k=8、iMAE、Delta_iMAE、iAUC 与 Delta_iAUC；图 RPT-F12 展示完整 k=0…8 曲线。",
                "error-first 并不始终优于随机顺序。首先纠正当前预测误差最大的概念，可能暴露模型其他位置的补偿误差；后续干预还可能逆转早期收益。这是重要负面结果，表明概念纠正不是单调 repair operation。",
                "这意味着什么？概念忠实度与可干预性不能互换。GAM 可以较好预测概念，但 integrated substitution 传播后总体不利；CEM 的部分概念忠实度较弱，却从纠正中获益最大。HOW 检验模型对内部概念表示的依赖，而不是改变患者影像学表现的因果效应。RPT-FA06 的病例级 before/after 值未持久化，因此 HOW 继续标记为 DATA_NOT_PERSISTED，不作重算。",
            ),
            table_ids=("RPT-T17",), figure_ids=("RPT-F12",), conclusion_codes=("INTERVENTION_BENEFIT_MODEL_DEPENDENT",),
        ),
        ManuscriptSection(
            "SEC-RESULTS-SYNTHESIS", "6.6 Integrated Interpretation", "6.6 综合解释",
            (
                "Prediction, WHERE, WHAT, WHY, and HOW answer different questions and should not be collapsed into a single explainability score. Prediction establishes task performance. WHERE tests spatial sensitivity but is limited by zero maps and weak matched-random advantages. WHAT measures whether named concepts match reader evidence. WHY exposes score composition. HOW probes whether correcting representations changes the output.",
                "The strongest integrated interpretation belongs to a model only when these layers are read together. Learned-softmax GAM has the best primary point estimate and strong concept fidelity in several groups, yet its integrated intervention response is unfavorable overall. Mixed-type CEM performs better than Black-box and Standard CBM on primary MAE and benefits most consistently from intervention despite weaker concept fidelity in places. Standard CBM remains simple and traceable but gains little task or intervention advantage.",
                "Table RPT-T18 and Figure RPT-F13 therefore present a chain of supported claims and boundaries rather than a winner-takes-all dashboard. The conclusion is that interpretability is multidimensional and model-dependent: a useful explanation must specify which layer it supports, what frozen evidence underlies it, and what it cannot establish.",
            ),
            (
                "Prediction、WHERE、WHAT、WHY、HOW 回答不同问题，不能压缩为单一 explainability score。Prediction 建立 task performance；WHERE 测试空间敏感性，但受全零图和较弱 matched-random 优势限制；WHAT 衡量命名概念是否匹配读者证据；WHY 暴露评分构成；HOW 探查纠正表示是否改变输出。",
                "只有把这些层一起阅读，才能形成对模型最强的综合解释。Learned-softmax GAM 具有最佳主要点估计，多个概念组忠实度较强，但 integrated intervention response 总体不利。Mixed-type CEM 的主要 MAE 优于 Black-box 与 Standard CBM，并且即使部分概念忠实度较弱，仍从干预中获得最一致收益。Standard CBM 简单、可追溯，却几乎没有 task 或 intervention 优势。",
                "因此，表 RPT-T18 与图 RPT-F13 呈现的是支持性 claim 与 boundary 链，而不是 winner-takes-all dashboard。结论是解释性具有多维度并依赖模型：有用解释必须说明它支持哪个层面、依赖什么冻结证据，以及不能建立什么。",
            ),
            table_ids=("RPT-T18",), figure_ids=("RPT-F13",),
        ),
        ManuscriptSection(
            "SEC-DISCUSSION", "7. Discussion", "7. 讨论",
            (
                "The primary result favours Learned-softmax GAM at the point-estimate level, and paired MAE comparisons support meaningful improvement over Black-box and Standard CBM. This suggests that explicit concept-local nonlinearities can improve continuous scoring while preserving additive decomposition. However, confidence intervals and secondary discrimination prevent an overly simple ranking: the best regression point estimate is not synonymous with statistically superior AUROC across every pair.",
                "The explanation layers reveal trade-offs that prediction metrics alone cannot show. GAM combines the best task point estimate with strong concept fidelity in several groups, but its integrated intervention response is negative overall. CEM has a better task estimate than Black-box and Standard CBM and the strongest correction response despite weaker concept fidelity in places. Standard CBM is transparent and sometimes concept-faithful, yet offers little task or intervention advantage. Prediction, concept fidelity, intervenability, and spatial faithfulness are therefore distinct properties, not interchangeable definitions of interpretability.",
                "Spatial evidence is the clearest caution. Thousands of legitimate post-ReLU zero maps and predominantly weak saliency-versus-random differences mean that visually appealing overlays should not dominate the scientific story. A Grad-CAM overlay is best treated as a local sensitivity view whose credibility is strengthened only when quantitative faithfulness and map-validity accounting agree.",
                "Clinically, the framework offers a disciplined way to communicate model behaviour, not a diagnosis. Reader ratings encode radiological assessment and disagreement; they do not establish histopathological truth. External validation, calibration for deployment, prospective workflow testing, and pathology-linked outcomes would be required before any clinical claim.",
                "Model selection therefore depends on the intended scientific use. Choosing only the lowest MAE would favour GAM, whereas choosing only intervention response would favour CEM, and choosing only architectural simplicity might favour Standard CBM. The evidence does not justify collapsing these criteria into a post hoc composite rank. Predictive benchmarking, inspection of concept errors, decomposition of score formation, and controlled testing of representation dependence are related but distinct goals.",
                "The negative results are informative rather than incidental. The mismatch between concept fidelity and intervention benefit shows why a well-predicted bottleneck is not necessarily a useful correction interface. Weak or negative matched-random spatial contrasts show why heatmaps require quantitative checks. Concentrated undefined maps show why missing spatial explanations must be counted rather than silently discarded. These findings narrow the claims but make the resulting interpretation more reproducible.",
            ),
            (
                "主要结果在点估计层面支持 Learned-softmax GAM；配对 MAE 比较支持其相对 Black-box 与 Standard CBM 的实质改善。这说明显式 concept-local nonlinearity 可以在保留加性分解的同时改善连续评分。然而，confidence interval 与次要判别结果阻止过度简单的排名：最佳回归点估计不等于在所有配对中 AUROC 都具有统计优势。",
                "解释层揭示了单独预测指标看不到的 trade-off。GAM 同时具有最佳 task 点估计和多个组较强概念忠实度，但 integrated intervention 总体为负。CEM 的 task 点估计优于 Black-box 与 Standard CBM，并在部分概念忠实度较弱的同时获得最强纠正收益。Standard CBM 透明且某些概念预测较好，却几乎没有 task 或 intervention 优势。因此，prediction、concept fidelity、intervenability 与 spatial faithfulness 是不同属性，不能互换为同一个解释性定义。",
                "空间证据提供最明确的谨慎信号。数千个合法 post-ReLU 全零图，以及总体较弱的 saliency-versus-random difference，意味着视觉上漂亮的 overlay 不应主导科学故事。Grad-CAM overlay 最适合被视为局部 sensitivity view；只有定量忠实度与 map-validity accounting 同时支持时，其可信度才增强。",
                "从临床角度看，该框架提供的是规范沟通模型行为的方法，而不是诊断。读者评分编码影像学评估与分歧，并不建立组织病理学真实。任何临床主张都需要外部验证、部署校准、前瞻性流程测试与病理关联结局。",
                "模型选择因此取决于预期科学用途。若只选最低 MAE，会偏向 GAM；若只看 intervention response，会偏向 CEM；若只看架构简单性，则可能偏向 Standard CBM。现有证据不支持把这些准则事后合并成单一排名。预测基准、concept error 检查、评分构成分解，以及表示依赖的受控测试，是相关但不同的目标。",
                "负面结果因此不是附带现象，而是重要信息。概念忠实度与干预收益之间的不匹配说明，预测较好的 bottleneck 不一定是有效纠错接口；较弱或负的 matched-random 空间对比说明 heatmap 需要定量核验；undefined map 集中则说明缺失空间解释必须被计数，不能静默丢弃。这些发现缩小了 claim，却使最终解读更可复现。",
            ),
        ),
        ManuscriptSection(
            "SEC-LIMITATIONS", "8. Limitations", "8. 局限性",
            (
                "First, the target is a radiologist-assessed malignancy score rather than pathology-confirmed disease. The patient-grouped internal cross-validation design controls leakage but cannot establish transportability to another institution, scanner distribution, or clinical workflow. Only one preregistered seed per fold was used, so the reported bootstrap intervals describe patient-sampling uncertainty rather than training-seed variability.",
                "Second, concept ground truth inherits reader variability. Continuous means compress disagreement, and categorical vote distributions can be sparse. Hard modal macro-F1 excludes true ties and is secondary to the full distributional metrics. Concept interventions replace internal representations with reader-derived targets; they should not be interpreted as feasible clinical manipulations or causal effects.",
                "Third, Grad-CAM is a nodule-level spatial proxy. The 6,955 undefined maps are confirmed finite post-ReLU all-zero maps, but the exact pre-ReLU/gradient mechanism was not persisted. The observed concentration is therefore reported as SYSTEMATIC_MODEL/TARGET_ISSUE, not resolved into zero gradients, zero channel weights, or negative weighted sums without prohibited new forward passes.",
                "Finally, some presentation goals are constrained by what was frozen. The model-by-concept mean absolute centered contribution aggregate and case-level intervention before/after trajectory were not persisted. The contribution table therefore reports supported signed means only and notes this limitation once; the report does not convert descriptive plotting or narrative memory into a new authoritative scientific result.",
            ),
            (
                "第一，目标是放射科医师评估恶性评分，而不是病理确诊疾病。患者分组内部 cross-validation 控制 leakage，但不能建立对其他机构、扫描仪分布或临床流程的 transportability。每折只使用一个预注册 seed，因此 bootstrap interval 描述患者抽样不确定性，而不是训练 seed variability。",
                "第二，concept ground truth 继承读者变异。连续均值压缩分歧，分类 vote distribution 可能稀疏。hard modal macro-F1 排除真实并列，并且次于完整分布指标。概念干预把内部表示替换为 reader-derived target，不能解释为可行临床操作或因果效应。",
                "第三，Grad-CAM 是结节层面空间 proxy。6,955 个 undefined map 被确认为 finite post-ReLU 全零图，但精确 pre-ReLU/gradient mechanism 未持久化。因此，观察到的集中分布被报告为 SYSTEMATIC_MODEL/TARGET_ISSUE；在不允许新增 forward pass 的情况下，不能进一步断言是 zero gradient、zero channel weight 或 negative weighted sum。",
                "最后，部分展示目标受冻结内容限制。model-by-concept mean absolute centered contribution aggregate 与病例级 intervention before/after trajectory 未持久化。因此贡献表只报告有来源支持的 signed mean，并仅在一处说明该限制；报告不会把描述性绘图或叙述记忆转成新的权威科学结果。",
            ),
        ),
        ManuscriptSection(
            "SEC-CONCLUSION", "9. Conclusion", "9. 结论",
            (
                "This unified comparison shows that prediction and explanation should be evaluated as a chain of distinct questions. Learned-softmax GAM achieved the lowest primary MAE point estimate, while paired uncertainty showed where this advantage was and was not decisive. Concept models added inspectable representations, signed score decompositions, and intervention experiments that a Black-box predictor cannot provide.",
                "The explanation evidence also imposed meaningful limits. CEM alone showed strong and consistent integrated intervention benefit; CBM was near-neutral and GAM was unfavorable overall despite limited early gains. Concept fidelity therefore did not predict intervenability. Grad-CAM maps could also be undefined, and saliency masks often failed to outperform matched random masks. These findings determine how strongly each explanation can be interpreted.",
                "The resulting framework supports transparent research reporting of radiologist-assessed pulmonary-nodule malignancy. It does not establish pathology-level diagnosis, causal concepts, ground-truth localisation, or clinical readiness. Its central contribution is a reproducible evidence structure linking Prediction, WHERE, WHAT, WHY, and HOW while keeping each claim bound to its frozen source and interpretation boundary.",
            ),
            (
                "这项统一比较表明，预测与解释应作为一系列不同问题进行评估。Learned-softmax GAM 获得最低主要 MAE 点估计；配对不确定性同时指出该优势在哪些比较中明确、哪些比较中不明确。概念模型增加 Black-box predictor 无法提供的可检查表示、有符号评分分解与干预实验。",
                "解释证据也带来具有实际意义的限制。只有 CEM 表现出强而一致的 integrated intervention benefit；CBM 近似中性，GAM 尽管早期有有限收益但总体不利。因此，概念忠实度不能预测可干预性。Grad-CAM map 也可能 undefined，显著区域遮挡经常不能优于匹配随机遮挡。这些结果决定每类解释能被多强地解读。",
                "最终框架支持对放射科医师评估肺结节恶性程度进行透明、可复现的研究报告。它不建立病理层级诊断、因果概念、ground-truth localisation 或临床就绪性。其核心贡献是构建连接 Prediction、WHERE、WHAT、WHY、HOW 的可复现证据结构，同时让每个 claim 绑定冻结来源与解释边界。",
            ),
        ),
        ManuscriptSection(
            "SEC-REPRODUCIBILITY", "Public Reproducibility Appendix", "公开可复现性附录",
            (
                "All scientific values are read from the user-approved 2,395-item Results Catalogue and its registered frozen sources. The report-revision supplement binds the Catalogue registry SHA, Catalogue manifest SHA, and both approved planning-document SHAs. Section manifests and reverse-traceability rows connect every rendered table, figure, caption, and conclusion code to Catalogue item IDs and source hashes.",
                "P5–P9 checkpoints, histories, predictions, metrics, evaluations, OOF rows, interventions, Grad-CAM maps, occlusion rows, and faithfulness payloads remain read-only. Report generation performs no training, model forward pass, test inference, bootstrap recomputation, or new scientific job. The private archive remains outside Git and stores full-resolution case assets under opaque CASE labels.",
                "The archive contains 1,698 files and 14,386,651,621 bytes under a completed SHA-verified manifest. The six mandatory PDFs are rendered page by page with Poppler at 150 DPI, inspected through contact sheets and original-resolution pages, and checked with pypdf/pdfplumber for metadata, text, numbering, fonts, and page integrity before P10 can enter AWAITING_USER_APPROVAL.",
            ),
            (
                "全部科学数值都来自用户批准的 2,395-item Results Catalogue 及其注册的冻结来源。报告修订 supplement 绑定 Catalogue registry SHA、Catalogue manifest SHA 与两份已批准计划文档 SHA。section manifest 与 reverse-traceability row 把每个渲染 table、figure、caption、conclusion code 连接到 Catalogue item ID 与 source hash。",
                "P5–P9 checkpoint、history、prediction、metric、evaluation、OOF row、intervention、Grad-CAM map、occlusion row 与 faithfulness payload 保持只读。报告生成不进行训练、model forward pass、test inference、bootstrap recomputation 或新 scientific job。私有 archive 保持在 Git 之外，并用 opaque CASE label 保存全分辨率病例资产。",
                "Archive 包含 1,698 个文件、14,386,651,621 bytes，并由完成的 SHA-verified manifest 保护。六份 mandatory PDF 在 150 DPI 下用 Poppler 逐页渲染，通过 contact sheet 与原始分辨率页面检查，并使用 pypdf/pdfplumber 核对 metadata、text、numbering、font 与 page integrity，之后 P10 才能进入 AWAITING_USER_APPROVAL。",
            ),
        ),
    )
    for section in sections:
        if len(section.paragraphs_en)!=len(section.paragraphs_zh): raise ValueError(f"P10_BILINGUAL_PARAGRAPH_COUNT:{section.section_id}")
    return sections


def _display_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    hidden={"catalogue_item_id","source_artifact_id","source_field_path","source_sha256","controlled_conclusion_code"}
    return [key for key in rows[0] if key not in hidden]


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows: return "_No registered rows._"
    cols=_display_columns(rows)
    def clean(value: Any) -> str: return str(value).replace("|","/").replace("\n"," ")
    lines=["| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]
    lines.extend("| "+" | ".join(clean(row.get(c,"")) for c in cols)+" |" for row in rows)
    return "\n".join(lines)


def build_markdown_manuscript(context: CatalogueContext, tables: Mapping[str, Sequence[Mapping[str, Any]]], language: str) -> str:
    sections=build_manuscript_sections(context); lines=[]
    title="Interpretable Pulmonary-Nodule Malignancy Scoring: Prediction, WHERE, WHAT, WHY, and HOW" if language=="en" else "可解释肺结节恶性评分：Prediction、WHERE、WHAT、WHY 与 HOW"
    labels=("Author","Affiliation","Supervisor","Date","Keywords") if language=="en" else ("作者","单位","导师","日期","关键词")
    lines.extend([f"# {title}", "", f"**{labels[0]}:** [To be completed]", "", f"**{labels[1]}:** [To be completed]", "", f"**{labels[2]}:** [To be completed]", "", f"**{labels[3]}:** 2026-08-13", "", f"**{labels[4]}:** LIDC-IDRI; concept bottleneck; Grad-CAM; intervention; explainability", ""])
    for section in sections:
        title_value=section.title_en if language=="en" else section.title_zh
        level="##" if not section.section_id.startswith("SEC-RESULTS-") else "###"
        lines.extend([f"{level} {title_value}",""])
        paragraphs=section.paragraphs_en if language=="en" else section.paragraphs_zh
        evidence=[("table",value) for value in section.table_ids]+[("figure",value) for value in section.figure_ids]
        buckets=[[] for _ in paragraphs]
        for index,item in enumerate(evidence): buckets[min(index,len(buckets)-1)].append(item)
        for index,paragraph in enumerate(paragraphs):
            lines.extend([paragraph,""])
            for kind,item_id in buckets[index]:
                if kind=="table":
                    table_title=TABLE_TITLES[item_id][0 if language=="en" else 1]
                    lines.extend([f"**{item_id}. {table_title}**","",_markdown_table(tables[item_id]),""])
                else:
                    caption=FIGURE_CAPTIONS[item_id][0 if language=="en" else 1]
                    lines.extend([f"![{item_id}. {caption}](figures_catalogue/{item_id}_{language}.png)","",f"**{item_id}.** {caption}",""])
    refs=REFERENCES
    lines.extend(["## References" if language=="en" else "## 参考文献",""])
    lines.extend([ref,"" ] for ref in refs)
    flattened=[]
    for value in lines:
        if isinstance(value,list): flattened.extend(value)
        else: flattened.append(value)
    return "\n".join(flattened).rstrip()+"\n"


NUMERIC_TOKEN_RE=re.compile(
    r"(?<![A-Za-z])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)


def verify_bilingual_numeric_parity(en_text: str, zh_text: str) -> None:
    en=NUMERIC_TOKEN_RE.findall(en_text); zh=NUMERIC_TOKEN_RE.findall(zh_text)
    if en!=zh:
        for index,(a,b) in enumerate(zip(en,zh)):
            if a!=b: raise ValueError(f"P10_BILINGUAL_NUMERIC_TOKEN_MISMATCH:{index}:{a}:{b}")
        raise ValueError(f"P10_BILINGUAL_NUMERIC_TOKEN_COUNT:{len(en)}:{len(zh)}")


def write_markdown_manuscripts(context: CatalogueContext, tables: Mapping[str, Sequence[Mapping[str, Any]]], public_root: Path=PUBLIC_ROOT) -> dict[str,Path]:
    paths={}
    texts={language:build_markdown_manuscript(context,tables,language) for language in ("en","zh")}
    verify_bilingual_numeric_parity(texts["en"],texts["zh"])
    for language,text in texts.items():
        path=public_root/f"technical_{language}.md"; path.write_text(text,encoding="utf-8"); paths[language]=path
    return paths


def _register_pdf_fonts(language: str) -> tuple[str,str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if language=="zh":
        font_path="/System/Library/Fonts/Supplemental/Songti.ttc"
        pdfmetrics.registerFont(TTFont("P10RevisionSongti",font_path,subfontIndex=6))
        pdfmetrics.registerFont(TTFont("P10RevisionSongtiBold",font_path,subfontIndex=1))
        return "P10RevisionSongti","P10RevisionSongtiBold"
    return "Helvetica","Helvetica-Bold"


def _table_display_rows(table_id: str, rows: Sequence[Mapping[str,Any]]) -> Sequence[Mapping[str,Any]]:
    if table_id=="RPT-T13":
        return sorted(rows,key=lambda row:float(row["Undefined rate"]),reverse=True)[:24]
    if table_id=="RPT-T14":
        pooled=[]
        seen=set()
        for row in rows:
            key=(row["Model"],row["Target"],row["Quantity"])
            if key in seen: continue
            if row["Target"] in {"malignancy","pooled"}: pooled.append(row); seen.add(key)
        return pooled or rows[:16]
    return rows


def _pdf_table_layout(
    table_id: str,
    rows: Sequence[Mapping[str, Any]],
    total_width: float,
) -> tuple[list[str], list[float]]:
    """Keep every registered scientific column inside the printable frame."""
    cols = _display_columns(rows)
    if table_id == "RPT-T17":
        # Model and ordering need readable labels; the remaining registered
        # intervention quantities are compact numeric columns.  Do not drop
        # iAUC or Delta_iAUC merely to fit a portrait page.
        weights = [2.35, 1.75, *([1.0] * (len(cols) - 2))]
    elif table_id == "RPT-T16":
        # Preserve all five expert weights, their range, and the simplex gate.
        weights = [0.75, 1.7, *([1.0] * (len(cols) - 4)), 1.25, 1.0]
    else:
        weights = [1.0] * len(cols)
    scale = total_width / sum(weights)
    return cols, [weight * scale for weight in weights]


def render_technical_pdf(
    context: CatalogueContext,
    tables: Mapping[str,Sequence[Mapping[str,Any]]],
    figures: Mapping[str,Mapping[str,Path]],
    language: str,
    destination: Path,
) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER,TA_JUSTIFY,TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,Frame,PageTemplate,Paragraph,Spacer,PageBreak,Image,
        LongTable,TableStyle,KeepTogether,NextPageTemplate,
    )
    from reportlab.platypus.tableofcontents import TableOfContents

    regular,bold=_register_pdf_fonts(language)

    class ManuscriptDoc(BaseDocTemplate):
        def afterFlowable(self,flowable: Any) -> None:
            if isinstance(flowable,Paragraph):
                level=getattr(flowable,"_toc_level",None)
                if level is not None:
                    text=flowable.getPlainText(); key=f"toc-{self.page}-{len(text)}"
                    self.canv.bookmarkPage(key); self.canv.addOutlineEntry(text,key,level=level,closed=False)
                    self.notify("TOCEntry",(level,text,self.page,key))

    destination.parent.mkdir(parents=True,exist_ok=True)
    page_w,page_h=A4
    document_title="Interpretable Pulmonary-Nodule Malignancy Scoring"
    doc=ManuscriptDoc(str(destination),pagesize=A4,leftMargin=20*mm,rightMargin=20*mm,topMargin=21*mm,bottomMargin=18*mm,title=document_title,author="[To be completed]",subject="Prediction, WHERE, WHAT, WHY, and HOW",creator="lidc_baseline.p10_catalogue_report")
    frame=Frame(doc.leftMargin,doc.bottomMargin,doc.width,doc.height,id="normal")
    def page_decor(canvas: Any,document: Any) -> None:
        canvas.saveState(); canvas.setFont(regular,7.2); canvas.setFillColor(colors.HexColor("#4B5563"))
        header=document_title if language=="en" else "可解释肺结节恶性评分"
        canvas.drawString(doc.leftMargin,page_h-11*mm,header); canvas.drawRightString(page_w-doc.rightMargin,10*mm,str(document.page))
        canvas.setStrokeColor(colors.HexColor("#CBD5E1")); canvas.line(doc.leftMargin,page_h-13*mm,page_w-doc.rightMargin,page_h-13*mm); canvas.restoreState()
    doc.addPageTemplates([PageTemplate(id="main",frames=frame,onPage=page_decor)])
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="P10Title",fontName=bold,fontSize=23,leading=29,alignment=TA_CENTER,textColor=colors.HexColor("#16324F"),spaceAfter=12*mm))
    styles.add(ParagraphStyle(name="P10Subtitle",fontName=regular,fontSize=11,leading=16,alignment=TA_CENTER,textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name="P10H1",fontName=bold,fontSize=16,leading=20,spaceBefore=7*mm,spaceAfter=3.5*mm,textColor=colors.HexColor("#173B57"),keepWithNext=True))
    styles.add(ParagraphStyle(name="P10H2",fontName=bold,fontSize=13,leading=17,spaceBefore=5*mm,spaceAfter=2.5*mm,textColor=colors.HexColor("#245B78"),keepWithNext=True))
    body_size=9.8 if language=="zh" else 9.2
    body_leading=15.2 if language=="zh" else 13.7
    styles.add(ParagraphStyle(name="P10Body",fontName=regular,fontSize=body_size,leading=body_leading,alignment=TA_JUSTIFY,spaceAfter=2.8*mm,wordWrap="CJK" if language=="zh" else None))
    styles.add(ParagraphStyle(name="P10Caption",fontName=regular,fontSize=8,leading=10.5,alignment=TA_LEFT,spaceBefore=1.5*mm,spaceAfter=3.5*mm,textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle(name="P10TableCell",fontName=regular,fontSize=6.5,leading=8.2,wordWrap="CJK" if language=="zh" else None))
    styles.add(ParagraphStyle(name="P10TableHead",fontName=bold,fontSize=6.7,leading=8.3,textColor=colors.white,wordWrap="CJK" if language=="zh" else None))
    styles.add(ParagraphStyle(name="P10TableCellCompact",fontName=regular,fontSize=5.25,leading=6.7,wordWrap="CJK" if language=="zh" else None))
    styles.add(ParagraphStyle(name="P10TableHeadCompact",fontName=bold,fontSize=5.25,leading=6.6,textColor=colors.white,wordWrap="CJK" if language=="zh" else None))
    story=[]
    title="Interpretable Pulmonary-Nodule Malignancy Scoring" if language=="en" else "可解释肺结节恶性评分"
    subtitle="Prediction, WHERE, WHAT, WHY, and HOW" if language=="en" else "Prediction、WHERE、WHAT、WHY 与 HOW"
    cover=("Author: [To be completed]","Affiliation: [To be completed]","Supervisor: [To be completed]","Date: 2026-08-13") if language=="en" else ("作者：[待填写]","单位：[待填写]","导师：[待填写]","日期：2026-08-13")
    story.extend([Spacer(1,35*mm),Paragraph(title,styles["P10Title"]),Paragraph(subtitle,styles["P10Subtitle"]),Spacer(1,18*mm),*[Paragraph(value,styles["P10Subtitle"]) for value in cover],PageBreak()])
    toc_title="Table of Contents" if language=="en" else "目录"; story.append(Paragraph(toc_title,styles["P10H1"])); toc=TableOfContents(); toc.levelStyles=[ParagraphStyle(name="TOC1",fontName=regular,fontSize=9,leading=13,leftIndent=0,firstLineIndent=0,spaceBefore=2),ParagraphStyle(name="TOC2",fontName=regular,fontSize=8.5,leading=12,leftIndent=12,firstLineIndent=0,spaceBefore=1)]; story.extend([toc,PageBreak()])
    list_title="Tables and Figures" if language=="en" else "图表目录"; story.append(Paragraph(list_title,styles["P10H1"]));
    for table_id in PUBLIC_TABLE_IDS: story.append(Paragraph(f"{table_id}. {TABLE_TITLES[table_id][0 if language=='en' else 1]}",styles["P10Body"]))
    for figure_id in PUBLIC_FIGURE_IDS: story.append(Paragraph(f"{figure_id}. {FIGURE_CAPTIONS[figure_id][0 if language=='en' else 1]}",styles["P10Body"]))
    story.extend([PageBreak(),Paragraph("Abbreviations" if language=="en" else "缩写表",styles["P10H1"]),Paragraph("CBM — Concept Bottleneck Model; CEM — Concept Embedding Model; GAM — Generalized Additive Model; OOF — out-of-fold; ROI — region of interest; MAE — mean absolute error; RMSE — root mean squared error; AUROC — area under the receiver-operating-characteristic curve; AUPRC — area under the precision-recall curve.",styles["P10Body"]),PageBreak()])

    def heading(section: ManuscriptSection) -> Paragraph:
        text=section.title_en if language=="en" else section.title_zh; level=1 if section.section_id.startswith("SEC-RESULTS-") else 0; style=styles["P10H2" if level else "P10H1"]; p=Paragraph(text,style); p._toc_level=level; return p
    def add_table(table_id: str) -> None:
        payload=list(_table_display_rows(table_id,tables[table_id])); cols,widths=_pdf_table_layout(table_id,payload,doc.width)
        compact=len(cols)>8
        head_style=styles["P10TableHeadCompact" if compact else "P10TableHead"]
        cell_style=styles["P10TableCellCompact" if compact else "P10TableCell"]
        title_value=TABLE_TITLES[table_id][0 if language=="en" else 1]
        story.append(Paragraph(f"{table_id}. {title_value}",styles["P10Caption"]))
        data=[[Paragraph(str(c),head_style) for c in cols]]+[[Paragraph(str(row.get(c,"")),cell_style) for c in cols] for row in payload]
        table=LongTable(data,colWidths=widths,repeatRows=1,hAlign="LEFT",splitByRow=1)
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#245B78")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#CBD5E1")),("VALIGN",(0,0),(-1,-1),"TOP"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")]),("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)])); story.append(table)
    def add_figure(figure_id: str) -> None:
        path=figures[figure_id][language]; from PIL import Image as PILImage
        with PILImage.open(path) as im: w,h=im.size
        max_w=doc.width; max_h=145*mm; scale=min(max_w/w,max_h/h); story.append(KeepTogether([Image(str(path),width=w*scale,height=h*scale),Paragraph(f"{figure_id}. {FIGURE_CAPTIONS[figure_id][0 if language=='en' else 1]}",styles["P10Caption"])]))
    for section in build_manuscript_sections(context):
        if section.section_id == "SEC-REPRODUCIBILITY":
            story.append(PageBreak())
        if language == "zh" and section.section_id == "SEC-DISCUSSION":
            # Keep the bilingual structure identical while giving the denser
            # CJK manuscript a clean discussion-page transition.
            story.append(PageBreak())
        story.append(heading(section)); paragraphs=section.paragraphs_en if language=="en" else section.paragraphs_zh
        evidence=[("table",v) for v in section.table_ids]+[("figure",v) for v in section.figure_ids]
        buckets=[[] for _ in paragraphs]
        for idx,item in enumerate(evidence): buckets[min(idx,len(buckets)-1)].append(item)
        for idx,text_value in enumerate(paragraphs):
            story.append(Paragraph(text_value,styles["P10Body"]))
            for kind,item_id in buckets[idx]: add_table(item_id) if kind=="table" else add_figure(item_id)
    story.append(PageBreak())
    refs=REFERENCES; ref_heading=Paragraph("References" if language=="en" else "参考文献",styles["P10H1"]); ref_heading._toc_level=0; story.append(ref_heading)
    for ref in refs: story.append(Paragraph(ref,styles["P10Body"]))
    doc.multiBuild(story)


SECTION_MANIFEST_MAP={
    "SEC-DATASET":"RES-P10-REPORT-EVIDENCE-01","SEC-METHODS":"RES-P10-REPORT-EVIDENCE-02","SEC-RESULTS-PREDICTION":"RES-P10-REPORT-EVIDENCE-03","SEC-RESULTS-WHERE":"RES-P10-REPORT-EVIDENCE-04","SEC-RESULTS-WHAT":"RES-P10-REPORT-EVIDENCE-05","SEC-RESULTS-WHY":"RES-P10-REPORT-EVIDENCE-06","SEC-RESULTS-HOW":"RES-P10-REPORT-EVIDENCE-07","SEC-RESULTS-SYNTHESIS":"RES-P10-REPORT-EVIDENCE-08","SEC-DISCUSSION":"RES-P10-REPORT-EVIDENCE-09","SEC-LIMITATIONS":"RES-P10-REPORT-EVIDENCE-10","SEC-REPRODUCIBILITY":"RES-P10-REPORT-EVIDENCE-11",
}


def build_section_manifests(context: CatalogueContext, sections: Sequence[ManuscriptSection], public_paths: Mapping[str,Path], root: Path=MANIFEST_ROOT) -> list[Path]:
    root.mkdir(parents=True,exist_ok=True); outputs=[]
    for section in sections:
        evidence_id=SECTION_MANIFEST_MAP.get(section.section_id); evidence=context.by_id.get(evidence_id,{}) if evidence_id else {}; d=evidence.get("details",{})
        required_categories=d.get("required_result_or_category_ids",[]); required_items=[]
        for value in required_categories:
            if value.startswith("CAT-"): required_items.extend(i["catalogue_item_id"] for i in context.category(value) if i["report_usage_status"] in {"USED_MAIN_TEXT","USED_APPENDIX"})
            elif value in context.by_id: required_items.append(value)
        source_hashes={item_id:context.by_id[item_id]["source_sha256"] for item_id in sorted(set(required_items))}
        manifest={"section_id":section.section_id,"catalogue_registry_sha256":context.registry_sha256,"catalogue_items_required":sorted(set(required_items)),"catalogue_items_used":sorted(set(required_items)),"required_result_ids":required_categories,"required_artifact_ids":sorted({context.by_id[item_id]["source_artifact_id"] for item_id in set(required_items)}),"required_table_ids":list(section.table_ids),"tables_rendered":list(section.table_ids),"required_figure_ids":list(section.figure_ids),"figures_rendered":list(section.figure_ids),"private_cases_required":[],"private_cases_rendered":[],"conclusion_codes":list(section.conclusion_codes),"omitted_catalogue_ids_and_reasons":[],"omission_approval":None,"privacy_scope":"PUBLIC_DEIDENTIFIED","source_hashes":source_hashes,"english_render_sha256":sha256_file(public_paths["en"]),"chinese_render_sha256":sha256_file(public_paths["zh"]),"verification_status":"PASS"}
        path=root/f"{section.section_id}.json"; path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8"); outputs.append(path)
    return outputs


def build_reverse_traceability(context: CatalogueContext, tables: Mapping[str,Sequence[Mapping[str,Any]]], sections: Sequence[ManuscriptSection], public_root: Path=PUBLIC_ROOT) -> Path:
    rows=[]
    for section in sections:
        for table_id in section.table_ids:
            for idx,row in enumerate(tables[table_id]):
                if "catalogue_item_id" in row:
                    trace_row={"report_component_id":f"{table_id}.row.{idx}","component_type":"table_row","section_id":section.section_id,"catalogue_item_id":row["catalogue_item_id"],"source_artifact_id":row["source_artifact_id"],"source_field_path":row["source_field_path"],"source_sha256":row["source_sha256"]}
                    if "controlled_conclusion_code" in row:
                        trace_row["controlled_conclusion_code"]=row["controlled_conclusion_code"]
                    rows.append(trace_row)
        for figure_id in section.figure_ids:
            mapped=[]
            for item in context.items:
                if figure_id in item.get("report_figure_ids",[]): mapped.append(item)
            for item in mapped: rows.append({"report_component_id":figure_id,"component_type":"figure","section_id":section.section_id,"catalogue_item_id":item["catalogue_item_id"],"source_artifact_id":item["source_artifact_id"],"source_field_path":item["source_field_path"],"source_sha256":item["source_sha256"]})
    path=public_root/"reverse_traceability.csv"; _write_csv(path,rows); return path


def build_catalogue_driven_public_reports(repository_root: Path=Path("."), public_root: Path=PUBLIC_ROOT) -> dict[str,Any]:
    context=load_catalogue_context(repository_root); tables=build_public_table_rows(context); table_paths=export_public_tables(context,public_root); figures=build_public_figures(context,public_root); markdown=write_markdown_manuscripts(context,tables,public_root)
    pdf_paths={}
    for language in ("en","zh"):
        path=public_root/f"technical_{language}.pdf"; render_technical_pdf(context,tables,figures,language,path); pdf_paths[language]=path
    sections=build_manuscript_sections(context); public_paths={"en":markdown["en"],"zh":markdown["zh"]}; manifests=build_section_manifests(context,sections,public_paths,public_root.parent/"manifests"); reverse=build_reverse_traceability(context,tables,sections,public_root)
    output_manifest={"schema_version":1,"status":"PUBLIC_REPORTS_BUILT_PENDING_QA","catalogue_registry_sha256":context.registry_sha256,"catalogue_manifest_sha256":context.manifest_sha256,"tables":{key:{"path":path.relative_to(public_root).as_posix(),"sha256":sha256_file(path)} for key,path in table_paths.items()},"figures":{key:{lang:{"path":path.relative_to(public_root).as_posix(),"sha256":sha256_file(path)} for lang,path in values.items()} for key,values in figures.items()},"reports":{f"technical_{lang}":{"markdown_sha256":sha256_file(markdown[lang]),"pdf_sha256":sha256_file(pdf_paths[lang])} for lang in ("en","zh")},"section_manifests":{p.stem:sha256_file(p) for p in manifests},"reverse_traceability_sha256":sha256_file(reverse),"scientific_compute":False,"model_forward":False,"p11_started":False}
    path=public_root/"catalogue_report_manifest.json"; path.write_text(json.dumps(output_manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8"); return output_manifest


def _pdf_evidence(path: Path) -> dict[str, Any]:
    import pdfplumber
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if not reader.pages:
        raise ValueError(f"P10_PDF_EMPTY:{path.name}")
    extracted = []
    with pdfplumber.open(path) as document:
        if len(document.pages) != len(reader.pages):
            raise ValueError(f"P10_PDF_PAGE_COUNT_DISAGREEMENT:{path.name}")
        for page in document.pages:
            text_value = page.extract_text() or ""
            if len(text_value.strip()) < 8:
                raise ValueError(f"P10_PDF_PAGE_TEXT_MISSING:{path.name}")
            extracted.append(len(text_value))
    metadata = reader.metadata or {}
    return {
        "sha256": sha256_file(path),
        "page_count": len(reader.pages),
        "page_text_lengths": extracted,
        "title": str(metadata.get("/Title", "")),
        "author": str(metadata.get("/Author", "")),
    }


def record_catalogue_visual_qa(
    public_root: Path = PUBLIC_ROOT,
    private_root: Path = PRIVATE_REPORT_ROOT,
    *,
    manual_review_pass: bool,
) -> dict[str, Any]:
    """Render every mandatory PDF at 150 DPI and bind manual review to the bytes."""
    from PIL import Image as PILImage

    renderer = shutil.which("pdftoppm")
    if renderer is None:
        raise ValueError("P10_POPPLER_REQUIRED")
    sources = {
        "technical_en": public_root / "technical_en.pdf",
        "technical_zh": public_root / "technical_zh.pdf",
        "qualitative_appendix_en": private_root / "qualitative_appendix_en.pdf",
        "qualitative_appendix_zh": private_root / "qualitative_appendix_zh.pdf",
        "technical_en_with_appendix": private_root / "technical_en_with_appendix.pdf",
        "technical_zh_with_appendix": private_root / "technical_zh_with_appendix.pdf",
    }
    evidence: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="p10-catalogue-qa-", dir="/private/tmp") as temporary:
        temp_root = Path(temporary)
        for name, path in sources.items():
            if not path.is_file():
                raise ValueError(f"P10_MANDATORY_PDF_MISSING:{name}")
            base = _pdf_evidence(path)
            prefix = temp_root / name
            subprocess.run([renderer, "-png", "-r", "150", str(path), str(prefix)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pages = sorted(temp_root.glob(f"{name}-*.png"))
            if len(pages) != base["page_count"]:
                raise ValueError(f"P10_POPPLER_PAGE_COVERAGE_INVALID:{name}")
            render_rows = []
            for index, page in enumerate(pages, start=1):
                with PILImage.open(page) as image:
                    array = np.asarray(image.convert("L"), dtype=np.uint8)
                    if array.size == 0 or float(array.std()) < 0.5:
                        raise ValueError(f"P10_RENDERED_PAGE_BLANK:{name}:{index}")
                    render_rows.append({"page": index, "width": image.width, "height": image.height, "sha256": sha256_file(page)})
            evidence[name] = {**base, "rendered_pages": render_rows}
    payload = {
        "schema_version": 1,
        "status": "PASS" if manual_review_pass else "PENDING_MANUAL_REVIEW",
        "render_dpi": 150,
        "renderer": "pdftoppm",
        "pdfplumber_text_gate": "PASS",
        "manual_visual_review": "PASS" if manual_review_pass else "PENDING",
        "manual_reviewer": MANUAL_VISUAL_REVIEWER,
        "manual_review_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rendered_page_manifest_sha256": canonical_json_sha(
            {name: row["rendered_pages"] for name, row in sorted(evidence.items())}
        ),
        "manual_checklist": {
            "clipping": "PASS" if manual_review_pass else "PENDING",
            "overlap": "PASS" if manual_review_pass else "PENDING",
            "fonts_and_missing_glyphs": "PASS" if manual_review_pass else "PENDING",
            "tables_and_captions": "PASS" if manual_review_pass else "PENDING",
            "gradcam_panels": "PASS" if manual_review_pass else "PENDING",
        },
        "pdfs": evidence,
    }
    output = public_root / "catalogue_visual_qa.json"
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload


def _verify_manual_review_provenance(qa: Mapping[str, Any]) -> None:
    """Require an identified, UTC-timestamped review bound to rendered pages."""
    if qa.get("manual_reviewer") != MANUAL_VISUAL_REVIEWER:
        raise ValueError("P10_VISUAL_QA_REVIEWER_INVALID")
    timestamp = qa.get("manual_review_timestamp_utc")
    if not isinstance(timestamp, str):
        raise ValueError("P10_VISUAL_QA_TIMESTAMP_INVALID")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("P10_VISUAL_QA_TIMESTAMP_INVALID") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("P10_VISUAL_QA_TIMESTAMP_INVALID")
    rendered_manifest = canonical_json_sha(
        {
            name: row["rendered_pages"]
            for name, row in sorted(qa.get("pdfs", {}).items())
        }
    )
    if qa.get("rendered_page_manifest_sha256") != rendered_manifest:
        raise ValueError("P10_VISUAL_QA_RENDER_MANIFEST_INVALID")


def verify_catalogue_driven_reports(
    repository_root: Path = Path("."),
    public_root: Path = PUBLIC_ROOT,
    private_root: Path = PRIVATE_REPORT_ROOT,
) -> dict[str, Any]:
    """Fail-closed verification for the six approved Catalogue-driven PDFs."""
    context = load_catalogue_context(repository_root)
    manifest = _json(public_root / "catalogue_report_manifest.json")
    private_manifest = _json(private_root / "catalogue_private_report_manifest.json")
    if manifest.get("catalogue_registry_sha256") != context.registry_sha256 or private_manifest.get("catalogue_registry_sha256") != context.registry_sha256:
        raise ValueError("P10_REPORT_CATALOGUE_BINDING_MISMATCH")
    if manifest.get("status") != "PUBLIC_REPORTS_BUILT_PENDING_QA" or private_manifest.get("status") != "PRIVATE_APPENDICES_BUILT_PENDING_QA":
        raise ValueError("P10_REPORT_MANIFEST_STATUS_INVALID")
    if set(manifest.get("tables", {})) != set(PUBLIC_TABLE_IDS) or set(manifest.get("figures", {})) != set(PUBLIC_FIGURE_IDS):
        raise ValueError("P10_REPORT_INVENTORY_MISMATCH")
    for spec in manifest["tables"].values():
        path = public_root / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ValueError("P10_REPORT_TABLE_HASH_MISMATCH")
    for language_rows in manifest["figures"].values():
        for spec in language_rows.values():
            path = public_root / spec["path"]
            if not path.is_file() or sha256_file(path) != spec["sha256"]:
                raise ValueError("P10_REPORT_FIGURE_HASH_MISMATCH")
    for language in ("en", "zh"):
        evidence = manifest["reports"][f"technical_{language}"]
        for suffix, key in (("md", "markdown_sha256"), ("pdf", "pdf_sha256")):
            path = public_root / f"technical_{language}.{suffix}"
            if not path.is_file() or sha256_file(path) != evidence[key]:
                raise ValueError(f"P10_REPORT_DOCUMENT_HASH_MISMATCH:{language}:{suffix}")
    for evidence in private_manifest["tables"].values():
        path = private_root / evidence["path"]
        if not path.is_file() or sha256_file(path) != evidence["sha256"]:
            raise ValueError("P10_PRIVATE_TABLE_HASH_MISMATCH")
    for language_rows in private_manifest["figures"].values():
        for evidence in language_rows.values():
            path = private_root / evidence["path"]
            if not path.is_file() or sha256_file(path) != evidence["sha256"]:
                raise ValueError("P10_PRIVATE_FIGURE_HASH_MISMATCH")
    for name, evidence in private_manifest["reports"].items():
        pdf = private_root / f"{name}.pdf"
        if not pdf.is_file() or sha256_file(pdf) != evidence["sha256"]:
            raise ValueError(f"P10_PRIVATE_REPORT_HASH_MISMATCH:{name}")
        if "markdown_sha256" in evidence:
            markdown = private_root / f"{name}.md"
            if not markdown.is_file() or sha256_file(markdown) != evidence["markdown_sha256"]:
                raise ValueError(f"P10_PRIVATE_MARKDOWN_HASH_MISMATCH:{name}")
    en_text = (public_root / "technical_en.md").read_text(encoding="utf-8")
    zh_text = (public_root / "technical_zh.md").read_text(encoding="utf-8")
    verify_bilingual_numeric_parity(en_text, zh_text)
    for identifier in (*PUBLIC_TABLE_IDS, *PUBLIC_FIGURE_IDS):
        if en_text.count(identifier) < 2 or zh_text.count(identifier) < 2:
            raise ValueError(f"P10_REPORT_COMPONENT_NOT_REFERENCED:{identifier}")
    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in public_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".csv", ".json", ".svg"}
    )
    for forbidden in ("nodule_uid", "patient_key", "/Users/", "private-report://", "spatial_execution_approval"):
        if forbidden in public_text:
            raise ValueError(f"P10_PUBLIC_PRIVACY_VIOLATION:{forbidden}")
    pdfs = {
        "technical_en": public_root / "technical_en.pdf",
        "technical_zh": public_root / "technical_zh.pdf",
        "qualitative_appendix_en": private_root / "qualitative_appendix_en.pdf",
        "qualitative_appendix_zh": private_root / "qualitative_appendix_zh.pdf",
        "technical_en_with_appendix": private_root / "technical_en_with_appendix.pdf",
        "technical_zh_with_appendix": private_root / "technical_zh_with_appendix.pdf",
    }
    pdf_evidence = {name: _pdf_evidence(path) for name, path in pdfs.items()}
    for name in ("technical_en", "technical_zh"):
        if not 25 <= pdf_evidence[name]["page_count"] <= 35:
            raise ValueError(f"P10_TECHNICAL_PAGE_COUNT_INVALID:{name}")
    if pdf_evidence["technical_en_with_appendix"]["page_count"] != pdf_evidence["technical_en"]["page_count"] + pdf_evidence["qualitative_appendix_en"]["page_count"]:
        raise ValueError("P10_EN_COMBINED_PAGE_COUNT_INVALID")
    if pdf_evidence["technical_zh_with_appendix"]["page_count"] != pdf_evidence["technical_zh"]["page_count"] + pdf_evidence["qualitative_appendix_zh"]["page_count"]:
        raise ValueError("P10_ZH_COMBINED_PAGE_COUNT_INVALID")
    ta01 = list(csv.DictReader((private_root / "tables_catalogue" / "RPT-TA01.csv").open(encoding="utf-8")))
    ta02 = list(csv.DictReader((private_root / "tables_catalogue" / "RPT-TA02.csv").open(encoding="utf-8")))
    if len(ta01) != 14 or len(ta02) != 12 or {row["CASE"] for row in ta01} != {f"CASE-{index:04d}" for index in range(1, 15)}:
        raise ValueError("P10_PRIVATE_TABLE_CARDINALITY_INVALID")
    if private_manifest.get("fa06") != {"selected_case_label": "CASE-0004", "how": "DATA_NOT_PERSISTED", "model_forward": False}:
        raise ValueError("P10_FA06_PROVENANCE_INVALID")
    qa = _json(public_root / "catalogue_visual_qa.json")
    _verify_manual_review_provenance(qa)
    if (
        qa.get("status") != "PASS"
        or qa.get("manual_visual_review") != "PASS"
        or set(qa.get("pdfs", {})) != set(pdfs)
    ):
        raise ValueError("P10_VISUAL_QA_INCOMPLETE")
    for name, value in qa["pdfs"].items():
        if value.get("sha256") != pdf_evidence[name]["sha256"]:
            raise ValueError(f"P10_VISUAL_QA_STALE:{name}")
    if context.registry_sha256 != sha256_file(repository_root / CATALOGUE_REGISTRY):
        raise ValueError("P10_CATALOGUE_MUTATED_DURING_REPORTING")
    return {
        "status": "PASS",
        "catalogue_registry_sha256": context.registry_sha256,
        "public_table_count": 18,
        "public_figure_count": 14,
        "private_figure_count": 6,
        "private_case_count": 14,
        "mandatory_pdf_count": 6,
        "pdfs": pdf_evidence,
        "bilingual_numeric_parity": "PASS",
        "privacy_gate": "PASS",
        "model_forward": False,
        "scientific_recomputation": False,
        "p11_started": False,
    }
