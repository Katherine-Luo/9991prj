"""Catalogue-driven private qualitative appendix renderer.

The renderer is deliberately presentation-only: it consumes the approved
Catalogue, frozen OOF rows, frozen Grad-CAM shards, frozen ROI arrays, and the
original read-only DICOM series.  It never executes a model or alters a
scientific artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image as PILImage

from lidc_baseline.p10_catalogue_report import (
    CATEGORICAL_CONCEPTS,
    CONCEPTS,
    CONTINUOUS_CONCEPTS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    PRIVATE_ARCHIVE,
    PRIVATE_REPORT_ROOT,
    PUBLIC_ROOT,
    CatalogueContext,
    load_catalogue_context,
    sha256_file,
)
from lidc_baseline.p10_private_appendix import _find_map_record, _load_roi, _merge_pdfs


REPOSITORY_MANIFEST = Path("artifacts/baseline_v2/manifests/nodules.parquet")
DICOM_ROOT = Path("/Users/katherine/Desktop/lidc_data/manifest-1600709154662/LIDC-IDRI")
ROI_ROOT = Path("artifacts/baseline_v2/rois")
OOF_FILENAMES = {
    "blackbox": "blackbox_oof_predictions.parquet",
    "standard_cbm": "standard_cbm_oof_predictions.parquet",
    "mixed_cem": "cem_oof_predictions.parquet",
    "learned_softmax_gam": "gam_oof_predictions.parquet",
}
CATEGORY_LABELS = {
    "internalStructure": ("Soft tissue", "Fluid", "Fat", "Air"),
    "calcification": ("Popcorn", "Laminated", "Solid", "Non-central", "Central", "Absent"),
}
PRIVATE_FIGURE_IDS = tuple(f"RPT-FA{i:02d}" for i in range(1, 7))


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    result = np.asarray(value, dtype=float).reshape(-1)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("P10_PRIVATE_VECTOR_INVALID")
    return result


def _parse_scalar(value: Any) -> float:
    vector = _parse_vector(value)
    if vector.size != 1:
        raise ValueError("P10_PRIVATE_SCALAR_INVALID")
    return float(vector[0])


def _configure_font(language: str) -> None:
    if language == "zh":
        source = Path("/System/Library/Fonts/Supplemental/Songti.ttc")
        if not source.is_file():
            raise ValueError("P10_SONGTI_SOURCE_MISSING")
        plt.rcParams.update(
            {"font.family": font_manager.FontProperties(fname=str(source)).get_name(), "axes.unicode_minus": False}
        )
    else:
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": True})


def _load_cases(context: CatalogueContext) -> list[dict[str, Any]]:
    index_path = PRIVATE_REPORT_ROOT / "private_case_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {str(row["case_label"]): dict(row) for row in index["cases"]}
    output: list[dict[str, Any]] = []
    for item in sorted(context.category("CAT-Q"), key=lambda row: row["details"]["case_label"]):
        label = str(item["details"]["case_label"])
        if label not in indexed:
            raise ValueError(f"P10_PRIVATE_CASE_INDEX_MISSING:{label}")
        row = indexed[label]
        if row["model"] != item["model"] or int(row["fold_index"]) != int(item["fold"]):
            raise ValueError(f"P10_PRIVATE_CASE_CATALOGUE_MISMATCH:{label}")
        row["catalogue_item_id"] = item["catalogue_item_id"]
        row["catalogue_details"] = dict(item["details"])
        row["report_figure_ids"] = list(item["report_figure_ids"])
        output.append(row)
    if len(output) != 14 or len({row["case_label"] for row in output}) != 14:
        raise ValueError("P10_PRIVATE_CASE_CARDINALITY_INVALID")
    return output


def _load_frames(repository_root: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    oof_root = PRIVATE_ARCHIVE / "p9" / "canonical_oof"
    frames = {model: pd.read_parquet(oof_root / filename) for model, filename in OOF_FILENAMES.items()}
    manifest = pd.read_parquet(repository_root / REPOSITORY_MANIFEST)
    return frames, manifest


def _contribution_column(model: str, concept: str) -> str:
    suffix = "rating_point_contribution" if model == "standard_cbm" else "rating_contribution"
    return f"{concept}_{suffix}"


def _case_science(case: Mapping[str, Any], oof: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    uid = str(case["nodule_uid"])
    rows = oof[oof["nodule_uid"].astype(str) == uid]
    source_rows = manifest[manifest["nodule_uid"].astype(str) == uid]
    if len(rows) != 1 or len(source_rows) != 1:
        raise ValueError(f"P10_PRIVATE_CASE_SOURCE_CARDINALITY:{case['case_label']}")
    row = rows.iloc[0]
    source = source_rows.iloc[0]
    malignancy_prediction = float(row["malignancy_score_1_to_5"])
    values: dict[str, Any] = {
        "malignancy_prediction": malignancy_prediction,
        "malignancy_target": float(source["mean_malignancy"]),
        "absolute_error": abs(malignancy_prediction - float(source["mean_malignancy"])),
        "concepts": {},
        "contributions": {},
    }
    if str(case["model"]) == "blackbox":
        return values
    for concept in CONTINUOUS_CONCEPTS:
        values["concepts"][concept] = {
            "prediction": _parse_scalar(row[f"{concept}_activated_prediction"]),
            "target": float(source[f"{concept}_target"]),
            "target_kind": "reader_mean_normalized",
        }
    for concept in CATEGORICAL_CONCEPTS:
        prediction = _parse_vector(row[f"{concept}_activated_prediction"])
        votes = np.asarray(source[f"{concept}_vote_distribution"], dtype=float).reshape(-1)
        if prediction.size != votes.size or not np.isfinite(votes).all() or not np.isclose(votes.sum(), 1.0):
            raise ValueError("P10_PRIVATE_CATEGORICAL_TARGET_INVALID")
        pred_class = int(np.argmax(prediction))
        labels = CATEGORY_LABELS[concept]
        modal_value = source[f"{concept}_modal_class"]
        modal_label = "Reader tie" if pd.isna(modal_value) else labels[int(modal_value) - 1]
        values["concepts"][concept] = {
            "prediction": prediction.tolist(),
            "predicted_label": labels[pred_class],
            "target_vote_distribution": votes.tolist(),
            "target_modal_label": modal_label,
            "target_kind": "full_reader_vote_distribution",
        }
    for concept in CONCEPTS:
        column = _contribution_column(str(case["model"]), concept)
        values["contributions"][concept] = float(row[column])
    return values


def build_private_tables(cases: Sequence[Mapping[str, Any]], science: Mapping[str, Mapping[str, Any]], destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    ta01_rows = []
    ta02_rows = []
    for case in cases:
        label = str(case["case_label"])
        values = science[label]
        ta01_rows.append(
            {
                "CASE": label,
                "Model": MODEL_LABELS[str(case["model"])],
                "Role": case["catalogue_details"]["case_role"],
                "Fold": case["fold_index"],
                "Target": case["target"],
                "Malignancy prediction": f"{values['malignancy_prediction']:.3f}",
                "Radiologist-mean target": f"{values['malignancy_target']:.3f}",
                "Absolute error": f"{values['absolute_error']:.3f}",
                "Map status": case["map_status"],
            }
        )
        if str(case["model"]) == "blackbox":
            continue
        row: dict[str, Any] = {
            "CASE": label,
            "Model": MODEL_LABELS[str(case["model"])],
            "Malignancy Pred": f"{values['malignancy_prediction']:.3f}",
            "Malignancy GT": f"{values['malignancy_target']:.3f}",
        }
        for concept in CONTINUOUS_CONCEPTS:
            detail = values["concepts"][concept]
            row[f"{concept} Pred/GT"] = f"{detail['prediction']:.3f} / {detail['target']:.3f}"
        for concept in CATEGORICAL_CONCEPTS:
            detail = values["concepts"][concept]
            votes = ",".join(f"{value:.3f}" for value in detail["target_vote_distribution"])
            row[f"{concept} Pred/GT"] = f"{detail['predicted_label']} / {detail['target_modal_label']} [votes={votes}]"
        ta02_rows.append(row)

    paths: dict[str, Path] = {}
    for table_id, rows in (("RPT-TA01", ta01_rows), ("RPT-TA02", ta02_rows)):
        csv_path = destination / f"{table_id}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        md_path = destination / f"{table_id}.md"
        headers = list(rows[0])
        lines = [f"# {table_id}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        lines.extend("| " + " | ".join(str(row[h]) for h in headers) + " |" for row in rows)
        if table_id == "RPT-TA02":
            lines.extend(["", "Categorical modal labels are display aids. The frozen ground-truth target is the full reader vote distribution shown in each cell."])
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[f"{table_id}_csv"] = csv_path
        paths[f"{table_id}_md"] = md_path
    return paths


@lru_cache(maxsize=32)
def _series_files(patient_id: str, series_uid: str) -> tuple[Path, ...]:
    patient_root = DICOM_ROOT / patient_id
    if not patient_root.is_dir():
        raise ValueError("P10_PRIVATE_DICOM_PATIENT_MISSING")
    rows: list[tuple[float, int, Path]] = []
    for path in patient_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=["SeriesInstanceUID", "ImagePositionPatient", "InstanceNumber"],
            )
        except Exception:
            continue
        if str(getattr(ds, "SeriesInstanceUID", "")) != series_uid:
            continue
        position = float(ds.ImagePositionPatient[2]) if "ImagePositionPatient" in ds else float(getattr(ds, "InstanceNumber", 0))
        rows.append((position, int(getattr(ds, "InstanceNumber", 0)), path))
    rows.sort(key=lambda value: (value[0], value[1], value[2].as_posix()))
    paths = tuple(row[2] for row in rows)
    if not paths:
        raise ValueError("P10_PRIVATE_DICOM_SERIES_MISSING")
    return paths


def _full_ct_slice(source: pd.Series, z_index: int) -> np.ndarray:
    paths = _series_files(str(source["patient_id"]), str(source["series_instance_uid"]))
    if z_index < 0 or z_index >= len(paths):
        raise ValueError("P10_PRIVATE_CONTEXT_Z_OUT_OF_RANGE")
    ds = pydicom.dcmread(paths[z_index])
    pixels = ds.pixel_array.astype(np.float32)
    hu = pixels * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if not np.isfinite(hu).all():
        raise ValueError("P10_PRIVATE_CT_NONFINITE")
    return np.clip((hu + 1000.0) / 1400.0, 0.0, 1.0)


def _resize_float(values: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    image = PILImage.fromarray(np.asarray(values, dtype=np.float32), mode="F")
    resized = image.resize((shape_hw[1], shape_hw[0]), resample=PILImage.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _mapped_map_slice(heatmap: np.ndarray, bbox: Sequence[Sequence[int]], context_z: int) -> np.ndarray:
    z0, z1 = (int(value) for value in bbox[0])
    fraction = (context_z - z0) / max(z1 - z0 - 1, 1)
    index = int(np.clip(round(fraction * 63), 0, 63))
    return heatmap[index]


def _display_normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    if maximum <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return np.asarray(values / maximum, dtype=np.float32)


def _case_panel(
    case: Mapping[str, Any],
    values: Mapping[str, Any],
    source: pd.Series,
    repository_root: Path,
    destination: Path,
    language: str,
) -> Path:
    _configure_font(language)
    details = case["catalogue_details"]
    bbox = details["roi_bbox_dhw"]
    z_index = int(details["frozen_context_z_index"])
    ct = _full_ct_slice(source, z_index)
    roi = _load_roi(str(case["nodule_uid"]), repository_root / ROI_ROOT)
    heatmap, status = _find_map_record(PRIVATE_ARCHIVE, case)
    if status != str(case["map_status"]):
        raise ValueError("P10_PRIVATE_MAP_STATUS_MISMATCH")
    map_slice = _mapped_map_slice(heatmap, bbox, z_index)
    display_map = _display_normalize(map_slice)
    y0, y1 = (int(value) for value in bbox[1])
    x0, x1 = (int(value) for value in bbox[2])
    if not (0 <= y0 < y1 <= ct.shape[0] and 0 <= x0 < x1 <= ct.shape[1]):
        raise ValueError("P10_PRIVATE_ROI_BBOX_INVALID")
    full_map = np.zeros_like(ct, dtype=np.float32)
    full_map[y0:y1, x0:x1] = _resize_float(display_map, (y1 - y0, x1 - x0))
    roi_index = int(np.clip(round((z_index - int(bbox[0][0])) / max(int(bbox[0][1]) - int(bbox[0][0]) - 1, 1) * 63), 0, 63))
    roi_slice = roi[roi_index]

    figure, axes = plt.subplots(1, 5, figsize=(17, 4.25), dpi=180)
    axes[0].imshow(ct, cmap="gray", vmin=0, vmax=1)
    axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#00E5FF", linewidth=1.5))
    axes[0].set_title("完整 CT + ROI 框" if language == "zh" else "Full CT + ROI box")
    axes[1].imshow(ct, cmap="gray", vmin=0, vmax=1)
    if status == "valid":
        axes[1].imshow(np.ma.masked_where(full_map <= 0, full_map), cmap="magma", alpha=0.58, vmin=0, vmax=1)
    axes[1].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor="#00E5FF", linewidth=1.2))
    axes[1].set_title("全切片叠加" if language == "zh" else "Full-slice overlay")
    axes[2].imshow(roi_slice, cmap="gray")
    axes[2].set_title("64³ 局部输入" if language == "zh" else "64³ model-input ROI")
    axes[3].imshow(roi_slice, cmap="gray")
    if status == "valid":
        axes[3].imshow(display_map, cmap="magma", alpha=0.58, vmin=0, vmax=1)
    axes[3].set_title(("ROI + Grad-CAM" if status == "valid" else "全零 Grad-CAM") if language == "zh" else ("ROI + Grad-CAM" if status == "valid" else "Zero Grad-CAM"))
    axes[4].axis("off")
    lines = [
        MODEL_LABELS[str(case["model"])],
        f"Prediction: {values['malignancy_prediction']:.3f}",
        f"Reader mean: {values['malignancy_target']:.3f}",
        f"|error|: {values['absolute_error']:.3f}",
        f"Target: {case['target']}",
        f"Map: {status}",
    ]
    if values["contributions"]:
        ranked = sorted(values["contributions"].items(), key=lambda item: abs(item[1]), reverse=True)[:4]
        lines.extend(["", "Top signed contributions:", *[f"{name}: {value:+.3f}" for name, value in ranked]])
    axes[4].text(0.02, 0.97, "\n".join(lines), va="top", fontsize=9.2, linespacing=1.35)
    for axis in axes[:4]:
        axis.axis("off")
    title = f"{case['case_label']} — {case['target']} — {status}"
    figure.suptitle(title, fontsize=14, fontweight="bold")
    note = ("显示叠加仅作可视化归一化；定量忠实度使用未归一化 raw FP32 map。ROI 是 64³ 局部模型输入，并非完整 CT 切片。" if language == "zh" else "Display overlays use visualization-only normalization; quantitative faithfulness used the unnormalized raw FP32 map. The ROI is the 64³ local model input, not a full CT slice.")
    figure.text(0.5, 0.01, note, ha="center", fontsize=7.8)
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Title": str(case["case_label"]), "Author": "P10"}, bbox_inches="tight")
    plt.close(figure)
    return destination


def _group_figure(figure_id: str, panel_paths: Sequence[Path], destination: Path, language: str) -> Path:
    _configure_font(language)
    count = len(panel_paths)
    cols = 2 if count <= 4 else 3
    rows = math.ceil(count / cols)
    figure, axes = plt.subplots(rows, cols, figsize=(16, 4.3 * rows), dpi=150)
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis, path in zip(axes_array, panel_paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
    for axis in axes_array[count:]:
        axis.axis("off")
    title_map = {
        "RPT-FA01": ("Representative cases", "代表性病例"),
        "RPT-FA02": ("Maximum-error failures", "最大误差失败病例"),
        "RPT-FA03": ("Case-level concept contributions", "病例级概念贡献"),
        "RPT-FA04": ("Occlusion error-worsening cases (not concept intervention)", "遮挡误差恶化病例（并非概念干预）"),
        "RPT-FA05": ("Undefined zero-map limitations", "未定义全零图限制"),
    }
    figure.suptitle(f"{figure_id} — {title_map[figure_id][0 if language == 'en' else 1]}", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Title": figure_id, "Author": "P10"}, bbox_inches="tight")
    plt.close(figure)
    return destination


def _fa06_figure(case: Mapping[str, Any], values: Mapping[str, Any], panel_path: Path, destination: Path, language: str) -> Path:
    _configure_font(language)
    figure = plt.figure(figsize=(16, 7), dpi=170)
    grid = figure.add_gridspec(2, 5, height_ratios=[0.18, 0.82], wspace=0.18)
    headers = ["Prediction", "WHERE", "WHAT", "WHY", "HOW"]
    for index, header in enumerate(headers):
        axis = figure.add_subplot(grid[0, index]); axis.axis("off")
        axis.text(0.5, 0.5, header, ha="center", va="center", fontsize=13, fontweight="bold", color="#173B57")
        if index < 4:
            axis.annotate("", xy=(1.08, 0.5), xytext=(0.92, 0.5), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.5})
    image = plt.imread(panel_path)
    height, width = image.shape[:2]
    prediction = figure.add_subplot(grid[1, 0]); prediction.axis("off")
    prediction.imshow(image[int(0.10*height):int(0.90*height), :int(0.20*width)])
    prediction.set_title(f"Pred {values['malignancy_prediction']:.3f}\nReader mean {values['malignancy_target']:.3f}",fontsize=9)
    where = figure.add_subplot(grid[1, 1]); where.axis("off")
    where.imshow(image[int(0.10*height):int(0.90*height), int(0.20*width):int(0.40*width)])
    where.set_title(f"{case['target']} Grad-CAM",fontsize=9)
    what = figure.add_subplot(grid[1, 2]); what.axis("off")
    concept_lines=[]
    for concept in CONTINUOUS_CONCEPTS:
        detail=values["concepts"][concept]
        concept_lines.append(f"{concept}: {detail['prediction']:.2f}/{detail['target']:.2f}")
    for concept in CATEGORICAL_CONCEPTS:
        detail=values["concepts"][concept]
        concept_lines.append(f"{concept}: {detail['predicted_label']} / {detail['target_modal_label']}")
    what.text(0.01,0.98,"\n".join(concept_lines),va="top",fontsize=8.6,linespacing=1.32)
    why = figure.add_subplot(grid[1, 3])
    contribution_names=list(values["contributions"])
    contribution_values=[values["contributions"][name] for name in contribution_names]
    order=np.argsort(np.abs(contribution_values))
    why.barh(np.arange(len(order)),np.asarray(contribution_values)[order],color=["#C44E52" if contribution_values[index]>0 else "#4C78A8" for index in order])
    why.set_yticks(np.arange(len(order)),[contribution_names[index] for index in order],fontsize=7)
    why.axvline(0,color="black",lw=.8); why.grid(axis="x",alpha=.2); why.tick_params(axis="x",labelsize=7)
    how = figure.add_subplot(grid[1, 4]); how.axis("off"); how.set_facecolor("#F1F5F9")
    message = ("病例级干预证据\nDATA_NOT_PERSISTED\n未重新计算" if language == "zh" else "Case-level intervention evidence\nDATA_NOT_PERSISTED\nNot recomputed")
    how.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#7F1D1D", bbox={"boxstyle": "round", "facecolor": "#FEE2E2", "edgecolor": "#B91C1C"})
    figure.suptitle(f"RPT-FA06 — {case['case_label']} — Integrated Prediction–WHERE–WHAT–WHY–HOW", fontsize=15, fontweight="bold")
    figure.text(0.5, 0.02, ("HOW 缺失被显式保留；未运行新 forward 或 intervention。" if language == "zh" else "The unavailable HOW component is explicit; no new forward or intervention was run."), ha="center", fontsize=9)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.09, top=0.88, wspace=0.30)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, metadata={"Title": "RPT-FA06", "Author": "P10"}, bbox_inches="tight")
    plt.close(figure)
    return destination


def _private_markdown(cases: Sequence[Mapping[str, Any]], science: Mapping[str, Mapping[str, Any]], figures: Mapping[str, Path], language: str, destination: Path) -> Path:
    zh = language == "zh"
    lines = ["# " + ("私有定性附录" if zh else "Private qualitative appendix"), "", ("本附录完全读取冻结 OOF、Grad-CAM、ROI 与 CT provenance；未运行模型 forward。" if zh else "This appendix reads only frozen OOF, Grad-CAM, ROI, and CT provenance; no model forward was run."), "", ("64³ ROI 是模型使用的局部输入，不是完整 CT 切片。显示叠加可作可视化归一化；所有定量忠实度分析仍基于未归一化 raw FP32 Grad-CAM maps。" if zh else "The 64³ ROI is the local model input, not a full CT slice. Display overlays may be normalized for visualization; every quantitative faithfulness result used the original unnormalized raw FP32 Grad-CAM maps."), ""]
    for figure_id in PRIVATE_FIGURE_IDS:
        lines.extend([f"## {figure_id}", "", f"![{figure_id}](figures_catalogue/{figures[figure_id].name})", ""])
    lines.extend(["## " + ("病例级结果说明" if zh else "Case-level result notes"), ""])
    for case in cases:
        values = science[str(case["case_label"])]
        lines.extend([f"### {case['case_label']}", "", f"- Model: {MODEL_LABELS[str(case['model'])]}", f"- Malignancy prediction / reader mean: {values['malignancy_prediction']:.3f} / {values['malignancy_target']:.3f}", f"- Selected Grad-CAM target / status: {case['target']} / {case['map_status']}", ""])
    lines.extend(["## RPT-TA01", "", "See `tables_catalogue/RPT-TA01.csv`.", "", "## RPT-TA02", "", ("分类概念的 modal label 仅用于可读展示；冻结 ground truth 是完整 reader vote distribution。" if zh else "Categorical modal labels are display aids; the frozen ground truth is the full reader vote distribution."), ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _register_pdf_fonts(language: str) -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if language == "zh":
        path = "/System/Library/Fonts/Supplemental/Songti.ttc"
        regular = "P10CataloguePrivateSongti"; bold = "P10CataloguePrivateSongtiBold"
        if regular not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(regular, path, subfontIndex=6)); pdfmetrics.registerFont(TTFont(bold, path, subfontIndex=1))
        return regular, bold
    return "Helvetica", "Helvetica-Bold"


def _render_appendix_pdf(cases: Sequence[Mapping[str, Any]], science: Mapping[str, Mapping[str, Any]], figures: Mapping[str, Path], panels: Mapping[str, Path], table_paths: Mapping[str, Path], language: str, destination: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle
    regular, bold = _register_pdf_fonts(language)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("P10PrivateTitle", fontName=bold, fontSize=22, leading=28, textColor=colors.HexColor("#173B57"), spaceAfter=8*mm)
    heading = ParagraphStyle("P10PrivateHeading", fontName=bold, fontSize=15, leading=19, textColor=colors.HexColor("#245B78"), spaceBefore=5*mm, spaceAfter=3*mm)
    body = ParagraphStyle("P10PrivateBody", fontName=regular, fontSize=9.5, leading=14, spaceAfter=3*mm, wordWrap="CJK" if language == "zh" else None)
    caption = ParagraphStyle("P10PrivateCaption", fontName=regular, fontSize=8.2, leading=11, textColor=colors.HexColor("#334155"), spaceAfter=4*mm)
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=18*mm, bottomMargin=16*mm, title="Private qualitative appendix" if language == "en" else "私有定性附录", author="LIDC-IDRI Baseline-v2", creator="lidc_baseline.p10_catalogue_private")
    story = [Spacer(1, 25*mm), Paragraph("Private qualitative appendix" if language == "en" else "私有定性附录", title), Paragraph("Frozen case evidence for Prediction, WHERE, WHAT, WHY, and HOW" if language == "en" else "Prediction、WHERE、WHAT、WHY 与 HOW 的冻结病例证据", body), Paragraph("No model forward or scientific recomputation was performed. The 64³ ROI is a local model-input patch, not a full CT slice. Display normalization never changes the raw FP32 maps used for quantitative faithfulness." if language == "en" else "未运行模型 forward 或科学重计算。64³ ROI 是局部模型输入，并非完整 CT 切片；显示归一化不会改变定量忠实度所用 raw FP32 map。", body), PageBreak()]
    for figure_id in PRIVATE_FIGURE_IDS[:-1]:
        selected = [case for case in cases if figure_id in case["report_figure_ids"]]
        story.append(Paragraph(figure_id, heading))
        for case in selected:
            label = str(case["case_label"])
            story.extend([
                Image(str(panels[label]), width=170*mm, height=42.5*mm),
                Paragraph(
                    (f"{label}: paper-style full-slice context, ROI, existing {case['target']} Grad-CAM, and frozen prediction/contribution evidence." if language == "en" else f"{label}：论文式完整切片背景、ROI、既有 {case['target']} Grad-CAM 与冻结 prediction/contribution 证据。"),
                    caption,
                ),
            ])
        story.append(PageBreak())
    story.extend([
        Paragraph("RPT-FA06", heading),
        Image(str(figures["RPT-FA06"]), width=170*mm, height=74*mm),
        Paragraph(("The integrated case explicitly preserves the unavailable HOW component as DATA_NOT_PERSISTED." if language == "en" else "综合病例图将缺失 HOW 组件明确保留为 DATA_NOT_PERSISTED。"), caption),
        PageBreak(),
    ])
    story.append(Paragraph("Private Table RPT-TA01 — Frozen 14-case index", heading))
    ta01 = list(csv.DictReader(table_paths["RPT-TA01_csv"].open(encoding="utf-8")))
    cols = list(ta01[0])
    data = [[Paragraph(c, ParagraphStyle("h", fontName=bold, fontSize=6.5, textColor=colors.white)) for c in cols]] + [[Paragraph(str(row[c]), ParagraphStyle("c", fontName=regular, fontSize=6.2, leading=8)) for c in cols] for row in ta01]
    table = LongTable(data, repeatRows=1, colWidths=[(A4[0]-36*mm)/len(cols)]*len(cols))
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#245B78")),("GRID",(0,0),(-1,-1),.3,colors.HexColor("#CBD5E1")),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.extend([table, PageBreak(), Paragraph("Private Table RPT-TA02 — Case-level concept and malignancy predictions", heading), Paragraph(("Categorical modal labels are display aids; the frozen evaluation target is the complete reader vote distribution retained in the CSV." if language == "en" else "分类 modal label 仅用于可读展示；冻结评估目标是 CSV 中保留的完整 reader vote distribution。"), body)])
    ta02 = list(csv.DictReader(table_paths["RPT-TA02_csv"].open(encoding="utf-8")))
    small = ParagraphStyle(
        "P10PrivateTableCell",
        fontName=regular,
        fontSize=6.5,
        leading=8.2,
        wordWrap="CJK" if language == "zh" else None,
    )
    # Keep machine-readable numeric tokens intact in both language PDFs.  CJK
    # line wrapping may otherwise split a decimal token at an arbitrary glyph
    # boundary, which defeats the fail-closed bilingual token comparison.
    numeric_cell = ParagraphStyle(
        "P10PrivateNumericTableCell",
        fontName=regular,
        fontSize=6.0,
        leading=8.2,
        wordWrap=None,
        splitLongWords=False,
    )
    small_header = ParagraphStyle(
        "P10PrivateTableHeader",
        fontName=bold,
        fontSize=6.8,
        leading=8.5,
        textColor=colors.white,
        wordWrap="CJK" if language == "zh" else None,
    )
    ta02_headers = (
        ("Case / model", "病例 / 模型"),
        ("Malignancy Pred / GT", "恶性度 Pred / GT"),
        ("Continuous concepts Pred / GT", "连续概念 Pred / GT"),
        ("Categorical concepts Pred / reader target", "分类概念 Pred / reader target"),
    )
    ta02_data = [[Paragraph(value[1 if language == "zh" else 0], small_header) for value in ta02_headers]]
    for row in ta02:
        continuous = "<br/>".join(
            f"{concept}: {row[f'{concept} Pred/GT']}" for concept in CONTINUOUS_CONCEPTS
        )
        categorical = "<br/>".join(
            f"{concept}: {row[f'{concept} Pred/GT']}" for concept in CATEGORICAL_CONCEPTS
        )
        ta02_data.append(
            [
                Paragraph(f"{row['CASE']}<br/>{row['Model']}", small),
                Paragraph(f"{row['Malignancy Pred']} / {row['Malignancy GT']}", numeric_cell),
                Paragraph(continuous, numeric_cell),
                Paragraph(categorical, numeric_cell),
            ]
        )
    ta02_table = LongTable(
        ta02_data,
        repeatRows=1,
        colWidths=[30 * mm, 30 * mm, 52 * mm, 58 * mm],
        splitByRow=1,
    )
    ta02_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#245B78")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
            ]
        )
    )
    story.append(ta02_table)
    doc.build(story)
    return destination


def build_catalogue_private_appendices(repository_root: Path = Path("."), private_root: Path = PRIVATE_REPORT_ROOT, public_root: Path = PUBLIC_ROOT) -> dict[str, Any]:
    context = load_catalogue_context(repository_root)
    cases = _load_cases(context)
    frames, manifest = _load_frames(repository_root)
    manifest_by_uid = manifest.set_index("nodule_uid", drop=False)
    science = {str(case["case_label"]): _case_science(case, frames[str(case["model"])], manifest) for case in cases}
    table_paths = build_private_tables(cases, science, private_root / "tables_catalogue")
    panels: dict[str, dict[str, Path]] = {"en": {}, "zh": {}}
    figures: dict[str, dict[str, Path]] = {"en": {}, "zh": {}}
    for language in ("en", "zh"):
        panel_root = private_root / "case_panels_catalogue" / language
        for case in cases:
            label = str(case["case_label"])
            panels[language][label] = _case_panel(case, science[label], manifest_by_uid.loc[str(case["nodule_uid"])], repository_root, panel_root / f"{label}.png", language)
        figure_root = private_root / "figures_catalogue" / language
        for figure_id in PRIVATE_FIGURE_IDS[:-1]:
            selected = [case for case in cases if figure_id in case["report_figure_ids"]]
            if not selected:
                raise ValueError(f"P10_PRIVATE_FIGURE_CASES_MISSING:{figure_id}")
            figures[language][figure_id] = _group_figure(figure_id, [panels[language][str(case["case_label"])] for case in selected], figure_root / f"{figure_id}_{language}.png", language)
        fa06 = [case for case in cases if "RPT-FA06" in case["report_figure_ids"]]
        if len(fa06) != 1 or str(fa06[0]["case_label"]) != "CASE-0004":
            raise ValueError("P10_FA06_CATALOGUE_SELECTION_INVALID")
        figures[language]["RPT-FA06"] = _fa06_figure(fa06[0], science["CASE-0004"], panels[language]["CASE-0004"], figure_root / f"RPT-FA06_{language}.png", language)
        _private_markdown(cases, science, figures[language], language, private_root / f"qualitative_appendix_{language}.md")
        _render_appendix_pdf(cases, science, figures[language], panels[language], table_paths, language, private_root / f"qualitative_appendix_{language}.pdf")
        technical = public_root / f"technical_{language}.pdf"
        if not technical.is_file():
            raise ValueError("P10_PUBLIC_TECHNICAL_PDF_MISSING")
        _merge_pdfs(technical, private_root / f"qualitative_appendix_{language}.pdf", private_root / f"technical_{language}_with_appendix.pdf")
    payload = {
        "schema_version": 1,
        "status": "PRIVATE_APPENDICES_BUILT_PENDING_QA",
        "catalogue_registry_sha256": context.registry_sha256,
        "case_count": len(cases),
        "case_labels": [case["case_label"] for case in cases],
        "tables": {key: {"path": value.relative_to(private_root).as_posix(), "sha256": sha256_file(value)} for key, value in table_paths.items()},
        "figures": {language: {key: {"path": value.relative_to(private_root).as_posix(), "sha256": sha256_file(value)} for key, value in rows.items()} for language, rows in figures.items()},
        "reports": {
            name: {
                "sha256": sha256_file(private_root / f"{name}.pdf"),
                **(
                    {"markdown_sha256": sha256_file(private_root / f"{name}.md")}
                    if name.startswith("qualitative_appendix_")
                    else {}
                ),
            }
            for name in (
                "qualitative_appendix_en",
                "qualitative_appendix_zh",
                "technical_en_with_appendix",
                "technical_zh_with_appendix",
            )
        },
        "fa06": {"selected_case_label": "CASE-0004", "how": "DATA_NOT_PERSISTED", "model_forward": False},
        "model_forward": False,
        "scientific_recomputation": False,
        "p11_started": False,
    }
    path = private_root / "catalogue_private_report_manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return payload
