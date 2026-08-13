"""Build the Mac-private bilingual P10 qualitative appendix without a forward pass."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidc_baseline.p10_archive import LOCAL_ROOT_DEFAULT, verify_archive
from lidc_baseline.p10_report import (
    MODEL_LABELS,
    MODEL_ORDER,
    _pdf_page_count,
    _pdf_text_and_font_evidence,
    _pdfplumber_page_text_gate,
    _rendered_pdf_page_rows,
    _resolve_pdftoppm,
    _verify_chinese_pdf_fonts,
    extract_numeric_tokens,
    sha256_file,
)


PRIVATE_ROOT_NAME = "p10_private_report"
ROI_ROOT_DEFAULT = Path("artifacts/baseline_v2/rois")
OOF_FILENAMES = {
    "blackbox": "blackbox_oof_predictions.parquet",
    "standard_cbm": "standard_cbm_oof_predictions.parquet",
    "mixed_cem": "cem_oof_predictions.parquet",
    "learned_softmax_gam": "gam_oof_predictions.parquet",
}
CASE_COUNT = 14
PRIVATE_VISUAL_QA_NAME = "private_visual_qa.json"
PRIVATE_MANUAL_VISUAL_REVIEWER = (
    "Codex primary agent (visual inspection of contact sheets and "
    "original-resolution critical pages)"
)
ROLE_LABELS_ZH = {
    "median_error_representative": "中位绝对误差代表案例",
    "maximum_error_failure": "最大绝对误差失败案例",
    "maximum_positive_error_worsening": "遮挡后误差增幅最大的案例",
    "highest_undefined_rate_concept_zero_map": "未定义率最高概念的全零图案例",
}
ROLE_LABELS_EN = {
    "median_error_representative": "median absolute-error representative",
    "maximum_error_failure": "maximum absolute-error failure",
    "maximum_positive_error_worsening": "largest positive error worsening after occlusion",
    "highest_undefined_rate_concept_zero_map": "zero map for the highest-undefined-rate concept",
}
MAP_STATUS_LABELS_ZH = {
    "valid": "有效",
    "undefined": "未定义（post-ReLU 全零）",
}
MAP_STATUS_LABELS_EN = {
    "valid": "valid",
    "undefined": "undefined (post-ReLU exact zero)",
}


def _stable_uid_hash(uid: str) -> str:
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()


def _score_column(frame: pd.DataFrame) -> str:
    for name in ("malignancy_raw_score", "raw_task_score", "prediction_normalized"):
        if name in frame:
            return name
    raise ValueError("P10_PRIVATE_OOF_SCORE_COLUMN_MISSING")


def _target_column(frame: pd.DataFrame) -> str:
    for name in ("target_normalized", "malignancy_target_normalized"):
        if name in frame:
            return name
    raise ValueError("P10_PRIVATE_OOF_TARGET_COLUMN_MISSING")


def select_task_cases(
    frame: pd.DataFrame,
    model: str,
    *,
    used_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    used = used_uids if used_uids is not None else set()
    score_column = _score_column(frame)
    target_column = _target_column(frame)
    work = frame.copy()
    work["absolute_error"] = np.abs(
        work[score_column].astype(float) - work[target_column].astype(float)
    )
    median = float(work["absolute_error"].median())
    work["median_distance"] = np.abs(work["absolute_error"] - median)
    work["stable_hash"] = work["nodule_uid"].astype(str).map(_stable_uid_hash)
    representatives = work.sort_values(
        ["median_distance", "fold_index", "stable_hash"], kind="mergesort"
    )
    failures = work.sort_values(
        ["absolute_error", "fold_index", "stable_hash"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    output = []
    for role, candidates in (
        ("median_error_representative", representatives),
        ("maximum_error_failure", failures),
    ):
        selected = next(
            row for _, row in candidates.iterrows() if str(row["nodule_uid"]) not in used
        )
        uid = str(selected["nodule_uid"])
        used.add(uid)
        output.append(
            {
                "model": model,
                "role": role,
                "nodule_uid": uid,
                "patient_key": str(selected["patient_key"]),
                "fold_index": int(selected["fold_index"]),
                "target": "malignancy",
                "score_original": float(selected[score_column]),
                "target_normalized": float(selected[target_column]),
                "absolute_error": float(selected["absolute_error"]),
            }
        )
    return output


def _map_metadata_rows(archive_root: Path) -> Iterable[dict[str, Any]]:
    spatial = archive_root / "p9" / "spatial"
    columns = (
        "nodule_uid",
        "model",
        "fold_index",
        "target",
        "status",
        "faithfulness_json",
    )
    for model in MODEL_ORDER[1:]:
        for fold in range(5):
            root = spatial / model / f"fold_{fold}"
            for shard in sorted(root.glob("shard_*.parquet")):
                frame = pd.read_parquet(shard, columns=list(columns))
                for row in frame.to_dict(orient="records"):
                    faithfulness = (
                        json.loads(row["faithfulness_json"])
                        if row["faithfulness_json"]
                        else None
                    )
                    yield {
                        **{
                            key: value
                            for key, value in row.items()
                            if key != "faithfulness_json"
                        },
                        "saliency_error_increase": (
                            None
                            if faithfulness is None
                            else float(faithfulness["saliency_error_increase"])
                        ),
                        "shard_relative_path": shard.relative_to(archive_root).as_posix(),
                    }


def select_concept_cases(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    oof: pd.DataFrame,
    *,
    used_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    used = used_uids if used_uids is not None else set()
    concept_rows = [
        row for row in rows if row["model"] == model and row["target"] != "malignancy"
    ]
    valid = []
    for row in concept_rows:
        value = row["saliency_error_increase"]
        if row["status"] != "valid" or value is None:
            continue
        value = float(value)
        if math_finite(value) and value > 0:
            valid.append((value, row))
    valid.sort(
        key=lambda item: (
            -item[0],
            int(item[1]["fold_index"]),
            str(item[1]["target"]),
            _stable_uid_hash(str(item[1]["nodule_uid"])),
        )
    )
    positive = next(row for _, row in valid if str(row["nodule_uid"]) not in used)
    used.add(str(positive["nodule_uid"]))
    counts: dict[str, list[int]] = {}
    for row in concept_rows:
        target = str(row["target"])
        if target not in counts:
            counts[target] = [0, 0]
        counts[target][0] += 1
        counts[target][1] += int(row["status"] == "undefined")
    highest_target = sorted(
        counts,
        key=lambda target: (-(counts[target][1] / counts[target][0]), target),
    )[0]
    undefined = sorted(
        (
            row
            for row in concept_rows
            if row["target"] == highest_target and row["status"] == "undefined"
        ),
        key=lambda row: (
            int(row["fold_index"]),
            str(row["target"]),
            _stable_uid_hash(str(row["nodule_uid"])),
        ),
    )
    zero = next(row for row in undefined if str(row["nodule_uid"]) not in used)
    used.add(str(zero["nodule_uid"]))
    oof_by_uid = oof.set_index("nodule_uid", drop=False)

    def enrich(row: Mapping[str, Any], role: str) -> dict[str, Any]:
        uid = str(row["nodule_uid"])
        source = oof_by_uid.loc[uid]
        return {
            "model": model,
            "role": role,
            "nodule_uid": uid,
            "patient_key": str(source["patient_key"]),
            "fold_index": int(row["fold_index"]),
            "target": str(row["target"]),
            "map_status": str(row["status"]),
            "saliency_error_increase": (
                None
                if row["saliency_error_increase"] is None
                else float(row["saliency_error_increase"])
            ),
            "shard_relative_path": str(row["shard_relative_path"]),
            "highest_undefined_rate_target": highest_target,
            "highest_undefined_rate": counts[highest_target][1] / counts[highest_target][0],
        }

    return [
        enrich(positive, "maximum_positive_error_worsening"),
        enrich(zero, "highest_undefined_rate_concept_zero_map"),
    ]


def math_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def select_private_cases(
    archive_root: Path,
) -> list[dict[str, Any]]:
    oof_root = archive_root / "p9" / "canonical_oof"
    oof_frames = {
        model: pd.read_parquet(oof_root / OOF_FILENAMES[model]) for model in MODEL_ORDER
    }
    selected: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        selected.extend(select_task_cases(oof_frames[model], model))
    map_rows = list(_map_metadata_rows(archive_root))
    for model in MODEL_ORDER[1:]:
        selected.extend(select_concept_cases(map_rows, model, oof_frames[model]))
    if len(selected) != CASE_COUNT:
        raise ValueError("P10_PRIVATE_CASE_SELECTION_CARDINALITY_INVALID")
    selected.sort(
        key=lambda row: (
            MODEL_ORDER.index(row["model"]),
            row["role"],
            row["fold_index"],
            row["target"],
            _stable_uid_hash(row["nodule_uid"]),
        )
    )
    for index, row in enumerate(selected, start=1):
        row["case_label"] = f"CASE-{index:04d}"
    return selected


def _find_map_record(
    archive_root: Path,
    case: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    if "shard_relative_path" in case:
        candidates = [archive_root / str(case["shard_relative_path"])]
    else:
        candidates = sorted(
            (
                archive_root
                / "p9"
                / "spatial"
                / str(case["model"])
                / f"fold_{case['fold_index']}"
            ).glob("shard_*.parquet")
        )
    for shard in candidates:
        frame = pd.read_parquet(shard)
        match = frame[
            (frame["nodule_uid"].astype(str) == str(case["nodule_uid"]))
            & (frame["target"].astype(str) == str(case["target"]))
        ]
        if len(match) != 1:
            continue
        row = match.iloc[0]
        values = np.frombuffer(row["map_bytes"], dtype="<f4").copy().reshape(64, 64, 64)
        if hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest() != row[
            "map_sha256"
        ]:
            raise ValueError("P10_PRIVATE_MAP_SHA256_MISMATCH")
        if row["status"] == "undefined" and np.count_nonzero(values) != 0:
            raise ValueError("P10_PRIVATE_UNDEFINED_MAP_NOT_ZERO")
        return values, str(row["status"])
    raise ValueError(f"P10_PRIVATE_MAP_RECORD_MISSING:{case['case_label']}")


def _load_roi(uid: str, roi_root: Path) -> np.ndarray:
    path = roi_root / f"{uid}.npz"
    if not path.is_file():
        raise ValueError("P10_PRIVATE_ROI_MISSING")
    with np.load(path, allow_pickle=False) as archive:
        if "image" not in archive.files:
            raise ValueError("P10_PRIVATE_ROI_SCHEMA_INVALID")
        roi = np.asarray(archive["image"], dtype=np.float32)
    if roi.shape == (1, 64, 64, 64):
        roi = roi[0]
    if roi.shape != (64, 64, 64):
        raise ValueError("P10_PRIVATE_ROI_SCHEMA_INVALID")
    if not np.isfinite(roi).all():
        raise ValueError("P10_PRIVATE_ROI_NONFINITE")
    return roi


def _slice_index(roi: np.ndarray, heatmap: np.ndarray) -> int:
    if np.count_nonzero(heatmap):
        scores = heatmap.reshape(64, -1).sum(axis=1)
    else:
        scores = np.std(roi.reshape(64, -1), axis=1)
    return int(np.argmax(scores))


def _render_panel(
    case: Mapping[str, Any],
    roi: np.ndarray,
    heatmap: np.ndarray,
    slice_index: int,
    panel_path: Path,
    source_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    image = roi[slice_index]
    saliency = heatmap[slice_index]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=180)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("ROI")
    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(saliency, cmap="magma", alpha=0.55)
    axes[1].set_title("ROI + Grad-CAM")
    axes[2].imshow(saliency, cmap="magma")
    axes[2].set_title("Grad-CAM (post-ReLU)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"{case['case_label']} | {MODEL_LABELS[case['model']]} | {case['target']} | z={slice_index}"
    )
    figure.tight_layout()
    figure.savefig(panel_path, metadata={"Title": str(case["case_label"]), "Author": "P10"})
    plt.close(figure)
    np.savez_compressed(
        source_path,
        roi_slice=image.astype(np.float32),
        raw_gradcam_slice=saliency.astype(np.float32),
        slice_index=np.asarray(slice_index, dtype=np.int16),
    )


def _write_private_index(path: Path, cases: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "status": "PRIVATE_CASE_INDEX",
        "cases": list(cases),
        "model_forward": False,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o400)
    temporary.replace(path)


def _appendix_markdown(cases: Sequence[Mapping[str, Any]], language: str) -> str:
    title = "Private qualitative appendix" if language == "en" else "私有定性附录"
    intro = (
        "The cases below were selected deterministically from frozen OOF and Grad-CAM artifacts. No model forward was run."
        if language == "en"
        else "以下案例由冻结的OOF与Grad-CAM产物确定性选择；未运行任何模型前向。"
    )
    lines = [f"# {title}", "", intro, ""]
    for case in cases:
        role = str(case["role"])
        if language == "zh":
            details = (
                f"- 模型：{MODEL_LABELS[case['model']]}",
                f"- 案例类型：{ROLE_LABELS_ZH[role]}",
                f"- 折：{case['fold_index']}",
                f"- 目标：{case['target']}",
                f"- 图状态：{MAP_STATUS_LABELS_ZH[case['map_status']]}",
                f"- 切片：{case['slice_index']}",
                *_case_numeric_lines(case, language),
            )
        else:
            details = (
                f"- Model: {MODEL_LABELS[case['model']]}",
                f"- Role: {ROLE_LABELS_EN[role]}",
                f"- Fold: {case['fold_index']}",
                f"- Target: {case['target']}",
                f"- Map status: {MAP_STATUS_LABELS_EN[case['map_status']]}",
                f"- Slice: {case['slice_index']}",
                *_case_numeric_lines(case, language),
            )
        lines.extend(
            (
                f"## {case['case_label']}",
                "",
                *details,
                "",
                f"![{case['case_label']}](panels/{case['case_label']}.png)",
                "",
            )
        )
    return "\n".join(lines)


def _case_numeric_lines(case: Mapping[str, Any], language: str) -> tuple[str, ...]:
    if all(field in case for field in ("score_original", "target_normalized", "absolute_error")):
        if language == "zh":
            return (
                f"- 原始分数：{case['score_original']:.6f}",
                f"- 归一化目标：{case['target_normalized']:.6f}",
                f"- 绝对误差：{case['absolute_error']:.6f}",
            )
        return (
            f"- Original score: {case['score_original']:.6f}",
            f"- Normalized target: {case['target_normalized']:.6f}",
            f"- Absolute error: {case['absolute_error']:.6f}",
        )
    if case["role"] == "maximum_positive_error_worsening" and "saliency_error_increase" in case:
        label = "遮挡后误差增量" if language == "zh" else "Occlusion error increase"
        return (f"- {label}: {case['saliency_error_increase']:.6f}",)
    if case["role"] == "highest_undefined_rate_concept_zero_map" and "highest_undefined_rate" in case:
        label = "最高概念未定义率" if language == "zh" else "Highest concept undefined rate"
        return (f"- {label}: {case['highest_undefined_rate']:.6f}",)
    return ()


def _render_appendix_pdf(
    cases: Sequence[Mapping[str, Any]],
    language: str,
    destination: Path,
    panel_root: Path,
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
        pdfmetrics.registerFont(TTFont("P10AppendixSongti", font_path, subfontIndex=6))
        pdfmetrics.registerFont(TTFont("P10AppendixSongtiBold", font_path, subfontIndex=1))
        regular, bold = "P10AppendixSongti", "P10AppendixSongtiBold"
    else:
        regular, bold = "Helvetica", "Helvetica-Bold"
    temporary = destination.with_name(f".{destination.name}.tmp")
    width, height = A4
    document = canvas.Canvas(
        str(temporary),
        pagesize=A4,
        pageCompression=1,
        initialFontName=regular,
        initialFontSize=10,
    )
    document.setTitle(
        "Private qualitative appendix" if language == "en" else "私有定性附录"
    )
    document.setAuthor("LIDC-IDRI Baseline-v2")
    document.setCreator("lidc_baseline.p10_private_appendix")
    for page, case in enumerate(cases, start=1):
        document.setFont(bold, 18)
        document.drawString(42, height - 50, str(case["case_label"]))
        document.setFont(regular, 10)
        if language == "zh":
            summary = (
                f"{MODEL_LABELS[case['model']]} | {ROLE_LABELS_ZH[case['role']]} | "
                f"第 {case['fold_index']} 折 | 目标 {case['target']}"
            )
        else:
            summary = (
                f"{MODEL_LABELS[case['model']]} | {ROLE_LABELS_EN[case['role']]} | "
                f"fold {case['fold_index']} | target {case['target']}"
            )
        document.drawString(42, height - 72, summary)
        y = height - 90
        document.setFont(regular, 9)
        for detail in _case_numeric_lines(case, language):
            document.drawString(42, y, detail.removeprefix("- "))
            y -= 13
        document.drawImage(
            str(panel_root / f"{case['case_label']}.png"),
            42,
            180,
            width=width - 84,
            height=height - 290,
            preserveAspectRatio=True,
            anchor="c",
        )
        document.setFont(regular, 8)
        document.drawRightString(width - 42, 30, f"{page}/{len(cases)}")
        document.showPage()
    document.save()
    temporary.replace(destination)


def _merge_pdfs(first: Path, second: Path, destination: Path) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as error:
        raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED") from error
    writer = PdfWriter()
    for source in (first, second):
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        writer.write(handle)
    temporary.replace(destination)


def record_private_visual_qa(
    archive_root: Path = LOCAL_ROOT_DEFAULT,
    *,
    manual_review_pass: bool,
) -> dict[str, Any]:
    """Bind Poppler, pdfplumber, and manual review to all four private PDFs."""
    root = archive_root / PRIVATE_ROOT_NAME
    renderer = _resolve_pdftoppm()
    names = (
        "qualitative_appendix_en",
        "qualitative_appendix_zh",
        "technical_en_with_appendix",
        "technical_zh_with_appendix",
    )
    pdfs: dict[str, Any] = {}
    total_pages = 0
    for name in names:
        path = root / f"{name}.pdf"
        page_count = _pdf_page_count(path)
        rendered = _rendered_pdf_page_rows(path, renderer)
        if len(rendered) != page_count or not all(row["nonblank"] for row in rendered):
            raise ValueError(f"P10_PRIVATE_PDF_RENDER_COVERAGE_INVALID:{name}")
        _pdfplumber_page_text_gate(path, page_count)
        total_pages += page_count
        pdfs[name] = {
            "pdf_sha256": sha256_file(path),
            "page_count": page_count,
            "rendered_pages": rendered,
        }
    payload = {
        "schema_version": 1,
        "status": "PASS" if manual_review_pass else "PENDING_MANUAL_REVIEW",
        "renderer": "pdftoppm",
        "render_dpi": 80,
        "rendered_page_count": total_pages,
        "pdfplumber_text_gate": "PASS",
        "manual_visual_review": "PASS" if manual_review_pass else "PENDING",
        "manual_reviewer": PRIVATE_MANUAL_VISUAL_REVIEWER,
        "manual_review_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rendered_page_manifest_sha256": hashlib.sha256(
            json.dumps(
                {name: row["rendered_pages"] for name, row in sorted(pdfs.items())},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "manual_checklist": {
            "clipping": "PASS" if manual_review_pass else "PENDING",
            "overlap": "PASS" if manual_review_pass else "PENDING",
            "fonts_and_missing_glyphs": "PASS" if manual_review_pass else "PENDING",
            "legends_and_images": "PASS" if manual_review_pass else "PENDING",
        },
        "pdfs": pdfs,
    }
    path = root / PRIVATE_VISUAL_QA_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(path)
    return payload


def _verify_private_manual_review_provenance(evidence: Mapping[str, Any]) -> None:
    if evidence.get("manual_reviewer") != PRIVATE_MANUAL_VISUAL_REVIEWER:
        raise ValueError("P10_PRIVATE_VISUAL_QA_REVIEWER_INVALID")
    timestamp = evidence.get("manual_review_timestamp_utc")
    if not isinstance(timestamp, str):
        raise ValueError("P10_PRIVATE_VISUAL_QA_TIMESTAMP_INVALID")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("P10_PRIVATE_VISUAL_QA_TIMESTAMP_INVALID") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("P10_PRIVATE_VISUAL_QA_TIMESTAMP_INVALID")
    expected_manifest = hashlib.sha256(
        json.dumps(
            {
                name: row["rendered_pages"]
                for name, row in sorted(evidence.get("pdfs", {}).items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if evidence.get("rendered_page_manifest_sha256") != expected_manifest:
        raise ValueError("P10_PRIVATE_VISUAL_QA_RENDER_MANIFEST_INVALID")


def _verify_private_visual_qa(archive_root: Path) -> dict[str, Any]:
    root = archive_root / PRIVATE_ROOT_NAME
    evidence = json.loads(
        (root / PRIVATE_VISUAL_QA_NAME).read_text(encoding="utf-8")
    )
    expected_names = {
        "qualitative_appendix_en",
        "qualitative_appendix_zh",
        "technical_en_with_appendix",
        "technical_zh_with_appendix",
    }
    _verify_private_manual_review_provenance(evidence)
    if (
        evidence.get("status") != "PASS"
        or evidence.get("manual_visual_review") != "PASS"
        or evidence.get("pdfplumber_text_gate") != "PASS"
        or set(evidence.get("pdfs", {})) != expected_names
        or any(value != "PASS" for value in evidence.get("manual_checklist", {}).values())
    ):
        raise ValueError("P10_PRIVATE_VISUAL_QA_NOT_APPROVED")
    renderer = _resolve_pdftoppm()
    total_pages = 0
    for name in sorted(expected_names):
        path = root / f"{name}.pdf"
        expected = evidence["pdfs"][name]
        page_count = _pdf_page_count(path)
        _pdfplumber_page_text_gate(path, page_count)
        if (
            expected.get("pdf_sha256") != sha256_file(path)
            or expected.get("page_count") != page_count
            or expected.get("rendered_pages") != _rendered_pdf_page_rows(path, renderer)
        ):
            raise ValueError(f"P10_PRIVATE_VISUAL_QA_BINDING_INVALID:{name}")
        total_pages += page_count
    if evidence.get("rendered_page_count") != total_pages:
        raise ValueError("P10_PRIVATE_VISUAL_QA_PAGE_COUNT_INVALID")
    return evidence


def build_private_appendix(
    *,
    language: str,
    archive_root: Path | None = None,
    roi_root: Path = ROI_ROOT_DEFAULT,
) -> dict[str, Any]:
    if language not in {"en", "zh"}:
        raise ValueError("P10_PRIVATE_LANGUAGE_INVALID")
    archive = archive_root or LOCAL_ROOT_DEFAULT
    verify_archive(archive)
    private_root = archive / PRIVATE_ROOT_NAME
    private_root.mkdir(parents=True, exist_ok=True)
    index_path = private_root / "private_case_index.json"
    if index_path.exists():
        cases = json.loads(index_path.read_text(encoding="utf-8"))["cases"]
    else:
        cases = select_private_cases(archive)
        for case in cases:
            heatmap, status = _find_map_record(archive, case)
            roi = _load_roi(case["nodule_uid"], roi_root)
            index = _slice_index(roi, heatmap)
            case["map_status"] = status
            case["slice_index"] = index
            panel = private_root / "panels" / f"{case['case_label']}.png"
            source = private_root / "panel_sources" / f"{case['case_label']}.npz"
            _render_panel(case, roi, heatmap, index, panel, source)
            case["panel_sha256"] = sha256_file(panel)
            case["panel_source_sha256"] = sha256_file(source)
        _write_private_index(index_path, cases)
    markdown = private_root / f"qualitative_appendix_{language}.md"
    markdown.write_text(_appendix_markdown(cases, language), encoding="utf-8")
    pdf = private_root / f"qualitative_appendix_{language}.pdf"
    _render_appendix_pdf(cases, language, pdf, private_root / "panels")
    public_technical = Path("reports/baseline_v2/p10/public") / f"technical_{language}.pdf"
    combined = private_root / f"technical_{language}_with_appendix.pdf"
    _merge_pdfs(public_technical, pdf, combined)
    expected_pdfs = tuple(
        private_root / name
        for name in (
            "qualitative_appendix_en.pdf",
            "qualitative_appendix_zh.pdf",
            "technical_en_with_appendix.pdf",
            "technical_zh_with_appendix.pdf",
        )
    )
    if all(path.is_file() for path in expected_pdfs):
        record_private_visual_qa(archive, manual_review_pass=False)
    return {
        "status": "PASS",
        "language": language,
        "case_count": len(cases),
        "same_case_index_sha256": sha256_file(index_path),
        "appendix_pdf_sha256": sha256_file(pdf),
        "combined_pdf_sha256": sha256_file(combined),
        "model_forward": False,
    }


def verify_private_appendices(archive_root: Path = LOCAL_ROOT_DEFAULT) -> dict[str, Any]:
    root = archive_root / PRIVATE_ROOT_NAME
    revision_manifest_path = root / "catalogue_private_report_manifest.json"
    revision_manifest = (
        json.loads(revision_manifest_path.read_text(encoding="utf-8"))
        if revision_manifest_path.is_file()
        else None
    )
    payload = json.loads((root / "private_case_index.json").read_text(encoding="utf-8"))
    cases = payload["cases"]
    expected_cases = select_private_cases(archive_root)
    _verify_selected_case_index(payload, cases, expected_cases)
    if len(cases) != CASE_COUNT or len({case["case_label"] for case in cases}) != CASE_COUNT:
        raise ValueError("P10_PRIVATE_CASE_INDEX_INVALID")
    if (root / "private_case_index.json").stat().st_mode & 0o777 != 0o400:
        raise ValueError("P10_PRIVATE_CASE_INDEX_PERMISSIONS_INVALID")
    private_tokens: list[str] = []
    for case in cases:
        if not re_case_label(str(case["case_label"])):
            raise ValueError("P10_PRIVATE_CASE_LABEL_INVALID")
        private_tokens.extend((str(case["nodule_uid"]), str(case["patient_key"])))
        for directory, suffix in (("panels", ".png"), ("panel_sources", ".npz")):
            path = root / directory / f"{case['case_label']}{suffix}"
            if not path.is_file():
                raise ValueError("P10_PRIVATE_PANEL_MISSING")
            hash_field = "panel_sha256" if directory == "panels" else "panel_source_sha256"
            if sha256_file(path) != case[hash_field]:
                raise ValueError("P10_PRIVATE_PANEL_HASH_MISMATCH")
    expected_labels = [str(case["case_label"]) for case in cases]
    technical_pages = _pdf_page_count(
        Path("reports/baseline_v2/p10/public/technical_en.pdf")
    )
    markdown_by_language: dict[str, str] = {}
    appendix_text_by_language: dict[str, str] = {}
    combined_text_by_language: dict[str, str] = {}
    for language in ("en", "zh"):
        markdown = root / f"qualitative_appendix_{language}.md"
        appendix = root / f"qualitative_appendix_{language}.pdf"
        combined = root / f"technical_{language}_with_appendix.pdf"
        if not markdown.is_file() or not appendix.is_file():
            raise ValueError("P10_PRIVATE_APPENDIX_MISSING")
        if not combined.is_file():
            raise ValueError("P10_PRIVATE_COMBINED_REPORT_MISSING")
        markdown_text = markdown.read_text(encoding="utf-8")
        appendix_text, _ = _pdf_text_and_font_evidence(appendix)
        combined_text, _ = _pdf_text_and_font_evidence(combined)
        public_technical = Path("reports/baseline_v2/p10/public") / f"technical_{language}.pdf"
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise RuntimeError("P10_REPORT_DEPENDENCIES_REQUIRED") from error
        expected_combined_pages = [
            page.extract_text() or ""
            for source in (public_technical, appendix)
            for page in PdfReader(str(source)).pages
        ]
        observed_combined_pages = [
            page.extract_text() or "" for page in PdfReader(str(combined)).pages
        ]
        if observed_combined_pages != expected_combined_pages:
            raise ValueError("P10_PRIVATE_COMBINED_NOT_EXACT_CONCATENATION")
        markdown_by_language[language] = markdown_text
        appendix_text_by_language[language] = appendix_text
        combined_text_by_language[language] = combined_text
        appendix_pages = _pdf_page_count(appendix)
        combined_pages = _pdf_page_count(combined)
        if revision_manifest is None:
            expected_appendix_pages = CASE_COUNT
        else:
            report_evidence = revision_manifest.get("reports", {})
            appendix_evidence = report_evidence.get(f"qualitative_appendix_{language}", {})
            combined_evidence = report_evidence.get(f"technical_{language}_with_appendix", {})
            if (
                revision_manifest.get("status") != "PRIVATE_APPENDICES_BUILT_PENDING_QA"
                or revision_manifest.get("case_count") != CASE_COUNT
                or revision_manifest.get("case_labels") != expected_labels
                or revision_manifest.get("model_forward") is not False
                or revision_manifest.get("scientific_recomputation") is not False
                or revision_manifest.get("p11_started") is not False
                or appendix_evidence.get("sha256") != sha256_file(appendix)
                or appendix_evidence.get("markdown_sha256") != sha256_file(markdown)
                or combined_evidence.get("sha256") != sha256_file(combined)
            ):
                raise ValueError("P10_PRIVATE_REVISION_MANIFEST_INVALID")
            expected_appendix_pages = appendix_pages
        if appendix_pages != expected_appendix_pages or combined_pages != technical_pages + appendix_pages:
            raise ValueError("P10_PRIVATE_PDF_PAGE_COUNT_INVALID")
        for label in expected_labels:
            expected_markdown_count = 3 if revision_manifest is None else 1
            if markdown_text.count(label) != expected_markdown_count or label not in appendix_text:
                raise ValueError("P10_PRIVATE_CASE_CORRESPONDENCE_INVALID")
            if label not in combined_text:
                raise ValueError("P10_PRIVATE_COMBINED_CASE_MISSING")
        for token in private_tokens:
            if token in markdown_text or token in appendix_text or token in combined_text:
                raise ValueError("P10_PRIVATE_IDENTIFIER_LEAK")
        if language == "zh":
            _verify_chinese_pdf_fonts(appendix)
            _verify_chinese_pdf_fonts(combined)
    # The combined PDFs are proved above to be exact page-for-page
    # concatenations. Public technical-report bilingual parity is independently
    # verified by verify_public_outputs, so this private gate compares the two
    # private evidence layers it owns rather than conflating language-specific
    # TOC/footer page numbers with scientific values.
    for payload in (markdown_by_language, appendix_text_by_language):
        if extract_numeric_tokens(payload["en"]) != extract_numeric_tokens(payload["zh"]):
            raise ValueError("P10_PRIVATE_BILINGUAL_NUMERIC_MISMATCH")
    visual_qa = _verify_private_visual_qa(archive_root)
    appendix_pages = _pdf_page_count(root / "qualitative_appendix_en.pdf")
    combined_pages = _pdf_page_count(root / "technical_en_with_appendix.pdf")
    return {
        "status": "PASS",
        "case_count": CASE_COUNT,
        "same_cases": True,
        "model_forward": False,
        "private_identifiers_only_in_index": True,
        "appendix_pages": appendix_pages,
        "combined_pages": combined_pages,
        "chinese_fonts_embedded": True,
        "page_render_visual_qa": visual_qa["status"],
        "pdfplumber_text_gate": visual_qa["pdfplumber_text_gate"],
        "rendered_page_count": visual_qa["rendered_page_count"],
    }


def _verify_selected_case_index(
    payload: Mapping[str, Any],
    observed_cases: Sequence[Mapping[str, Any]],
    expected_cases: Sequence[Mapping[str, Any]],
) -> None:
    if payload.get("model_forward") is not False:
        raise ValueError("P10_PRIVATE_MODEL_FORWARD_BOUNDARY_INVALID")
    if len(observed_cases) != CASE_COUNT or len(expected_cases) != CASE_COUNT:
        raise ValueError("P10_PRIVATE_CASE_SELECTION_CARDINALITY_INVALID")
    for observed, expected in zip(observed_cases, expected_cases, strict=True):
        for field, value in expected.items():
            if observed.get(field) != value:
                raise ValueError(
                    f"P10_PRIVATE_CASE_SELECTION_MISMATCH:{expected['case_label']}:{field}"
                )


def re_case_label(value: str) -> bool:
    return len(value) == 9 and value.startswith("CASE-") and value[5:].isdigit()
