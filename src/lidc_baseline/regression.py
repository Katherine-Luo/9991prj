"""Baseline-v2 malignancy regression primitives."""

from __future__ import annotations

import math


def normalize_malignancy_target(mean_malignancy: float) -> float:
    """Map a valid mean LIDC malignancy rating from [1, 5] to [0, 1]."""
    value = float(mean_malignancy)
    if not math.isfinite(value) or value < 1.0 or value > 5.0:
        raise ValueError(f"mean_malignancy must be finite and in [1, 5], got {value}")
    return (value - 1.0) / 4.0


def normalized_score_to_rating_scale(score: float) -> float:
    """Convert an unconstrained normalized-scale prediction without clipping."""
    value = float(score)
    if not math.isfinite(value):
        raise ValueError(f"score must be finite, got {value}")
    return 1.0 + 4.0 * value


def extreme_binary_label(mean_malignancy: float) -> int | None:
    """Return the secondary extreme label while retaining middle-spectrum cases."""
    value = float(mean_malignancy)
    if not math.isfinite(value) or value < 1.0 or value > 5.0:
        raise ValueError(f"mean_malignancy must be finite and in [1, 5], got {value}")
    if value <= 2.0:
        return 0
    if value >= 4.0:
        return 1
    return None


def malignancy_stratum(mean_malignancy: float) -> str:
    """Return the pre-registered five-level patient-split stratum."""
    value = float(mean_malignancy)
    normalize_malignancy_target(value)
    if value <= 2.0:
        return "mean_le_2"
    if value < 3.0:
        return "mean_gt_2_lt_3"
    if value == 3.0:
        return "mean_eq_3"
    if value < 4.0:
        return "mean_gt_3_lt_4"
    return "mean_ge_4"
