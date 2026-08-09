"""Small deterministic helpers shared by the score-processing pipelines."""

from __future__ import annotations

import numpy as np


def nonnegative_quantile_threshold(
    raw_scores: np.ndarray, quantile: float
) -> float:
    """Return the max-clipped fallback threshold used for score anchoring."""

    values = np.asarray(raw_scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("raw_scores must be a nonempty one-dimensional array")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return max(0.0, float(np.quantile(values, quantile)))
