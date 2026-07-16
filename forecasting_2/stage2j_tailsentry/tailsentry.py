from __future__ import annotations

import numpy as np

LOWER_QUANTILE = 0.125
UPPER_QUANTILE = 0.875
VERSION = "tailsentry-q125-v1"


def guard_forecast(
    routed_mean: np.ndarray,
    routed_sigma: np.ndarray,
    candidate_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the frozen pointwise central-candidate envelope."""
    routed_mean = np.asarray(routed_mean, dtype=np.float64)
    routed_sigma = np.maximum(np.asarray(routed_sigma, dtype=np.float64), 1e-12)
    candidate_means = np.asarray(candidate_means, dtype=np.float64)
    if candidate_means.ndim != 2:
        raise ValueError(f"candidate_means must be 2D, got {candidate_means.shape}")
    if candidate_means.shape[1] != routed_mean.shape[0]:
        raise ValueError("candidate horizon does not match routed forecast")
    if not np.isfinite(routed_mean).all() or not np.isfinite(routed_sigma).all():
        raise ValueError("routed forecast is non-finite")
    if not np.isfinite(candidate_means).all():
        raise ValueError("candidate means are non-finite")
    lower = np.quantile(candidate_means, LOWER_QUANTILE, axis=0)
    upper = np.quantile(candidate_means, UPPER_QUANTILE, axis=0)
    guarded_mean = np.clip(routed_mean, lower, upper)
    displacement = routed_mean - guarded_mean
    guarded_sigma = np.sqrt(np.maximum(routed_sigma**2 + displacement**2, 1e-12))
    diagnostics = {
        "changed_fraction": float(np.mean(displacement != 0.0)),
        "mean_absolute_displacement": float(np.mean(np.abs(displacement))),
        "max_absolute_displacement": float(np.max(np.abs(displacement))),
    }
    return guarded_mean, guarded_sigma, diagnostics
