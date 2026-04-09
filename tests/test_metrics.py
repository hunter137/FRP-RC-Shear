"""
Unit tests for evaluation metrics.

Run:
    python -m pytest tests/test_metrics.py -v

Compatibility notes
-------------------
* This test file intentionally avoids re-importing calc_metrics from the
  main app (which requires PyQt5 / GUI stack).  It defines a local copy.
* scipy >= 1.9 : pearsonr() returns PearsonRResult instead of a plain
  tuple.  We use the _pearson_r() helper to handle both versions.
"""

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr as _pearsonr


def _pearson_r(a, b):
    """Return Pearson r scalar, compatible with scipy < 1.9 and >= 1.9."""
    res = _pearsonr(a, b)
    return float(res.statistic if hasattr(res, 'statistic') else res[0])


def calc_metrics(y_true, y_pred):
    """Local reproduction of the main app's metric function."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ok = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    yt, yp = yt[ok], yp[ok]
    if len(yt) < 2:
        return {k: float("nan") for k in
                ("R2", "r", "RMSE", "MAE", "MAPE",
                 "mean_ratio", "cov", "safety_pct")}
    ratio = yp / yt
    # np.sqrt(mse) avoids the deprecated/removed squared= kwarg (sklearn 1.4+)
    return {
        "R2":         round(float(r2_score(yt, yp)), 4),
        "r":          round(_pearson_r(yt, yp), 4),
        "RMSE":       round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
        "MAE":        round(float(mean_absolute_error(yt, yp)), 3),
        "MAPE":       round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 2),
        "mean_ratio": round(float(ratio.mean()), 4),
        "cov":        round(float(ratio.std() / ratio.mean()), 4),
        "safety_pct": round(float(np.mean(ratio <= 1.0) * 100), 1),
    }


class TestPerfectPrediction:
    """Metrics when prediction = true values."""

    def test_r2_is_one(self):
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        m = calc_metrics(y, y)
        assert m["R2"] == 1.0

    def test_rmse_is_zero(self):
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        m = calc_metrics(y, y)
        assert m["RMSE"] == 0.0

    def test_mae_is_zero(self):
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        m = calc_metrics(y, y)
        assert m["MAE"] == 0.0

    def test_mean_ratio_is_one(self):
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        m = calc_metrics(y, y)
        assert m["mean_ratio"] == 1.0


class TestKnownValues:
    """Metrics with known computed values."""

    def test_rmse_known(self):
        yt = np.array([10.0, 20.0, 30.0])
        yp = np.array([12.0, 18.0, 33.0])
        m = calc_metrics(yt, yp)
        # RMSE = sqrt(mean([4, 4, 9])) = sqrt(17/3) ≈ 2.380
        assert abs(m["RMSE"] - 2.380) < 0.01

    def test_mae_known(self):
        yt = np.array([10.0, 20.0, 30.0])
        yp = np.array([12.0, 18.0, 33.0])
        m = calc_metrics(yt, yp)
        # MAE = mean([2, 2, 3]) = 7/3 ≈ 2.333
        assert abs(m["MAE"] - 2.333) < 0.01

    def test_r_positive_correlation(self):
        yt = np.array([10, 20, 30, 40, 50], dtype=float)
        yp = np.array([11, 22, 28, 42, 48], dtype=float)
        m = calc_metrics(yt, yp)
        assert m["r"] > 0.99

    def test_safety_pct(self):
        yt = np.array([10, 20, 30, 40, 50], dtype=float)
        yp = np.array([8, 15, 25, 35, 55], dtype=float)
        m = calc_metrics(yt, yp)
        # ratio = [0.8, 0.75, 0.833, 0.875, 1.1] → 4/5 ≤ 1.0 → 80%
        assert m["safety_pct"] == 80.0


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_nan_values_filtered(self):
        yt = np.array([10, np.nan, 30])
        yp = np.array([11, 22, 28])
        m = calc_metrics(yt, yp)
        assert np.isfinite(m["R2"])

    def test_zero_true_filtered(self):
        yt = np.array([0, 10, 20])
        yp = np.array([5, 11, 22])
        m = calc_metrics(yt, yp)
        assert np.isfinite(m["R2"])

    def test_too_few_samples(self):
        m = calc_metrics([10], [10])
        assert np.isnan(m["R2"])

    def test_empty_input(self):
        m = calc_metrics([], [])
        assert np.isnan(m["R2"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
