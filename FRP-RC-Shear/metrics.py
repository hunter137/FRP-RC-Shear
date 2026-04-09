"""
metrics.py — Regression evaluation metrics for shear capacity prediction.

Compatibility notes
-------------------
* scipy >= 1.9 : pearsonr() returns a PearsonRResult object instead of a
  plain 2-tuple.  Both support index-0 access, but using .statistic is
  cleaner and future-proof.  We detect which API is available at import.
* sklearn any  : mean_squared_error() had a `squared` kwarg (removed in
  sklearn 1.4). We use np.sqrt(mse()) universally to avoid touching that
  parameter at all.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr as _pearsonr

# ── scipy pearsonr API shim ──────────────────────────────────────────
# scipy < 1.9  → returns plain tuple  (r, p)
# scipy >= 1.9 → returns PearsonRResult with .statistic / .pvalue
# In either case, result[0] works, but we wrap it once here so the rest
# of the code never has to think about it.
def _pearson_r(a, b):
    """Return only the Pearson r scalar, compatible with all scipy versions."""
    result = _pearsonr(a, b)
    # PearsonRResult (scipy >= 1.9) exposes .statistic; plain tuple does not.
    return float(result.statistic if hasattr(result, 'statistic') else result[0])


def calc_metrics(y_true, y_pred):
    """
    Compute a comprehensive set of regression metrics.

    Returns
    -------
    dict with keys:
        R2          : coefficient of determination
        r           : Pearson correlation coefficient
        RMSE        : root mean squared error [kN]
        MAE         : mean absolute error [kN]
        MAPE        : mean absolute percentage error [%]
        mean_ratio  : mean of V_pred / V_exp  (k̄)
        cov         : coefficient of variation of V_pred / V_exp
        safety_pct  : percentage of specimens with V_pred ≤ V_exp  [%]
    """
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ok = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    yt, yp = yt[ok], yp[ok]
    if len(yt) < 2:
        return {k: np.nan for k in
                ('R2', 'r', 'RMSE', 'MAE', 'MAPE',
                 'mean_ratio', 'cov', 'safety_pct')}
    ratio = yp / yt
    # Use np.sqrt(mse) — avoids the deprecated/removed `squared` kwarg
    rmse       = float(np.sqrt(mean_squared_error(yt, yp)))
    mean_ratio = float(ratio.mean())
    # Guard against division-by-zero: cov is undefined when mean_ratio == 0
    cov = float(ratio.std() / mean_ratio) if mean_ratio != 0.0 else float('nan')
    return {
        'R2':         round(float(r2_score(yt, yp)),                   4),
        'r':          round(_pearson_r(yt, yp),                        4),
        'RMSE':       round(rmse,                                       3),
        'MAE':        round(float(mean_absolute_error(yt, yp)),        3),
        'MAPE':       round(float(np.mean(np.abs((yt-yp)/yt)) * 100), 2),
        'mean_ratio': round(mean_ratio,                                4),
        'cov':        round(cov,                                       4),
        'safety_pct': round(float(np.mean(ratio <= 1.0) * 100),       1),
    }
