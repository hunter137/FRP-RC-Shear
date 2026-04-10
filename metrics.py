"""
metrics.py — Regression evaluation metrics for shear capacity prediction.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr as _pearsonr

# scipy < 1.9 returns a plain tuple; >= 1.9 returns PearsonRResult.
def _pearson_r(a, b):
    result = _pearsonr(a, b)
    return float(result.statistic if hasattr(result, 'statistic') else result[0])


def calc_metrics(y_true, y_pred):
    """Compute R2, r, RMSE, MAE, MAPE, mean_ratio, CoV, and safety_pct."""
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    ok = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
    yt, yp = yt[ok], yp[ok]
    if len(yt) < 2:
        return {k: np.nan for k in
                ('R2', 'r', 'RMSE', 'MAE', 'MAPE',
                 'mean_ratio', 'cov', 'safety_pct')}
    ratio      = yp / yt
    rmse       = float(np.sqrt(mean_squared_error(yt, yp)))
    mean_ratio = float(ratio.mean())
    cov        = float(ratio.std() / mean_ratio) if mean_ratio != 0.0 else float('nan')
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
