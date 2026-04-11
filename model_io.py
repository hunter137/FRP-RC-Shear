"""
model_io.py — Model wrapper and .frpmdl persistence.
"""
import os, io, contextlib
import numpy as np
import joblib
from datetime import datetime

from config import APP_VERSION, _SHAP_BUNDLE_SAMPLES


class FittedModel:
    """Single fitted model, picklable (no lambdas)."""
    def __init__(self, model):
        self.model = model
        if hasattr(model, "feature_importances_"):
            self.feature_importances_ = model.feature_importances_
    def predict(self, X):
        return self.model.predict(X)

# Stub retained so that joblib can deserialise bundles saved by older versions.
class CNN1DRegressor:
    """Stub — PyTorch has been removed from this project."""
    def predict(self, X):
        raise RuntimeError(
            "1D-CNN model is no longer supported in this version. "
            "This bundle was saved with an older version that included PyTorch."
        )

# NumPy 1.x bundles embed a 2-arg __randomstate_ctor call that breaks
# on some NumPy 2.x builds.  Patch it temporarily during joblib.load().

@contextlib.contextmanager
def _numpy2_compat_patch():
    """Patch __randomstate_ctor to accept the 2-arg form used in NumPy 1.x bundles."""
    import numpy.random._pickle as _nrp
    from numpy.random import RandomState as _RS

    _orig = getattr(_nrp, '__randomstate_ctor', None)

    def _compat_ctor(*args, **kwargs):

        if len(args) == 0:
            bg_name = kwargs.get('bit_generator_name', 'MT19937')
            # Use caller-supplied constructor if provided, else numpy's default
            ctor    = kwargs.get('bit_generator_ctor', None)
            bg_obj  = ctor(bg_name) if callable(ctor) \
                      else _nrp.__bit_generator_ctor(bg_name)
            return _RS(bg_obj)
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, str):
                return _RS(_nrp.__bit_generator_ctor(arg))
            else:
                # arg is already a BitGenerator instance (legacy format)
                return _RS(arg)
        else:
            # args = (bit_generator_name, bit_generator_obj_or_ctor)
            _, bg = args[0], args[1]
            if callable(bg):
                return _RS(bg(args[0]))
            else:
                return _RS(bg)

    _nrp.__randomstate_ctor = _compat_ctor
    try:
        yield
    finally:
        if _orig is None:
            try:
                del _nrp.__randomstate_ctor
            except AttributeError:
                pass
        else:
            _nrp.__randomstate_ctor = _orig

class ModelIO:
    """Save and load compressed model bundles (.frpmdl)."""
    EXT = '.frpmdl'

    @staticmethod
    def save(path, results, scaler, feat_cols, ohe,
             X_all=None, X_shape=None, extra_meta=None):

        shap_cache = None
        if X_all is not None and len(X_all) > 0:
            n     = min(_SHAP_BUNDLE_SAMPLES, len(X_all))
            idx   = np.random.default_rng(42).choice(
                len(X_all), n, replace=False)
            shap_cache = X_all[idx]

        bundle = {
            'models':     {n: r['model'] for n, r in results.items()},
            'metrics':    {n: {k: r[k] for k in
                               ('tr_metrics', 'te_metrics', 'cv_mean', 'cv_std')}
                           for n, r in results.items()},

            'predictions': {n: {
                                'tr_pred': np.asarray(r.get('tr_pred', [])),
                                'te_pred': np.asarray(r.get('te_pred', [])),
                            } for n, r in results.items()},

            'y_splits':   {
                'y_tr': np.asarray(results[next(iter(results))].get('_y_tr', [])) if results else np.array([]),
                'y_te': np.asarray(results[next(iter(results))].get('_y_te', [])) if results else np.array([]),
            },
            'scaler':     scaler,
            'feat_cols':  feat_cols,
            'ohe':        ohe,
            'shap_cache': shap_cache,
            'meta': {
                'version':    APP_VERSION,
                'saved_at':   datetime.now().isoformat(timespec='seconds'),
                'n_train':    X_shape[0] if X_shape else None,
                'n_features': X_shape[1] if X_shape else None,
                **(extra_meta or {}),
            },
        }
        joblib.dump(bundle, path, compress=3)

    @staticmethod
    def load(path):
        # Ensure pickle can resolve class names saved under __main__.
        import sys as _sys
        _main = _sys.modules.get('__main__')
        if _main is not None:
            if not hasattr(_main, 'FittedModel'):
                _main.FittedModel = FittedModel
            if not hasattr(_main, 'CNN1DRegressor'):
                _main.CNN1DRegressor = CNN1DRegressor


        with _numpy2_compat_patch():
            bundle = joblib.load(path)

        results = {}
        preds   = bundle.get('predictions', {})
        ysplits = bundle.get('y_splits', {})
        y_tr    = ysplits.get('y_tr', np.array([]))
        y_te    = ysplits.get('y_te', np.array([]))
        for name, model in bundle['models'].items():
            met = bundle['metrics'].get(name, {})
            p   = preds.get(name, {})
            results[name] = {
                'model':      model,
                'tr_pred':    p.get('tr_pred', np.array([])),
                'te_pred':    p.get('te_pred', np.array([])),
                'tr_metrics': met.get('tr_metrics', {}),
                'te_metrics': met.get('te_metrics', {}),
                'cv_mean':    met.get('cv_mean', float('nan')),
                'cv_std':     met.get('cv_std',  float('nan')),
            }
        return (results,
                bundle['scaler'],
                bundle['feat_cols'],
                bundle.get('ohe'),
                bundle.get('shap_cache'),
                bundle.get('meta', {}),
                y_tr, y_te)
