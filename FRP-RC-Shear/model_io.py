"""
model_io.py — Model wrapper, 1D-CNN regressor, and .frpmdl persistence.
"""
import os, copy, io, warnings, contextlib
import numpy as np
import joblib
from datetime import datetime
from sklearn.base import BaseEstimator as _BaseEstimator, RegressorMixin as _RegressorMixin

from config import APP_VERSION, _SHAP_BUNDLE_SAMPLES

# ═══════════════════════════════════════════════════════════════════════
#  FittedModel + CNN1DRegressor — needed to unpickle .frpmdl bundles
#  saved by train_frp_models.py  (pickle stores '__main__.<ClassName>')
# ═══════════════════════════════════════════════════════════════════════

class FittedModel:
    """Single fitted model, picklable (no lambdas)."""
    def __init__(self, model):
        self.model = model
        if hasattr(model, "feature_importances_"):
            self.feature_importances_ = model.feature_importances_
    def predict(self, X):
        return self.model.predict(X)

try:
    import torch as _torch
    import torch.nn as _nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    import io as _io
    from sklearn.base import BaseEstimator as _BaseEstimator
    from sklearn.base import RegressorMixin as _RegressorMixin

    class CNN1DRegressor(_BaseEstimator, _RegressorMixin):
        """1D-CNN regressor — sklearn API wrapper over PyTorch."""
        def __init__(self, n_filters1=64, n_filters2=128, kernel_size=3,
                     hidden_dim=64, dropout=0.2, lr=0.001, epochs=300,
                     batch_size=32, patience=30, random_state=42):
            self.n_filters1   = n_filters1
            self.n_filters2   = n_filters2
            self.kernel_size  = kernel_size
            self.hidden_dim   = hidden_dim
            self.dropout      = dropout
            self.lr           = lr
            self.epochs       = epochs
            self.batch_size   = batch_size
            self.patience     = patience
            self.random_state = random_state

        class _Net(_nn.Module):
            def __init__(self, in_features, f1, f2, ks, hid, drop):
                super().__init__()
                pad = ks // 2
                self.conv = _nn.Sequential(
                    _nn.Conv1d(1, f1, ks, padding=pad), _nn.ReLU(),
                    _nn.BatchNorm1d(f1),
                    _nn.Conv1d(f1, f2, ks, padding=pad), _nn.ReLU(),
                    _nn.BatchNorm1d(f2),
                    _nn.AdaptiveAvgPool1d(1),
                )
                self.head = _nn.Sequential(
                    _nn.Linear(f2, hid), _nn.ReLU(), _nn.Dropout(drop),
                    _nn.Linear(hid, 1),
                )
            def forward(self, x):
                h = self.conv(x).squeeze(-1)
                return self.head(h).squeeze(-1)

        def _build(self, n_features):
            _torch.manual_seed(self.random_state)
            return self._Net(n_features, self.n_filters1, self.n_filters2,
                             self.kernel_size, self.hidden_dim, self.dropout)

        def predict(self, X):
            X = np.asarray(X, dtype=np.float32)
            net = self._build(X.shape[1])
            buf = _io.BytesIO(self._weights_bytes)
            try:
                sd = _torch.load(buf, map_location="cpu", weights_only=True)
            except TypeError:
                buf.seek(0)
                sd = _torch.load(buf, map_location="cpu")
            net.load_state_dict(sd)
            net.eval()
            with _torch.no_grad():
                pred = net(_torch.from_numpy(X).unsqueeze(1)).numpy()
            return pred
else:
    # Stub so pickle can find the class name even without PyTorch
    class CNN1DRegressor:
        def predict(self, X):
            raise RuntimeError("PyTorch is required for 1D-CNN prediction. "
                               "Install it: pip install torch")

# ═══════════════════════════════════════════════════════════════════════
#  NumPy 2.x compatibility patch
#
#  Root cause:
#    .frpmdl bundles saved under NumPy 1.x embed a pickle call:
#      numpy.random._pickle.__randomstate_ctor('MT19937', <bit_gen_obj>)
#    i.e. two positional arguments.
#
#    Certain NumPy 2.x builds trimmed that function to accept 0–1
#    positional arguments, raising:
#      "__randomstate_ctor() takes from 0 to 1 positional arguments
#       but 2 were given"
#
#  Fix strategy:
#    Wrap joblib.load() in a context manager that temporarily replaces
#    numpy.random._pickle.__randomstate_ctor with a shim accepting
#    0, 1, or 2 positional arguments, then restores the original.
# ═══════════════════════════════════════════════════════════════════════

@contextlib.contextmanager
def _numpy2_compat_patch():
    """
    Temporarily patch numpy.random._pickle.__randomstate_ctor so that
    .frpmdl bundles saved under NumPy 1.x can be deserialised on NumPy
    2.x without a signature-mismatch error.

    Background:
      Bundles are saved with NumPy 1.x.  The embedded pickle call is:
        numpy.random._pickle.__randomstate_ctor('MT19937', <bit_gen_obj>)
      (two positional args).  Some NumPy 2.x releases accept only 0–1
      positional args, raising "takes from 0 to 1 positional arguments
      but 2 were given".

    Approach:
      During joblib.load() the function is replaced by a shim that
      handles all historical call signatures (0, 1, or 2 positional
      args).  The original is restored unconditionally via finally.
    """
    import numpy.random._pickle as _nrp
    from numpy.random import RandomState as _RS

    _orig = getattr(_nrp, '__randomstate_ctor', None)

    def _compat_ctor(*args, **kwargs):
        """
        Handle all historical pickle formats:
          - 0 args : modern code path
          - 1 arg  : bit_generator_name (str) or a BitGenerator instance
          - 2 args : (bit_generator_name, bit_generator_obj)  ← triggered here
        """
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
    """
    Save / load a complete model bundle (compressed joblib archive).

    The bundle stores trained models, preprocessing objects, metadata,
    and a calibration subsample of the training data (≤ 400 rows) to
    enable SHAP analysis after loading without access to the original dataset.
    """
    EXT = '.frpmdl'

    @staticmethod
    def save(path, results, scaler, feat_cols, ohe,
             X_all=None, X_shape=None, extra_meta=None):
        # Store a calibration subsample for post-load SHAP
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
            # Store raw predictions so scatter/error/ratio plots work after loading
            'predictions': {n: {
                                'tr_pred': np.asarray(r.get('tr_pred', [])),
                                'te_pred': np.asarray(r.get('te_pred', [])),
                            } for n, r in results.items()},
            # Store true labels (same for every model, taken from first entry)
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
        # When models were saved by train_frp_models.py (run as __main__),
        # pickle stored the class references as '__main__.FittedModel' etc.
        # Now the app's __main__ is main.py, so we patch it here so pickle
        # can find the classes during deserialization.
        import sys as _sys
        _main = _sys.modules.get('__main__')
        if _main is not None:
            if not hasattr(_main, 'FittedModel'):
                _main.FittedModel = FittedModel
            if not hasattr(_main, 'CNN1DRegressor'):
                _main.CNN1DRegressor = CNN1DRegressor

        # Wrap joblib.load() in _numpy2_compat_patch() to neutralise the
        # __randomstate_ctor signature mismatch; the original is restored
        # automatically via the context manager's finally block.
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
