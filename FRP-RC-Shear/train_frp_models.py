"""
train_frp_models.py  —  FRP-RC Shear Capacity (Paper-Aligned + Extended)
=========================================================================
Original paper methodology (6 numeric feats + OHE, MinMaxScaler, Bayesian
Optuna TPE 1000 trials, 10-fold CV) PLUS additional models.

Models (11 total):
  ① GBDT          ② XGBoost       ③ LightGBM
  ④ CatBoost      ⑤ Random Forest ⑥ Extra Trees
  ⑦ MLP           ⑧ SVR           ⑨ AdaBoost
  ⑩ KNN           ⑪ 1D-CNN (requires PyTorch)

Usage:
  python train_frp_models.py --data "database.xls"
  python train_frp_models.py --data "..." --trials 1000
  python train_frp_models.py --data "..." --only LightGBM MLP 1D-CNN
  python train_frp_models.py --data "..." --time-limit 240
"""

import sys, os, re, argparse, time, warnings, io, copy
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing   import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import pearsonr as _pearsonr
from model_io import FittedModel as _FittedModelIO, ModelIO

# ── Compatibility helpers ─────────────────────────────────────────────

def _pearson_r(a, b):
    """scipy-version-agnostic Pearson r scalar.
    scipy < 1.9  → plain tuple (r, p)
    scipy >= 1.9 → PearsonRResult with .statistic
    """
    res = _pearsonr(a, b)
    return float(res.statistic if hasattr(res, 'statistic') else res[0])

def _ohe_sparse_kwarg():
    """Return {sparse_output: False} or {sparse: False} for installed sklearn."""
    import sklearn
    try:
        from packaging.version import Version
        return ({'sparse_output': False}
                if Version(sklearn.__version__) >= Version('1.2')
                else {'sparse': False})
    except Exception:
        try:
            OneHotEncoder(sparse_output=False)
            return {'sparse_output': False}
        except TypeError:
            return {'sparse': False}

def _xgb_gpu_kwargs():
    """Return correct XGBoost GPU kwargs for the installed version."""
    try:
        import xgboost as _x
        from packaging.version import Version
        return ({'device': 'cuda'} if Version(_x.__version__) >= Version('2.0.0')
                else {'tree_method': 'gpu_hist'})
    except Exception:
        return {'device': 'cuda'}

try:    import xgboost  as xgb;  HAS_XGB = True
except ImportError:              HAS_XGB = False
try:    import lightgbm as lgb;  HAS_LGB = True
except ImportError:              HAS_LGB = False
try:    import catboost as cb;   HAS_CAT = True
except ImportError:              HAS_CAT = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError: HAS_OPTUNA = False
try:
    import torch
    import torch.nn as _nn
    import torch.optim as _optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError: HAS_TORCH = False

def _safe_torch_load(buf):
    """torch.load compatible with both old (<1.13) and new PyTorch."""
    try:
        return torch.load(buf, map_location="cpu", weights_only=True)
    except TypeError:
        buf.seek(0)
        return torch.load(buf, map_location="cpu")

APP_VERSION      = "4.0.0"
SHAP_BUNDLE_ROWS = 400
FRP_TYPES        = ["A", "B", "C", "G"]
PAPER_NUM_FEATS  = ["a/d", "d(mm)", "b(mm)", "f`c(Mpa)", "\u03c1f(%)", "Ef(GPa)"]

# ══════════════════════════════════════════════════════════════════════
#  1D-CNN sklearn-compatible wrapper  (requires PyTorch)
# ══════════════════════════════════════════════════════════════════════

class CNN1DRegressor(BaseEstimator, RegressorMixin):
    """
    1D-CNN regressor for tabular data, wrapped in sklearn API.

    Features are treated as a 1-channel 1D signal of length n_features.
    Architecture:
        Conv1d → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d
        → Flatten → Linear → ReLU → Dropout → Linear(1)

    Fully picklable: stores weights as bytes via state_dict.
    """
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
                _nn.Conv1d(1, f1, ks, padding=pad), _nn.ReLU(), _nn.BatchNorm1d(f1),
                _nn.Conv1d(f1, f2, ks, padding=pad), _nn.ReLU(), _nn.BatchNorm1d(f2),
                _nn.AdaptiveAvgPool1d(1),
            )
            self.head = _nn.Sequential(
                _nn.Linear(f2, hid), _nn.ReLU(), _nn.Dropout(drop),
                _nn.Linear(hid, 1),
            )
        def forward(self, x):
            # x: (batch, 1, n_features)
            h = self.conv(x).squeeze(-1)     # (batch, f2)
            return self.head(h).squeeze(-1)  # (batch,)

    def _build(self, n_features):
        torch.manual_seed(self.random_state)
        return self._Net(n_features, self.n_filters1, self.n_filters2,
                         self.kernel_size, self.hidden_dim, self.dropout)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        net = self._build(X.shape[1])
        opt = _optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = _nn.MSELoss()

        # 10% validation for early stopping
        n_val = max(1, int(len(X) * 0.1))
        idx = np.random.RandomState(self.random_state).permutation(len(X))
        Xt, Xv = X[idx[n_val:]], X[idx[:n_val]]
        yt, yv = y[idx[n_val:]], y[idx[:n_val]]

        ds = TensorDataset(torch.from_numpy(Xt).unsqueeze(1),
                           torch.from_numpy(yt))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        Xv_t = torch.from_numpy(Xv).unsqueeze(1)
        yv_t = torch.from_numpy(yv)

        best_loss, wait, best_state = 1e18, 0, None
        net.train()
        for epoch in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
            # validation
            net.eval()
            with torch.no_grad():
                vl = loss_fn(net(Xv_t), yv_t).item()
            net.train()
            if vl < best_loss:
                best_loss, wait = vl, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        # store weights as bytes for pickling
        buf = io.BytesIO()
        torch.save(net.state_dict(), buf)
        self._weights_bytes = buf.getvalue()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        net = self._build(X.shape[1])
        buf = io.BytesIO(self._weights_bytes)
        net.load_state_dict(_safe_torch_load(buf))
        net.eval()
        with torch.no_grad():
            pred = net(torch.from_numpy(X).unsqueeze(1)).numpy()
        return pred

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ══════════════════════════════════════════════════════════════════════
#  Column detection  (same as original)
# ══════════════════════════════════════════════════════════════════════

_INTERNAL = {
    "Vexp":     "Vexp(kN)", "d_mm": "d(mm)", "b_mm": "b(mm)",
    "fc_mpa":   "f`c(Mpa)", "rho_f": "\u03c1f(%)", "Ef_gpa": "Ef(GPa)",
    "ffu_mpa":  "ffu(MPa)", "frp_type": "FRP-type", "ad": "a/d",
    "stirrup_s": "s(mm)", "rho_fv": "\u03c1fv(%)",
}
_ALIASES = {
    "Vexp":     ["vexp(kn)","vexp(kN)","vexp(KN)","vu(kn)","vexp"],
    "d_mm":     ["d(mm)","d_mm","h0","h_0","h0(mm)"],
    "b_mm":     ["b(mm)","b_mm","bw","b_w"],
    "fc_mpa":   ["fc","f'c(mpa)","fck","f`c(mpa)","fcu","fc(mpa)"],
    "rho_f":    ["\u03c1f(%)","\u03c1f/%","\u03c1f","\u03c1f/\u914d\u7b4b\u7387",
                 "\u03c1f/\u914d\u7b4b\u7387.1","rhof(%)","rhof","rho_f(%)","rho_f","rf(%)","pf(%)"],
    "Ef_gpa":   ["ef(gpa)","ef_gpa","e_f","ef","ef(GPa)","ef(Gpa)"],
    "ffu_mpa":  ["ffu","ffu(mpa)","fu(mpa)","ffu_mpa"],
    "frp_type": ["frp-type","frp_type","frptype","type","frp type"],
    "ad":       ["a/d","ad","a_d","shear_span_ratio"],
    "stirrup_s":["s(mm)","s","spacing","\u7b4b\u7b4b\u95f4\u8ddd","stirrup_spacing","s_mm"],
    "rho_fv":   ["\u03c1fv(%)","rfv(%)","rho_fv","\u914d\u7b4b\u7387","\u03c1fv"],
}

def _strip(s):
    return re.sub(r"[\s\(\)\[\]_\-/]", "", str(s)).lower()

def _auto_map(raw_cols):
    low_map = {c.lower(): c for c in raw_cols}
    str_map = {_strip(c): c for c in raw_cols}
    result  = {}
    for key, aliases in _ALIASES.items():
        for a in aliases:
            if a.lower() in low_map: result[key] = low_map[a.lower()]; break
            if _strip(a) in str_map: result[key] = str_map[_strip(a)]; break
    return result

# ── Load & filter stirrup-free beams ─────────────────────────────────
def load_and_filter(path):
    ext = os.path.splitext(path)[1].lower()
    raw = (pd.read_csv(path) if ext==".csv" else
           pd.read_excel(path, engine="xlrd") if ext==".xls" else
           pd.read_excel(path))
    print(f"  Columns ({len(raw.columns)}): {raw.columns.tolist()}")
    mapping = _auto_map(raw.columns.tolist())
    print(f"  Mapped: { {k:v for k,v in mapping.items()} }\n")

    out = {}
    for key, col in mapping.items():
        if col in raw.columns:
            out[_INTERNAL[key]] = raw[col].values
    df = pd.DataFrame(out).reset_index(drop=True)
    df["Vexp(kN)"] = pd.to_numeric(df["Vexp(kN)"], errors="coerce")
    df = df[df["Vexp(kN)"].notna() & (df["Vexp(kN)"]>0)].reset_index(drop=True)
    n_total = len(df)

    has_s   = "s(mm)"  in df.columns
    has_rfv = "\u03c1fv(%)" in df.columns
    mask_rfv = (pd.to_numeric(df["\u03c1fv(%)"],errors="coerce").isna() |
                (pd.to_numeric(df["\u03c1fv(%)"],errors="coerce")==0)) if has_rfv \
               else pd.Series([True]*n_total)
    mask_s   = (pd.to_numeric(df["s(mm)"],errors="coerce").isna() |
                (pd.to_numeric(df["s(mm)"],errors="coerce")==0)) if has_s \
               else pd.Series([True]*n_total)
    sf_mask  = mask_rfv & mask_s

    if not has_rfv and not has_s:
        print("  [WARN] No stirrup column found — using all records.")
        df_sf = df.copy()
    else:
        df_sf = df[sf_mask].reset_index(drop=True)
        print(f"  Stirrup-free: {len(df_sf)} beams  "
              f"(excluded {n_total-len(df_sf)} with-stirrup beams)")
        if abs(len(df_sf)-581) > 30:
            print(f"  [WARN] Expected ~581, got {len(df_sf)} — check mapping")
    return df_sf, mapping

# ── Feature matrix ────────────────────────────────────────────────────
def build_features(df):
    num_cols = [c for c in PAPER_NUM_FEATS if c in df.columns]
    miss     = [c for c in PAPER_NUM_FEATS if c not in df.columns]
    if miss: print(f"  [WARN] Missing: {miss}")
    X_num = df[num_cols].values.astype(float)
    for j in range(X_num.shape[1]):
        bad = ~np.isfinite(X_num[:,j])
        if bad.any(): X_num[bad,j] = np.nanmedian(X_num[:,j])
    if "FRP-type" in df.columns:
        ohe   = OneHotEncoder(**_ohe_sparse_kwarg(), handle_unknown="ignore",
                              categories=[FRP_TYPES])
        X_cat = ohe.fit_transform(df[["FRP-type"]].astype(str))
        X     = np.hstack([X_num, X_cat])
        flabs = num_cols + [f"FRP={t}" for t in FRP_TYPES]
    else:
        X=X_num; flabs=num_cols; ohe=None
    y    = pd.to_numeric(df["Vexp(kN)"],errors="coerce").values
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & (y>0)
    print(f"  Feature matrix: {X[mask].shape}  features: {flabs}")
    return X[mask], y[mask], flabs, ohe

# ── Metrics ───────────────────────────────────────────────────────────
def _metrics(yt, yp):
    yt=np.asarray(yt,float); yp=np.asarray(yp,float)
    ok=np.isfinite(yt)&np.isfinite(yp)&(yt>0); yt,yp=yt[ok],yp[ok]
    if len(yt)<2: return {k:float("nan") for k in
        ("R2","r","RMSE","MAE","MAPE","mean_ratio","cov","safety_pct")}
    ratio=yp/yt
    return {"R2":round(float(r2_score(yt,yp)),4),
            "r":round(_pearson_r(yt,yp),4),
            "RMSE":round(float(np.sqrt(mean_squared_error(yt,yp))),3),
            "MAE":round(float(mean_absolute_error(yt,yp)),3),
            "MAPE":round(float(np.mean(np.abs((yt-yp)/yt))*100),2),
            "mean_ratio":round(float(ratio.mean()),4),
            "cov":round(float(ratio.std()/ratio.mean()),4),
            "safety_pct":round(float(np.mean(ratio<=1.0)*100),1)}

# ══════════════════════════════════════════════════════════════════════
#  Paper hyperparameters (warm-start for Bayesian search)
# ══════════════════════════════════════════════════════════════════════

PAPER_PARAMS = {
    "GBDT":         {"n_estimators":2300,"learning_rate":0.1483,
                     "min_samples_leaf":1,"min_samples_split":50,"max_depth":3},
    "XGBoost":      {"n_estimators":3000,"learning_rate":0.12,
                     "min_child_weight":2,"max_leaves":31,"max_depth":9},
    "LightGBM":     {"n_estimators":800,"learning_rate":0.1776,
                     "min_child_samples":1,"num_leaves":512,"max_depth":3},
    "CatBoost":     {"iterations":800,"learning_rate":0.1343,
                     "min_data_in_leaf":5,"l2_leaf_reg":9.6047,
                     "bagging_temperature":0.8238},
    "Random Forest":{"n_estimators":1200,"max_depth":25,
                     "min_samples_leaf":1,"min_samples_split":5,
                     "max_features":None},
    # ── New models ──
    "Extra Trees":  {"n_estimators":1500,"max_depth":30,
                     "min_samples_leaf":1,"min_samples_split":2,
                     "max_features":None},
    "MLP":          {"hidden1":256,"hidden2":128,"hidden3":64,
                     "learning_rate_init":0.005,"alpha":0.0001,
                     "batch_size":32},
    "SVR":          {"C":500.0,"epsilon":1.0,"gamma_val":0.05,
                     "kernel":"rbf"},
    "AdaBoost":     {"n_estimators":1000,"learning_rate":0.05,
                     "loss":"square"},
    "KNN":          {"n_neighbors":5,"weights":"distance","p":2},
    "1D-CNN":       {"n_filters1":64,"n_filters2":128,"kernel_size":3,
                     "hidden_dim":64,"dropout":0.2,"lr":0.001,
                     "epochs":300,"batch_size":32},
}

# ══════════════════════════════════════════════════════════════════════
#  Search spaces  (paper Tables 3-1~3-5 + new models)
# ══════════════════════════════════════════════════════════════════════

def _suggest(name, trial):
    # ── Original 5 tree models ──
    if name=="GBDT": return dict(
        n_estimators=trial.suggest_int("n_estimators",50,3000),
        learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
        min_samples_leaf=trial.suggest_int("min_samples_leaf",1,10),
        min_samples_split=trial.suggest_int("min_samples_split",2,100),
        max_depth=trial.suggest_int("max_depth",3,100))
    if name=="XGBoost": return dict(
        n_estimators=trial.suggest_int("n_estimators",50,3000),
        learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
        min_child_weight=trial.suggest_int("min_child_weight",2,100),
        max_leaves=trial.suggest_int("max_leaves",15,512),
        max_depth=trial.suggest_int("max_depth",3,100))
    if name=="LightGBM": return dict(
        n_estimators=trial.suggest_int("n_estimators",50,3000),
        learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
        min_child_samples=trial.suggest_int("min_child_samples",1,10),
        num_leaves=trial.suggest_int("num_leaves",15,512),
        max_depth=trial.suggest_int("max_depth",3,100))
    if name=="CatBoost": return dict(
        iterations=trial.suggest_int("iterations",50,3000),
        learning_rate=trial.suggest_float("learning_rate",0.01,0.2),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf",1,10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg",0,10),
        bagging_temperature=trial.suggest_float("bagging_temperature",0,1))
    if name=="Random Forest":
        mf=trial.suggest_categorical("max_features",["log2","sqrt","None"])
        return dict(
            n_estimators=trial.suggest_int("n_estimators",50,3000),
            max_depth=trial.suggest_int("max_depth",3,100),
            min_samples_leaf=trial.suggest_int("min_samples_leaf",1,10),
            min_samples_split=trial.suggest_int("min_samples_split",2,100),
            max_features=None if mf=="None" else mf)

    # ── Extra Trees ──
    if name=="Extra Trees":
        mf=trial.suggest_categorical("max_features",["log2","sqrt","None"])
        return dict(
            n_estimators=trial.suggest_int("n_estimators",50,3000),
            max_depth=trial.suggest_int("max_depth",3,100),
            min_samples_leaf=trial.suggest_int("min_samples_leaf",1,10),
            min_samples_split=trial.suggest_int("min_samples_split",2,100),
            max_features=None if mf=="None" else mf)

    # ── MLP ──
    if name=="MLP":
        h1 = trial.suggest_int("hidden1", 32, 256)
        h2 = trial.suggest_int("hidden2", 16, 128)
        h3 = trial.suggest_int("hidden3", 0, 64)
        return dict(
            hidden1=h1, hidden2=h2, hidden3=h3,
            learning_rate_init=trial.suggest_float("learning_rate_init",1e-4,0.01,log=True),
            alpha=trial.suggest_float("alpha",1e-5,0.1,log=True),
            batch_size=trial.suggest_categorical("batch_size",[16,32,64,128]))

    # ── SVR ──
    if name=="SVR":
        return dict(
            C=trial.suggest_float("C",0.1,1000,log=True),
            epsilon=trial.suggest_float("epsilon",0.001,1.0,log=True),
            gamma_val=trial.suggest_float("gamma_val",1e-4,1.0,log=True),
            kernel=trial.suggest_categorical("kernel",["rbf","poly"]))

    # ── AdaBoost ──
    if name=="AdaBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators",50,3000),
            learning_rate=trial.suggest_float("learning_rate",0.01,2.0,log=True),
            loss=trial.suggest_categorical("loss",["linear","square","exponential"]))

    # ── KNN ──
    if name=="KNN":
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors",1,30),
            weights=trial.suggest_categorical("weights",["uniform","distance"]),
            p=trial.suggest_int("p",1,3))

    # ── 1D-CNN ──
    if name=="1D-CNN":
        return dict(
            n_filters1=trial.suggest_categorical("n_filters1",[32,64,128]),
            n_filters2=trial.suggest_categorical("n_filters2",[64,128,256]),
            kernel_size=trial.suggest_int("kernel_size",2,5),
            hidden_dim=trial.suggest_categorical("hidden_dim",[32,64,128]),
            dropout=trial.suggest_float("dropout",0.0,0.5),
            lr=trial.suggest_float("lr",1e-4,0.01,log=True),
            epochs=trial.suggest_int("epochs",100,500),
            batch_size=trial.suggest_categorical("batch_size",[16,32,64]))

    raise ValueError(f"Unknown model: {name}")


# ══════════════════════════════════════════════════════════════════════
#  Model constructor
# ══════════════════════════════════════════════════════════════════════

def _make(name, params, seed):
    p = dict(params)

    # ── Original 5 ──
    if name=="GBDT":
        return GradientBoostingRegressor(random_state=seed, **p)
    if name=="XGBoost":
        return xgb.XGBRegressor(random_state=seed, verbosity=0, n_jobs=-1, **p)
    if name=="LightGBM":
        return lgb.LGBMRegressor(random_state=seed, verbose=-1, n_jobs=-1, **p)
    if name=="CatBoost":
        return cb.CatBoostRegressor(random_state=seed, verbose=0, **p)
    if name=="Random Forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **p)

    # ── Extra Trees ──
    if name=="Extra Trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **p)

    # ── MLP ──
    if name=="MLP":
        layers = [p.pop("hidden1"), p.pop("hidden2")]
        h3 = p.pop("hidden3")
        if h3 > 0:
            layers.append(h3)
        return MLPRegressor(
            hidden_layer_sizes=tuple(layers),
            activation="relu", solver="adam",
            max_iter=1000, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=30,
            random_state=seed, **p)

    # ── SVR ──
    if name=="SVR":
        gamma_val = p.pop("gamma_val")
        return SVR(gamma=gamma_val, **p)

    # ── AdaBoost ──
    if name=="AdaBoost":
        return AdaBoostRegressor(random_state=seed, **p)

    # ── KNN ──
    if name=="KNN":
        return KNeighborsRegressor(n_jobs=-1, **p)

    # ── 1D-CNN ──
    if name=="1D-CNN":
        return CNN1DRegressor(random_state=seed, **p)

    raise ValueError(f"Unknown model: {name}")


def _avail(name):
    if name=="XGBoost"  and not HAS_XGB:   return False
    if name=="LightGBM" and not HAS_LGB:   return False
    if name=="CatBoost" and not HAS_CAT:   return False
    if name=="1D-CNN"   and not HAS_TORCH: return False
    return True

# ── Models that should use n_jobs=1 in CV (avoid multiprocess issues) ─
_SINGLE_THREAD_MODELS = {"MLP", "SVR", "1D-CNN"}


# ══════════════════════════════════════════════════════════════════════
#  Bayesian search
# ══════════════════════════════════════════════════════════════════════

def bayesian_search(name, X_tr, y_tr, cv, n_trials, seed, time_limit=None):
    if not HAS_OPTUNA:
        print("    [SKIP] optuna not installed")
        return PAPER_PARAMS.get(name, {})

    # Reduce trials for slow models
    if name in _SINGLE_THREAD_MODELS and n_trials > 200:
        adj = min(n_trials, 200)
        print(f"    [NOTE] {name}: reducing trials {n_trials} → {adj} (slower model)")
        n_trials = adj

    t0 = time.time()
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    njobs = 1 if name in _SINGLE_THREAD_MODELS else -1

    def objective(trial):
        try:
            p = _suggest(name, trial)
            m = _make(name, p, seed)
            sc = cross_val_score(m, X_tr, y_tr, cv=kf,
                                 scoring="neg_mean_squared_error", n_jobs=njobs)
            return float(sc.mean())
        except Exception:
            return -1e9

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start with default params
    ws = PAPER_PARAMS.get(name, {})
    if ws:
        ws2 = {k: ("None" if k == "max_features" and v is None else v)
               for k, v in ws.items()}
        try:
            study.enqueue_trial(ws2)
        except Exception:
            pass

    def cb_fn(s, t):
        e = time.time() - t0
        if time_limit and e > time_limit:
            s.stop()
        if t.number > 0 and t.number % 50 == 0:
            print(f"      trial {t.number:4d}  "
                  f"CV MSE={-s.best_value:.2f}  {e:.0f}s")

    study.optimize(objective, n_trials=n_trials,
                   callbacks=[cb_fn], show_progress_bar=False)
    best_p = study.best_params.copy()
    if "max_features" in best_p and best_p["max_features"] == "None":
        best_p["max_features"] = None
    print(f"    [Bayesian] CV MSE={-study.best_value:.2f}  "
          f"trials={len(study.trials)}")
    print(f"    best: {best_p}")
    return best_p


# ══════════════════════════════════════════════════════════════════════
#  Picklable model wrapper
# ══════════════════════════════════════════════════════════════════════

class FittedModel:
    """Single fitted model, picklable (no lambdas)."""
    def __init__(self, model):
        self.model = model
        if hasattr(model, "feature_importances_"):
            self.feature_importances_ = model.feature_importances_
    def predict(self, X):
        return self.model.predict(X)


# ══════════════════════════════════════════════════════════════════════
#  Save bundle
# ══════════════════════════════════════════════════════════════════════

def _save(path, name, fmodel, scaler, feat_cols, ohe,
          tr_m, te_m, cv_mean, cv_std, X_all, shape, best_p,
          tr_pred=None, te_pred=None, y_tr=None, y_te=None):
    """Save bundle via ModelIO.save() — includes predictions & y_splits
    so the GUI scatter-plot can always be drawn after loading."""
    results = {
        name: {
            "model":      fmodel,
            "tr_pred":    np.asarray(tr_pred) if tr_pred is not None else np.array([]),
            "te_pred":    np.asarray(te_pred) if te_pred is not None else np.array([]),
            "tr_metrics": tr_m,
            "te_metrics": te_m,
            "cv_mean":    cv_mean,
            "cv_std":     cv_std,
            "_y_tr":      np.asarray(y_tr) if y_tr is not None else np.array([]),
            "_y_te":      np.asarray(y_te) if y_te is not None else np.array([]),
        }
    }
    ModelIO.save(
        path, results, scaler, feat_cols, ohe,
        X_all=X_all, X_shape=shape,
        extra_meta={
            "algo":        name,
            "best_params": best_p,
            "strategy":    "Bayesian-Optuna",
            "scaler_type": "MinMaxScaler",
            "filter":      "stirrup-free only",
            "note":        "6 numeric feats + OHE, MinMaxScaler [0,1]",
        }
    )
    print(f"    Saved → {path}")



# ══════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════

ALL_MODELS = [
    "GBDT", "XGBoost", "LightGBM", "CatBoost", "Random Forest",
    "Extra Trees", "MLP", "SVR", "AdaBoost", "KNN", "1D-CNN",
]

PAPER_REF = {
    "GBDT": 0.84, "XGBoost": 0.79, "LightGBM": 0.91,
    "CatBoost": 0.75, "Random Forest": 0.79,
}


def train(args):
    t0 = time.time()
    print("\n" + "=" * 70)
    print(f"  FRP-RC Extended Training  v{APP_VERSION}")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Models: {len(ALL_MODELS)} total  "
          f"({5} paper + {len(ALL_MODELS)-5} new)")
    print("=" * 70)

    print(f"\n[1] Loading & filtering: {args.data}")
    df_sf, mapping = load_and_filter(args.data)

    os.makedirs(args.outdir, exist_ok=True)
    filt_path = os.path.join(os.path.dirname(os.path.abspath(args.data)),
                             "dataset_stirrup_free_filtered.xlsx")
    df_sf.to_excel(filt_path, index=False)
    print(f"\n[2] Filtered dataset saved → {filt_path}")
    print(f"    Rows: {len(df_sf)}   Cols: {df_sf.columns.tolist()}")

    print(f"\n[3] Building features (Sec 2.4.3 & 2.4.4) …")
    X, y, feat_cols, ohe = build_features(df_sf)

    seed = args.seed
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test, random_state=seed)
    scaler = MinMaxScaler()
    scaler.fit(X_tr)
    Xts = scaler.transform(X_tr)
    Xes = scaler.transform(X_te)
    Xas = scaler.transform(X)

    print(f"  Total={len(y)}  Train={len(y_tr)}  Test={len(y_te)}")
    print(f"  Scaler: MinMaxScaler [0,1]  (paper eq.2-32)")
    print(f"  CV={args.cv}  Trials={args.trials}  Seed={seed}")

    names = [n for n in (args.only or ALL_MODELS) if _avail(n)]
    unavail = [n for n in (args.only or ALL_MODELS) if not _avail(n)]
    if unavail:
        print(f"  [SKIP] Unavailable: {unavail}")
    summary = []
    print(f"\n[4] Training {len(names)} model(s) …\n")

    for model_name in names:
        elapsed = (time.time() - t0) / 60
        print(f"{'─' * 70}")
        print(f"  {model_name}  (elapsed {elapsed:.1f} min)")
        per_lim = None
        if args.time_limit:
            rem = args.time_limit * 60 - (time.time() - t0)
            per_lim = max(60, rem / max(1, len(names) - names.index(model_name)) * 0.85)

        best_p = bayesian_search(model_name, Xts, y_tr,
                                 cv=args.cv, n_trials=args.trials,
                                 seed=seed, time_limit=per_lim)

        print("  Fitting final model …")
        model = _make(model_name, best_p, seed)
        model.fit(Xts, y_tr)

        tr_m = _metrics(y_tr, model.predict(Xts))
        te_m = _metrics(y_te, model.predict(Xes))

        njobs = 1 if model_name in _SINGLE_THREAD_MODELS else -1
        kf = KFold(n_splits=max(2, min(args.cv, len(Xts))),
                   shuffle=True, random_state=seed)
        cv_sc = cross_val_score(model, Xts, y_tr, cv=kf,
                                scoring="r2", n_jobs=njobs)
        cv_mean, cv_std = float(cv_sc.mean()), float(cv_sc.std())

        dt = (time.time() - t0) / 60
        ref = PAPER_REF.get(model_name, float("nan"))
        diff = f"{te_m['R2'] - ref:+.4f}" if ref == ref else "  —"
        print(f"\n  R²(train)={tr_m['R2']:.4f}  RMSE(train)={tr_m['RMSE']:.2f}kN")
        print(f"  R²(test) ={te_m['R2']:.4f}  RMSE(test) ={te_m['RMSE']:.2f}kN  "
              f"r={te_m['r']:.4f}  MAE={te_m['MAE']:.2f}kN")
        print(f"  CV R²    ={cv_mean:.4f}±{cv_std:.4f}  "
              f"vs paper R²={ref}  diff={diff}  [{dt:.1f}min]")

        out = os.path.join(
            args.outdir,
            model_name.replace(" ", "_").replace("-", "_") + ".frpmdl")
        _tr_pred = model.predict(Xts)
        _te_pred = model.predict(Xes)
        _save(out, model_name, FittedModel(model), scaler, feat_cols, ohe,
              tr_m, te_m, cv_mean, cv_std, Xas, X.shape, best_p,
              tr_pred=_tr_pred, te_pred=_te_pred, y_tr=y_tr, y_te=y_te)
        summary.append((model_name, te_m["R2"], te_m["RMSE"],
                         te_m["r"], cv_mean, cv_std))

    total = (time.time() - t0) / 60
    print("\n" + "=" * 70)
    print(f"  Done  |  {total:.1f} min  |  {len(summary)} model(s)")
    print(f"  {'Model':22}  {'R²(test)':>8}  {'r':>6}  {'RMSE(kN)':>9}  "
          f"{'CV R²':>8}  {'vs paper':>9}")
    print("  " + "-" * 68)
    for nm, r2, rmse, r, cvm, cvs in sorted(summary, key=lambda x: -x[1]):
        ref = PAPER_REF.get(nm, float("nan"))
        diff = f"{r2 - ref:+.4f}" if ref == ref else "       —"
        print(f"  {nm:22}  {r2:>8.4f}  {r:>6.4f}  {rmse:>9.2f}  "
              f"{cvm:>8.4f}  {diff:>9}")
    print("=" * 70)
    print(f"\n  Filtered dataset : {filt_path}")
    print(f"  Model bundles    : {os.path.abspath(args.outdir)}/")
    print("  Copy *.frpmdl to FRP.py directory.\n")


def main():
    ap = argparse.ArgumentParser(
        description="FRP-RC extended trainer (11 models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--data",       required=True)
    ap.add_argument("--outdir",     default="models")
    ap.add_argument("--test",       type=float, default=0.20)
    ap.add_argument("--cv",         type=int,   default=10)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--trials",     type=int,   default=1000,
                    help="Bayesian trials per model (paper uses 1000)")
    ap.add_argument("--time-limit", type=int,   default=None, dest="time_limit",
                    help="Total wall-clock limit in minutes")
    ap.add_argument("--only",       nargs="+",  default=None, metavar="MODEL",
                    help=f"Train only specified models. Available: {ALL_MODELS}")
    args = ap.parse_args()
    if not os.path.isfile(args.data):
        print(f"[ERROR] File not found: {args.data}")
        sys.exit(1)
    if not HAS_OPTUNA:
        print("[WARN] optuna not installed — using default params")

    # Print availability
    print("\n  Library check:")
    for lib, flag, label in [
        ("xgboost",  HAS_XGB,   "XGBoost"),
        ("lightgbm", HAS_LGB,   "LightGBM"),
        ("catboost", HAS_CAT,   "CatBoost"),
        ("torch",    HAS_TORCH, "1D-CNN"),
        ("optuna",   HAS_OPTUNA,"Bayesian Search"),
    ]:
        status = "✓" if flag else "✗ (pip install " + lib + ")"
        print(f"    {label:20s} {status}")

    train(args)


if __name__ == "__main__":
    main()
