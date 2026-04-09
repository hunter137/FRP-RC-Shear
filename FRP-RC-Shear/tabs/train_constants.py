"""
train_constants.py — Shared constants and helper functions for the training module.

Contents
--------
_ohe_sparse_kwarg   sklearn OHE sparse-keyword compatibility shim
_ALGO_CATALOGUE     Master list of algorithm names, groups, and default params
_PARAM_LABELS       Human-readable labels and search bounds for each hyperparameter
_CURVE_PALETTE      Colour palette for the live optimisation curve
_is_available       Returns True when an optional backend library is installed
"""
from sklearn.preprocessing import OneHotEncoder
from config import HAS_XGB, HAS_LGB, HAS_CAT, HAS_TORCH

# ── PyQt5 — complete import set inherited from the original train_tab ──
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QTextEdit, QPlainTextEdit, QProgressBar, QMessageBox,
    QRadioButton, QFileDialog, QDialog, QDialogButtonBox,
    QScrollArea, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QListWidget, QListWidgetItem, QGroupBox,
    QAbstractItemView, QSizePolicy, QPushButton, QStackedWidget,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPixmap, QPainter, QPen, QBrush

# ── config — complete set ──────────────────────────────────────────────
from config import (
    APP_VERSION, _SHAP_BUNDLE_SAMPLES,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    C_WIN_BG, C_PANEL_BG, C_ALT_ROW, C_HEADER_BG,
    C_SUCCESS, C_SUCCESS_BG, C_DANGER,
    HAS_XGB, HAS_LGB, HAS_CAT, HAS_OPTUNA, HAS_PYMOO, HAS_TORCH,
    HAS_CUDA, HAS_SHAP, CUDA_DEVICE_NAME,
    NUM_FEAT_COLS, FRP_TYPES, FEAT_LABELS,
    ALGO_COLORS, CODE_COLORS,
)
from optimization import (
    _ps_gbdt, _ps_xgb, _ps_lgb, _ps_cat, _ps_rf,
    _factory_for, tlbo_optimize, _optuna_optimize, nsga2_optimize,
    PARAM_SPACES, NSGA2_OBJECTIVES, _GPU_CAPABLE,
)


# ── sklearn OneHotEncoder sparse kwarg shim ──────────────────────────
# sklearn < 1.2 : sparse=False  |  sklearn >= 1.2 : sparse_output=False
def _ohe_sparse_kwarg():
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

from config import (
    APP_VERSION, _SHAP_BUNDLE_SAMPLES,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    HAS_XGB, HAS_LGB, HAS_CAT, HAS_OPTUNA, HAS_PYMOO, HAS_TORCH,
    NUM_FEAT_COLS, FRP_TYPES,
    HAS_CUDA, CUDA_DEVICE_NAME,
)
from widgets import flat_btn, panel, _stat_textbox
from metrics import calc_metrics
from optimization import (
    _ps_gbdt, _ps_xgb, _ps_lgb, _ps_cat, _ps_rf,
    _factory_for, tlbo_optimize, _optuna_optimize, nsga2_optimize,
    PARAM_SPACES, NSGA2_OBJECTIVES, _GPU_CAPABLE,
)
from model_io import FittedModel, ModelIO

if HAS_XGB:    import xgboost as xgb
if HAS_LGB:    import lightgbm as lgb
if HAS_CAT:    import catboost as cb

# ── Algorithm catalogue ───────────────────────────────────────────────
# Default hyperparameters set to the paper's optimal values for the
# no-stirrup FRP-RC beam dataset (Table 3-6):
#   GBDT:     n_est=2300, lr=0.1483, min_split=50, min_leaf=1, depth=3
#   XGBoost:  n_est=3000, lr=0.1200, min_child_weight=2, max_leaves=31, depth=9
#   LightGBM: n_est=800,  lr=0.1776, num_leaves=512, min_child_samples=1, depth=3
#   CatBoost: n_est=800,  lr=0.1343, min_data_in_leaf=5, l2=9.60, bag_temp=0.82
#   RF:       n_est=1200, depth=25, min_leaf=1, min_split=5, max_features=1.0(None)
_ALGO_CATALOGUE = [
    ('GBDT',         'Boosting', None,      {'n_estimators':2300,'max_depth':3,'learning_rate':0.1483,'min_samples_split':50,'min_samples_leaf':1}),
    ('XGBoost',      'Boosting', 'xgboost', {'n_estimators':3000,'max_depth':9,'learning_rate':0.1200,'min_child_weight':2,'subsample':0.8}),
    ('LightGBM',     'Boosting', 'lightgbm',{'n_estimators':800, 'max_depth':3,'learning_rate':0.1776,'num_leaves':512,'min_child_samples':1}),
    ('CatBoost',     'Boosting', 'catboost',{'iterations':800,   'learning_rate':0.1343,'min_data_in_leaf':5,'l2_leaf_reg':9.6,'bagging_temperature':0.82}),
    ('AdaBoost',     'Boosting', None,      {'n_estimators':100,'learning_rate':0.5}),
    ('Random Forest','Bagging',  None,      {'n_estimators':1200,'max_depth':25,'min_samples_leaf':1,'min_samples_split':5,'max_features':1.0}),
    ('Extra Trees',  'Bagging',  None,      {'n_estimators':500,'max_depth':20,'min_samples_leaf':1,'min_samples_split':4,'max_features':0.7}),
    ('KNN',          'Instance', None,      {'n_neighbors':5,'leaf_size':30}),
    ('MLP',          'Neural',   None,      {'hidden_layer_sizes_n':128,'alpha':0.001,'learning_rate_init':0.001,'max_iter':2000}),
    ('SVR',          'Kernel',   None,      {'C':10.0,'epsilon':0.1}),
]

_PARAM_LABELS = {
    # ── Common tree params ─────────────────────────────────────────────
    'n_estimators':         ('No. estimators',    50,  3000, False),
    'iterations':           ('Iterations',         50,  3000, False),
    'max_depth':            ('Max depth',           3,    30, False),
    'depth':                ('Max depth',           2,    10, False),
    'learning_rate':        ('Learning rate',    0.005,  0.3, True),
    'learning_rate_init':   ('LR (init)',       0.0001,  0.1, True),
    # ── GBDT-specific ─────────────────────────────────────────────────
    'min_samples_split':    ('Min split',            2,  100, False),
    'min_samples_leaf':     ('Min leaf samples',     1,   10, False),
    # ── XGBoost-specific ──────────────────────────────────────────────
    'min_child_weight':     ('Min child weight',     2,  100, False),
    'max_leaves':           ('Max leaves',          15,  512, False),
    # ── LightGBM-specific ─────────────────────────────────────────────
    'num_leaves':           ('Num leaves',          15,  512, False),
    'min_child_samples':    ('Min child samples',    1,   10, False),
    # ── CatBoost-specific ─────────────────────────────────────────────
    'l2_leaf_reg':          ('L2 reg',              0.0, 10.0, True),
    'min_data_in_leaf':     ('Min leaf samples',    1,    10, False),
    'bagging_temperature':  ('Bagging temp',        0.0,  1.0, True),
    # ── RF / ET ───────────────────────────────────────────────────────
    'max_features':         ('Max features',        0.1,  1.0, True),
    # ── Other ─────────────────────────────────────────────────────────
    'subsample':            ('Subsample',           0.4,  1.0, True),
    'colsample_bytree':     ('Col sample',          0.4,  1.0, True),
    'reg_alpha':            ('L1 reg (α)',         1e-4, 10.0, True),
    'n_neighbors':          ('Neighbors',             2,   30, False),
    'leaf_size':            ('Leaf size',            10,   80, False),
    'hidden_layer_sizes_n': ('Hidden units',         32,  512, False),
    'alpha':                ('Weight decay',        1e-6,  0.1, True),
    'C':                    ('Regulariz. C',        0.01, 200.0, True),
    'epsilon':              ('Epsilon',            0.001,  2.0, True),
}

# Colour palette for optimisation curves
_CURVE_PALETTE = [
    '#1f77b4','#ff7f0e','#2ca02c','#d62728',
    '#9467bd','#8c564b','#e377c2','#7f7f7f',
    '#bcbd22','#17becf',
]

def _is_available(requires):
    if requires is None: return True
    return {'xgboost': HAS_XGB, 'lightgbm': HAS_LGB,
            'catboost': HAS_CAT, 'torch': HAS_TORCH}.get(requires, True)


# ══════════════════════════════════════════════════════════════════════
#  _TLBOPreviewThread  — lightweight TLBO run inside the config dialog
# ══════════════════════════════════════════════════════════════════════
