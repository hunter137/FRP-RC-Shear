"""
config.py — Global constants, colour palette, feature definitions.
"""
APP_VERSION = '2.0.0'
_SHAP_BUNDLE_SAMPLES = 400

# ── Optional-library availability flags ─────────────────────────────
try:    import xgboost  as xgb; HAS_XGB    = True
except ImportError:             HAS_XGB    = False
try:    import lightgbm as lgb; HAS_LGB    = True
except ImportError:             HAS_LGB    = False
try:    import catboost as cb;  HAS_CAT    = True
except ImportError:             HAS_CAT    = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:             HAS_OPTUNA = False
try:    import shap;            HAS_SHAP   = True
except ImportError:             HAS_SHAP   = False
try:    import torch;           HAS_TORCH  = True
except ImportError:             HAS_TORCH  = False
try:
    from sklearn.inspection import PartialDependenceDisplay
    HAS_PDP = True
except ImportError:             HAS_PDP    = False
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2          as _NSGA2
    from pymoo.core.problem          import Problem        as _Problem
    from pymoo.optimize              import minimize       as _pymoo_min
    from pymoo.termination           import get_termination
    HAS_PYMOO = True
except ImportError:             HAS_PYMOO  = False

# ── GPU / CUDA detection ─────────────────────────────────────────────
HAS_CUDA        = False
CUDA_DEVICE_NAME = 'Not detected'
try:
    import torch
    if torch.cuda.is_available():
        HAS_CUDA         = True
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
except Exception:
    pass

if not HAS_CUDA:
    try:
        import subprocess as _sp
        _r = _sp.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=3)
        if _r.returncode == 0 and _r.stdout.strip():
            HAS_CUDA         = True
            CUDA_DEVICE_NAME = _r.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass

# ═══════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════
C_WIN_BG    = '#F5F5F5'
C_PANEL_BG  = '#FFFFFF'
C_ALT_ROW   = '#F9F9F9'
C_ACCENT    = '#2B6CB0'
C_ACCENT_LT = '#6EA8D8'
C_ACCENT_BG = '#EDF4FB'
C_TEXT      = '#111111'
C_TEXT2     = '#555555'
C_BORDER    = '#CCCCCC'
C_BORDER_LT = '#E2E2E2'
C_SUCCESS   = '#276227'
C_SUCCESS_BG= '#EBF5EB'
C_DANGER    = '#B30000'
C_HEADER_BG = '#EFEFEF'

ALGO_COLORS = {
    'GBDT':          '#C0392B',
    'XGBoost':       '#1A7A40',
    'LightGBM':      '#1A5FA0',
    'CatBoost':      '#7B2D8B',
    'Random Forest': '#4A4A8A',
}
CODE_COLORS = {
    'GB 50608-2020': '#C0392B',
    'ACI 440.1R-15': '#D35400',
    'CSA S806-12':   '#7A4D00',
    'BISE (1999)':   '#5B2D8A',
    'JSCE (1997)':   '#276227',
    'Proposed':      '#1A5FA0',
}

# ── Feature definitions ──────────────────────────────────────────────
FRP_TYPES     = ['A', 'B', 'C', 'G']
NUM_FEAT_COLS = ['a/d', 'd(mm)', 'b(mm)', "f`c(Mpa)", '\u03c1f(%)', 'Ef(GPa)']
FEAT_LABELS   = NUM_FEAT_COLS + [f'FRP={t}' for t in FRP_TYPES]

# ── Matplotlib backend — auto-detect, never crash ────────────────────
# Do NOT call matplotlib.use() here at module level: it runs on every
# `import config` (including from train_frp_models.py and tests) and
# will crash headless environments.  Backend selection is deferred to
# _configure_mpl() which is called only inside the GUI entry point.

def _configure_mpl():
    """
    Select the best available Qt/Tk matplotlib backend and apply
    journal-quality defaults.  Safe to call multiple times (no-op after
    the first successful switch).

    Backend priority:
      Qt5Agg  (PyQt5 / PySide2)
      Qt6Agg  (PyQt6 / PySide6)
      TkAgg   (Tcl/Tk — ships with CPython, always available on Windows/macOS)
      Agg     (headless fallback — no interactive window, but no crash)
    """
    import matplotlib
    import matplotlib.pyplot as plt

    current = matplotlib.get_backend().lower()
    # Already set to something interactive — leave it alone
    _interactive = {'qt5agg', 'qt6agg', 'tkagg', 'wxagg', 'macosx'}
    if current in _interactive:
        pass
    else:
        for backend in ('Qt5Agg', 'Qt6Agg', 'TkAgg', 'Agg'):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue

    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ('DejaVu Sans', 'Arial', 'Liberation Sans'):
        if name in available:
            plt.rcParams['font.family'] = name
            break

    plt.rcParams.update({
        'axes.unicode_minus':  False,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.grid':           True,
        'axes.facecolor':      '#FFFFFF',
        'figure.facecolor':    '#FFFFFF',
        'grid.alpha':          0.22,
        'grid.linewidth':      0.5,
        'grid.color':          '#CCCCCC',
        'xtick.direction':     'in',
        'ytick.direction':     'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.top':           False,
        'ytick.right':         False,
        'axes.linewidth':      0.8,
        'axes.edgecolor':      '#888888',
        'font.size':           10,
    })

# ── Variable-name dictionaries ───────────────────────────────────────
VAR_LATEX = {
    'Vexp(kN)': r'$V_\mathrm{exp}$\,(kN)',
    'a/d':       r'$a/d$',
    'd(mm)':     r'$d$\,(mm)',
    'b(mm)':     r'$b$\,(mm)',
    "f`c(Mpa)":  r"$f'_\mathrm{c}$\,(MPa)",
    'ρf(%)':     r'$\rho_\mathrm{f}$\,(%)',
    'Ef(GPa)':   r'$E_\mathrm{f}$\,(GPa)',
    'FRP=A':     'AFRP',
    'FRP=B':     'BFRP',
    'FRP=C':     'CFRP',
    'FRP=G':     'GFRP',
}

VAR_PLAIN = {
    'Vexp(kN)': 'V\u2091\u2093\u2095 (kN)',
    'a/d':       'a/d',
    'd(mm)':     'd (mm)',
    'b(mm)':     'b (mm)',
    "f`c(Mpa)":  "f\u2032c (MPa)",
    'ρf(%)':     '\u03c1f (%)',
    'Ef(GPa)':   'Ef (GPa)',
    'FRP=A':     'AFRP',
    'FRP=B':     'BFRP',
    'FRP=C':     'CFRP',
    'FRP=G':     'GFRP',
}

PRED_COLS = ['Vexp(kN)'] + NUM_FEAT_COLS   # type: list[str]
