import warnings
import traceback
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QSpinBox, QRadioButton, QCheckBox,
    QTextEdit, QProgressBar, QMessageBox, QFileDialog,
    QDialog, QStackedWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

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
    _factory_for, PARAM_SPACES, NSGA2_OBJECTIVES, _GPU_CAPABLE,
)
from model_io import FittedModel, ModelIO

from .train_constants import (
    _ohe_sparse_kwarg, _ALGO_CATALOGUE, _CURVE_PALETTE, _is_available,
)
from .train_threads import TrainingThread
from .train_hyperparams import (
    AlgorithmConfigDialog, TLBOMultiOptimizeDialog, SearchRangeDialog,
)
from .train_dialogs import (
    BundleFolderDialog, Nsga2ObjectiveDialog,
    TlboSettingsDialog, BayesianSettingsDialog,
    PreTrainingDialog, _TrainSummaryDialog,
)

if HAS_XGB:    import xgboost as xgb
if HAS_LGB:    import lightgbm as lgb
if HAS_CAT:    import catboost as cb

class TrainTab(QWidget):
    sig_done = pyqtSignal(dict, object, object, object, object, object, object)

    def __init__(self):
        super().__init__()
        self.df        = None
        self.scaler    = MinMaxScaler()
        self.feat_cols = None
        self.ohe       = None
        self._results  = {}
        self._X_all    = None
        self._y_all    = None
        self._algo_selection = {n: _is_available(r)
                                for n, _, r, _ in _ALGO_CATALOGUE}
        self._algo_params    = {n: dict(dp)
                                for n, _, _, dp in _ALGO_CATALOGUE}
        self._algo_locked    = {}   # {algo: set_of_locked_param_names}
        self._custom_ranges  = {}   # {algo: {param: (lo, hi)}}
        self._nsga2_objectives = ['R2', 'safety_pct']   # default bi-objective targets
        self._tlbo_settings    = {'mode': 'auto', 'n_pop': 10, 'n_iter': 10}
        self._bayes_settings   = {'n_startup_trials': 10, 'multivariate': False}
        self._use_gpu          = False
        # Live-curve data: {algo_name: [(eval_idx, best_cv), ...]}
        self._curve_data = {}
        self._build_ui()

    def set_data(self, df):
        self.df = df

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        left = QWidget()
        left.setFixedWidth(330)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)
        ll.setContentsMargins(0, 0, 0, 0)

        # Predictive Models
        algo_grp = panel('Algorithm Selection')
        ag = QVBoxLayout(algo_grp)
        ag.setSpacing(5)
        self._algo_summary = QLabel(self._algo_summary_text())
        self._algo_summary.setWordWrap(True)
        self._algo_summary.setStyleSheet(f'font-size:11px;color:{C_TEXT2};')
        ag.addWidget(self._algo_summary)
        self._cfg_algo_btn = flat_btn('Algorithm Configuration', accent=True)
        self._cfg_algo_btn.setFixedHeight(28)
        self._cfg_algo_btn.setToolTip(
            'Select algorithms and set hyperparameters.\n'
            'Supports parameter locking and TLBO preview.')
        self._cfg_algo_btn.clicked.connect(self._open_algo_dialog)
        ag.addWidget(self._cfg_algo_btn)
        ll.addWidget(algo_grp)

        # Data Partitioning
        cfg_grp = panel('Data Partitioning')
        cg = QGridLayout(cfg_grp)
        cg.setSpacing(5)
        cg.setColumnStretch(1, 1)
        self.cfg = {}
        for i, (lbl, key, mn, mx, dv, tip) in enumerate([
                ('Test Set (%):',   'test_pct', 5, 40, 20,
                 'Proportion of samples reserved for hold-out evaluation.'),
                ('Cross-Val. Folds:', 'cv_folds', 2, 20, 10,
                 'Number of folds used in k-fold cross-validation.'),
                ('Random Seed:',    'seed',     0, 9999, 42,
                 'Random state for reproducible train/test splits.'),
        ]):
            lbl_w = QLabel(lbl)
            lbl_w.setStyleSheet('font-size:11px;')
            lbl_w.setToolTip(tip)
            cg.addWidget(lbl_w, i, 0)
            w = QSpinBox()
            w.setRange(mn, mx); w.setValue(dv)
            w.setStyleSheet('font-size:11px;')
            self.cfg[key] = w
            cg.addWidget(w, i, 1)
        ll.addWidget(cfg_grp)

        # Optimisation Strategy
        opt_grp = panel('Optimisation Strategy')
        og = QVBoxLayout(opt_grp)
        og.setSpacing(4)
        self.opt_btns = {}
        for key, lbl, avail, tip in [
            ('none',
             'Fixed Hyperparameters',
             True,
             'Use exact values set in Model & Parameter Setup. No search is performed.'),
            ('bayesian',
             'Bayesian Optimisation  (Optuna TPE)',
             HAS_OPTUNA,
             'Bayesian optimisation via Optuna TPE sampler.'),
            ('tlbo',
             'TLBO  (Teaching-Learning)',
             True,
             'Teaching-Learning Based Optimisation — population-based metaheuristic.'),
            ('nsga2',
             'NSGA-II  (Multi-Objective)',
             HAS_PYMOO,
             'Multi-objective genetic algorithm — objectives configurable via "Objectives …".'),
        ]:
            suffix = ('' if avail else
                      '  [pip install optuna]' if key == 'bayesian'
                      else '  [pip install pymoo]')

            rb = QRadioButton(lbl + suffix)
            rb.setChecked(key == 'bayesian')   # Paper uses Bayesian (1000 trials)
            rb.setEnabled(avail)
            rb.setStyleSheet('font-size:11px;')
            rb.setToolTip(tip)
            rb.toggled.connect(self._on_strategy_changed)
            self.opt_btns[key] = rb

            if key in ('bayesian', 'tlbo', 'nsga2'):
                # These three strategies each get a config button on the right
                row_layout = QHBoxLayout()
                row_layout.setSpacing(4)
                row_layout.addWidget(rb)
                row_layout.addStretch()

                if key == 'bayesian':
                    self._bayes_btn = flat_btn('Settings')
                    self._bayes_btn.setFixedHeight(22)
                    self._bayes_btn.setFixedWidth(70)
                    self._bayes_btn.setEnabled(False)
                    self._bayes_btn.setToolTip(
                        'Optuna TPE sampler settings.\n'
                        'Only available when Bayesian strategy is selected.')
                    self._bayes_btn.clicked.connect(self._open_bayes_dialog)
                    row_layout.addWidget(self._bayes_btn)

                elif key == 'tlbo':
                    self._tlbo_btn = flat_btn('Settings')
                    self._tlbo_btn.setFixedHeight(22)
                    self._tlbo_btn.setFixedWidth(70)
                    self._tlbo_btn.setEnabled(False)
                    self._tlbo_btn.setToolTip(
                        'TLBO search settings.\n'
                        'Only available when TLBO strategy is selected.')
                    self._tlbo_btn.clicked.connect(self._open_tlbo_dialog)
                    row_layout.addWidget(self._tlbo_btn)

                elif key == 'nsga2':
                    self._obj_btn = flat_btn('Objectives')
                    self._obj_btn.setFixedHeight(22)
                    self._obj_btn.setFixedWidth(70)
                    self._obj_btn.setEnabled(False)
                    self._obj_btn.setToolTip(
                        'Choose the two metrics NSGA-II will optimise simultaneously.\n'
                        'Only available when NSGA-II strategy is selected.')
                    self._obj_btn.clicked.connect(self._open_obj_dialog)
                    row_layout.addWidget(self._obj_btn)

                og.addLayout(row_layout)
            else:
                og.addWidget(rb)

        sep_line = QFrame()
        sep_line.setFrameShape(QFrame.HLine)
        sep_line.setStyleSheet(f'color:{C_BORDER_LT};')
        og.addWidget(sep_line)

        trials_row = QHBoxLayout()
        trials_row.setSpacing(4)
        t_lbl = QLabel('Max. Iterations:')
        t_lbl.setStyleSheet('font-size:11px;')
        t_lbl.setToolTip(
            'Total hyperparameter evaluations per model\n'
            '(ignored when Fixed Parameters is active).')
        trials_row.addWidget(t_lbl)
        self.opt_trials = QSpinBox()
        self.opt_trials.setRange(10, 2000); self.opt_trials.setValue(300)
        self.opt_trials.setFixedWidth(58)
        self.opt_trials.setStyleSheet('font-size:11px;')
        trials_row.addWidget(self.opt_trials)
        trials_row.addStretch()
        self._range_btn = flat_btn('Search Bounds')
        self._range_btn.setFixedHeight(24)
        self._range_btn.setEnabled(False)
        self._range_btn.setToolTip(
            'Specify the minimum and maximum search bounds for each hyperparameter.\n'
            '(Available for all non-fixed search strategies.)')
        self._range_btn.clicked.connect(self._open_range_dialog)
        trials_row.addWidget(self._range_btn)
        og.addLayout(trials_row)

        # Early Stop row (Bayesian only)
        es_row = QHBoxLayout()
        es_row.setSpacing(4)
        self._early_stop_cb = QCheckBox('Early Stop')
        self._early_stop_cb.setChecked(True)
        self._early_stop_cb.setStyleSheet('font-size:11px;')
        self._early_stop_cb.setToolTip(
            'Halt Bayesian search early if no improvement is found\n'
            'after the specified number of consecutive trials.\n'
            'Ignored for Fixed / TLBO / NSGA-II strategies.')
        es_row.addWidget(self._early_stop_cb)
        p_lbl = QLabel('Patience:')
        p_lbl.setStyleSheet('font-size:11px;')
        p_lbl.setToolTip('Consecutive no-improvement trials before stopping.')
        es_row.addWidget(p_lbl)
        self._patience_sp = QSpinBox()
        self._patience_sp.setRange(10, 500)
        self._patience_sp.setValue(80)
        self._patience_sp.setFixedWidth(58)
        self._patience_sp.setStyleSheet('font-size:11px;')
        self._patience_sp.setToolTip('Default 80 — stop after 80 trials with no improvement.')
        # Patience spinbox enabled only when checkbox is ticked
        self._patience_sp.setEnabled(self._early_stop_cb.isChecked())
        self._early_stop_cb.toggled.connect(self._patience_sp.setEnabled)
        es_row.addWidget(self._patience_sp)
        es_row.addStretch()
        og.addLayout(es_row)

        ll.addWidget(opt_grp)

        # Run
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.run_btn  = flat_btn('Train Models', accent=True)
        self.run_btn.setFixedHeight(32)
        self.run_btn.setToolTip('Train all selected models and compute evaluation metrics.')
        self.stop_btn = flat_btn('Stop')
        self.stop_btn.setFixedHeight(32)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setToolTip('Stop the current training run.')
        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        ll.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setStyleSheet(
            f'QProgressBar{{border:1px solid {C_BORDER};border-radius:2px;'
            f'background:#F0F0F0;height:12px;text-align:center;font-size:10px;}}'
            f'QProgressBar::chunk{{background:{C_ACCENT};}}')
        ll.addWidget(self.progress)

        self._eta_lbl = QLabel('')
        self._eta_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        self._eta_lbl.setWordWrap(True)
        ll.addWidget(self._eta_lbl)

        # Model Archive
        io_grp = panel('Model Archive')
        ig = QVBoxLayout(io_grp)
        ig.setSpacing(4)
        self.save_btn = flat_btn('Save Bundle')
        self.save_btn.setEnabled(False)
        self.save_btn.setToolTip(
            'Save all trained models as a .frpmdl bundle.\n'
            '(Enabled automatically after Run Training completes.)')
        self.save_btn.clicked.connect(self._save)
        ig.addWidget(self.save_btn)
        load_file_btn = flat_btn('Load Bundle')
        load_file_btn.setToolTip('Load a single .frpmdl bundle file.')
        load_file_btn.clicked.connect(self._load_single)
        ig.addWidget(load_file_btn)
        load_folder_btn = flat_btn('Load Folder')
        load_folder_btn.setToolTip('Load all .frpmdl files from a selected folder.')
        load_folder_btn.clicked.connect(self._load_folder)
        ig.addWidget(load_folder_btn)
        ll.addWidget(io_grp)
        ll.addStretch()
        root.addWidget(left)

        right_splitter = QSplitter(Qt.Vertical)

        log_grp = panel('Training Log')
        lg      = QVBoxLayout(log_grp)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont('Courier New', 10))
        self.log.setStyleSheet(
            f'background:#FEFEFE;color:{C_TEXT};'
            f'border:1px solid {C_BORDER};padding:4px;')
        clear_btn = flat_btn('Clear')
        clear_btn.setFixedWidth(90)
        clear_btn.clicked.connect(self.log.clear)
        lg.addWidget(self.log)
        lg.addWidget(clear_btn)
        right_splitter.addWidget(log_grp)

        # Bottom panel: stacked — curve (TLBO/Bayesian) or summary table (Fixed)
        self._bottom_panel = panel('Training Summary')
        bp_layout = QVBoxLayout(self._bottom_panel)
        bp_layout.setContentsMargins(4, 4, 4, 4)
        self._bottom_stack = QStackedWidget()

        # Page 0: live optimisation curve
        curve_page = QWidget()
        cp_layout  = QVBoxLayout(curve_page)
        cp_layout.setContentsMargins(0, 0, 0, 0)
        self._curve_fig    = mfig.Figure(figsize=(8, 3.2), dpi=100)
        self._curve_canvas = FigureCanvas(self._curve_fig)
        self._curve_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        cp_layout.addWidget(self._curve_canvas)
        self._bottom_stack.addWidget(curve_page)   # index 0

        # Page 1: post-training summary table
        tbl_page = QWidget()
        tp_layout = QVBoxLayout(tbl_page)
        tp_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_tbl = QTableWidget()
        self._summary_tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self._summary_tbl.setAlternatingRowColors(True)
        self._summary_tbl.verticalHeader().setVisible(False)
        self._summary_tbl.setStyleSheet('font-size:11px;')
        self._summary_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        tp_layout.addWidget(self._summary_tbl)
        self._bottom_stack.addWidget(tbl_page)     # index 1

        bp_layout.addWidget(self._bottom_stack)
        right_splitter.addWidget(self._bottom_panel)

        right_splitter.setSizes([440, 260])
        root.addWidget(right_splitter)

        self._init_curve_plot()

        # radio buttons are pre-checked in code (no toggle signal fires),
        # so call the handler once explicitly to sync all dependent controls.
        self._on_strategy_changed()

    def _on_strategy_changed(self):
        manual   = self.opt_btns['none'].isChecked()
        bayesian = self.opt_btns['bayesian'].isChecked()
        tlbo     = self.opt_btns['tlbo'].isChecked()
        nsga2    = self.opt_btns['nsga2'].isChecked()
        self._range_btn.setEnabled(not manual)
        self._bayes_btn.setEnabled(bayesian)
        self._tlbo_btn.setEnabled(tlbo)
        self._obj_btn.setEnabled(nsga2)
        # Early Stop only applies to Bayesian — dim controls otherwise
        self._early_stop_cb.setEnabled(bayesian)
        self._patience_sp.setEnabled(
            bayesian and self._early_stop_cb.isChecked())

    def _algo_summary_text(self):
        sel = [n for n, ok in self._algo_selection.items() if ok]
        return (f'{len(sel)} selected:  ' + ',  '.join(sel)
                if sel else 'No algorithms selected.')

    def _open_algo_dialog(self):
        # Pass training data if available so TLBO preview can run
        train_data = None
        if self.df is not None:
            try:
                X, y = self._prepare_data()
                train_data = (X, y)
            except Exception:
                pass
        dlg = AlgorithmConfigDialog(
            self._algo_selection, self._algo_params,
            train_data=train_data,
            locked_params=self._algo_locked,
            custom_ranges=self._custom_ranges,
            parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._algo_selection = dlg.get_selection()
            self._algo_params    = dlg.get_params()
            self._algo_locked    = dlg.get_locked()
            # Fully replace per-algo range entries so ↺ resets are honoured.
            # Using update() would leave stale custom values in place.
            for algo, param_map in dlg.get_ranges().items():
                self._custom_ranges[algo] = param_map
            self._algo_summary.setText(self._algo_summary_text())

    def _open_range_dialog(self):
        selected = [n for n, ok in self._algo_selection.items()
                    if ok and n in PARAM_SPACES]
        if not selected:
            QMessageBox.information(
                self, 'No Optimisable Algorithms',
                'Select at least one tree-based algorithm with an '
                'optimisation parameter space.')
            return
        dlg = SearchRangeDialog(selected, self._custom_ranges, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._custom_ranges = dlg.get_ranges()

    def _open_bayes_dialog(self):
        dlg = BayesianSettingsDialog(self._bayes_settings, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._bayes_settings = dlg.get_settings()
            s = self._bayes_settings
            mv = 'multivariate' if s['multivariate'] else 'independent'
            self._bayes_btn.setToolTip(
                f'Bayesian TPE — startup={s["n_startup_trials"]}, {mv}\n'
                'Click to reconfigure.')
            self.opt_btns['bayesian'].setToolTip(
                f'Bayesian optimisation via Optuna TPE sampler.\n'
                f'Startup trials: {s["n_startup_trials"]}   '
                f'Multivariate: {"yes" if s["multivariate"] else "no"}')

    def _open_tlbo_dialog(self):
        budget = self.opt_trials.value()
        dlg = TlboSettingsDialog(self._tlbo_settings, budget_hint=budget, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._tlbo_settings = dlg.get_settings()
            s = self._tlbo_settings
            if s['mode'] == 'auto':
                summary = f'Auto (budget={budget})'
                tip_extra = (f'n_pop=max(5,budget÷5)={max(5,budget//5)}, '
                             f'n_iter=max(3,budget÷5)={max(3,budget//5)}')
            else:
                total = s['n_pop'] + s['n_iter'] * 2 * s['n_pop']
                summary = f'Population {s["n_pop"]}, Iterations {s["n_iter"]}, Total {total}'
                tip_extra = f'Total evals = {total}'
            self._tlbo_btn.setToolTip(
                f'TLBO — {summary}\nClick to reconfigure.')
            self.opt_btns['tlbo'].setToolTip(
                f'Teaching-Learning Based Optimisation.\n{tip_extra}')

    def _open_obj_dialog(self):
        dlg = Nsga2ObjectiveDialog(self._nsga2_objectives, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._nsga2_objectives = dlg.get_objectives()
            lbl1 = NSGA2_OBJECTIVES.get(self._nsga2_objectives[0],
                                        (self._nsga2_objectives[0],))[0]
            lbl2 = NSGA2_OBJECTIVES.get(self._nsga2_objectives[1],
                                        (self._nsga2_objectives[1],))[0]
            self._obj_btn.setToolTip(
                f'Obj 1: {lbl1}\nObj 2: {lbl2}\nClick to reconfigure.')
            self.opt_btns['nsga2'].setToolTip(
                f'Multi-objective genetic algorithm.\n'
                f'Objective 1: {lbl1}\n'
                f'Objective 2: {lbl2}')

    def _init_curve_plot(self):
        self._curve_data = {}
        self._bottom_stack.setCurrentIndex(0)   # show curve page
        self._bottom_panel.setTitle('Optimisation History  (CV R² per Evaluation)')
        with plt.rc_context({'font.family': 'serif', 'font.size': 9,
                             'axes.spines.top': False,
                             'axes.spines.right': False}):
            self._curve_fig.clear()
            ax = self._curve_fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                'Convergence curve will appear here during iterative search.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='#AAAAAA', style='italic',
                fontfamily='serif')
        try: self._curve_fig.tight_layout()
        except Exception: pass
        self._curve_canvas.draw()

    def _update_curve(self, name, eval_idx, best_cv):
        """Called from main thread via trial_score signal.

        Throttled: accumulates data every call but only schedules a real
        redraw at most once every 500 ms via QTimer.singleShot.  This
        prevents the Qt event queue from being flooded when TLBO/Bayesian
        emits hundreds of trial_score signals in rapid succession.
        """
        if name not in self._curve_data:
            self._curve_data[name] = []
        self._curve_data[name].append((eval_idx, best_cv))
        if not getattr(self, '_curve_timer_pending', False):
            self._curve_timer_pending = True
            QTimer.singleShot(500, self._do_deferred_redraw)

    def _do_deferred_redraw(self):
        """Deferred redraw slot — fires at most once per 500 ms."""
        self._curve_timer_pending = False
        self._redraw_curve()

    def _redraw_curve(self):
        with plt.rc_context({'font.family': 'serif', 'font.size': 9,
                             'axes.linewidth': 0.8,
                             'xtick.direction': 'in',
                             'ytick.direction': 'in',
                             'axes.spines.top': False,
                             'axes.spines.right': False,
                             'figure.dpi': 100}):
            self._curve_fig.clear()
            ax = self._curve_fig.add_subplot(111)

        for i, (name, pts) in enumerate(self._curve_data.items()):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            col = _CURVE_PALETTE[i % len(_CURVE_PALETTE)]
            ax.plot(xs, ys, lw=1.4, color=col, label=name)
            # Mark current best
            if pts:
                ax.scatter([xs[-1]], [ys[-1]], s=28, color=col, zorder=5)

        ax.set_xlabel('Evaluation index', fontsize=8)
        ax.set_ylabel('Best CV R²', fontsize=8)
        ax.set_title('Convergence Progress', fontsize=9, fontweight='bold')
        if self._curve_data:
            ax.legend(fontsize=7, frameon=False,
                      loc='lower right')

        try: self._curve_fig.tight_layout(pad=0.6)
        except Exception: pass
        self._curve_canvas.draw()

    def _prepare_data(self):
        df      = self.df.copy()
        n_feats = [c for c in NUM_FEAT_COLS if c in df.columns]
        X_num   = df[n_feats].values.astype(float)
        for j, col in enumerate(n_feats):
            n_nan = int(np.sum(~np.isfinite(X_num[:, j])))
            if n_nan > 0:
                med = float(np.nanmedian(X_num[:, j]))
                X_num[~np.isfinite(X_num[:, j]), j] = med
                self.log.append(
                    f'[INFO] {col}: {n_nan} missing → median ({med:.4g})')
        if 'FRP-type' in df.columns:
            ohe   = OneHotEncoder(**_ohe_sparse_kwarg(),
                                  handle_unknown='ignore',
                                  categories=[FRP_TYPES])
            X_cat = ohe.fit_transform(df[['FRP-type']].astype(str))
            X     = np.hstack([X_num, X_cat])
            flabs = n_feats + [f'FRP={t}' for t in FRP_TYPES]
            self.ohe = ohe
        else:
            X = X_num; flabs = n_feats; self.ohe = None
        self.feat_cols = flabs
        y    = pd.to_numeric(df['Vexp(kN)'], errors='coerce').values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & (y > 0)
        if mask.sum() < 10:
            raise ValueError(
                f'Only {mask.sum()} valid rows. Check column mapping.')
        return X[mask], y[mask]

    def _build_cfg(self, use_gpu=False):
        seed = self.cfg['seed'].value()
        cfg  = {}
        for name, checked in self._algo_selection.items():
            if not checked: continue
            entry = next((e for e in _ALGO_CATALOGUE if e[0] == name), None)
            if entry and not _is_available(entry[2]): continue
            do_gpu = use_gpu and (name in _GPU_CAPABLE)
            cfg[name] = {
                'factory':       _factory_for(name, seed, use_gpu=do_gpu),
                'fixed_params':  dict(self._algo_params.get(name, {})),
                'locked_params': set(self._algo_locked.get(name, set())),
                'use_gpu':       do_gpu,
            }
        return cfg

    def _get_opt(self):
        for k, rb in self.opt_btns.items():
            if rb.isChecked(): return k
        return 'none'

    def _start(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Dataset',
                'Load a dataset in the Data Management tab first.')
            return
        # Build cfg without GPU first — just to validate algo selection
        cfg = self._build_cfg(use_gpu=False)
        if not cfg:
            QMessageBox.warning(self, 'No Algorithm',
                'Select at least one algorithm.')
            return

        strategy_labels = {
            'none':     'Fixed Hyperparameters',
            'bayesian': 'Bayesian Optimisation (TPE)',
            'tlbo':     'TLBO (Teaching-Learning)',
            'nsga2':    'NSGA-II (Multi-Objective)',
        }
        cur_strategy = self._get_opt()
        summary = {
            'algorithms': ', '.join(cfg.keys()),
            'n_models':   len(cfg),
            'strategy':   strategy_labels.get(cur_strategy, cur_strategy),
            'budget':     self.opt_trials.value(),
            'cv_folds':   self.cfg['cv_folds'].value(),
            'test_pct':   self.cfg['test_pct'].value(),
        }
        pre_dlg = PreTrainingDialog(summary, last_use_gpu=self._use_gpu, parent=self)
        if pre_dlg.exec_() != QDialog.Accepted:
            return
        self._use_gpu = pre_dlg.use_gpu()
        # Rebuild cfg now that we know the actual device choice — factories
        # for XGBoost / LightGBM / CatBoost get the correct GPU kwargs.
        cfg = self._build_cfg(use_gpu=self._use_gpu)

        self.log.clear()
        self.log.append('[INFO] Preparing feature matrix …')
        try:
            X, y = self._prepare_data()
        except Exception as e:
            QMessageBox.critical(self, 'Data Error', str(e)); return

        ts   = self.cfg['test_pct'].value() / 100
        seed = self.cfg['seed'].value()
        cv   = self.cfg['cv_folds'].value()
        opt  = self._get_opt()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=ts, random_state=seed)
        self.scaler.fit(X_tr)
        X_tr_s  = self.scaler.transform(X_tr)
        X_te_s  = self.scaler.transform(X_te)
        X_all_s = self.scaler.transform(X)
        self._X_all = X_all_s; self._y_all = y

        device_str = f'GPU ({CUDA_DEVICE_NAME})' if self._use_gpu else 'CPU'
        self.log.append(
            f'\n[INFO] Dataset: total={len(X)}  '
            f'train={len(X_tr)}  test={len(X_te)}\n'
            f'       Features ({len(self.feat_cols)}): {self.feat_cols}\n'
            f'       Strategy: {opt}  Budget: {self.opt_trials.value()}  '
            f'CV: {cv}  Device: {device_str}\n{"="*60}')

        # Reset live curve
        self._init_curve_plot()

        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._cfg_algo_btn.setEnabled(False)   # prevent opening during training
        self._eta_lbl.setText(
            f'Training {len(cfg)} model(s)  ·  strategy: {opt}  ·  {cv}-fold CV')

        # Disconnect previous thread's signals before creating a new one.
        # Without this, every re-run accumulates an extra connection so
        # _on_done (and sig_done) fire N times after the N-th training run.
        if hasattr(self, '_thread') and self._thread is not None:
            try:
                self._thread.done.disconnect()
                self._thread.log.disconnect()
                self._thread.progress.disconnect()
                self._thread.trial_score.disconnect()
            except Exception:
                pass
            self._thread = None

        self._thread = TrainingThread(
            X_tr_s, X_te_s, y_tr, y_te, cfg,
            cv, X_all_s, y,
            opt_strategy=opt,
            opt_trials=self.opt_trials.value(),
            seed=seed,
            custom_ranges=self._custom_ranges,
            nsga2_objectives=self._nsga2_objectives,
            tlbo_settings=self._tlbo_settings,
            bayes_settings=self._bayes_settings,
            use_gpu=self._use_gpu,
            early_stop=self._early_stop_cb.isChecked(),
            patience=self._patience_sp.value())
        self._thread.progress.connect(self.progress.setValue)
        # Cap log at 5000 lines to prevent O(n) append cost over long runs.
        self.log.document().setMaximumBlockCount(5000)
        self._thread.log.connect(self._append_log)
        self._thread.trial_score.connect(self._update_curve)
        self._thread.done.connect(self._on_done)

        # Wrap sig_done emission in a try/except so that any exception inside
        # _on_train_done_inner (MainWindow side) cannot escape the lambda into
        # PyQt5's C++ slot dispatcher.  An unguarded exception there would call
        # sys.excepthook which — if itself nested inside exec_() — may terminate
        # the process.
        def _forward_results(r):
            try:
                self.sig_done.emit(r, X_te_s, y_te, X_tr_s, y_tr, X_all_s, y)
            except Exception:
                import traceback
                self._append_log(
                    f'\n[ERROR] Post-training update failed:\n'
                    f'{traceback.format_exc()}')

        self._thread.done.connect(_forward_results)
        self._thread.start()

    def _append_log(self, text):
        """Append a log line and auto-scroll.  Called via signal from TrainingThread.
        Using a dedicated slot (vs two separate connects) halves the signal
        dispatch overhead and keeps scroll behaviour in one place.
        """
        self.log.append(text)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _stop(self):
        if hasattr(self, '_thread') and self._thread.isRunning():
            self._thread.stop(); self.stop_btn.setEnabled(False)

    def _on_done(self, results):
        self._results = results
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(bool(results))
        self._cfg_algo_btn.setEnabled(True)    # re-enable after training
        self._eta_lbl.setText('')
        try:
            self._on_done_inner(results)
        except Exception:
            msg = traceback.format_exc()
            self.log.append(f'\n[ERROR] Post-training display error:\n{msg}')
            QMessageBox.warning(
                self, 'Display Error',
                'Training completed successfully, but an error occurred '
                'while updating the results panel.\n\n'
                'Your trained models are intact — use "Save Bundle" to '
                'save them.\n\n'
                f'Details:\n{msg}')

    def _on_done_inner(self, results):
        # Force final curve redraw — throttle may have skipped last points
        self._curve_timer_pending = False
        if self._curve_data and self._get_opt() != 'none':
            self._redraw_curve()
        self.log.append('\n' + '='*60)
        self.log.append('[INFO] Training complete — test-set ranking:')
        for name, res in sorted(
                results.items(),
                key=lambda x: x[1]['te_metrics'].get('R2', -1),
                reverse=True):
            tm = res['te_metrics']
            self.log.append(
                f'    {name:<18}  '
                f'R² = {tm["R2"]:.4f}  '
                f'r = {tm["r"]:.4f}  '
                f'RMSE = {tm["RMSE"]:.2f} kN  '
                f'CV = {res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}')

        if self._get_opt() != 'none' and results:
            newly_locked = 0
            for name, res in results.items():
                bp = res.get('best_params', {})
                if not bp:
                    continue
                if name not in self._algo_params:
                    self._algo_params[name] = {}
                self._algo_params[name].update(bp)
                if name not in self._algo_locked:
                    self._algo_locked[name] = set()
                before = len(self._algo_locked[name])
                self._algo_locked[name].update(bp.keys())
                newly_locked += len(self._algo_locked[name]) - before
            self.log.append(
                f'[INFO] Optimal hyperparameters written back and locked '
                f'({newly_locked} parameter(s) across {len(results)} model(s)).\n'
                f'       Open Algorithm Configuration to inspect the locked values (🔒).')

        # If manual mode, show a flat "no curve" placeholder in the panel
        if self._get_opt() == 'none':
            self._show_curve_placeholder_manual(results)

        # Auto-popup summary dialog.
        # IMPORTANT: use show() instead of exec_() to avoid creating a nested
        # Qt event loop inside the done-signal slot.  exec_() would cause the
        # second done-connected slot (the sig_done lambda) to be dispatched
        # *inside* the nested loop; any exception escaping that lambda bypasses
        # all Python try/except blocks and kills the process.  show() keeps the
        # dialog modeless and returns immediately, so the entire signal chain
        # completes before Qt processes any more events.
        if results:
            self._summary_dlg = _TrainSummaryDialog(
                results,
                opt_strategy=self._get_opt(),
                feat_cols=self.feat_cols or [],
                save_fn=self._save_selected_models,
                parent=self)
            self._summary_dlg.show()

    def _show_curve_placeholder_manual(self, results):
        """For Fixed Parameters mode: populate the summary table and switch to it."""
        self._bottom_panel.setTitle('Performance Summary')
        sorted_names = sorted(
            results,
            key=lambda n: results[n]['te_metrics'].get('R2', -99),
            reverse=True)

        hdrs = ['Rank', 'Model', 'R² (test)', 'r (test)',
                'RMSE (kN)', 'MAE (kN)', 'CV R²  (mean ± S.D.)']
        tbl = self._summary_tbl
        tbl.setColumnCount(len(hdrs))
        tbl.setRowCount(len(sorted_names))
        tbl.setHorizontalHeaderLabels(hdrs)
        hdr = tbl.horizontalHeader()
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in [0] + list(range(2, len(hdrs))):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)

        bold     = QFont(); bold.setBold(True)
        green_bg = QColor('#D5ECD4')

        def _fv(m, k, fmt='.4f'):
            v = m.get(k)
            return f'{v:{fmt}}' if isinstance(v, float) and not np.isnan(v) else '—'

        for row, name in enumerate(sorted_names):
            res = results[name]
            te  = res['te_metrics']
            cv_str = (f'{res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}'
                      if not np.isnan(res.get('cv_mean', float('nan'))) else '—')
            vals = [str(row + 1), name,
                    _fv(te, 'R2'), _fv(te, 'r'),
                    _fv(te, 'RMSE', '.2f'), _fv(te, 'MAE', '.2f'),
                    cv_str]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(
                    Qt.AlignLeft | Qt.AlignVCenter if col == 1
                    else Qt.AlignCenter)
                if row == 0:
                    it.setBackground(green_bg)
                    it.setFont(bold)
                tbl.setItem(row, col, it)

        self._bottom_stack.setCurrentIndex(1)   # switch to table page

    def _models_dir(self):
        """Return project-root/models/, creating it if needed."""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        d = os.path.join(base, 'models')
        os.makedirs(d, exist_ok=True)
        return d

    def _make_bundle_name(self, model_names):
        """
        Generate a descriptive filename from trained model names.
        1 model  → 'GBDT.frpmdl'
        2-3      → 'GBDT_XGBoost_LightGBM.frpmdl'
        4+       → 'Bundle_9models_YYYYMMDD_HHMM.frpmdl'
        Spaces in model names are replaced with underscores.
        """
        from datetime import datetime
        # Sanitize: replace spaces with underscores for valid filenames
        names = [n.replace(' ', '_') for n in model_names]
        if len(names) == 0:
            stem = 'frp_model'
        elif len(names) <= 3:
            stem = '_'.join(names)
        else:
            stamp = datetime.now().strftime('%Y%m%d_%H%M')
            stem = f'Bundle_{len(names)}models_{stamp}'
        return stem + '.frpmdl'

    def _save(self):
        if not self._results:
            QMessageBox.warning(self, 'No Models', 'No trained models.'); return
        default_name = self._make_bundle_name(self._results.keys())
        default_path = os.path.join(self._models_dir(), default_name)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Bundle', default_path,
            'FRP Model Bundle (*.frpmdl)')
        if not path: return
        try:
            ModelIO.save(path, self._results, self.scaler,
                         self.feat_cols, self.ohe,
                         X_all=self._X_all,
                         X_shape=self._X_all.shape if self._X_all is not None else None)
            QMessageBox.information(self, 'Saved', f'Saved to:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save Failed', str(e))

    def _save_selected_models(self, selected_names):
        """Save only the models whose names are in selected_names."""
        if not selected_names:
            return
        default_name = self._make_bundle_name(selected_names)
        default_path = os.path.join(self._models_dir(), default_name)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Bundle', default_path,
            'FRP Model Bundle (*.frpmdl)')
        if not path:
            return
        subset = {n: self._results[n] for n in selected_names
                  if n in self._results}
        try:
            ModelIO.save(path, subset, self.scaler,
                         self.feat_cols, self.ohe,
                         X_all=self._X_all,
                         X_shape=self._X_all.shape if self._X_all is not None else None)
            QMessageBox.information(
                self, 'Saved',
                f'Saved {len(subset)} model(s) to:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save Failed', str(e))

    def _load_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Bundle', '', 'FRP Model Bundle (*.frpmdl)')
        if path: self._load_bundle(path)

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, 'Select Folder Containing .frpmdl Files')
        if not folder: return
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith('.frpmdl')]
        if not files:
            QMessageBox.information(
                self, 'No Bundles', f'No .frpmdl files in:\n{folder}'); return
        dlg = BundleFolderDialog(folder, parent=self)
        if dlg.exec_() != QDialog.Accepted: return
        paths = dlg.selected_paths()
        if not paths: return
        self.log.clear()
        merged = {}
        last_scaler = last_feat_cols = last_ohe = last_shap = None
        last_meta = {}
        y_tr_all = y_te_all = np.array([])
        for p in paths:
            try:
                r = self._load_raw(p)
                if r is None: continue
                results, scaler, feat_cols, ohe, shap, meta, y_tr, y_te = r
                merged.update(results)
                last_scaler, last_feat_cols = scaler, feat_cols
                last_ohe, last_shap, last_meta = ohe, shap, meta
                if len(y_te) > 0: y_te_all = y_te
                if len(y_tr) > 0: y_tr_all = y_tr
                self.log.append(
                    f'[OK] {os.path.basename(p)} → {list(results.keys())}')
            except Exception as e:
                self.log.append(f'[FAIL] {os.path.basename(p)}: {e}')
        if not merged:
            QMessageBox.warning(self, 'Load Failed', 'No models loaded.'); return
        self._finalise_load(merged, last_scaler, last_feat_cols,
                            last_ohe, last_shap, last_meta,
                            y_tr_all, y_te_all)

    def _load_raw(self, path):
        r = ModelIO.load(path)
        if len(r) == 8: return r
        results, scaler, feat_cols, ohe, shap, meta = r
        return results, scaler, feat_cols, ohe, shap, meta, \
               np.array([]), np.array([])

    def _load_bundle(self, path):
        try:
            r = self._load_raw(path)
            results, scaler, feat_cols, ohe, shap, meta, y_tr, y_te = r
            self.log.clear()
            self._finalise_load(results, scaler, feat_cols,
                                ohe, shap, meta, y_tr, y_te)
        except Exception as e:
            QMessageBox.critical(self, 'Load Failed', str(e))

    def _finalise_load(self, results, scaler, feat_cols,
                       ohe, shap_cache, meta, y_tr, y_te):
        self._results  = results
        self.scaler    = scaler
        self.feat_cols = feat_cols
        self.ohe       = ohe
        self._X_all    = shap_cache
        self.save_btn.setEnabled(True)
        self.log.append(
            f'[INFO] Loaded {len(results)} model(s): {list(results.keys())}')
        self.log.append(f'       Features: {feat_cols}')
        self.log.append(
            f'       SHAP cache: '
            f'{len(shap_cache) if shap_cache is not None else "n/a"} rows')
        for name, res in results.items():
            te = res['te_metrics']
            self.log.append(
                f'    {name:<18}  '
                f'R² = {te.get("R2","—")}  '
                f'RMSE = {te.get("RMSE","—")} kN')
        dummy = np.array([])
        self.sig_done.emit(
            results, dummy, y_te, dummy, y_tr,
            shap_cache if shap_cache is not None else dummy, dummy)
