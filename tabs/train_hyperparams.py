"""
train_hyperparams.py — Algorithm hyperparameter configuration dialogs.

Classes
-------
TLBOMultiOptimizeDialog  Batch TLBO optimisation across multiple algorithms
AlgorithmConfigDialog    Per-algorithm hyperparameter editor with lock/unlock and TLBO preview
SearchRangeDialog        Custom [min, max] search-bound editor for each hyperparameter
"""
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QCheckBox, QSpinBox, QDoubleSpinBox,
    QDialog, QScrollArea, QFrame, QGroupBox,
    QPushButton, QSizePolicy, QAbstractItemView,
    QPlainTextEdit, QProgressBar, QMessageBox,
    QRadioButton, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    HAS_XGB, HAS_LGB, HAS_CAT, HAS_OPTUNA,
    HAS_CUDA, CUDA_DEVICE_NAME, _CAT_IMPORT_ERROR,
)
from widgets import flat_btn
from optimization import PARAM_SPACES, _GPU_CAPABLE, _factory_for
from .train_constants import _ALGO_CATALOGUE, _PARAM_LABELS, _is_available
from .train_threads import _TLBOPreviewThread, _TLBOMultiThread

# PyQt5 — complete import set inherited from the original train_tab
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

# config — complete set
from config import (
    APP_VERSION, _SHAP_BUNDLE_SAMPLES,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    C_WIN_BG, C_PANEL_BG, C_ALT_ROW, C_HEADER_BG,
    C_SUCCESS, C_SUCCESS_BG, C_DANGER,
    HAS_XGB, HAS_LGB, HAS_CAT, HAS_OPTUNA, HAS_PYMOO,
    HAS_CUDA, HAS_SHAP, CUDA_DEVICE_NAME,
    NUM_FEAT_COLS, FRP_TYPES, FEAT_LABELS,
    ALGO_COLORS, CODE_COLORS,
)
from optimization import (
    _ps_gbdt, _ps_xgb, _ps_lgb, _ps_cat, _ps_rf,
    _factory_for, tlbo_optimize, _optuna_optimize, nsga2_optimize,
    PARAM_SPACES, NSGA2_OBJECTIVES, _GPU_CAPABLE,
)

class TLBOMultiOptimizeDialog(QDialog):
    """
    Batch TLBO hyperparameter optimisation dialog.

    The user selects which algorithms to optimise, sets pop / iter / CV,
    optionally enables GPU, then clicks ‘Run Optimisation’.  TLBO runs
    sequentially for each selected algorithm; when an algorithm finishes
    its best parameters are written back into the parent
    AlgorithmConfigDialog’s spinboxes and every updated parameter is
    locked (🔒).
    """

    def __init__(self, config_dlg, avail_algos, train_data,
                 init_pop=20, init_iter=50, init_cv=5,
                 preview_ranges=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('TLBO Hyperparameter Search')
        self.resize(960, 600)
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint)

        self._config_dlg     = config_dlg
        self._avail_algos    = avail_algos
        self._train_data     = train_data
        self._preview_ranges = dict(preview_ranges or {})
        self._thread         = None
        self._n_done         = 0
        self._n_total        = 0
        self._model_cbs      = {}   # {name: QCheckBox}

        self._build_ui(init_pop, init_iter, init_cv)

    # UI
    def _build_ui(self, init_pop, init_iter, init_cv):
        root = QHBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(14, 14, 14, 14)

        # LEFT panel (fixed width)
        left = QWidget()
        left.setFixedWidth(270)
        ll = QVBoxLayout(left)
        ll.setSpacing(8)
        ll.setContentsMargins(0, 0, 0, 0)

        _grp_style = (
            'QGroupBox{font-size:11px;font-weight:bold;' +
            f'border:1px solid {C_BORDER};border-radius:4px;' +
            'margin-top:9px;padding:6px 6px 4px 6px;}' +
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}')

        # Models — simple checkboxes, no GPU/CPU badges (search always CPU)
        model_grp = QGroupBox('Algorithm Selection')
        model_grp.setStyleSheet(_grp_style)
        mg = QVBoxLayout(model_grp)
        mg.setSpacing(2)

        for name in self._avail_algos:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setStyleSheet('font-size:11px;')
            self._model_cbs[name] = cb
            mg.addWidget(cb)

        # Select-All / None shortcuts
        sel_row = QHBoxLayout()
        sel_row.setSpacing(3)
        sel_row.setContentsMargins(0, 0, 0, 0)
        for txt, val in (('All', True), ('None', False)):
            b = flat_btn(txt)
            b.setFixedHeight(24)
            _v = val
            b.clicked.connect(
                lambda _, v=_v: [c.setChecked(v)
                                  for c in self._model_cbs.values()])
            sel_row.addWidget(b, 1)
        sel_row.addStretch()
        mg.addLayout(sel_row)
        ll.addWidget(model_grp)

        # Settings
        sgrp = QGroupBox('Search Settings')
        sgrp.setStyleSheet(_grp_style)
        sg = QVBoxLayout(sgrp)
        sg.setSpacing(5)
        self._pop_sp  = self._isp(sg, 'Population:',  init_pop,  2,  50)
        self._iter_sp = self._isp(sg, 'Iterations:',  init_iter, 2, 200)
        self._cv_sp   = self._isp(sg, 'CV Folds:',    init_cv,   2,  15)
        ll.addWidget(sgrp)

        bounds_btn = flat_btn('Search Bounds')
        bounds_btn.setFixedHeight(28)
        bounds_btn.clicked.connect(self._open_bounds)
        ll.addWidget(bounds_btn)

        ll.addStretch()

        self._run_btn = flat_btn('Execute Optimisation', accent=True)
        self._run_btn.setFixedHeight(36)
        self._run_btn.clicked.connect(self._run)
        ll.addWidget(self._run_btn)

        self._abort_btn = flat_btn('Stop', danger=True)
        self._abort_btn.setFixedHeight(28)
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self._do_abort)
        ll.addWidget(self._abort_btn)

        close_btn = flat_btn('Close')
        close_btn.setFixedHeight(28)
        close_btn.clicked.connect(self.close)
        ll.addWidget(close_btn)

        root.addWidget(left)

        # RIGHT panel (stretch)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setSpacing(8)
        rl.setContentsMargins(0, 0, 0, 0)

        pgrp = QGroupBox('Progress')
        pgrp.setStyleSheet(_grp_style)
        pg = QVBoxLayout(pgrp)
        pg.setSpacing(6)

        for bar_attr, lbl_attr, row_lbl in [
            ('_overall_bar', '_overall_lbl', 'Overall:'),
            ('_cur_bar',     '_cur_lbl',     'Current:'),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(row_lbl))
            bar = QProgressBar()
            bar.setRange(0, 100); bar.setValue(0)
            bar.setFixedHeight(14); bar.setTextVisible(False)
            setattr(self, bar_attr, bar)
            row.addWidget(bar)
            lbl = QLabel('—')
            lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
            lbl.setFixedWidth(130)
            setattr(self, lbl_attr, lbl)
            row.addWidget(lbl)
            pg.addLayout(row)

        self._status_lbl = QLabel(
            'Select algorithms and click Execute to begin.')
        self._status_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        self._status_lbl.setWordWrap(True)
        pg.addWidget(self._status_lbl)
        rl.addWidget(pgrp)

        lgrp = QGroupBox('Log')
        lgrp.setStyleSheet(_grp_style)
        lg = QVBoxLayout(lgrp)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            'font-family:Consolas,Courier New,monospace;font-size:10px;' +
            f'background:#F9F9F9;border:none;')
        lg.addWidget(self._log)
        rl.addWidget(lgrp, stretch=1)
        root.addWidget(right, stretch=1)

    @staticmethod
    def _isp(layout, label, default, lo, hi):
        """Add a labelled QSpinBox row to *layout* and return the spinbox."""
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setStyleSheet('font-size:11px;')
        row.addWidget(lbl)
        sp = QSpinBox()
        sp.setRange(lo, hi); sp.setValue(default)
        sp.setFixedWidth(65)
        sp.setStyleSheet('font-size:11px;')
        row.addWidget(sp)
        row.addStretch()
        layout.addLayout(row)
        return sp

    # Helpers
    def _refresh_device_warning(self, *_):
        """No-op: GPU selection is handled in the main training dialog."""
        pass

    def _selected_algos(self):
        return [n for n, cb in self._model_cbs.items() if cb.isChecked()]

    def _open_bounds(self):
        sel = self._selected_algos() or list(self._model_cbs)
        dlg = SearchRangeDialog(sel, self._preview_ranges, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._preview_ranges.update(dlg.get_ranges())

    def _run(self):
        selected = self._selected_algos()
        if not selected:
            QMessageBox.warning(self, 'No Models',
                                'Select at least one model to optimise.')
            return

        use_gpu = False   # Hyperparameter search always runs on CPU.
        # Rationale: TLBO does hundreds of small cross-validation fits.
        # GPU overhead dominates for small CV batches and gives unreliable
        # R² values.  GPU is reserved for the final model training step.
        n_pop   = self._pop_sp.value()
        n_iter  = self._iter_sp.value()
        cv      = self._cv_sp.value()
        seed    = 42
        X, y    = self._train_data

        tasks = []
        for name in selected:
            if name not in PARAM_SPACES:
                continue
            full_space = PARAM_SPACES[name]()
            custom     = self._preview_ranges.get(name, {})
            locked     = self._config_dlg.get_locked().get(name, set())
            space = []
            for p, lo, hi, is_int in full_space:
                if p in locked:
                    continue
                if p in custom:
                    lo, hi = custom[p]
                space.append((p, lo, hi, is_int))

            if not space:
                self._log.appendPlainText(
                    f'[SKIP] {name}: all parameters locked.')
                continue

            fixed = {
                p: (int(round(w.value())) if isinstance(w, QSpinBox)
                    else float(w.value()))
                for p, w in self._config_dlg._widgets.get(name, {}).items()
                if p in locked
            }
            do_gpu  = use_gpu and (name in _GPU_CAPABLE)
            base_fn = _factory_for(name, seed, use_gpu=do_gpu)
            if fixed:
                _fx = dict(fixed)
                def _make_factory(_b=base_fn, _f=_fx):
                    def factory(**kw):
                        return _b(**{**_f, **kw})
                    return factory
                factory = _make_factory()
            else:
                factory = base_fn

            tasks.append({
                'name': name, 'factory': factory, 'space': space,
                'X': X, 'y': y,
                'n_pop': n_pop, 'n_iter': n_iter,
                'cv': cv, 'seed': seed,
                'use_gpu': use_gpu and (name in _GPU_CAPABLE),
            })

        if not tasks:
            QMessageBox.information(
                self, 'Nothing to Optimise',
                'All parameters of the selected models are locked — '
                'unlock at least one parameter per model to proceed.')
            return

        self._n_done  = 0
        self._n_total = len(tasks)
        self._overall_bar.setValue(0)
        self._overall_lbl.setText(f'0 / {self._n_total} models')
        self._cur_bar.setValue(0)
        self._cur_lbl.setText('—')
        self._log.clear()
        self._run_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)
        gpu_note = ' [GPU]' if use_gpu else ''
        self._status_lbl.setText(
            f'Starting{gpu_note} — {len(tasks)} model(s) queued…')

        self._thread = _TLBOMultiThread(tasks)
        self._thread.algo_started.connect(self._on_algo_started)
        self._thread.algo_done.connect(self._on_algo_done)
        self._thread.trial_update.connect(self._on_trial_update)
        self._thread.log_line.connect(self._log.appendPlainText)
        self._thread.all_done.connect(self._on_all_done)
        self._thread.start()

    def _do_abort(self):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
        self._abort_btn.setEnabled(False)
        self._status_lbl.setText('Aborting …')

    # Slot handlers
    def _on_algo_started(self, name):
        self._cur_bar.setValue(0)
        self._cur_lbl.setText(name)
        self._status_lbl.setText(f'Optimising: {name} …')

    def _on_trial_update(self, name, i, total, best):
        pct = int(100 * i / max(total, 1))
        self._cur_bar.setValue(pct)
        self._cur_lbl.setText(f'Trial {i}/{total}')
        self._status_lbl.setText(
            f'{name}  —  Trial {i} of {total}  |  Best CV R² = {best:.4f}')

    def _on_algo_done(self, name, best_params, best_score):
        self._config_dlg._apply_tlbo_result(name, best_params)
        self._n_done += 1
        pct = int(100 * self._n_done / max(self._n_total, 1))
        self._overall_bar.setValue(pct)
        self._overall_lbl.setText(
            f'{self._n_done} / {self._n_total} models')
        self._cur_bar.setValue(100)

    def _on_all_done(self):
        self._run_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
        self._overall_bar.setValue(100)
        self._cur_bar.setValue(100)
        n = self._n_done
        self._status_lbl.setText(
            f'Complete — {n} model(s) optimised.  '
            'Parameters applied and locked in the configuration.')
        self._log.appendPlainText(
            f'\n[INFO] Done — {n} model(s) updated.')

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(3000)
        event.accept()

#  AlgorithmConfigDialog
class AlgorithmConfigDialog(QDialog):
    def __init__(self, current_selection, current_params,
                 train_data=None, locked_params=None,
                 custom_ranges=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Algorithm Configuration')
        self.resize(980, 620)   # slightly wider for range inputs
        self._checks         = {}
        self._widgets        = {}
        self._range_widgets  = {}   # {name: {param: (lo_spin, hi_spin)}}
        self._param_checks   = {}
        self._preview_ranges = dict(custom_ranges or {})  # seed from external ranges
        self._train_data     = train_data
        self._current_algo   = None

        self._params = {n: dict(dp) for n, _, _, dp in _ALGO_CATALOGUE}
        for name, params in current_params.items():
            if name in self._params:
                self._params[name].update(params)
        self._build_ui()
        for name, checked in current_selection.items():
            if name in self._checks:
                self._checks[name].setChecked(checked)
        # Apply externally locked params (e.g. written back after training)
        for name, locked_set in (locked_params or {}).items():
            for param in locked_set:
                ck = self._param_checks.get(name, {}).get(param)
                if ck is not None:
                    ck.setChecked(True)   # 🔒
        first = next((n for n, _, r, _ in _ALGO_CATALOGUE if _is_available(r)), None)
        if first:
            self._show_params(first)

    # UI construction
    def _build_ui(self):
        vl   = QVBoxLayout(self)
        body = QHBoxLayout()
        vl.addLayout(body)

        # Left: algorithm list
        left_grp = QGroupBox('Algorithms')
        left_grp.setFixedWidth(265)
        left_grp.setStyleSheet('QGroupBox{font-weight:bold;font-size:12px;}')
        ll = QVBoxLayout(left_grp)
        current_group = None
        for name, group, requires, _ in _ALGO_CATALOGUE:
            avail = _is_available(requires)
            if group != current_group:
                current_group = group
                sep = QLabel(f'── {group} ──')
                sep.setStyleSheet(
                    f'color:{C_TEXT2};font-size:10px;margin-top:4px;')
                ll.addWidget(sep)
            cb = QCheckBox(
                name if avail else (
                    f'{name}  [import error: {_CAT_IMPORT_ERROR}]'
                    if requires == 'catboost' and _CAT_IMPORT_ERROR
                    else f'{name}  [pip install {requires}]'
                ))
            cb.setChecked(avail)
            cb.setEnabled(avail)
            cb.setStyleSheet('font-size:12px;' if avail
                             else f'color:{C_TEXT2};font-size:11px;')
            if avail:
                cb.clicked.connect(lambda _, n=name: self._show_params(n))
            self._checks[name] = cb
            ll.addWidget(cb)
        ll.addStretch()
        btn_row = QHBoxLayout()
        for lbl, fn in [('All', self._select_all), ('None', self._select_none)]:
            b = flat_btn(lbl); b.setFixedHeight(26); b.clicked.connect(fn)
            btn_row.addWidget(b)
        ll.addLayout(btn_row)
        body.addWidget(left_grp)

        # Right: hyperparameters + TLBO
        right_grp = QGroupBox('Hyperparameters')
        right_grp.setStyleSheet('QGroupBox{font-weight:bold;font-size:12px;}')
        rl = QVBoxLayout(right_grp)

        # Title
        self._algo_title = QLabel('Select an algorithm')
        self._algo_title.setStyleSheet(
            f'font-size:13px;font-weight:bold;color:{C_ACCENT};')
        rl.addWidget(self._algo_title)

        # Hint
        hint = QLabel(
            '  🔓 free to optimise  │  🔒 fixed  │  '
            'Current value  │  Search range [min – max]  │  ↺ reset range')
        hint.setStyleSheet(f'font-size:10px;color:{C_TEXT2};margin-bottom:4px;')
        rl.addWidget(hint)

        # Parameter scroll area
        self._param_area      = QScrollArea()
        self._param_area.setWidgetResizable(True)
        self._param_area.setFrameShape(QFrame.NoFrame)
        self._param_container = QWidget()
        self._param_layout    = QGridLayout(self._param_container)
        self._param_layout.setSpacing(6)
        self._param_layout.setColumnMinimumWidth(0, 20)   # checkbox col
        self._param_layout.setColumnStretch(1, 1)          # label col
        self._param_area.setWidget(self._param_container)
        rl.addWidget(self._param_area, stretch=1)

        # Pre-build all spinboxes + per-param checkboxes + range spinboxes
        for name, _, _, defaults in _ALGO_CATALOGUE:
            self._widgets[name]       = {}
            self._param_checks[name]  = {}
            self._range_widgets[name] = {}
            for param, val in defaults.items():
                meta = _PARAM_LABELS.get(param)
                if meta is None:
                    continue
                _, mn, mx, is_float = meta

                # Value spinbox
                w = QDoubleSpinBox() if is_float else QSpinBox()
                if is_float:
                    w.setDecimals(5)
                    w.setSingleStep((mx - mn) / 100)
                w.setRange(mn, mx)
                w.setValue(self._params[name].get(param, val))
                self._widgets[name][param] = w

                # Lock toggle button 🔓/🔒
                ck = QPushButton('\U0001f513')
                ck.setCheckable(True)
                ck.setChecked(False)
                ck.setFixedSize(32, 28)
                _lock_font = 'Segoe UI Emoji, Apple Color Emoji, Noto Emoji, sans-serif'
                ck.setStyleSheet(
                    f'QPushButton{{font-size:15px;font-family:{_lock_font};'
                    f'border:1px solid #aaa;border-radius:4px;background:#f5f5f5;}}'
                    f'QPushButton:checked{{background:#ffe0e0;border-color:#c00;}}'
                )
                ck.setToolTip(
                    'Unlocked: parameter will be tuned by TLBO / Bayesian / NSGA-II\n'
                    'Locked: parameter is fixed at its current value')
                ck.toggled.connect(
                    lambda locked, b=ck: b.setText(
                        '\U0001f512' if locked else '\U0001f513'))
                self._param_checks[name][param] = ck

                # Range spinboxes (Min / Max)
                c_range = self._preview_ranges.get(name, {}).get(param)
                r_lo = c_range[0] if c_range else mn
                r_hi = c_range[1] if c_range else mx

                def _make_range_spin(val_r, is_f):
                    if is_f:
                        rs = QDoubleSpinBox()
                        rs.setDecimals(6)
                        rs.setRange(-1e9, 1e9)
                        rs.setSingleStep((mx - mn) / 50)
                    else:
                        rs = QSpinBox()
                        rs.setRange(-999999, 999999)
                    rs.setValue(val_r)
                    rs.setFixedWidth(80)
                    rs.setStyleSheet('font-size:10px;')
                    return rs

                lo_sp = _make_range_spin(r_lo, is_float)
                hi_sp = _make_range_spin(r_hi, is_float)
                lo_sp.setToolTip(f'Minimum search bound for {meta[0]}')
                hi_sp.setToolTip(f'Maximum search bound for {meta[0]}')
                self._range_widgets[name][param] = (lo_sp, hi_sp)

        body.addWidget(right_grp, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        vl.addWidget(bb)

    # Param panel helpers
    def _clear_param_layout(self):
        while self._param_layout.count():
            item = self._param_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

    def _show_params(self, name):
        self._current_algo = name
        self._algo_title.setText(f'{name}  —  hyperparameters')
        self._clear_param_layout()
        params = self._widgets.get(name, {})

        for row_idx, (param, w) in enumerate(params.items()):
            r = row_idx
            meta = _PARAM_LABELS.get(param)
            _, mn, mx, is_float = meta if meta else (param, 0, 1, False)

            lbl = QLabel((meta[0] if meta else param) + ':')
            lbl.setStyleSheet('font-size:11px;')
            ck  = self._param_checks[name][param]

            # Value spinbox — slightly wider, visually distinct
            w.setFixedWidth(110)

            # Separator label between value and range
            sep_lbl = QLabel('│')
            sep_lbl.setAlignment(Qt.AlignCenter)
            sep_lbl.setStyleSheet(
                f'color:{C_BORDER};font-size:14px;'
                f'padding: 0 6px;')

            # Range spinboxes
            lo_sp, hi_sp = self._range_widgets[name][param]
            lo_sp.setFixedWidth(82)
            hi_sp.setFixedWidth(82)

            # Small greyed labels above the range spinboxes (first row only)
            # — use tooltip instead of labels to keep rows compact
            lo_sp.setToolTip(
                f'Search range minimum for {meta[0] if meta else param}\n'
                f'Default: {mn:.4g}')
            hi_sp.setToolTip(
                f'Search range maximum for {meta[0] if meta else param}\n'
                f'Default: {mx:.4g}')

            # Tilde separator
            tilde = QLabel('–')
            tilde.setAlignment(Qt.AlignCenter)
            tilde.setStyleSheet(
                f'font-size:11px;color:{C_TEXT2};padding:0 2px;')

            # Reset button
            rst = QPushButton('↺')
            rst.setFixedSize(22, 22)
            rst.setStyleSheet(
                f'QPushButton{{font-size:12px;border:1px solid #ccc;'
                f'border-radius:3px;background:#f9f9f9;color:{C_TEXT2};}}'
                f'QPushButton:hover{{background:#e8f0fe;color:{C_ACCENT};}}')
            rst.setToolTip('Reset to default range')
            rst.clicked.connect(
                lambda _, ls=lo_sp, hs=hi_sp, lo=mn, hi=mx:
                    (ls.setValue(lo), hs.setValue(hi)))

            # col: 0=lock  1=label  2=value  3=sep  4=lo  5=dash  6=hi  7=reset
            self._param_layout.addWidget(ck,      r, 0, Qt.AlignCenter)
            self._param_layout.addWidget(lbl,     r, 1)
            self._param_layout.addWidget(w,       r, 2)
            self._param_layout.addWidget(sep_lbl, r, 3, Qt.AlignCenter)
            self._param_layout.addWidget(lo_sp,   r, 4)
            self._param_layout.addWidget(tilde,   r, 5, Qt.AlignCenter)
            self._param_layout.addWidget(hi_sp,   r, 6)
            self._param_layout.addWidget(rst,     r, 7, Qt.AlignCenter)

        self._param_layout.setColumnStretch(1, 1)
        self._param_layout.setColumnMinimumWidth(2, 110)
        self._param_layout.setColumnMinimumWidth(3, 18)   # separator gap
        self._param_layout.setColumnMinimumWidth(4, 84)
        self._param_layout.setColumnMinimumWidth(6, 84)
        self._param_layout.setVerticalSpacing(7)

        if not params:
            self._param_layout.addWidget(
                QLabel('No configurable parameters.'), 0, 0, 1, 8)

    def _select_all(self):
        for n, cb in self._checks.items():
            if cb.isEnabled():
                cb.setChecked(True)

    def _select_none(self):
        for cb in self._checks.values():
            cb.setChecked(False)

    def get_selection(self):
        return {n: cb.isChecked() for n, cb in self._checks.items()}

    def get_params(self):
        return {n: {p: w.value() for p, w in pw.items()}
                for n, pw in self._widgets.items()}

    def get_locked(self):
        """Return {algo_name: set_of_locked_param_names}."""
        return {
            name: {p for p, btn in checks.items() if btn.isChecked()}
            for name, checks in self._param_checks.items()
        }

    def get_ranges(self):
        """
        Return {algo: {param: (lo, hi)}} for ALL params with range widgets.
        Returns every value (including defaults) so the caller can fully
        replace the custom_ranges dict — this ensures that ↺ resets are
        not lost when merging back into the parent's _custom_ranges.
        """
        result = {}
        for name, param_map in self._range_widgets.items():
            for param, (lo_sp, hi_sp) in param_map.items():
                meta = _PARAM_LABELS.get(param)
                if meta is None:
                    continue
                _, def_mn, def_mx, _ = meta
                lo, hi = lo_sp.value(), hi_sp.value()
                # Guard: enforce lo < hi.  If the user accidentally set
                # min > max, silently swap so the optimiser never receives
                # an invalid search interval (numpy raises on uniform(hi, lo)).
                if lo >= hi:
                    lo, hi = min(lo, hi), max(lo, hi)
                    if lo == hi:                 # degenerate — use defaults
                        lo, hi = def_mn, def_mx
                # Store ALL values; skip only if exactly equal to default
                # (keeps dict lean but correctly clears stale overrides)
                if lo != def_mn or hi != def_mx:
                    result.setdefault(name, {})[param] = (lo, hi)
                else:
                    # Explicitly mark as default — caller removes stale entry
                    result.setdefault(name, {})   # ensure algo key exists
        return result

    # TLBO preview
    def _open_preview_ranges(self):
        name = self._current_algo
        if name is None or name not in PARAM_SPACES:
            return
        dlg = SearchRangeDialog(
            [name], self._preview_ranges, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._preview_ranges.update(dlg.get_ranges())

    def _apply_tlbo_result(self, name: str, best_params: dict):
        """
        Write TLBO best values into the config spinboxes for *name*
        and lock every updated parameter (🔒).
        Called from TLBOMultiOptimizeDialog via signal on the GUI thread.
        """
        for param, val in best_params.items():
            w = self._widgets.get(name, {}).get(param)
            if w is not None:
                w.setValue(val)
            ck = self._param_checks.get(name, {}).get(param)
            if ck is not None:
                ck.setChecked(True)   # True = locked (🔒)

    def closeEvent(self, event):
        super().closeEvent(event)

#  SearchRangeDialog  — per-parameter [min, max] customisation
class SearchRangeDialog(QDialog):
    """
    Configure search bounds for every tuneable parameter of every
    selected algorithm.  The table has columns:

        Algorithm | Parameter | Min | Max

    The user edits Min/Max cells directly.  On OK, returns a dict
    {algo_name: [(param, lo, hi, is_int), ...]} that replaces
    the default PARAM_SPACES entry for the optimiser.
    """

    def __init__(self, selected_algos, current_ranges, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Hyperparameter Search Bounds')
        self.resize(660, 480)
        self._rows = []   # (algo, param, is_int, lo_w, hi_w)
        self._build_ui(selected_algos, current_ranges)

    def _build_ui(self, selected_algos, current_ranges):
        vl = QVBoxLayout(self)
        info = QLabel(
            'Edit the Min / Max bounds for each parameter.\n'
            'These ranges are used by TLBO, Bayesian, and NSGA-II search.')
        info.setStyleSheet(f'font-size:11px;color:{C_TEXT2};')
        vl.addWidget(info)

        self._tbl = QTableWidget()
        self._tbl.setColumnCount(4)
        self._tbl.setHorizontalHeaderLabels(
            ['Algorithm', 'Parameter', 'Min', 'Max'])
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.setStyleSheet('font-size:11px;')
        hdr = self._tbl.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        vl.addWidget(self._tbl)

        # Populate rows
        row_idx = 0
        for algo in selected_algos:
            if algo not in PARAM_SPACES:
                continue
            default_space = PARAM_SPACES[algo]()
            custom_space  = current_ranges.get(algo, {})
            self._tbl.setRowCount(row_idx + len(default_space))
            for param, d_lo, d_hi, is_int in default_space:
                c_lo = custom_space.get(param, (d_lo, d_hi))[0]
                c_hi = custom_space.get(param, (d_lo, d_hi))[1]
                meta = _PARAM_LABELS.get(param)
                label = meta[0] if meta else param

                algo_item = QTableWidgetItem(algo)
                algo_item.setFlags(algo_item.flags() & ~Qt.ItemIsEditable)
                param_item = QTableWidgetItem(label)
                param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)
                self._tbl.setItem(row_idx, 0, algo_item)
                self._tbl.setItem(row_idx, 1, param_item)

                def _make_spin(val, lo, hi, is_f):
                    if is_f:
                        w = QDoubleSpinBox()
                        w.setDecimals(6)
                        w.setRange(-1e9, 1e9)
                        w.setSingleStep((hi - lo) / 50)
                    else:
                        w = QSpinBox()
                        w.setRange(-999999, 999999)
                    w.setValue(val)
                    return w

                is_f = not is_int
                lo_w = _make_spin(c_lo, d_lo, d_hi, is_f)
                hi_w = _make_spin(c_hi, d_lo, d_hi, is_f)
                self._tbl.setCellWidget(row_idx, 2, lo_w)
                self._tbl.setCellWidget(row_idx, 3, hi_w)
                self._rows.append((algo, param, is_int, lo_w, hi_w))
                row_idx += 1

        reset_btn = flat_btn('Restore Defaults')
        reset_btn.setFixedHeight(28)
        reset_btn.clicked.connect(self._reset)
        vl.addWidget(reset_btn)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        vl.addWidget(bb)

    def _reset(self):
        idx = 0
        for algo, param, is_int, lo_w, hi_w in self._rows:
            space = PARAM_SPACES.get(algo, lambda: [])()
            for p, d_lo, d_hi, _ in space:
                if p == param:
                    lo_w.setValue(d_lo); hi_w.setValue(d_hi)
            idx += 1

    def get_ranges(self):
        """Returns {algo: {param: (lo, hi)}}"""
        result = {}
        for algo, param, is_int, lo_w, hi_w in self._rows:
            result.setdefault(algo, {})
            result[algo][param] = (lo_w.value(), hi_w.value())
        return result

#  BundleFolderDialog
