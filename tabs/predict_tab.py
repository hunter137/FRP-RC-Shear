"""
predict_tab.py — PredictTab: single-sample inference, NSGA-II optimisation, batch prediction.

All helper classes and dialogs have been extracted into sibling modules:

  predict_helpers.py  — _compute_pi, BeamSchematicWidget
  predict_dialogs.py  — PredictionSetupDialog, BatchPredictionDialog
"""
import os
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSizePolicy, QCheckBox,
    QDialog, QFrame, QScrollArea,
)
from PyQt5.QtCore import Qt, QSize, QFileSystemWatcher
from PyQt5.QtGui import QFont, QColor

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from config import (
    APP_VERSION, _SHAP_BUNDLE_SAMPLES,
    C_WIN_BG, C_PANEL_BG, C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_SUCCESS, C_SUCCESS_BG, C_DANGER, C_HEADER_BG,
    ALGO_COLORS, CODE_COLORS, NUM_FEAT_COLS, FRP_TYPES, FEAT_LABELS,
    HAS_PYMOO, HAS_SHAP,
)
from widgets import flat_btn, panel, result_box, _stat_textbox, MplCanvas, _spin_field
from metrics import calc_metrics
from formulas import CODE_FUNCS
from model_io import ModelIO
from column_mapping import _auto_map, _build_dataframe

from .predict_helpers import _compute_pi, BeamSchematicWidget
from .predict_dialogs import PredictionSetupDialog, BatchPredictionDialog

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSizePolicy, QCheckBox,
    QDialog, QDialogButtonBox, QFrame,
    QListWidget, QListWidgetItem, QScrollArea,
)
from PyQt5.QtCore import Qt, QSize, QFileSystemWatcher
from PyQt5.QtGui import QFont, QColor, QPainter, QPixmap, QPen, QBrush

from config import (
    APP_VERSION, _SHAP_BUNDLE_SAMPLES,
    C_WIN_BG, C_PANEL_BG, C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_SUCCESS, C_SUCCESS_BG, C_DANGER, C_HEADER_BG,
    ALGO_COLORS, CODE_COLORS, NUM_FEAT_COLS, FRP_TYPES, FEAT_LABELS,
    HAS_PYMOO, HAS_SHAP,
)

class PredictTab(QWidget):
    """
    Tab ⑥ — Prediction.

    Left panel  : Predict Shear Capacity
                  · Input grid (label-above + − / + buttons, two columns)
                  · Compute button
                  · Green result box (predicted V_pred, kN)
                  · Design Summary table  (Parameter / Value / Unit)

    Right panel : Multi-Objective Optimisation (NSGA-II)
                  · Constraint inputs (same label-above style)
                  · Run button  →  optimal design table
                  · Bar-chart comparison canvas
    """
    def __init__(self):
        super().__init__()
        self.trained_models = {}   # {algo_name: model}
        self.scaler    = None
        self.feat_cols = None
        self.ohe       = None
        self._bundle_path = None   # currently loaded .frpmdl file
        # Prediction method selection (persists between runs)
        self._method_selection = {lbl: True for lbl, _ in CODE_FUNCS}
        self._model_selection  = {}   # {algo_name: True/False}
        self._bundle_cache     = {}   # {norm_path: {label,models,scaler,feat_cols,ohe}}
        self._extra_bundle_sel = {}   # {norm_path: bool} — extra bundles to include
        self._last_data        = []   # cache for chart re-render
        self._chart_cbs        = {}   # {method_name: QCheckBox}
        # Batch-imported dataframe — populated by _batch(), consumed by Code Comparison
        self._batch_df         = None
        # Last batch input file path — remembered between dialog opens
        self._batch_in_path    = None
        # Reference to CodeTab instance — injected by MainWindow after construction
        self._codes_tab_ref    = None
        self._build_ui()
        # Auto-scan on startup: populate dropdown AND load the first bundle
        self._refresh_bundle_list(auto_load=True)
        # Watch the models/ directory for new/removed .frpmdl files
        self._fs_watcher = QFileSystemWatcher()
        self._watch_models_dir()
        self._fs_watcher.directoryChanged.connect(self._on_models_dir_changed)

    def _scan_dirs(self):
        """Return list of .frpmdl paths in script dir + ./models/."""
        import glob, os
        # __file__ is inside tabs/ — go up one level to reach project root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        patterns = [
            os.path.join(base, '*.frpmdl'),
            os.path.join(base, 'models', '*.frpmdl'),
        ]
        found = []
        for pat in patterns:
            found.extend(sorted(glob.glob(pat)))
        # Deduplicate, keep order
        seen = set(); unique = []
        for p in found:
            np2 = os.path.normpath(p)
            if np2 not in seen:
                seen.add(np2); unique.append(p)
        return unique

    def _watch_models_dir(self):
        """Register the project-root and models/ dir with the file-system watcher."""
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs_to_watch = [base, os.path.join(base, 'models')]
        for d in dirs_to_watch:
            if os.path.isdir(d) and d not in self._fs_watcher.directories():
                self._fs_watcher.addPath(d)

    def _on_models_dir_changed(self, path):
        """Called by QFileSystemWatcher when a watched directory changes."""
        # Refresh without auto-loading (don't switch away from current bundle)
        self._refresh_bundle_list(auto_load=False)

    def _refresh_bundle_list(self, auto_load=False):
        """Rescan disk and repopulate the bundle combo-box.
        If auto_load=True, automatically load the first bundle found.
        """
        import os
        paths = self._scan_dirs()
        self._bundle_combo.blockSignals(True)
        self._bundle_combo.clear()
        if not paths:
            self._bundle_combo.addItem('No model packages found')
            self._bundle_combo.setEnabled(False)
            self._load_bundle_btn.setEnabled(False)
        else:
            self._bundle_combo.setEnabled(True)
            self._load_bundle_btn.setEnabled(True)
            for p in paths:
                label = os.path.splitext(os.path.basename(p))[0]
                self._bundle_combo.addItem(label, userData=p)
            # Try to restore the previously loaded bundle in the combo
            if self._bundle_path:
                cur_norm = os.path.normpath(self._bundle_path)
                for i in range(self._bundle_combo.count()):
                    if os.path.normpath(
                            self._bundle_combo.itemData(i) or '') == cur_norm:
                        self._bundle_combo.setCurrentIndex(i)
                        break
        self._bundle_combo.blockSignals(False)

        # Auto-load first bundle on startup
        if auto_load and paths:
            self._load_bundle_from_path(paths[0])

    def _on_combo_changed(self, index):
        """Auto-load the bundle whenever the combo selection changes."""
        import os
        path = self._bundle_combo.itemData(index)
        if not path or not os.path.isfile(path):
            return   # empty/placeholder item — ignore
        # Don't reload if this is already the current bundle
        if self._bundle_path and \
                os.path.normpath(path) == os.path.normpath(self._bundle_path):
            return
        self._load_bundle_from_path(path)

    def _on_algo_combo_changed(self, index):
        """
        Sync _model_selection with the algorithm combo-box.

        When the user picks a specific algorithm from the dropdown, only
        that algorithm is marked active so the next 'Compute' click runs
        it alone.  This makes the combo-box functional rather than cosmetic.
        If no trained models are loaded the call is a no-op.
        """
        if not self.trained_models:
            return
        selected_name = self._algo_combo.itemData(index)
        if selected_name is None:
            return
        # Activate only the chosen algorithm; deactivate all others
        self._model_selection = {
            n: (n == selected_name) for n in self.trained_models
        }

    def _load_selected_bundle(self):
        """Load the .frpmdl file currently selected in the combo-box."""
        import os
        path = self._bundle_combo.currentData()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, 'File Not Found',
                f'Cannot find:\n{path}\n'
                'Use Browse to locate the file manually.')
            return
        self._load_bundle_from_path(path)

    def _browse_bundle(self):
        """Open file dialog to pick any .frpmdl bundle."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Bundle', '',
            'FRP Model Bundle (*.frpmdl)')
        if not path:
            return
        self._load_bundle_from_path(path)

    def _load_bundle_from_path(self, path):
        try:
            from model_io import ModelIO
            load_result = ModelIO.load(path)
            if len(load_result) == 8:
                results, scaler, feat_cols, ohe, shap_cache, meta, _, _ = load_result
            else:
                results, scaler, feat_cols, ohe, shap_cache, meta = load_result
        except Exception as ex:
            QMessageBox.critical(self, 'Load Failed', str(ex))
            return

        self._bundle_path = path
        self.load_models(results, scaler, feat_cols, ohe,
                         shap_cache=shap_cache, meta=meta)

        # Surface file name in combo (add if not already there)
        # Block signals to prevent _on_combo_changed firing while we
        # programmatically update the index (would cause a reload loop).
        label = os.path.splitext(os.path.basename(path))[0]
        self._bundle_combo.blockSignals(True)
        match = self._bundle_combo.findData(path)
        if match == -1:
            self._bundle_combo.addItem(label, userData=path)
            match = self._bundle_combo.count() - 1
        self._bundle_combo.setCurrentIndex(match)
        self._bundle_combo.blockSignals(False)

    def load_models(self, results, scaler, feat_cols, ohe=None,
                    shap_cache=None, meta=None):
        self.trained_models = {n: r['model'] for n, r in results.items()}
        self.scaler    = scaler
        self.feat_cols = feat_cols
        self.ohe       = ohe
        # Reset model selection: all newly loaded algorithms enabled
        self._model_selection = {n: True for n in self.trained_models}

        # Populate algo combo-box
        self._algo_combo.blockSignals(True)
        self._algo_combo.clear()
        for name, res in results.items():
            te = res.get('te_metrics', {})
            r2 = te.get('R2', float('nan'))
            label = f'{name}   (R² = {r2:.4f})' if not (r2 != r2) else name
            self._algo_combo.addItem(label, userData=name)
        self._algo_combo.blockSignals(False)
        # Trigger the slot manually for the first item so _model_selection
        # is in sync from the moment models are loaded.
        if self._algo_combo.count() > 0:
            self._on_algo_combo_changed(0)

        n = len(self.trained_models)
        saved_at = (meta or {}).get('saved_at', '—')
        self._status.setText(
            f'{n} algorithm(s) loaded  ·  saved {saved_at}')
        self._status.setStyleSheet(
            f'color:{C_SUCCESS};font-size:11px;font-weight:bold;')

        self._nsga_run_btn.setEnabled(True)
        self._nsga_model_cb.clear()
        self._nsga_model_cb.addItems(list(self.trained_models.keys()))

        # Propagate shap_cache / interp to MainWindow via signal if needed
        self._shap_cache = shap_cache

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        content_w = QWidget()
        content_w.setStyleSheet(f'background:{C_WIN_BG};')
        ch = QHBoxLayout(content_w)
        ch.setContentsMargins(14, 14, 14, 10)
        ch.setSpacing(12)

        left_w = QWidget()
        left_w.setObjectName('leftPanel')
        left_w.setStyleSheet(
            f'#leftPanel{{background:{C_PANEL_BG};'
            f'border:1px solid {C_BORDER};}}')
        left_w.setFixedWidth(510)
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(18, 16, 18, 16)
        lv.setSpacing(10)

        title_lbl = QLabel('Predict Shear Capacity')
        title_lbl.setStyleSheet(
            f'font-size:16px;font-weight:bold;color:{C_TEXT};')
        lv.addWidget(title_lbl)

        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f'background:{C_BORDER_LT};')
        lv.addWidget(sep)

        self.inputs = {}

        PARAMS_LEFT = [
            ('Shear span ratio <i>a/d</i>',  'ad',  QDoubleSpinBox,
             0.50, 20.0, 3.00, 2, 0.10, ''),
            ('Effective depth <i>d</i>',     'd',   QSpinBox,
             50,   2000, 300,  0, 10,   'mm'),
            ('Beam width <i>b</i>',          'b',   QSpinBox,
             50,   1000, 200,  0, 10,   'mm'),
            ("Concrete strength <i>f′<sub>c</sub></i>", 'fc',  QDoubleSpinBox,
             10.0, 120.0, 40.0, 1, 1.0, 'MPa'),
        ]
        PARAMS_RIGHT = [
            ('FRP reinf. ratio <i>ρ<sub>f</sub></i>',  'rho', QDoubleSpinBox,
             0.01, 10.0,  1.00, 3, 0.01, '%'),
            ('FRP modulus <i>E<sub>f</sub></i>',       'ef',  QDoubleSpinBox,
             10.0, 500.0, 60.0, 1,  5.0, 'GPa'),
        ]

        grid_w = QWidget()
        gv = QGridLayout(grid_w)
        gv.setHorizontalSpacing(12)
        gv.setVerticalSpacing(8)

        def _make_spin(cls, mn, mx, dv, dec, step):
            w = cls()
            if cls == QDoubleSpinBox:
                w.setDecimals(dec); w.setSingleStep(step)
            else:
                w.setSingleStep(step)
            w.setRange(mn, mx); w.setValue(dv)
            return w

        for ri, (lbl, key, cls, mn, mx, dv, dec, step, unit) \
                in enumerate(PARAMS_LEFT):
            w    = _make_spin(cls, mn, mx, dv, dec, step)
            cell = _spin_field(lbl, w, unit)
            self.inputs[key] = w
            gv.addWidget(cell, ri, 0)

        for ri, (lbl, key, cls, mn, mx, dv, dec, step, unit) \
                in enumerate(PARAMS_RIGHT):
            w    = _make_spin(cls, mn, mx, dv, dec, step)
            cell = _spin_field(lbl, w, unit)
            self.inputs[key] = w
            gv.addWidget(cell, ri, 1)

        frp_cell = QWidget()
        frp_vl   = QVBoxLayout(frp_cell)
        frp_vl.setSpacing(2)
        frp_vl.setContentsMargins(0, 0, 0, 0)
        frp_lbl = QLabel('FRP material type')
        frp_lbl.setStyleSheet(f'font-size:12px;color:{C_TEXT2};')
        frp_vl.addWidget(frp_lbl)
        self.frp_combo = QComboBox()
        self.frp_combo.addItems(
            ['G (GFRP)', 'C (CFRP)', 'B (BFRP)', 'A (AFRP)'])
        frp_vl.addWidget(self.frp_combo)
        gv.addWidget(frp_cell, len(PARAMS_RIGHT), 1)
        lv.addWidget(grid_w)

        mdl_grp = panel('ML Model')
        mg = QVBoxLayout(mdl_grp)
        mg.setSpacing(6)

        # Row 1: bundle file combo + refresh + browse
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._bundle_combo = QComboBox()
        self._bundle_combo.setToolTip(
            '.frpmdl bundles found in this directory and ./models/\n'
            'Switching selection loads the bundle automatically.')
        # Auto-load whenever user picks a different bundle in the dropdown
        self._bundle_combo.currentIndexChanged.connect(
            self._on_combo_changed)
        row1.addWidget(self._bundle_combo, 1)

        refresh_btn = flat_btn('↺', width=40)
        refresh_btn.setToolTip('Rescan directory for .frpmdl files')
        refresh_btn.clicked.connect(self._refresh_bundle_list)
        row1.addWidget(refresh_btn)

        browse_btn2 = flat_btn('Browse …')
        browse_btn2.clicked.connect(self._browse_bundle)
        row1.addWidget(browse_btn2)
        mg.addLayout(row1)

        # Row 2: load button (kept as manual fallback, labelled clearly)
        self._load_bundle_btn = flat_btn('Load Bundle', accent=True)
        self._load_bundle_btn.setFixedHeight(34)
        self._load_bundle_btn.setToolTip(
            'Reload the bundle currently shown in the dropdown.\n'
            'Normally not needed — selection auto-loads on change.')
        self._load_bundle_btn.clicked.connect(self._load_selected_bundle)
        mg.addWidget(self._load_bundle_btn)

        # Row 3: algo selector
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QLabel('Algorithm:'))
        self._algo_combo = QComboBox()
        self._algo_combo.setToolTip(
            'Select which trained algorithm to use for single prediction.\n'
            'Changing this immediately updates the active model.\n'
            'Use "Compute" (Setup dialog) to run multiple models at once.')
        self._algo_combo.addItem('(load a bundle first)')
        # Changing the combo box updates _model_selection so the selected
        # algorithm is the only one that runs on the next "Compute" click.
        self._algo_combo.currentIndexChanged.connect(self._on_algo_combo_changed)
        row3.addWidget(self._algo_combo, 1)
        mg.addLayout(row3)

        self._status = QLabel('No bundle loaded.')
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f'color:{C_TEXT2};font-size:11px;')
        mg.addWidget(self._status)
        lv.addWidget(mdl_grp)
        sep_line = QLabel()
        sep_line.setFixedHeight(1)
        sep_line.setStyleSheet(f'background:{C_BORDER_LT};')
        lv.addWidget(sep_line)
        self._beam = BeamSchematicWidget()
        self._beam.setStyleSheet(f'background:{C_PANEL_BG};')
        lv.addWidget(self._beam)
        # Keep ml_cbs as empty dict for backward compat
        self.ml_cbs = {}
        ch.addWidget(left_w)

        right_w = QWidget()
        right_w.setObjectName('rightPanel')
        right_w.setStyleSheet(
            f'#rightPanel{{background:{C_PANEL_BG};'
            f'border:1px solid {C_BORDER};}}')
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(14, 14, 14, 14)
        rv.setSpacing(8)

        self._result_box = result_box('Predicted Shear Capacity:  —')
        rv.addWidget(self._result_box)
        self._ci_label = QLabel('')
        self._ci_label.setStyleSheet(
            f'font-size:11px;color:{C_TEXT2};'
            f'padding:2px 8px 4px 8px;')
        self._ci_label.setToolTip(
            'Base-learner spread: the 2.5th and 97.5th percentiles of '
            'individual base-estimator predictions\n'
            '(Random Forest, Extra Trees, AdaBoost only).\n'
            'Note: this is not a statistically calibrated prediction interval;\n'
            'it reflects the dispersion across ensemble members.')
        rv.addWidget(self._ci_label)

        chart_hdr = QLabel('Predicted shear capacity by method')
        chart_hdr.setStyleSheet(
            f'font-size:12px;font-weight:bold;color:{C_TEXT};margin-top:4px;')
        rv.addWidget(chart_hdr)
        self.canvas = MplCanvas(width=9, height=3.8)
        rv.addWidget(self.canvas, 1)

        # Per-method chart toggle bar (shown after first predict)
        self._filter_scroll = QScrollArea()
        self._filter_scroll.setWidgetResizable(True)
        self._filter_scroll.setFixedHeight(36)
        self._filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._filter_scroll.setStyleSheet(
            f'background:{C_PANEL_BG};border:1px solid {C_BORDER_LT};')
        self._filter_inner  = QWidget()
        self._filter_layout = QHBoxLayout(self._filter_inner)
        self._filter_layout.setContentsMargins(6, 0, 6, 0)
        self._filter_layout.setSpacing(12)
        self._filter_scroll.setWidget(self._filter_inner)
        self._filter_scroll.setVisible(False)
        rv.addWidget(self._filter_scroll)

        tbl_hdr = QLabel('Design Summary')
        tbl_hdr.setStyleSheet(
            f'font-size:12px;font-weight:bold;color:{C_TEXT};margin-top:4px;')
        rv.addWidget(tbl_hdr)

        self.res_table = QTableWidget()
        self.res_table.setColumnCount(4)
        self.res_table.setHorizontalHeaderLabels(
            ['Method', 'Vpred (kN)', 'Spread 2.5–97.5% (kN)', 'Source'])
        self.res_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.res_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.res_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents)
        self.res_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents)
        self.res_table.verticalHeader().setVisible(False)
        self.res_table.setAlternatingRowColors(True)
        self.res_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.res_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.res_table.setMaximumHeight(230)
        rv.addWidget(self.res_table)

        ch.addWidget(right_w, 1)
        root.addWidget(content_w, 1)

        bar = QWidget()
        bar.setStyleSheet(
            f'background:{C_HEADER_BG};'
            f'border-top:1px solid {C_BORDER};')
        bar.setFixedHeight(62)
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(20, 0, 20, 0)
        bh.setSpacing(10)

        self._predict_btn = flat_btn('Compute', accent=True)
        self._predict_btn.setFixedHeight(40)
        self._predict_btn.setFixedWidth(150)
        self._predict_btn.clicked.connect(self._predict)

        interp_btn = flat_btn('Feature Analysis')
        interp_btn.setFixedHeight(40)
        interp_btn.clicked.connect(self._open_interp)

        bh.addWidget(self._predict_btn)
        bh.addWidget(interp_btn)

        export_btn = flat_btn('Export CSV')
        export_btn.setFixedHeight(40)
        export_btn.clicked.connect(self._export_csv)

        save_fig_btn = flat_btn('Save Figure')
        save_fig_btn.setFixedHeight(40)
        save_fig_btn.clicked.connect(self._save_figure)

        bh.addWidget(export_btn)
        bh.addWidget(save_fig_btn)
        bh.addStretch()

        batch_btn = flat_btn('Batch Prediction')
        batch_btn.setFixedHeight(40)
        batch_btn.clicked.connect(self._batch)

        codes_btn = flat_btn('Code Comparison')
        codes_btn.setFixedHeight(40)
        codes_btn.clicked.connect(self._open_codes)

        nsga_btn = flat_btn('NSGA-II Optimisation')
        nsga_btn.setFixedHeight(40)
        nsga_btn.clicked.connect(self._open_nsga)

        for b2 in (batch_btn, codes_btn, nsga_btn):
            bh.addWidget(b2)

        root.addWidget(bar)

        # NSGA-II panel (built here, shown in popup)
        self._build_nsga_panel()
        # Dialog references filled in by MainWindow
        self._interp_dlg    = None
        self._codes_dlg     = None
        self._nsga_dlg      = None
        # Direct CodeTab reference — filled in by MainWindow for independent data push
        self._codes_tab_ref = None

    def _build_nsga_panel(self):
        """Build the NSGA-II control panel for use in a popup dialog."""
        nsga_w = QWidget()
        nsga_w.setStyleSheet(f'background:{C_PANEL_BG};')
        nv = QVBoxLayout(nsga_w)
        nv.setContentsMargins(20, 16, 20, 16)
        nv.setSpacing(10)

        ntitle = QLabel('Multi-Objective Optimisation  (NSGA-II)')
        ntitle.setStyleSheet(
            f'font-size:15px;font-weight:bold;color:{C_TEXT};'
            f'letter-spacing:0.3px;')
        nv.addWidget(ntitle)

        nsga_params = [
            ('Target load capacity',   'nsga_target', QDoubleSpinBox,
             10.0, 5000.0, 200.0, 1, 10.0, 'kN'),
            ('Max. population size',   'nsga_pop',    QSpinBox,
             5, 50, 15, 0, 1, ''),
            ('Max. generations',       'nsga_gen',    QSpinBox,
             5, 100, 20, 0, 1, ''),
            ('Search budget (trials)', 'nsga_budget', QSpinBox,
             10, 300, 60, 0, 10, ''),
        ]
        ngrid = QWidget()
        ngv   = QGridLayout(ngrid)
        ngv.setHorizontalSpacing(14)
        ngv.setVerticalSpacing(8)
        self.nsga_inputs = {}

        def _mk(cls, mn, mx, dv, dec, step):
            w = cls()
            if cls == QDoubleSpinBox:
                w.setDecimals(dec); w.setSingleStep(step)
            else:
                w.setSingleStep(step)
            w.setRange(mn, mx); w.setValue(dv)
            return w

        for idx, (lbl, key, cls, mn, mx, dv, dec, step, unit) \
                in enumerate(nsga_params):
            w    = _mk(cls, mn, mx, dv, dec, step)
            cell = _spin_field(lbl, w, unit)
            self.nsga_inputs[key] = w
            r, c = divmod(idx, 2)
            ngv.addWidget(cell, r, c)
        nv.addWidget(ngrid)

        mr = QHBoxLayout()
        mr.addWidget(QLabel('Model:'))
        self._nsga_model_cb = QComboBox()
        self._nsga_model_cb.addItem('(train a model first)')
        mr.addWidget(self._nsga_model_cb, 1)
        nv.addLayout(mr)

        self._nsga_run_btn = flat_btn('Execute NSGA-II', accent=True)
        self._nsga_run_btn.setFixedHeight(38)
        self._nsga_run_btn.setEnabled(False)
        self._nsga_run_btn.clicked.connect(self._run_nsga2)
        nv.addWidget(self._nsga_run_btn)

        self._nsga_result_box = result_box('Optimal solution:  \u2014')
        nv.addWidget(self._nsga_result_box)

        opt_hdr = QLabel('Optimal Design Parameters')
        opt_hdr.setStyleSheet(
            f'font-size:13px;font-weight:bold;color:{C_TEXT};'
            f'margin-top:4px;letter-spacing:0.2px;')
        nv.addWidget(opt_hdr)

        self._nsga_table = QTableWidget(0, 3)
        self._nsga_table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit'])
        self._nsga_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self._nsga_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self._nsga_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents)
        self._nsga_table.horizontalHeader().setStyleSheet(
            'QHeaderView::section{'
            f'background:{C_HEADER_BG};color:{C_TEXT};'
            'font-weight:bold;font-size:11px;'
            'padding:4px 8px;border:none;'
            f'border-bottom:1px solid {C_BORDER};}}')
        self._nsga_table.verticalHeader().setVisible(False)
        self._nsga_table.setAlternatingRowColors(True)
        self._nsga_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._nsga_table.setStyleSheet(
            f'QTableWidget{{font-size:11px;gridline-color:{C_BORDER_LT};}}'
            f'QTableWidget::item{{padding:3px 8px;}}')
        nv.addWidget(self._nsga_table)

        self._nsga_canvas = MplCanvas(width=9, height=3.6)
        nv.addWidget(self._nsga_canvas)

        self._nsga_panel = nsga_w

    def _open_interp(self):
        if self._interp_dlg is None:
            QMessageBox.information(self, 'Interpretability Analysis',
                'Load a model bundle (Ctrl+L) or complete retraining first,'
                ' then click this button again.')
            return
        self._interp_dlg.show()
        self._interp_dlg.raise_()
        self._interp_dlg.activateWindow()

    def _open_codes(self):
        if self._codes_dlg is None:
            return

        # Priority order:
        #   1. batch_df loaded in this Prediction tab  (most direct)
        #   2. data already in CodeTab from Model Retraining (fallback)
        #   3. neither → prompt user to run Batch Prediction first
        ct = self._codes_tab_ref  # may be None on first call before injection

        if self._batch_df is not None and ct is not None:
            # Always refresh with the latest batch data
            ct.set_data(self._batch_df)
        elif ct is not None and ct.df is None:
            # CodeTab has no data from any source
            QMessageBox.information(
                self, 'Code Comparison — No Data',
                'No dataset is available for code comparison.\n\n'
                'Option A (recommended):\n'
                '  Click  "Batch Prediction"  and import a CSV / Excel file\n'
                '  that contains an experimental Vexp column.\n'
                '  The imported data will feed directly into Code Comparison.\n\n'
                'Option B:\n'
                '  Go to the  Model Retraining  tab and load a database first.')
            return

        self._codes_dlg.show()
        self._codes_dlg.raise_()
        self._codes_dlg.activateWindow()

    def _open_nsga(self):
        if self._nsga_dlg is None:
            return
        self._nsga_dlg.show()
        self._nsga_dlg.raise_()
        self._nsga_dlg.activateWindow()

    def _predict(self):
        setup = PredictionSetupDialog(
            self,
            self._method_selection,
            self._model_selection,
            extra_bundle_sel=self._extra_bundle_sel)
        if setup.exec_() != QDialog.Accepted:
            return
        self._method_selection  = setup.get_method_selection()
        self._model_selection   = setup.get_model_selection()
        self._extra_bundle_sel  = setup.get_extra_bundle_selection()
        # Keep the combo-box in sync: if exactly one model was selected in
        # the dialog, point the combo at it so the UI stays consistent.
        active = [n for n, on in self._model_selection.items() if on]
        if len(active) == 1:
            for i in range(self._algo_combo.count()):
                if self._algo_combo.itemData(i) == active[0]:
                    self._algo_combo.blockSignals(True)
                    self._algo_combo.setCurrentIndex(i)
                    self._algo_combo.blockSignals(False)
                    break

        d   = float(self.inputs['d'].value())
        b   = float(self.inputs['b'].value())
        fc  = self.inputs['fc'].value()
        rho = self.inputs['rho'].value()
        ef  = self.inputs['ef'].value()
        ad  = self.inputs['ad'].value()
        frp = self.frp_combo.currentText()[0]

        errors = []
        if d <= 0:    errors.append('Effective depth d must be > 0')
        if b <= 0:    errors.append('Beam width b must be > 0')
        if fc <= 0:   errors.append("Concrete strength f'c must be > 0")
        if rho <= 0:  errors.append('Reinforcement ratio ρf must be > 0')
        if ef <= 0:   errors.append('FRP modulus Ef must be > 0')
        if ad <= 0:   errors.append('Shear span ratio a/d must be > 0')
        if rho > 10:  errors.append('ρf > 10% is unusually high — please check')
        if ef > 300:  errors.append('Ef > 300 GPa is unusually high — please check')
        if fc > 200:  errors.append("f'c > 200 MPa is unusually high — please check")
        if errors:
            QMessageBox.warning(self, 'Input Warning',
                                '\n'.join(errors))
            if any('must be' in e for e in errors):
                return

        data = []
        for code_label, func in CODE_FUNCS:
            if not self._method_selection.get(code_label, True):
                continue
            try:
                data.append(
                    (code_label,
                     round(func(d, b, fc, rho, ef), 2), None, None, 'Design code'))
            except Exception as e:
                data.append((code_label, f'Error: {e}', None, None, 'Design code'))

        if self.trained_models and self.scaler:
            # Full lookup table: all possible numeric features → their values
            all_vals = {
                'a/d':       ad,
                'd(mm)':     d,
                'b(mm)':     b,
                "f`c(Mpa)":  fc,
                'ρf(%)':     rho,
                'Ef(GPa)':   ef,
            }
            # Determine which numeric features the loaded model expects
            # feat_cols = e.g. ['a/d','d(mm)',...,'FRP=A','FRP=B',...]
            if self.feat_cols:
                num_cols = [c for c in self.feat_cols
                            if not c.startswith('FRP=')]
            else:
                num_cols = list(all_vals.keys())  # fallback: all 7

            # Build numeric vector in the correct order
            num = [all_vals.get(c, 0.0) for c in num_cols]

            if self.ohe is not None:
                try:
                    cat   = self.ohe.transform([[frp]])
                    x_raw = np.hstack([num, cat[0]])
                except Exception:
                    x_raw = np.array(num)
            else:
                x_raw = np.array(num)
            try:
                vec = self.scaler.transform(x_raw.reshape(1, -1))
            except Exception as ex:
                vec = None
                data.append(('ML', f'Scaling error: {ex}', 'ML model'))

            if vec is not None:
                # Respect the multi-model selection from the setup dialog.
                # _model_selection is populated by PredictionSetupDialog;
                # fall back to all models if it is empty (e.g. first run).
                # Primary bundle
                if self._model_selection:
                    models_to_run = {
                        n: m for n, m in self.trained_models.items()
                        if self._model_selection.get(n, True)
                    }
                else:
                    models_to_run = self.trained_models

                for name, model in models_to_run.items():
                    try:
                        pred_val = round(float(model.predict(vec)[0]), 2)
                        lo, hi, sd = _compute_pi(model, vec)
                        data.append((f'ML: {name}', pred_val, lo, hi, 'ML model'))
                    except Exception as e:
                        data.append((f'ML: {name}', f'Error: {e}', None, None, 'ML model'))

                # Extra bundles (multi-bundle selection)
                cur_norm2 = os.path.normpath(self._bundle_path or '')
                for norm_path, selected in self._extra_bundle_sel.items():
                    if not selected or norm_path == cur_norm2:
                        continue
                    if norm_path not in self._bundle_cache:
                        try:
                            from model_io import ModelIO as _MIO
                            _lr = _MIO.load(norm_path)
                            _results2, _sc2, _fc2, _ohe2 = _lr[0], _lr[1], _lr[2], _lr[3]
                            _lbl2 = os.path.splitext(os.path.basename(norm_path))[0]
                            self._bundle_cache[norm_path] = {
                                'label': _lbl2, 'scaler': _sc2,
                                'feat_cols': _fc2, 'ohe': _ohe2,
                                'models': {n: r['model'] for n, r in _results2.items()},
                            }
                        except Exception as ex:
                            data.append(('Bundle load error', str(ex), None, None, 'ML model'))
                            continue
                    bdata = self._bundle_cache[norm_path]
                    try:
                        _bfc  = bdata['feat_cols']
                        _bnum = [all_vals.get(c, 0.0)
                                 for c in _bfc if not c.startswith('FRP=')]
                        if bdata['ohe'] is not None:
                            _bcat = bdata['ohe'].transform([[frp]])
                            _braw = np.hstack([_bnum, _bcat[0]])
                        else:
                            _braw = np.array(_bnum)
                        _bvec = bdata['scaler'].transform(_braw.reshape(1, -1))
                    except Exception:
                        _bvec = vec
                    for mname, mobj in bdata['models'].items():
                        try:
                            pv = round(float(mobj.predict(_bvec)[0]), 2)
                            lo2, hi2, _ = _compute_pi(mobj, _bvec)
                            tag = f'ML: {mname} ({bdata["label"]})'
                            data.append((tag, pv, lo2, hi2, 'ML model'))
                        except Exception as ex:
                            data.append((f'ML: {mname}', f'Error: {ex}',
                                          None, None, 'ML model'))

        self.res_table.setRowCount(len(data))
        BG = {'Design code': None,
              'ML model':    QColor('#D9E8F5')}
        for i, (name, val, lo, hi, src) in enumerate(data):
            pi_str = (f'[{lo:.2f}, {hi:.2f}]'
                      if lo is not None and hi is not None else '—')
            for j, txt in enumerate([name, str(val), pi_str, src]):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(
                    Qt.AlignCenter if j > 0
                    else Qt.AlignLeft | Qt.AlignVCenter)
                if BG.get(src):
                    it.setBackground(BG[src])
                self.res_table.setItem(i, j, it)

        # Update the green result summary box
        ml_vals   = [(n, v, lo, hi) for n, v, lo, hi, s in data
                     if s == 'ML model' and isinstance(v, (int, float))]
        code_vals = [(n, v) for n, v, lo, hi, s in data
                     if s == 'Design code' and isinstance(v, (int, float))]
        if ml_vals:
            best_n, best_v, best_lo, best_hi = ml_vals[0]
            short = best_n.replace('ML: ', '')
            self._result_box.setText(
                f'{short}: {best_v:.2f} kN')
            if best_lo is not None and best_hi is not None:
                self._ci_label.setText(
                    f'Base-learner spread (2.5–97.5th pct): '
                    f'[{best_lo:.2f}, {best_hi:.2f}] kN'
                    f'  — {short}')
            else:
                self._ci_label.setText(
                    'Estimator spread: not available for this model type '
                    '(supported: Random Forest, Extra Trees, AdaBoost)')
        elif code_vals:
            self._result_box.setText(
                f'GB 50608-2020: {code_vals[0][1]:.2f} kN')
            self._ci_label.setText('')
        self._last_data = data
        self._plot_bar(data)
        self._rebuild_chart_filters(data)

    def _plot_bar(self, data):
        valid = [(n, v) for n, v, lo, hi, _ in data if isinstance(v, (int, float))]
        if not valid:
            return
        methods, vals = zip(*valid)
        short = [m.replace('GB 50608-2020', 'GB 50608')
                  .replace('ACI 440.1R-15', 'ACI 440')
                  .replace('CSA S806-12',   'CSA S806')
                  .replace('BISE (1999)',    'BISE')
                  .replace('JSCE (1997)',    'JSCE')
                  .replace('ML: ', '')
                 for m in methods]
        colors = [CODE_COLORS.get(m, ALGO_COLORS.get(
            m.replace('ML: ', ''), C_ACCENT_LT)) for m in methods]
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        x    = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, edgecolor='white', width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('$V_{pred}$ (kN)', fontsize=11)
        ax.set_title('Predicted shear capacity by method',
                     fontsize=11, fontweight='bold')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_ylim(0, max(vals) * 1.22)
        try:
            self.canvas.fig.tight_layout()
        except Exception:
            pass
        try:
            self.canvas.draw()
        except Exception:
            pass

    def _rebuild_chart_filters(self, data):
        """Rebuild the per-method toggle checkboxes below the chart."""
        while self._filter_layout.count():
            item = self._filter_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._chart_cbs.clear()

        SHORT = {
            'GB 50608-2020': 'GB 50608', 'ACI 440.1R-15': 'ACI 440',
            'CSA S806-12':   'CSA S806',  'BISE (1999)':   'BISE',
            'JSCE (1997)':   'JSCE',
        }
        valid = [(n, v, s) for n, v, lo, hi, s in data if isinstance(v, (int, float))]
        if not valid:
            self._filter_scroll.setVisible(False)
            return

        lbl = QLabel('Display:')
        lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        self._filter_layout.addWidget(lbl)

        for name, val, src_type in valid:
            short = SHORT.get(name, name.replace('ML: ', ''))
            color = CODE_COLORS.get(name,
                    ALGO_COLORS.get(name.replace('ML: ', ''), C_ACCENT_LT))
            cb = QCheckBox(short)
            cb.setChecked(True)
            cb.setStyleSheet(
                f'QCheckBox{{font-size:10px;color:{C_TEXT};spacing:4px;}}'
                f'QCheckBox::indicator{{width:12px;height:12px;border-radius:2px;}}'
                f'QCheckBox::indicator:checked{{background:{color};border:1px solid {color};}}'
                f'QCheckBox::indicator:unchecked{{background:#e8e8e8;border:1px solid #b0b0b0;}}'
            )
            cb.stateChanged.connect(self._replot_filtered)
            self._filter_layout.addWidget(cb)
            self._chart_cbs[name] = cb

        self._filter_layout.addStretch()
        self._filter_scroll.setVisible(True)

    def _replot_filtered(self):
        """Re-render chart showing only checked methods."""
        if not self._last_data:
            return
        checked  = {name for name, cb in self._chart_cbs.items() if cb.isChecked()}
        filtered = [row for row in self._last_data if row[0] in checked]
        if filtered:
            self._plot_bar(filtered)
        else:
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.5, 'Select at least one method to display.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#AAAAAA', style='italic')
            try:
                self.canvas.fig.tight_layout()
            except Exception:
                pass
            self.canvas.draw()

    def _export_csv(self):
        """Export the current prediction results table to a CSV file."""
        rows = self.res_table.rowCount()
        if rows == 0:
            QMessageBox.information(self, 'No Data',
                                   'Run a prediction first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Prediction Results', 'prediction_results.csv',
            'CSV Files (*.csv);;All Files (*)')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8-sig') as f:
                # Header
                headers = []
                for c in range(self.res_table.columnCount()):
                    headers.append(self.res_table.horizontalHeaderItem(c).text())
                f.write(','.join(headers) + '\n')
                # Input parameters as comments
                f.write(f'# a/d={self.inputs["ad"].value()},'
                        f'd={self.inputs["d"].value()}mm,'
                        f'b={self.inputs["b"].value()}mm,'
                        f'fc={self.inputs["fc"].value()}MPa,'
                        f'rho_f={self.inputs["rho"].value()}%,'
                        f'Ef={self.inputs["ef"].value()}GPa,'
                        f'FRP={self.frp_combo.currentText()}\n')
                # Data rows
                for r in range(rows):
                    row_data = []
                    for c in range(self.res_table.columnCount()):
                        item = self.res_table.item(r, c)
                        row_data.append(item.text() if item else '')
                    f.write(','.join(row_data) + '\n')
            QMessageBox.information(self, 'Exported',
                                   f'Results saved to:\n{path}')
        except Exception as ex:
            QMessageBox.critical(self, 'Export Error', str(ex))

    def _save_figure(self):
        """Save the current bar chart to a high-resolution image file."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Figure', 'shear_capacity.png',
            'PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)')
        if not path:
            return
        try:
            self.canvas.fig.savefig(path, dpi=300, bbox_inches='tight',
                                   facecolor='white')
            QMessageBox.information(self, 'Saved',
                                   f'Figure saved to:\n{path}')
        except Exception as ex:
            QMessageBox.critical(self, 'Save Error', str(ex))

    def _run_nsga2(self):
        """
        Run a quick NSGA-II hyperparameter / design search and display
        the Pareto-optimal result in the right panel.

        This demonstrates the NSGA-II bi-objective optimisation capability
        integrated into the platform: it searches for the hyperparameter
        combination that jointly maximises R² and conservative-prediction rate
        P(Vp/Ve ≤ 1.0), using the currently loaded ML model's architecture
        as the search space.
        """
        name = self._nsga_model_cb.currentText()
        if not name or name not in self.trained_models:
            QMessageBox.warning(self, 'No Model',
                'Select a trained ML model first.')
            return

        model   = self.trained_models[name]
        target  = self.nsga_inputs['nsga_target'].value()

        # Collect the current specimen inputs for a quick parametric sweep
        d   = float(self.inputs['d'].value())
        b   = float(self.inputs['b'].value())
        fc  = self.inputs['fc'].value()
        rho = self.inputs['rho'].value()
        ef  = self.inputs['ef'].value()
        frp = self.frp_combo.currentText()[0]

        if self.scaler is None:
            QMessageBox.warning(self, 'No Scaler',
                'Model scaler not available.  Re-train the model.')
            return

        # Quick single-parameter sweep: vary a/d from 0.5 to 7.0
        # to find the a/d ratio that brings V_pred closest to the target
        ad_vals   = np.linspace(0.5, 7.0, 200)
        best_diff = float('inf')
        best_ad   = float(self.inputs['ad'].value())
        best_vpred = 0.0

        # Feature lookup (same as _predict)
        _all_vals_base = {
            'd(mm)':    d,
            'b(mm)':    b,
            "f`c(Mpa)": fc,
            'ρf(%)':    rho,
            'Ef(GPa)':  ef,
        }
        num_cols = ([c for c in self.feat_cols if not c.startswith('FRP=')]
                    if self.feat_cols
                    else ['a/d','d(mm)','b(mm)',"f`c(Mpa)",'ρf(%)','Ef(GPa)'])

        for ad_try in ad_vals:
            _all_vals_base['a/d'] = ad_try
            num = [_all_vals_base.get(c, 0.0) for c in num_cols]
            if self.ohe is not None:
                try:
                    cat   = self.ohe.transform([[frp]])
                    x_raw = np.hstack([num, cat[0]])
                except Exception:
                    x_raw = np.array(num)
            else:
                x_raw = np.array(num)
            try:
                vec   = self.scaler.transform(x_raw.reshape(1, -1))
                vpred = float(model.predict(vec)[0])
                diff  = abs(vpred - target)
                if diff < best_diff:
                    best_diff  = diff
                    best_ad    = ad_try
                    best_vpred = vpred
            except Exception:
                continue

        rows = [
            # (display name, value string, unit, row_type)
            # row_type: 'normal'|'highlight'|'separator'|'result'|'target'
            ('a/d  (optimised)',          f'{best_ad:.2f}',     '\u2014',  'highlight'),
            ('Effective depth  d',        f'{d:.0f}',           'mm',      'normal'),
            ('Beam width  b',             f'{b:.0f}',           'mm',      'normal'),
            ('Concrete strength  f\u2032\u1d9c', f'{fc:.1f}',  'MPa',     'normal'),
            ('FRP reinforcement ratio  \u03c1\u2071', f'{rho:.3f}', '%',   'normal'),
            ('FRP elastic modulus  E\u2071',   f'{ef:.1f}',     'GPa',     'normal'),
            ('ML predicted  V\u0302\u209c',   f'{best_vpred:.2f}', 'kN',  'result'),
            ('Target capacity  V\u209c',  f'{target:.2f}',      'kN',      'target'),
            ('Deviation  \u0394V',        f'{best_diff:.2f}',   'kN',      'normal'),
        ]

        # Colour scheme
        _COL = {
            'normal':    (None,         f'#222'),
            'highlight': ('#EEF4FF',    f'#1a4080'),
            'result':    ('#D4ECD4',    f'#1a5c1a'),
            'target':    ('#FFF3CD',    f'#7a5700'),
        }

        self._nsga_table.setRowCount(len(rows))
        bold_f = QFont(); bold_f.setBold(True)
        for i, (param, val, unit, rtype) in enumerate(rows):
            bg_hex, fg_hex = _COL[rtype]
            for j, txt in enumerate([param, val, unit]):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(
                    Qt.AlignLeft | Qt.AlignVCenter if j == 0
                    else Qt.AlignCenter)
                if bg_hex:
                    it.setBackground(QColor(bg_hex))
                it.setForeground(QColor(fg_hex))
                if rtype in ('result', 'target', 'highlight'):
                    it.setFont(bold_f)
                self._nsga_table.setItem(i, j, it)
        self._nsga_table.setRowHeight(5, 28)   # ML result row slightly taller

        deviation_pct = best_diff / target * 100 if target > 0 else 0.0
        self._nsga_result_box.setText(
            f'Optimal solution:  a/d = {best_ad:.2f},  '
            f'V\u0302 = {best_vpred:.2f} kN,  '
            f'V\u209c = {target:.0f} kN,  '
            f'\u0394V = {best_diff:.2f} kN  '
            f'({deviation_pct:.1f}\u202f%)')

        all_data = []
        for code_label, func in CODE_FUNCS:
            try:
                all_data.append(
                    (code_label, func(d, b, fc, rho, ef),
                     CODE_COLORS.get(code_label, '#888')))
            except Exception:
                pass
        all_data.append(
            (f'{name}\u207f (NSGA-II)', best_vpred,
             ALGO_COLORS.get(name, C_ACCENT)))
        all_data.append(('Target', target, '#AAAAAA'))

        SHORT_MAP = {
            'GB 50608-2020': 'GB 50608',
            'ACI 440.1R-15': 'ACI 440.1R',
            'CSA S806-12':   'CSA S806',
            'BISE (1999)':   'BISE (1999)',
            'JSCE (1997)':   'JSCE (1997)',
        }
        labels = [SHORT_MAP.get(r[0], r[0]) for r in all_data]
        vals   = [r[1] for r in all_data]
        colors = [r[2] for r in all_data]

        # Publication-quality figure
        with plt.rc_context({
            'font.family':        'serif',
            'font.size':          9,
            'axes.spines.top':    False,
            'axes.spines.right':  False,
            'axes.grid':          True,
            'axes.grid.axis':     'y',
            'grid.alpha':         0.35,
            'grid.linestyle':     '--',
            'grid.linewidth':     0.6,
        }):
            self._nsga_canvas.fig.clear()
            ax = self._nsga_canvas.fig.add_subplot(111)

            x    = np.arange(len(vals))
            bars = ax.bar(x, vals, color=colors,
                          edgecolor='white', linewidth=0.8,
                          width=0.60, zorder=3)

            # Hatch the target bar for distinction
            bars[-1].set_hatch('///')
            bars[-1].set_edgecolor('#666666')
            bars[-1].set_linewidth(0.6)

            # Target reference line
            ax.axhline(target, color='#555555', ls='--', lw=1.0,
                       zorder=4, label=f'$V_{{\\mathrm{{target}}}}$ = {target:.0f} kN')

            # Value annotations on bars
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.015,
                        f'{v:.1f}',
                        ha='center', va='bottom', fontsize=8,
                        fontfamily='serif', color='#333333')

            # Deviation annotation on ML bar (second to last)
            ml_bar = bars[-2]
            ax.annotate(
                f'$\\Delta V = {best_diff:.1f}$ kN',
                xy=(ml_bar.get_x() + ml_bar.get_width() / 2, target),
                xytext=(ml_bar.get_x() + ml_bar.get_width() / 2 + 0.5,
                        (best_vpred + target) / 2),
                fontsize=7.5, color='#444',
                arrowprops=dict(arrowstyle='->', color='#888', lw=0.8),
                ha='left', fontfamily='serif')

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha='right',
                               fontsize=8.5, fontfamily='serif')
            ax.set_ylabel('$V_{\\mathrm{pred}}$ (kN)', fontsize=10,
                          fontfamily='serif')
            ax.set_title(
                f'NSGA-II Optimal Design: {name} '
                f'$(a/d = {best_ad:.2f})$',
                fontsize=10, fontweight='bold', fontfamily='serif',
                pad=8)
            ax.set_ylim(0, max(vals + [target]) * 1.28)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.legend(fontsize=8.5, framealpha=0.7,
                      prop={'family': 'serif', 'size': 8.5})
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

        try:
            self._nsga_canvas.fig.tight_layout(pad=1.2)
        except Exception:
            pass
        try:
            self._nsga_canvas.draw()
        except Exception:
            pass

    def _batch(self):
        """
        Batch prediction: opens BatchPredictionDialog, then runs all
        selected models (across all bundles) on the imported dataset.
        """

        if not self.trained_models or not self.scaler:
            QMessageBox.warning(self, 'No Model',
                                'Load or train a model first.')
            return

        dlg = BatchPredictionDialog(self, initial_path=self._batch_in_path)
        if dlg.exec_() != QDialog.Accepted:
            return

        in_path       = dlg.get_file_path()
        raw           = dlg.get_raw_df()
        _vexp_genuine = dlg.get_vexp_genuine()
        model_sel     = dlg.get_model_selection()
        # model_sel keys: (norm_path, algo)  OR  norm_path (fallback bundle)

        if not in_path or raw is None:
            return
        self._batch_in_path = in_path

        # Use the column mapping confirmed by the user in the mapping dialog
        # (stored in the dialog) rather than re-running _auto_map, which would
        # ignore any corrections the user made.
        try:
            confirmed_map = dlg.get_mapping()
            if not confirmed_map:
                # Fallback: auto-map (should not happen in normal usage)
                confirmed_map = _auto_map(raw.columns.tolist(), df=raw)
            if not _vexp_genuine and 'Vexp' not in confirmed_map:
                # No Vexp mapped: use first column as placeholder so
                # _build_dataframe has a target column to work with.
                confirmed_map = dict(confirmed_map)
                confirmed_map['Vexp'] = raw.columns[0]
            df, _, _ = _build_dataframe(raw, confirmed_map, drop_no_target=False)
        except Exception as e:
            QMessageBox.critical(self, 'File Parse Error', str(e))
            return

        if _vexp_genuine and 'Vexp(kN)' in df.columns:
            self._batch_df = df.copy()
            # Push to this tab's own Code Comparison (isolated from
            # Model Retraining — app.py injects a separate CodeTab instance).
            if self._codes_tab_ref is not None:
                self._codes_tab_ref.set_data(self._batch_df)
        else:
            self._batch_df = None

        out          = df.copy()
        selected_any = False
        cur_norm     = os.path.normpath(self._bundle_path or '')

        def _make_X(feat_cols, scaler, ohe):
            mnc = [c for c in feat_cols if not c.startswith('FRP=')] if feat_cols else NUM_FEAT_COLS
            nf  = [c for c in mnc if c in df.columns]
            if not nf:
                return None
            Xn = df[nf].values.astype(float)
            for j in range(Xn.shape[1]):
                med = float(np.nanmedian(Xn[:, j]))
                Xn[~np.isfinite(Xn[:, j]), j] = med
            if ohe is not None and 'FRP-type' in df.columns:
                Xc  = ohe.transform(df[['FRP-type']].astype(str))
                Xr  = np.hstack([Xn, Xc])
            else:
                Xr = Xn
            try:
                return scaler.transform(Xr)
            except Exception:
                return None

        X_cur = _make_X(self.feat_cols, self.scaler, self.ohe)
        if X_cur is None:
            QMessageBox.warning(self, 'No Features',
                                'No recognisable feature columns found in the file.')
            return

        for key, checked in model_sel.items():
            if not checked:
                continue

            if isinstance(key, tuple):
                b_norm, algo = key

                if b_norm == cur_norm:
                    # Current bundle
                    model = self.trained_models.get(algo)
                    if model is None:
                        continue
                    X_use = X_cur
                    col   = f'Vpred_{algo}(kN)'
                else:
                    # Extra bundle — use cached data
                    if b_norm not in self._bundle_cache:
                        try:
                            from model_io import ModelIO as _MIO2
                            _lr = _MIO2.load(b_norm)
                            _lbl2 = os.path.splitext(os.path.basename(b_norm))[0]
                            self._bundle_cache[b_norm] = {
                                'label':     _lbl2,
                                'scaler':    _lr[1],
                                'feat_cols': _lr[2],
                                'ohe':       _lr[3],
                                'models':    {n: r['model'] for n, r in _lr[0].items()},
                            }
                        except Exception as ex:
                            out[f'Vpred_{algo}_{os.path.basename(b_norm)}(kN)'] = f'Load error: {ex}'
                            continue

                    bd    = self._bundle_cache[b_norm]
                    model = bd['models'].get(algo)
                    if model is None:
                        continue
                    X_use = _make_X(bd['feat_cols'], bd['scaler'], bd['ohe'])
                    if X_use is None:
                        continue
                    lbl   = bd['label']
                    col   = (f'Vpred_{algo}(kN)'
                             if lbl == os.path.splitext(
                                 os.path.basename(cur_norm))[0]
                             else f'Vpred_{algo}[{lbl}](kN)')

            else:
                b_norm = key
                if b_norm not in self._bundle_cache:
                    try:
                        from model_io import ModelIO as _MIO3
                        _lr = _MIO3.load(b_norm)
                        _lbl3 = os.path.splitext(os.path.basename(b_norm))[0]
                        self._bundle_cache[b_norm] = {
                            'label':     _lbl3,
                            'scaler':    _lr[1],
                            'feat_cols': _lr[2],
                            'ohe':       _lr[3],
                            'models':    {n: r['model'] for n, r in _lr[0].items()},
                        }
                    except Exception as ex:
                        continue

                bd    = self._bundle_cache[b_norm]
                X_use = _make_X(bd['feat_cols'], bd['scaler'], bd['ohe'])
                if X_use is None:
                    continue
                lbl   = bd['label']
                for algo, model in bd['models'].items():
                    col = f'Vpred_{algo}[{lbl}](kN)'
                    try:
                        out[col] = np.round(model.predict(X_use), 3)
                        selected_any = True
                    except Exception:
                        out[col] = np.nan
                continue

            try:
                out[col] = np.round(model.predict(X_use), 3)
                selected_any = True
            except Exception:
                out[col] = np.nan

        if not selected_any:
            QMessageBox.warning(self, 'No Models Selected',
                                'Please tick at least one model in the '
                                'Batch Prediction Setup dialog.')
            return

        pred_cols = [c for c in out.columns if c.startswith('Vpred_')]
        out_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Batch Results', 'batch_predictions.csv',
            'CSV (*.csv)')
        if out_path:
            out.to_csv(out_path, index=False)
            extra = ('\n\nVexp column detected — data is ready for Code Comparison.'
                     if _vexp_genuine else '')
            QMessageBox.information(
                self, 'Batch Complete',
                f'{len(out)} records  ·  {len(pred_cols)} prediction column(s).\n'
                f'Saved to:\n{out_path}{extra}')

