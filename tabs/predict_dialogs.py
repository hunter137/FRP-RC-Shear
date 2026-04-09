import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QCheckBox, QSpinBox, QComboBox,
    QDialog, QDialogButtonBox, QFrame, QGroupBox,
    QListWidget, QListWidgetItem, QScrollArea,
    QFileDialog, QMessageBox, QHeaderView, QPushButton,
    QTableWidget, QTableWidgetItem, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT, C_ACCENT_LT,
    C_PANEL_BG, C_ACCENT_BG, C_HEADER_BG, C_SUCCESS,
    ALGO_COLORS, NUM_FEAT_COLS,
)
from widgets import flat_btn, panel
from model_io import ModelIO
from formulas import CODE_FUNCS

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
from column_mapping import _auto_map, _build_dataframe

class PredictionSetupDialog(QDialog):
    """
    Opens when the user clicks Run Prediction.

    Left column  : checkboxes for each design-code method and the
                   proposed formula.
    Right column : checkboxes for each ML algorithm in the currently
                   loaded bundle, plus a list of other bundles found
                   in the models/ folder so the user can switch.
    """

    def __init__(self, parent_tab, method_sel: dict, model_sel: dict,
                 extra_bundle_sel: dict = None):
        super().__init__(parent_tab)
        self.setWindowTitle('Prediction Configuration')
        self.setMinimumWidth(540)
        self._tab              = parent_tab
        self._method_cbs       = {}   # {label: QCheckBox}
        self._model_cbs        = {}   # {algo:  QCheckBox} — current bundle
        self._bundle_cbs       = {}   # {norm_path: QCheckBox} — extra bundles
        self._method_sel       = dict(method_sel)
        self._model_sel        = dict(model_sel)
        self._extra_bundle_sel = dict(extra_bundle_sel or {})
        self._build_ui()

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setSpacing(14)
        vl.setContentsMargins(20, 16, 20, 14)

        title = QLabel('Select Calculation Methods')
        title.setStyleSheet(f'font-size:13px;font-weight:bold;color:{C_TEXT};')
        vl.addWidget(title)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep)

        body = QHBoxLayout()
        body.setSpacing(24)

        left_grp = QGroupBox('Design Code Methods')
        left_grp.setStyleSheet('QGroupBox{font-weight:bold;font-size:11px;}')
        lv = QVBoxLayout(left_grp)
        lv.setSpacing(5)
        for code_label, _ in CODE_FUNCS:
            cb = QCheckBox(code_label)
            cb.setChecked(self._method_sel.get(code_label, True))
            cb.setStyleSheet('font-size:11px;')
            self._method_cbs[code_label] = cb
            lv.addWidget(cb)
        # Proposed formula
        cb_prop = QCheckBox('Proposed Formula  (this work)')
        cb_prop.setChecked(self._method_sel.get('Proposed', True))
        cb_prop.setStyleSheet('font-size:11px;')
        self._method_cbs['Proposed'] = cb_prop
        lv.addWidget(cb_prop)
        lv.addStretch()

        # Toggle-all button
        tog_codes = QPushButton('Toggle All')
        tog_codes.setFixedHeight(24)
        tog_codes.setStyleSheet('font-size:10px;')
        tog_codes.clicked.connect(
            lambda: self._toggle_all(self._method_cbs))
        lv.addWidget(tog_codes)
        body.addWidget(left_grp, 1)

        right_grp = QGroupBox('ML Models')
        right_grp.setStyleSheet('QGroupBox{font-weight:bold;font-size:11px;}')
        rv = QVBoxLayout(right_grp)
        rv.setSpacing(4)

        all_paths  = self._tab._scan_dirs()
        cur_path   = self._tab._bundle_path or ''
        cur_norm   = os.path.normpath(cur_path) if cur_path else ''

        if not all_paths:
            no_mdl = QLabel('No model packages found in models/ folder.')
            no_mdl.setStyleSheet(f'font-size:11px;color:{C_TEXT2};font-style:italic;')
            no_mdl.setWordWrap(True)
            rv.addWidget(no_mdl)
        else:
            for path in all_paths:
                norm  = os.path.normpath(path)
                label = os.path.splitext(os.path.basename(path))[0]
                is_cur = (norm == cur_norm)

                # Bundle header row
                hdr_row = QHBoxLayout()
                hdr_row.setSpacing(4)
                pkg_lbl = QLabel(label + ('  [loaded]' if is_cur else ''))
                pkg_lbl.setStyleSheet(
                    f'font-size:11px;font-weight:bold;'
                    f'color:{C_ACCENT if is_cur else C_TEXT};')
                hdr_row.addWidget(pkg_lbl)
                hdr_row.addStretch()
                hdr_w = QWidget()
                hdr_w.setLayout(hdr_row)
                rv.addWidget(hdr_w)

                if is_cur:
                    # Per-model checkboxes for the loaded bundle
                    for algo in self._tab.trained_models:
                        r2_str = ''
                        for i in range(self._tab._algo_combo.count()):
                            if self._tab._algo_combo.itemData(i) == algo:
                                txt = self._tab._algo_combo.itemText(i)
                                r2_str = '  ' + txt.split('(')[1].rstrip(')')                                     if '(' in txt else ''
                                break
                        cb = QCheckBox(f'  {algo}{r2_str}')
                        cb.setChecked(self._model_sel.get(algo, True))
                        cb.setStyleSheet('font-size:11px;margin-left:14px;')
                        self._model_cbs[algo] = cb
                        rv.addWidget(cb)
                else:
                    # Extra bundle: single checkbox to include all its models
                    cb = QCheckBox('  Include all models from this package')
                    cb.setChecked(self._extra_bundle_sel.get(norm, False))
                    cb.setStyleSheet(
                        f'font-size:11px;margin-left:14px;color:{C_TEXT2};')
                    self._bundle_cbs[norm] = cb
                    rv.addWidget(cb)
                    note = QLabel('     (loaded automatically on Run Prediction)')
                    note.setStyleSheet(
                        f'font-size:9px;color:{C_TEXT2};font-style:italic;')
                    rv.addWidget(note)

                pkg_sep = QFrame(); pkg_sep.setFrameShape(QFrame.HLine)
                pkg_sep.setStyleSheet(f'color:{C_BORDER_LT};')
                rv.addWidget(pkg_sep)

        rv.addStretch()
        all_cbs = {**self._model_cbs, **self._bundle_cbs}
        if all_cbs:
            tog_mdl = QPushButton('Toggle All')
            tog_mdl.setFixedHeight(24)
            tog_mdl.setStyleSheet('font-size:10px;')
            tog_mdl.clicked.connect(lambda: self._toggle_all(all_cbs))
            rv.addWidget(tog_mdl)

        body.addWidget(right_grp, 1)
        vl.addLayout(body)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep3)

        btn_box = QDialogButtonBox()
        ok_btn  = btn_box.addButton('Run Prediction', QDialogButtonBox.AcceptRole)
        ok_btn.setFixedHeight(30)
        ok_btn.setStyleSheet(
            f'QPushButton{{background:{C_ACCENT};color:#fff;'
            f'border:none;border-radius:3px;font-size:11px;padding:0 14px;}}'
            f'QPushButton:hover{{background:{C_ACCENT_LT};}}')
        can_btn = btn_box.addButton('Cancel', QDialogButtonBox.RejectRole)
        can_btn.setFixedHeight(30)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        vl.addWidget(btn_box)

    @staticmethod
    def _toggle_all(cbs: dict):
        target = not all(cb.isChecked() for cb in cbs.values())
        for cb in cbs.values():
            cb.setChecked(target)

    def get_method_selection(self) -> dict:
        return {lbl: cb.isChecked() for lbl, cb in self._method_cbs.items()}

    def get_model_selection(self) -> dict:
        return {algo: cb.isChecked() for algo, cb in self._model_cbs.items()}

    def get_extra_bundle_selection(self) -> dict:
        return {norm: cb.isChecked() for norm, cb in self._bundle_cbs.items()}

# ══════════════════════════════════════════════════════════════════════
#  BatchPredictionDialog  — data-source lock + full multi-bundle model selector
# ══════════════════════════════════════════════════════════════════════
class BatchPredictionDialog(QDialog):
    """
    Batch Prediction Setup dialog.

    ┌─ Data Source ──────────────────────────────────────────────────┐
    │  🔒  filename.xlsx                        [Unlock & Replace]   │
    │  728 rows · 23 columns · Vexp column detected ✓               │
    └────────────────────────────────────────────────────────────────┘
    ┌─ Model Selection ──────────────────────────────────────────────┐
    │  Scans ALL .frpmdl bundles in the models/ folder.             │
    │                                                                │
    │  ── GBDT  [loaded] ─────────────────────────────────          │
    │   ☑ GBDT  R²=0.79   ☑ XGBoost R²=0.81  ☑ LightGBM R²=0.82  │
    │   ☑ AdaBoost        ☑ Random Forest     ☑ Extra Trees        │
    │   ☑ KNN             ☑ MLP               ☑ SVR               │
    │                                                                │
    │  ── LightGBM_bundle ──────────────────────────────────        │
    │   ☑ Include all models (loaded on run)                        │
    │  [Toggle All]                                                  │
    └────────────────────────────────────────────────────────────────┘
    [Run Batch Prediction]                               [Cancel]
    """

    def __init__(self, parent_tab, initial_path: str = None):
        super().__init__(parent_tab)
        self.setWindowTitle('Batch Prediction')
        self.setMinimumWidth(640)
        self.setMinimumHeight(520)
        self._tab = parent_tab

        self._file_path    = initial_path
        self._raw_df       = None
        self._is_locked    = False
        self._vexp_genuine = False

        # Key: (norm_bundle_path, algo_name) → QCheckBox   (current bundle)
        # Key: norm_bundle_path              → QCheckBox   (extra bundles)
        self._model_cbs   = {}   # {(norm_path, algo): QCheckBox}
        self._bundle_cbs  = {}   # {norm_path: QCheckBox}  extra bundles

        self._build_ui()
        self._rebuild_model_checkboxes()
        if initial_path:
            self._load_file(initial_path)

    # ═══════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(20, 18, 20, 14)

        title = QLabel('Batch Prediction')
        title.setStyleSheet(
            f'font-size:14px;font-weight:bold;color:{C_TEXT};')
        root.addWidget(title)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.HLine)
        sep0.setStyleSheet(f'color:{C_BORDER_LT};')
        root.addWidget(sep0)

        src_grp = QGroupBox('Data Source')
        src_grp.setStyleSheet(self._grp_style())
        sv = QVBoxLayout(src_grp)
        sv.setSpacing(6)
        sv.setContentsMargins(12, 10, 12, 10)

        row1 = QHBoxLayout(); row1.setSpacing(8)
        self._lock_icon = QLabel('[--]')
        self._lock_icon.setFixedWidth(22)
        self._lock_icon.setStyleSheet('font-size:14px;')
        row1.addWidget(self._lock_icon)

        self._path_lbl = QLabel('No file selected — click Browse to load data.')
        self._path_lbl.setStyleSheet(
            f'font-size:11px;color:{C_TEXT2};font-style:italic;')
        self._path_lbl.setWordWrap(True)
        row1.addWidget(self._path_lbl, 1)

        self._toggle_lock_btn = QPushButton('Browse …')
        self._toggle_lock_btn.setFixedHeight(28)
        self._toggle_lock_btn.setFixedWidth(140)
        self._toggle_lock_btn.setStyleSheet(self._accent_btn_style())
        self._toggle_lock_btn.clicked.connect(self._on_lock_toggle)
        row1.addWidget(self._toggle_lock_btn)
        sv.addLayout(row1)

        self._info_lbl = QLabel('')
        self._info_lbl.setStyleSheet(
            f'font-size:10px;color:{C_TEXT2};padding-left:30px;')
        sv.addWidget(self._info_lbl)

        self._preview_lbl = QLabel('')
        self._preview_lbl.setStyleSheet(
            f'font-size:10px;color:{C_TEXT2};padding-left:30px;font-style:italic;')
        self._preview_lbl.setWordWrap(True)
        sv.addWidget(self._preview_lbl)
        root.addWidget(src_grp)

        mdl_grp = QGroupBox('Model Selection')
        mdl_grp.setStyleSheet(self._grp_style())
        mv = QVBoxLayout(mdl_grp)
        mv.setSpacing(6)
        mv.setContentsMargins(12, 10, 12, 10)

        hint = QLabel(
            'All .frpmdl bundles found in the models/ folder are listed below. '
            'Tick the individual algorithms you want included in the batch output.')
        hint.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        hint.setWordWrap(True)
        mv.addWidget(hint)

        # Scrollable area for bundle/model checkboxes
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(180)
        self._scroll.setStyleSheet(
            f'QScrollArea{{border:1px solid {C_BORDER_LT};border-radius:3px;}}')
        self._scroll_inner = QWidget()
        self._scroll_inner.setStyleSheet(f'background:{C_PANEL_BG};')
        self._cb_layout = QVBoxLayout(self._scroll_inner)
        self._cb_layout.setSpacing(0)
        self._cb_layout.setContentsMargins(8, 6, 8, 6)
        self._scroll.setWidget(self._scroll_inner)
        mv.addWidget(self._scroll)

        self._no_mdl_lbl = QLabel(
            'No .frpmdl model bundles found in the models/ folder.')
        self._no_mdl_lbl.setStyleSheet(
            f'font-size:11px;color:{C_TEXT2};font-style:italic;padding:6px;')
        self._no_mdl_lbl.setVisible(False)
        mv.addWidget(self._no_mdl_lbl)

        # Toggle-all + bundle count row
        tog_row = QHBoxLayout()
        self._bundle_count_lbl = QLabel('')
        self._bundle_count_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        tog_row.addWidget(self._bundle_count_lbl)
        tog_row.addStretch()
        self._tog_btn = QPushButton('Toggle All')
        self._tog_btn.setFixedHeight(24)
        self._tog_btn.setFixedWidth(90)
        self._tog_btn.setStyleSheet('font-size:10px;')
        self._tog_btn.clicked.connect(self._toggle_all)
        tog_row.addWidget(self._tog_btn)
        mv.addLayout(tog_row)

        root.addWidget(mdl_grp)

        sep_bot = QFrame(); sep_bot.setFrameShape(QFrame.HLine)
        sep_bot.setStyleSheet(f'color:{C_BORDER_LT};')
        root.addWidget(sep_bot)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        self._run_btn = QPushButton('Run Batch')
        self._run_btn.setFixedHeight(32)
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet(
            f'QPushButton{{background:{C_ACCENT};color:#fff;border:none;'
            f'border-radius:3px;font-size:11px;padding:0 18px;}}'
            f'QPushButton:hover{{background:{C_ACCENT_LT};}}'
            f'QPushButton:disabled{{background:#b0b8c8;color:#e8e8e8;}}')
        self._run_btn.clicked.connect(self.accept)

        can_btn = QPushButton('Cancel')
        can_btn.setFixedHeight(32)
        can_btn.setFixedWidth(80)
        can_btn.setStyleSheet('font-size:11px;')
        can_btn.clicked.connect(self.reject)

        btn_row.addWidget(self._run_btn)
        btn_row.addWidget(can_btn)
        root.addLayout(btn_row)

    # ═══════════════════════════════════════════════════════════════
    # Model checkbox rebuild — scans ALL bundles
    # ═══════════════════════════════════════════════════════════════
    def _rebuild_model_checkboxes(self):
        """
        Scan every .frpmdl file in the models/ directory.
        For the currently-loaded bundle: show one checkbox per algorithm.
        For extra bundles: show a single 'Include all' checkbox.
        """

        # Clear existing widgets
        while self._cb_layout.count():
            item = self._cb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._model_cbs.clear()
        self._bundle_cbs.clear()

        all_paths = self._tab._scan_dirs()
        cur_path  = self._tab._bundle_path or ''
        cur_norm  = os.path.normpath(cur_path) if cur_path else ''

        if not all_paths:
            self._no_mdl_lbl.setVisible(True)
            self._scroll.setVisible(False)
            self._tog_btn.setEnabled(False)
            self._bundle_count_lbl.setText('')
            return

        self._no_mdl_lbl.setVisible(False)
        self._scroll.setVisible(True)
        self._tog_btn.setEnabled(True)

        total_models = 0
        n_bundles    = 0

        for path in all_paths:
            norm  = os.path.normpath(path)
            label = os.path.splitext(os.path.basename(path))[0]
            is_cur = (norm == cur_norm)
            n_bundles += 1

            hdr_w   = QWidget()
            hdr_w.setStyleSheet(
                f'background:{"#eaf0fb" if is_cur else "#f5f5f5"};'
                f'border-bottom:1px solid {C_BORDER_LT};')
            hdr_row = QHBoxLayout(hdr_w)
            hdr_row.setContentsMargins(6, 5, 8, 5)
            hdr_row.setSpacing(6)

            _loaded_tag = '  [loaded]' if is_cur else ''
            pkg_lbl = QLabel(f'\U0001f4e6 {label}{_loaded_tag}')
            pkg_lbl.setStyleSheet(
                f'font-size:11px;font-weight:bold;'
                f'color:{C_ACCENT if is_cur else C_TEXT};'
                f'background:transparent;')
            hdr_row.addWidget(pkg_lbl)
            hdr_row.addStretch()

            self._cb_layout.addWidget(hdr_w)

            cb_container = QWidget()
            cb_container.setStyleSheet('background:transparent;')
            cb_vl = QVBoxLayout(cb_container)
            cb_vl.setContentsMargins(16, 4, 8, 8)
            cb_vl.setSpacing(4)

            if is_cur:
                # Current bundle: one checkbox per trained algorithm
                trained = self._tab.trained_models
                if trained:
                    # Lay out in rows of 3
                    row_w = None; col_idx = 0
                    for algo in trained:
                        if col_idx % 3 == 0:
                            row_w = QWidget()
                            row_w.setStyleSheet('background:transparent;')
                            row_hl = QHBoxLayout(row_w)
                            row_hl.setContentsMargins(0, 0, 0, 0)
                            row_hl.setSpacing(20)
                            cb_vl.addWidget(row_w)

                        # R² label from combo
                        r2_str = ''
                        combo = self._tab._algo_combo
                        for i in range(combo.count()):
                            if combo.itemData(i) == algo:
                                txt = combo.itemText(i)
                                if '(' in txt:
                                    r2_str = '  ' + txt.split('(')[1].rstrip(')')
                                break

                        cb = QCheckBox(f'{algo}{r2_str}')
                        cb.setChecked(True)
                        cb.setStyleSheet(
                            'font-size:11px;'
                            'QCheckBox::indicator{width:13px;height:13px;}')
                        row_w.layout().addWidget(cb)
                        self._model_cbs[(norm, algo)] = cb
                        total_models += 1
                        col_idx += 1

                    # Fill last row with stretch
                    if row_w and col_idx % 3 != 0:
                        row_w.layout().addStretch()
                else:
                    no_algo = QLabel('(no algorithms in this bundle)')
                    no_algo.setStyleSheet(f'font-size:10px;color:{C_TEXT2};font-style:italic;')
                    cb_vl.addWidget(no_algo)
            else:
                # Extra bundle — try to read algo names from cached data
                # or attempt a lightweight load to get model names only
                algo_names = []
                if norm in self._tab._bundle_cache:
                    algo_names = list(self._tab._bundle_cache[norm]['models'].keys())
                else:
                    try:
                        from model_io import ModelIO as _MIO
                        _lr = _MIO.load(norm)
                        _results2 = _lr[0]
                        algo_names = list(_results2.keys())
                        # Cache the full bundle for later use
                        self._tab._bundle_cache[norm] = {
                            'label':     label,
                            'scaler':    _lr[1],
                            'feat_cols': _lr[2],
                            'ohe':       _lr[3],
                            'models':    {n: r['model'] for n, r in _results2.items()},
                        }
                    except Exception:
                        algo_names = []

                if algo_names:
                    # Show individual checkboxes for extra bundles too
                    row_w = None; col_idx = 0
                    for algo in algo_names:
                        if col_idx % 3 == 0:
                            row_w = QWidget()
                            row_w.setStyleSheet('background:transparent;')
                            row_hl = QHBoxLayout(row_w)
                            row_hl.setContentsMargins(0, 0, 0, 0)
                            row_hl.setSpacing(20)
                            cb_vl.addWidget(row_w)

                        cb = QCheckBox(algo)
                        cb.setChecked(False)   # extra bundles unchecked by default
                        cb.setStyleSheet(
                            f'font-size:11px;color:{C_TEXT2};'
                            'QCheckBox::indicator{width:13px;height:13px;}')
                        row_w.layout().addWidget(cb)
                        self._model_cbs[(norm, algo)] = cb
                        total_models += 1
                        col_idx += 1

                    if row_w and col_idx % 3 != 0:
                        row_w.layout().addStretch()
                else:
                    # Cannot read — fall back to single 'include all' checkbox
                    cb = QCheckBox('  Include all models from this bundle')
                    cb.setChecked(False)
                    cb.setStyleSheet(
                        f'font-size:11px;color:{C_TEXT2};'
                        'QCheckBox::indicator{width:13px;height:13px;}')
                    cb_vl.addWidget(cb)
                    self._bundle_cbs[norm] = cb
                    note = QLabel('     (will be loaded automatically when Run is clicked)')
                    note.setStyleSheet(
                        f'font-size:9px;color:{C_TEXT2};font-style:italic;')
                    cb_vl.addWidget(note)
                    total_models += 1

            self._cb_layout.addWidget(cb_container)

        # Trailing spacer
        spacer_w = QWidget()
        spacer_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cb_layout.addWidget(spacer_w)

        self._bundle_count_lbl.setText(
            f'{n_bundles} bundle(s) found  ·  {total_models} algorithm(s) total')

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════
    def _toggle_all(self):
        all_cbs = list(self._model_cbs.values()) + list(self._bundle_cbs.values())
        target  = not all(cb.isChecked() for cb in all_cbs)
        for cb in all_cbs:
            cb.setChecked(target)

    @staticmethod
    def _grp_style():
        return (
            'QGroupBox{font-weight:bold;font-size:11px;'
            'border:1px solid #d0d0d0;border-radius:4px;'
            'margin-top:8px;padding-top:4px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}')

    @staticmethod
    def _accent_btn_style():
        return (
            f'QPushButton{{background:{C_ACCENT};color:#fff;border:none;'
            f'border-radius:3px;font-size:11px;padding:0 10px;}}'
            f'QPushButton:hover{{background:{C_ACCENT_LT};}}')

    def _on_lock_toggle(self):
        if self._is_locked:
            self._unlock()
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, 'Select Input File', self._file_path or '',
                'Excel / CSV (*.xls *.xlsx *.csv)')
            if path:
                self._load_file(path)

    def _unlock(self):
        self._is_locked = False
        self._lock_icon.setText('[--]')
        self._path_lbl.setStyleSheet(
            f'font-size:11px;color:{C_TEXT2};font-style:italic;')
        self._toggle_lock_btn.setText('Browse …')
        self._toggle_lock_btn.setStyleSheet(self._accent_btn_style())
        self._run_btn.setEnabled(False)
        self._info_lbl.setText('Data unlocked — browse to select a replacement file.')
        self._preview_lbl.setText('')

    def _load_file(self, path: str):
        import os
        try:
            if path.endswith('.csv'):
                raw = pd.read_csv(path)
            elif path.endswith('.xls'):
                raw = pd.read_excel(path, engine='xlrd')
            else:
                raw = pd.read_excel(path)
        except Exception as ex:
            QMessageBox.critical(self, 'File Read Error', str(ex))
            return

        # Auto-detect first, then let the user verify / correct every
        # assignment before the file is locked.  Vexp is optional here
        # (metrics shown only when it is mapped).
        from column_mapping import _auto_map, ColumnMappingDialog
        auto_map = _auto_map(raw.columns.tolist(), df=raw)

        map_dlg = ColumnMappingDialog(raw.columns.tolist(), auto_map,
                                      parent=self)
        # Batch mode: Vexp is *optional* — override _validate so the
        # dialog can be accepted even without a Vexp assignment.
        map_dlg._validate = map_dlg.accept
        if map_dlg.exec_() != QDialog.Accepted:
            return   # user cancelled — do NOT lock the file

        confirmed_mapping  = map_dlg.get_mapping()
        self._col_mapping  = confirmed_mapping
        self._vexp_genuine = 'Vexp' in confirmed_mapping

        self._file_path  = path
        self._raw_df     = raw
        self._is_locked  = True

        fname = os.path.basename(path)
        self._path_lbl.setText(fname)
        self._path_lbl.setStyleSheet(
            f'font-size:11px;color:{C_TEXT};font-weight:bold;font-style:normal;')
        self._lock_icon.setText('[OK]')
        self._toggle_lock_btn.setText('Unlock & Replace')
        self._toggle_lock_btn.setStyleSheet(
            f'QPushButton{{background:#e8eef8;color:{C_TEXT};'
            f'border:1px solid {C_BORDER};border-radius:3px;'
            f'font-size:11px;padding:0 10px;}}'
            f'QPushButton:hover{{background:#d8e4f4;}}')

        nrow, ncol = raw.shape
        mapped_n  = len(confirmed_mapping)
        vexp_note = '  Vexp mapped' if self._vexp_genuine else '  Vexp not mapped'
        self._info_lbl.setText(
            f'{nrow:,} rows  ·  {ncol} columns  '
            f'·  {mapped_n} variables mapped{vexp_note}')

        mapped_vars = ', '.join(confirmed_mapping.values())
        self._preview_lbl.setText(f'Mapped columns: {mapped_vars}')
        self._run_btn.setEnabled(True)

    # Public accessors used by _batch()
    def get_file_path(self) -> str:
        return self._file_path

    def get_raw_df(self):
        return self._raw_df

    def get_vexp_genuine(self) -> bool:
        return self._vexp_genuine

    def get_mapping(self) -> dict:
        """Return the column mapping confirmed by the user."""
        return getattr(self, '_col_mapping', {})

    def get_model_selection(self) -> dict:
        """
        Returns {(norm_bundle_path, algo_name): bool}
        plus {norm_bundle_path: bool} for fallback-only bundles.
        """
        sel = {key: cb.isChecked() for key, cb in self._model_cbs.items()}
        sel.update({key: cb.isChecked() for key, cb in self._bundle_cbs.items()})
        return sel

