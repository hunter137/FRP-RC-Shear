"""
eval_dialogs.py — Plot dialogs for the Model Evaluation tab.

Contents
--------
_sample_pair             Draw a stratified random subsample from two parallel arrays
ScatterPlotDialog        Full-featured Predicted vs. Measured scatter popup
_ResponseSurfaceDialog   Feature-axis selector for 2D / 3D response surface plots
_PctDialog               Simple sample-percentage selector
"""
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QButtonGroup, QRadioButton,
    QFrame, QGroupBox, QSpinBox, QAbstractItemView,
    QPushButton, QSpacerItem,
    QListWidget, QListWidgetItem, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_ACCENT, C_ACCENT_BG, C_DANGER,
    C_WIN_BG, C_PANEL_BG,
    ALGO_COLORS,
)
from widgets import flat_btn, MplCanvas

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QButtonGroup, QRadioButton,
    QFrame, QGroupBox, QPushButton, QSizePolicy, QSpacerItem,
    QListWidget, QListWidgetItem, QSpinBox, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_ACCENT, C_ACCENT_BG, C_DANGER, C_SUCCESS,
    C_WIN_BG, C_PANEL_BG,
    ALGO_COLORS, FEAT_LABELS, HAS_SHAP,
)

def _sample_pair(a, b, pct):
    """Return a stratified random sample of two parallel arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if pct >= 100 or len(a) == 0:
        return a, b
    n   = max(1, int(len(a) * pct / 100))
    idx = np.random.default_rng(42).choice(len(a), n, replace=False)
    idx.sort()
    return a[idx], b[idx]

class ScatterPlotDialog(QDialog):
    """Full-featured Predicted vs. Measured scatter-plot popup.

    Behaviour
    ---------
    Training mode (data available for both splits):
        Dataset split radio buttons: Training Set / Test Set /
        Full Dataset (Training + Test overlaid).
        Sampling ratio: 25 / 50 / 75 / 100 %.

    Bundle mode (only test-set predictions stored):
        Split selector is replaced by an explanatory note.
        Sampling ratio is shown prominently as the primary control.

    CSV Export
        Saves V_exp (kN), V_pred (kN), split tag, absolute error,
        relative error (%), V_pred/V_exp ratio; appends a per-split
        statistics block.
    """

    _SPLITS_TRAIN = [
        ('both',  'Full Dataset  (Training + Test, overlaid)'),
        ('test',  'Test Set  (holdout evaluation)'),
        ('train', 'Training Set  (in-sample)'),
    ]
    _PCTS = [25, 50, 75, 100]

    def __init__(self, parent, name, res, y_tr, y_te,
                 loaded_model, algo_colors):
        super().__init__(parent)
        self._name     = name
        self._res      = res
        self._y_tr     = np.asarray(y_tr) if y_tr is not None else np.array([])
        self._y_te     = np.asarray(y_te) if y_te is not None else np.array([])
        self._loaded   = loaded_model
        self._colors   = algo_colors
        self._csv_rows  = []
        self._plot_split = 'test'   # split used in last replot (for export)
        self._plot_pct   = 100      # pct   used in last replot (for export)

        self.setWindowTitle(f'Scatter Plot — {name}')
        self.setMinimumSize(1020, 700)
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint)
        self._build_ui()
        self._replot()

    def _draw_placeholder(self):
        """Render a blank canvas with a descriptive hint."""
        self._current_plot_data = {}
        if self._save_plot_data_btn:
            self._save_plot_data_btn.setEnabled(False)
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.55,
                'Select a diagnostic plot from the panel on the left.',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=12, color='#AAAAAA',
                style='italic')
        ax.text(0.5, 0.44,
                'Plots are generated on demand and do not auto-refresh.',
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=10, color='#CCCCCC',
                style='italic')
        try:
            self.canvas.fig.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left control panel
        ctrl = QWidget()
        ctrl.setFixedWidth(258)
        ctrl.setStyleSheet(
            f'background:{C_WIN_BG};'
            f'border-right:1px solid {C_BORDER};')
        cv = QVBoxLayout(ctrl)
        cv.setContentsMargins(14, 18, 14, 14)
        cv.setSpacing(14)

        hdr = QLabel('Plot Settings')
        hdr.setStyleSheet(
            f'font-size:13px;font-weight:bold;color:{C_TEXT};'
            f'padding-bottom:8px;'
            f'border-bottom:2px solid {C_BORDER_LT};')
        cv.addWidget(hdr)

        grp_style = (
            f'QGroupBox{{'
            f'  font-size:11px;font-weight:bold;color:{C_TEXT2};'
            f'  border:1px solid {C_BORDER_LT};border-radius:3px;'
            f'  margin-top:8px;padding-top:10px;}}'
            f'QGroupBox::title{{'
            f'  subcontrol-origin:margin;left:8px;'
            f'  padding:0 4px;background:{C_WIN_BG};}}')

        # Dataset split
        split_box = QGroupBox('Dataset Split')
        split_box.setStyleSheet(grp_style)
        sb = QVBoxLayout(split_box)
        sb.setSpacing(5)
        sb.setContentsMargins(8, 8, 8, 8)
        self._split_grp = QButtonGroup(self)

        if self._loaded:
            note = QLabel(
                '\u24d8  Bundle mode\n\n'
                'Only test-set predictions\n'
                'are serialised. Training-\n'
                'split plots require full\n'
                'retraining.')
            note.setWordWrap(False)
            note.setStyleSheet(
                'font-size:10px;color:#555;'
                'background:#FFF8E1;'
                'padding:6px 8px;'
                'border:1px solid #FFE082;'
                'border-radius:3px;')
            sb.addWidget(note)
            locked = QLabel('\u2713  Test Set  (locked)')
            locked.setStyleSheet(
                f'font-size:11px;font-weight:bold;'
                f'color:{C_ACCENT};margin-top:4px;')
            sb.addWidget(locked)
        else:
            for key, label in self._SPLITS_TRAIN:
                rb = QRadioButton(label)
                rb.setProperty('key', key)
                rb.setStyleSheet('font-size:11px;padding:2px 0;')
                rb.setChecked(key == 'both')
                self._split_grp.addButton(rb)
                sb.addWidget(rb)
        cv.addWidget(split_box)

        # Sampling ratio
        pct_box = QGroupBox('Display Density  (%  of data rendered)')
        pct_box.setStyleSheet(grp_style)
        pb = QVBoxLayout(pct_box)
        pb.setSpacing(5)
        pb.setContentsMargins(8, 8, 8, 8)
        note2 = QLabel(
            'Reduce to mitigate overplotting\n'
            'on large datasets. Sampling is\n'
            'reproducible (fixed seed = 42).')
        note2.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        pb.addWidget(note2)
        self._pct_grp = QButtonGroup(self)
        pct_row = QHBoxLayout()
        pct_row.setSpacing(4)
        for pct in self._PCTS:
            rb = QRadioButton(f'{pct}')
            rb.setProperty('pct', pct)
            rb.setStyleSheet('font-size:12px;padding:2px 3px;')
            rb.setChecked(pct == 100)
            self._pct_grp.addButton(rb)
            pct_row.addWidget(rb)
        pct_row.addStretch()
        pb.addLayout(pct_row)
        cv.addWidget(pct_box)

        cv.addSpacerItem(QSpacerItem(0, 4))

        replot_btn = QPushButton('Apply')
        replot_btn.setFixedHeight(34)
        replot_btn.setStyleSheet(
            f'background:{C_ACCENT};color:#fff;'
            f'border:none;font-size:12px;font-weight:bold;'
            f'border-radius:3px;')
        replot_btn.clicked.connect(self._replot)
        cv.addWidget(replot_btn)

        self._export_btn = QPushButton('Export Scatter (CSV)')
        self._export_btn.setFixedHeight(34)
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet(
            'background:#4A4A8A;color:#fff;'
            'border:none;font-size:12px;border-radius:3px;')
        self._export_btn.clicked.connect(self._export_csv)
        cv.addWidget(self._export_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        cv.addWidget(sep)

        close_btn = QPushButton('Close')
        close_btn.setFixedHeight(30)
        close_btn.setStyleSheet(
            f'background:#fff;color:{C_TEXT};'
            f'border:1px solid {C_BORDER};font-size:12px;'
            f'border-radius:3px;')
        close_btn.clicked.connect(self.close)
        cv.addWidget(close_btn)
        cv.addStretch()
        root.addWidget(ctrl)

        # Canvas
        cw = QWidget()
        cw.setStyleSheet(f'background:{C_PANEL_BG};')
        cl = QVBoxLayout(cw)
        cl.setContentsMargins(6, 6, 6, 6)
        cl.setSpacing(2)
        self._fig    = MplFigure(figsize=(9, 6.5), dpi=100)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._toolbar = NavigationToolbar(self._canvas, self)
        cl.addWidget(self._toolbar)
        cl.addWidget(self._canvas, 1)
        root.addWidget(cw, 1)

    def _sel_split(self):
        if self._loaded:
            return 'test'
        b = self._split_grp.checkedButton()
        return b.property('key') if b else 'both'

    def _sel_pct(self):
        b = self._pct_grp.checkedButton()
        return b.property('pct') if b else 100

    def _replot(self):
        split   = self._sel_split()
        pct     = self._sel_pct()
        res     = self._res
        name    = self._name
        color   = self._colors.get(name, C_ACCENT)
        y_te    = self._y_te
        y_tr    = self._y_tr
        te_pred = np.asarray(res.get('te_pred', []))
        tr_pred = np.asarray(res.get('tr_pred', []))

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        all_true, all_pred = [], []
        self._csv_rows = []

        if split in ('both', 'train') and len(tr_pred) > 0 and len(y_tr) > 0:
            yt_s, yp_s = _sample_pair(y_tr, tr_pred, pct)
            ax.scatter(yt_s, yp_s,
                       s=14, alpha=0.28, color='#9DC3E6',
                       linewidths=0.3, edgecolors='#6FA8D8',
                       label=f'Training set  ($n$ = {len(yt_s)})',
                       zorder=2)
            all_true.append(yt_s); all_pred.append(yp_s)
            self._csv_rows += [
                {'model': name, 'split': 'train',
                 'V_exp_kN': float(a), 'V_pred_kN': float(b)}
                for a, b in zip(yt_s, yp_s)]

        if split in ('both', 'test') and len(te_pred) > 0 and len(y_te) > 0:
            yte_s, yp_s = _sample_pair(y_te, te_pred, pct)
            ax.scatter(yte_s, yp_s,
                       s=22, alpha=0.78, color=color,
                       linewidths=0.4, edgecolors='#1F1F1F',
                       label=f'Test set  ($n$ = {len(yte_s)})',
                       zorder=3)
            all_true.append(yte_s); all_pred.append(yp_s)
            self._csv_rows += [
                {'model': name, 'split': 'test',
                 'V_exp_kN': float(a), 'V_pred_kN': float(b)}
                for a, b in zip(yte_s, yp_s)]

        self._export_btn.setEnabled(bool(self._csv_rows))
        # Record the settings used for this plot — used by export
        self._plot_split = split
        self._plot_pct   = pct

        if not all_true:
            ax.text(0.5, 0.5,
                    'No prediction data available for the selected split.\n'
                    'Please retrain the model or select a different split.',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=11, color='#888', multialignment='center')
            self._canvas.draw()
            return

        cat_t = np.concatenate(all_true)
        cat_p = np.concatenate(all_pred)
        vmax  = max(float(cat_t.max()), float(cat_p.max())) * 1.12

        ax.plot([0, vmax], [0, vmax],
                color='#1F1F1F', ls='--', lw=1.5,
                label='Perfect agreement  (1:1)', zorder=1)
        ax.plot([0, vmax], [0, vmax * 1.20],
                ':', color='#70AD47', lw=1.1, alpha=0.80, zorder=1)
        ax.plot([0, vmax], [0, vmax * 0.80],
                ':', color='#70AD47', lw=1.1, alpha=0.80,
                label='\u00b120\u202f% bounds', zorder=1)

        ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(
            r'Measured shear capacity  $V_\mathrm{exp}$  (kN)',
            fontsize=12)
        ax.set_ylabel(
            r'Predicted shear capacity  $V_\mathrm{pred}$  (kN)',
            fontsize=12)

        split_lbl = {
            'train': 'Training Set  (in-sample)',
            'test':  'Test Set  (holdout evaluation)',
            'both':  'Full Dataset  (Training + Test)',
        }[split]
        pct_note = f'  \u2014  {pct}\u202f% sample' if pct < 100 else ''
        ax.set_title(
            f'{name}  \u2014  Scatter Diagram  [{split_lbl}]{pct_note}',
            fontsize=12, fontweight='bold', pad=10)

        # For training split: show tr_metrics.
        # For test / bundle mode: show te_metrics.
        # For both: show test metrics with a note.
        tr_m = res.get('tr_metrics', {})
        te_m = res.get('te_metrics', {})

        if split == 'train' and tr_m:
            ann_m    = tr_m
            ann_tag  = 'train'
            ann_note = '(in-sample / training set)'
        elif split == 'both' and te_m:
            ann_m    = te_m
            ann_tag  = 'test'
            ann_note = '(test-set metrics shown)'
        else:
            ann_m    = te_m
            ann_tag  = 'test'
            ann_note = '(holdout / test set)'

        if ann_m:
            ax.text(0.04, 0.97,
                    f'$R^2$  ({ann_tag}) = {ann_m.get("R2",  float("nan")):.4f}\n'
                    f'$r$    ({ann_tag}) = {ann_m.get("r",   float("nan")):.4f}\n'
                    f'RMSE ({ann_tag}) = {ann_m.get("RMSE", float("nan")):.2f} kN\n'
                    f'MAE  ({ann_tag}) = {ann_m.get("MAE",  float("nan")):.2f} kN\n'
                    f'MAPE ({ann_tag}) = {ann_m.get("MAPE", float("nan")):.2f} %\n'
                    f'\u2014 {ann_note}',
                    transform=ax.transAxes,
                    fontsize=9.5, va='top', linespacing=1.6,
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.45',
                              facecolor='white',
                              edgecolor=C_BORDER, alpha=0.93))

        ax.legend(fontsize=10, framealpha=0.92,
                  loc='lower right', edgecolor=C_BORDER)
        ax.tick_params(labelsize=9.5)
        ax.grid(True, alpha=0.18, linewidth=0.6)
        try:
            self._fig.tight_layout(pad=1.4)
        except Exception:
            pass
        self._canvas.draw()

    def _export_csv(self):
        rows = self._csv_rows
        if not rows:
            QMessageBox.warning(self, 'No Data',
                'No plot data available. Click \u201cReplot\u201d first.')
            return
        # Use settings from the LAST replot — not current radio selection
        split   = self._plot_split
        pct     = self._plot_pct
        pct_tag = f'_sample{pct}pct' if pct < 100 else ''
        split_labels = {
            'both':  'full_dataset',
            'test':  'test_set',
            'train': 'training_set',
        }
        default = f'{self._name}_scatter_{split_labels.get(split, split)}{pct_tag}.csv'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Scatter Data  (≡ currently displayed chart)',
            default, 'CSV files (*.csv)')
        if not path:
            return

        df = pd.DataFrame(rows)[
            ['model', 'split', 'V_exp_kN', 'V_pred_kN']]
        df['abs_error_kN']     = df['V_pred_kN'] - df['V_exp_kN']
        df['rel_error_pct']    = (df['abs_error_kN'] / df['V_exp_kN']) * 100
        df['ratio_Vpred_Vexp'] = df['V_pred_kN'] / df['V_exp_kN']
        df.to_csv(path, index=False)

        # Append per-split statistics
        split_stats = []
        for sp in df['split'].unique():
            sub = df[df['split'] == sp]
            r2 = float(np.corrcoef(
                sub['V_exp_kN'], sub['V_pred_kN'])[0, 1] ** 2)
            split_stats.append({
                'model':        f'{self._name}  ({sp})',
                'n':            len(sub),
                'R2':           round(r2, 6),
                'RMSE_kN':      round(float(np.sqrt(
                                    np.mean(sub['abs_error_kN']**2))), 4),
                'MAE_kN':       round(float(sub['abs_error_kN'].abs().mean()), 4),
                'MAPE_pct':     round(float(sub['rel_error_pct'].abs().mean()), 4),
                'mean_ratio':   round(float(sub['ratio_Vpred_Vexp'].mean()), 6),
                'CoV_ratio':    round(float(sub['ratio_Vpred_Vexp'].std() /
                                            sub['ratio_Vpred_Vexp'].mean()), 6),
            })
        with open(path, 'a', newline='') as f:
            f.write('\n# Per-split summary statistics\n')
            pd.DataFrame(split_stats).to_csv(f, index=False)

        split_note = {
            'both':  'Full Dataset (Training + Test)',
            'test':  'Test Set',
            'train': 'Training Set',
        }.get(split, split)
        pct_info = (f'{pct}\u202f% random sample'
                    if pct < 100 else '100% (all points)')
        QMessageBox.information(
            self, 'Save Successful',
            f'Data saved matches the chart currently displayed.\n\n'
            f'  Split  : {split_note}\n'
            f'  Sample : {pct_info}  (seed = 42)\n\n'
            f'File: {path}\n\n'
            f'Data columns:\n'
            f'  model, split, V_exp_kN, V_pred_kN,\n'
            f'  abs_error_kN, rel_error_pct, ratio_Vpred_Vexp\n\n'
            f'Total rows: {len(df)}\n'
            f'A per-split summary statistics block is appended.')

class _ResponseSurfaceDialog(QDialog):
    """Variable selection for response surface analysis.

    1 feature  → 2-D line plot.
    2 features → 3-D surface or 2-D filled contour map.
    All other features are fixed at their per-column median.
    """

    def __init__(self, feat_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Response Surface: Variable Selection')
        self.setFixedWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        outer = QVBoxLayout(self)
        outer.setSpacing(12)
        outer.setContentsMargins(18, 16, 18, 14)

        info = QLabel(
            'Select 1 or 2 input features.\n'
            'All other features are held fixed at their median value.\n'
            'Hold Ctrl to select multiple items.')
        info.setWordWrap(True)
        info.setStyleSheet('font-size:11px;color:#555;')
        outer.addWidget(info)

        self._lst = QListWidget()
        self._lst.setSelectionMode(QAbstractItemView.MultiSelection)
        self._lst.setFixedHeight(200)
        # Separate numeric and OHE features for clarity
        numeric = [n for n in feat_names if not n.startswith('FRP=')]
        ohe     = [n for n in feat_names if n.startswith('FRP=')]
        from PyQt5.QtGui import QColor, QBrush, QFont
        for nm in numeric:
            it = QListWidgetItem(nm)
            self._lst.addItem(it)
        if ohe:
            sep = QListWidgetItem('── One-Hot features (binary, less informative) ──')
            sep.setFlags(Qt.NoItemFlags)  # not selectable
            sep.setForeground(QBrush(QColor('#AAAAAA')))
            _f = QFont(); _f.setItalic(True); sep.setFont(_f)
            self._lst.addItem(sep)
            for nm in ohe:
                it = QListWidgetItem(nm)
                it.setForeground(QBrush(QColor('#999999')))
                self._lst.addItem(it)
        outer.addWidget(self._lst)
        # Hint about OHE features
        if ohe:
            hint = QLabel(
                '⚠  One-Hot features (FRP=...) are binary (0/1). '
                'Tree models produce flat surfaces for these; '
                'select numeric features for meaningful results.')
            hint.setWordWrap(True)
            hint.setStyleSheet('font-size:10px;color:#B07800;')
            outer.addWidget(hint)

        type_box = QGroupBox('Plot Type  (for 2-feature selection)')
        type_box.setStyleSheet(
            'QGroupBox{font-size:11px;border:1px solid #DDD;'
            'border-radius:3px;margin-top:6px;padding-top:8px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:8px;}')
        tb = QHBoxLayout(type_box)
        self._rb_3d = QRadioButton('3D Surface')
        self._rb_2d = QRadioButton('2D Contour Map')
        self._rb_3d.setChecked(True)
        tb.addWidget(self._rb_3d)
        tb.addWidget(self._rb_2d)
        outer.addWidget(type_box)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel('Grid resolution:'))
        self._res_spin = QSpinBox()
        self._res_spin.setRange(20, 100)
        self._res_spin.setValue(40)
        self._res_spin.setSuffix(' pts / axis')
        res_row.addWidget(self._res_spin)
        res_row.addStretch()
        outer.addLayout(res_row)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('Generate Plot')
        bb.button(QDialogButtonBox.Ok).setStyleSheet(
            'background:#2B6CB0;color:#fff;border:none;'
            'padding:5px 18px;border-radius:2px;')
        bb.button(QDialogButtonBox.Cancel).setStyleSheet(
            'background:#fff;color:#111;border:1px solid #CCC;'
            'padding:5px 18px;border-radius:2px;')
        bb.accepted.connect(self._validate)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def _validate(self):
        sel = self._lst.selectedItems()
        if not (1 <= len(sel) <= 2):
            QMessageBox.warning(self, 'Selection Error',
                'Please select exactly 1 or 2 features.')
            return
        self.accept()

    def selected_features(self):
        return [item.text() for item in self._lst.selectedItems()]

    def plot_type(self):
        return '3d' if self._rb_3d.isChecked() else '2d'

    def resolution(self):
        return self._res_spin.value()

class _PctDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(340)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        outer = QVBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(18, 16, 18, 14)
        lbl = QLabel('Proportion of observations to include:')
        lbl.setStyleSheet(
            f'font-size:13px;font-weight:bold;color:{C_TEXT};')
        outer.addWidget(lbl)
        self._pct_group = QButtonGroup(self)
        row = QHBoxLayout()
        for pct in (25, 50, 75, 100):
            rb = QRadioButton(f'{pct}\u202f%')
            rb.setProperty('pct', pct)
            rb.setStyleSheet('font-size:12px;padding:2px 6px;')
            rb.setChecked(pct == 100)
            self._pct_group.addButton(rb)
            row.addWidget(rb)
        outer.addLayout(row)
        hint = QLabel(
            'Reducing the proportion mitigates overplotting\n'
            'while preserving the distributional shape.\n'
            'Random seed is fixed (42) for reproducibility.')
        hint.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        outer.addWidget(hint)
        bb = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText('Generate Plot')
        bb.button(QDialogButtonBox.Ok).setStyleSheet(
            f'background:{C_ACCENT};color:#fff;'
            f'border:1px solid {C_ACCENT};padding:5px 18px;border-radius:2px;')
        bb.button(QDialogButtonBox.Cancel).setStyleSheet(
            f'background:#fff;color:{C_TEXT};'
            f'border:1px solid {C_BORDER};padding:5px 18px;border-radius:2px;')
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    def selected_pct(self):
        b = self._pct_group.checkedButton()
        return b.property('pct') if b else 100

from .shap_dialog import ShapBeeswarmDialog   # noqa: F401

