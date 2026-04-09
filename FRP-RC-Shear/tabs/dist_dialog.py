"""
dist_dialog.py
==============
Variable Distribution Analysis dialog for the FRP-RC Shear platform.

Variables displayed are restricted to the seven prediction-relevant
numeric columns (PRED_COLS = ['Vexp(kN)', 'a/d', 'd(mm)', 'b(mm)',
'f`c(Mpa)', 'ρf(%)', 'Ef(GPa)']) — exactly matching the Prediction tab.

Classes
-------
_GridConfigDialog   Helper: user sets rows × cols for grid layout.
DistributionDialog  Main dialog: checkboxes, histogram + KDE,
                    single-plot or grid-plot mode.
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavBar,
)
import matplotlib.figure as mfig

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QWidget, QLabel, QCheckBox, QPushButton,
    QButtonGroup, QRadioButton, QSpinBox,
    QScrollArea, QFrame, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt

from config import C_ACCENT, C_TEXT2, C_BORDER, PRED_COLS, VAR_LATEX, VAR_PLAIN
from widgets import flat_btn, panel

# ── Matplotlib academic style ─────────────────────────────────────────
_ACADEMIC_RC = {
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'axes.linewidth':     0.8,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         110,
}

_PALETTE = ['#1f4e79', '#c0392b', '#1a7a4a',
            '#7d3c98', '#d35400', '#2e86c1',
            '#117a65', '#6e2f1a']


def _kde_curve(data, n_pts=300):
    """Return (x, y) for a KDE. Falls back to histogram density if
    scipy is not available."""
    vals = data.dropna().values.astype(float)
    x = np.linspace(vals.min(), vals.max(), n_pts)
    try:
        from scipy.stats import gaussian_kde
        return x, gaussian_kde(vals, bw_method='scott')(x)
    except ImportError:
        counts, edges = np.histogram(vals, bins=30, density=True)
        return (edges[:-1] + edges[1:]) / 2, counts


def _draw_hist_kde(ax, series, col_name, color='#1f4e79'):
    """Draw histogram + KDE on *ax* for one numeric variable."""
    data = series.dropna()
    if len(data) < 2:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='#AAAAAA', style='italic')
        return

    latex_label = VAR_LATEX.get(col_name, col_name)

    ax.hist(data, bins='auto', density=True,
            color=color, alpha=0.32,
            edgecolor='white', linewidth=0.4)

    xk, yk = _kde_curve(data)
    ax.plot(xk, yk, color=color, lw=1.6)

    ax.axvline(data.mean(),   color='#c0392b', lw=1.0, ls='--',
               label=f'Mean = {data.mean():.3g}')
    ax.axvline(data.median(), color='#27ae60', lw=1.0, ls=':',
               label=f'Median = {data.median():.3g}')

    ax.set_xlabel(latex_label)
    ax.set_ylabel('Probability density')
    ax.set_title(latex_label, fontweight='bold')
    ax.legend(fontsize=7, frameon=False)


# ══════════════════════════════════════════════════════════════════════
#  _GridConfigDialog
# ══════════════════════════════════════════════════════════════════════

class _GridConfigDialog(QDialog):
    """Tiny helper: ask the user for rows × cols before drawing a grid."""

    def __init__(self, n_vars, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Grid Layout')
        self.setFixedSize(280, 165)
        vl = QVBoxLayout(self)
        vl.setSpacing(10)

        cols_default = math.ceil(math.sqrt(n_vars))
        rows_default = math.ceil(n_vars / cols_default)

        info = QLabel(
            f'{n_vars} variable(s) selected.\n'
            'Set the grid dimensions:')
        info.setStyleSheet(f'font-size:11px;color:{C_TEXT2};')
        vl.addWidget(info)

        grid = QGridLayout()
        grid.addWidget(QLabel('Columns:'), 0, 0)
        self._cols_spin = QSpinBox()
        self._cols_spin.setRange(1, 8)
        self._cols_spin.setValue(cols_default)
        self._cols_spin.setFixedHeight(28)
        grid.addWidget(self._cols_spin, 0, 1)

        grid.addWidget(QLabel('Rows:'), 1, 0)
        self._rows_spin = QSpinBox()
        self._rows_spin.setRange(1, 8)
        self._rows_spin.setValue(rows_default)
        self._rows_spin.setFixedHeight(28)
        grid.addWidget(self._rows_spin, 1, 1)
        vl.addLayout(grid)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton('OK')
        ok_btn.setFixedHeight(30)
        ok_btn.setStyleSheet(
            f'background:{C_ACCENT};color:#fff;border-radius:3px;')
        ok_btn.clicked.connect(self.accept)
        ca_btn = QPushButton('Cancel')
        ca_btn.setFixedHeight(30)
        ca_btn.setStyleSheet(
            f'border:1px solid {C_BORDER};border-radius:3px;')
        ca_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(ca_btn)
        vl.addLayout(btn_row)

    @property
    def rows(self):
        return self._rows_spin.value()

    @property
    def cols(self):
        return self._cols_spin.value()


# ══════════════════════════════════════════════════════════════════════
#  DistributionDialog
# ══════════════════════════════════════════════════════════════════════

class DistributionDialog(QDialog):
    """
    Variable Distribution Analysis dialog.

    Shows histogram + KDE for the seven prediction-relevant numeric
    variables (PRED_COLS), with Single or Grid plot mode.
    """

    def __init__(self, df, parent=None):
        super().__init__(
            parent,
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint,
        )
        self.setWindowTitle('Distribution Analysis')
        self.resize(1080, 660)

        # Restrict to prediction-relevant numeric columns present in df
        self._df      = df
        self._cols    = [c for c in PRED_COLS if c in df.columns
                         and pd.api.types.is_numeric_dtype(df[c])]
        self._checks  = {}   # col_name → QCheckBox

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)
        root.addWidget(self._build_left())
        root.addWidget(self._build_canvas_panel())

    def _build_left(self):
        left = QWidget()
        left.setFixedWidth(220)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)
        ll.setContentsMargins(0, 0, 0, 0)

        # Variable checklist
        var_grp = panel('Variables')
        vg = QVBoxLayout(var_grp)

        row = QHBoxLayout()
        all_btn = flat_btn('Select All')
        all_btn.setFixedHeight(26)
        all_btn.clicked.connect(self._select_all)
        clr_btn = flat_btn('Clear')
        clr_btn.setFixedHeight(26)
        clr_btn.clicked.connect(self._clear_all)
        row.addWidget(all_btn)
        row.addWidget(clr_btn)
        vg.addLayout(row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner  = QWidget()
        il     = QVBoxLayout(inner)
        il.setSpacing(4)
        il.setContentsMargins(4, 4, 4, 4)

        for col in self._cols:
            plain_label = VAR_PLAIN.get(col, col)
            cb = QCheckBox(plain_label)
            cb.setStyleSheet('font-size:11px;')
            self._checks[col] = cb
            il.addWidget(cb)

        il.addStretch()
        scroll.setWidget(inner)
        vg.addWidget(scroll)
        ll.addWidget(var_grp, stretch=1)

        # Plot mode
        mode_grp = panel('Plot Mode')
        mg = QVBoxLayout(mode_grp)
        self._mode_grp  = QButtonGroup(self)
        self._rb_single = QRadioButton('Single plot')
        self._rb_grid   = QRadioButton('Grid plot')
        self._rb_single.setChecked(True)
        self._mode_grp.addButton(self._rb_single, 0)
        self._mode_grp.addButton(self._rb_grid,   1)
        mg.addWidget(self._rb_single)
        mg.addWidget(self._rb_grid)
        hint = QLabel('Single: one figure per click.\n'
                      'Grid: all selected in one figure.')
        hint.setStyleSheet(f'font-size:9px;color:{C_TEXT2};')
        mg.addWidget(hint)
        ll.addWidget(mode_grp)

        # Buttons
        self._plot_btn = flat_btn('Plot Selected', accent=True)
        self._plot_btn.setFixedHeight(36)
        self._plot_btn.clicked.connect(self._on_plot)
        ll.addWidget(self._plot_btn)

        close_btn = flat_btn('Close')
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.close)
        ll.addWidget(close_btn)

        return left

    def _build_canvas_panel(self):
        grp = panel('Histogram + KDE')
        cl  = QVBoxLayout(grp)
        self._fig    = mfig.Figure(figsize=(10, 6))
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        cl.addWidget(NavBar(self._canvas, self))
        cl.addWidget(self._canvas)
        self._draw_placeholder()
        return grp

    # ── Helpers ───────────────────────────────────────────────────────

    def _select_all(self):
        for cb in self._checks.values():
            cb.setChecked(True)

    def _clear_all(self):
        for cb in self._checks.values():
            cb.setChecked(False)

    def _selected_cols(self):
        return [c for c, cb in self._checks.items() if cb.isChecked()]

    def _draw_placeholder(self):
        with plt.rc_context(_ACADEMIC_RC):
            self._fig.clear()
            ax = self._fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.54,
                'Select variables on the left and click "Plot Selected".',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#AAAAAA', style='italic',
                fontfamily='serif')
        ax.text(0.5, 0.44,
                'Single: one figure per click.   Grid: all selections in one figure.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=9, color='#CCCCCC', style='italic',
                fontfamily='serif')
        try:
            self._fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    # ── Plotting ──────────────────────────────────────────────────────

    def _on_plot(self):
        cols = self._selected_cols()
        if not cols:
            QMessageBox.information(self, 'Nothing Selected',
                                    'Please tick at least one variable.')
            return
        if self._rb_single.isChecked():
            self._plot_single(cols)
        else:
            self._plot_grid(cols)

    def _plot_single(self, cols):
        n = len(cols)
        with plt.rc_context(_ACADEMIC_RC):
            self._fig.clear()
            axes = self._fig.subplots(n, 1, squeeze=False)
            for i, col in enumerate(cols):
                _draw_hist_kde(axes[i][0], self._df[col], col,
                               _PALETTE[i % len(_PALETTE)])
            try:
                self._fig.tight_layout(h_pad=1.4)
            except Exception:
                pass
        self._canvas.draw()

    def _plot_grid(self, cols):
        n   = len(cols)
        cfg = _GridConfigDialog(n, parent=self)
        if cfg.exec_() != QDialog.Accepted:
            return

        rows, ncols = cfg.rows, cfg.cols
        if rows * ncols < n:
            QMessageBox.warning(
                self, 'Grid Too Small',
                f'The grid ({rows}\u00d7{ncols}\u202f=\u202f{rows*ncols} cells) '
                f'is smaller than the number of selected variables ({n}).\n'
                'Increase rows or columns.')
            return

        with plt.rc_context(_ACADEMIC_RC):
            self._fig.clear()
            axes = self._fig.subplots(rows, ncols, squeeze=False)
            for idx, col in enumerate(cols):
                r, c = divmod(idx, ncols)
                _draw_hist_kde(axes[r][c], self._df[col], col,
                               _PALETTE[idx % len(_PALETTE)])
            for idx in range(n, rows * ncols):
                r, c = divmod(idx, ncols)
                axes[r][c].set_visible(False)
            try:
                self._fig.tight_layout(h_pad=1.4, w_pad=1.2)
            except Exception:
                pass
        self._canvas.draw()
