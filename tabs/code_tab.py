"""
code_tab.py — Design Code Comparison tab.

Compares international design-code formulas and trained ML models against
the experimental database using the following views:

  1. Metrics table    — R², r, RMSE, MAE, MAPE, k̄, CoV, P(Vp/Ve ≤ 1)
  2. Scatter plot     — Vpred vs Vexp (single method or all-codes grid popup)
  3. Ratio box plot   — Vpred/Vexp distribution per method
  4. Error CDF        — cumulative P(|ε| < x %) per method

All figures follow the project academic style (serif, inward ticks,
no top/right spines).  Export buttons produce publication-ready outputs.
"""
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavBar,
)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDialog, QSizePolicy,
    QFileDialog, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT,
    ALGO_COLORS, CODE_COLORS,
)
from widgets import flat_btn, panel, MplCanvas
from metrics import calc_metrics
from formulas import apply_code_formulas

_RC = {
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
    'figure.dpi':         120,
}

_METHOD_COLORS = {
    'GB 50608-2020': '#1f77b4',
    'ACI 440.1R-15': '#ff7f0e',
    'CSA S806-12':   '#2ca02c',
    'BISE (1999)':   '#9467bd',
    'JSCE (1997)':   '#8c564b',
}
_METHOD_COLORS.update(CODE_COLORS)

def _method_color(name):
    for k, v in _METHOD_COLORS.items():
        if k in name:
            return v
    return ALGO_COLORS.get(name.replace('ML: ', ''), C_ACCENT)

def _scatter_ax(ax, vexp, vpred, label, color, fontsize=9):
    mask = np.isfinite(vpred)
    ve, vp = vexp[mask], vpred[mask]
    if len(ve) < 2:
        return
    ax.scatter(ve, vp, alpha=0.55, s=12, color=color, linewidths=0)
    lim = [0, max(ve.max(), vp.max()) * 1.10]
    ax.plot(lim, lim,               color='#333333', ls='--', lw=1.0)
    ax.plot(lim, [v*1.2 for v in lim], ':', color='#70AD47', lw=0.8, alpha=0.7)
    ax.plot(lim, [v*0.8 for v in lim], ':', color='#70AD47', lw=0.8, alpha=0.7)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$V_\mathrm{exp}$ (kN)', fontsize=fontsize)
    ax.set_ylabel(r'$V_\mathrm{pred}$ (kN)', fontsize=fontsize)
    ax.set_title(label, fontsize=fontsize, fontweight='bold')
    m = calc_metrics(ve, vp)
    ax.text(0.04, 0.96,
            f'$R^2$ = {m["R2"]:.3f}\n'
            f'$r$   = {m["r"]:.3f}\n'
            f'RMSE = {m["RMSE"]:.1f} kN\n'
            f'$\\bar{{k}}$ = {m["mean_ratio"]:.3f}',
            transform=ax.transAxes, fontsize=fontsize-1, va='top',
            fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C_BORDER, alpha=0.88))

class _AllCodesDialog(QDialog):
    """Large pop-up grid: all methods in generous subplots."""

    def __init__(self, vexp, code_preds, ml_pairs, parent=None):
        super().__init__(parent,
                         Qt.Window |
                         Qt.WindowMinimizeButtonHint |
                         Qt.WindowMaximizeButtonHint |
                         Qt.WindowCloseButtonHint)
        self.setWindowTitle('All Methods: Predicted vs. Measured')
        self.resize(1200, 740)
        vl = QVBoxLayout(self)
        vl.setContentsMargins(8, 8, 8, 8)

        all_items = list(code_preds.items()) + ml_pairs
        n     = len(all_items)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig = mfig.Figure(figsize=(4.2*ncols, 3.6*nrows))
        with plt.rc_context(_RC):
            for i, (label, preds) in enumerate(all_items):
                ax = fig.add_subplot(nrows, ncols, i+1)
                ve = vexp if label in code_preds else np.asarray(preds[1])
                vp = np.asarray(preds) if label in code_preds \
                     else np.asarray(preds[0])
                _scatter_ax(ax, ve, vp, label,
                            _method_color(label), fontsize=8)
                ax.tick_params(labelsize=7)
            try:
                fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.8)
            except Exception:
                pass

        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(NavBar(canvas, self))
        vl.addWidget(canvas)

        btn_row = QHBoxLayout()
        save_btn = flat_btn('Export Figure  (PNG / PDF)')
        save_btn.setFixedHeight(32)
        save_btn.clicked.connect(lambda: self._save(fig))
        close_btn = flat_btn('Close')
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        vl.addLayout(btn_row)

    def _save(self, fig):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Figure', 'all_codes_scatter.pdf',
            'PDF (*.pdf);;PNG 300 dpi (*.png);;SVG (*.svg)')
        if path:
            fig.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, 'Saved', f'Figure saved to:\n{path}')

class CodeTab(QWidget):

    def __init__(self):
        super().__init__()
        self.df          = None
        self.ml_results  = {}
        self.y_te        = None
        self.code_preds  = {}
        self._rows_cache = []
        self._build_ui()

    def set_data(self, df):
        self.df = df
        if self.df is not None:
            self._calc()

    def set_ml(self, results, y_te):
        self.ml_results = results
        _yte = np.asarray(y_te) if y_te is not None else np.array([])
        self.y_te = _yte if len(_yte) > 0 else None
        # Auto-refresh the table whenever new ML results arrive so the user
        # does not have to manually click "Evaluate Design Codes" again.
        if self.df is not None and self.ml_results:
            try:
                self._calc()
            except Exception:
                pass

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Action bar
        top = QHBoxLayout()
        self._calc_btn = flat_btn('Evaluate Design Codes', accent=True)
        self._calc_btn.setFixedHeight(34)
        self._calc_btn.clicked.connect(self._calc)
        info = QLabel(
            'Design codes: full database.  ML models: held-out test set.')
        info.setStyleSheet(f'color:{C_TEXT2};font-size:11px;')
        top.addWidget(self._calc_btn)
        top.addSpacing(8)
        top.addWidget(info)
        top.addStretch()
        root.addLayout(top)

        # Body
        body = QHBoxLayout()
        body.setSpacing(8)

        # Left: table
        tbl_grp = panel('Design Code and ML Model Comparison')
        tl      = QVBoxLayout(tbl_grp)
        self.tbl = QTableWidget()
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setStyleSheet('font-size:12px;')
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.tbl.verticalHeader().setVisible(False)
        tl.addWidget(self.tbl)

        tbl_btns = QHBoxLayout()
        for lbl, fn in [('Export Metrics (CSV)',     self._export_metrics_csv),
                        ('Export Predictions (CSV)', self._export_pred_csv)]:
            b = flat_btn(lbl)
            b.setFixedHeight(28)
            b.clicked.connect(fn)
            tbl_btns.addWidget(b)
        tbl_btns.addStretch()
        tl.addLayout(tbl_btns)
        body.addWidget(tbl_grp, stretch=3)

        # Right: plot panel
        right = QWidget()
        right.setMinimumWidth(380)
        rl = QVBoxLayout(right)
        rl.setSpacing(5)
        rl.setContentsMargins(0, 0, 0, 0)

        ctrl_grp = panel('Visualisation')
        cl = QVBoxLayout(ctrl_grp)
        cl.addWidget(QLabel('Method (for scatter):'))
        self.method_cb = QComboBox()
        self.method_cb.setStyleSheet('font-size:11px;')
        cl.addWidget(self.method_cb)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        cl.addWidget(sep)

        for lbl, fn in [
            ('Scatter Plot: Selected Method',       self._plot_single),
            ('Scatter: All Methods (Overlay)', self._plot_all_popup),
            ('Vₚ/Vₑ ratio box plot',           self._plot_box),
            ('Error CDF',                      self._plot_cdf),
        ]:
            b = flat_btn(lbl)
            b.setFixedHeight(32)
            b.clicked.connect(fn)
            cl.addWidget(b)

        cl.addSpacing(4)
        exp_fig = flat_btn('Export Figure  (PNG / PDF)')
        exp_fig.setFixedHeight(28)
        exp_fig.clicked.connect(self._export_fig)
        cl.addWidget(exp_fig)
        rl.addWidget(ctrl_grp)

        self._canvas = MplCanvas(width=5, height=4.5)
        rl.addWidget(NavBar(self._canvas, right))
        rl.addWidget(self._canvas, stretch=1)
        body.addWidget(right, stretch=2)
        root.addLayout(body)
        self._draw_placeholder()

    def _prep_df(self):
        df = self.df.copy()
        if 'Vexp(kN)' not in df.columns and 'Vexp(KN)' in df.columns:
            df['Vexp(kN)'] = df['Vexp(KN)']
        return df

    def _vexp(self):
        return self._prep_df()['Vexp(kN)'].values.astype(float)

    def _calc(self):
        if self.df is None:
            QMessageBox.warning(self, 'No Data', 'Load a dataset first.')
            return
        try:
            df   = self._prep_df()
            self.code_preds = apply_code_formulas(df)
            vexp = df['Vexp(kN)'].values.astype(float)

            rows = [(label, 'Full dataset', calc_metrics(vexp, preds))
                    for label, preds in self.code_preds.items()]

            if self.ml_results:
                for n, res in self.ml_results.items():
                    te_m = res.get('te_metrics', {})
                    if te_m.get('R2') is not None:
                        ds = ('Test set' if (self.y_te is not None
                                             and len(self.y_te) > 0)
                              else 'Test set (bundle)')
                        rows.append((f'ML: {n}', ds, te_m))

            self._rows_cache = rows
            self._fill_table(rows)
            self.method_cb.clear()
            self.method_cb.addItems(list(self.code_preds.keys()))
            if self.ml_results:
                self.method_cb.addItems(
                    [f'ML: {n}' for n in self.ml_results])
            self._plot_single()
        except Exception as exc:
            QMessageBox.critical(self, 'Computation Error',
                                 f'{exc}\n\n{traceback.format_exc()}')

    def _fill_table(self, rows):
        hdrs = ['Method', 'Dataset', 'R²', 'r',
                'RMSE (kN)', 'MAE (kN)', 'MAPE (%)',
                'k̄', 'CoV', 'P(Vp/Ve≤1) (%)']
        self.tbl.setRowCount(len(rows))
        self.tbl.setColumnCount(len(hdrs))
        self.tbl.setHorizontalHeaderLabels(hdrs)
        bold = QFont(); bold.setBold(True)

        def fv(m, k):
            v = m.get(k)
            return f'{v:.4f}' if isinstance(v, float) and not np.isnan(v) else '—'

        for i, (nm, ds, m) in enumerate(rows):
            is_ml = nm.startswith('ML:')
            vals = [nm, ds, fv(m,'R2'), fv(m,'r'), fv(m,'RMSE'),
                    fv(m,'MAE'), fv(m,'MAPE'), fv(m,'mean_ratio'),
                    fv(m,'cov'), fv(m,'safety_pct')]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(
                    Qt.AlignLeft|Qt.AlignVCenter if j==0 else Qt.AlignCenter)
                if is_ml:
                    it.setBackground(QColor('#D9E8F5'))
                self.tbl.setItem(i, j, it)

        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(hdrs)):
            self.tbl.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)

    def _draw_placeholder(self):
        with plt.rc_context(_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.55, 'Click "Evaluate Design Codes" to begin.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, color='#AAAAAA', style='italic',
                fontfamily='serif')
        try: self._canvas.fig.tight_layout()
        except Exception: pass
        self._canvas.draw()

    def _plot_single(self):
        method = self.method_cb.currentText()
        if not method:
            return
        if method in self.code_preds:
            vexp, vpred = self._vexp(), self.code_preds[method]
        elif method.startswith('ML: '):
            n = method[4:]
            if n not in self.ml_results: return
            if self.y_te is None or len(self.y_te) == 0:
                QMessageBox.warning(self, 'No Labels',
                    'Test labels unavailable. Retrain or load v2.0+ bundle.')
                return
            te = np.asarray(self.ml_results[n].get('te_pred', []))
            if len(te) == 0: return
            vexp, vpred = self.y_te, te
        else:
            return

        with plt.rc_context(_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)
            _scatter_ax(ax, vexp, vpred, method, _method_color(method))
            try: self._canvas.fig.tight_layout()
            except Exception: pass
        self._canvas.draw()

    def _plot_all_popup(self):
        if not self.code_preds:
            QMessageBox.information(self, 'No Data',
                                    'Evaluate design codes first.')
            return
        vexp     = self._vexp()
        ml_pairs = []
        if self.ml_results and self.y_te is not None:
            for n, res in self.ml_results.items():
                te = np.asarray(res.get('te_pred', []))
                if len(te) > 0 and len(te) == len(self.y_te):
                    # Store as (vexp_for_ml, vpred) tuple for the dialog
                    ml_pairs.append((f'ML: {n}', (te, self.y_te)))

        dlg = _AllCodesDialog(vexp, self.code_preds, ml_pairs, parent=self)
        dlg.show()

    def _plot_box(self):
        if not self.code_preds:
            QMessageBox.information(self, 'No Data',
                                    'Evaluate design codes first.')
            return
        vexp = self._vexp()
        labels, data, colors = [], [], []
        for label, preds in self.code_preds.items():
            mask  = np.isfinite(preds) & np.isfinite(vexp) & (vexp > 0)
            ratio = preds[mask] / vexp[mask]
            labels.append(label)
            data.append(ratio)
            colors.append(_method_color(label))

        with plt.rc_context(_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)
            bp = ax.boxplot(data, patch_artist=True,
                            medianprops=dict(color='#333333', lw=1.5),
                            whiskerprops=dict(lw=0.8),
                            capprops=dict(lw=0.8),
                            flierprops=dict(marker='o', markersize=3,
                                            alpha=0.4, linestyle='none'))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c); patch.set_alpha(0.55)

            ax.axhline(1.0, color='#c0392b', lw=1.2, ls='--',
                       label=r'$V_\mathrm{pred}/V_\mathrm{exp}=1.0$')
            ax.axhline(1.2, color='#27ae60', lw=0.8, ls=':', alpha=0.7)
            ax.axhline(0.8, color='#27ae60', lw=0.8, ls=':',
                       alpha=0.7, label='±20%')

            ax.set_xticks(range(1, len(labels)+1))
            ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
            ax.set_ylabel(r'$V_\mathrm{pred}\,/\,V_\mathrm{exp}$')
            ax.set_title(
                r'Safety Factor Distribution — $V_\mathrm{pred}/V_\mathrm{exp}$',
                fontweight='bold')
            ax.legend(fontsize=8, frameon=False)
            ax.set_ylim(bottom=0)
            try: self._canvas.fig.tight_layout()
            except Exception: pass
        self._canvas.draw()

    def _plot_cdf(self):
        if not self.code_preds:
            QMessageBox.information(self, 'No Data',
                                    'Evaluate design codes first.')
            return
        vexp = self._vexp()
        with plt.rc_context(_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)
            for label, preds in self.code_preds.items():
                mask = (np.isfinite(preds) & np.isfinite(vexp) & (vexp > 0))
                err  = np.abs(preds[mask] - vexp[mask]) / vexp[mask] * 100
                xs   = np.sort(err)
                ys   = np.arange(1, len(xs)+1) / len(xs)
                ax.plot(xs, ys, lw=1.5, label=label,
                        color=_method_color(label))

            ax.axvline(20, color='#555555', lw=0.8, ls='--',
                       alpha=0.7, label='20% error')
            ax.set_xlabel(
                r'Absolute relative error $|\varepsilon|$ (%)')
            ax.set_ylabel(
                r'Cumulative probability $P\,(|\varepsilon|<x)$')
            ax.set_title('Error Cumulative Distribution Function',
                         fontweight='bold')
            ax.set_xlim(left=0); ax.set_ylim(0, 1)
            ax.legend(fontsize=8, frameon=False, loc='lower right')
            try: self._canvas.fig.tight_layout()
            except Exception: pass
        self._canvas.draw()

    def _export_metrics_csv(self):
        if not self._rows_cache:
            QMessageBox.information(self, 'No Data',
                                    'Evaluate design codes first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Metrics', 'code_comparison_metrics.csv',
            'CSV files (*.csv)')
        if not path: return
        cols = ['Method','Dataset','R2','r','RMSE_kN','MAE_kN',
                'MAPE_pct','k_bar','CoV','P_safe_pct']
        records = [{'Method': nm, 'Dataset': ds,
                    'R2': m.get('R2'), 'r': m.get('r'),
                    'RMSE_kN': m.get('RMSE'), 'MAE_kN': m.get('MAE'),
                    'MAPE_pct': m.get('MAPE'), 'k_bar': m.get('mean_ratio'),
                    'CoV': m.get('cov'), 'P_safe_pct': m.get('safety_pct')}
                   for nm, ds, m in self._rows_cache]
        pd.DataFrame(records, columns=cols).to_csv(path, index=False)
        QMessageBox.information(self, 'Done', f'Saved to:\n{path}')

    def _export_pred_csv(self):
        if not self.code_preds or self.df is None:
            QMessageBox.information(self, 'No Data',
                                    'Evaluate design codes first.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Predictions', 'code_predictions.csv',
            'CSV files (*.csv)')
        if not path: return
        vexp   = self._vexp()
        df_out = pd.DataFrame({'Vexp_kN': vexp})
        for label, preds in self.code_preds.items():
            safe = label.replace(' ', '_').replace('(','').replace(')','')
            df_out[f'Vpred_{safe}']  = preds
            df_out[f'ratio_{safe}']  = np.where(vexp > 0, preds/vexp, np.nan)
            df_out[f'error_pct_{safe}'] = np.where(
                vexp > 0, np.abs(preds-vexp)/vexp*100, np.nan)
        df_out.to_csv(path, index=False)
        QMessageBox.information(self, 'Done', f'Saved to:\n{path}')

    def _export_fig(self):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Figure', 'code_comparison.pdf',
            'PDF (*.pdf);;PNG 300 dpi (*.png);;SVG (*.svg)')
        if path:
            self._canvas.fig.savefig(path, dpi=300, bbox_inches='tight')
            QMessageBox.information(self, 'Saved', f'Saved to:\n{path}')
