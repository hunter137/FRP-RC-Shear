"""
eval_tab.py — EvalTab: model evaluation metrics and diagnostic plots.

Responsibilities
----------------
- Display a ranked metrics table for all loaded models
- Orchestrate diagnostic plots (scatter, importance, error, safety factor,
  response surface, SHAP beeswarm) via the left-panel buttons
- Expose CSV export of plot data and full metrics

Dialog classes have been extracted into eval_dialogs.py to keep this
file focused on the main EvalTab widget.
"""
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import matplotlib.figure as mfig
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3d projection

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox,
    QDialog, QFrame, QGroupBox,
    QSizePolicy, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_ACCENT, C_ACCENT_BG, C_DANGER, C_SUCCESS,
    C_WIN_BG, C_PANEL_BG,
    ALGO_COLORS, FEAT_LABELS, HAS_SHAP,
)
from widgets import flat_btn, panel, MplCanvas
from metrics import calc_metrics

from .eval_dialogs import (
    _sample_pair, ScatterPlotDialog,
    _ResponseSurfaceDialog, _PctDialog,
)

try:
    from .shap_dialog import ShapBeeswarmDialog
except ImportError:
    ShapBeeswarmDialog = None

class EvalTab(QWidget):

    def __init__(self):
        super().__init__()
        self.results            = {}
        self.X_te = self.y_te  = None
        self.X_tr = self.y_tr  = None
        self.X_all             = None  # fallback for bundle-load mode
        self.feat_names         = FEAT_LABELS
        self._loaded_model_mode  = False
        self._data_plot_btns     = []
        self._current_plot_data  = {}   # {type, name, df} for export
        self._save_plot_data_btn = None
        self._shap_dlg           = None   # lazy-created ShapBeeswarmDialog
        self._build_ui()
        self._draw_placeholder()

    def load(self, results, X_te, y_te, X_tr, y_tr, X_all=None):
        self.results = results
        self.X_te, self.y_te = X_te, y_te
        self.X_tr, self.y_tr = X_tr, y_tr
        # X_all: shap_cache from bundle, or full training set — used
        # as fallback when X_tr is empty (bundle-load mode).
        if X_all is not None and len(np.asarray(X_all)) > 0:
            self.X_all = np.asarray(X_all)

        # dict.get(key, default) only falls back when the key is MISSING,
        # not when the value is an empty array.  Injecting here guarantees
        # that _y_te / _y_tr are always non-empty when the caller supplies
        # real labels (training mode) and avoids the scatter-plot failure.
        _yte = np.asarray(y_te) if y_te is not None else np.array([])
        _ytr = np.asarray(y_tr) if y_tr is not None else np.array([])
        for _r in self.results.values():
            if len(_yte) > 0:
                _r['_y_te'] = _yte
            if len(_ytr) > 0:
                _r['_y_tr'] = _ytr

        has_te_labels = len(_yte) > 0
        has_te_pred = has_tr_pred = False
        if results:
            first = next(iter(results.values()))
            has_te_pred = len(np.asarray(first.get('te_pred', []))) > 0
            has_tr_pred = len(np.asarray(first.get('tr_pred', []))) > 0

        self._loaded_model_mode = not has_tr_pred
        has_data = has_te_labels and has_te_pred
        self._set_data_buttons_enabled(has_data)
        self._refresh_table()
        self.model_cb.clear()
        self.model_cb.addItems(list(results.keys()))

        # Do NOT auto-plot — canvas starts blank.
        # User selects a plot type from the buttons on the left.
        self._draw_placeholder()

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
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        tbl_grp = panel('Model Evaluation Metrics')
        tl      = QVBoxLayout(tbl_grp)
        self.tbl = QTableWidget()
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setStyleSheet('font-size:13px;')
        _hdr = self.tbl.horizontalHeader()
        _hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        _hdr.setStretchLastSection(True)   # fill remaining width
        self.tbl.setMaximumHeight(165)
        tl.addWidget(self.tbl)
        root.addWidget(tbl_grp)

        bottom   = QHBoxLayout()
        ctrl_grp = panel('Diagnostic Analysis')
        cl       = QVBoxLayout(ctrl_grp)
        cl.setSpacing(5)
        ctrl_grp.setFixedWidth(280)

        lm = QLabel('Selected Algorithm:')
        lm.setStyleSheet(f'font-size:11px;color:{C_TEXT2};')
        cl.addWidget(lm)
        self.model_cb = QComboBox()
        self.model_cb.setStyleSheet('font-size:12px;')
        cl.addWidget(self.model_cb)
        cl.addSpacing(8)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        cl.addWidget(sep)

        lp = QLabel('Diagnostic Plots:')
        lp.setStyleSheet(
            f'font-size:11px;font-weight:bold;color:{C_TEXT2};'
            f'margin-top:4px;')
        cl.addWidget(lp)

        btn_spec = [
            # Academic naming: concise, standard ML/engineering terminology
            ('Scatter Plot',                                   self._plot_scatter),
            ('Feature Importance',                             self._plot_importance),
            ('Error Histogram',                                self._plot_error),
            ('Safety Factor Plot',                             self._plot_ratio),
            ('Response Surface Analysis',                      self._plot_response_surface),
            ('SHAP Beeswarm Analysis',                         self._open_shap_dialog),
        ]
        self._data_plot_btns = []
        for lbl, fn in btn_spec:
            b = flat_btn(lbl)
            b.setMinimumHeight(40)
            b.clicked.connect(fn)
            cl.addWidget(b)
            self._data_plot_btns.append(b)

        cl.addSpacing(8)
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f'color:{C_BORDER_LT};')
        cl.addWidget(sep2)

        # Export current plot raw data
        lp2 = QLabel('Data Export:')
        lp2.setStyleSheet(
            f'font-size:11px;font-weight:bold;color:{C_TEXT2};margin-top:2px;')
        cl.addWidget(lp2)

        self._save_plot_data_btn = flat_btn('Save Plot Data  (CSV)')
        self._save_plot_data_btn.setMinimumHeight(40)
        self._save_plot_data_btn.setEnabled(False)
        self._save_plot_data_btn.setToolTip(
            'Exports the raw data underlying the currently displayed plot,\n'
            'enabling independent figure reproduction in any plotting tool.')
        self._save_plot_data_btn.clicked.connect(self._export_current_plot_data)
        cl.addWidget(self._save_plot_data_btn)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setStyleSheet(f'color:{C_BORDER_LT};')
        cl.addWidget(sep3)

        ex = flat_btn('Export (CSV)')
        ex.setMinimumHeight(40)
        ex.clicked.connect(self._export)
        cl.addWidget(ex)
        cl.addStretch()
        bottom.addWidget(ctrl_grp)

        canvas_grp = panel('Plot Preview')
        cv_l       = QVBoxLayout(canvas_grp)
        self.canvas = MplCanvas(width=9, height=5)
        toolbar     = NavigationToolbar(self.canvas, self)
        cv_l.addWidget(toolbar)
        cv_l.addWidget(self.canvas)
        bottom.addWidget(canvas_grp)
        root.addLayout(bottom)

    def _set_data_buttons_enabled(self, enabled):
        for btn in self._data_plot_btns:
            btn.setEnabled(enabled)
        if self._data_plot_btns and self.results:
            self._data_plot_btns[0].setEnabled(True)   # Scatter plot: always on when models loaded
            self._data_plot_btns[1].setEnabled(True)   # Variable Importance: always on

    def _refresh_table(self):
        hdrs = [
            'Model',
            'R\u00b2 (test)',  'r (test)',
            'RMSE (kN)',       'MAE (kN)',
            'MAPE (%)',        '\u03ba\u0305',
            'CoV',             'P(Vp/Ve\u22641) (%)',
            'R\u00b2 (train)',
            'CV R\u00b2  mean \u00b1 S.D.',
        ]
        items = sorted(
            self.results.items(),
            key=lambda x: x[1]['te_metrics'].get('R2', -1),
            reverse=True)
        self.tbl.setRowCount(len(items))
        self.tbl.setColumnCount(len(hdrs))
        self.tbl.setHorizontalHeaderLabels(hdrs)
        for i, (name, res) in enumerate(items):
            te = res['te_metrics']
            tr = res['tr_metrics']
            vals = [
                name,
                f'{te["R2"]:.4f}',     f'{te["r"]:.4f}',
                f'{te["RMSE"]:.3f}',   f'{te["MAE"]:.3f}',
                f'{te["MAPE"]:.2f}',   f'{te["mean_ratio"]:.4f}',
                f'{te["cov"]:.4f}',    f'{te["safety_pct"]:.1f}',
                f'{tr["R2"]:.4f}',
                f'{res["cv_mean"]:.4f} \u00b1 {res["cv_std"]:.4f}',
            ]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(Qt.AlignCenter)
                if j == 1 and i == 0:
                    it.setBackground(QColor('#D5ECD4'))
                self.tbl.setItem(i, j, it)

    def _check_ready(self, need_data=True):
        name = self.model_cb.currentText()
        if not name:
            raise RuntimeError(
                'No model is selected.\n'
                'Please choose a model from the drop-down list.')
        if name not in self.results:
            raise RuntimeError(
                f'No results found for \u201c{name}\u201d.\n'
                'Complete training or load a model bundle first.')
        res = self.results[name]
        if need_data:
            te_pred = np.asarray(res.get('te_pred', []))
            _y_te   = res.get('_y_te', self.y_te)
            y_te    = np.asarray(_y_te) if _y_te is not None else np.array([])
            if len(te_pred) == 0 or len(y_te) == 0:
                raise RuntimeError(
                    'Prediction data are unavailable.\n\n'
                    'Possible causes: the bundle predates v2.0 (no raw\n'
                    'predictions stored), or the model has not been\n'
                    'trained. Please retrain to enable this visualisation.')
        return name, res

    def _open_shap_dialog(self):
        """Open the SHAP Beeswarm Analysis pop-up for the selected model."""
        name = self.model_cb.currentText()
        if not name or name not in self.results:
            QMessageBox.warning(self, 'No Model',
                                'Train or load a model first.')
            return
        if not HAS_SHAP:
            QMessageBox.warning(self, 'SHAP Not Installed',
                                'Run:  pip install shap')
            return
        model = self.results[name]['model']
        raw   = getattr(model, 'model', model)
        if not hasattr(raw, 'feature_importances_'):
            QMessageBox.information(
                self, 'Unsupported Model',
                f'SHAP TreeExplainer requires a tree-based model.\n'
                f'"{name}" does not support it.\n\n'
                'Supported: GBDT, XGBoost, LightGBM, CatBoost, Random Forest.')
            return

        # Pick data source: prefer full training set, fall back to shap_cache
        X_src = self.X_all
        if X_src is None or len(X_src) == 0:
            QMessageBox.warning(
                self, 'No Calibration Data',
                'No data available for SHAP.\n'
                'Re-train the model to enable SHAP analysis.')
            return

        # Create dialog once, reuse afterwards
        if self._shap_dlg is None:
            self._shap_dlg = ShapBeeswarmDialog(self)
        self._shap_dlg.load(model, name, X_src, self.feat_names)
        self._shap_dlg.show()
        self._shap_dlg.raise_()
        self._shap_dlg.activateWindow()

    def _plot_scatter(self):
        try:
            name, res = self._check_ready(need_data=False)
            te_pred = np.asarray(res.get('te_pred', []))
            # Labels are injected into result entries by load().
            # Use res['_y_te'] first; fall back to self.y_te only when absent.
            _yte_r = res.get('_y_te')
            _ytr_r = res.get('_y_tr')
            y_te_src = _yte_r if (_yte_r is not None and len(np.asarray(_yte_r)) > 0) else self.y_te
            y_tr_src = _ytr_r if (_ytr_r is not None and len(np.asarray(_ytr_r)) > 0) else self.y_tr
            y_te = np.asarray(y_te_src) if y_te_src is not None else np.array([])
            if len(te_pred) == 0 or len(y_te) == 0:
                QMessageBox.warning(
                    self, 'Prediction Data Unavailable',
                    'No prediction data are associated with this model.\n\n'
                    'Possible causes:\n'
                    '  • The bundle was saved without raw predictions\n'
                    '    (format predates v2.0).\n'
                    '  • The model has not been trained yet.\n\n'
                    'Please retrain the model to enable this plot.')
                return
            dlg = ScatterPlotDialog(
                parent=self, name=name, res=res,
                y_tr=y_tr_src, y_te=y_te_src,
                loaded_model=self._loaded_model_mode,
                algo_colors=ALGO_COLORS)
            dlg.exec_()
        except RuntimeError as exc:
            QMessageBox.warning(self, 'Cannot Generate Plot', str(exc))
        except Exception as exc:
            QMessageBox.critical(
                self,
                'Plot Error \u2014 Predicted vs. Measured',
                f'{exc}\n\n{traceback.format_exc()}')

    # Inline preview auto-drawn on load
    def _plot_scatter_inline(self, split='both', pct=100,
                             name=None, res=None):
        try:
            if name is None: name = self.model_cb.currentText()
            if res  is None: res  = self.results.get(name)
            if not name or res is None: return
            color   = ALGO_COLORS.get(name, C_ACCENT)
            # Prefer labels stored inside result entry (training mode)
            _y_te   = res.get('_y_te', self.y_te)
            _y_tr   = res.get('_y_tr', self.y_tr)
            y_te    = np.asarray(_y_te) if _y_te is not None else np.array([])
            y_tr    = np.asarray(_y_tr) if _y_tr is not None else np.array([])
            te_pred = np.asarray(res.get('te_pred', []))
            tr_pred = np.asarray(res.get('tr_pred', []))
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            all_true, all_pred = [], []
            if split in ('both', 'train') and len(tr_pred) > 0 and len(y_tr) > 0:
                yt_s, yp_s = _sample_pair(y_tr, tr_pred, pct)
                ax.scatter(yt_s, yp_s, s=10, alpha=0.22, color='#9DC3E6',
                           label=f'Training  ($n$ = {len(yt_s)})', zorder=2)
                all_true.append(y_tr); all_pred.append(tr_pred)
            if split in ('both', 'test') and len(te_pred) > 0 and len(y_te) > 0:
                yte_s, yp_s = _sample_pair(y_te, te_pred, pct)
                ax.scatter(yte_s, yp_s, s=16, alpha=0.70, color=color,
                           label=f'Test  ($n$ = {len(yte_s)})', zorder=3)
                all_true.append(y_te); all_pred.append(te_pred)
            if not all_true:
                ax.text(0.5, 0.5, 'No data for the selected split.',
                        ha='center', va='center',
                        transform=ax.transAxes, fontsize=11, color='#888')
                self.canvas.draw(); return
            vmax = max(float(np.concatenate(all_true).max()),
                       float(np.concatenate(all_pred).max())) * 1.08
            ax.plot([0, vmax], [0, vmax], color='#333', ls='--', lw=1.2,
                    label='Perfect agreement  (1:1)')
            ax.plot([0, vmax], [0, vmax * 1.2], ':', color='#70AD47',
                    lw=1.0, alpha=0.70, label='\u00b120\u202f% bounds')
            ax.plot([0, vmax], [0, vmax * 0.8], ':', color='#70AD47',
                    lw=1.0, alpha=0.70)
            ax.set_xlim(0, vmax); ax.set_ylim(0, vmax)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(r'Measured  $V_\mathrm{exp}$  (kN)', fontsize=11)
            ax.set_ylabel(r'Predicted  $V_\mathrm{pred}$  (kN)', fontsize=11)
            slbl = {'train': 'Training Set', 'test': 'Test Set',
                    'both': 'Full Dataset'}[split]
            ax.set_title(
                f'{name}  \u2014  Predicted vs. Measured  [{slbl}]',
                fontsize=11, fontweight='bold')
            tm = res.get('te_metrics', {})
            if tm:
                ax.text(0.04, 0.96,
                        f'$R^2$ = {tm["R2"]:.4f}   RMSE = {tm["RMSE"]:.2f} kN',
                        transform=ax.transAxes, fontsize=9, va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=C_BORDER, alpha=0.9))
            ax.legend(fontsize=9, loc='lower right', framealpha=0.9, edgecolor=C_BORDER)
            ax.tick_params(labelsize=9)
            try: self.canvas.fig.tight_layout()
            except Exception: pass
            self.canvas.draw()
        except Exception as exc:
            QMessageBox.critical(self, 'Preview Error',
                                 f'{exc}\n\n{traceback.format_exc()}')

    def _plot_importance(self):
        try:
            name, res = self._check_ready(need_data=False)
            model = res['model']
            if not hasattr(model, 'feature_importances_'):
                QMessageBox.information(
                    self, 'Method Not Applicable',
                    f'Variable importance is unavailable for \u201c{name}\u201d.\n\n'
                    'Gini impurity-based importance is defined only for\n'
                    'tree-ensemble models (GBDT, XGBoost, LightGBM,\n'
                    'CatBoost, Random Forest).')
                return
            imp    = np.asarray(model.feature_importances_)
            # Guard: feat_names may be shorter than imp if the bundle was
            # trained with different columns.  Pad with generic names so the
            # lengths always match before indexing.
            feat_n = list(self.feat_names)
            if len(feat_n) < len(imp):
                feat_n += [f'Feature {i}' for i in range(len(feat_n), len(imp))]
            labels = feat_n[:len(imp)]
            idx    = np.argsort(imp)
            norm   = imp[idx] / (imp.max() if imp.max() > 0 else 1.0)
            clrs   = [plt.cm.Blues(0.30 + 0.70 * float(v)) for v in norm]
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            ax.barh(range(len(imp)), imp[idx],
                    color=clrs, edgecolor='white', height=0.65)
            ax.set_yticks(range(len(imp)))
            ax.set_yticklabels([labels[i] for i in idx], fontsize=10)
            ax.set_xlabel(
                'Gini Impurity-based Feature Importance', fontsize=11)
            ax.set_title(
                f'{name}  \u2014  Feature Importance Ranking',
                fontsize=11, fontweight='bold')
            # idx is ascending (least→most important); reverse for rank 1 = most important
            imp_df = pd.DataFrame({
                'rank':       range(1, len(idx) + 1),
                'feature':    [labels[i] for i in reversed(idx)],
                'importance': imp[list(reversed(idx))],
            })
            self._current_plot_data = {
                'type': 'feature_importance',
                'name': name,
                'df':   imp_df,
            }
            if self._save_plot_data_btn:
                self._save_plot_data_btn.setEnabled(True)
                self._save_plot_data_btn.setToolTip(
                    f'Export feature importance data for {name}\n'
                    f'Columns: feature, importance, rank')
            try: self.canvas.fig.tight_layout()
            except Exception: pass
            self.canvas.draw()
        except RuntimeError as exc:
            QMessageBox.warning(self, 'Cannot Generate Plot', str(exc))
        except Exception as exc:
            QMessageBox.critical(
                self, 'Plot Error \u2014 Variable Importance',
                f'{exc}\n\n{traceback.format_exc()}')

    def _plot_error(self):
        try:
            name, res = self._check_ready(need_data=True)
            dlg = _PctDialog('Residual Error Distribution', self)
            if dlg.exec_() != QDialog.Accepted: return
            pct = dlg.selected_pct()
            _y_te   = res.get('_y_te', self.y_te)
            y_te    = np.asarray(_y_te) if _y_te is not None else np.array([])
            te_pred = np.asarray(res['te_pred'])
            y_s, p_s = _sample_pair(y_te, te_pred, pct)
            err      = (p_s - y_s) / y_s * 100
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            ax.hist(err, bins=40,
                    color=ALGO_COLORS.get(name, C_ACCENT),
                    edgecolor='white', alpha=0.85)
            ax.axvline(0, color=C_DANGER, lw=2.0,
                       label='Zero relative error  (reference)')
            ax.axvline(err.mean(), color='#D35400', lw=1.8, ls='--',
                       label=f'Sample mean = {err.mean():.2f}\u202f%')
            ax.axvspan(-20, 20, alpha=0.07, color='green',
                       label='\u00b120\u202f% acceptance region')
            ax.set_xlabel(
                r'Relative Prediction Error  '
                r'$(V_\mathrm{pred}-V_\mathrm{exp})\,/\,V_\mathrm{exp}$  (%)',
                fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            pct_note = f'  [{pct}\u202f% sample]' if pct < 100 else ''
            ax.set_title(
                f'{name}  \u2014  Residual Error Distribution{pct_note}',
                fontsize=11, fontweight='bold')
            in20 = np.mean(np.abs(err) <= 20) * 100
            ax.text(0.985, 0.97,
                    f'Within \u00b120\u202f%: {in20:.1f}\u202f%\n'
                    f'\u03c3 = {err.std():.2f}\u202f%',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor=C_BORDER, alpha=0.93))
            # Legend: upper-left — avoids stats box (top-right)
            ax.legend(fontsize=9, loc='upper left', framealpha=0.9,
                      edgecolor=C_BORDER)
            err_df = pd.DataFrame({
                'V_exp_kN':         y_s.tolist(),
                'V_pred_kN':        p_s.tolist(),
                'abs_error_kN':     (p_s - y_s).tolist(),
                'rel_error_pct':    err.tolist(),
                'split':            'test',
                'model':            name,
            })
            self._current_plot_data = {
                'type': 'error_distribution',
                'name': name,
                'pct':  pct,
                'df':   err_df,
            }
            if self._save_plot_data_btn:
                self._save_plot_data_btn.setEnabled(True)
                self._save_plot_data_btn.setToolTip(
                    f'Export error distribution data for {name} ({pct}% sample)\n'
                    f'Columns: V_exp_kN, V_pred_kN, abs_error_kN, rel_error_pct')
            try: self.canvas.fig.tight_layout()
            except Exception: pass
            self.canvas.draw()
        except RuntimeError as exc:
            QMessageBox.warning(self, 'Cannot Generate Plot', str(exc))
        except Exception as exc:
            QMessageBox.critical(
                self, 'Plot Error \u2014 Residual Error Distribution',
                f'{exc}\n\n{traceback.format_exc()}')

    def _plot_ratio(self):
        try:
            name, res = self._check_ready(need_data=True)
            dlg = _PctDialog(
                'Safety Factor  (Vpred / Vexp)  Distribution', self)
            if dlg.exec_() != QDialog.Accepted: return
            pct = dlg.selected_pct()
            _y_te   = res.get('_y_te', self.y_te)
            y_te    = np.asarray(_y_te) if _y_te is not None else np.array([])
            te_pred = np.asarray(res['te_pred'])
            y_s, p_s = _sample_pair(y_te, te_pred, pct)
            ratio    = p_s / y_s
            ratio    = ratio[np.isfinite(ratio) & (ratio > 0)]
            if not len(ratio):
                raise RuntimeError(
                    'No valid ratio values (all non-finite or \u22640).')
            self.canvas.fig.clear()
            ax = self.canvas.fig.add_subplot(111)
            _, bins, patches = ax.hist(
                ratio, bins=40,
                color=ALGO_COLORS.get(name, C_ACCENT),
                edgecolor='white', alpha=0.85)
            for patch, left in zip(patches, bins[:-1]):
                if left < 1.0: patch.set_facecolor(C_DANGER)
            ax.axvline(1.0, color='#1F4E79', lw=2.0,
                       label=r'$V_\mathrm{pred}/V_\mathrm{exp} = 1.0$  (reference)')
            ax.axvline(ratio.mean(), color='#D35400', lw=1.8, ls='--',
                       label=f'$\\bar{{\\kappa}}$ = {ratio.mean():.4f}  (sample mean)')
            ax.set_xlabel(
                r'$V_\mathrm{pred}$ / $V_\mathrm{exp}$', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            pct_note = f'  [{pct}\u202f% sample]' if pct < 100 else ''
            ax.set_title(
                f'{name}  \u2014  '
                r'Safety Factor  ($V_\mathrm{pred}/V_\mathrm{exp}$)'
                f'  Distribution{pct_note}',
                fontsize=11, fontweight='bold')
            _mean_ratio = ratio.mean()
            _cov_str    = (f'{ratio.std() / _mean_ratio:.4f}'
                           if _mean_ratio != 0.0 else 'N/A')
            safe = np.mean(ratio <= 1.0) * 100
            ax.text(0.985, 0.97,
                    f'$\\bar{{\\kappa}}$ = {_mean_ratio:.4f}\n'
                    f'CoV = {_cov_str}\n'
                    f'$P(\\kappa\\leq1)$ = {safe:.1f}\u202f%',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                              edgecolor=C_BORDER, alpha=0.93))
            # Legend: upper-left — avoids stats box (top-right)
            ax.legend(fontsize=9, loc='upper left', framealpha=0.9,
                      edgecolor=C_BORDER)
            _valid = np.isfinite(p_s / y_s) & (y_s > 0) & (p_s / y_s > 0)
            ratio_df = pd.DataFrame({
                'V_exp_kN':         y_s[_valid].tolist(),
                'V_pred_kN':        p_s[_valid].tolist(),
                'ratio_Vpred_Vexp': (p_s[_valid] / y_s[_valid]).tolist(),
                'split':            'test',
                'model':            name,
            })
            self._current_plot_data = {
                'type': 'safety_index',
                'name': name,
                'pct':  pct,
                'df':   ratio_df,
            }
            if self._save_plot_data_btn:
                self._save_plot_data_btn.setEnabled(True)
                self._save_plot_data_btn.setToolTip(
                    f'Export safety index data for {name} ({pct}% sample)\n'
                    f'Columns: V_exp_kN, V_pred_kN, ratio_Vpred_Vexp')
            try: self.canvas.fig.tight_layout()
            except Exception: pass
            self.canvas.draw()
        except RuntimeError as exc:
            QMessageBox.warning(self, 'Cannot Generate Plot', str(exc))
        except Exception as exc:
            QMessageBox.critical(
                self, 'Plot Error \u2014 Safety Factor Distribution',
                f'{exc}\n\n{traceback.format_exc()}')

    def _plot_response_surface(self):
        """Generate a response surface (2D line or 3D/contour surface)
        showing how the selected model's prediction varies with 1 or 2
        input features, with all other features held at their median.
        """
        try:
            name, res = self._check_ready(need_data=False)
            model = res['model']

            # Build feature matrix: train+test (training mode)
            # or shap_cache (bundle-load mode, up to 400 rows)
            parts = []
            if self.X_tr is not None and len(np.asarray(self.X_tr)) > 0:
                parts.append(np.asarray(self.X_tr))
            if self.X_te is not None and len(np.asarray(self.X_te)) > 0:
                parts.append(np.asarray(self.X_te))
            if not parts and self.X_all is not None:
                parts.append(np.asarray(self.X_all))
            if not parts:
                QMessageBox.warning(self, 'No Data',
                    'Feature matrix is not available for this model.\n\n'
                    'Possible fix: the bundle was saved without a SHAP\n'
                    'cache. Re-run train_frp_models.py or retrain via\n'
                    'the GUI to generate a bundle with embedded data.')
                return
            X_all = np.vstack(parts)

            feat_names = list(self.feat_names)
            n_feats    = min(X_all.shape[1], len(feat_names))
            feat_names = feat_names[:n_feats]

            dlg = _ResponseSurfaceDialog(feat_names, self)
            if dlg.exec_() != QDialog.Accepted:
                return

            sel_names = dlg.selected_features()
            plot_type = dlg.plot_type()
            n_pts     = dlg.resolution()
            n_sel     = len(sel_names)

            sel_idx = [feat_names.index(n) for n in sel_names]

            # Reference point: per-column median of all data
            x_ref = np.median(X_all, axis=0)

            # Feature ranges: 5th-95th percentile (robust to outliers)
            def _rng(col_idx):
                col = X_all[:, col_idx]
                lo  = float(np.percentile(col, 5))
                hi  = float(np.percentile(col, 95))
                if hi <= lo:   # constant or near-constant (e.g. OHE)
                    lo, hi = float(col.min()), float(col.max())
                margin = (hi - lo) * 0.05
                return lo - margin, hi + margin

            self.canvas.fig.clear()
            color = ALGO_COLORS.get(name, C_ACCENT)

            if n_sel == 1:
                i0   = sel_idx[0]
                lo, hi = _rng(i0)
                xs   = np.linspace(lo, hi, n_pts)
                X_g  = np.tile(x_ref, (n_pts, 1))
                X_g[:, i0] = xs
                ys   = model.predict(X_g)

                ax = self.canvas.fig.add_subplot(111)
                ax.plot(xs, ys, color=color, lw=2.0)
                ax.fill_between(xs, ys, alpha=0.12, color=color)
                ax.axvline(x_ref[i0], color='#555', ls='--', lw=1.2,
                           label='Median (reference)')
                ax.set_xlabel(
                    f'{sel_names[0]}  (normalised)',
                    fontsize=11)
                ax.set_ylabel(
                    r'Predicted  $V_\mathrm{pred}$  (kN)',
                    fontsize=11)
                ax.set_title(
                    f'{name}  \u2014  Response Surface\n'
                    f'Feature: {sel_names[0]}  '
                    f'(others fixed at median)',
                    fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.2)

                # Store for export
                df_out = pd.DataFrame({
                    'feature_value': xs,
                    'V_pred_kN':     ys,
                    'feature_name':  sel_names[0],
                    'model':         name,
                })
                self._current_plot_data = {
                    'type': 'response_surface_1d',
                    'name': name,
                    'df':   df_out,
                }

            else:
                i0, i1 = sel_idx[0], sel_idx[1]
                lo0, hi0 = _rng(i0)
                lo1, hi1 = _rng(i1)
                xs0 = np.linspace(lo0, hi0, n_pts)
                xs1 = np.linspace(lo1, hi1, n_pts)
                g0, g1 = np.meshgrid(xs0, xs1)

                # Show a "Computing…" placeholder before the (potentially
                # slow) predict call so the user knows the app is working.
                self.canvas.fig.clear()
                _ax_tmp = self.canvas.fig.add_subplot(111)
                _ax_tmp.set_axis_off()
                _ax_tmp.text(0.5, 0.5,
                    f'Computing response surface ({n_pts}×{n_pts} grid) …',
                    ha='center', va='center', transform=_ax_tmp.transAxes,
                    fontsize=10, color='#888888', style='italic')
                self.canvas.draw()
                from PyQt5.QtWidgets import QApplication
                QApplication.processEvents()

                X_g = np.tile(x_ref, (n_pts * n_pts, 1))
                X_g[:, i0] = g0.ravel()
                X_g[:, i1] = g1.ravel()
                Z = model.predict(X_g).reshape(n_pts, n_pts)

                if plot_type == '3d':
                    ax = self.canvas.fig.add_subplot(
                        111, projection='3d')
                    surf = ax.plot_surface(
                        g0, g1, Z,
                        cmap='RdYlGn', alpha=0.88,
                        linewidth=0, antialiased=True)
                    self.canvas.fig.colorbar(
                        surf, ax=ax, shrink=0.5, aspect=12,
                        label=r'$V_\mathrm{pred}$  (kN)')
                    ax.set_xlabel(
                        f'{sel_names[0]}\n(normalised)',
                        fontsize=9, labelpad=8)
                    ax.set_ylabel(
                        f'{sel_names[1]}\n(normalised)',
                        fontsize=9, labelpad=8)
                    ax.set_zlabel(
                        r'$V_\mathrm{pred}$  (kN)',
                        fontsize=9, labelpad=8)
                    ax.set_title(
                        f'{name}  \u2014  Response Surface\n'
                        f'{sel_names[0]}  \u00d7  {sel_names[1]}  '
                        f'(others fixed at median)',
                        fontsize=10, fontweight='bold')
                    ax.tick_params(labelsize=8)
                else:
                    ax  = self.canvas.fig.add_subplot(111)
                    cf  = ax.contourf(g0, g1, Z, levels=20,
                                      cmap='RdYlGn', alpha=0.90)
                    cs  = ax.contour(g0, g1, Z, levels=10,
                                     colors='k', linewidths=0.4,
                                     alpha=0.35)
                    ax.clabel(cs, inline=True, fontsize=7, fmt='%.0f')
                    self.canvas.fig.colorbar(
                        cf, ax=ax,
                        label=r'$V_\mathrm{pred}$  (kN)')
                    ax.scatter(
                        x_ref[i0], x_ref[i1],
                        color='white', edgecolors='k', s=60, zorder=5,
                        label='Median (reference)')
                    ax.set_xlabel(
                        f'{sel_names[0]}  (normalised)',
                        fontsize=11)
                    ax.set_ylabel(
                        f'{sel_names[1]}  (normalised)',
                        fontsize=11)
                    ax.set_title(
                        f'{name}  \u2014  Response Surface  '
                        f'({sel_names[0]}  \u00d7  {sel_names[1]})\n'
                        f'(others fixed at median)',
                        fontsize=11, fontweight='bold')
                    ax.legend(fontsize=9)

                # Store for export
                df_out = pd.DataFrame({
                    sel_names[0]:  g0.ravel(),
                    sel_names[1]:  g1.ravel(),
                    'V_pred_kN':   Z.ravel(),
                    'model':       name,
                })
                self._current_plot_data = {
                    'type': 'response_surface_2d',
                    'name': name,
                    'df':   df_out,
                }

            if self._save_plot_data_btn:
                self._save_plot_data_btn.setEnabled(True)
                self._save_plot_data_btn.setToolTip(
                    f'Export response surface data for {name}\n'
                    f'Features: {", ".join(sel_names)}')

            try:
                self.canvas.fig.tight_layout(pad=1.5)
            except Exception:
                pass
            self.canvas.draw()

        except RuntimeError as exc:
            QMessageBox.warning(self, 'Cannot Generate Plot', str(exc))
        except Exception as exc:
            QMessageBox.critical(
                self,
                'Plot Error \u2014 Response Surface Analysis',
                f'{exc}\n\n{traceback.format_exc()}')

    def _export_current_plot_data(self):
        """Export the raw data underlying the current inline plot to CSV.

        The exported file contains all data points rendered in the last
        plot, enabling independent figure reproduction in Python, R, or
        any other analysis environment.
        """
        pd_info = self._current_plot_data
        if not pd_info or 'df' not in pd_info:
            QMessageBox.warning(self, 'No Plot Data',
                'No plot has been generated yet.\n'
                'Generate a diagnostic plot first, then export its data.')
            return

        plot_type = pd_info.get('type', 'plot')
        model_name = pd_info.get('name', 'model')
        pct        = pd_info.get('pct', 100)
        df         = pd_info['df']

        type_labels = {
            'feature_importance':   'Feature_Importance_Ranking',
            'error_distribution':   'Relative_Error_Distribution',
            'safety_index':         'Safety_Index_Distribution',
            'response_surface_1d':  'Response_Surface_1D',
            'response_surface_2d':  'Response_Surface_2D',
        }
        type_lbl = type_labels.get(plot_type, plot_type)
        pct_tag  = f'_sample{pct}pct' if pct < 100 else ''
        default  = f'{model_name}_{type_lbl}{pct_tag}.csv'

        path, _ = QFileDialog.getSaveFileName(
            self,
            f'Export Plot Data  —  {type_lbl.replace("_", " ")}',
            default,
            'CSV files (*.csv)')
        if not path:
            return

        df.to_csv(path, index=False)

        # Append metadata block
        with open(path, 'a', newline='') as f:
            f.write('\n# Metadata\n')
            f.write(f'# model,{model_name}\n')
            f.write(f'# plot_type,{type_lbl}\n')
            f.write(f'# sampling_pct,{pct}\n')
            f.write(f'# random_seed,42\n')
            f.write(f'# n_rows,{len(df)}\n')

        col_info = ', '.join(df.columns.tolist())
        QMessageBox.information(
            self, 'Export Successful',
            f'Plot data exported for figure reproduction.\n\n'
            f'File: {path}\n'
            f'Rows: {len(df)}\n'
            f'Columns: {col_info}\n\n'
            f'A metadata block is appended at the end of the file.')

    def _export(self):
        if not self.results:
            QMessageBox.warning(self, 'No Results',
                'No evaluation results are available to export.')
            return

        active = self.model_cb.currentText()
        from PyQt5.QtWidgets import QInputDialog
        items  = ['Active model only  (' + active + ')',
                  'All loaded models  (' + str(len(self.results)) + ' total)']
        choice, ok = QInputDialog.getItem(
            self, 'Export Scope',
            'Select which models to include in the export:',
            items, 0, False)
        if not ok:
            return

        export_all = choice.startswith('All')
        names_to_export = (list(self.results.keys())
                           if export_all else [active])
        default_name = ('all_models_metrics.csv'
                        if export_all else f'{active}_metrics.csv')

        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Performance Metrics',
            default_name, 'CSV files (*.csv)')
        if not path:
            return

        rows = []
        for name in names_to_export:
            res = self.results.get(name)
            if res is None:
                continue
            row = {'Model': name}
            row.update({f'Test_{k}':  v for k, v in res['te_metrics'].items()})
            row.update({f'Train_{k}': v for k, v in res['tr_metrics'].items()})
            row['CV_R2_mean'] = res['cv_mean']
            row['CV_R2_std']  = res['cv_std']
            rows.append(row)

        pd.DataFrame(rows).to_csv(path, index=False)
        scope_note = ('all models' if export_all
                      else f'active model: {active}')
        QMessageBox.information(
            self, 'Export Successful',
            f'Performance metrics exported ({scope_note}).\n'
            f'File: {path}')
