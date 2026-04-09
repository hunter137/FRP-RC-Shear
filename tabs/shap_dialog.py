"""
shap_dialog.py
==============
SHAP Interpretability Analysis dialog for the FRP-RC Shear platform.

Classes
-------
_ShapWorker         QThread — computes SHAP values off the GUI thread.
ShapBeeswarmDialog  QDialog — four plot types with academic matplotlib style.

Academic style notes
--------------------
* Serif fonts, inward ticks, no top/right spines (``_ACADEMIC_RC``).
* Colorbars: ``extend='neither'`` removes end-arrows; tick labels use
  text tiers ('Low' / 'Medium' / 'High') instead of numeric annotation.
* All plotting code executes *inside* ``plt.rc_context(_ACADEMIC_RC)``
  so every text element inherits the correct family and size.
"""
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QSpinBox, QComboBox, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavBar

from config import C_ACCENT, C_TEXT2, HAS_SHAP, FEAT_LABELS, VAR_LATEX
from widgets import flat_btn, panel, MplCanvas

import pandas as pd

if HAS_SHAP:
    import shap

_ACADEMIC_RC = {
    'font.family':        'serif',
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'axes.linewidth':     0.8,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         110,
}

def _add_colorbar(fig, ax, sc, label='Feature value'):
    """
    Attach an academic-style colorbar to *ax*.

    Design choices
    --------------
    * ``extend='neither'``  — no pointed end-arrows.
    * Tick labels are text tiers (Low / Medium / High) rather than
      numerics, consistent with SHAP beeswarm conventions in the
      literature (Lundberg & Lee, 2017).
    * Vertical label rotated 270°; serif font; thin outline.
    """
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02,
                      extend='neither')
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(['Low', 'Medium', 'High'])
    cb.set_label(label, fontsize=9, rotation=270,
                 labelpad=14, fontfamily='serif')
    cb.ax.tick_params(labelsize=8, direction='in', length=3)
    cb.outline.set_linewidth(0.6)
    return cb

_TREE_EXPLAINER_CLASSES = {
    'GradientBoostingRegressor',
    'RandomForestRegressor',
    'ExtraTreesRegressor',
    'XGBRegressor',
    'LGBMRegressor',
    'CatBoostRegressor',
    'DecisionTreeRegressor',
}

def _supports_tree_explainer(model) -> bool:
    """Return True if shap.TreeExplainer supports this model class."""
    raw = getattr(model, 'model', model)
    return type(raw).__name__ in _TREE_EXPLAINER_CLASSES

class _ShapWorker(QThread):
    """
    Compute SHAP values on a background thread to keep the GUI responsive.

    Uses shap.TreeExplainer for native tree models (GBDT / RF / XGB / LGB …)
    and falls back to shap.KernelExplainer for everything else
    (AdaBoost, SVR, KNN, MLP …).  KernelExplainer is model-agnostic but
    slower — a background-data subsample of ≤100 rows is used to keep it
    tractable.

    Signals
    -------
    done(shap_values, X_sample)
    log_s(str)
    failed(str)
    """
    done   = pyqtSignal(object, object)
    log_s  = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model, X_sample, X_background=None):
        super().__init__()
        self._model       = model
        self._X_sample    = X_sample
        # Background data for KernelExplainer (subset of training set)
        self._X_background = X_background

    def run(self):
        if not HAS_SHAP:
            self.failed.emit(
                'shap is not installed.\n'
                'Install it with:  pip install shap')
            return
        try:
            n   = len(self._X_sample)
            raw = getattr(self._model, 'model', self._model)

            if _supports_tree_explainer(self._model):
                self.log_s.emit(
                    f'[INFO] TreeExplainer — computing SHAP for {n} samples …')
                explainer   = shap.TreeExplainer(raw)
                shap_values = explainer.shap_values(self._X_sample)
            else:
                cls_name = type(raw).__name__
                # Use a small background dataset (≤100 rows) via k-means
                # summary to keep KernelExplainer tractable.
                if self._X_background is not None and len(self._X_background) > 0:
                    bg_size = min(100, len(self._X_background))
                    bg = shap.kmeans(self._X_background, bg_size)
                else:
                    bg = shap.kmeans(self._X_sample, min(50, n))
                self.log_s.emit(
                    f'[INFO] KernelExplainer ({cls_name}) — '
                    f'computing SHAP for {n} samples  '
                    f'[this may take a minute] …')
                explainer   = shap.KernelExplainer(raw.predict, bg)
                shap_values = explainer.shap_values(
                    self._X_sample, silent=True)

            # shap >= 0.40 may return an Explanation object — unwrap to ndarray
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            self.log_s.emit('[INFO] SHAP computation complete.')
            self.done.emit(shap_values, self._X_sample)
        except Exception as exc:
            self.failed.emit(str(exc))

class ShapBeeswarmDialog(QDialog):
    """
    SHAP Interpretability Analysis pop-up launched from the Model
    Evaluation tab.

    Plot types
    ----------
    1. SHAP Bar Summary        Mean |φᵢ| horizontal bar chart.
    2. SHAP Beeswarm Plot      Per-sample dot cloud (top-10 features).
    3. SHAP Dependence Plot    φᵢ vs. feature value for a chosen feature.
    4. Partial Dependence Plot Marginal model response for a chosen feature.

    All plots are rendered inside ``plt.rc_context(_ACADEMIC_RC)`` so
    every text element uses serif fonts and consistent sizing.
    """

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint,
        )
        self.setWindowTitle('SHAP Analysis')
        self.resize(1040, 650)

        self._model      = None
        self._model_name = ''
        self._X_all      = None
        self._feat_names = list(FEAT_LABELS)
        self._shap_vals  = None
        self._X_sample   = None
        self._worker     = None

        self._build_ui()

    def load(self, model, model_name, X_all, feat_names):
        """Populate dialog with a new model and calibration dataset."""
        self._model      = model
        self._model_name = model_name
        self._X_all      = np.asarray(X_all) if X_all is not None else None
        self._feat_names = list(feat_names) if feat_names else list(FEAT_LABELS)
        self._shap_vals  = None
        self._X_sample   = None

        n = len(self._X_all) if self._X_all is not None else 0
        self.setWindowTitle(f'SHAP Analysis — {model_name}')
        self._model_lbl.setText(model_name)
        self._avail_lbl.setText(f'{n} samples available')
        self._status_lbl.setText(
            'Click "Compute SHAP Values" to enable plot types.')

        self._feat_cb.clear()
        self._feat_cb.addItems(self._feat_names)
        self._set_plot_btns_enabled(False)
        self._export_btn.setEnabled(False)
        self._draw_placeholder()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_canvas_panel())

    def _build_left_panel(self):
        left = QWidget()
        left.setFixedWidth(252)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)
        ll.setContentsMargins(0, 0, 0, 0)

        ll.addWidget(self._build_model_group())
        ll.addWidget(self._build_compute_group())
        ll.addWidget(self._build_plot_type_group())
        ll.addWidget(self._build_feature_group())
        ll.addStretch()
        ll.addWidget(self._build_export_btn())
        ll.addWidget(self._build_close_btn())

        return left

    def _build_model_group(self):
        grp = panel('Model')
        mg  = QVBoxLayout(grp)
        self._model_lbl = QLabel('—')
        self._model_lbl.setStyleSheet(
            f'font-size:12px;font-weight:bold;color:{C_ACCENT};')
        self._avail_lbl = QLabel('')
        self._avail_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        mg.addWidget(self._model_lbl)
        mg.addWidget(self._avail_lbl)
        return grp

    def _build_compute_group(self):
        grp = panel('SHAP Computation')
        sg  = QVBoxLayout(grp)

        sg.addWidget(QLabel('No. of samples for SHAP:'))

        self._spin = QSpinBox()
        self._spin.setRange(50, 1000)
        self._spin.setValue(200)
        self._spin.setFixedHeight(28)
        sg.addWidget(self._spin)

        note = QLabel('Seed fixed (42).  Range: 50\u20131000.')
        note.setStyleSheet(f'font-size:9px;color:{C_TEXT2};')
        sg.addWidget(note)

        self._compute_btn = flat_btn('Compute SHAP', accent=True)
        self._compute_btn.setFixedHeight(36)
        self._compute_btn.clicked.connect(self._run_shap)
        sg.addWidget(self._compute_btn)

        self._status_lbl = QLabel('Load a model first.')
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        sg.addWidget(self._status_lbl)

        return grp

    def _build_plot_type_group(self):
        grp = panel('Plot Type')
        pg  = QVBoxLayout(grp)
        pg.setSpacing(4)

        self._plot_btns = []
        for label, handler in [
            ('SHAP Bar Summary',        self._plot_bar),
            ('SHAP Beeswarm Plot',      self._plot_beeswarm),
            ('SHAP Dependence Plot',    self._plot_dependence),
            ('Partial Dependence Plot', self._plot_pdp),
        ]:
            btn = flat_btn(label)
            btn.setFixedHeight(34)
            btn.setEnabled(False)
            btn.clicked.connect(handler)
            pg.addWidget(btn)
            self._plot_btns.append(btn)

        return grp

    def _build_feature_group(self):
        grp = panel('Feature (Dependence / PDP)')
        fg  = QVBoxLayout(grp)
        self._feat_cb = QComboBox()
        self._feat_cb.addItems(FEAT_LABELS)
        fg.addWidget(self._feat_cb)
        return grp

    def _build_export_btn(self):
        self._export_btn = flat_btn('Export SHAP (CSV)')
        self._export_btn.setFixedHeight(34)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        return self._export_btn

    def _build_close_btn(self):
        btn = flat_btn('Close')
        btn.setFixedHeight(34)
        btn.clicked.connect(self.close)
        return btn

    def _build_canvas_panel(self):
        grp = panel('Interpretability / Feature Analysis')
        cl  = QVBoxLayout(grp)
        self._canvas = MplCanvas(width=10, height=6)
        cl.addWidget(_NavBar(self._canvas, self))
        cl.addWidget(self._canvas)
        return grp

    def _set_plot_btns_enabled(self, flag: bool):
        for btn in self._plot_btns:
            btn.setEnabled(flag)

    def _draw_placeholder(self):
        with plt.rc_context(_ACADEMIC_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.54,
                    'Select a plot type from the panel on the left.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='#AAAAAA', style='italic')
            ax.text(0.5, 0.44,
                    'SHAP-based plots require computing SHAP values first.',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#CCCCCC', style='italic')
        try:
            self._canvas.fig.tight_layout()
        except Exception:
            pass
        self._canvas.draw()

    def _check_shap(self) -> bool:
        if self._shap_vals is None:
            QMessageBox.information(
                self, 'SHAP Values Required',
                'Please click "Compute SHAP Values" first.')
            return False
        return True

    def _feat_index(self):
        # Returns the index of the currently selected feature, or None.
        name   = self._feat_cb.currentText()
        n_feat = self._shap_vals.shape[1] if self._shap_vals is not None \
                 else len(self._feat_names)
        labels = self._feat_names[:n_feat]
        return labels.index(name) if name in labels else None

    def _run_shap(self):
        if not HAS_SHAP:
            QMessageBox.warning(self, 'SHAP Not Installed',
                                'Run:  pip install shap')
            return
        if self._model is None:
            QMessageBox.warning(self, 'No Model', 'No model loaded.')
            return
        if self._X_all is None or len(self._X_all) == 0:
            QMessageBox.warning(
                self, 'No Calibration Data',
                'No data available for SHAP.\n'
                'Re-train the model to enable SHAP analysis.')
            return

        # Notify user when falling back to KernelExplainer (slower)
        if not _supports_tree_explainer(self._model):
            raw      = getattr(self._model, 'model', self._model)
            cls_name = type(raw).__name__
            reply = QMessageBox.question(
                self, 'Slower SHAP Method',
                f'{cls_name} is not supported by TreeExplainer.\n\n'
                'SHAP will use KernelExplainer instead — this is '
                'model-agnostic but significantly slower.\n\n'
                'Recommended: keep the sample count ≤ 100 for '
                'this model type.\n\nContinue?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return

        n   = min(self._spin.value(), len(self._X_all))
        idx = np.random.default_rng(42).choice(
            len(self._X_all), n, replace=False)
        X_s = self._X_all[idx]

        self._compute_btn.setEnabled(False)
        self._set_plot_btns_enabled(False)
        self._export_btn.setEnabled(False)
        self._status_lbl.setText(
            f'Computing SHAP values for {n} samples …')

        self._worker = _ShapWorker(self._model, X_s,
                                   X_background=self._X_all)
        self._worker.log_s.connect(self._status_lbl.setText)
        self._worker.done.connect(self._on_shap_done)
        self._worker.failed.connect(self._on_shap_fail)
        self._worker.start()

    def _on_shap_done(self, shap_values, X_sample):
        self._shap_vals = shap_values
        self._X_sample  = X_sample
        self._compute_btn.setEnabled(True)
        self._set_plot_btns_enabled(True)
        self._export_btn.setEnabled(True)
        n_feat = shap_values.shape[1]
        self._status_lbl.setText(
            f'SHAP values ready  '
            f'({len(X_sample)} samples, {n_feat} features).')
        self._plot_beeswarm()   # default view

    def _on_shap_fail(self, msg):
        self._compute_btn.setEnabled(True)
        self._status_lbl.setText(f'[ERROR] {msg}')
        QMessageBox.critical(self, 'SHAP Error', msg)

    def _plot_bar(self):
        if not self._check_shap():
            return
        sv     = self._shap_vals
        n_feat = min(sv.shape[1], len(self._feat_names))
        means  = np.abs(sv).mean(axis=0)[:n_feat]
        labels = self._feat_names[:n_feat]
        idx    = np.argsort(means)          # ascending → bottom of chart first
        m_max  = means.max() if means.max() > 0 else 1.0

        with plt.rc_context(_ACADEMIC_RC):
            self._canvas.fig.clear()
            ax   = self._canvas.fig.add_subplot(111)
            cmap = plt.cm.Blues
            clrs = [cmap(0.30 + 0.65 * float(v / m_max)) for v in means[idx]]

            bars = ax.barh(range(n_feat), means[idx],
                           color=clrs, edgecolor='white',
                           height=0.60, linewidth=0.5)
            ax.set_yticks(range(n_feat))
            ax.set_yticklabels(
                [VAR_LATEX.get(labels[i], labels[i]) for i in idx])
            ax.set_xlabel(
                r'Mean $|\phi_i|$ — average absolute SHAP value (kN)')
            ax.set_title(
                f'SHAP Feature Importance — {self._model_name}'
                f'  (n\u202f=\u202f{len(self._X_sample)})',
                fontweight='bold')
            ax.tick_params(axis='y', length=0)
            ax.xaxis.set_tick_params(which='both', direction='in')

            for bar, val in zip(bars, means[idx]):
                ax.text(
                    val + m_max * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}',
                    va='center', fontsize=8)
            ax.set_xlim(right=m_max * 1.16)

            try:
                self._canvas.fig.tight_layout()
            except Exception:
                pass

        self._canvas.draw()

    def _plot_beeswarm(self):
        if not self._check_shap():
            return
        sv     = self._shap_vals
        X      = self._X_sample
        n_feat = min(sv.shape[1], len(self._feat_names))
        means  = np.abs(sv).mean(axis=0)[:n_feat]
        top_k  = min(10, n_feat)
        top_idx = np.argsort(means)[::-1][:top_k]
        ordered = top_idx[::-1]             # bottom-to-top on y-axis
        labels  = [VAR_LATEX.get(self._feat_names[i], self._feat_names[i])
                   for i in ordered]
        rng     = np.random.default_rng(0)

        with plt.rc_context(_ACADEMIC_RC):
            self._canvas.fig.clear()
            ax   = self._canvas.fig.add_subplot(111)
            cmap = plt.cm.RdBu_r
            sc   = None

            for rank, fi in enumerate(ordered):
                vals   = sv[:, fi]
                feats  = X[:, fi]
                span   = feats.max() - feats.min()
                fnorm  = (feats - feats.min()) / (span + 1e-9)
                jitter = rng.uniform(-0.22, 0.22, len(vals))
                y_pos  = np.full(len(vals), rank) + jitter
                sc = ax.scatter(
                    vals, y_pos, c=fnorm, cmap=cmap,
                    s=8, alpha=0.60, linewidths=0, vmin=0, vmax=1)

            ax.set_yticks(range(top_k))
            ax.set_yticklabels(labels)
            ax.axvline(0, color='#555555', lw=0.7, ls='--')
            ax.set_xlabel(
                r'SHAP value $\phi_i$ — impact on predicted'
                r' $V_\mathrm{pred}$ (kN)')
            ax.set_title(
                f'SHAP Beeswarm Plot — {self._model_name}'
                f'  (n\u202f=\u202f{len(X)}, top {top_k} features)',
                fontweight='bold')

            if sc is not None:
                _add_colorbar(self._canvas.fig, ax, sc,
                              label='Feature value')

            try:
                self._canvas.fig.tight_layout()
            except Exception:
                pass

        self._canvas.draw()

    def _plot_dependence(self):
        if not self._check_shap():
            return
        fi = self._feat_index()
        if fi is None:
            return
        feat_name = self._feat_cb.currentText()
        vals      = self._X_sample[:, fi]
        sv_col    = self._shap_vals[:, fi]

        feat_label = VAR_LATEX.get(feat_name, feat_name)

        with plt.rc_context(_ACADEMIC_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)
            sc = ax.scatter(
                vals, sv_col, c=sv_col, cmap=plt.cm.RdBu_r,
                s=18, alpha=0.75, linewidths=0.3, edgecolors='white')
            ax.axhline(0, color='#555555', lw=0.7, ls='--')
            ax.set_xlabel(feat_label)
            ax.set_ylabel(
                rf'SHAP value $\phi$ for {feat_label} (kN)')
            ax.set_title(
                f'SHAP Dependence Plot — {feat_label}'
                f'  [{self._model_name}]',
                fontweight='bold')

            # For dependence plot the colorbar represents SHAP magnitude,
            # so numeric ticks are more informative than Low/Medium/High.
            cb = self._canvas.fig.colorbar(
                sc, ax=ax, fraction=0.025, pad=0.02, extend='neither')
            cb.set_label(
                rf'$\phi$ for {feat_name} (kN)',
                fontsize=9, rotation=270, labelpad=14, fontfamily='serif')
            cb.ax.tick_params(labelsize=8, direction='in', length=3)
            cb.outline.set_linewidth(0.6)

            try:
                self._canvas.fig.tight_layout()
            except Exception:
                pass

        self._canvas.draw()

    def _plot_pdp(self):
        try:
            from sklearn.inspection import PartialDependenceDisplay
        except ImportError:
            QMessageBox.warning(
                self, 'Not Available',
                'PartialDependenceDisplay requires scikit-learn ≥ 1.1.\n'
                'Upgrade with:  pip install -U scikit-learn')
            return

        if self._model is None or self._X_all is None:
            QMessageBox.warning(self, 'No Data',
                                'No model or calibration data loaded.')
            return
        fi = self._feat_index() if self._shap_vals is not None \
             else self._feat_names.index(self._feat_cb.currentText()) \
             if self._feat_cb.currentText() in self._feat_names else None
        if fi is None:
            return

        feat_name = self._feat_cb.currentText()
        n_feat    = min(len(self._feat_names), self._X_all.shape[1])
        labels    = self._feat_names[:n_feat]
        feat_label = VAR_LATEX.get(feat_name, feat_name)
        raw       = getattr(self._model, 'model', self._model)

        with plt.rc_context(_ACADEMIC_RC):
            self._canvas.fig.clear()
            ax = self._canvas.fig.add_subplot(111)

            # Build kwargs — ice_lines_kw added in sklearn 1.1
            pdp_kw  = dict(color='#1f4e79', lw=1.5)
            try:
                disp = PartialDependenceDisplay.from_estimator(
                    raw, self._X_all, [fi],
                    feature_names=labels,
                    ax=ax,
                    pd_line_kw=pdp_kw,
                    ice_lines_kw=dict(color='#9ecae1',
                                      alpha=0.20, lw=0.5),
                )
            except TypeError:
                # Older sklearn: no ice_lines_kw
                disp = PartialDependenceDisplay.from_estimator(
                    raw, self._X_all, [fi],
                    feature_names=labels,
                    ax=ax,
                    pd_line_kw=pdp_kw,
                )

            pdp_ax = disp.axes_[0][0]
            pdp_ax.set_xlabel(feat_label)
            pdp_ax.set_ylabel(
                r'Partial dependence on $V_\mathrm{pred}$ (kN)')
            pdp_ax.set_title(
                f'Partial Dependence Plot — {feat_label}'
                f'  [{self._model_name}]',
                fontweight='bold')
            pdp_ax.tick_params(direction='in')
            # Remove top/right spines if PDP added them back
            pdp_ax.spines['top'].set_visible(False)
            pdp_ax.spines['right'].set_visible(False)

            try:
                self._canvas.fig.tight_layout()
            except Exception:
                pass

        self._canvas.draw()

    def _export_csv(self):
        if self._shap_vals is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export SHAP Values', 'shap_values.csv',
            'CSV files (*.csv)')
        if not path:
            return
        sv     = self._shap_vals
        n_feat = min(sv.shape[1], len(self._feat_names))
        cols   = self._feat_names[:n_feat]
        df     = pd.DataFrame(sv[:, :n_feat], columns=cols)
        df.insert(0, 'sample_index', range(len(df)))
        df.to_csv(path, index=False)
        QMessageBox.information(
            self, 'Export Complete',
            f'SHAP values saved to:\n{path}')
