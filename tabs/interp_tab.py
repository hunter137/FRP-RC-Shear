"""Model interpretability tab — SHAP and partial dependence analysis.

Changes vs v2.0:
  * Load Model Bundle directly from this tab (no training required).
  * Load Data File works without a pre-fitted scaler.
  * shap.Explainer auto-selects Tree / Linear / Kernel — all algorithms work.
  * Clear step-by-step UI guides the user.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QMessageBox,
    QFileDialog, QFrame, QPushButton, QGroupBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT, C_ACCENT_LT,
    C_PANEL_BG, C_SUCCESS, C_DANGER,
    HAS_SHAP, HAS_PDP,
    ALGO_COLORS, FEAT_LABELS, VAR_LATEX, NUM_FEAT_COLS,
)
from widgets import flat_btn, panel, MplCanvas

if HAS_SHAP:
    import shap
if HAS_PDP:
    from sklearn.inspection import PartialDependenceDisplay

class PdpWorker(QThread):
    """Computes PartialDependenceDisplay off the GUI thread.

    On GBDT / XGBoost with thousands of estimators and ≥400 background
    rows the PDP calculation can take 5-30 s.  Running it on the main
    thread freezes the entire UI.  This worker offloads the computation
    and emits done/failed signals so the GUI can redraw safely.
    """
    done   = pyqtSignal(object)   # emits the fitted PartialDependenceDisplay
    failed = pyqtSignal(str)

    def __init__(self, raw_model, X_all, feat_idx, feat_names):
        super().__init__()
        self._model      = raw_model
        self._X          = X_all
        self._feat_idx   = feat_idx
        self._feat_names = feat_names

    def run(self):
        try:
            from sklearn.inspection import PartialDependenceDisplay
            disp = PartialDependenceDisplay.from_estimator(
                self._model, self._X, [self._feat_idx],
                feature_names=self._feat_names)
            self.done.emit(disp)
        except Exception as exc:
            self.failed.emit(str(exc))

class ShapWorker(QThread):
    done   = pyqtSignal(object, object, list)
    log_s  = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, model, X_sample, feat_names, algo_name=''):
        super().__init__()
        self.model      = model
        self.X_sample   = X_sample
        self.feat_names = feat_names
        self.algo_name  = algo_name

    def run(self):
        if not HAS_SHAP:
            self.failed.emit('shap not installed.  Run: pip install shap')
            return
        try:
            raw = getattr(self.model, 'model', self.model)
            n   = len(self.X_sample)
            self.log_s.emit(f'Computing SHAP for {n} samples ...')

            if hasattr(raw, 'feature_importances_'):
                explainer = shap.TreeExplainer(raw)
                sv = explainer.shap_values(self.X_sample)
            elif hasattr(raw, 'coef_'):
                explainer = shap.LinearExplainer(raw, self.X_sample)
                sv = explainer.shap_values(self.X_sample)
            else:
                bg_n = min(50, n)
                # shap.sample() was removed in shap >= 0.42 — use numpy instead
                rng = np.random.default_rng(42)
                idx = rng.choice(len(self.X_sample), bg_n, replace=False)
                bg  = self.X_sample[idx]
                self.log_s.emit(
                    f'KernelExplainer (background={bg_n} samples, may be slow)...')
                explainer = shap.KernelExplainer(raw.predict, bg)
                sv = explainer.shap_values(self.X_sample, nsamples=100)

            # shap >= 0.40 may return an Explanation object instead of ndarray
            if hasattr(sv, 'values'):
                sv = sv.values
            if isinstance(sv, list):
                sv = sv[0]
            self.log_s.emit('SHAP complete.')
            self.done.emit(sv, self.X_sample, self.feat_names)
        except Exception as e:
            self.failed.emit(str(e))

class InterpTab(QWidget):
    def __init__(self):
        super().__init__()
        self.results      = {}
        self.X_all        = None
        self.feat_names   = list(FEAT_LABELS)
        self._scaler      = None
        self._feat_cols   = None
        self._ohe         = None
        self._shap_vals   = None
        self._X_sample    = None
        self._data_source = 'none'
        self._bundle_label = ''
        self._build_ui()

    def load(self, results, X_all, feat_names=None,
             scaler=None, feat_cols=None, ohe=None):
        self.results = results
        self.X_all   = X_all
        if feat_names: self.feat_names = list(feat_names)
        if scaler:     self._scaler    = scaler
        if feat_cols:  self._feat_cols = feat_cols
        if ohe:        self._ohe       = ohe
        self._shap_vals = None
        self._X_sample  = None

        self._refresh_model_cb()

        has_cache = (X_all is not None
                     and isinstance(X_all, np.ndarray)
                     and X_all.size > 0)
        if has_cache:
            self._data_source = 'bundle'
            self._set_data_lbl(f'Bundle cache ready — {len(X_all):,} rows', ok=True)
        else:
            self._data_source = 'none'
            self._set_data_lbl(
                'No background data — load a CSV/Excel file below.', ok=False)

        self.dep_feat_cb.clear()
        self.dep_feat_cb.addItems(self.feat_names)
        self._update_step_status()

    def push_bundle(self, results, scaler, feat_cols, ohe,
                    shap_cache=None, feat_names=None):
        """Called by PredictTab / MainWindow whenever a bundle is loaded."""
        good_cache = (shap_cache is not None
                      and isinstance(shap_cache, np.ndarray)
                      and shap_cache.size > 0)
        self.load(results,
                  shap_cache if good_cache else None,
                  feat_names=feat_names or self.feat_names,
                  scaler=scaler, feat_cols=feat_cols, ohe=ohe)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        left = QWidget()
        left.setFixedWidth(268)
        ll = QVBoxLayout(left)
        ll.setSpacing(6)
        ll.setContentsMargins(0, 0, 0, 0)

        s1 = self._section_box('Step 1: Model Bundle')
        v1 = QVBoxLayout(s1)
        v1.setSpacing(5)

        self._bundle_lbl = QLabel('No bundle loaded.')
        self._bundle_lbl.setWordWrap(True)
        self._bundle_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        v1.addWidget(self._bundle_lbl)

        load_btn = flat_btn('Load Bundle')
        load_btn.setToolTip('Load any .frpmdl bundle — no training needed.')
        load_btn.clicked.connect(self._load_bundle)
        v1.addWidget(load_btn)

        v1.addWidget(QLabel('Active model:'))
        self.model_cb = QComboBox()
        self.model_cb.setToolTip(
            'All algorithms in the bundle are listed.\n'
            'Tree models use TreeExplainer (fast);\n'
            'others fall back to KernelExplainer.')
        self.model_cb.currentIndexChanged.connect(self._on_model_changed)
        v1.addWidget(self.model_cb)

        self._model_type_lbl = QLabel('')
        self._model_type_lbl.setStyleSheet(
            f'font-size:9px;color:{C_TEXT2};font-style:italic;')
        v1.addWidget(self._model_type_lbl)
        ll.addWidget(s1)

        s2 = self._section_box('Step 2: Background Data')
        v2 = QVBoxLayout(s2)
        v2.setSpacing(5)

        hint = QLabel(
            'SHAP needs a background dataset.\n'
            'Bundle cache is used if available;\n'
            'otherwise load any CSV / Excel file:')
        hint.setWordWrap(True)
        hint.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        v2.addWidget(hint)

        self._data_lbl = QLabel('No data loaded.')
        self._data_lbl.setWordWrap(True)
        self._data_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        v2.addWidget(self._data_lbl)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        v2.addWidget(sep)

        browse_btn = flat_btn('Load Dataset')
        browse_btn.setToolTip(
            'Load a CSV or Excel file as SHAP background data.\n'
            'Bundle scaler applied if available; otherwise raw values used.')
        browse_btn.clicked.connect(self._browse_data)
        v2.addWidget(browse_btn)

        self._file_lbl = QLabel('')
        self._file_lbl.setWordWrap(True)
        self._file_lbl.setStyleSheet(
            f'font-size:10px;color:{C_TEXT2};font-style:italic;')
        v2.addWidget(self._file_lbl)
        ll.addWidget(s2)

        s3 = self._section_box('Step 3: Compute SHAP Values')
        v3 = QVBoxLayout(s3)
        v3.setSpacing(5)

        row_spin = QHBoxLayout()
        row_spin.addWidget(QLabel('Max samples:'))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(50, 2000)
        self.max_spin.setValue(400)
        self.max_spin.setToolTip(
            'Samples drawn for SHAP background.\n'
            'Fewer = faster; more = more accurate.')
        row_spin.addWidget(self.max_spin)
        v3.addLayout(row_spin)

        self.compute_btn = flat_btn('Compute SHAP', accent=True)
        self.compute_btn.setFixedHeight(32)
        self.compute_btn.clicked.connect(self._run_shap)
        v3.addWidget(self.compute_btn)

        self._status_lbl = QLabel('Complete steps 1 & 2 first.')
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
        v3.addWidget(self._status_lbl)
        ll.addWidget(s3)

        pg = self._section_box('Plot Type')
        pgv = QVBoxLayout(pg)
        self._plot_btns = []
        for lbl, fn in [
            ('SHAP Bar Summary',        self._plot_bar),
            ('SHAP Beeswarm Plot',      self._plot_beeswarm),
            ('SHAP Dependence Plot',    self._plot_dependence),
            ('Feature Importance',      self._plot_importance),
            ('Partial Dependence Plot', self._plot_pdp),
        ]:
            b = flat_btn(lbl)
            b.clicked.connect(fn)
            pgv.addWidget(b)
            self._plot_btns.append(b)
        ll.addWidget(pg)

        dep = self._section_box('Dependence Variable')
        depv = QVBoxLayout(dep)
        self.dep_feat_cb = QComboBox()
        self.dep_feat_cb.addItems(self.feat_names)
        depv.addWidget(self.dep_feat_cb)
        ll.addWidget(dep)
        ll.addStretch()
        root.addWidget(left)

        canvas_grp = panel('Interpretability / feature analysis')
        cl = QVBoxLayout(canvas_grp)
        self.canvas = MplCanvas(width=10, height=6)
        toolbar = NavigationToolbar(self.canvas, self)
        cl.addWidget(toolbar)
        cl.addWidget(self.canvas)
        root.addWidget(canvas_grp)

    @staticmethod
    def _section_box(title):
        g = QGroupBox(title)
        g.setStyleSheet(
            'QGroupBox{font-weight:bold;font-size:10px;'
            'border:1px solid #d4d4d4;border-radius:3px;'
            'margin-top:7px;padding-top:3px;}'
            'QGroupBox::title{subcontrol-origin:margin;left:6px;}')
        return g

    def _set_data_lbl(self, text, ok=True):
        color = C_SUCCESS if ok else C_TEXT2
        self._data_lbl.setText(text)
        self._data_lbl.setStyleSheet(f'font-size:10px;color:{color};')

    def _load_bundle(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Bundle', '',
            'FRP Model Bundle (*.frpmdl)')
        if not path:
            return
        try:
            from model_io import ModelIO
            lr = ModelIO.load(path)
            results    = lr[0]
            scaler     = lr[1]
            feat_cols  = lr[2]
            ohe        = lr[3]
            shap_cache = lr[4] if len(lr) > 4 else None
            meta       = lr[5] if len(lr) > 5 else {}
        except Exception as ex:
            QMessageBox.critical(self, 'Bundle Load Error', str(ex))
            return

        label = os.path.splitext(os.path.basename(path))[0]
        self._bundle_label = label
        saved_at = (meta or {}).get('saved_at', '--')
        self._bundle_lbl.setText(
            f'{label}  ({len(results)} model(s))\nSaved: {saved_at}')
        self._bundle_lbl.setStyleSheet(
            f'font-size:10px;color:{C_SUCCESS};font-weight:bold;')

        _fc    = feat_cols or []
        fnames = [c for c in _fc if not c.startswith('FRP=')] or None
        self.push_bundle(results, scaler, feat_cols, ohe,
                         shap_cache=shap_cache, feat_names=fnames)

    def _refresh_model_cb(self):
        """Show ALL models, not just tree-based ones."""
        self.model_cb.blockSignals(True)
        self.model_cb.clear()
        for name in self.results:
            self.model_cb.addItem(name, userData=name)
        self.model_cb.blockSignals(False)
        self._on_model_changed()

    def _on_model_changed(self):
        name = self.model_cb.currentText()
        if not name or name not in self.results:
            self._model_type_lbl.setText('')
            return
        model = self.results[name]['model']
        raw   = getattr(model, 'model', model)
        if hasattr(raw, 'feature_importances_'):
            kind = 'Tree-based  (TreeExplainer - fast)'
        elif hasattr(raw, 'coef_'):
            kind = 'Linear  (LinearExplainer - fast)'
        else:
            kind = 'Black-box  (KernelExplainer - slower)'
        self._model_type_lbl.setText(kind)
        # Invalidate SHAP cache on model switch
        self._shap_vals = None
        self._X_sample  = None
        self._update_step_status()

    def _update_step_status(self):
        has_model = bool(self.results)
        has_data  = (self.X_all is not None
                     and isinstance(self.X_all, np.ndarray)
                     and self.X_all.size > 0)
        has_shap  = self._shap_vals is not None

        if not has_model:
            msg = 'Load a model bundle (Step 1).'
            col = C_TEXT2
        elif not has_data:
            msg = 'Load a background dataset (Step 2).'
            col = C_TEXT2
        elif not has_shap:
            msg = 'Click "Compute SHAP Values" above.'
            col = C_ACCENT
        else:
            msg = f'SHAP ready ({len(self._X_sample)} samples). Pick a plot.'
            col = C_SUCCESS

        self._status_lbl.setText(msg)
        self._status_lbl.setStyleSheet(f'font-size:10px;color:{col};')

    def _browse_data(self):
        if not self.results:
            QMessageBox.information(
                self, 'No Model Loaded',
                'Please load a model bundle first (Step 1):\n'
                '  Click "Load Model Bundle (.frpmdl) ..."')
            return

        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Background Dataset', '',
            'Excel / CSV (*.xlsx *.xls *.csv)')
        if not path:
            return

        try:
            if path.endswith('.csv'):
                raw = pd.read_csv(path)
            elif path.endswith('.xls'):
                raw = pd.read_excel(path, engine='xlrd')
            else:
                raw = pd.read_excel(path)
        except Exception as ex:
            QMessageBox.critical(self, 'File Error', str(ex))
            return

        # Auto-map columns
        try:
            from column_mapping import _auto_map, _build_dataframe
            am = _auto_map(raw.columns.tolist(), df=raw)
            if 'Vexp' not in am:
                am['Vexp'] = raw.columns[0]
            df, _, _ = _build_dataframe(raw, am, drop_no_target=False)
        except Exception:
            df = raw

        feat_cols = self._feat_cols or []
        num_cols  = ([c for c in feat_cols if not c.startswith('FRP=')]
                     or NUM_FEAT_COLS)
        avail     = [c for c in num_cols if c in df.columns]
        if not avail:
            avail = df.select_dtypes(include=[np.number]).columns.tolist()
        if not avail:
            QMessageBox.warning(
                self, 'No Matching Columns',
                f'Expected features: {num_cols}\n'
                f'Available columns: {df.columns.tolist()}')
            return

        X_num = df[avail].values.astype(float)
        for j in range(X_num.shape[1]):
            med = float(np.nanmedian(X_num[:, j]))
            X_num[~np.isfinite(X_num[:, j]), j] = med

        if self._ohe is not None and 'FRP-type' in df.columns:
            try:
                X_cat = self._ohe.transform(df[['FRP-type']].astype(str))
                X_raw = np.hstack([X_num, X_cat])
            except Exception:
                X_raw = X_num
        else:
            X_raw = X_num

        # Apply scaler if available, otherwise use raw values
        if self._scaler is not None:
            try:
                X_sc = self._scaler.transform(X_raw)
            except Exception as ex:
                reply = QMessageBox.question(
                    self, 'Scaler Mismatch',
                    f'Applying bundle scaler failed:\n{ex}\n\n'
                    'Use raw (unscaled) values instead?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply != QMessageBox.Yes:
                    return
                X_sc = X_raw
        else:
            X_sc = X_raw

        self.X_all        = X_sc
        self._shap_vals   = None
        self._X_sample    = None
        self._data_source = 'file'

        fname = os.path.basename(path)
        self._set_data_lbl(
            f'Ready: {len(X_sc):,} rows, {len(avail)} feature(s)', ok=True)
        self._file_lbl.setText(f'{fname}')
        self._update_step_status()

    def _current_model(self):
        name = self.model_cb.currentData() or self.model_cb.currentText()
        if name and name in self.results:
            return self.results[name]['model'], name
        return None, None

    def _run_shap(self):
        if not HAS_SHAP:
            QMessageBox.warning(self, 'SHAP Not Installed',
                                'Run:  pip install shap')
            return

        model, name = self._current_model()
        if model is None:
            QMessageBox.information(self, 'No Model',
                'Load a model bundle first (Step 1).')
            return

        no_data = (self.X_all is None
                   or not isinstance(self.X_all, np.ndarray)
                   or self.X_all.size == 0)
        if no_data:
            reply = QMessageBox.question(
                self, 'No Background Data',
                'No background dataset is loaded (Step 2).\n\n'
                'Load a CSV / Excel file now?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self._browse_data()
            if self.X_all is None or self.X_all.size == 0:
                return

        n        = min(self.max_spin.value(), len(self.X_all))
        idx      = np.random.default_rng(0).choice(len(self.X_all), n, replace=False)
        X_sample = self.X_all[idx]

        self._status_lbl.setText(f'Computing SHAP for {n} samples ...')
        self._status_lbl.setStyleSheet(f'font-size:10px;color:{C_ACCENT};')
        self.compute_btn.setEnabled(False)

        self._worker = ShapWorker(model, X_sample, self.feat_names, name)
        self._worker.log_s.connect(self._status_lbl.setText)
        self._worker.done.connect(self._on_shap_done)
        self._worker.failed.connect(self._on_shap_fail)
        self._worker.start()

    def _on_shap_done(self, sv, X_sample, feat_names):
        self._shap_vals = sv
        self._X_sample  = X_sample
        self.compute_btn.setEnabled(True)
        n_feat = sv.shape[1]
        self.dep_feat_cb.clear()
        self.dep_feat_cb.addItems(feat_names[:n_feat])
        self._update_step_status()
        self._plot_bar()

    def _on_shap_fail(self, msg):
        self.compute_btn.setEnabled(True)
        self._status_lbl.setText(f'[ERROR] {msg}')
        self._status_lbl.setStyleSheet(f'font-size:10px;color:{C_DANGER};')
        QMessageBox.critical(self, 'SHAP Error', msg)

    def _check_shap(self):
        if self._shap_vals is None:
            QMessageBox.information(
                self, 'SHAP Values Required',
                'SHAP values have not been computed yet.\n\n'
                'Click "Compute SHAP Values" in Step 3 first.')
            return False
        return True

    def _finish_plot(self):
        try: self.canvas.fig.tight_layout()
        except Exception: pass
        try: self.canvas.draw()
        except Exception: pass

    def _plot_bar(self):
        if not self._check_shap(): return
        sv     = self._shap_vals
        n_feat = min(sv.shape[1], len(self.feat_names))
        means  = np.abs(sv).mean(axis=0)[:n_feat]
        labels = self.feat_names[:n_feat]
        idx    = np.argsort(means)
        self.canvas.fig.clear()
        ax   = self.canvas.fig.add_subplot(111)
        norm = means[idx] / (means.max() + 1e-9)
        clrs = [plt.cm.Blues(0.3 + 0.7 * v) for v in norm]
        ax.barh(range(n_feat), means[idx], color=clrs, edgecolor='white', height=0.65)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(
            [VAR_LATEX.get(labels[i], labels[i]) for i in idx], fontsize=10)
        ax.set_xlabel(
            r'Mean $|\phi_i|$ — average absolute SHAP value (kN)', fontsize=10)
        ax.set_title(
            f'SHAP Feature Importance — {self.model_cb.currentText()}',
            fontsize=11, fontweight='bold')
        self._finish_plot()

    def _plot_beeswarm(self):
        if not self._check_shap(): return
        sv = self._shap_vals; X = self._X_sample
        n_feat  = min(sv.shape[1], len(self.feat_names))
        means   = np.abs(sv).mean(axis=0)[:n_feat]
        top_k   = min(10, n_feat)
        ordered = np.argsort(means)[::-1][:top_k][::-1]
        labels  = [VAR_LATEX.get(self.feat_names[i], self.feat_names[i])
                   for i in ordered]
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        sc = None
        for rank, fi in enumerate(ordered):
            vals  = sv[:, fi]
            feats = X[:, fi] if fi < X.shape[1] else np.zeros(len(sv))
            fnorm = (feats - feats.min()) / (feats.max() - feats.min() + 1e-9)
            y_pos = (np.full(len(vals), rank)
                     + np.random.default_rng(rank).uniform(-0.25, 0.25, len(vals)))
            sc = ax.scatter(vals, y_pos, c=fnorm, cmap='RdBu_r',
                            s=8, alpha=0.55, vmin=0, vmax=1)
        ax.set_yticks(range(top_k)); ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color='#555555', lw=0.8, ls='--')
        ax.set_xlabel(r'SHAP $\phi_i$ — impact on $V_\mathrm{pred}$ (kN)', fontsize=10)
        ax.set_title(f'SHAP Beeswarm — {self.model_cb.currentText()}',
                     fontsize=11, fontweight='bold')
        if sc is not None:
            cb = self.canvas.fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
            cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(['Low', 'Medium', 'High'])
            cb.set_label('Feature value', fontsize=8, rotation=270, labelpad=12)
            cb.ax.tick_params(labelsize=7, direction='in', length=3)
            cb.outline.set_linewidth(0.6)
        self._finish_plot()

    def _plot_dependence(self):
        if not self._check_shap(): return
        feat_name = self.dep_feat_cb.currentText()
        n_feat    = self._shap_vals.shape[1]
        labels    = self.feat_names[:n_feat]
        if feat_name not in labels: return
        fi   = labels.index(feat_name)
        vals = (self._X_sample[:, fi]
                if fi < self._X_sample.shape[1] else np.zeros(len(self._shap_vals)))
        sv   = self._shap_vals[:, fi]
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        sc = ax.scatter(vals, sv, c=sv, cmap='RdBu_r', s=16, alpha=0.7)
        ax.axhline(0, color='#888888', lw=0.8, ls='--')
        ax.set_xlabel(VAR_LATEX.get(feat_name, feat_name), fontsize=11)
        ax.set_ylabel(
            rf'SHAP $\phi$ for {VAR_LATEX.get(feat_name, feat_name)} (kN)', fontsize=11)
        ax.set_title(f'SHAP Dependence — {VAR_LATEX.get(feat_name, feat_name)}',
                     fontsize=11, fontweight='bold')
        cb = self.canvas.fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label(r'$\phi$ (kN)', fontsize=8, rotation=270, labelpad=12)
        cb.ax.tick_params(labelsize=7, direction='in', length=3)
        cb.outline.set_linewidth(0.6)
        self._finish_plot()

    @staticmethod
    def _is_tree(model):
        return hasattr(getattr(model, 'model', model), 'feature_importances_')

    def _plot_importance(self):
        model, name = self._current_model()
        if model is None: return
        if not self._is_tree(model):
            QMessageBox.information(
                self, 'Not Available',
                'Gini feature importance is only available for tree-based models.\n'
                'Use "SHAP Bar Summary" for all model types.')
            return
        raw = getattr(model, 'model', model)
        imp = raw.feature_importances_
        n_feat = min(len(imp), len(self.feat_names))
        labels = self.feat_names[:n_feat]; idx = np.argsort(imp[:n_feat])
        self.canvas.fig.clear()
        ax   = self.canvas.fig.add_subplot(111)
        norm = imp[idx] / (imp.max() + 1e-9)
        clrs = [plt.cm.Oranges(0.3 + 0.7 * v) for v in norm]
        ax.barh(range(n_feat), imp[idx], color=clrs, edgecolor='white', height=0.65)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(
            [VAR_LATEX.get(labels[i], labels[i]) for i in idx], fontsize=10)
        ax.set_xlabel('Gini impurity-based feature importance', fontsize=11)
        ax.set_title(f'Feature Importance — {name}', fontsize=11, fontweight='bold')
        self._finish_plot()

    def _plot_pdp(self):
        if not HAS_PDP:
            QMessageBox.warning(self, 'Not Available',
                'PartialDependenceDisplay requires scikit-learn >= 1.1.')
            return
        model, name = self._current_model()
        if model is None: return
        if self.X_all is None:
            QMessageBox.warning(self, 'No Data',
                'Load a background dataset first (Step 2).')
            return
        feat_name = self.dep_feat_cb.currentText()
        if feat_name not in self.feat_names: return
        fi  = self.feat_names.index(feat_name)
        raw = getattr(model, 'model', model)

        # Show a placeholder while the worker runs so the user knows
        # something is happening (PDP on large ensemble can take >10 s).
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5,
                f'Computing PDP for {feat_name} …\nThis may take a moment.',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#888888', style='italic')
        self.canvas.draw()

        # Disable the plot buttons while computing to prevent double-clicks.
        for btn in getattr(self, '_plot_btns', []):
            btn.setEnabled(False)

        self._pdp_worker = PdpWorker(raw, self.X_all, fi, self.feat_names)
        self._pdp_meta   = (feat_name, name)

        def _on_pdp_done(disp):
            fn, nm = self._pdp_meta
            self.canvas.fig.clear()
            ax2 = self.canvas.fig.add_subplot(111)
            disp.plot(ax=ax2)
            ax2.set_xlabel(VAR_LATEX.get(fn, fn), fontsize=11)
            ax2.set_ylabel('Partial dependence  (kN)', fontsize=11)
            ax2.set_title(
                f'Partial Dependence — {fn}  ({nm})',
                fontsize=11, fontweight='bold')
            self._finish_plot()
            for btn in getattr(self, '_plot_btns', []):
                btn.setEnabled(True)

        def _on_pdp_failed(msg):
            for btn in getattr(self, '_plot_btns', []):
                btn.setEnabled(True)
            QMessageBox.warning(self, 'PDP Error', msg)

        self._pdp_worker.done.connect(_on_pdp_done)
        self._pdp_worker.failed.connect(_on_pdp_failed)
        self._pdp_worker.start()
