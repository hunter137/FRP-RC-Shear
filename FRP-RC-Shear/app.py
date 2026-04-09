"""
app.py — MainWindow: assembles all tabs and applies global styling.
"""
import sys
import traceback
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QAction,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QDialog, QFileDialog, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette

from config import (
    APP_VERSION,
    C_WIN_BG, C_PANEL_BG, C_ALT_ROW,
    C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_HEADER_BG,
    FEAT_LABELS,
)
from tabs import DataTab, TrainTab, EvalTab, CodeTab, PredictTab, InterpTab
from model_io import ModelIO
from widgets import flat_btn

#  SECTION 14 — MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f'FRP-RC Shear Strength Prediction Platform  v{APP_VERSION}  '
            f'(Stirrup-Free FRP-RC Beams)')
        self.setMinimumSize(1200, 800)
        self.resize(1500, 920)
        self._build_ui()
        self._build_menu()
        self._connect()
        self.statusBar().showMessage(
            f'Ready  (v{APP_VERSION})  —  '
            'Load a database in Model Retraining, or load a saved bundle (Ctrl+L) to begin.')

    def _make_dialog(self, widget, title, w=1300, h=800):
        """Create a persistent popup dialog that houses widget permanently."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(w, h)
        dlg.setWindowFlags(
            Qt.Dialog |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint)
        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(0, 0, 0, 8)
        vl.addWidget(widget, 1)
        row = QHBoxLayout()
        row.setContentsMargins(10, 0, 10, 0)
        row.addStretch()
        cb = flat_btn('Close', width=100)
        cb.clicked.connect(dlg.hide)
        row.addWidget(cb)
        vl.addLayout(row)
        # Override close button — hide, never destroy
        def _hide_close(e, _dlg=dlg):
            e.ignore()
            _dlg.hide()
        dlg.closeEvent = _hide_close
        return dlg

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane{{
                border:none;
                background:{C_WIN_BG};
            }}
            QTabBar{{
                background:{C_WIN_BG};
                border-bottom:2px solid {C_BORDER};
            }}
            QTabBar::tab{{
                font-size:14px;
                font-weight:bold;
                padding:11px 52px;
                background:{C_WIN_BG};
                color:{C_TEXT2};
                border:1px solid {C_BORDER};
                border-bottom:none;
                margin-right:3px;
                border-radius:0px;
                min-width:220px;
            }}
            QTabBar::tab:selected{{
                background:{C_PANEL_BG};
                color:{C_TEXT};
                border-top:3px solid {C_ACCENT};
                border-bottom:2px solid {C_PANEL_BG};
            }}
            QTabBar::tab:hover:!selected{{
                background:{C_ACCENT_BG};
                color:{C_ACCENT};
            }}
        """)

        self.tab_data    = DataTab()
        self.tab_train   = TrainTab()
        self.tab_eval    = EvalTab()
        self.tab_interp  = InterpTab()
        self.tab_predict = PredictTab()

        # Model Retraining has its own CodeTab — receives data from DataTab
        # and ML results after training.
        self.tab_codes         = CodeTab()

        # Prediction tab gets a completely SEPARATE CodeTab — receives data
        # exclusively from Batch Prediction imports.  The two instances never
        # share state, so loading data in one tab cannot overwrite the other.
        self._predict_codes_tab = CodeTab()

        self._interp_dlg = self._make_dialog(
            self.tab_interp, 'Feature Analysis  (SHAP / PDP)',
            1320, 820)
        # Model Retraining → its own Code Comparison dialog
        self._codes_dlg  = self._make_dialog(
            self.tab_codes,  'Design Code Comparison',
            1250, 780)
        # Prediction → its own isolated Code Comparison dialog
        self._predict_codes_dlg = self._make_dialog(
            self._predict_codes_tab,
            'Design Code Comparison',
            1250, 780)
        self._eval_dlg   = self._make_dialog(
            self.tab_eval,   'Model Evaluation',
            1350, 800)
        self._data_dlg   = self._make_dialog(
            self.tab_data,   'Data Manager',
            1280, 780)

        # Inject dialog refs into PredictTab — uses the PREDICTION-only instances
        self.tab_predict._interp_dlg    = self._interp_dlg
        self.tab_predict._codes_dlg     = self._predict_codes_dlg
        self.tab_predict._codes_tab_ref = self._predict_codes_tab
        # Build NSGA-II dialog using the panel created inside PredictTab
        self.tab_predict._nsga_dlg   = self._make_dialog(
            self.tab_predict._nsga_panel,
            'NSGA-II Multi-Objective Optimisation',
            1100, 760)

        self.tabs.addTab(self.tab_predict, 'Prediction')

        self.tabs.addTab(self._build_retrain_panel(), 'Model Retraining')

        self.setCentralWidget(self.tabs)

    def _build_retrain_panel(self):
        """Clean Model Retraining panel with bottom popup buttons."""
        w  = QWidget()
        w.setStyleSheet(f'background:{C_WIN_BG};')
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        db_strip = QWidget()
        db_strip.setStyleSheet(
            f'background:{C_PANEL_BG};'
            f'border-bottom:1px solid {C_BORDER};')
        db_strip.setFixedHeight(52)
        ds = QHBoxLayout(db_strip)
        ds.setContentsMargins(16, 8, 16, 8)
        ds.setSpacing(10)
        db_lbl = QLabel('Database:')
        db_lbl.setStyleSheet(f'font-size:12px;color:{C_TEXT2};')
        ds.addWidget(db_lbl)

        # Mirror path_edit text (DataTab.path_edit stays in its dialog)
        self._path_mirror = QLineEdit()
        self._path_mirror.setReadOnly(True)
        self._path_mirror.setPlaceholderText(
            'No database loaded. Use Browse to select a file.')
        self._path_mirror.setStyleSheet(
            f'background:#FFFFFF;border:1px solid {C_BORDER};'
            f'padding:3px 6px;font-size:12px;')
        self.tab_data.path_edit.textChanged.connect(
            self._path_mirror.setText)
        ds.addWidget(self._path_mirror, 1)

        browse_btn = flat_btn('Browse …', width=95)
        browse_btn.clicked.connect(self.tab_data._browse)
        load_btn   = flat_btn('Load & Map Columns', accent=True)
        load_btn.clicked.connect(self.tab_data._load)
        ds.addWidget(browse_btn)
        ds.addWidget(load_btn)
        vl.addWidget(db_strip)

        vl.addWidget(self.tab_train, 1)

        bar = QWidget()
        bar.setStyleSheet(
            f'background:{C_HEADER_BG};'
            f'border-top:1px solid {C_BORDER};')
        bar.setFixedHeight(62)
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(20, 0, 20, 0)
        bh.setSpacing(10)

        bh.addStretch()

        for label, dlg_attr in [
            ('Data Manager',       '_data_dlg'),
            ('Evaluation', '_eval_dlg'),
            ('Code Comparison', '_codes_dlg'),
        ]:
            def _make_slot(attr):
                def _slot():
                    d = getattr(self, attr)
                    d.show()
                    d.raise_()
                    d.activateWindow()
                return _slot
            btn = flat_btn(label)
            btn.setFixedHeight(40)
            btn.clicked.connect(_make_slot(dlg_attr))
            bh.addWidget(btn)

        vl.addWidget(bar)
        return w

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu('File')
        for text, shortcut, slot in [
            ('Open Database',          'Ctrl+O',
             lambda: self.tabs.setCurrentIndex(1)),
            ('Load Bundle',      'Ctrl+L',
             self.tab_train._load_single),
            ('Save Bundle',      'Ctrl+S',
             self.tab_train._save),
            (None, None, None),
            ('Quit',                     'Ctrl+Q',
             self.close),
        ]:
            if text is None:
                file_m.addSeparator()
            else:
                act = QAction(text, self)
                if shortcut:
                    act.setShortcut(shortcut)
                act.triggered.connect(slot)
                file_m.addAction(act)

        view_m = mb.addMenu('View')
        for text, shortcut, idx in [
            ('Prediction',       'Ctrl+1', 0),
            ('Model Retraining', 'Ctrl+2', 1),
        ]:
            act = QAction(text, self)
            act.setShortcut(shortcut)
            act.triggered.connect(lambda checked, i=idx:
                                  self.tabs.setCurrentIndex(i))
            view_m.addAction(act)

        help_m = mb.addMenu('Help')
        about_act = QAction('About …', self)
        about_act.triggered.connect(self._about)
        help_m.addAction(about_act)

    def _connect(self):
        self.tab_data.sig_data_loaded.connect(self.tab_train.set_data)
        self.tab_data.sig_data_loaded.connect(self.tab_codes.set_data)
        self.tab_data.sig_data_loaded.connect(
            lambda df: self.statusBar().showMessage(
                f'{len(df)} records loaded  ·  '
                f'FRP types: '
                f'{df["FRP-type"].value_counts().to_dict() if "FRP-type" in df.columns else "not mapped"}'
            ))
        self.tab_train.sig_done.connect(self._on_train_done)

    def _on_train_done(self, results, X_te, y_te, X_tr, y_tr,
                       X_all, y_all):
        try:
            self._on_train_done_inner(results, X_te, y_te,
                                      X_tr, y_tr, X_all, y_all)
        except Exception:
            msg = traceback.format_exc()
            QMessageBox.critical(
                self, 'Post-Training Initialisation Error',
                'An error occurred while updating the evaluation tabs '
                'after training completed.\n\n'
                'Training results have been saved; you can still export '
                'the bundle from Model Repository.\n\n'
                f'Details:\n{msg}')

    def _on_train_done_inner(self, results, X_te, y_te, X_tr, y_tr,
                             X_all, y_all):
        feat_names = self.tab_train.feat_cols or FEAT_LABELS

        self.tab_eval.feat_names = feat_names
        # Pass X_all (shap_cache / full scaled data) so that
        # Response Surface Analysis works for bundle-loaded models.
        _X_all_eval = (X_all if isinstance(X_all, np.ndarray)
                       and X_all.size > 0 else None)
        self.tab_eval.load(results, X_te, y_te, X_tr, y_tr,
                           X_all=_X_all_eval)

        self.tab_codes.set_ml(results, y_te)

        # Pass X_all to interpretability — may be a SHAP cache array
        X_for_shap = (X_all if isinstance(X_all, np.ndarray)
                      and X_all.size > 0 else None)
        self.tab_interp.load(results, X_for_shap, feat_names)
        self.tab_interp.dep_feat_cb.clear()
        self.tab_interp.dep_feat_cb.addItems(feat_names)

        self.tab_predict.load_models(
            results, self.tab_train.scaler,
            feat_names,
            getattr(self.tab_train, 'ohe', None),
            shap_cache=getattr(self.tab_train, '_X_all', None))
        # Refresh bundle list in case a new .frpmdl was just saved
        self.tab_predict._refresh_bundle_list()

        if results:
            best = max(
                results,
                key=lambda k: results[k]['te_metrics'].get('R2', -1))
            tm   = results[best]['te_metrics']
            self.statusBar().showMessage(
                f'Training complete — {len(results)} model(s)  ·  '
                f'Best: {best}  '
                f'R² = {tm["R2"]:.4f}  '
                f'r = {tm["r"]:.4f}  '
                f'RMSE = {tm["RMSE"]:.2f} kN  '
                f'·  Switch to Prediction tab to run inference.')

    def _about(self):
        QMessageBox.about(
            self, f'About — v{APP_VERSION}',
            f'<b>FRP-RC Shear Strength Prediction Platform</b><br>'
            f'Stirrup-Free Rectangular Beams<br>'
            f'<hr>'
            f'Version {APP_VERSION}  ·  MIT Licence<br><br>'
            f'<b>Algorithms</b><br>'
            f'GBDT, XGBoost, LightGBM, CatBoost, Random Forest<br><br>'
            f'<b>Search strategies</b><br>'
            f'Manual, Bayesian (Optuna TPE), TLBO '
            f'[Rao et al., 2011], NSGA-II [Deb et al., 2002]<br><br>'
            f'<b>Design codes</b><br>'
            f'GB 50608-2020, ACI 440.1R-15, CSA S806-12, '
            f'BISE (1999), JSCE (1997)<br><br>'
            f'<b>Portability</b><br>'
            f'Any database with arbitrary column names is supported '
            f'via the interactive column-mapping dialog.<br><br>'
            f'<b>Dependencies</b><br>'
            f'PyQt5, matplotlib, numpy, pandas, scikit-learn, scipy,<br>'
            f'xgboost, lightgbm, catboost, joblib, '
            f'optuna, shap, pymoo'
        )

