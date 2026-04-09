import traceback
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QDialog, QDialogButtonBox, QFileDialog, QFrame, QGroupBox,
    QPushButton, QRadioButton, QMessageBox,
    QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    HAS_CUDA, CUDA_DEVICE_NAME, APP_VERSION,
)
from widgets import flat_btn, panel
from optimization import NSGA2_OBJECTIVES, _GPU_CAPABLE

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

from config import (
    APP_VERSION, _SHAP_BUNDLE_SAMPLES,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT,
    C_ACCENT, C_ACCENT_LT, C_ACCENT_BG,
    C_WIN_BG, C_PANEL_BG, C_ALT_ROW, C_HEADER_BG,
    C_SUCCESS, C_SUCCESS_BG, C_DANGER,
    HAS_XGB, HAS_LGB, HAS_CAT, HAS_OPTUNA, HAS_PYMOO, HAS_TORCH,
    HAS_CUDA, HAS_SHAP, CUDA_DEVICE_NAME,
    NUM_FEAT_COLS, FRP_TYPES, FEAT_LABELS,
    ALGO_COLORS, CODE_COLORS,
)
from optimization import (
    _ps_gbdt, _ps_xgb, _ps_lgb, _ps_cat, _ps_rf,
    _factory_for, tlbo_optimize, _optuna_optimize, nsga2_optimize,
    PARAM_SPACES, NSGA2_OBJECTIVES, _GPU_CAPABLE,
)

class BundleFolderDialog(QDialog):
    def __init__(self, folder, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select Model Packages to Import')
        self.resize(540, 360)
        self._folder = folder
        self._files  = sorted([f for f in os.listdir(folder)
                                if f.lower().endswith('.frpmdl')])
        vl = QVBoxLayout(self)
        info = QLabel(
            f'Found {len(self._files)} bundle(s) in:\n{folder}')
        info.setStyleSheet(f'font-size:11px;color:{C_TEXT2};')
        vl.addWidget(info)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        for f in self._files:
            item = QListWidgetItem(f)
            item.setCheckState(Qt.Checked)
            self._list.addItem(item)
        vl.addWidget(self._list)
        row = QHBoxLayout()
        for lbl, fn in [('Select All', self._check_all),
                        ('Clear All',  self._uncheck_all)]:
            b = flat_btn(lbl); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(); vl.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        vl.addWidget(bb)

    def _check_all(self):
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)
    def _uncheck_all(self):
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Unchecked)
    def selected_paths(self):
        return [os.path.join(self._folder, self._list.item(i).text())
                for i in range(self._list.count())
                if self._list.item(i).checkState() == Qt.Checked]

class Nsga2ObjectiveDialog(QDialog):
    """
    Let the user pick exactly 2 objectives for the NSGA-II bi-objective search.
    Each objective can be any metric produced by calc_metrics(), with its
    optimisation direction (maximise / minimise / closest-to-1) shown.
    """
    # Ordered list so the combo keeps a consistent display order
    _OBJ_KEYS = ['R2', 'safety_pct', 'RMSE', 'MAE', 'MAPE', 'cov', 'mean_ratio']
    _DIRECTION_ICON = {'maximize': '[Max]', 'minimize': '[Min]', 'ratio': '[~1.0]'}

    def __init__(self, current_objectives=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('NSGA-II Objective Setup')
        self.setFixedSize(460, 320)
        self._objectives = list(current_objectives) if current_objectives else ['R2', 'safety_pct']
        self._build_ui()

    def _combo_label(self, key):
        lbl, direction = NSGA2_OBJECTIVES.get(key, (key, 'minimize'))
        icon = self._DIRECTION_ICON.get(direction, '')
        return f'{icon}   {lbl}'

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setSpacing(14)
        vl.setContentsMargins(20, 16, 20, 16)

        desc = QLabel(
            'NSGA-II searches a <b>Pareto-optimal front</b> by simultaneously optimising '
            'two competing objectives. Select any two metrics; '
            'the optimisation direction is applied automatically.')
        desc.setWordWrap(True)
        desc.setStyleSheet(f'font-size:11px; color:{C_TEXT};')
        vl.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setSpacing(8)

        self._combos = []
        for row, (label_txt, default_key) in enumerate([
                ('Objective 1:', self._objectives[0]),
                ('Objective 2:', self._objectives[1]),
        ]):
            lbl = QLabel(label_txt)
            lbl.setStyleSheet('font-size:11px; font-weight:bold;')
            combo = QComboBox()
            combo.setStyleSheet('font-size:11px;')
            for key in self._OBJ_KEYS:
                combo.addItem(self._combo_label(key), userData=key)
            idx = self._OBJ_KEYS.index(default_key) if default_key in self._OBJ_KEYS else 0
            combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(self._validate)
            grid.addWidget(lbl,   row, 0, Qt.AlignRight)
            grid.addWidget(combo, row, 1)
            self._combos.append(combo)
        vl.addLayout(grid)

        self._warn_lbl = QLabel('')
        self._warn_lbl.setStyleSheet('font-size:10px; color:#C0392B;')
        vl.addWidget(self._warn_lbl)

        legend = QLabel('  [Max] = Maximise     [Min] = Minimise     [~1.0] = Closest to 1.0')
        legend.setStyleSheet(f'font-size:10px; color:{C_TEXT2};')
        vl.addWidget(legend)

        vl.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._accept)
        btn_box.rejected.connect(self.reject)
        self._ok_btn = btn_box.button(QDialogButtonBox.Ok)
        vl.addWidget(btn_box)

        self._validate()

    def _validate(self):
        k1 = self._combos[0].currentData()
        k2 = self._combos[1].currentData()
        conflict = (k1 == k2)
        self._warn_lbl.setText(
            'Warning: both objectives are identical. Please select two different metrics.'
            if conflict else '')
        self._ok_btn.setEnabled(not conflict)

    def _accept(self):
        self._objectives = [c.currentData() for c in self._combos]
        self.accept()

    def get_objectives(self):
        return list(self._objectives)

class TlboSettingsDialog(QDialog):
    """
    Configure Teaching-Learning Based Optimisation search parameters.

    Two modes:
      Auto   – n_pop / n_iter derived from the global Max. trials spinbox
               (same as the original behaviour).
      Manual – user sets n_pop and n_iter directly; total evaluations are
               shown live so the user can gauge the computational cost.

    Total evaluations formula:  n_pop  +  n_iter × 2 × n_pop
    """
    def __init__(self, current_settings=None, budget_hint=50, parent=None):
        super().__init__(parent)
        self.setWindowTitle('TLBO Search Configuration')
        self.setFixedSize(420, 310)
        s = current_settings or {}
        self._mode   = s.get('mode',   'auto')
        self._n_pop  = s.get('n_pop',  max(5, budget_hint // 5))
        self._n_iter = s.get('n_iter', max(3, budget_hint // 5))
        self._budget = budget_hint
        self._build_ui()

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setSpacing(10)
        vl.setContentsMargins(20, 16, 20, 14)

        desc = QLabel(
            '<b>TLBO</b> alternates a <i>teaching</i> phase (population moves '
            'toward the best member) and a <i>learning</i> phase (pairwise '
            'improvement). Configure the population and iteration budget below.')
        desc.setWordWrap(True)
        desc.setStyleSheet(f'font-size:11px; color:{C_TEXT};')
        vl.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep)

        self._rb_auto   = QRadioButton('Automatic  (derived from iteration budget)')
        self._rb_manual = QRadioButton('User-Defined Specification')
        for rb in (self._rb_auto, self._rb_manual):
            rb.setStyleSheet('font-size:11px;')
        self._rb_auto.setToolTip(
            f'n_pop = max(5, budget÷5)\nn_iter = max(3, budget÷5)\n'
            f'With current budget={self._budget}: '
            f'n_pop={max(5, self._budget//5)}, '
            f'n_iter={max(3, self._budget//5)}, '
            f'total={self._auto_total()} evals')
        self._rb_manual.setToolTip('Set population size and iterations directly.')
        (self._rb_auto if self._mode == 'auto' else self._rb_manual).setChecked(True)
        # NOTE: connect AFTER setChecked so the signal doesn't fire before
        # _total_lbl is constructed (connected below, post widget creation).
        vl.addWidget(self._rb_auto)
        vl.addWidget(self._rb_manual)

        self._manual_frame = QWidget()
        mg = QGridLayout(self._manual_frame)
        mg.setContentsMargins(20, 4, 0, 0)
        mg.setSpacing(6)

        for row, (label, attr, lo, hi, tip) in enumerate([
            ('Population Size:',  '_sp_pop',  3, 200,
             'Number of candidate solutions evaluated in parallel each iteration.'),
            ('Iterations:',      '_sp_iter', 1, 200,
             'Number of teaching + learning cycles.'),
        ]):
            lbl = QLabel(label)
            lbl.setStyleSheet('font-size:11px;')
            sp  = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(self._n_pop if attr == '_sp_pop' else self._n_iter)
            sp.setFixedWidth(70)
            sp.setStyleSheet('font-size:11px;')
            sp.setToolTip(tip)
            sp.valueChanged.connect(self._refresh_total)
            setattr(self, attr, sp)
            mg.addWidget(lbl, row, 0)
            mg.addWidget(sp,  row, 1)

        # Live total-evals display
        self._total_lbl = QLabel()
        self._total_lbl.setStyleSheet(
            f'font-size:11px; color:{C_TEXT2}; font-style:italic;')
        mg.addWidget(self._total_lbl, 2, 0, 1, 2)
        vl.addWidget(self._manual_frame)

        # Safe to connect now — _total_lbl exists
        self._rb_auto.toggled.connect(self._on_mode)
        self._on_mode()          # apply initial enable state + total label

        vl.addStretch()
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        vl.addWidget(btn_box)

    def _auto_total(self):
        p = max(5, self._budget // 5)
        i = max(3, self._budget // 5)
        return p + i * 2 * p

    def _refresh_total(self):
        p = self._sp_pop.value()
        i = self._sp_iter.value()
        total = p + i * 2 * p
        self._total_lbl.setText(
            f'Total evaluations:  {p} + {i} × 2 × {p}  =  {total}')

    def _on_mode(self):
        manual = self._rb_manual.isChecked()
        self._manual_frame.setEnabled(manual)
        if manual:
            self._refresh_total()
        else:
            if hasattr(self, '_total_lbl'):
                p = max(5, self._budget // 5)
                i = max(3, self._budget // 5)
                self._total_lbl.setText(
                    f'Auto total:  {p} + {i} × 2 × {p}  =  {self._auto_total()}')

    def get_settings(self):
        manual = self._rb_manual.isChecked()
        return {
            'mode':   'manual' if manual else 'auto',
            'n_pop':  self._sp_pop.value()  if manual else max(5, self._budget // 5),
            'n_iter': self._sp_iter.value() if manual else max(3, self._budget // 5),
        }

class BayesianSettingsDialog(QDialog):
    """
    Configure the Optuna TPE Bayesian search.

    Exposed knobs:
      n_startup_trials – how many random trials run before the TPE model
                         activates (warm-up exploration).
      multivariate     – whether TPE models correlations between parameters
                         (more accurate but slower per trial).
    """
    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Bayesian Optimisation Settings')
        self.setFixedSize(440, 290)
        s = current_settings or {}
        self._n_startup   = s.get('n_startup_trials', 10)
        self._multivariate = s.get('multivariate',    False)
        self._build_ui()

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setSpacing(12)
        vl.setContentsMargins(20, 16, 20, 14)

        desc = QLabel(
            '<b>Tree-structured Parzen Estimator (TPE)</b> builds a probabilistic '
            'model of the objective and proposes promising hyperparameter '
            'combinations.  The settings below control exploration vs. exploitation.')
        desc.setWordWrap(True)
        desc.setStyleSheet(f'font-size:11px; color:{C_TEXT};')
        vl.addWidget(desc)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1)

        # n_startup_trials
        lbl_s = QLabel('Warm-Up Trials (random):')
        lbl_s.setStyleSheet('font-size:11px; font-weight:bold;')
        self._sp_startup = QSpinBox()
        self._sp_startup.setRange(1, 100)
        self._sp_startup.setValue(self._n_startup)
        self._sp_startup.setFixedWidth(70)
        self._sp_startup.setStyleSheet('font-size:11px;')
        tip_s = QLabel(
            'Random trials before the TPE model activates.\n'
            'Higher values produce more exploration and better global coverage.\n'
            'Recommended: 10–20 % of total budget.')
        tip_s.setStyleSheet(f'font-size:10px; color:{C_TEXT2};')
        tip_s.setWordWrap(True)
        grid.addWidget(lbl_s,           0, 0, Qt.AlignTop | Qt.AlignRight)
        grid.addWidget(self._sp_startup, 0, 1, Qt.AlignTop)
        grid.addWidget(tip_s,            1, 1)

        # multivariate
        lbl_m = QLabel('Multivariate Modelling:')
        lbl_m.setStyleSheet('font-size:11px; font-weight:bold;')
        self._cb_multi = QCheckBox('Enable  (model parameter correlations)')
        self._cb_multi.setChecked(self._multivariate)
        self._cb_multi.setStyleSheet('font-size:11px;')
        tip_m = QLabel(
            'Fits a joint model across all parameters instead of independent '
            'univariate models.  More accurate for spaces where parameters '
            'interact (e.g. learning_rate ↔ n_estimators), but slightly '
            'slower per trial.')
        tip_m.setStyleSheet(f'font-size:10px; color:{C_TEXT2};')
        tip_m.setWordWrap(True)
        grid.addWidget(lbl_m,        2, 0, Qt.AlignTop | Qt.AlignRight)
        grid.addWidget(self._cb_multi, 2, 1, Qt.AlignTop)
        grid.addWidget(tip_m,          3, 1)

        vl.addLayout(grid)
        vl.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        vl.addWidget(btn_box)

    def get_settings(self):
        return {
            'n_startup_trials': self._sp_startup.value(),
            'multivariate':     self._cb_multi.isChecked(),
        }

class PreTrainingDialog(QDialog):
    """
    Shown when the user clicks Execute Training.

    Presents a read-only summary of the current training configuration
    (selected models, search strategy, data split) and lets the user
    choose the compute device (CPU or GPU/CUDA) before committing.
    """
    def __init__(self, summary: dict, last_use_gpu: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Train Models')
        self.setFixedSize(460, 400)
        self._use_gpu = last_use_gpu
        self._build_ui(summary)

    def _build_ui(self, s: dict):
        vl = QVBoxLayout(self)
        vl.setSpacing(14)
        vl.setContentsMargins(22, 18, 22, 16)

        title = QLabel('Training Configuration')
        title.setStyleSheet(
            f'font-size:13px; font-weight:bold; color:{C_TEXT};')
        vl.addWidget(title)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep1)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)

        def _row(r, label, value, highlight=False):
            lbl = QLabel(label)
            lbl.setStyleSheet(
                f'font-size:11px; font-weight:bold; color:{C_TEXT2};')
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet(
                f'font-size:11px; color:{C_ACCENT if highlight else C_TEXT};')
            grid.addWidget(lbl, r, 0, Qt.AlignTop | Qt.AlignRight)
            grid.addWidget(val, r, 1, Qt.AlignTop)

        algo_str = s.get('algorithms', 'None selected')
        strategy = s.get('strategy', 'Fixed Hyperparameters')
        budget   = s.get('budget', 50)
        cv       = s.get('cv_folds', 10)
        test_pct = s.get('test_pct', 20)
        n_models = s.get('n_models', 0)

        strategy_detail = strategy
        if strategy not in ('none', 'Fixed Hyperparameters'):
            strategy_detail = f'{strategy}  (max {budget} iterations)'

        _row(0, 'Models selected:',   f'{n_models} model(s)  —  {algo_str}')
        _row(1, 'Search strategy:',   strategy_detail)
        _row(2, 'Cross-validation:',  f'{cv}-fold')
        _row(3, 'Train / test split:', f'{100 - test_pct}% training  /  {test_pct}% test')

        vl.addLayout(grid)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f'color:{C_BORDER_LT};')
        vl.addWidget(sep2)

        dev_title = QLabel('Compute Device')
        dev_title.setStyleSheet(
            f'font-size:12px; font-weight:bold; color:{C_TEXT};')
        vl.addWidget(dev_title)

        dev_row = QHBoxLayout()
        dev_row.setSpacing(18)

        self._rb_cpu = QRadioButton('CPU')
        self._rb_cpu.setChecked(not self._use_gpu)
        self._rb_cpu.setStyleSheet('font-size:11px;')
        self._rb_cpu.setToolTip('Use the central processing unit for all models.')
        dev_row.addWidget(self._rb_cpu)

        gpu_label = 'GPU  (CUDA)'
        self._rb_gpu = QRadioButton(gpu_label)
        self._rb_gpu.setChecked(self._use_gpu and HAS_CUDA)
        self._rb_gpu.setEnabled(HAS_CUDA)
        self._rb_gpu.setStyleSheet('font-size:11px;')
        if HAS_CUDA:
            self._rb_gpu.setToolTip(
                f'Detected device: {CUDA_DEVICE_NAME}\n'
                'Applies to XGBoost, LightGBM, and CatBoost.\n'
                'Other models run on CPU regardless.')
        else:
            self._rb_gpu.setToolTip(
                'No CUDA-capable GPU detected.\n'
                'Install the CUDA toolkit and a compatible driver to enable.')
        dev_row.addWidget(self._rb_gpu)
        dev_row.addStretch()
        vl.addLayout(dev_row)

        dev_note = QLabel(
            CUDA_DEVICE_NAME if HAS_CUDA
            else 'No CUDA device detected on this system.')
        dev_note.setStyleSheet(
            f'font-size:10px; color:{C_TEXT2}; font-style:italic;')
        vl.addWidget(dev_note)

        if HAS_CUDA:
            # Build per-model device list based on current algo selection
            algos_in_cfg = list(s.get('algorithms', '').split(', '))
            gpu_algos  = [a for a in algos_in_cfg if a in _GPU_CAPABLE]
            cpu_algos  = [a for a in algos_in_cfg if a not in _GPU_CAPABLE]
            lines = []
            if gpu_algos:
                lines.append(f'[GPU]  {" | ".join(gpu_algos)}')
            if cpu_algos:
                lines.append(f'[CPU]  {" | ".join(cpu_algos)}  (sklearn — CPU only)')
            gpu_scope = QLabel('\n'.join(lines) if lines else
                'GPU acceleration applies to XGBoost, LightGBM, CatBoost.')
            gpu_scope.setStyleSheet(
                f'font-size:10px; color:{C_TEXT2}; '
                f'background:#F0F6FF; padding:5px 8px; '
                f'border:1px solid {C_BORDER_LT};')
            gpu_scope.setWordWrap(True)
            vl.addWidget(gpu_scope)

        vl.addStretch()

        btn_box = QDialogButtonBox()
        ok_btn  = btn_box.addButton('Confirm & Run', QDialogButtonBox.AcceptRole)
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

    def use_gpu(self) -> bool:
        return self._rb_gpu.isChecked() and HAS_CUDA

class _TrainSummaryDialog(QDialog):
    """
    Auto-popup after training completes.
    Ranked metrics table + model selection for bundle export.
    """

    def __init__(self, results, opt_strategy, feat_cols,
                 save_fn=None, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.Dialog |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint)
        self.setWindowTitle('Training Summary')
        self.resize(980, 540)
        self._results      = results
        self._opt_strategy = opt_strategy
        self._feat_cols    = feat_cols
        self._save_fn      = save_fn
        self._save_checks  = {}
        self._build_ui()

    def _build_ui(self):
        vl = QVBoxLayout(self)
        vl.setSpacing(8)
        vl.setContentsMargins(12, 12, 12, 12)

        sorted_names = sorted(
            self._results,
            key=lambda n: self._results[n]['te_metrics'].get('R2', -99),
            reverse=True)
        best = sorted_names[0] if sorted_names else '—'

        info_lbl = QLabel(
            f'<b>Best model:</b>  {best}  &nbsp;|&nbsp;  '
            f'<b>Strategy:</b>  {self._opt_strategy}  &nbsp;|&nbsp;  '
            f'<b>Features ({len(self._feat_cols)}):</b>  '
            f'{", ".join(self._feat_cols)}')
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet(f'font-size:11px;color:{C_TEXT2};')
        vl.addWidget(info_lbl)

        hdrs = ['Rank', 'Model', 'R² (test)', 'r (test)',
                'RMSE (kN)', 'MAE (kN)', 'MAPE (%)',
                'k̄', 'CoV', 'CV R² (mean ± S.D.)']
        tbl = QTableWidget()
        tbl.setColumnCount(len(hdrs))
        tbl.setRowCount(len(sorted_names))
        tbl.setHorizontalHeaderLabels(hdrs)
        tbl.verticalHeader().setVisible(False)
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.setStyleSheet('font-size:12px;')
        tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for c in [0] + list(range(2, len(hdrs))):
            tbl.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)

        bold     = QFont(); bold.setBold(True)
        green_bg = QColor('#D5ECD4')

        def fv(m, k, fmt='.4f'):
            v = m.get(k)
            return f'{v:{fmt}}' if isinstance(v, float) and not np.isnan(v) else '—'

        for row, name in enumerate(sorted_names):
            res = self._results[name]
            te  = res['te_metrics']
            cv_str = (
                f'{res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}'
                if not np.isnan(res.get('cv_mean', float('nan'))) else '—')
            vals = [str(row + 1), name,
                    fv(te,'R2'), fv(te,'r'), fv(te,'RMSE','.2f'),
                    fv(te,'MAE','.2f'), fv(te,'MAPE','.2f'),
                    fv(te,'mean_ratio'), fv(te,'cov'), cv_str]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                it.setTextAlignment(
                    Qt.AlignLeft | Qt.AlignVCenter if col == 1
                    else Qt.AlignCenter)
                if row == 0:
                    it.setBackground(green_bg)
                    it.setFont(bold)
                tbl.setItem(row, col, it)

        vl.addWidget(tbl)

        if self._save_fn is not None:
            save_grp = QGroupBox('Save Model Bundle')
            save_grp.setStyleSheet(
                f'QGroupBox{{font-weight:bold;font-size:11px;'
                f'border:1px solid {C_BORDER};border-radius:4px;'
                f'margin-top:6px;padding:8px 8px 6px 8px;}}')
            sg = QVBoxLayout(save_grp)
            sg.setSpacing(5)

            hint = QLabel(
                'Select the models to include in the exported bundle. '
                'The best-performing model is pre-selected by default.')
            hint.setStyleSheet(f'font-size:10px;color:{C_TEXT2};')
            hint.setWordWrap(True)
            sg.addWidget(hint)

            cb_row = QHBoxLayout()
            cb_row.setSpacing(14)
            for name in sorted_names:
                cb = QCheckBox(name)
                cb.setChecked(name == sorted_names[0])
                cb.setStyleSheet('font-size:11px;')
                self._save_checks[name] = cb
                cb_row.addWidget(cb)
            cb_row.addStretch()
            sg.addLayout(cb_row)

            ctrl_row = QHBoxLayout()
            ctrl_row.setSpacing(6)
            all_btn  = flat_btn('Select All')
            none_btn = flat_btn('Deselect All')
            all_btn.setFixedHeight(24)
            none_btn.setFixedHeight(24)
            all_btn.clicked.connect(
                lambda: [c.setChecked(True)  for c in self._save_checks.values()])
            none_btn.clicked.connect(
                lambda: [c.setChecked(False) for c in self._save_checks.values()])
            ctrl_row.addWidget(all_btn)
            ctrl_row.addWidget(none_btn)
            ctrl_row.addStretch()
            save_btn = flat_btn('Save Selected', accent=True)
            save_btn.setFixedHeight(28)
            save_btn.clicked.connect(self._save_selected)
            ctrl_row.addWidget(save_btn)
            sg.addLayout(ctrl_row)
            vl.addWidget(save_grp)

        btn_row = QHBoxLayout()
        exp_btn   = flat_btn('Export Performance Table (CSV)')
        exp_btn.setFixedHeight(30)
        exp_btn.clicked.connect(lambda: self._export(sorted_names))
        close_btn = flat_btn('Close')
        close_btn.setFixedHeight(30)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(exp_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        vl.addLayout(btn_row)

    def _save_selected(self):
        selected = [n for n, cb in self._save_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, 'No Selection',
                                'Please select at least one model.')
            return
        self._save_fn(selected)

    def _export(self, sorted_names):
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Training Results',
            'training_results.csv', 'CSV files (*.csv)')
        if not path:
            return
        rows = []
        for rank, name in enumerate(sorted_names, 1):
            res = self._results[name]
            te  = res['te_metrics']
            rows.append({
                'rank': rank, 'model': name,
                'R2_test':    te.get('R2'),   'r_test':     te.get('r'),
                'RMSE_kN':    te.get('RMSE'), 'MAE_kN':     te.get('MAE'),
                'MAPE_pct':   te.get('MAPE'), 'k_bar':      te.get('mean_ratio'),
                'CoV':        te.get('cov'),  'safety_pct': te.get('safety_pct'),
                'CV_R2_mean': res.get('cv_mean'), 'CV_R2_std': res.get('cv_std'),
            })
        pd.DataFrame(rows).to_csv(path, index=False)
        QMessageBox.information(self, 'Saved', f'Results saved to:\n{path}')

