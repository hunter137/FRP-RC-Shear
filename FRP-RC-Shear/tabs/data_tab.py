"""Data management tab — load, preview, and inspect databases."""
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QDialog, QSizePolicy, QCheckBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtCore import pyqtSignal

from config import (
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_ACCENT,
    C_PANEL_BG, C_HEADER_BG, C_ALT_ROW,
    NUM_FEAT_COLS, FRP_TYPES, FEAT_LABELS,
    VAR_PLAIN, PRED_COLS,
)
from widgets import flat_btn, panel, _stat_textbox, MplCanvas
from column_mapping import _VAR_CATALOGUE, _auto_map, _build_dataframe, ColumnMappingDialog
from .dist_dialog import DistributionDialog

class DataTab(QWidget):
    sig_data_loaded = pyqtSignal(pd.DataFrame)

    def __init__(self):
        super().__init__()
        self.df          = None
        self._raw_df     = None
        self.col_mapping = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        # Section heading
        hdr = QLabel('Data Management')
        hdr.setStyleSheet(
            f'font-size:13px;font-weight:bold;color:{C_TEXT};'
            f'padding:4px 0px 2px 0px;'
            f'border-bottom:1px solid {C_BORDER_LT};')
        root.addWidget(hdr)

        # File selector row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel('Database file:'))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            'Select a .xls / .xlsx / .csv specimen database  '
            '(any column format accepted — '
            'use the mapping dialog to connect your data)')
        self.path_edit.setReadOnly(True)
        browse_btn = flat_btn('Browse …')
        load_btn   = flat_btn('Load & Map Columns', accent=True)
        browse_btn.clicked.connect(self._browse)
        load_btn.clicked.connect(self._load)
        file_row.addWidget(self.path_edit)
        file_row.addWidget(browse_btn)
        file_row.addWidget(load_btn)
        root.addLayout(file_row)

        portability_note = _stat_textbox(
            'Portability: this platform accepts any experimental database '
            'regardless of column naming.  After loading, an interactive '
            'column-mapping dialog connects your variable names to the '
            'required model inputs.  The mapping can be edited at any time.')
        root.addWidget(portability_note)

        splitter = QSplitter(Qt.Horizontal)

        # Left: data table
        tbl_grp = panel('Data preview  (first 300 rows)')
        tl      = QVBoxLayout(tbl_grp)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet('font-size:11px;')
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        tl.addWidget(self.table)

        tbl_btn_row = QHBoxLayout()
        self._dist_btn = flat_btn('Distribution Analysis …')
        self._dist_btn.setFixedHeight(32)
        self._dist_btn.setToolTip(
            'Open distribution analysis: histogram + KDE for each variable')
        self._dist_btn.clicked.connect(self._show_dist_dialog)
        self._dist_dlg = None   # lazy-created DistributionDialog
        tbl_btn_row.addStretch()
        tbl_btn_row.addWidget(self._dist_btn)
        tl.addLayout(tbl_btn_row)

        splitter.addWidget(tbl_grp)

        # Right panel
        right = QWidget()
        rl    = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        rl.setSpacing(6)

        stat_grp = panel('Descriptive statistics')
        sl       = QVBoxLayout(stat_grp)
        self._info_lbl = QLabel('No data loaded.')
        self._info_lbl.setAlignment(Qt.AlignCenter)
        self._info_lbl.setStyleSheet(
            f'color:{C_TEXT2};font-size:12px;'
            f'padding:3px;')
        sl.addWidget(self._info_lbl)
        stat_btn = flat_btn('View Full Statistics …')
        stat_btn.clicked.connect(self._show_stat_dialog)
        sl.addWidget(stat_btn)
        rl.addWidget(stat_grp)

        # Hidden: stat data stored for the dialog
        self._stat_df = None

        map_grp = panel('Active column mapping')
        ml      = QVBoxLayout(map_grp)
        ml.setContentsMargins(4, 22, 4, 4)
        self._map_table = QTableWidget()
        self._map_table.setColumnCount(2)
        self._map_table.setHorizontalHeaderLabels(['Variable', 'Mapped to'])
        self._map_table.verticalHeader().setVisible(False)
        self._map_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._map_table.setAlternatingRowColors(True)
        self._map_table.setSelectionMode(QTableWidget.NoSelection)
        self._map_table.setStyleSheet('font-size:12px;')
        self._map_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self._map_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self._map_table.setMaximumHeight(230)
        ml.addWidget(self._map_table)
        self._map_status = QLabel('')
        self._map_status.setStyleSheet(f'font-size:11px;color:{C_TEXT2};padding:2px;')
        ml.addWidget(self._map_status)
        remap_btn = flat_btn('Edit Column Mapping …')
        remap_btn.clicked.connect(self._remap)
        ml.addWidget(remap_btn)
        rl.addWidget(map_grp)

        pie_grp = panel('FRP type distribution')
        pl      = QVBoxLayout(pie_grp)
        self.pie_canvas = MplCanvas(width=3.2, height=2.5, dpi=100)
        pl.addWidget(self.pie_canvas)
        rl.addWidget(pie_grp)
        rl.addStretch()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)   # data table gets 3/4
        splitter.setStretchFactor(1, 1)   # right panel gets 1/4
        splitter.setSizes([780, 380])
        root.addWidget(splitter)

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self, 'Select Database File', '',
            'Excel / CSV (*.xls *.xlsx *.csv);;All (*)')
        if p:
            self.path_edit.setText(p)

    def _load(self):
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, 'No File Selected',
                                'Please select a database file first.')
            return
        try:
            if path.endswith('.csv'):
                raw = pd.read_csv(path)
            elif path.endswith('.xls'):
                raw = pd.read_excel(path, engine='xlrd')
            else:
                raw = pd.read_excel(path)
        except Exception as e:
            QMessageBox.critical(self, 'File Read Error', str(e))
            return

        self._raw_df = raw
        auto_map     = _auto_map(raw.columns.tolist(), df=raw)
        dlg          = ColumnMappingDialog(
            raw.columns.tolist(), auto_map, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        self.col_mapping = dlg.get_mapping()
        self._apply_mapping(raw, self.col_mapping)

    def _remap(self):
        if self._raw_df is None:
            QMessageBox.warning(self, 'No Data',
                                'Load a database file first.')
            return
        dlg = ColumnMappingDialog(
            self._raw_df.columns.tolist(), self.col_mapping, self)
        if dlg.exec_() == QDialog.Accepted:
            self.col_mapping = dlg.get_mapping()
            self._apply_mapping(self._raw_df, self.col_mapping)

    def _apply_mapping(self, raw, mapping):
        try:
            df_all,   n_valid, n_total = _build_dataframe(raw, mapping, False)
            df_train, _,       _       = _build_dataframe(raw, mapping, True)
        except Exception as e:
            QMessageBox.critical(self, 'Column Extraction Failed', str(e))
            return

        n_drop = n_total - n_valid
        if n_valid == 0:
            QMessageBox.critical(
                self, 'No Valid Records',
                f'Zero records with a positive V_exp were found '
                f'(total rows: {n_total}).\n'
                f'Verify that the correct column is mapped to V_exp.')
            return

        if n_drop > 0 and n_valid / n_total < 0.5:
            ans = QMessageBox.warning(
                self, 'Low Record Yield',
                f'Only {n_valid} of {n_total} rows have a valid V_exp > 0.\n'
                f'{n_drop} rows will be excluded.  Continue?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        self.df = df_all
        self._fill_table(df_all)
        self._fill_stats(df_all)
        self._update_map_label(mapping, n_valid, n_total)
        self._plot_pie(df_all)
        self.sig_data_loaded.emit(df_train)
        QMessageBox.information(
            self, 'Database Loaded',
            f'{n_valid} valid records loaded '
            f'({n_drop} rows excluded).\n'
            f'{len(mapping)} columns mapped.')

    # Display-friendly column headers (Unicode subscripts where available)
    # Table-header / stat-dialog display names — sourced from VAR_PLAIN
    _COL_DISPLAY = {
        'Vexp(kN)': VAR_PLAIN['Vexp(kN)'],
        'd(mm)':    VAR_PLAIN['d(mm)'],
        'b(mm)':    VAR_PLAIN['b(mm)'],
        "f`c(Mpa)": VAR_PLAIN["f`c(Mpa)"],
        'ρf(%)':    VAR_PLAIN['ρf(%)'],
        'Ef(GPa)':  VAR_PLAIN['Ef(GPa)'],
        'a/d':      VAR_PLAIN['a/d'],
        'FRP-type': 'FRP type',
    }

    def _fill_table(self, df):
        # Only show prediction-relevant columns
        keep = ['Vexp(kN)'] + list(NUM_FEAT_COLS) + ['FRP-type']
        show_cols = [c for c in keep if c in df.columns]
        display_headers = [self._COL_DISPLAY.get(c, c) for c in show_cols]
        n = min(300, len(df))
        self.table.setRowCount(n)
        self.table.setColumnCount(len(show_cols))
        self.table.setHorizontalHeaderLabels(display_headers)
        for i in range(n):
            for j, col in enumerate(show_cols):
                v   = df.at[df.index[i], col]
                txt = f'{v:.3f}' if isinstance(v, float) else str(v)
                it  = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(i, j, it)

    def _fill_stats(self, df):
        """Store data for the statistics popup and update summary label."""
        self._stat_df = df
        keep = ['Vexp(kN)'] + list(NUM_FEAT_COLS)
        n_feat = sum(1 for c in keep if c in df.columns)
        self._info_lbl.setText(
            f'{len(df)} records  ·  {n_feat} prediction variables')

    def _show_dist_dialog(self):
        """Open the Distribution Analysis dialog for the loaded dataset."""
        if self.df is None:
            QMessageBox.information(
                self, 'No Data', 'Load a database first.')
            return
        # Lazy-create; destroy and recreate whenever df changes so that
        # the variable list always reflects the current dataset.
        if self._dist_dlg is None or self._dist_dlg._df is not self.df:
            self._dist_dlg = DistributionDialog(self.df, parent=self)
        self._dist_dlg.show()
        self._dist_dlg.raise_()
        self._dist_dlg.activateWindow()

    def _show_stat_dialog(self):
        """Open a popup dialog — only prediction-relevant variables."""
        df = self._stat_df
        if df is None:
            QMessageBox.information(self, 'No Data',
                                    'Load a database first.')
            return

        # Only show columns used for prediction (6 features + target)
        keep = ['Vexp(kN)'] + list(NUM_FEAT_COLS)
        stat_cols = [c for c in keep if c in df.columns]

        dlg = QDialog(self)
        dlg.setWindowTitle('Descriptive Statistics: Prediction Variables')
        dlg.resize(720, 360)
        vl = QVBoxLayout(dlg)

        tbl = QTableWidget()
        tbl.setColumnCount(7)
        tbl.setHorizontalHeaderLabels(
            ['Variable', 'Count', 'Min', 'Max', 'Mean', 'S.D.', 'Median'])
        tbl.setRowCount(len(stat_cols))
        tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        tbl.verticalHeader().setVisible(False)
        tbl.setStyleSheet('font-size:12px;')

        for row, col in enumerate(stat_cols):
            s = df[col].dropna()
            display_name = self._COL_DISPLAY.get(col, col)
            vals = [display_name,
                    f'{len(s)}',
                    f'{s.min():.4g}',
                    f'{s.max():.4g}',
                    f'{s.mean():.4g}',
                    f'{s.std():.4g}',
                    f'{s.median():.4g}']
            for c, txt in enumerate(vals):
                it = QTableWidgetItem(txt)
                it.setTextAlignment(
                    Qt.AlignLeft | Qt.AlignVCenter if c == 0
                    else Qt.AlignCenter)
                tbl.setItem(row, c, it)

        tbl.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        for c in range(1, 7):
            tbl.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeToContents)
        vl.addWidget(tbl)

        close_btn = flat_btn('Close', width=100)
        close_btn.clicked.connect(dlg.accept)
        vl.addWidget(close_btn, alignment=Qt.AlignRight)
        dlg.exec_()

    # Pretty HTML names for column-mapping display
    _PRETTY_KEYS = {
        'Vexp':     'V<sub>exp</sub>',
        'd_mm':     'd',
        'b_mm':     'b',
        'fc_mpa':   "f′<sub>c</sub>",
        'rho_f':    'ρ<sub>f</sub>',
        'Ef_gpa':   'E<sub>f</sub>',
        'ad':       'a/d',
        'frp_type': 'FRP type',
    }

    # Unicode display names for mapping table (canonical key → display)
    _PRETTY_DISPLAY = {
        'Vexp':     'Vₑₓₚ',
        'd_mm':     'd',
        'b_mm':     'b',
        'fc_mpa':   "f′c",
        'rho_f':    'ρf',
        'Ef_gpa':   'Ef',
        'ad':       'a/d',
        'frp_type': 'FRP type',
    }

    # Prediction-relevant keys (matching Prediction tab's 7 input variables)
    _PREDICTION_KEYS = {'Vexp', 'd_mm', 'b_mm', 'fc_mpa',
                        'rho_f', 'Ef_gpa', 'ad', 'frp_type'}

    def _update_map_label(self, mapping, n_valid, n_total):
        # Filter: only show prediction-relevant variables
        pred_mapping = {k: v for k, v in mapping.items()
                        if k in self._PREDICTION_KEYS}
        self._map_table.setRowCount(len(pred_mapping))

        for row, (k, v) in enumerate(pred_mapping.items()):
            var_item = QTableWidgetItem(self._PRETTY_DISPLAY.get(k, k))
            var_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            # Mapped-to cell with blue highlight
            col_item = QTableWidgetItem(v)
            col_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            col_item.setForeground(QColor('#1A5FA0'))
            col_item.setBackground(QColor('#F0F6FC'))

            self._map_table.setItem(row, 0, var_item)
            self._map_table.setItem(row, 1, col_item)

        self._map_table.resizeRowsToContents()

        status = f'Records: {n_valid} / {n_total}'
        if n_valid == n_total:
            status += '  ✓'
        unmapped = [k for k in _VAR_CATALOGUE
                    if k in self._PREDICTION_KEYS and k not in mapping]
        if unmapped:
            names = ', '.join(self._PRETTY_DISPLAY.get(u, u) for u in unmapped)
            status += f'\nNot mapped: {names}'
        self._map_status.setText(status)

    def _plot_pie(self, df):
        """
        Donut chart with external legend — no label overlap.
        Percentages displayed inside large slices only; a legend box
        beneath the chart shows every category with count + percentage.
        """
        w = self.pie_canvas.width()
        h = self.pie_canvas.height()
        if w <= 0 or h <= 0:
            # Canvas not yet laid out — defer once via a single-shot timer.
            # Guard with a flag so multiple rapid resizes only schedule one
            # callback instead of stacking up dozens of deferred redraws.
            if not getattr(self, '_pie_timer_pending', False):
                self._pie_timer_pending = True
                from PyQt5.QtCore import QTimer
                def _deferred(_df=df):
                    self._pie_timer_pending = False
                    self._plot_pie(_df)
                QTimer.singleShot(80, _deferred)
            return

        self.pie_canvas.fig.clear()
        ax = self.pie_canvas.fig.add_subplot(111)

        if 'FRP-type' not in df.columns:
            ax.text(0.5, 0.5, 'FRP-type not mapped',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=9, color=C_TEXT2)
            try:
                self.pie_canvas.draw()
            except Exception:
                pass
            return

        counts = df['FRP-type'].value_counts()
        name_map = {'G': 'GFRP', 'C': 'CFRP', 'B': 'BFRP', 'A': 'AFRP'}
        labels = [name_map.get(k, k) for k in counts.index]
        total  = counts.sum()
        colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000',
                  '#A5A5A5', '#5B9BD5']

        def _autopct(pct):
            """Show percentage only for slices ≥ 8 %."""
            return f'{pct:.0f}%' if pct >= 8 else ''

        wedges, _, autotexts = ax.pie(
            counts.values,
            colors=colors[:len(counts)],
            autopct=_autopct,
            startangle=140,
            pctdistance=0.75,
            wedgeprops=dict(width=0.45, edgecolor='#FFFFFF', linewidth=1.5),
            textprops={'fontsize': 8, 'fontweight': 'bold', 'color': '#FFFFFF'},
        )

        # White text inside donut slices
        for t in autotexts:
            t.set_fontsize(8)
            t.set_fontweight('bold')
            t.set_color('#FFFFFF')

        legend_labels = [
            f'{lbl}  ({counts.values[i]/total*100:.0f}%)'
            for i, lbl in enumerate(labels)
        ]
        leg = ax.legend(
            wedges, legend_labels,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            fontsize=7.5,
            frameon=True,
            fancybox=False,
            edgecolor='#DDDDDD',
            ncol=1,
            handlelength=1.0,
            handleheight=0.8,
        )
        leg.get_frame().set_linewidth(0.5)

        ax.set_aspect('equal')
        try:
            self.pie_canvas.fig.tight_layout(pad=0.2)
            self.pie_canvas.fig.subplots_adjust(right=0.58)
        except Exception:
            pass
        try:
            self.pie_canvas.draw()
        except Exception:
            pass

