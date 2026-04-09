"""
column_mapping.py — Database column auto-detection and mapping dialog.
"""
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QMessageBox,
)
from PyQt5.QtCore import Qt

from config import C_TEXT, C_TEXT2, C_BORDER, C_ACCENT, C_ACCENT_LT, C_ACCENT_BG
from widgets import panel, _stat_textbox

#  SECTION 7 — COLUMN MAPPING (database portability)
# ═══════════════════════════════════════════════════════════════════════

# Internal canonical keys → human-readable descriptions
_VAR_CATALOGUE = {
    'Vexp':     'Experimental shear capacity V_exp (kN)  [REQUIRED]',
    'd_mm':     'Effective depth d (mm)',
    'b_mm':     'Beam width b (mm)',
    'fc_mpa':   "Concrete cylinder strength f'c (MPa)",
    'rho_f':    'FRP longitudinal reinforcement ratio ρ_f (%)',
    'Ef_gpa':   'FRP elastic modulus E_f (GPa)',
    'ad':       'Shear span-to-depth ratio a/d',
    'frp_type': 'FRP material type (C = CFRP, G = GFRP, B = BFRP, A = AFRP)',
}

# Pretty HTML for the mapping dialog (Variable column + Description column)
_VAR_HTML = {
    'Vexp':     ('V<sub>exp</sub>',
                 'Experimental shear capacity V<sub>exp</sub> (kN)'
                 '  <b style="color:#B30000">[REQUIRED]</b>'),
    'd_mm':     ('d',
                 'Effective depth <i>d</i> (mm)'),
    'b_mm':     ('b',
                 'Beam width <i>b</i> (mm)'),
    'fc_mpa':   ("f′<sub>c</sub>",
                 "Concrete cylinder strength <i>f′<sub>c</sub></i> (MPa)"),
    'rho_f':    ('ρ<sub>f</sub>',
                 'FRP longitudinal reinforcement ratio <i>ρ<sub>f</sub></i> (%)'),
    'Ef_gpa':   ('E<sub>f</sub>',
                 'FRP elastic modulus <i>E<sub>f</sub></i> (GPa)'),
    'ad':       ('<i>a/d</i>',
                 'Shear span-to-depth ratio <i>a/d</i>'),
    'frp_type': ('FRP type',
                 'FRP material type (C, G, B, A)'),
}

# Maps canonical key → standardised internal column name
_INTERNAL_NAME = {
    'Vexp':     'Vexp(kN)',
    'd_mm':     'd(mm)',
    'b_mm':     'b(mm)',
    'fc_mpa':   "f`c(Mpa)",
    'rho_f':    'ρf(%)',
    'Ef_gpa':   'Ef(GPa)',
    'ad':       'a/d',
    'ffu_mpa':  'ffu(MPa)',
    'frp_type': 'FRP-type',
}

# Auto-detection aliases (lowercase, stripped)
_ALIASES = {
    'Vexp':     ['vexp(kn)','vexp(kN)','vexp(KN)','vu(kn)',
                 'vtest(kn)','vexp','shear_capacity'],
    'd_mm':     ['d(mm)','d_mm','h0','h_0','effective_depth'],
    'b_mm':     ['b(mm)','b_mm','bw','b_w','beam_width'],
    'fc_mpa':   ["f`c(mpa)","f'c(mpa)","f`cmpa","f'cmpa",
                 'fck','fcu','fcu_mpa','fc_mpa',"f'c",'fc'],
    'rho_f':    ['ρf(%)','ρf/配筋率','rhof','rho_f','rf(%)'],
    'Ef_gpa':   ['ef(gpa)','ef_gpa','e_f','ef'],
    'frp_type': ['frp-type','frp_type','frptype','material_type'],
    'ad':       ['a/d','ad','a_d','shear_span_ratio'],
}


def _auto_map(raw_cols, df=None):
    """Return {canonical_key: raw_col_name} via exact + stripped matching.

    When *df* is provided, ties between equally-scored alias matches are
    broken by preferring the column with the most non-null values.  This
    prevents statistics-summary columns (with only a few rows of data)
    from winning over the actual data column.
    """
    import re
    strip     = lambda s: re.sub(r'[\s\(\)\[\]_\-`\']', '', str(s)).lower()
    low_map   = {str(c).lower(): c for c in raw_cols}
    strip_map = {}
    # Build strip_map: when multiple columns strip to the same key, keep ALL
    # candidates so we can choose the most data-dense one.
    strip_multi = {}
    for c in raw_cols:
        k = strip(c)
        strip_multi.setdefault(k, []).append(c)
    # For backward compat, strip_map still holds one value (the first);
    # _best_col() below resolves ties using df if available.
    strip_map = {k: v[0] for k, v in strip_multi.items()}

    def _best_col(candidates):
        """Return the candidate with the most non-null rows (needs df)."""
        if df is None or len(candidates) == 1:
            return candidates[0]
        counts = []
        for c in candidates:
            try:
                counts.append((df[c].notna().sum(), c))
            except Exception:
                counts.append((0, c))
        return max(counts)[1]

    result = {}
    for key, aliases in _ALIASES.items():
        for a in aliases:
            if str(a).lower() in low_map:
                result[key] = low_map[str(a).lower()]; break
            sk = strip(a)
            if sk in strip_multi:
                result[key] = _best_col(strip_multi[sk]); break
    return result


def _build_dataframe(raw, mapping, drop_no_target=True):
    """
    Construct a standardised DataFrame from raw data + column mapping.
    Returns (df, n_valid, n_total).
    """
    out = {}
    for key, raw_col in mapping.items():
        if raw_col in raw.columns:
            out[_INTERNAL_NAME[key]] = raw[raw_col].values
    if 'Vexp(kN)' not in out:
        raise ValueError(
            "Target column (V_exp) is not mapped.\n"
            "Please assign it via the column-mapping dialog.")
    df = pd.DataFrame(out).reset_index(drop=True)
    df['Vexp(kN)'] = pd.to_numeric(df['Vexp(kN)'], errors='coerce')
    n_total = len(df)
    valid   = df['Vexp(kN)'].notna() & (df['Vexp(kN)'] > 0)
    n_valid = int(valid.sum())
    if drop_no_target:
        df = df[valid].reset_index(drop=True)
    return df, n_valid, n_total


class ColumnMappingDialog(QDialog):
    """
    Interactive dialog for connecting any user database to the platform.

    Presents auto-detected column assignments in editable dropdowns,
    enabling the user to correct or override each mapping before proceeding.
    This mechanism ensures that the platform is portable to any database
    with arbitrary column naming conventions.
    """
    def __init__(self, raw_cols, auto_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Column Mapping — Connect Your Database')
        self.setMinimumWidth(700)
        self.setMinimumHeight(460)
        self.combos = {}
        self._build(raw_cols, auto_map)

    def _build(self, raw_cols, auto_map):
        layout = QVBoxLayout(self)

        info = _stat_textbox(
            'Map your database column names to the variables required by '
            'the platform.  Columns marked [REQUIRED] must be assigned; '
            'all others are optional but improve prediction accuracy.\n'
            'Auto-detected assignments are highlighted in blue.')
        layout.addWidget(info)

        grp  = panel('Variable → Column assignment')
        grid = QGridLayout(grp)
        grid.setSpacing(6)
        for col_i, text in enumerate(
                ('Variable', 'Description', 'Database column')):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                'font-weight:bold;font-size:12px;')
            grid.addWidget(lbl, 0, col_i)

        OPTIONS = ['— not mapped —'] + raw_cols
        for row, (key, desc) in enumerate(_VAR_CATALOGUE.items(), start=1):
            # Variable name — rich text with subscripts
            var_html, desc_html = _VAR_HTML.get(key, (key, desc))
            key_lbl = QLabel(var_html)
            key_lbl.setTextFormat(Qt.RichText)
            key_lbl.setStyleSheet('font-size:13px; padding:2px 4px;')

            # Description — rich text
            desc_lbl = QLabel(desc_html)
            desc_lbl.setTextFormat(Qt.RichText)
            desc_lbl.setStyleSheet(
                f'color:{C_TEXT2};font-size:12px; padding:2px 4px;')
            cb = QComboBox()
            cb.addItems(OPTIONS)
            if key in auto_map and auto_map[key] in raw_cols:
                cb.setCurrentIndex(OPTIONS.index(auto_map[key]))
                cb.setStyleSheet(
                    f'background:{C_ACCENT_BG};'
                    f'border:1px solid {C_ACCENT_LT};')
            else:
                cb.setCurrentIndex(0)
                if '[REQUIRED]' in desc:
                    cb.setStyleSheet(
                        'border:1px solid #D08080;'
                        'background:#FFF5F5;')
            self.combos[key] = cb
            grid.addWidget(key_lbl,  row, 0)
            grid.addWidget(desc_lbl, row, 1)
            grid.addWidget(cb,       row, 2)
        layout.addWidget(grp)

        btns = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._validate)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _validate(self):
        if self.combos['Vexp'].currentIndex() == 0:
            QMessageBox.warning(
                self, 'Required Column Missing',
                'The experimental shear capacity (V_exp) column must be '
                'assigned before proceeding.')
            return
        self.accept()

    def get_mapping(self):
        """Return {canonical_key: raw_col_name} for all assigned variables."""
        return {k: cb.currentText()
                for k, cb in self.combos.items()
                if cb.currentIndex() > 0}

