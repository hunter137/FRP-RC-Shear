#!/usr/bin/env python3
"""
FRP-RC Shear Strength Prediction Platform
==========================================
Launch:  python main.py

A PyQt5 desktop application for predicting the shear capacity of
FRP-reinforced concrete beams without stirrups, combining five
international design codes with eight ensemble ML algorithms.
"""
# Allow multiple Intel OpenMP copies to coexist (harmless when only one is present).
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_WARNINGS']         = 'FALSE'


# Write a native stack trace to stderr on C-level crashes.
import faulthandler, sys as _sys
faulthandler.enable(file=_sys.stderr, all_threads=True)


import qt_compat
qt_compat.patch()

import sys
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Propagate warning suppression to loky subprocess workers.
os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# Suppress LightGBM C++-level stderr messages.
try:
    import lightgbm as _lgb
    _lgb.basic._LightGBMLibrary
    os.environ.setdefault('LIGHTGBM_VERBOSITY', '-1')
except Exception:
    pass
from config import _configure_mpl
_configure_mpl()

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt

from config import (
    C_WIN_BG, C_ALT_ROW, C_ACCENT, C_ACCENT_BG,
    C_TEXT, C_TEXT2, C_BORDER, C_BORDER_LT, C_HEADER_BG,
)
from app import MainWindow

def _install_excepthook():
    """Show an error dialog for unhandled exceptions instead of silently exiting."""
    from PyQt5.QtWidgets import QMessageBox
    import traceback as _tb

    def _hook(exc_type, exc_value, exc_tb):
        _tb.print_exception(exc_type, exc_value, exc_tb)          # keep stderr log
        msg = ''.join(_tb.format_exception(exc_type, exc_value, exc_tb))
        dlg = QMessageBox()
        dlg.setIcon(QMessageBox.Critical)
        dlg.setWindowTitle('Unexpected Error')
        dlg.setText(
            '<b>An unexpected error occurred.</b><br>'
            'The application will attempt to continue.<br><br>'
            'Please copy the details below and report this issue.')
        dlg.setDetailedText(msg)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.exec_()

    sys.excepthook = _hook

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    font = QFont()
    if   sys.platform == 'win32':   font.setFamily('Segoe UI')
    elif sys.platform == 'darwin':  font.setFamily('SF Pro Text')
    else:                           font.setFamily('DejaVu Sans')
    font.setPointSize(11)
    app.setFont(font)

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(C_WIN_BG))
    pal.setColor(QPalette.Base,            QColor('#FFFFFF'))
    pal.setColor(QPalette.AlternateBase,   QColor(C_ALT_ROW))
    pal.setColor(QPalette.Highlight,       QColor(C_ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor('#FFFFFF'))
    pal.setColor(QPalette.Text,            QColor(C_TEXT))
    pal.setColor(QPalette.WindowText,      QColor(C_TEXT))
    pal.setColor(QPalette.Button,          QColor('#E8E8E8'))
    pal.setColor(QPalette.ButtonText,      QColor(C_TEXT))
    app.setPalette(pal)

    app.setStyleSheet(f"""
        /* ── Global base ── */
        QWidget{{
            font-size:12px;
            color:{C_TEXT};
        }}
        /* ── Input widgets — flat, square corners ── */
        QLineEdit{{
            background:#FFFFFF;
            border:1px solid {C_BORDER};
            border-radius:0px;
            padding:3px 6px;
            font-size:12px;
        }}
        QLineEdit:focus{{
            border:1px solid {C_ACCENT};
        }}
        QSpinBox,QDoubleSpinBox{{
            background:#FFFFFF;
            border:1px solid {C_BORDER};
            border-radius:0px;
            padding:2px 4px;
            font-size:12px;
        }}
        QSpinBox:focus,QDoubleSpinBox:focus{{
            border:1px solid {C_ACCENT};
        }}
        QComboBox{{
            background:#FFFFFF;
            border:1px solid {C_BORDER};
            border-radius:0px;
            padding:3px 6px;
            font-size:12px;
        }}
        QComboBox:focus{{
            border:1px solid {C_ACCENT};
        }}
        QComboBox::drop-down{{
            border:none;
            width:18px;
        }}
        /* ── Tables ── */
        QTableWidget{{
            gridline-color:{C_BORDER_LT};
            font-size:12px;
            background:#FFFFFF;
            alternate-background-color:{C_ALT_ROW};
            border:1px solid {C_BORDER};
        }}
        QHeaderView::section{{
            background:{C_HEADER_BG};
            color:{C_TEXT};
            font-size:12px;
            font-weight:bold;
            padding:4px 6px;
            border:none;
            border-right:1px solid {C_BORDER_LT};
            border-bottom:1px solid {C_BORDER};
        }}
        /* ── Scrollbars — minimal ── */
        QScrollBar:vertical{{
            width:9px;background:{C_WIN_BG};margin:0;
        }}
        QScrollBar::handle:vertical{{
            background:{C_BORDER};min-height:20px;margin:1px;
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical{{height:0;}}
        QScrollBar:horizontal{{
            height:9px;background:{C_WIN_BG};margin:0;
        }}
        QScrollBar::handle:horizontal{{
            background:{C_BORDER};min-width:20px;margin:1px;
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal{{width:0;}}
        /* ── Tooltip ── */
        QToolTip{{
            background:#FFFFFF;color:{C_TEXT};
            border:1px solid {C_BORDER};
            font-size:12px;padding:3px 6px;
        }}
        /* ── Menu bar ── */
        QMenuBar{{
            background:{C_WIN_BG};
            border-bottom:1px solid {C_BORDER_LT};
            font-size:12px;
        }}
        QMenuBar::item:selected{{
            background:{C_ACCENT_BG};color:{C_ACCENT};
        }}
        QMenu{{
            background:#FFFFFF;border:1px solid {C_BORDER};
            font-size:12px;
        }}
        QMenu::item:selected{{
            background:{C_ACCENT_BG};color:{C_ACCENT};
        }}
        /* ── Progress bar ── */
        QProgressBar{{
            border:1px solid {C_BORDER};
            background:#F0F0F0;
            text-align:center;
            font-size:11px;
            border-radius:0px;
        }}
        QProgressBar::chunk{{background:{C_TEXT};}}
        /* ── CheckBox / RadioButton ── */
        QCheckBox,QRadioButton{{font-size:12px;}}
        /* ── Status bar ── */
        QStatusBar{{
            background:{C_WIN_BG};
            border-top:1px solid {C_BORDER_LT};
            font-size:11px;color:{C_TEXT2};
        }}
        /* ── Splitter ── */
        QSplitter::handle{{
            background:{C_BORDER_LT};
        }}
    """)

    win = MainWindow()
    _install_excepthook()        # must be after QApplication is created
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
