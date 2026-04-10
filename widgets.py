"""
widgets.py — Shared UI helper widgets.

Provides: flat_btn, panel, result_box, _stat_textbox, MplCanvas, _spin_field.
"""
from PyQt5.QtWidgets import (
    QGroupBox, QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QSizePolicy, QSpinBox, QDoubleSpinBox,
)
from PyQt5.QtCore import Qt

# Matplotlib backend
# Import matplotlib BEFORE setting the backend, and guard against the
# backend already being set (e.g. when running tests or in a Jupyter
# notebook where the backend is managed externally).
import matplotlib
_WANT = ('Qt5Agg', 'Qt6Agg', 'TkAgg', 'Agg')
_current = matplotlib.get_backend().lower()
if _current not in {b.lower() for b in _WANT}:
    for _b in _WANT:
        try:
            matplotlib.use(_b)
            break
        except Exception:
            continue

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from config import (
    C_ACCENT, C_ACCENT_LT, C_ACCENT_BG, C_TEXT, C_TEXT2,
    C_BORDER, C_BORDER_LT, C_DANGER,
    C_SUCCESS, C_SUCCESS_BG, C_PANEL_BG,
)

def flat_btn(text, accent=False, danger=False, width=None):
    """
    Clean rectangular button matching the reference UI aesthetic.
    Primary actions (accent=True) use a muted blue fill.
    Secondary actions are white with a hairline border.
    """
    btn = QPushButton(text)
    base_font = 'font-size:12px;font-family:inherit;'
    if accent:
        s = (f'QPushButton{{{base_font}background:{C_ACCENT};color:#FFFFFF;'
             f'border:1px solid {C_ACCENT};padding:6px 18px;border-radius:0px;}}'
             f'QPushButton:hover{{background:#1D4F8A;border-color:#1D4F8A;}}'
             f'QPushButton:pressed{{background:#163A6B;}}'
             f'QPushButton:disabled{{background:#B8B8B8;color:#808080;'
             f'border-color:#B8B8B8;}}')
    elif danger:
        s = (f'QPushButton{{{base_font}background:#FFFFFF;color:{C_DANGER};'
             f'border:1px solid {C_DANGER};padding:6px 18px;border-radius:0px;}}'
             f'QPushButton:hover{{background:#FFF0F0;}}'
             f'QPushButton:disabled{{color:#CCCCCC;border-color:#DDDDDD;}}')
    else:
        s = (f'QPushButton{{{base_font}background:#FFFFFF;color:{C_TEXT};'
             f'border:1px solid {C_BORDER};padding:6px 18px;border-radius:0px;}}'
             f'QPushButton:hover{{background:{C_ACCENT_BG};'
             f'border-color:{C_ACCENT_LT};color:{C_ACCENT};}}'
             f'QPushButton:pressed{{background:#DCE8F4;}}'
             f'QPushButton:disabled{{color:#AAAAAA;border-color:#DDDDDD;}}')
    btn.setStyleSheet(s)
    if width:
        btn.setFixedWidth(width)
    return btn

def panel(title=''):
    """
    Section panel — bold black title, thin border, white bg.
    """
    g = QGroupBox(title)
    g.setStyleSheet(
        f'QGroupBox{{background:{C_PANEL_BG};'
        f'border:1px solid {C_BORDER};'
        f'border-radius:0px;margin-top:20px;padding:8px 6px 6px 6px;}}'
        f'QGroupBox::title{{subcontrol-origin:margin;left:10px;'
        f'padding:0 4px;color:{C_TEXT};'
        f'font-size:13px;font-weight:bold;}}')
    return g

def result_box(text):
    """
    Green-tinted result display.
    """
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    lbl.setStyleSheet(
        f'background:{C_SUCCESS_BG};color:{C_SUCCESS};'
        f'font-size:12px;font-weight:bold;'
        f'padding:6px 10px;'
        f'border:1px solid #A8D5A8;')
    return lbl

def _stat_textbox(text):
    """Inline informational note — light tint, fine border."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f'background:{C_ACCENT_BG};color:{C_TEXT2};font-size:11px;'
        f'padding:5px 8px;border:1px solid {C_BORDER_LT};')
    return lbl

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=5, dpi=110):
        self.fig = Figure(figsize=(width, height), dpi=dpi,
                          facecolor='#FFFFFF')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f'border:1px solid {C_BORDER_LT};'
                           f'background:#FFFFFF;')

def _spin_field(label_text, widget, unit=''):
    """
    Build a reference-style input cell:
      Label (unit)
      [  −  ][   value field   ][  +  ]

    label_text may contain HTML (e.g. subscripts).
    Returns (container QWidget, the spinbox).
    """
    from PyQt5.QtCore import Qt as _Qt
    cell    = QWidget()
    vl      = QVBoxLayout(cell)
    vl.setSpacing(2)
    vl.setContentsMargins(0, 0, 0, 0)

    lbl_text = f'{label_text} ({unit})' if unit else label_text
    lbl      = QLabel(lbl_text)
    lbl.setTextFormat(_Qt.RichText)
    lbl.setStyleSheet(f'font-size:12px;color:{C_TEXT2};')
    vl.addWidget(lbl)

    row = QHBoxLayout()
    row.setSpacing(0)
    row.setContentsMargins(0, 0, 0, 0)

    def _btn(sym):
        b = QPushButton(sym)
        b.setFixedSize(26, 28)
        b.setStyleSheet(
            f'QPushButton{{background:#FFFFFF;color:{C_TEXT};'
            f'border:1px solid {C_BORDER};font-size:14px;'
            f'padding:0px;border-radius:0px;}}'
            f'QPushButton:hover{{background:{C_ACCENT_BG};'
            f'color:{C_ACCENT};}}'
            f'QPushButton:pressed{{background:#DCE8F4;}}')
        return b

    minus_btn = _btn('−')
    plus_btn  = _btn('+')
    widget.setStyleSheet(
        f'QSpinBox,QDoubleSpinBox{{'
        f'background:#FFFFFF;border:1px solid {C_BORDER};'
        f'border-left:none;border-right:none;'
        f'border-radius:0px;padding:2px 6px;'
        f'font-size:12px;min-height:28px;}}')
    widget.setButtonSymbols(QSpinBox.NoButtons
                            if isinstance(widget, QSpinBox)
                            else QDoubleSpinBox.NoButtons)

    minus_btn.clicked.connect(
        lambda: widget.setValue(widget.value() - widget.singleStep()))
    plus_btn.clicked.connect(
        lambda: widget.setValue(widget.value() + widget.singleStep()))

    row.addWidget(minus_btn)
    row.addWidget(widget, 1)
    row.addWidget(plus_btn)
    vl.addLayout(row)
    return cell
