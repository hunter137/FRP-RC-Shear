"""
qt_compat.py — PyQt5 / PySide6 unified compatibility layer
===========================================================

Problem
-------
When PySide6 is installed (with or without PyQt5), matplotlib may pick
PySide6 as its Qt backend while the application code uses PyQt5 enum
values.  This causes crashes such as:

    TypeError: 'PySide6.QtWidgets.QWidget.setSizePolicy' called with
    wrong argument types:
      Supported signatures:
        setSizePolicy(PySide6.QtWidgets.QSizePolicy.Policy, ...)

because PyQt5-style ``QSizePolicy.Expanding`` (an int) is not the same
type as ``QSizePolicy.Policy.Expanding`` (a PySide6 enum member).

Solution
--------
Call ``patch()`` **once**, before any Qt or matplotlib import.

1. Detect which binding is "real" (PYQT_VERSION_STR on PyQt5.QtCore
   confirms it's a native binding, not a PySide6 wrapper).
2. Set the ``QT_API`` environment variable so that matplotlib uses the
   SAME binding as the rest of the application.
3. If only PySide6 is available, install flat enum aliases on PySide6
   classes (``QSizePolicy.Expanding = QSizePolicy.Policy.Expanding``,
   etc.) so that existing PyQt5-style code runs unchanged, and register
   synthetic ``PyQt5.*`` modules in ``sys.modules`` so that
   ``from PyQt5.QtWidgets import ...`` continues to work.

Usage (top of main.py, before any other import)
-----------------------------------------------
    import qt_compat
    qt_compat.patch()
"""

import os
import sys

# Binding detected by patch() — 'pyqt5' | 'pyside6' | None
BINDING = None   # type: str | None  (comment only — supports Python 3.8+)
_PATCHED = False

def patch() -> str:
    """
    Detect the active Qt binding, set QT_API, and install compatibility
    shims if needed.  Safe to call multiple times (no-op after first).

    Returns
    -------
    str : 'pyqt5' or 'pyside6'
    """
    global BINDING, _PATCHED
    if _PATCHED:
        return BINDING
    _PATCHED = True

    if _has_real_pyqt5():
        BINDING = 'pyqt5'
        os.environ.setdefault('QT_API', 'pyqt5')
    elif _has_pyside6():
        BINDING = 'pyside6'
        os.environ['QT_API'] = 'pyside6'
        _install_pyside6_shim()
    else:
        raise ImportError(
            "No Qt binding found.  Install one and restart:\n"
            "    pip install PyQt5\n"
            "    pip install PySide6\n"
        )

    return BINDING

def _has_real_pyqt5() -> bool:
    """Return True only for a native PyQt5 installation."""
    try:
        import PyQt5.QtCore as _qc
        # PYQT_VERSION_STR is defined by the real C extension;
        # PySide6-based shims do not expose it.
        return hasattr(_qc, 'PYQT_VERSION_STR')
    except ImportError:
        return False

def _has_pyside6() -> bool:
    try:
        import PySide6.QtCore  # noqa: F401
        return True
    except ImportError:
        return False

def _install_pyside6_shim():
    """
    Make PySide6 look like PyQt5 by:
      a) adding flat enum aliases to PySide6 widget classes, and
      b) registering fake PyQt5.* modules in sys.modules.
    """
    import PySide6.QtCore    as C6
    import PySide6.QtGui     as G6
    import PySide6.QtWidgets as W6

    # Qt namespace
    _flat(C6.Qt, {
        # Alignment
        'AlignLeft':      'AlignmentFlag.AlignLeft',
        'AlignRight':     'AlignmentFlag.AlignRight',
        'AlignCenter':    'AlignmentFlag.AlignCenter',
        'AlignHCenter':   'AlignmentFlag.AlignHCenter',
        'AlignVCenter':   'AlignmentFlag.AlignVCenter',
        'AlignTop':       'AlignmentFlag.AlignTop',
        'AlignBottom':    'AlignmentFlag.AlignBottom',
        # Orientation
        'Horizontal':     'Orientation.Horizontal',
        'Vertical':       'Orientation.Vertical',
        # Text format
        'RichText':       'TextFormat.RichText',
        'PlainText':      'TextFormat.PlainText',
        'AutoText':       'TextFormat.AutoText',
        # Window flags
        'Window':                       'WindowType.Window',
        'Dialog':                       'WindowType.Dialog',
        'WindowMinimizeButtonHint':     'WindowType.WindowMinimizeButtonHint',
        'WindowMaximizeButtonHint':     'WindowType.WindowMaximizeButtonHint',
        'WindowCloseButtonHint':        'WindowType.WindowCloseButtonHint',
        'WindowContextHelpButtonHint':  'WindowType.WindowContextHelpButtonHint',
        'FramelessWindowHint':          'WindowType.FramelessWindowHint',
        # Item data roles
        'DisplayRole':    'ItemDataRole.DisplayRole',
        'UserRole':       'ItemDataRole.UserRole',
        'ToolTipRole':    'ItemDataRole.ToolTipRole',
        # Item flags  (used in setFlags / checkState)
        'ItemIsEditable':    'ItemFlag.ItemIsEditable',
        'ItemIsEnabled':     'ItemFlag.ItemIsEnabled',
        'ItemIsSelectable':  'ItemFlag.ItemIsSelectable',
        'ItemIsCheckable':   'ItemFlag.ItemIsCheckable',
        'ItemIsDropEnabled': 'ItemFlag.ItemIsDropEnabled',
        'ItemIsDragEnabled': 'ItemFlag.ItemIsDragEnabled',
        'NoItemFlags':       'ItemFlag.NoItemFlags',
        # Check state
        'Checked':          'CheckState.Checked',
        'Unchecked':        'CheckState.Unchecked',
        'PartiallyChecked': 'CheckState.PartiallyChecked',
        # Sort
        'AscendingOrder':  'SortOrder.AscendingOrder',
        'DescendingOrder': 'SortOrder.DescendingOrder',
        # Scroll bar policy
        'ScrollBarAlwaysOff':    'ScrollBarPolicy.ScrollBarAlwaysOff',
        'ScrollBarAlwaysOn':     'ScrollBarPolicy.ScrollBarAlwaysOn',
        'ScrollBarAsNeeded':     'ScrollBarPolicy.ScrollBarAsNeeded',
        # Pen / brush styles (used in matplotlib canvas drawing)
        'DashLine':    'PenStyle.DashLine',
        'SolidLine':   'PenStyle.SolidLine',
        'NoPen':       'PenStyle.NoPen',
        'NoBrush':     'BrushStyle.NoBrush',
        'SolidPattern':'BrushStyle.SolidPattern',
        # Image / pixmap transformation
        'KeepAspectRatio':     'AspectRatioMode.KeepAspectRatio',
        'IgnoreAspectRatio':   'AspectRatioMode.IgnoreAspectRatio',
        'SmoothTransformation':'TransformationMode.SmoothTransformation',
        'FastTransformation':  'TransformationMode.FastTransformation',
        # Misc
        'ElideRight':    'TextElideMode.ElideRight',
    })

    # QSizePolicy
    _flat(W6.QSizePolicy, {
        'Fixed':            'Policy.Fixed',
        'Minimum':          'Policy.Minimum',
        'Maximum':          'Policy.Maximum',
        'Preferred':        'Policy.Preferred',
        'Expanding':        'Policy.Expanding',
        'MinimumExpanding': 'Policy.MinimumExpanding',
        'Ignored':          'Policy.Ignored',
    })

    # QFrame
    _flat(W6.QFrame, {
        'NoFrame':  'Shape.NoFrame',
        'Box':      'Shape.Box',
        'Panel':    'Shape.Panel',
        'StyledPanel': 'Shape.StyledPanel',
        'HLine':    'Shape.HLine',
        'VLine':    'Shape.VLine',
        'Plain':    'Shadow.Plain',
        'Raised':   'Shadow.Raised',
        'Sunken':   'Shadow.Sunken',
    })

    # QDialogButtonBox
    _flat(W6.QDialogButtonBox, {
        'Ok':           'StandardButton.Ok',
        'Open':         'StandardButton.Open',
        'Save':         'StandardButton.Save',
        'Cancel':       'StandardButton.Cancel',
        'Close':        'StandardButton.Close',
        'Discard':      'StandardButton.Discard',
        'Apply':        'StandardButton.Apply',
        'Reset':        'StandardButton.Reset',
        'Yes':          'StandardButton.Yes',
        'No':           'StandardButton.No',
        'Abort':        'StandardButton.Abort',
        'Retry':        'StandardButton.Retry',
        'Ignore':       'StandardButton.Ignore',
        'AcceptRole':      'ButtonRole.AcceptRole',
        'RejectRole':      'ButtonRole.RejectRole',
        'DestructiveRole': 'ButtonRole.DestructiveRole',
        'ActionRole':      'ButtonRole.ActionRole',
        'HelpRole':        'ButtonRole.HelpRole',
    })

    # QMessageBox
    _flat(W6.QMessageBox, {
        'Ok':        'StandardButton.Ok',
        'Save':      'StandardButton.Save',
        'SaveAll':   'StandardButton.SaveAll',
        'Open':      'StandardButton.Open',
        'Yes':       'StandardButton.Yes',
        'YesToAll':  'StandardButton.YesToAll',
        'No':        'StandardButton.No',
        'NoToAll':   'StandardButton.NoToAll',
        'Abort':     'StandardButton.Abort',
        'Retry':     'StandardButton.Retry',
        'Ignore':    'StandardButton.Ignore',
        'Close':     'StandardButton.Close',
        'Cancel':    'StandardButton.Cancel',
        'Discard':   'StandardButton.Discard',
        'Help':      'StandardButton.Help',
        'Apply':     'StandardButton.Apply',
        'Reset':     'StandardButton.Reset',
        'RestoreDefaults': 'StandardButton.RestoreDefaults',
        'NoButton':  'StandardButton.NoButton',
        # Icons
        'NoIcon':      'Icon.NoIcon',
        'Information': 'Icon.Information',
        'Warning':     'Icon.Warning',
        'Critical':    'Icon.Critical',
        'Question':    'Icon.Question',
    })

    # QDialog
    _flat(W6.QDialog, {
        'Accepted': 'DialogCode.Accepted',
        'Rejected': 'DialogCode.Rejected',
    })

    # QAbstractItemView
    _flat(W6.QAbstractItemView, {
        'NoSelection':          'SelectionMode.NoSelection',
        'SingleSelection':      'SelectionMode.SingleSelection',
        'MultiSelection':       'SelectionMode.MultiSelection',
        'ExtendedSelection':    'SelectionMode.ExtendedSelection',
        'ContiguousSelection':  'SelectionMode.ContiguousSelection',
        'SelectRows':           'SelectionBehavior.SelectRows',
        'SelectColumns':        'SelectionBehavior.SelectColumns',
        'SelectItems':          'SelectionBehavior.SelectItems',
        'NoEditTriggers':       'EditTrigger.NoEditTriggers',
        'CurrentChanged':       'EditTrigger.CurrentChanged',
        'DoubleClicked':        'EditTrigger.DoubleClicked',
    })

    # QHeaderView
    _flat(W6.QHeaderView, {
        'Interactive':      'ResizeMode.Interactive',
        'Fixed':            'ResizeMode.Fixed',
        'Stretch':          'ResizeMode.Stretch',
        'ResizeToContents': 'ResizeMode.ResizeToContents',
    })

    # QSpinBox / QDoubleSpinBox — NoButtons lives on QAbstractSpinBox
    _flat(W6.QSpinBox,       {'NoButtons': 'ButtonSymbols.NoButtons'},
          source=W6.QAbstractSpinBox)
    _flat(W6.QDoubleSpinBox, {'NoButtons': 'ButtonSymbols.NoButtons'},
          source=W6.QAbstractSpinBox)

    # QAbstractSpinBox itself (used directly in some older code)
    _flat(W6.QAbstractSpinBox, {
        'NoButtons':      'ButtonSymbols.NoButtons',
        'UpDownArrows':   'ButtonSymbols.UpDownArrows',
        'PlusMinus':      'ButtonSymbols.PlusMinus',
    })

    # pyqtSignal / pyqtSlot aliases on QtCore
    if not hasattr(C6, 'pyqtSignal'):
        C6.pyqtSignal = C6.Signal
    if not hasattr(C6, 'pyqtSlot'):
        C6.pyqtSlot = C6.Slot

    _register_fake_pyqt5(C6, G6, W6)

def _flat(cls, mapping: dict, source=None):
    """
    For each (flat_name, dotted_path) in *mapping*, add or OVERWRITE
    ``cls.flat_name`` with the resolved PySide6 enum member.

    We always overwrite (not skip) because PySide6 sometimes exposes
    legacy integer aliases like ``QSizePolicy.Expanding = 7`` that have
    the wrong type and are rejected by shiboken's type-strict dispatch.
    Replacing those integers with the real enum member fixes the crash:

        TypeError: setSizePolicy called with wrong argument types:
          setSizePolicy(Policy, Policy)

    *source* overrides the class used for attribute resolution
    (useful when the enum lives on a base class).
    """
    src = source if source is not None else cls
    for flat, dotted in mapping.items():
        try:
            obj = src
            for part in dotted.split('.'):
                obj = getattr(obj, part)
            setattr(cls, flat, obj)   # unconditional overwrite
        except AttributeError:
            pass   # enum member absent in this PySide6 version — skip

def _register_fake_pyqt5(C6, G6, W6):
    """
    Add synthetic ``PyQt5``, ``PyQt5.QtCore``, ``PyQt5.QtGui``, and
    ``PyQt5.QtWidgets`` entries to ``sys.modules`` so that all existing
    ``from PyQt5.Xxx import Yyy`` statements continue to work.
    """
    import types

    def _mirror(name: str, base) -> types.ModuleType:
        """Return a module that exposes everything from *base*."""
        m = types.ModuleType(name)
        m.__dict__.update(
            {k: v for k, v in vars(base).items() if not k.startswith('__')}
        )
        return m

    # Build fake sub-modules
    qtcore    = _mirror('PyQt5.QtCore',    C6)
    qtgui     = _mirror('PyQt5.QtGui',     G6)
    qtwidgets = _mirror('PyQt5.QtWidgets', W6)

    # Ensure pyqtSignal / pyqtSlot are available on the fake QtCore
    qtcore.pyqtSignal = C6.Signal
    qtcore.pyqtSlot   = C6.Slot

    # In PySide6, QAction moved from QtWidgets → QtGui.
    # Add it to the fake QtWidgets so `from PyQt5.QtWidgets import QAction`
    # continues to work unchanged.
    qtwidgets.QAction = G6.QAction

    # Patch exec_() onto dialogs / application so that dlg.exec_() works
    # regardless of which path the class was imported from.
    for _cls in (W6.QDialog, W6.QApplication, W6.QMenu):
        if not hasattr(_cls, 'exec_'):
            _cls.exec_ = _cls.exec

    # Root PyQt5 package
    root = types.ModuleType('PyQt5')
    root.QtCore    = qtcore
    root.QtGui     = qtgui
    root.QtWidgets = qtwidgets

    # Only register if not already present (real PyQt5 takes precedence)
    for key, mod in [
        ('PyQt5',          root),
        ('PyQt5.QtCore',   qtcore),
        ('PyQt5.QtGui',    qtgui),
        ('PyQt5.QtWidgets', qtwidgets),
    ]:
        sys.modules.setdefault(key, mod)
