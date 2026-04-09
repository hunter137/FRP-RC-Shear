"""
predict_helpers.py — Utility function and beam schematic widget for prediction.

Contents
--------
_compute_pi          Estimate base-learner prediction spread for ensemble models
BeamSchematicWidget  Displays a beam cross-section image loaded from disk
"""

# ── PyQt5 — complete import set ───────────────────────────────────────
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPainter, QPixmap, QColor, QFont, QPen, QBrush

# ── config ─────────────────────────────────────────────────────────────
from config import C_BORDER, C_TEXT2

def _compute_pi(model, vec, confidence=0.95):
    """
    Estimate the base-learner prediction spread for supported ensemble models.

    Method: collect predictions from each individual base estimator, then
    take the (alpha/2) and (1-alpha/2) percentiles as the spread bounds.

    Note: this is *not* a statistically calibrated prediction interval (PI).
    It reflects the dispersion of predictions across the ensemble's base
    learners, which approximates (but does not equal) a formal PI.

    Supported: RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor.
    Returns (lo, hi, std) as floats, or (None, None, None) if unsupported.
    """
    alpha = (1.0 - confidence) / 2.0
    cls   = type(model).__name__

    # Random Forest and Extra Trees — individual tree predictions
    if cls in ('RandomForestRegressor', 'ExtraTreesRegressor'):
        try:
            preds = np.array([t.predict(vec)[0] for t in model.estimators_])
            return (float(np.percentile(preds, alpha * 100)),
                    float(np.percentile(preds, (1.0 - alpha) * 100)),
                    float(preds.std()))
        except Exception:
            return None, None, None

    # AdaBoost — individual base estimator predictions
    if cls == 'AdaBoostRegressor':
        try:
            preds = np.array([est.predict(vec)[0] for est in model.estimators_])
            return (float(np.percentile(preds, alpha * 100)),
                    float(np.percentile(preds, (1.0 - alpha) * 100)),
                    float(preds.std()))
        except Exception:
            return None, None, None

    # Other models (GBDT, XGBoost, LightGBM, SVR, KNN, MLP):
    # individual-tree PI not accessible without retraining — return None
    return None, None, None


# ═══════════════════════════════════════════════════════════════════════
#  BEAM SCHEMATIC WIDGET  —  Loads an external image file
#
#  Place one of these files next to main.py:
#    beam_schematic.jpg / beam_schematic.png / beam_schematic.svg
#
#  If no image is found a clean placeholder is shown with instructions.
# ═══════════════════════════════════════════════════════════════════════

class BeamSchematicWidget(QWidget):
    """
    Displays a beam schematic image loaded from disk.

    On construction the widget scans the script directory for
    ``beam_schematic.{png,jpg,jpeg,bmp,svg}``.  If a match is found the
    image is painted scaled-to-fit with preserved aspect ratio.
    Otherwise a light placeholder rectangle with guidance text is shown.

    Call ``load_image(path)`` at any time to switch to a different file.
    """

    _STEM = 'beam_schematic'
    _EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.svg')

    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt5.QtGui import QPixmap
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setToolTip(
            'FRP-RC beam schematic\n'
            'Place beam_schematic.jpg next to main.py to display.')
        self._pixmap = QPixmap()
        self._auto_discover()

    # ── public API ────────────────────────────────────────────────────
    def load_image(self, path: str):
        """Load an image file and repaint."""
        from PyQt5.QtGui import QPixmap
        pm = QPixmap(path)
        if not pm.isNull():
            self._pixmap = pm
            self.update()

    # ── internals ─────────────────────────────────────────────────────
    def _auto_discover(self):
        """Look for beam_schematic.* next to this script."""
        import os, glob
        base = os.path.dirname(os.path.abspath(__file__))
        for ext in self._EXTS:
            candidates = glob.glob(
                os.path.join(base, self._STEM + ext))
            if candidates:
                self.load_image(candidates[0])
                return

    def sizeHint(self):
        return QSize(500, 188)

    def paintEvent(self, evt):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        W, H = self.width(), self.height()

        if not self._pixmap.isNull():
            # ── Draw loaded image, scaled to fit, centred ────────────
            pm = self._pixmap.scaled(
                W, H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x0 = (W - pm.width())  // 2
            y0 = (H - pm.height()) // 2
            p.drawPixmap(x0, y0, pm)
        else:
            # ── Placeholder ──────────────────────────────────────────
            margin = 6
            p.setPen(QPen(QColor(C_BORDER), 1.0, Qt.DashLine))
            p.setBrush(QBrush(QColor('#FAFAF8')))
            p.drawRect(margin, margin, W - 2*margin, H - 2*margin)
            p.setBrush(Qt.NoBrush)

            f1 = QFont('DejaVu Sans', 11)
            f1.setBold(True)
            p.setFont(f1)
            p.setPen(QPen(QColor(C_TEXT2)))
            p.drawText(0, 0, W, H, Qt.AlignCenter,
                       'Beam Schematic Placeholder')

            f2 = QFont('DejaVu Sans', 9)
            p.setFont(f2)
            p.setPen(QPen(QColor(C_BORDER)))
            p.drawText(0, H // 2 + 12, W, 40, Qt.AlignHCenter,
                       'Place  beam_schematic.jpg  next to main.py')

        p.end()




# ══════════════════════════════════════════════════════════════════════
#  PredictionSetupDialog  — select calculation methods before predicting
# ══════════════════════════════════════════════════════════════════════
