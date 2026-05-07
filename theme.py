"""Catppuccin Mocha dark theme constants and styling."""


class T:
    """Color palette."""
    BG = "#1e1e2e"
    S1 = "#2a2a3d"
    S2 = "#353550"
    PRI = "#89b4fa"
    GRN = "#a6e3a1"
    PEA = "#fab387"
    RED = "#f38ba8"
    YEL = "#f9e2af"
    MAU = "#cba6f7"
    TXT = "#cdd6f4"
    SUB = "#a6adc8"
    DIM = "#6c7086"
    OVR = "#45475a"
    TEAL = "#94e2d5"
    SKY = "#74c7ec"

    @staticmethod
    def qc(h, a=255):
        return _qcolor(h, a)

    @staticmethod
    def css():
        return _CSS


def _qcolor(h, a=255):
    return __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(
        int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16), a
    )


_CSS = f"""
QWidget{{background:{T.BG};color:{T.TXT};font-family:'Segoe UI',sans-serif;font-size:12px}}
QGroupBox{{border:1px solid {T.OVR};border-radius:8px;margin-top:14px;padding-top:18px;
            font-weight:bold;color:{T.SUB}}}
QGroupBox::title{{subcontrol-origin:margin;left:12px;padding:0 6px}}
QPushButton{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;
            padding:6px 10px;color:{T.TXT};font-weight:bold;min-height:16px;font-size:11px}}
QPushButton:hover{{background:{T.OVR}}}
QPushButton:pressed{{background:{T.PRI};color:{T.BG}}}
QPushButton:disabled{{color:{T.DIM};border-color:{T.DIM}}}
QPushButton#startBtn{{background:#2d5a3e;border-color:{T.GRN}}}
QPushButton#startBtn:hover{{background:#3a7a52}}
QPushButton#stopBtn{{background:#5a2d2d;border-color:{T.RED}}}
QPushButton#stopBtn:hover{{background:#7a3a3a}}
QPushButton#resetBtn{{background:#5a4a2d;border-color:{T.YEL}}}
QPushButton#resetBtn:hover{{background:#7a6a3a}}
QComboBox{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;padding:5px 8px;
          color:{T.TXT};font-size:11px}}
QComboBox::drop-down{{border:none;width:18px}}
QComboBox QAbstractItemView{{background:{T.S2};color:{T.TXT};selection-background-color:{T.PRI}}}
QDoubleSpinBox,QSpinBox{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;
                         padding:3px 6px;color:{T.TXT};font-size:11px}}
QLineEdit{{background:{T.S2};border:1px solid {T.OVR};border-radius:6px;padding:4px 8px;
           color:{T.TXT};font-size:11px}}
QLabel{{color:{T.SUB}}}
QLabel#desc{{color:{T.DIM};font-size:10px;font-style:italic;padding:2px}}
QCheckBox{{color:{T.SUB};spacing:6px}}
QSlider::groove:horizontal{{height:4px;background:{T.OVR};border-radius:2px}}
QSlider::handle:horizontal{{background:{T.PRI};width:14px;margin:-5px 0;border-radius:7px}}
QTabWidget::pane{{border:1px solid {T.OVR};border-radius:6px;background:{T.S1}}}
QTabBar::tab{{background:{T.S2};color:{T.SUB};padding:6px 12px;border-top-left-radius:6px;
              border-top-right-radius:6px;margin-right:2px;font-size:10px}}
QTabBar::tab:selected{{background:{T.S1};color:{T.PRI};border-bottom:2px solid {T.PRI}}}
QTableWidget{{background:{T.S1};gridline-color:{T.OVR};color:{T.TXT};font-size:10px;
              border:1px solid {T.OVR};border-radius:4px}}
QTableWidget::item{{padding:2px}}
QHeaderView::section{{background:{T.S2};color:{T.SUB};padding:4px;border:1px solid {T.OVR};
                     font-size:10px}}
QScrollBar:vertical{{background:{T.S1};width:8px;border:none}}
QScrollBar::handle:vertical{{background:{T.OVR};border-radius:4px;min-height:20px}}
QScrollArea{{border:none}}
QStatusBar{{background:{T.S1};color:{T.DIM};font-size:10px;border-top:1px solid {T.OVR}}}
"""


def apply_theme(app):
    """Apply dark palette to the QApplication."""
    from PySide6.QtGui import QPalette
    p = app.palette()
    p.setColor(QPalette.ColorRole.Window, T.qc(T.BG))
    p.setColor(QPalette.ColorRole.WindowText, T.qc(T.TXT))
    p.setColor(QPalette.ColorRole.Base, T.qc(T.S1))
    p.setColor(QPalette.ColorRole.Text, T.qc(T.TXT))
    p.setColor(QPalette.ColorRole.Button, T.qc(T.S2))
    p.setColor(QPalette.ColorRole.ButtonText, T.qc(T.TXT))
    p.setColor(QPalette.ColorRole.Highlight, T.qc(T.PRI))
    p.setColor(QPalette.ColorRole.HighlightedText, T.qc(T.BG))
    p.setColor(QPalette.ColorRole.AlternateBase, T.qc(T.S2))
    app.setPalette(p)