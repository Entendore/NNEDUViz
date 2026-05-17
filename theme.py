"""Catppuccin Mocha dark theme for PySide6."""

from PySide6.QtGui import QColor, QPalette


class ThemeColors:
    BG = "#1e1e2e"
    S1 = "#2a2a3d"
    S2 = "#353550"
    PRI = "#89b4fa"
    GRN = "#a6e3a1"
    PEA = "#fab387"
    RED = "#f38ba8"
    YEL = "#f9e2af"
    MAU = "#cba6f7"
    TEAL = "#94e2d5"
    SKY = "#74c7ec"
    TXT = "#cdd6f4"
    SUB = "#a6adc8"
    DIM = "#6c7086"
    OVR = "#45475a"

    @staticmethod
    def qc(hex_color: str, alpha: int = 255) -> QColor:
        return QColor(int(hex_color[1:3], 16), int(hex_color[3:5], 16),
                      int(hex_color[5:7], 16), alpha)

    @staticmethod
    def css() -> str:
        return _STYLESHEET


def apply_theme(app) -> None:
    T = ThemeColors
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, T.qc(T.BG))
    palette.setColor(QPalette.ColorRole.WindowText, T.qc(T.TXT))
    palette.setColor(QPalette.ColorRole.Base, T.qc(T.S1))
    palette.setColor(QPalette.ColorRole.Text, T.qc(T.TXT))
    palette.setColor(QPalette.ColorRole.Button, T.qc(T.S2))
    palette.setColor(QPalette.ColorRole.ButtonText, T.qc(T.TXT))
    palette.setColor(QPalette.ColorRole.Highlight, T.qc(T.PRI))
    palette.setColor(QPalette.ColorRole.HighlightedText, T.qc(T.BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, T.qc(T.S2))
    app.setPalette(palette)
    app.setStyleSheet(_STYLESHEET)


T = ThemeColors

_STYLESHEET = f"""
QWidget {{
    background: {T.BG};
    color: {T.TXT};
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    border: 1px solid {T.OVR};
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 18px;
    font-weight: bold;
    color: {T.SUB};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}}
QPushButton {{
    background: {T.S2};
    border: 1px solid {T.OVR};
    border-radius: 6px;
    padding: 6px 10px;
    color: {T.TXT};
    font-weight: bold;
    min-height: 16px;
    font-size: 11px;
}}
QPushButton:hover {{ background: {T.OVR}; }}
QPushButton:pressed {{ background: {T.PRI}; color: {T.BG}; }}
QPushButton:disabled {{ color: {T.DIM}; border-color: {T.DIM}; }}
QPushButton#startBtn {{ background: #2d5a3e; border-color: {T.GRN}; }}
QPushButton#startBtn:hover {{ background: #3a7a52; }}
QPushButton#stopBtn {{ background: #5a2d2d; border-color: {T.RED}; }}
QPushButton#stopBtn:hover {{ background: #7a3a3a; }}
QPushButton#resetBtn {{ background: #5a4a2d; border-color: {T.YEL}; }}
QPushButton#resetBtn:hover {{ background: #7a6a3a; }}
QComboBox {{
    background: {T.S2};
    border: 1px solid {T.OVR};
    border-radius: 6px;
    padding: 5px 8px;
    color: {T.TXT};
    font-size: 11px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {T.S2};
    color: {T.TXT};
    selection-background-color: {T.PRI};
}}
QDoubleSpinBox, QSpinBox {{
    background: {T.S2};
    border: 1px solid {T.OVR};
    border-radius: 6px;
    padding: 3px 6px;
    color: {T.TXT};
    font-size: 11px;
}}
QLineEdit {{
    background: {T.S2};
    border: 1px solid {T.OVR};
    border-radius: 6px;
    padding: 4px 8px;
    color: {T.TXT};
    font-size: 11px;
}}
QLabel {{ color: {T.SUB}; }}
QLabel#desc {{ color: {T.DIM}; font-size: 10px; font-style: italic; padding: 2px; }}
QLabel#tip {{ color: {T.TEAL}; font-size: 10px; padding: 4px; }}
QLabel#info {{ color: {T.PRI}; font-size: 10px; padding: 4px; }}
QLabel#eigen_info {{ color: {T.MAU}; font-size: 10px; padding: 4px; }}
QCheckBox {{ color: {T.SUB}; spacing: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {T.OVR}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {T.PRI}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QTabWidget::pane {{ border: 1px solid {T.OVR}; border-radius: 6px; background: {T.S1}; }}
QTabBar::tab {{
    background: {T.S2};
    color: {T.SUB};
    padding: 6px 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-size: 10px;
}}
QTabBar::tab:selected {{ background: {T.S1}; color: {T.PRI}; border-bottom: 2px solid {T.PRI}; }}
QTableWidget {{
    background: {T.S1};
    gridline-color: {T.OVR};
    color: {T.TXT};
    font-size: 10px;
    border: 1px solid {T.OVR};
    border-radius: 4px;
}}
QTableWidget::item {{ padding: 2px; }}
QHeaderView::section {{ background: {T.S2}; color: {T.SUB}; padding: 4px; border: 1px solid {T.OVR}; font-size: 10px; }}
QScrollBar:vertical {{ background: {T.S1}; width: 8px; border: none; }}
QScrollBar::handle:vertical {{ background: {T.OVR}; border-radius: 4px; min-height: 20px; }}
QScrollArea {{ border: none; }}
QStatusBar {{ background: {T.S1}; color: {T.DIM}; font-size: 10px; border-top: 1px solid {T.OVR}; }}
QToolTip {{
    background: {T.S2};
    color: {T.TXT};
    border: 1px solid {T.OVR};
    padding: 6px;
    font-size: 11px;
}}
QTextEdit {{
    background: {T.S1};
    border: 1px solid {T.OVR};
    border-radius: 6px;
    color: {T.TXT};
    font-size: 11px;
    padding: 8px;
}}
"""