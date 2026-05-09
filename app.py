"""Entry point for the Neural Network Training Visualizer."""

import sys
from PySide6.QtWidgets import QApplication
from theme import apply_theme
from ui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_theme(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()