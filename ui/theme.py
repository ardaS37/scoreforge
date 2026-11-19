"""Modern dark DAW / AnthemScore-inspired Qt stylesheet for ScoreForge."""

DARK_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #14161D;
    color: #E6E9F0;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QMenuBar {
    background-color: #101218;
    color: #D0D5E2;
    border-bottom: 1px solid #252A38;
    padding: 2px 6px;
}
QMenuBar::item:selected {
    background-color: #252B3B;
    border-radius: 4px;
}
QMenu {
    background-color: #1A1D27;
    border: 1px solid #2E3446;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px 6px 14px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #3A56D4;
    color: #FFFFFF;
}

QToolBar {
    background-color: #181B24;
    border-bottom: 1px solid #262C3D;
    spacing: 6px;
    padding: 4px 8px;
}

QPushButton {
    background-color: #232838;
    color: #E6E9F0;
    border: 1px solid #32394F;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #2C3347;
    border-color: #4CC9F0;
}
QPushButton:pressed {
    background-color: #1D2230;
}
QPushButton:checked {
    background-color: #3A56D4;
    border-color: #5C7CFA;
    color: #FFFFFF;
}
QPushButton#PrimaryBtn {
    background-color: #4361EE;
    border: 1px solid #5C7CFA;
    color: #FFFFFF;
    font-weight: 600;
}
QPushButton#PrimaryBtn:hover {
    background-color: #4E6EF2;
}
QPushButton#DangerBtn {
    background-color: #8B2635;
    border: 1px solid #B83347;
    color: #FFFFFF;
}
QPushButton#MuteBtn:checked {
    background-color: #E63946;
    border-color: #FF5C6C;
    color: #FFFFFF;
}
QPushButton#SoloBtn:checked {
    background-color: #F4A261;
    border-color: #FFB87A;
    color: #14161D;
    font-weight: 700;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1C202C;
    color: #E6E9F0;
    border: 1px solid #30374C;
    border-radius: 5px;
    padding: 4px 8px;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1C202C;
    border: 1px solid #3A4259;
    selection-background-color: #4361EE;
}

QSlider::groove:horizontal {
    height: 5px;
    background: #2A3042;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #4CC9F0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #E6E9F0;
    border: 1px solid #4CC9F0;
    width: 13px;
    height: 13px;
    margin: -4px 0;
    border-radius: 6px;
}

QTabWidget::pane {
    border: 1px solid #262C3D;
    background-color: #14161D;
}
QTabBar::tab {
    background-color: #1A1D27;
    color: #9AA3B8;
    padding: 7px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #252B3B;
    color: #4CC9F0;
    border-bottom: 2px solid #4CC9F0;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background: #14161D;
    width: 12px;
    height: 12px;
    margin: 0;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #2E3548;
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::handle:hover {
    background: #434D69;
}
QScrollBar::add-line, QScrollBar::sub-line {
    width: 0;
    height: 0;
}

QGroupBox {
    border: 1px solid #2A3042;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
    color: #B6C0D4;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QProgressBar {
    background-color: #1C202C;
    border: 1px solid #30374C;
    border-radius: 5px;
    text-align: center;
    color: #E6E9F0;
    height: 16px;
}
QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4361EE, stop:1 #4CC9F0);
    border-radius: 4px;
}

QStatusBar {
    background-color: #101218;
    color: #9AA3B8;
    border-top: 1px solid #252A38;
}

QLabel#BadgeLabel {
    background-color: #202636;
    border: 1px solid #323B52;
    border-radius: 5px;
    padding: 3px 9px;
    color: #4CC9F0;
    font-weight: 600;
}

QFrame#TrackCard {
    background-color: #1A1E2A;
    border: 1px solid #2A3144;
    border-radius: 7px;
}
QFrame#TrackCard[selected="true"] {
    border: 1.5px solid #4CC9F0;
    background-color: #202638;
}
"""
