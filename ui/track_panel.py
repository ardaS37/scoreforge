"""Left-hand Stem & MIDI Track Mixer panel."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from core.project import GM_INSTRUMENTS, MidiTrack, ScoreProject


class TrackCard(QFrame):
    """Single track control card in the mixer sidebar."""

    selected = Signal(int)
    changed = Signal()

    def __init__(self, index: int, track: MidiTrack, parent=None) -> None:
        super().__init__(parent)
        self.index = index
        self.track = track
        self.setObjectName("TrackCard")
        self.setProperty("selected", False)
        self.setCursor(Qt.PointingHandCursor)

        root = QVBoxLayout(self)
        root.setContentsMargins(9, 8, 9, 8)
        root.setSpacing(6)

        # Row 1: Color dot, Visibility checkbox, Name, Note count, Mute, Solo
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        color_dot = QLabel()
        color_dot.setFixedSize(11, 11)
        color_dot.setStyleSheet(f"background-color: {track.color}; border-radius: 5px;")
        row1.addWidget(color_dot)

        self.vis_chk = QCheckBox()
        self.vis_chk.setChecked(track.visible)
        self.vis_chk.setToolTip("Kanalı göster/gizle")
        self.vis_chk.toggled.connect(self._on_vis_changed)
        row1.addWidget(self.vis_chk)

        self.name_lbl = QLabel(f"<b>{track.name}</b>")
        row1.addWidget(self.name_lbl, 1)

        self.count_lbl = QLabel(f"{len(track.active_notes())} nota")
        self.count_lbl.setStyleSheet("color: #8F99B0; font-size: 11px;")
        row1.addWidget(self.count_lbl)

        self.mute_btn = QPushButton("M")
        self.mute_btn.setObjectName("MuteBtn")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(track.muted)
        self.mute_btn.setFixedSize(26, 22)
        self.mute_btn.setStyleSheet("padding: 0;")
        self.mute_btn.toggled.connect(self._on_mute_changed)
        row1.addWidget(self.mute_btn)

        self.solo_btn = QPushButton("S")
        self.solo_btn.setObjectName("SoloBtn")
        self.solo_btn.setCheckable(True)
        self.solo_btn.setChecked(track.solo)
        self.solo_btn.setFixedSize(26, 22)
        self.solo_btn.setStyleSheet("padding: 0;")
        self.solo_btn.toggled.connect(self._on_solo_changed)
        row1.addWidget(self.solo_btn)

        root.addLayout(row1)

        # Row 2: GM Instrument selector (for non-drum tracks)
        if not track.is_drum:
            self.inst_combo = QComboBox()
            self.inst_combo.setStyleSheet("font-size: 11px; padding: 2px 6px;")
            sel_idx = 0
            for idx_i, (prog, label) in enumerate(GM_INSTRUMENTS):
                self.inst_combo.addItem(f"{prog}: {label}", prog)
                if prog == track.program:
                    sel_idx = idx_i
            self.inst_combo.setCurrentIndex(sel_idx)
            self.inst_combo.currentIndexChanged.connect(self._on_program_changed)
            root.addWidget(self.inst_combo)

        # Row 3: Audio & MIDI volume sliders
        vol_row = QHBoxLayout()
        vol_row.setSpacing(6)
        vol_lbl = QLabel("MIDI")
        vol_lbl.setStyleSheet("color: #8F99B0; font-size: 11px;")
        vol_lbl.setFixedWidth(32)
        self.midi_vol = QSlider(Qt.Horizontal)
        self.midi_vol.setRange(0, 150)
        self.midi_vol.setValue(int(track.midi_volume * 100))
        self.midi_vol.valueChanged.connect(self._on_midi_vol_changed)
        vol_row.addWidget(vol_lbl)
        vol_row.addWidget(self.midi_vol)
        root.addLayout(vol_row)

        # Row 4: Live Confidence Filter slider
        conf_row = QHBoxLayout()
        conf_row.setSpacing(6)
        conf_lbl = QLabel("Eşik")
        conf_lbl.setStyleSheet("color: #8F99B0; font-size: 11px;")
        conf_lbl.setFixedWidth(32)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 90)
        self.conf_slider.setValue(int(track.min_confidence * 100))
        self.conf_slider.setToolTip("Düşük güvenilirlikli notaları canlı filtrele")
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(conf_lbl)
        conf_row.addWidget(self.conf_slider)
        root.addLayout(conf_row)

    def set_selected(self, sel: bool) -> None:
        self.setProperty("selected", sel)
        self.style().unpolish(self)
        self.style().polish(self)

    def refresh_count(self) -> None:
        self.count_lbl.setText(f"{len(self.track.active_notes())} nota")

    def mousePressEvent(self, event) -> None:
        self.selected.emit(self.index)
        super().mousePressEvent(event)

    def _on_vis_changed(self, checked: bool) -> None:
        self.track.visible = checked
        self.changed.emit()

    def _on_mute_changed(self, checked: bool) -> None:
        self.track.muted = checked
        self.changed.emit()

    def _on_solo_changed(self, checked: bool) -> None:
        self.track.solo = checked
        self.changed.emit()

    def _on_program_changed(self, _idx: int) -> None:
        self.track.program = int(self.inst_combo.currentData())
        self.changed.emit()

    def _on_midi_vol_changed(self, val: int) -> None:
        self.track.midi_volume = val / 100.0
        self.changed.emit()

    def _on_conf_changed(self, val: int) -> None:
        self.track.min_confidence = val / 100.0
        self.refresh_count()
        self.changed.emit()


class TrackPanel(QWidget):
    """Scrollable sidebar containing all Stem / MIDI tracks."""

    active_track_changed = Signal(int)
    tracks_modified = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(250)
        self._project: Optional[ScoreProject] = None
        self._cards: list[TrackCard] = []
        self._active_index: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QLabel("KANALLAR / STEMS")
        header.setStyleSheet("font-weight: 700; color: #8F99B0; letter-spacing: 1px; font-size: 11px;")
        layout.addWidget(header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.container = QWidget()
        self.cards_layout = QVBoxLayout(self.container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch(1)

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

    @property
    def active_track_index(self) -> int:
        return self._active_index

    def set_project(self, project: Optional[ScoreProject]) -> None:
        self._project = project
        self._cards.clear()

        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if project is None or not project.tracks:
            self._active_index = -1
            return

        for idx, track in enumerate(project.tracks):
            card = TrackCard(idx, track)
            card.selected.connect(self.select_track)
            card.changed.connect(self.tracks_modified.emit)
            self.cards_layout.insertWidget(idx, card)
            self._cards.append(card)

        self.select_track(0)

    def select_track(self, index: int) -> None:
        if not self._cards:
            self._active_index = -1
            return
        index = max(0, min(index, len(self._cards) - 1))
        self._active_index = index
        for i, card in enumerate(self._cards):
            card.set_selected(i == index)
        self.active_track_changed.emit(index)

    def refresh_counts(self) -> None:
        for card in self._cards:
            card.refresh_count()
