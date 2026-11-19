"""Interactive CQT Spectrogram + Piano Roll + Chord Ruler + Velocity Editor widget."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QUndoCommand,
    QUndoStack,
    QWheelEvent,
)
from PySide6.QtWidgets import QWidget

from core.audio import render_spectrogram_rgba
from core.project import MidiTrack, NoteEvent, ScoreProject


MIDI_LOW = 21       # A0
MIDI_HIGH = 108     # C8
N_KEYS = MIDI_HIGH - MIDI_LOW + 1  # 88 keys
BLACK_PCS = {1, 3, 6, 8, 10}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{(pitch // 12) - 1}"


# ---------------------------------------------------------------------------
# Undo / Redo Commands
# ---------------------------------------------------------------------------

class AddNoteCommand(QUndoCommand):
    def __init__(self, track: MidiTrack, note: NoteEvent, on_change) -> None:
        super().__init__("Nota Ekle")
        self.track = track
        self.note = note
        self.on_change = on_change

    def redo(self) -> None:
        if self.note not in self.track.notes:
            self.track.notes.append(self.note)
            self.track.sort_notes()
        self.on_change()

    def undo(self) -> None:
        if self.note in self.track.notes:
            self.track.notes.remove(self.note)
        self.on_change()


class DeleteNotesCommand(QUndoCommand):
    def __init__(self, items: list[tuple[MidiTrack, NoteEvent]], on_change) -> None:
        super().__init__(f"{len(items)} Nota Sil")
        self.items = list(items)
        self.on_change = on_change

    def redo(self) -> None:
        for tr, n in self.items:
            if n in tr.notes:
                tr.notes.remove(n)
        self.on_change()

    def undo(self) -> None:
        touched = set()
        for tr, n in self.items:
            if n not in tr.notes:
                tr.notes.append(n)
                touched.add(id(tr))
        for tr, _ in self.items:
            if id(tr) in touched:
                tr.sort_notes()
        self.on_change()


class ModifyNotesCommand(QUndoCommand):
    """Generic undoable state change for move / resize / velocity edits."""

    def __init__(
        self,
        changes: list[tuple[NoteEvent, tuple[int, float, float, int], tuple[int, float, float, int]]],
        label: str,
        on_change,
    ) -> None:
        super().__init__(label)
        self.changes = changes  # (note, (old_p, old_s, old_e, old_v), (new_p, new_s, new_e, new_v))
        self.on_change = on_change

    def redo(self) -> None:
        for n, _, new_state in self.changes:
            n.pitch, n.start, n.end, n.velocity = new_state
        self.on_change()

    def undo(self) -> None:
        for n, old_state, _ in self.changes:
            n.pitch, n.start, n.end, n.velocity = old_state
        self.on_change()


class ReplaceTrackNotesCommand(QUndoCommand):
    """Undoable bulk replacement (used for Quantize / Scale Snap)."""

    def __init__(
        self,
        track: MidiTrack,
        new_notes: list[NoteEvent],
        label: str,
        on_change,
    ) -> None:
        super().__init__(label)
        self.track = track
        self.old_notes = list(track.notes)
        self.new_notes = list(new_notes)
        self.on_change = on_change

    def redo(self) -> None:
        self.track.notes = list(self.new_notes)
        self.on_change()

    def undo(self) -> None:
        self.track.notes = list(self.old_notes)
        self.on_change()


# ---------------------------------------------------------------------------
# Piano Roll + Spectrogram Widget
# ---------------------------------------------------------------------------

class PianoRollWidget(QWidget):
    """High-FPS custom QPainter widget combining CQT spectrogram, piano roll, chord ruler, and velocity lane."""

    notes_changed = Signal()
    playhead_scrubbed = Signal(float)
    audition_requested = Signal(int, int, bool)  # pitch, velocity, is_drum

    KEY_WIDTH = 62
    RULER_HEIGHT = 48
    VEL_HEIGHT = 78

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self._project: Optional[ScoreProject] = None
        self._active_track_idx: int = 0
        self.undo_stack = QUndoStack(self)

        # Viewport state
        self.px_per_sec: float = 120.0
        self.row_height: float = 14.0
        self.scroll_x: float = 0.0          # left visible time in seconds
        self.scroll_y: float = 28.0         # top visible row index (0 = C8, 87 = A0)

        # Spectrogram rendering cache
        self._spec_pixmap: Optional[QPixmap] = None
        self._spec_opacity: float = 0.65
        self._spec_colormap: str = "inferno"

        # Editing state
        self.tool_mode: str = "select"      # "select", "draw", "erase"
        self.snap_to_grid: bool = True
        self.grid_division: float = 0.25    # 16th note
        self._playhead: float = 0.0

        # Selection & active drag interaction
        self._selected: set[str] = set()    # NoteEvent.id
        self._drag_mode: Optional[str] = None  # "move", "resize", "draw", "rubberband", "pan", "velocity", "ruler", "erase"
        self._drag_start_pos: QPointF = QPointF()
        self._drag_anchor_time: float = 0.0
        self._drag_anchor_pitch: int = 60
        self._drag_orig_states: dict[str, tuple[NoteEvent, tuple[int, float, float, int]]] = {}
        self._drawing_note: Optional[NoteEvent] = None
        self._rubberband_rect: Optional[QRectF] = None
        self._pan_start_scroll: tuple[float, float] = (0.0, 0.0)
        self._last_audition_pitch: int = -1

    # ---- Public API ----

    def set_project(self, project: Optional[ScoreProject]) -> None:
        self._project = project
        self._selected.clear()
        self.undo_stack.clear()
        self.scroll_x = 0.0
        self.scroll_y = 24.0  # Center around C4–C5
        self._playhead = 0.0
        self._rebuild_spectrogram_pixmap()
        self.update()

    def set_active_track(self, idx: int) -> None:
        self._active_track_idx = idx
        self.update()

    def set_playhead(self, seconds: float, auto_follow: bool = True) -> None:
        self._playhead = max(0.0, seconds)
        if auto_follow:
            roll_w = max(1, self.width() - self.KEY_WIDTH)
            visible_sec = roll_w / self.px_per_sec
            if self._playhead > self.scroll_x + visible_sec * 0.88:
                self.scroll_x = max(0.0, self._playhead - visible_sec * 0.15)
            elif self._playhead < self.scroll_x:
                self.scroll_x = max(0.0, self._playhead - visible_sec * 0.1)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        self.tool_mode = mode
        if mode == "draw":
            self.setCursor(Qt.CrossCursor)
        elif mode == "erase":
            self.setCursor(Qt.ForbiddenCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_spectrogram_opacity(self, opacity: float) -> None:
        self._spec_opacity = float(np.clip(opacity, 0.0, 1.0))
        self.update()

    def set_spectrogram_colormap(self, cmap: str) -> None:
        if cmap != self._spec_colormap:
            self._spec_colormap = cmap
            self._rebuild_spectrogram_pixmap()
            self.update()

    def select_all(self) -> None:
        if self._project is None:
            return
        self._selected.clear()
        for tr in self._project.tracks:
            if tr.visible:
                for n in tr.active_notes():
                    self._selected.add(n.id)
        self.update()

    def delete_selected(self) -> None:
        if self._project is None or not self._selected:
            return
        to_delete: list[tuple[MidiTrack, NoteEvent]] = []
        for tr in self._project.tracks:
            for n in tr.notes:
                if n.id in self._selected:
                    to_delete.append((tr, n))
        if to_delete:
            self._selected.clear()
            self.undo_stack.push(DeleteNotesCommand(to_delete, self._emit_changed))

    def _emit_changed(self) -> None:
        self.notes_changed.emit()
        self.update()

    def _rebuild_spectrogram_pixmap(self) -> None:
        self._spec_pixmap = None
        if self._project is None or self._project.cqt_db is None:
            return
        rgba = render_spectrogram_rgba(self._project.cqt_db, colormap=self._spec_colormap)
        h, w, _ = rgba.shape
        qimg = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        self._spec_pixmap = QPixmap.fromImage(qimg)

    # ---- Coordinate Conversions ----

    def _roll_rect(self) -> QRectF:
        return QRectF(
            self.KEY_WIDTH,
            self.RULER_HEIGHT,
            max(1, self.width() - self.KEY_WIDTH),
            max(1, self.height() - self.RULER_HEIGHT - self.VEL_HEIGHT),
        )

    def _vel_rect(self) -> QRectF:
        return QRectF(
            self.KEY_WIDTH,
            self.height() - self.VEL_HEIGHT,
            max(1, self.width() - self.KEY_WIDTH),
            self.VEL_HEIGHT,
        )

    def _time_to_x(self, t: float) -> float:
        return self.KEY_WIDTH + (t - self.scroll_x) * self.px_per_sec

    def _x_to_time(self, x: float) -> float:
        return max(0.0, self.scroll_x + (x - self.KEY_WIDTH) / self.px_per_sec)

    def _pitch_to_y(self, pitch: int) -> float:
        row_idx = MIDI_HIGH - pitch
        return self.RULER_HEIGHT + (row_idx - self.scroll_y) * self.row_height

    def _y_to_pitch(self, y: float) -> int:
        row_idx = int((y - self.RULER_HEIGHT) / self.row_height + self.scroll_y)
        return int(np.clip(MIDI_HIGH - row_idx, MIDI_LOW, MIDI_HIGH))

    def _snap_time(self, t: float) -> float:
        if not self.snap_to_grid or self._project is None:
            return max(0.0, t)
        bpm = max(30.0, self._project.beat_grid.bpm or 120.0)
        step = (60.0 / bpm) * self.grid_division
        return max(0.0, round(t / step) * step)

    def _active_track(self) -> Optional[MidiTrack]:
        if self._project is None or not self._project.tracks:
            return None
        idx = max(0, min(self._active_track_idx, len(self._project.tracks) - 1))
        return self._project.tracks[idx]

    def _hit_test_note(self, pos: QPointF) -> tuple[Optional[MidiTrack], Optional[NoteEvent], bool]:
        """Return (track, note, is_right_edge) under mouse pos, prioritizing active track."""
        if self._project is None:
            return None, None, False

        ordered_tracks = list(self._project.tracks)
        act = self._active_track()
        if act in ordered_tracks:
            ordered_tracks.remove(act)
            ordered_tracks.append(act)  # Check active track on top (last drawn)

        for tr in reversed(ordered_tracks):
            if not tr.visible:
                continue
            for n in tr.active_notes():
                x0 = self._time_to_x(n.start)
                x1 = max(x0 + 5.0, self._time_to_x(n.end))
                y0 = self._pitch_to_y(n.pitch)
                y1 = y0 + self.row_height
                if x0 <= pos.x() <= x1 and y0 <= pos.y() <= y1:
                    is_edge = (x1 - pos.x()) <= 7.0
                    return tr, n, is_edge
        return None, None, False

    # ---- Painting ----

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)

        w, h = self.width(), self.height()
        roll = self._roll_rect()
        vel = self._vel_rect()

        # Background fill
        p.fillRect(0, 0, w, h, QColor("#12141B"))

        # 1. Piano Roll Background & Grid
        p.save()
        p.setClipRect(roll)
        self._draw_roll_background(p, roll)
        self._draw_spectrogram(p)
        self._draw_time_grid(p, roll)
        self._draw_notes(p, roll)

        if self._rubberband_rect is not None:
            p.setPen(QPen(QColor("#4CC9F0"), 1, Qt.DashLine))
            p.setBrush(QColor(76, 201, 240, 38))
            p.drawRect(self._rubberband_rect)
        p.restore()

        # 2. Left 88-Key Piano Keyboard
        p.save()
        p.setClipRect(QRectF(0, self.RULER_HEIGHT, self.KEY_WIDTH, roll.height()))
        self._draw_piano_keys(p, roll)
        p.restore()

        # 3. Top Ruler (Measures, Beats & Chords)
        self._draw_ruler(p, w)

        # 4. Bottom Velocity Lane
        self._draw_velocity_lane(p, vel)

        # 5. Playhead Line (across ruler, roll, and velocity)
        px = self._time_to_x(self._playhead)
        if self.KEY_WIDTH <= px <= w:
            p.setPen(QPen(QColor("#FF4D6D"), 2))
            p.drawLine(QPointF(px, 0), QPointF(px, h))
            # Playhead triangle cap in ruler
            p.setBrush(QColor("#FF4D6D"))
            p.setPen(Qt.NoPen)
            p.drawPolygon([QPointF(px - 6, 0), QPointF(px + 6, 0), QPointF(px, 10)])

        # Empty state hint
        if self._project is None:
            p.setPen(QColor("#6E778F"))
            f = QFont("Segoe UI", 14)
            p.setFont(f)
            p.drawText(
                roll,
                Qt.AlignCenter,
                "Başlamak için üstten 'Ses Aç' butonuna tıklayın\n(WAV, MP3, FLAC, OGG desteklenir)",
            )

        p.end()

    def _draw_roll_background(self, p: QPainter, roll: QRectF) -> None:
        first_row = max(0, int(self.scroll_y))
        last_row = min(N_KEYS - 1, int(self.scroll_y + roll.height() / self.row_height) + 1)

        for r in range(first_row, last_row + 1):
            pitch = MIDI_HIGH - r
            y = self.RULER_HEIGHT + (r - self.scroll_y) * self.row_height
            is_black = (pitch % 12) in BLACK_PCS
            bg = QColor("#151822") if is_black else QColor("#1B1F2B")
            p.fillRect(QRectF(roll.left(), y, roll.width(), self.row_height), bg)

            # Subtle horizontal divider; slightly brighter below C notes
            line_col = QColor("#2D3448") if (pitch % 12) == 0 else QColor("#202432")
            p.setPen(line_col)
            p.drawLine(QPointF(roll.left(), y + self.row_height), QPointF(roll.right(), y + self.row_height))

    def _draw_spectrogram(self, p: QPainter, ) -> None:
        if self._spec_pixmap is None or self._project is None or self._spec_opacity <= 0.01:
            return
        x0 = self._time_to_x(0.0)
        x1 = self._time_to_x(self._project.duration)
        y0 = self._pitch_to_y(MIDI_HIGH)
        y1 = self._pitch_to_y(MIDI_LOW) + self.row_height
        target = QRectF(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))

        p.save()
        p.setOpacity(self._spec_opacity)
        p.drawPixmap(target, self._spec_pixmap, QRectF(self._spec_pixmap.rect()))
        p.restore()

    def _draw_time_grid(self, p: QPainter, roll: QRectF) -> None:
        if self._project is None:
            return
        bpm = max(30.0, self._project.beat_grid.bpm or 120.0)
        ts_num = self._project.beat_grid.time_signature[0]
        beat_sec = 60.0 / bpm
        sub_sec = beat_sec * self.grid_division

        t_start = max(0.0, self.scroll_x)
        t_end = self._x_to_time(roll.right())

        # Sub-beat grid lines (if zoomed in enough)
        if sub_sec * self.px_per_sec >= 14.0:
            p.setPen(QColor(255, 255, 255, 16))
            k = int(t_start / sub_sec)
            while k * sub_sec <= t_end:
                x = self._time_to_x(k * sub_sec)
                p.drawLine(QPointF(x, roll.top()), QPointF(x, roll.bottom()))
                k += 1

        # Beat & Measure lines
        b = int(t_start / beat_sec)
        while b * beat_sec <= t_end:
            x = self._time_to_x(b * beat_sec)
            is_measure = (b % ts_num) == 0
            p.setPen(QColor(255, 255, 255, 65) if is_measure else QColor(255, 255, 255, 28))
            p.drawLine(QPointF(x, roll.top()), QPointF(x, roll.bottom()))
            b += 1

    def _draw_notes(self, p: QPainter, roll: QRectF) -> None:
        if self._project is None:
            return

        p.setRenderHint(QPainter.Antialiasing, True)
        font = QFont("Segoe UI", 8)
        p.setFont(font)

        act = self._active_track()
        t_vis_start = self.scroll_x - 1.0
        t_vis_end = self._x_to_time(roll.right()) + 1.0

        # Draw non-active tracks first (slightly translucent), active track on top
        tracks_order = [t for t in self._project.tracks if t is not act]
        if act is not None:
            tracks_order.append(act)

        for tr in tracks_order:
            if not tr.visible:
                continue
            is_active = tr is act
            base_color = QColor(tr.color)

            for n in tr.active_notes():
                if n.end < t_vis_start or n.start > t_vis_end:
                    continue
                if n.pitch < MIDI_LOW or n.pitch > MIDI_HIGH:
                    continue

                x0 = self._time_to_x(n.start)
                x1 = max(x0 + 5.0, self._time_to_x(n.end))
                y0 = self._pitch_to_y(n.pitch) + 1.0
                nh = max(4.0, self.row_height - 2.0)
                rect = QRectF(x0, y0, x1 - x0, nh)

                is_sel = n.id in self._selected
                col = QColor(base_color)
                alpha = 235 if is_active else 140
                # Modulate brightness gently by velocity
                vel_factor = 0.65 + 0.35 * (n.velocity / 127.0)
                col.setRgb(
                    int(min(255, col.red() * vel_factor)),
                    int(min(255, col.green() * vel_factor)),
                    int(min(255, col.blue() * vel_factor)),
                    alpha,
                )

                p.setBrush(QBrush(col))
                if is_sel:
                    p.setPen(QPen(QColor("#FFFFFF"), 1.8))
                else:
                    p.setPen(QPen(col.darker(155), 1.0))

                p.drawRoundedRect(rect, 3.0, 3.0)

                if rect.width() >= 28 and nh >= 11:
                    p.setPen(QColor("#FFFFFF"))
                    p.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignVCenter | Qt.AlignLeft, midi_to_name(n.pitch))

        p.setRenderHint(QPainter.Antialiasing, False)

    def _draw_piano_keys(self, p: QPainter, roll: QRectF) -> None:
        first_row = max(0, int(self.scroll_y))
        last_row = min(N_KEYS - 1, int(self.scroll_y + roll.height() / self.row_height) + 1)
        font = QFont("Segoe UI", 8)
        p.setFont(font)

        for r in range(first_row, last_row + 1):
            pitch = MIDI_HIGH - r
            y = self.RULER_HEIGHT + (r - self.scroll_y) * self.row_height
            is_black = (pitch % 12) in BLACK_PCS

            if is_black:
                p.fillRect(QRectF(0, y, self.KEY_WIDTH, self.row_height), QColor("#181A22"))
                p.fillRect(QRectF(0, y + 1, self.KEY_WIDTH * 0.66, self.row_height - 2), QColor("#0B0C10"))
            else:
                p.fillRect(QRectF(0, y, self.KEY_WIDTH, self.row_height), QColor("#DDE2EC"))
                p.setPen(QColor("#A3AAB8"))
                p.drawLine(QPointF(0, y + self.row_height), QPointF(self.KEY_WIDTH, y + self.row_height))

            if (pitch % 12) == 0 and self.row_height >= 9:
                p.setPen(QColor("#202636"))
                p.drawText(QRectF(4, y, self.KEY_WIDTH - 8, self.row_height), Qt.AlignVCenter | Qt.AlignRight, midi_to_name(pitch))

        # Right border
        p.setPen(QColor("#32394F"))
        p.drawLine(QPointF(self.KEY_WIDTH - 1, roll.top()), QPointF(self.KEY_WIDTH - 1, roll.bottom()))

    def _draw_ruler(self, p: QPainter, width: int) -> None:
        # Top-left corner block
        p.fillRect(QRectF(0, 0, self.KEY_WIDTH, self.RULER_HEIGHT), QColor("#161922"))
        p.fillRect(QRectF(self.KEY_WIDTH, 0, width - self.KEY_WIDTH, self.RULER_HEIGHT), QColor("#1A1E2B"))
        p.setPen(QColor("#2D3448"))
        p.drawLine(QPointF(0, self.RULER_HEIGHT - 1), QPointF(width, self.RULER_HEIGHT - 1))

        p.setPen(QColor("#7E879E"))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(4, 2, self.KEY_WIDTH - 8, 20), Qt.AlignVCenter | Qt.AlignLeft, "AKOR")
        p.drawText(QRectF(4, 24, self.KEY_WIDTH - 8, 20), Qt.AlignVCenter | Qt.AlignLeft, "ÖLÇÜ")

        if self._project is None:
            return

        p.save()
        p.setClipRect(QRectF(self.KEY_WIDTH, 0, width - self.KEY_WIDTH, self.RULER_HEIGHT))

        # Row 1 (0..24px): Detected Chord Badges
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        for ch in self._project.beat_grid.chords:
            x0 = self._time_to_x(ch.start)
            x1 = self._time_to_x(ch.end)
            if x1 < self.KEY_WIDTH or x0 > width:
                continue
            crect = QRectF(x0 + 1, 3, max(22.0, x1 - x0 - 2), 20)
            p.setBrush(QColor("#252D42"))
            p.setPen(QColor("#3E4C6E"))
            p.drawRoundedRect(crect, 4, 4)
            p.setPen(QColor("#4CC9F0"))
            p.drawText(crect.adjusted(5, 0, -3, 0), Qt.AlignVCenter | Qt.AlignLeft, ch.symbol)

        # Row 2 (24..48px): Measure & Beat numbers
        bpm = max(30.0, self._project.beat_grid.bpm or 120.0)
        ts_num = self._project.beat_grid.time_signature[0]
        beat_sec = 60.0 / bpm
        t_start = max(0.0, self.scroll_x)
        t_end = self._x_to_time(width)

        p.setFont(QFont("Segoe UI", 8))
        b = int(t_start / beat_sec)
        while b * beat_sec <= t_end:
            x = self._time_to_x(b * beat_sec)
            if (b % ts_num) == 0:
                measure_num = (b // ts_num) + 1
                p.setPen(QColor("#CBD2E2"))
                p.drawLine(QPointF(x, 26), QPointF(x, self.RULER_HEIGHT))
                p.drawText(QPointF(x + 4, 41), f"{measure_num}")
            else:
                p.setPen(QColor("#5B6378"))
                p.drawLine(QPointF(x, 38), QPointF(x, self.RULER_HEIGHT))
            b += 1

        p.restore()

    def _draw_velocity_lane(self, p: QPainter, vel: QRectF) -> None:
        p.fillRect(QRectF(0, vel.top(), self.KEY_WIDTH, vel.height()), QColor("#141720"))
        p.fillRect(vel, QColor("#11131A"))
        p.setPen(QColor("#2D3448"))
        p.drawLine(QPointF(0, vel.top()), QPointF(self.width(), vel.top()))

        p.setPen(QColor("#7E879E"))
        p.setFont(QFont("Segoe UI", 8, QFont.Bold))
        p.drawText(QRectF(4, vel.top() + 4, self.KEY_WIDTH - 8, 20), Qt.AlignLeft, "VELOCITY")

        if self._project is None:
            return

        p.save()
        p.setClipRect(vel)

        act = self._active_track()
        if act is not None and act.visible:
            col = QColor(act.color)
            for n in act.active_notes():
                x = self._time_to_x(n.start)
                if x < vel.left() - 6 or x > vel.right() + 6:
                    continue
                frac = np.clip(n.velocity / 127.0, 0.05, 1.0)
                bar_h = frac * (vel.height() - 12)
                y_top = vel.bottom() - 4 - bar_h
                is_sel = n.id in self._selected
                c = QColor("#FFFFFF") if is_sel else col
                p.setPen(QPen(c, 2.2))
                p.drawLine(QPointF(x, vel.bottom() - 4), QPointF(x, y_top))
                p.setBrush(c)
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(x, y_top), 3.2, 3.2)

        p.restore()

    # ---- Mouse & Keyboard Interaction ----

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        mods = event.modifiers()

        if mods & Qt.ControlModifier:
            # Horizontal zoom centered around mouse X
            anchor_t = self._x_to_time(event.position().x())
            factor = 1.15 if delta > 0 else (1.0 / 1.15)
            self.px_per_sec = float(np.clip(self.px_per_sec * factor, 25.0, 650.0))
            self.scroll_x = max(0.0, anchor_t - (event.position().x() - self.KEY_WIDTH) / self.px_per_sec)
        elif mods & Qt.ShiftModifier:
            # Vertical zoom
            factor = 1.12 if delta > 0 else (1.0 / 1.12)
            self.row_height = float(np.clip(self.row_height * factor, 6.0, 32.0))
        else:
            # Scroll vertically or horizontally
            if abs(event.angleDelta().x()) > abs(delta):
                dx = -event.angleDelta().x() / self.px_per_sec * 0.5
                self.scroll_x = max(0.0, self.scroll_x + dx)
            else:
                dy = -delta / 120.0 * 3.0
                max_sy = max(0.0, N_KEYS - self._roll_rect().height() / self.row_height)
                self.scroll_y = float(np.clip(self.scroll_y + dy, 0.0, max_sy))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        roll = self._roll_rect()
        vel = self._vel_rect()

        # Middle button pan
        if event.button() == Qt.MiddleButton:
            self._drag_mode = "pan"
            self._drag_start_pos = pos
            self._pan_start_scroll = (self.scroll_x, self.scroll_y)
            return

        # Left Piano Keyboard click -> audition note
        if pos.x() < self.KEY_WIDTH and roll.top() <= pos.y() <= roll.bottom():
            pitch = self._y_to_pitch(pos.y())
            act = self._active_track()
            self.audition_requested.emit(pitch, 100, bool(act.is_drum) if act else False)
            return

        # Top Ruler click -> scrub playhead
        if pos.y() < self.RULER_HEIGHT and pos.x() >= self.KEY_WIDTH:
            self._drag_mode = "ruler"
            t = self._x_to_time(pos.x())
            self._playhead = t
            self.playhead_scrubbed.emit(t)
            self.update()
            return

        if self._project is None:
            return

        # Bottom Velocity Lane click
        if vel.contains(pos):
            self._drag_mode = "velocity"
            self._edit_velocity_at(pos)
            return

        # Inside Piano Roll
        if not roll.contains(pos):
            return

        if event.button() == Qt.RightButton or self.tool_mode == "erase":
            self._drag_mode = "erase"
            self._erase_at(pos)
            return

        if self.tool_mode == "draw":
            act = self._active_track()
            if act is not None:
                t0 = self._snap_time(self._x_to_time(pos.x()))
                pitch = self._y_to_pitch(pos.y())
                bpm = max(30.0, self._project.beat_grid.bpm or 120.0)
                default_dur = (60.0 / bpm) * self.grid_division
                new_note = NoteEvent(pitch=pitch, start=t0, end=t0 + default_dur, velocity=95)
                self._drawing_note = new_note
                self._drag_mode = "draw"
                self._selected = {new_note.id}
                self.undo_stack.push(AddNoteCommand(act, new_note, self._emit_changed))
                self.audition_requested.emit(pitch, 95, act.is_drum)
            return

        # Select / Edit mode
        tr, hit_note, is_edge = self._hit_test_note(pos)
        if hit_note is not None and tr is not None:
            if not (event.modifiers() & Qt.ShiftModifier) and hit_note.id not in self._selected:
                self._selected = {hit_note.id}
            else:
                self._selected.add(hit_note.id)

            self._drag_mode = "resize" if is_edge else "move"
            self._drag_start_pos = pos
            self._drag_anchor_time = self._x_to_time(pos.x())
            self._drag_anchor_pitch = self._y_to_pitch(pos.y())
            self._last_audition_pitch = hit_note.pitch
            self._capture_selected_orig_states()
            self.audition_requested.emit(hit_note.pitch, hit_note.velocity, tr.is_drum)
            self.update()
        else:
            if not (event.modifiers() & Qt.ShiftModifier):
                self._selected.clear()
            self._drag_mode = "rubberband"
            self._drag_start_pos = pos
            self._rubberband_rect = QRectF(pos, pos)
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._project is None or not self._roll_rect().contains(pos):
            return
        act = self._active_track()
        if act is None:
            return
        t0 = self._snap_time(self._x_to_time(pos.x()))
        pitch = self._y_to_pitch(pos.y())
        bpm = max(30.0, self._project.beat_grid.bpm or 120.0)
        dur = (60.0 / bpm) * self.grid_division
        new_note = NoteEvent(pitch=pitch, start=t0, end=t0 + dur, velocity=95)
        self._selected = {new_note.id}
        self.undo_stack.push(AddNoteCommand(act, new_note, self._emit_changed))
        self.audition_requested.emit(pitch, 95, act.is_drum)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        if self._drag_mode is None:
            # Update hover cursor on note resize edge
            if self.tool_mode == "select" and self._roll_rect().contains(pos):
                _, hit_n, is_edge = self._hit_test_note(pos)
                if hit_n is not None and is_edge:
                    self.setCursor(Qt.SizeHorCursor)
                elif hit_n is not None:
                    self.setCursor(Qt.OpenHandCursor)
                else:
                    self.setCursor(Qt.ArrowCursor)
            return

        if self._drag_mode == "pan":
            dx = (pos.x() - self._drag_start_pos.x()) / self.px_per_sec
            dy = (pos.y() - self._drag_start_pos.y()) / self.row_height
            self.scroll_x = max(0.0, self._pan_start_scroll[0] - dx)
            max_sy = max(0.0, N_KEYS - self._roll_rect().height() / self.row_height)
            self.scroll_y = float(np.clip(self._pan_start_scroll[1] - dy, 0.0, max_sy))
            self.update()
        elif self._drag_mode == "ruler":
            t = self._x_to_time(pos.x())
            self._playhead = t
            self.playhead_scrubbed.emit(t)
            self.update()
        elif self._drag_mode == "erase":
            self._erase_at(pos)
        elif self._drag_mode == "velocity":
            self._edit_velocity_at(pos)
        elif self._drag_mode == "draw" and self._drawing_note is not None:
            t1 = self._snap_time(self._x_to_time(pos.x()))
            self._drawing_note.end = max(self._drawing_note.start + 0.04, t1)
            self.update()
        elif self._drag_mode == "move":
            dt = self._x_to_time(pos.x()) - self._drag_anchor_time
            dp = self._y_to_pitch(pos.y()) - self._drag_anchor_pitch
            for n, (op, os_, oe, ov) in self._drag_orig_states.values():
                new_start = max(0.0, self._snap_time(os_ + dt) if self.snap_to_grid else (os_ + dt))
                dur = oe - os_
                n.start = round(new_start, 4)
                n.end = round(new_start + dur, 4)
                n.pitch = int(np.clip(op + dp, MIDI_LOW, MIDI_HIGH))
            cur_p = self._y_to_pitch(pos.y())
            if cur_p != self._last_audition_pitch:
                self._last_audition_pitch = cur_p
                act = self._active_track()
                self.audition_requested.emit(cur_p, 85, bool(act.is_drum) if act else False)
            self.update()
        elif self._drag_mode == "resize":
            dt = self._x_to_time(pos.x()) - self._drag_anchor_time
            for n, (op, os_, oe, ov) in self._drag_orig_states.values():
                new_end = self._snap_time(oe + dt) if self.snap_to_grid else (oe + dt)
                n.end = round(max(os_ + 0.03, new_end), 4)
            self.update()
        elif self._drag_mode == "rubberband":
            self._rubberband_rect = QRectF(self._drag_start_pos, pos).normalized()
            self._update_rubberband_selection()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode in ("move", "resize") and self._drag_orig_states:
            changes = []
            for n, old_state in self._drag_orig_states.values():
                new_state = (n.pitch, n.start, n.end, n.velocity)
                if new_state != old_state:
                    # Temporarily restore old_state so ModifyNotesCommand.redo applies cleanly
                    n.pitch, n.start, n.end, n.velocity = old_state
                    changes.append((n, old_state, new_state))
            if changes:
                label = "Nota Taşı" if self._drag_mode == "move" else "Nota Süresi Değiştir"
                self.undo_stack.push(ModifyNotesCommand(changes, label, self._emit_changed))
        elif self._drag_mode in ("draw", "velocity"):
            self._emit_changed()

        self._drag_mode = None
        self._drawing_note = None
        self._rubberband_rect = None
        self._drag_orig_states.clear()
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif event.matches(QKeyEvent.Undo):
            self.undo_stack.undo()
        elif event.matches(QKeyEvent.Redo):
            self.undo_stack.redo()
        elif event.matches(QKeyEvent.SelectAll):
            self.select_all()
        else:
            super().keyPressEvent(event)

    # ---- Helpers ----

    def _capture_selected_orig_states(self) -> None:
        self._drag_orig_states.clear()
        if self._project is None:
            return
        for tr in self._project.tracks:
            for n in tr.notes:
                if n.id in self._selected:
                    self._drag_orig_states[n.id] = (n, (n.pitch, n.start, n.end, n.velocity))

    def _erase_at(self, pos: QPointF) -> None:
        tr, n, _ = self._hit_test_note(pos)
        if tr is not None and n is not None:
            self._selected.discard(n.id)
            self.undo_stack.push(DeleteNotesCommand([(tr, n)], self._emit_changed))

    def _edit_velocity_at(self, pos: QPointF) -> None:
        act = self._active_track()
        if act is None:
            return
        vel = self._vel_rect()
        frac = 1.0 - (pos.y() - vel.top() - 6.0) / max(1.0, vel.height() - 12.0)
        new_v = int(np.clip(round(frac * 127.0), 10, 127))
        for n in act.active_notes():
            x = self._time_to_x(n.start)
            if abs(x - pos.x()) <= 10.0:
                n.velocity = new_v
        self.update()

    def _update_rubberband_selection(self) -> None:
        if self._project is None or self._rubberband_rect is None:
            return
        rb = self._rubberband_rect
        self._selected.clear()
        for tr in self._project.tracks:
            if not tr.visible:
                continue
            for n in tr.active_notes():
                x0 = self._time_to_x(n.start)
                x1 = max(x0 + 5.0, self._time_to_x(n.end))
                y0 = self._pitch_to_y(n.pitch)
                nrect = QRectF(x0, y0, x1 - x0, self.row_height)
                if rb.intersects(nrect):
                    self._selected.add(n.id)
