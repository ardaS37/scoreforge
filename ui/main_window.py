"""Main desktop window for ScoreForge (AnthemScore+ Audio-to-MIDI Suite)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.audio import compute_cqt_spectrogram, load_audio
from core.export import export_midi, export_musicxml
from core.playback import SyncPlaybackEngine
from core.project import ScoreProject
from core.stems import get_device_info
from core.theory import quantize_notes
from core.transcribe import AnalysisConfig
from ui.analysis_dialog import AnalysisConfigDialog
from ui.piano_roll import PianoRollWidget, ReplaceTrackNotesCommand
from ui.sheet_view import SheetMusicView
from ui.track_panel import TrackPanel
from ui.workers import AnalysisWorker


def _fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    m = int(sec // 60)
    s = sec % 60.0
    return f"{m:02d}:{s:04.1f}"


class MainWindow(QMainWindow):
    """ScoreForge main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ScoreForge — Gelişmiş Audio-to-MIDI & Nota Stüdyosu")
        self.resize(1420, 860)

        self._project: Optional[ScoreProject] = None
        self._last_config = AnalysisConfig()
        self._worker: Optional[AnalysisWorker] = None
        self.playback = SyncPlaybackEngine()

        self._build_menus()
        self._build_toolbars()
        self._build_central_ui()
        self._build_statusbar()

        # 30 FPS playhead & time display refresh timer
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._on_playback_tick)
        self._ui_timer.start()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File Menu
        m_file = mb.addMenu("Dosya")
        act_open_audio = QAction("Ses Dosyası Aç ve Analiz Et...", self)
        act_open_audio.setShortcut(QKeySequence.Open)
        act_open_audio.triggered.connect(self.open_audio_dialog)
        m_file.addAction(act_open_audio)

        act_open_proj = QAction("Proje Aç (.sforge)...", self)
        act_open_proj.triggered.connect(self.open_project_dialog)
        m_file.addAction(act_open_proj)

        act_save_proj = QAction("Projeyi Kaydet (.sforge)...", self)
        act_save_proj.setShortcut(QKeySequence.Save)
        act_save_proj.triggered.connect(self.save_project_dialog)
        m_file.addAction(act_save_proj)

        m_file.addSeparator()

        act_exp_midi = QAction("MIDI Dışa Aktar (.mid)...", self)
        act_exp_midi.setShortcut("Ctrl+E")
        act_exp_midi.triggered.connect(self.export_midi_dialog)
        m_file.addAction(act_exp_midi)

        act_exp_xml = QAction("MusicXML Dışa Aktar (.musicxml)...", self)
        act_exp_xml.triggered.connect(lambda: self.sheet_view.export_musicxml_dialog())
        m_file.addAction(act_exp_xml)

        act_exp_pdf = QAction("Nota Kağıdı PDF Dışa Aktar (.pdf)...", self)
        act_exp_pdf.triggered.connect(lambda: self.sheet_view.export_pdf_dialog())
        m_file.addAction(act_exp_pdf)

        m_file.addSeparator()
        act_quit = QAction("Çıkış", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # Edit Menu
        m_edit = mb.addMenu("Düzenle")
        self.act_undo = QAction("Geri Al", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(lambda: self.piano_roll.undo_stack.undo())
        m_edit.addAction(self.act_undo)

        self.act_redo = QAction("İleri Al", self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.triggered.connect(lambda: self.piano_roll.undo_stack.redo())
        m_edit.addAction(self.act_redo)

        m_edit.addSeparator()
        act_sel_all = QAction("Tüm Notaları Seç", self)
        act_sel_all.setShortcut(QKeySequence.SelectAll)
        act_sel_all.triggered.connect(lambda: self.piano_roll.select_all())
        m_edit.addAction(act_sel_all)

        act_del = QAction("Seçili Notaları Sil", self)
        act_del.setShortcut(QKeySequence.Delete)
        act_del.triggered.connect(lambda: self.piano_roll.delete_selected())
        m_edit.addAction(act_del)

        m_edit.addSeparator()
        act_quant = QAction("Aktif Kanalı Kuantize Et", self)
        act_quant.setShortcut("Ctrl+Q")
        act_quant.triggered.connect(lambda: self.apply_quantize(snap_scale=False))
        m_edit.addAction(act_quant)

        act_scale = QAction("Aktif Kanalı Algılanan Tona Oturt (Scale Snap)", self)
        act_scale.triggered.connect(lambda: self.apply_quantize(snap_scale=True))
        m_edit.addAction(act_scale)

        # Analysis Menu
        m_ai = mb.addMenu("Analiz")
        act_reanalyze = QAction("Mevcut Sesi Yeniden Analiz Et...", self)
        act_reanalyze.triggered.connect(self.reanalyze_current_audio)
        m_ai.addAction(act_reanalyze)

        # Playback Menu
        m_play = mb.addMenu("Oynatma")
        act_play = QAction("Oynat / Duraklat", self)
        act_play.setShortcut(Qt.Key_Space)
        act_play.triggered.connect(self.toggle_play)
        m_play.addAction(act_play)

        act_stop = QAction("Durdur", self)
        act_stop.triggered.connect(self.stop_playback)
        m_play.addAction(act_stop)

    def _build_toolbars(self) -> None:
        # Row 1: Transport, Theory & Audio/MIDI Mix
        tb1 = QToolBar("Transport")
        tb1.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb1)

        self.open_btn = QPushButton("Ses Aç")
        self.open_btn.setObjectName("PrimaryBtn")
        self.open_btn.clicked.connect(self.open_audio_dialog)
        tb1.addWidget(self.open_btn)

        self.reanalyze_btn = QPushButton("AI Analiz")
        self.reanalyze_btn.setEnabled(False)
        self.reanalyze_btn.clicked.connect(self.reanalyze_current_audio)
        tb1.addWidget(self.reanalyze_btn)

        tb1.addSeparator()

        self.play_btn = QPushButton("▶ Oynat")
        self.play_btn.setFixedWidth(92)
        self.play_btn.clicked.connect(self.toggle_play)
        tb1.addWidget(self.play_btn)

        self.stop_btn = QPushButton("⏹ Durdur")
        self.stop_btn.clicked.connect(self.stop_playback)
        tb1.addWidget(self.stop_btn)

        self.loop_btn = QPushButton("Döngü")
        self.loop_btn.setCheckable(True)
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        tb1.addWidget(self.loop_btn)

        self.time_lbl = QLabel("00:00.0 / 00:00.0")
        self.time_lbl.setObjectName("BadgeLabel")
        self.time_lbl.setFixedWidth(135)
        self.time_lbl.setAlignment(Qt.AlignCenter)
        tb1.addWidget(self.time_lbl)

        tb1.addSeparator()

        tb1.addWidget(QLabel(" BPM: "))
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(30.0, 260.0)
        self.bpm_spin.setDecimals(1)
        self.bpm_spin.setValue(120.0)
        self.bpm_spin.setFixedWidth(74)
        self.bpm_spin.valueChanged.connect(self._on_bpm_changed)
        tb1.addWidget(self.bpm_spin)

        self.ts_lbl = QLabel("4/4")
        self.ts_lbl.setObjectName("BadgeLabel")
        tb1.addWidget(self.ts_lbl)

        self.key_lbl = QLabel("Ton: —")
        self.key_lbl.setObjectName("BadgeLabel")
        tb1.addWidget(self.key_lbl)

        tb1.addSeparator()

        # Audio <-> MIDI Mix slider
        tb1.addWidget(QLabel(" Ses "))
        self.mix_slider = QSlider(Qt.Horizontal)
        self.mix_slider.setRange(0, 100)
        self.mix_slider.setValue(50)
        self.mix_slider.setFixedWidth(120)
        self.mix_slider.setToolTip("Sol: Orijinal Ses | Orta: Senkron | Sağ: Sadece MIDI")
        self.mix_slider.valueChanged.connect(lambda v: self.playback.set_mix(v / 100.0))
        tb1.addWidget(self.mix_slider)
        tb1.addWidget(QLabel(" MIDI "))

        tb1.addSeparator()

        self.export_midi_btn = QPushButton("MIDI Kaydet")
        self.export_midi_btn.setObjectName("PrimaryBtn")
        self.export_midi_btn.setEnabled(False)
        self.export_midi_btn.clicked.connect(self.export_midi_dialog)
        tb1.addWidget(self.export_midi_btn)

        # Row 2: Editor Tools, Grid, Quantize & Spectrogram Controls
        self.addToolBarBreak(Qt.TopToolBarArea)
        tb2 = QToolBar("Editor")
        tb2.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb2)

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)

        self.btn_select = QPushButton("Seç (V)")
        self.btn_select.setCheckable(True)
        self.btn_select.setChecked(True)
        self.btn_select.clicked.connect(lambda: self.piano_roll.set_tool_mode("select"))

        self.btn_draw = QPushButton("Çiz (B)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(lambda: self.piano_roll.set_tool_mode("draw"))

        self.btn_erase = QPushButton("Sil (E)")
        self.btn_erase.setCheckable(True)
        self.btn_erase.clicked.connect(lambda: self.piano_roll.set_tool_mode("erase"))

        for b in (self.btn_select, self.btn_draw, self.btn_erase):
            self.tool_group.addButton(b)
            tb2.addWidget(b)

        tb2.addSeparator()

        self.snap_btn = QPushButton("Snap")
        self.snap_btn.setCheckable(True)
        self.snap_btn.setChecked(True)
        self.snap_btn.toggled.connect(self._on_snap_toggled)
        tb2.addWidget(self.snap_btn)

        self.grid_combo = QComboBox()
        self.grid_combo.addItem("1/4", 1.0)
        self.grid_combo.addItem("1/8", 0.5)
        self.grid_combo.addItem("1/16", 0.25)
        self.grid_combo.addItem("1/32", 0.125)
        self.grid_combo.addItem("1/8T", 1.0 / 3.0)
        self.grid_combo.setCurrentIndex(2)
        self.grid_combo.currentIndexChanged.connect(self._on_grid_changed)
        tb2.addWidget(self.grid_combo)

        self.quant_btn = QPushButton("Kuantize Et")
        self.quant_btn.clicked.connect(lambda: self.apply_quantize(snap_scale=False))
        tb2.addWidget(self.quant_btn)

        self.scale_snap_btn = QPushButton("Tona Oturt")
        self.scale_snap_btn.setToolTip("Aktif kanaldaki notaları algılanan gama oturtur")
        self.scale_snap_btn.clicked.connect(lambda: self.apply_quantize(snap_scale=True))
        tb2.addWidget(self.scale_snap_btn)

        tb2.addSeparator()

        tb2.addWidget(QLabel(" Spektrogram: "))
        self.spec_slider = QSlider(Qt.Horizontal)
        self.spec_slider.setRange(0, 100)
        self.spec_slider.setValue(65)
        self.spec_slider.setFixedWidth(100)
        self.spec_slider.valueChanged.connect(lambda v: self.piano_roll.set_spectrogram_opacity(v / 100.0))
        tb2.addWidget(self.spec_slider)

        self.cmap_combo = QComboBox()
        for cmap in ("inferno", "magma", "viridis", "plasma"):
            self.cmap_combo.addItem(cmap.title(), cmap)
        self.cmap_combo.currentIndexChanged.connect(
            lambda _: self.piano_roll.set_spectrogram_colormap(self.cmap_combo.currentData())
        )
        tb2.addWidget(self.cmap_combo)

    def _build_central_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)

        self.track_panel = TrackPanel()
        self.track_panel.active_track_changed.connect(self._on_active_track_changed)
        self.track_panel.tracks_modified.connect(self._on_tracks_modified)
        splitter.addWidget(self.track_panel)

        self.tabs = QTabWidget()

        self.piano_roll = PianoRollWidget()
        self.piano_roll.notes_changed.connect(self._on_notes_changed)
        self.piano_roll.playhead_scrubbed.connect(self._on_playhead_scrubbed)
        self.piano_roll.audition_requested.connect(self.playback.audition_note)
        self.tabs.addTab(self.piano_roll, "Spektrogram & Piano Roll")

        self.sheet_view = SheetMusicView()
        self.sheet_view.status_message.connect(lambda m: self.statusBar().showMessage(m, 5000))
        self.tabs.addTab(self.sheet_view, "Nota Kağıdı (Sheet Music)")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    def _build_statusbar(self) -> None:
        sb = self.statusBar()
        dev = get_device_info()
        gpu_txt = f"CUDA: {dev['device_name']}" if dev["cuda_available"] else "CPU Modu"
        self.gpu_lbl = QLabel(f" {gpu_txt} ")
        self.gpu_lbl.setStyleSheet("color: #4CC9F0; font-weight: 600;")
        sb.addPermanentWidget(self.gpu_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setVisible(False)
        sb.addPermanentWidget(self.progress_bar)

        self.cancel_btn = QPushButton("İptal")
        self.cancel_btn.setObjectName("DangerBtn")
        self.cancel_btn.setFixedHeight(22)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_analysis)
        sb.addPermanentWidget(self.cancel_btn)

        sb.showMessage("Hazır — Başlamak için bir ses dosyası açın.")

    # ------------------------------------------------------------------
    # Audio & Project Lifecycle
    # ------------------------------------------------------------------

    def open_audio_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ses Dosyası Seç",
            "",
            "Ses Dosyaları (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;Tüm Dosyalar (*.*)",
        )
        if not path:
            return
        self._start_analysis_flow(Path(path))

    def reanalyze_current_audio(self) -> None:
        if self._project is None or not self._project.audio_path:
            return
        self._start_analysis_flow(Path(self._project.audio_path))

    def _start_analysis_flow(self, audio_path: Path) -> None:
        dlg = AnalysisConfigDialog(self._last_config, self)
        if dlg.exec() != AnalysisConfigDialog.Accepted:
            return

        self._last_config = dlg.get_config()
        self.stop_playback()

        self.open_btn.setEnabled(False)
        self.reanalyze_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.cancel_btn.setVisible(True)

        self._worker = AnalysisWorker(audio_path, self._last_config, parent=self)
        self._worker.progress.connect(self._on_analysis_progress)
        self._worker.finished_project.connect(self._on_analysis_finished)
        self._worker.failed.connect(self._on_analysis_failed)
        self._worker.start()

    def _cancel_analysis(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.statusBar().showMessage("Analiz iptal ediliyor...")

    def _on_analysis_progress(self, pct: int, msg: str) -> None:
        self.progress_bar.setValue(pct)
        self.statusBar().showMessage(msg)

    def _on_analysis_finished(self, project: ScoreProject) -> None:
        self._finish_worker_ui()
        self._load_project_into_ui(project)
        self.statusBar().showMessage(
            f"Analiz tamamlandı: {project.total_notes()} nota, {len(project.tracks)} kanal "
            f"({project.beat_grid.bpm} BPM, {project.beat_grid.key_name})",
            8000,
        )

    def _on_analysis_failed(self, err: str) -> None:
        self._finish_worker_ui()
        self.statusBar().showMessage(err, 8000)
        QMessageBox.warning(self, "Analiz", err)

    def _finish_worker_ui(self) -> None:
        self.open_btn.setEnabled(True)
        self.reanalyze_btn.setEnabled(self._project is not None and bool(self._project.audio_path))
        self.progress_bar.setVisible(False)
        self.cancel_btn.setVisible(False)

    def _load_project_into_ui(self, project: ScoreProject) -> None:
        self._project = project
        self.setWindowTitle(f"ScoreForge — {project.title}")
        self.reanalyze_btn.setEnabled(bool(project.audio_path))
        self.export_midi_btn.setEnabled(True)

        self.bpm_spin.blockSignals(True)
        self.bpm_spin.setValue(project.beat_grid.bpm)
        self.bpm_spin.blockSignals(False)

        ts_n, ts_d = project.beat_grid.time_signature
        self.ts_lbl.setText(f"{ts_n}/{ts_d}")
        self.key_lbl.setText(f"Ton: {project.beat_grid.key_name}")
        self.time_lbl.setText(f"00:00.0 / {_fmt_time(project.duration)}")

        self.track_panel.set_project(project)
        self.piano_roll.set_project(project)
        self.playback.set_project(project)
        self.sheet_view.set_project(project)

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "ScoreForge Projesi Aç", "", "ScoreForge (*.sforge)")
        if not path:
            return
        try:
            proj = ScoreProject.load(path)
            if proj.audio_path and Path(proj.audio_path).exists():
                mono, stereo, sr = load_audio(proj.audio_path, target_sr=44100)
                proj.audio_mono = mono
                proj.audio_stereo = stereo
                proj.sample_rate = sr
                proj.cqt_db, proj.cqt_times = compute_cqt_spectrogram(mono, sr=sr)
            self._load_project_into_ui(proj)
            self.statusBar().showMessage(f"Proje yüklendi: {Path(path).name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", f"Proje açılamadı:\n{exc}")

    def save_project_dialog(self) -> None:
        if self._project is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Projeyi Kaydet", f"{self._project.title}.sforge", "ScoreForge (*.sforge)"
        )
        if not path:
            return
        try:
            self._project.save(path)
            self.statusBar().showMessage(f"Proje kaydedildi: {Path(path).name}", 5000)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", f"Proje kaydedilemedi:\n{exc}")

    def export_midi_dialog(self) -> None:
        if self._project is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "MIDI Olarak Kaydet", f"{self._project.title}.mid", "MIDI Dosyası (*.mid)"
        )
        if not path:
            return
        try:
            export_midi(self._project, path)
            self.statusBar().showMessage(f"MIDI dışa aktarıldı: {Path(path).name}", 6000)
        except Exception as exc:
            QMessageBox.critical(self, "Hata", f"MIDI kaydedilemedi:\n{exc}")

    # ------------------------------------------------------------------
    # Playback & Editor Callbacks
    # ------------------------------------------------------------------

    def toggle_play(self) -> None:
        if self._project is None:
            return
        if self.playback.is_playing:
            self.playback.pause()
            self.play_btn.setText("▶ Oynat")
        else:
            self.playback.play()
            self.play_btn.setText("⏸ Duraklat")

    def stop_playback(self) -> None:
        self.playback.stop()
        self.play_btn.setText("▶ Oynat")
        self.piano_roll.set_playhead(0.0, auto_follow=False)

    def _on_loop_toggled(self, checked: bool) -> None:
        if self._project is not None:
            self.playback.set_loop(checked, 0.0, self._project.duration)

    def _on_playback_tick(self) -> None:
        if self._project is None:
            return
        pos = self.playback.position_seconds
        self.time_lbl.setText(f"{_fmt_time(pos)} / {_fmt_time(self._project.duration)}")
        if self.playback.is_playing:
            self.piano_roll.set_playhead(pos, auto_follow=True)
        elif self.play_btn.text().startswith("⏸"):
            self.play_btn.setText("▶ Oynat")

    def _on_playhead_scrubbed(self, sec: float) -> None:
        self.playback.seek(sec)

    def _on_bpm_changed(self, val: float) -> None:
        if self._project is not None:
            self._project.beat_grid.bpm = float(val)
            self.piano_roll.update()

    def _on_snap_toggled(self, checked: bool) -> None:
        self.piano_roll.snap_to_grid = checked

    def _on_grid_changed(self, _idx: int) -> None:
        self.piano_roll.grid_division = float(self.grid_combo.currentData())
        self.piano_roll.update()

    def _on_active_track_changed(self, idx: int) -> None:
        self.piano_roll.set_active_track(idx)

    def _on_tracks_modified(self) -> None:
        self.playback.mark_midi_dirty()
        self.piano_roll.update()

    def _on_notes_changed(self) -> None:
        self.playback.mark_midi_dirty()
        self.track_panel.refresh_counts()

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1 and self._project is not None:
            self.sheet_view.refresh_sheet()

    def apply_quantize(self, snap_scale: bool = False) -> None:
        if self._project is None:
            return
        track = self._project.get_track(self.track_panel.active_track_index)
        if track is None or not track.notes:
            return

        div = float(self.grid_combo.currentData())
        bg = self._project.beat_grid
        new_notes = quantize_notes(
            track.notes,
            bpm=bg.bpm,
            division=div,
            strength=0.85,
            snap_to_scale=(snap_scale and not track.is_drum),
            key_tonic=bg.key_tonic,
            key_mode=bg.key_mode,
        )
        label = "Tona Oturt & Kuantize" if snap_scale else "Kuantize Et"
        self.piano_roll.undo_stack.push(
            ReplaceTrackNotesCommand(track, new_notes, label, self.piano_roll._emit_changed)
        )
        self.statusBar().showMessage(f"{track.name}: {label} uygulandı ({len(new_notes)} nota).", 4000)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_V:
            self.btn_select.click()
        elif event.key() == Qt.Key_B:
            self.btn_draw.click()
        elif event.key() == Qt.Key_E:
            self.btn_erase.click()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(1500)
        if self.sheet_view._worker is not None and self.sheet_view._worker.isRunning():
            self.sheet_view._worker.wait(2000)
        self.playback.close()
        super().closeEvent(event)
