"""Analysis configuration dialog for stem separation, AI sensitivity, and quantization."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)

from core.stems import get_device_info
from core.transcribe import AnalysisConfig


class AnalysisConfigDialog(QDialog):
    """Modal dialog to configure Demucs stem separation & Basic Pitch transcription settings."""

    def __init__(self, initial: AnalysisConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Analiz Ayarları — ScoreForge")
        self.setMinimumWidth(460)
        cfg = initial or AnalysisConfig()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Hardware info banner
        dev_info = get_device_info()
        hw_text = (
            f"GPU Algılandı: {dev_info['device_name']} ({dev_info['vram_gb']} GB VRAM — CUDA Aktif)"
            if dev_info["cuda_available"]
            else "GPU Bulunamadı: CPU modunda çalışacak"
        )
        hw_lbl = QLabel(hw_text)
        hw_lbl.setObjectName("BadgeLabel")
        layout.addWidget(hw_lbl)

        # 1. Stem Separation Group
        stem_grp = QGroupBox("1. Enstrüman Ayrıştırma (Demucs)")
        stem_form = QFormLayout(stem_grp)

        self.stem_mode_combo = QComboBox()
        self.stem_mode_combo.addItem("6 Stem (Piyano, Vokal, Gitar, Bas, Davul, Diğer) — Önerilen", "htdemucs_6s")
        self.stem_mode_combo.addItem("4 Stem (Vokal, Bas, Davul, Diğer) — Daha Hızlı", "htdemucs")
        self.stem_mode_combo.addItem("Ayrıştırma Yok (Solo Enstrüman / Hızlı Mod)", "none")

        if not cfg.separate_stems:
            self.stem_mode_combo.setCurrentIndex(2)
        elif cfg.demucs_model == "htdemucs":
            self.stem_mode_combo.setCurrentIndex(1)
        else:
            self.stem_mode_combo.setCurrentIndex(0)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Otomatik (CUDA varsa GPU)", "auto")
        self.device_combo.addItem("NVIDIA GPU (CUDA)", "cuda")
        self.device_combo.addItem("Sadece CPU", "cpu")
        if cfg.device == "cuda":
            self.device_combo.setCurrentIndex(1)
        elif cfg.device == "cpu":
            self.device_combo.setCurrentIndex(2)

        stem_form.addRow("Ayrıştırma Modu:", self.stem_mode_combo)
        stem_form.addRow("İşlemci:", self.device_combo)
        layout.addWidget(stem_grp)

        # 2. Transcription Sensitivity Group
        tr_grp = QGroupBox("2. Nota Algılama Hassasiyeti (Basic Pitch & Davul)")
        tr_form = QFormLayout(tr_grp)

        self.onset_slider, onset_row = self._make_slider(int(cfg.onset_threshold * 100), 15, 85)
        self.frame_slider, frame_row = self._make_slider(int(cfg.frame_threshold * 100), 10, 80)
        self.drum_slider, drum_row = self._make_slider(int(cfg.drum_sensitivity * 100), 10, 95)

        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(25.0, 400.0)
        self.min_dur_spin.setSingleStep(10.0)
        self.min_dur_spin.setSuffix(" ms")
        self.min_dur_spin.setValue(cfg.min_note_length_ms)

        self.pitch_bend_chk = QCheckBox("Pitch Bend (Kaydırma/Vibrato) kaydet")
        self.pitch_bend_chk.setChecked(cfg.include_pitch_bends)

        tr_form.addRow("Nota Başlangıç Eşiği (Onset):", onset_row)
        tr_form.addRow("Nota Devam Eşiği (Frame):", frame_row)
        tr_form.addRow("Davul Hassasiyeti:", drum_row)
        tr_form.addRow("Minimum Nota Süresi:", self.min_dur_spin)
        tr_form.addRow("", self.pitch_bend_chk)
        layout.addWidget(tr_grp)

        # 3. Post-processing / Quantization Group
        q_grp = QGroupBox("3. Otomatik Kuantizasyon")
        q_form = QFormLayout(q_grp)

        self.auto_quant_chk = QCheckBox("Analiz sonrası notaları ritim ızgarasına oturt")
        self.auto_quant_chk.setChecked(cfg.auto_quantize)

        self.quant_div_combo = QComboBox()
        self.quant_div_combo.addItem("1/4 (Dörtlük)", 1.0)
        self.quant_div_combo.addItem("1/8 (Sekizlik)", 0.5)
        self.quant_div_combo.addItem("1/16 (Onaltılık)", 0.25)
        self.quant_div_combo.addItem("1/32 (Otuzikilik)", 0.125)
        self.quant_div_combo.setCurrentIndex(2)

        q_form.addRow("", self.auto_quant_chk)
        q_form.addRow("Izgara Çözünürlüğü:", self.quant_div_combo)
        layout.addWidget(q_grp)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = buttons.button(QDialogButtonBox.Ok)
        ok_btn.setText("Analizi Başlat")
        ok_btn.setObjectName("PrimaryBtn")
        buttons.button(QDialogButtonBox.Cancel).setText("İptal")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_slider(self, value: int, lo: int, hi: int) -> tuple[QSlider, QHBoxLayout]:
        row = QHBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(value)
        lbl = QLabel(f"{value}%")
        lbl.setFixedWidth(38)
        slider.valueChanged.connect(lambda v: lbl.setText(f"{v}%"))
        row.addWidget(slider)
        row.addWidget(lbl)
        return slider, row

    def get_config(self) -> AnalysisConfig:
        mode = self.stem_mode_combo.currentData()
        return AnalysisConfig(
            separate_stems=(mode != "none"),
            demucs_model=mode if mode != "none" else "htdemucs_6s",
            device=self.device_combo.currentData(),
            onset_threshold=self.onset_slider.value() / 100.0,
            frame_threshold=self.frame_slider.value() / 100.0,
            min_note_length_ms=float(self.min_dur_spin.value()),
            drum_sensitivity=self.drum_slider.value() / 100.0,
            auto_quantize=self.auto_quant_chk.isChecked(),
            quantize_division=float(self.quant_div_combo.currentData()),
            include_pitch_bends=self.pitch_bend_chk.isChecked(),
        )
