"""Interactive Sheet Music (Nota Kağıdı) view powered by OpenSheetMusicDisplay + PDF Export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.export import export_musicxml
from core.project import ScoreProject
from ui.workers import MusicXmlWorker

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    HAS_WEBENGINE = True
except Exception:
    HAS_WEBENGINE = False


_OSMD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  html, body {
    margin: 0;
    padding: 0;
    background: #FFFFFF;
    color: #111111;
    font-family: "Segoe UI", sans-serif;
  }
  #status {
    padding: 36px;
    text-align: center;
    color: #555;
    font-size: 15px;
  }
  #osmd-container {
    padding: 12px 24px 48px 24px;
  }
</style>
<script src="https://cdn.jsdelivr.net/npm/opensheetmusicdisplay@1.8.9/build/opensheetmusicdisplay.min.js"></script>
</head>
<body>
<div id="status">Nota kağıdı oluşturmak için bir ses dosyası analiz edin veya "Yenile"ye tıklayın.</div>
<div id="osmd-container"></div>
<script>
  let osmd = null;
  let currentZoom = 0.9;

  async function renderMusicXml(xmlString) {
    const status = document.getElementById("status");
    const container = document.getElementById("osmd-container");
    if (typeof opensheetmusicdisplay === "undefined") {
      status.style.display = "block";
      status.innerText = "OpenSheetMusicDisplay yüklenemedi (internet bağlantısı gerekli). Yine de MusicXML dışa aktarabilirsiniz.";
      return;
    }
    try {
      status.style.display = "block";
      status.innerText = "Nota kağıdı çiziliyor...";
      container.innerHTML = "";
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, {
        autoResize: true,
        backend: "svg",
        drawTitle: true,
        drawComposer: true,
        drawPartNames: true,
        drawMeasureNumbers: true,
        drawCredits: true
      });
      await osmd.load(xmlString);
      osmd.zoom = currentZoom;
      osmd.render();
      status.style.display = "none";
    } catch (err) {
      status.style.display = "block";
      status.innerText = "Nota işleme hatası: " + err;
    }
  }

  function setSheetZoom(z) {
    currentZoom = Math.max(0.4, Math.min(2.0, z));
    if (osmd) {
      osmd.zoom = currentZoom;
      osmd.render();
    }
  }
</script>
</body>
</html>
"""


class SheetMusicView(QWidget):
    """Embedded Sheet Music preview with live MusicXML engraving and one-click PDF export."""

    status_message = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._project: Optional[ScoreProject] = None
        self._worker: Optional[MusicXmlWorker] = None
        self._zoom: float = 0.9
        self._page_ready: bool = False
        self._pending_xml: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Toolbar
        bar = QHBoxLayout()
        self.refresh_btn = QPushButton("Nota Kağıdını Yenile")
        self.refresh_btn.setObjectName("PrimaryBtn")
        self.refresh_btn.clicked.connect(self.refresh_sheet)

        bar.addWidget(self.refresh_btn)

        bar.addWidget(QLabel("Kuantizasyon:"))
        self.div_combo = QComboBox()
        self.div_combo.addItem("1/4 (Dörtlük)", 1)
        self.div_combo.addItem("1/8 (Sekizlik)", 2)
        self.div_combo.addItem("1/16 (Onaltılık)", 4)
        self.div_combo.setCurrentIndex(2)
        self.div_combo.currentIndexChanged.connect(lambda _: self.refresh_sheet())
        bar.addWidget(self.div_combo)

        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedWidth(30)
        self.zoom_out_btn.clicked.connect(lambda: self._adjust_zoom(-0.12))
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(30)
        self.zoom_in_btn.clicked.connect(lambda: self._adjust_zoom(0.12))
        bar.addWidget(self.zoom_out_btn)
        bar.addWidget(self.zoom_in_btn)

        bar.addStretch(1)

        self.export_xml_btn = QPushButton("MusicXML Kaydet")
        self.export_xml_btn.clicked.connect(self.export_musicxml_dialog)
        bar.addWidget(self.export_xml_btn)

        self.export_pdf_btn = QPushButton("PDF Kaydet")
        self.export_pdf_btn.setObjectName("PrimaryBtn")
        self.export_pdf_btn.clicked.connect(self.export_pdf_dialog)
        bar.addWidget(self.export_pdf_btn)

        layout.addLayout(bar)

        # Sheet renderer (skip Chromium WebEngine in headless/offscreen test mode)
        if HAS_WEBENGINE and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            self.web = QWebEngineView()
            self.web.loadFinished.connect(self._on_page_loaded)
            self.web.setHtml(_OSMD_HTML)
            layout.addWidget(self.web, 1)
        else:
            self.web = None
            fallback = QLabel(
                "Canlı nota önizlemesi hazır (QtWebEngine).\n"
                "MusicXML ve PDF dışa aktarımını üst araç çubuğundan kullanabilirsiniz."
            )
            fallback.setAlignment(Qt.AlignCenter)
            layout.addWidget(fallback, 1)

    def set_project(self, project: Optional[ScoreProject]) -> None:
        self._project = project
        if project is not None and project.total_notes() > 0:
            self.refresh_sheet()

    def _on_page_loaded(self, ok: bool) -> None:
        self._page_ready = ok
        if ok and self._pending_xml:
            xml = self._pending_xml
            self._pending_xml = None
            self._inject_xml(xml)

    def refresh_sheet(self) -> None:
        if self._project is None or self._project.total_notes() == 0:
            return
        if self._worker is not None and self._worker.isRunning():
            return

        self.status_message.emit("Nota kağıdı (MusicXML) hazırlanıyor...")
        div = int(self.div_combo.currentData())
        self._worker = MusicXmlWorker(self._project, quantize_div=div, parent=self)
        self._worker.finished_xml.connect(self._on_xml_ready)
        self._worker.failed.connect(lambda err: self.status_message.emit(err))
        self._worker.start()

    def _on_xml_ready(self, xml_str: str) -> None:
        self.status_message.emit("Nota kağıdı güncellendi.")
        if self.web is None:
            return
        if not self._page_ready:
            self._pending_xml = xml_str
        else:
            self._inject_xml(xml_str)

    def _inject_xml(self, xml_str: str) -> None:
        if self.web is None:
            return
        js_arg = json.dumps(xml_str)
        self.web.page().runJavaScript(f"renderMusicXml({js_arg});")

    def _adjust_zoom(self, delta: float) -> None:
        self._zoom = max(0.4, min(2.0, self._zoom + delta))
        if self.web is not None and self._page_ready:
            self.web.page().runJavaScript(f"setSheetZoom({self._zoom:.2f});")

    def export_musicxml_dialog(self) -> None:
        if self._project is None:
            return
        default_name = f"{self._project.title}.musicxml"
        path, _ = QFileDialog.getSaveFileName(
            self, "MusicXML Olarak Kaydet", default_name, "MusicXML (*.musicxml *.xml)"
        )
        if not path:
            return
        try:
            div = int(self.div_combo.currentData())
            export_musicxml(self._project, path, quantize_quarter_div=div)
            self.status_message.emit(f"MusicXML kaydedildi: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Hata", f"MusicXML kaydedilemedi:\n{exc}")

    def export_pdf_dialog(self) -> None:
        if self._project is None:
            return
        if self.web is None:
            QMessageBox.warning(self, "PDF", "PDF dışa aktarımı için QtWebEngine gereklidir.")
            return
        default_name = f"{self._project.title}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "PDF Olarak Kaydet", default_name, "PDF (*.pdf)"
        )
        if not path:
            return
        self.web.page().printToPdf(path)
        self.status_message.emit(f"PDF dışa aktarıldı: {Path(path).name}")
