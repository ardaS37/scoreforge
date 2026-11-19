"""Background QThread workers for non-blocking AI transcription and MusicXML generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal

from core.export import generate_musicxml_string
from core.project import ScoreProject
from core.transcribe import AnalysisConfig, run_transcription_pipeline


class AnalysisWorker(QThread):
    """Runs the full AI audio-to-MIDI pipeline on a background thread."""

    progress = Signal(int, str)
    finished_project = Signal(object)   # ScoreProject
    failed = Signal(str)

    def __init__(
        self,
        audio_path: str | Path,
        config: AnalysisConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.audio_path = Path(audio_path)
        self.config = config
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            project = run_transcription_pipeline(
                audio_path=self.audio_path,
                config=self.config,
                progress_cb=lambda pct, msg: self.progress.emit(pct, msg),
                cancel_check=lambda: self._cancelled,
            )
            if not self._cancelled:
                self.finished_project.emit(project)
        except InterruptedError:
            self.failed.emit("Analiz kullanıcı tarafından iptal edildi.")
        except Exception as exc:
            self.failed.emit(f"Analiz hatası: {exc}")


class MusicXmlWorker(QThread):
    """Generates MusicXML string asynchronously for live Sheet Music preview."""

    finished_xml = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        project: ScoreProject,
        quantize_div: int = 4,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.quantize_div = quantize_div

    def run(self) -> None:
        try:
            xml_str = generate_musicxml_string(self.project, quantize_quarter_div=self.quantize_div)
            self.finished_xml.emit(xml_str)
        except Exception as exc:
            self.failed.emit(f"MusicXML oluşturma hatası: {exc}")
