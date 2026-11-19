"""End-to-end verification tests for ScoreForge core engine and PySide6 UI."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure workspace root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _generate_synthetic_song(sr: int = 44100, bpm: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 4-second synthetic C-Am-F-G chord progression + simple kick/snare."""
    beat = 60.0 / bpm
    duration = 8 * beat  # 4.0 seconds (2 measures of 4/4)
    n_samples = int(duration * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr
    y = np.zeros(n_samples, dtype=np.float32)

    chords = [
        (0.0, 2 * beat, [48, 60, 64, 67]),      # C major
        (2 * beat, 4 * beat, [45, 57, 60, 64]), # A minor
        (4 * beat, 6 * beat, [41, 53, 57, 60]), # F major
        (6 * beat, 8 * beat, [43, 55, 59, 62]), # G major
    ]

    for t0, t1, pitches in chords:
        s0, s1 = int(t0 * sr), min(n_samples, int(t1 * sr))
        lt = np.arange(s1 - s0, dtype=np.float32) / sr
        env = np.minimum(1.0, lt / 0.015) * np.exp(-1.2 * lt)
        for p in pitches:
            f = 440.0 * (2.0 ** ((p - 69) / 12.0))
            sig = 0.65 * np.sin(2 * np.pi * f * lt) + 0.25 * np.sin(4 * np.pi * f * lt)
            y[s0:s1] += 0.14 * sig * env

    # Add percussive kicks on beats 0, 2, 4, 6 and snares on 1, 3, 5, 7
    drums = np.zeros(n_samples, dtype=np.float32)
    rng = np.random.default_rng(7)
    for b in range(8):
        s0 = int(b * beat * sr)
        s1 = min(n_samples, s0 + int(0.15 * sr))
        lt = np.arange(s1 - s0, dtype=np.float32) / sr
        if b % 2 == 0:
            f_env = 55.0 + 110.0 * np.exp(-30.0 * lt)
            drums[s0:s1] += 0.5 * np.sin(2 * np.pi * np.cumsum(f_env) / sr) * np.exp(-15.0 * lt)
        else:
            noise = rng.standard_normal(s1 - s0).astype(np.float32)
            drums[s0:s1] += 0.35 * noise * np.exp(-24.0 * lt)

    mix = np.clip(y + drums * 0.4, -0.95, 0.95)
    return mix, drums


def main() -> None:
    print("=== 1. CUDA / PyTorch Check ===")
    from core.stems import get_device_info

    dev = get_device_info()
    print("Device info:", dev)
    assert dev["cuda_available"], "Expected CUDA to be available on RTX 4060!"

    print("\n=== 2. Synthetic Audio, CQT Spectrogram & Theory Check ===")
    from core.audio import compute_cqt_spectrogram, render_spectrogram_rgba
    from core.drums import transcribe_drums
    from core.export import export_midi, export_musicxml
    from core.project import ScoreProject
    from core.theory import analyze_music_theory, quantize_notes
    from core.transcribe import AnalysisConfig, run_transcription_pipeline

    mix, drums = _generate_synthetic_song()
    cqt, times = compute_cqt_spectrogram(mix, sr=44100)
    rgba = render_spectrogram_rgba(cqt)
    print(f"CQT shape: {cqt.shape}, RGBA shape: {rgba.shape}")
    assert cqt.shape[0] == 88 * 3 and rgba.shape[2] == 4

    bg = analyze_music_theory(mix, sr=44100)
    print(f"Detected BPM: {bg.bpm}, Key: {bg.key_name} (conf={bg.key_confidence}), Chords: {[c.symbol for c in bg.chords]}")
    assert bg.bpm > 40 and len(bg.chords) >= 1

    drum_notes = transcribe_drums(drums, sr=44100)
    print(f"Detected drum hits: {len(drum_notes)}")
    assert len(drum_notes) >= 4

    print("\n=== 3. Polyphonic Transcription + MIDI / MusicXML / Project Export ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        wav_path = tmp / "test_song.wav"
        sf.write(str(wav_path), mix, 44100)

        cfg = AnalysisConfig(separate_stems=False, auto_quantize=True)
        project = run_transcription_pipeline(wav_path, config=cfg)
        print(f"Transcribed notes: {project.total_notes()}")
        assert project.total_notes() >= 6, f"Expected >= 6 notes, got {project.total_notes()}"

        mid_path = export_midi(project, tmp / "out.mid")
        xml_path = export_musicxml(project, tmp / "out.musicxml")
        proj_path = tmp / "out.sforge"
        project.save(proj_path)
        reloaded = ScoreProject.load(proj_path)

        print(f"MIDI size: {mid_path.stat().st_size} bytes")
        print(f"MusicXML size: {xml_path.stat().st_size} bytes")
        assert mid_path.stat().st_size > 100
        assert xml_path.stat().st_size > 500
        assert reloaded.total_notes() == project.total_notes()

    print("\n=== 4. PySide6 MainWindow Smoke Test ===")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win._load_project_into_ui(project)
    win.close()
    print("PySide6 MainWindow initialized and loaded project cleanly!")

    print("\nALL TESTS PASSED!")


if __name__ == "__main__":
    main()
