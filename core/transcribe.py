"""Polyphonic Audio-to-MIDI Transcription Engine.

Supports:
1. Spotify Basic Pitch (ONNX Runtime) for accurate polyphonic transcription with pitch bends
2. Specialized Drum Transcription (core/drums.py) for drum stems
3. CQT Harmonic Peak-Picking fallback if Basic Pitch is unavailable
4. End-to-end multi-stem transcription pipeline orchestrated with Demucs + Theory analysis
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import librosa
import numpy as np
import soundfile as sf

from core.audio import compute_cqt_spectrogram, load_audio
from core.drums import transcribe_drums
from core.project import MidiTrack, NoteEvent, STEM_DEFAULTS, ScoreProject
from core.stems import clear_gpu_cache, separate_stems
from core.theory import analyze_music_theory, quantize_notes


@dataclass
class AnalysisConfig:
    """User-configurable parameters for the AI transcription pipeline."""

    separate_stems: bool = True
    demucs_model: str = "htdemucs_6s"       # "htdemucs_6s" (6 stems) or "htdemucs" (4 stems)
    device: str = "auto"                    # "auto", "cuda", "cpu"
    onset_threshold: float = 0.50           # 0.1..0.9 (Basic Pitch onset sensitivity)
    frame_threshold: float = 0.30           # 0.1..0.9 (Basic Pitch frame sensitivity)
    min_note_length_ms: float = 80.0        # Minimum note duration in milliseconds
    min_freq_hz: float = 32.7               # C1
    max_freq_hz: float = 2093.0             # C7
    drum_sensitivity: float = 0.50
    auto_quantize: bool = False
    quantize_division: float = 0.25         # 16th note
    quantize_strength: float = 0.75
    include_pitch_bends: bool = True


def _stem_has_energy(audio: np.ndarray, rms_threshold: float = 0.003) -> bool:
    """Skip silent/empty stems (e.g. piano stem in a guitar-only track)."""
    if audio.size == 0:
        return False
    rms = float(np.sqrt(np.mean(audio**2)))
    return rms >= rms_threshold


def transcribe_polyphonic_basic_pitch(
    mono_audio: np.ndarray,
    sr: int,
    onset_threshold: float = 0.50,
    frame_threshold: float = 0.30,
    min_note_length_ms: float = 80.0,
    min_freq_hz: Optional[float] = 32.7,
    max_freq_hz: Optional[float] = 2093.0,
    include_pitch_bends: bool = True,
) -> list[NoteEvent]:
    """Transcribe polyphonic audio to NoteEvents using Spotify Basic Pitch (ONNX)."""
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    # Write temporary WAV for basic-pitch (which resamples internally to 22050 Hz)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, mono_audio.astype(np.float32), sr)
        _, _, note_events = predict(
            audio_path=tmp_path,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length_ms,
            minimum_frequency=min_freq_hz,
            maximum_frequency=max_freq_hz,
            melodia_trick=True,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    notes: list[NoteEvent] = []
    for ev in note_events:
        # ev tuple: (start_time_s, end_time_s, pitch_midi, amplitude, pitch_bends)
        start_s = float(ev[0])
        end_s = float(ev[1])
        pitch = int(ev[2])
        amp = float(ev[3])
        bends = list(ev[4]) if (include_pitch_bends and len(ev) > 4 and ev[4] is not None) else []

        if end_s <= start_s:
            continue

        vel = int(np.clip(round(30 + amp * 97), 20, 127))
        notes.append(
            NoteEvent(
                pitch=int(np.clip(pitch, 0, 127)),
                start=round(start_s, 4),
                end=round(end_s, 4),
                velocity=vel,
                confidence=round(float(np.clip(amp, 0.0, 1.0)), 3),
                pitch_bends=[int(b) for b in bends],
            )
        )

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def transcribe_polyphonic_cqt_fallback(
    mono_audio: np.ndarray,
    sr: int,
    threshold: float = 0.55,
    min_note_sec: float = 0.08,
) -> list[NoteEvent]:
    """Pure-DSP CQT peak-picking fallback if Basic Pitch is not installed."""
    cqt_sr = 22050
    y = librosa.resample(mono_audio, orig_sr=sr, target_sr=cqt_sr) if sr != cqt_sr else mono_audio
    hop = 512
    fmin_midi = 21
    n_bins = 88

    C = np.abs(
        librosa.cqt(
            y,
            sr=cqt_sr,
            hop_length=hop,
            fmin=float(librosa.midi_to_hz(fmin_midi)),
            n_bins=n_bins,
            bins_per_octave=12,
        )
    )
    C_db = librosa.amplitude_to_db(C, ref=np.max)
    C_norm = np.clip((C_db + 65.0) / 65.0, 0.0, 1.0)
    times = librosa.frames_to_time(np.arange(C_norm.shape[1]), sr=cqt_sr, hop_length=hop)

    notes: list[NoteEvent] = []
    for bin_idx in range(n_bins):
        pitch = fmin_midi + bin_idx
        active = C_norm[bin_idx] >= threshold
        # Suppress harmonic bleed: require local peak across neighboring pitches
        if 0 < bin_idx < n_bins - 1:
            is_peak = (C_norm[bin_idx] >= C_norm[bin_idx - 1]) & (
                C_norm[bin_idx] >= C_norm[bin_idx + 1]
            )
            active = active & is_peak

        in_note = False
        start_idx = 0
        for f_idx, on in enumerate(active):
            if on and not in_note:
                in_note = True
                start_idx = f_idx
            elif not on and in_note:
                in_note = False
                t0, t1 = float(times[start_idx]), float(times[f_idx])
                if t1 - t0 >= min_note_sec:
                    peak_val = float(np.max(C_norm[bin_idx, start_idx:f_idx]))
                    notes.append(
                        NoteEvent(
                            pitch=pitch,
                            start=round(t0, 4),
                            end=round(t1, 4),
                            velocity=int(np.clip(35 + peak_val * 92, 30, 127)),
                            confidence=round(peak_val, 3),
                        )
                    )
        if in_note:
            t0, t1 = float(times[start_idx]), float(times[-1])
            if t1 - t0 >= min_note_sec:
                peak_val = float(np.max(C_norm[bin_idx, start_idx:]))
                notes.append(
                    NoteEvent(
                        pitch=pitch,
                        start=round(t0, 4),
                        end=round(t1, 4),
                        velocity=int(np.clip(35 + peak_val * 92, 30, 127)),
                        confidence=round(peak_val, 3),
                    )
                )

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def transcribe_audio_array(
    mono_audio: np.ndarray,
    sr: int,
    stem_role: str,
    config: AnalysisConfig,
) -> list[NoteEvent]:
    """Transcribe a single stem or mix array using the best engine for its role."""
    if stem_role == "drums":
        return transcribe_drums(mono_audio, sr=sr, sensitivity=config.drum_sensitivity)

    # Stem-specific frequency bounds for cleaner results
    freq_bounds = {
        "bass": (28.0, 520.0),
        "vocals": (75.0, 1600.0),
        "guitar": (70.0, 1800.0),
        "piano": (27.5, 4186.0),
        "other": (config.min_freq_hz, config.max_freq_hz),
        "mix": (config.min_freq_hz, config.max_freq_hz),
    }
    f_lo, f_hi = freq_bounds.get(stem_role, (config.min_freq_hz, config.max_freq_hz))

    try:
        notes = transcribe_polyphonic_basic_pitch(
            mono_audio,
            sr=sr,
            onset_threshold=config.onset_threshold,
            frame_threshold=config.frame_threshold,
            min_note_length_ms=config.min_note_length_ms,
            min_freq_hz=f_lo,
            max_freq_hz=f_hi,
            include_pitch_bends=config.include_pitch_bends,
        )
    except Exception:
        notes = transcribe_polyphonic_cqt_fallback(
            mono_audio,
            sr=sr,
            threshold=max(0.45, config.onset_threshold),
            min_note_sec=config.min_note_length_ms / 1000.0,
        )

    return notes


def run_transcription_pipeline(
    audio_path: str | Path,
    config: Optional[AnalysisConfig] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ScoreProject:
    """Full end-to-end ScoreForge pipeline:
    1. Load audio
    2. Compute high-res note-aligned CQT spectrogram
    3. Music theory analysis (BPM, beats, downbeats, key, chords)
    4. Optional Demucs stem separation on CUDA
    5. Per-stem AI polyphonic / drum transcription
    6. Optional smart quantization
    """
    cfg = config or AnalysisConfig()
    p = Path(audio_path)

    def emit(pct: int, msg: str) -> None:
        if progress_cb:
            progress_cb(pct, msg)

    def check_cancel() -> None:
        if cancel_check and cancel_check():
            raise InterruptedError("Analysis cancelled by user")

    emit(2, f"Loading audio: {p.name}...")
    mono, stereo, sr = load_audio(p, target_sr=44100)
    duration = float(len(mono) / sr)
    check_cancel()

    project = ScoreProject(
        audio_path=str(p.resolve()),
        title=p.stem,
        duration=duration,
        sample_rate=sr,
        audio_mono=mono,
        audio_stereo=stereo,
    )

    emit(10, "Computing pitch-aligned CQT spectrogram...")
    cqt_norm, cqt_times = compute_cqt_spectrogram(
        mono,
        sr=sr,
        fmin_midi=project.cqt_fmin_midi,
        n_semitones=88,
        bins_per_semitone=project.cqt_bins_per_semitone,
    )
    project.cqt_db = cqt_norm
    project.cqt_times = cqt_times
    check_cancel()

    emit(20, "Analyzing tempo, beats, key, and chords...")
    project.beat_grid = analyze_music_theory(mono, sr=sr)
    check_cancel()

    tracks: list[MidiTrack] = []

    if cfg.separate_stems:
        dev = None if cfg.device == "auto" else cfg.device
        stems_dir = Path(tempfile.gettempdir()) / "scoreforge_stems" / p.stem

        def stem_progress(pct: int, msg: str) -> None:
            # Map 0..100 -> 25..62
            emit(25 + int(pct * 0.37), msg)

        stems = separate_stems(
            stereo_audio=stereo,
            sr=sr,
            model_name=cfg.demucs_model,
            device=dev,
            output_dir=stems_dir,
            progress_cb=stem_progress,
            cancel_check=cancel_check,
        )
        check_cancel()

        # Order stems nicely for the mixer
        preferred_order = ["piano", "vocals", "guitar", "bass", "other", "drums"]
        ordered_names = [s for s in preferred_order if s in stems] + [
            s for s in stems if s not in preferred_order
        ]

        for idx, stem_name in enumerate(ordered_names):
            check_cancel()
            stem_stereo, stem_sr, saved_wav = stems[stem_name]
            stem_mono = np.mean(stem_stereo, axis=0).astype(np.float32)

            pct = 65 + int(30 * idx / max(1, len(ordered_names)))
            emit(pct, f"Transcribing stem: {stem_name.title()}...")

            meta = STEM_DEFAULTS.get(stem_name, STEM_DEFAULTS["other"])
            if _stem_has_energy(stem_mono):
                notes = transcribe_audio_array(stem_mono, stem_sr, stem_name, cfg)
                if cfg.auto_quantize and notes:
                    notes = quantize_notes(
                        notes,
                        bpm=project.beat_grid.bpm,
                        division=cfg.quantize_division,
                        strength=cfg.quantize_strength,
                    )
            else:
                notes = []

            tracks.append(
                MidiTrack(
                    name=meta["label"],
                    stem_role=stem_name,
                    program=meta["program"],
                    is_drum=meta["is_drum"],
                    color=meta["color"],
                    notes=notes,
                    audio_path=saved_wav,
                    stem_audio=stem_stereo,
                    stem_sr=stem_sr,
                )
            )
    else:
        emit(40, "Transcribing full mix with AI...")
        notes = transcribe_audio_array(mono, sr, "mix", cfg)
        if cfg.auto_quantize and notes:
            notes = quantize_notes(
                notes,
                bpm=project.beat_grid.bpm,
                division=cfg.quantize_division,
                strength=cfg.quantize_strength,
            )
        meta = STEM_DEFAULTS["mix"]
        tracks.append(
            MidiTrack(
                name=meta["label"],
                stem_role="mix",
                program=meta["program"],
                is_drum=False,
                color=meta["color"],
                notes=notes,
                audio_path=str(p.resolve()),
                stem_audio=stereo,
                stem_sr=sr,
            )
        )

    project.tracks = tracks
    clear_gpu_cache()
    emit(
        100,
        f"Done! {project.total_notes()} notes across {len(tracks)} tracks "
        f"({project.beat_grid.bpm} BPM, {project.beat_grid.key_name})",
    )
    return project
