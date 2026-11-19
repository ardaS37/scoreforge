"""Drum stem transcription to General MIDI Channel 10 drum notes.

Uses multi-band spectral flux onset detection to separate:
- Kick Drum (GM 36): 30–140 Hz sub/low transient energy
- Snare Drum (GM 38): 160–1200 Hz body + 2000–6000 Hz snare wire burst
- Closed Hi-Hat (GM 42) / Open Hi-Hat (GM 46): 6000–14000 Hz high-frequency transients
- Crash / Ride Cymbal (GM 49): Sustained high-energy broadband >4000 Hz
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt

from core.project import NoteEvent


# General MIDI Drum Map (Channel 10)
GM_KICK = 36
GM_SNARE = 38
GM_HIHAT_CLOSED = 42
GM_HIHAT_OPEN = 46
GM_TOM_MID = 47
GM_CRASH = 49


def _bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = 0.5 * sr
    lo = max(1.0 / nyq, low_hz / nyq)
    hi = min(0.99, high_hz / nyq)
    if lo >= hi:
        return y
    sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def _detect_band_hits(
    y_band: np.ndarray,
    sr: int,
    hop_length: int,
    delta: float,
    min_wait_sec: float,
    pitch: int,
    duration: float = 0.08,
) -> list[NoteEvent]:
    """Detect percussive hits in a frequency band and convert to NoteEvents."""
    rms = float(np.sqrt(np.mean(y_band**2))) if y_band.size > 0 else 0.0
    if rms < 1e-4:
        return []

    onset_env = librosa.onset.onset_strength(
        y=y_band, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    if onset_env.size == 0 or float(np.max(onset_env)) < 1e-5:
        return []

    env_norm = onset_env / (float(np.max(onset_env)) + 1e-8)
    wait_frames = max(1, int(round(min_wait_sec * sr / hop_length)))

    peaks = librosa.util.peak_pick(
        env_norm,
        pre_max=2,
        post_max=2,
        pre_avg=4,
        post_avg=4,
        delta=delta,
        wait=wait_frames,
    )

    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    notes: list[NoteEvent] = []
    for frame_idx, t in zip(peaks, times):
        strength = float(env_norm[frame_idx])
        vel = int(np.clip(40 + strength * 87, 35, 127))
        notes.append(
            NoteEvent(
                pitch=pitch,
                start=float(t),
                end=float(t + duration),
                velocity=vel,
                confidence=round(min(1.0, 0.5 + 0.5 * strength), 3),
            )
        )
    return notes


def transcribe_drums(
    mono_audio: np.ndarray,
    sr: int = 44100,
    sensitivity: float = 0.5,
) -> list[NoteEvent]:
    """Transcribe a drum stem into General MIDI drum notes (Channel 10).

    Args:
        mono_audio: 1D float32 audio array of isolated drums
        sr: sample rate
        sensitivity: 0.0 (strict) to 1.0 (sensitive), default 0.5
    """
    if mono_audio.size == 0 or float(np.max(np.abs(mono_audio))) < 1e-4:
        return []

    # Work at 22050 Hz for fast, clean transient analysis
    work_sr = 22050
    y = (
        librosa.resample(mono_audio, orig_sr=sr, target_sr=work_sr)
        if sr != work_sr
        else mono_audio
    )
    hop = 256

    # Map sensitivity [0..1] to peak-picking delta [0.35..0.08]
    base_delta = float(np.clip(0.32 - 0.22 * sensitivity, 0.08, 0.35))

    # 1. Kick (30 - 135 Hz)
    y_kick = _bandpass(y, work_sr, 30.0, 135.0)
    kicks = _detect_band_hits(
        y_kick, work_sr, hop, delta=base_delta * 0.95, min_wait_sec=0.09, pitch=GM_KICK, duration=0.10
    )

    # 2. Snare (180 - 2500 Hz, excluding kick sub-bass)
    y_snare = _bandpass(y, work_sr, 180.0, 2500.0)
    snares = _detect_band_hits(
        y_snare, work_sr, hop, delta=base_delta * 1.1, min_wait_sec=0.09, pitch=GM_SNARE, duration=0.10
    )

    # 3. Hi-Hat & Cymbals (5000 - 10500 Hz)
    y_hi = _bandpass(y, work_sr, 5000.0, min(10500.0, work_sr * 0.48))
    hi_hits = _detect_band_hits(
        y_hi, work_sr, hop, delta=base_delta * 0.85, min_wait_sec=0.06, pitch=GM_HIHAT_CLOSED, duration=0.06
    )

    # Distinguish Closed Hi-Hat vs Open Hi-Hat / Crash by post-onset decay energy
    refined_hi: list[NoteEvent] = []
    for n in hi_hits:
        s0 = int(n.start * work_sr)
        s_short = min(len(y_hi), s0 + int(0.05 * work_sr))
        s_tail = min(len(y_hi), s0 + int(0.28 * work_sr))
        if s_tail > s_short > s0:
            e_attack = float(np.mean(y_hi[s0:s_short] ** 2)) + 1e-10
            e_tail = float(np.mean(y_hi[s_short:s_tail] ** 2))
            decay_ratio = e_tail / e_attack
            if decay_ratio > 0.45 and n.velocity > 95:
                n.pitch = GM_CRASH
                n.end = n.start + 0.30
            elif decay_ratio > 0.28:
                n.pitch = GM_HIHAT_OPEN
                n.end = n.start + 0.18
        refined_hi.append(n)

    all_notes = kicks + snares + refined_hi
    all_notes.sort(key=lambda x: (x.start, x.pitch))
    return all_notes
