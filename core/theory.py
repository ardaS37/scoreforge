"""Music theory engine: BPM/beat tracking, key detection, chord recognition, and quantization."""

from __future__ import annotations

from typing import Callable, Optional

import librosa
import numpy as np

from core.project import BeatGrid, ChordEvent, NoteEvent


PITCH_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Schmuckler key profiles
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Scale pitch-class sets relative to tonic
SCALE_INTERVALS = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "minor": {0, 2, 3, 5, 7, 8, 10},
}

# Chord templates (intervals from root, suffix, penalty weight so triads aren't over-classified as 7ths)
_CHORD_TEMPLATES: list[tuple[str, tuple[int, ...], float]] = [
    ("", (0, 4, 7), 1.00),
    ("m", (0, 3, 7), 1.00),
    ("7", (0, 4, 7, 10), 0.94),
    ("maj7", (0, 4, 7, 11), 0.94),
    ("m7", (0, 3, 7, 10), 0.95),
    ("dim", (0, 3, 6), 0.92),
    ("aug", (0, 4, 8), 0.90),
    ("sus4", (0, 5, 7), 0.92),
    ("sus2", (0, 2, 7), 0.92),
]


def _build_chord_matrix() -> tuple[list[tuple[str, str, str]], np.ndarray]:
    labels: list[tuple[str, str, str]] = []
    vectors: list[np.ndarray] = []
    for root_idx, root_name in enumerate(PITCH_NAMES_SHARP):
        for suffix, intervals, weight in _CHORD_TEMPLATES:
            vec = np.zeros(12, dtype=np.float32)
            for iv in intervals:
                vec[(root_idx + iv) % 12] = 1.0
            vec /= np.linalg.norm(vec)
            vec *= weight
            labels.append((f"{root_name}{suffix}", root_name, suffix or "maj"))
            vectors.append(vec)
    return labels, np.stack(vectors, axis=0)


_CHORD_LABELS, _CHORD_MATRIX = _build_chord_matrix()


def detect_key(chroma: np.ndarray) -> tuple[str, int, str, float]:
    """Detect musical key (tonic, mode, confidence) from a (12, T) chromagram."""
    if chroma.size == 0:
        return "C Major", 0, "major", 0.0

    profile = np.mean(chroma, axis=1)
    if float(np.sum(profile)) < 1e-6:
        return "C Major", 0, "major", 0.0

    best_corr = -2.0
    best_tonic = 0
    best_mode = "major"

    for tonic in range(12):
        maj_rot = np.roll(_KS_MAJOR, tonic)
        min_rot = np.roll(_KS_MINOR, tonic)

        c_maj = float(np.corrcoef(profile, maj_rot)[0, 1])
        c_min = float(np.corrcoef(profile, min_rot)[0, 1])

        if c_maj > best_corr:
            best_corr = c_maj
            best_tonic = tonic
            best_mode = "major"
        if c_min > best_corr:
            best_corr = c_min
            best_tonic = tonic
            best_mode = "minor"

    use_flats = best_tonic in {1, 3, 5, 8, 10} and best_mode == "major"
    name_root = (PITCH_NAMES_FLAT if use_flats else PITCH_NAMES_SHARP)[best_tonic]
    mode_label = "Major" if best_mode == "major" else "Minor"
    confidence = float(np.clip((best_corr + 1.0) * 0.5, 0.0, 1.0))
    return f"{name_root} {mode_label}", best_tonic, best_mode, round(confidence, 3)


def detect_chords(
    chroma: np.ndarray,
    sr: int,
    hop_length: int,
    beat_times: list[float],
    duration: float,
) -> list[ChordEvent]:
    """Detect chords beat-synchronously (or every ~0.5s if no beats) and merge consecutive duplicates."""
    if chroma.size == 0 or duration <= 0.0:
        return []

    if len(beat_times) >= 2:
        boundaries = list(beat_times)
        if boundaries[0] > 0.15:
            boundaries.insert(0, 0.0)
        if boundaries[-1] < duration - 0.1:
            boundaries.append(duration)
    else:
        boundaries = list(np.arange(0.0, duration, 0.5))
        if boundaries[-1] < duration:
            boundaries.append(duration)

    raw_chords: list[ChordEvent] = []
    n_frames = chroma.shape[1]

    for i in range(len(boundaries) - 1):
        t0, t1 = float(boundaries[i]), float(boundaries[i + 1])
        if t1 <= t0 + 1e-4:
            continue
        f0 = int(np.clip(librosa.time_to_frames(t0, sr=sr, hop_length=hop_length), 0, n_frames - 1))
        f1 = int(np.clip(librosa.time_to_frames(t1, sr=sr, hop_length=hop_length), f0 + 1, n_frames))

        seg = np.mean(chroma[:, f0:f1], axis=1)
        norm = float(np.linalg.norm(seg))
        if norm < 1e-5:
            continue
        seg_norm = seg / norm

        scores = _CHORD_MATRIX @ seg_norm
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < 0.52:
            continue

        symbol, root, quality = _CHORD_LABELS[best_idx]
        raw_chords.append(
            ChordEvent(
                symbol=symbol,
                root=root,
                quality=quality,
                start=t0,
                end=t1,
                confidence=round(min(1.0, best_score), 3),
            )
        )

    # Merge consecutive identical chords
    merged: list[ChordEvent] = []
    for ch in raw_chords:
        if merged and merged[-1].symbol == ch.symbol and abs(ch.start - merged[-1].end) < 0.05:
            merged[-1].end = ch.end
            merged[-1].confidence = round(0.5 * (merged[-1].confidence + ch.confidence), 3)
        else:
            merged.append(ch)

    return merged


def analyze_music_theory(
    mono_audio: np.ndarray,
    sr: int = 44100,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> BeatGrid:
    """Run full rhythmic & tonal analysis: BPM, beats, downbeats, key, and chords."""
    duration = float(len(mono_audio) / sr) if sr > 0 else 0.0
    if duration < 0.2:
        return BeatGrid()

    work_sr = 22050
    y = (
        librosa.resample(mono_audio, orig_sr=sr, target_sr=work_sr)
        if sr != work_sr
        else mono_audio
    )
    hop = 512

    if progress_cb:
        progress_cb(15, "Tracking tempo and beats...")

    onset_env = librosa.onset.onset_strength(y=y, sr=work_sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=work_sr, hop_length=hop, trim=False
    )
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm < 40.0 or bpm > 240.0:
        bpm = 120.0
    bpm = round(bpm, 1)

    beat_times = librosa.frames_to_time(beat_frames, sr=work_sr, hop_length=hop).tolist()

    # Fallback regular grid if beat tracking returned too few beats (e.g., ambient/solo)
    if len(beat_times) < 2:
        beat_period = 60.0 / bpm
        beat_times = list(np.arange(0.0, duration, beat_period))

    # Estimate time signature (4/4 vs 3/4) via onset autocorrelation at beat multiples
    ts_num = 4
    if len(beat_frames) >= 8:
        beat_strengths = onset_env[np.clip(beat_frames, 0, len(onset_env) - 1)]
        s3 = float(np.mean(beat_strengths[0::3]))
        s4 = float(np.mean(beat_strengths[0::4]))
        if s3 > s4 * 1.18:
            ts_num = 3

    downbeat_times = beat_times[0::ts_num]

    if progress_cb:
        progress_cb(55, "Detecting key and chords...")

    # Harmonic CQT chromagram for clean key & chord estimation
    y_harm = librosa.effects.harmonic(y, margin=2.0)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=work_sr, hop_length=hop)

    key_name, tonic, mode, key_conf = detect_key(chroma)
    chords = detect_chords(chroma, work_sr, hop, beat_times, duration)

    if progress_cb:
        progress_cb(100, f"Theory analysis complete ({bpm} BPM, {key_name})")

    return BeatGrid(
        bpm=bpm,
        time_signature=(ts_num, 4),
        beat_times=[round(t, 4) for t in beat_times],
        downbeat_times=[round(t, 4) for t in downbeat_times],
        key_name=key_name,
        key_tonic=tonic,
        key_mode=mode,
        key_confidence=key_conf,
        chords=chords,
    )


def quantize_notes(
    notes: list[NoteEvent],
    bpm: float,
    division: float = 0.25,
    strength: float = 1.0,
    min_duration_sec: float = 0.04,
    snap_to_scale: bool = False,
    key_tonic: int = 0,
    key_mode: str = "major",
) -> list[NoteEvent]:
    """Quantize note start/end times to a rhythmic grid and optionally snap pitches to scale.

    Args:
        notes: list of NoteEvent (modified copies returned)
        bpm: tempo in beats per minute
        division: fraction of a quarter note (1.0=quarter, 0.5=eighth, 0.25=16th, 0.125=32nd, 1/3=triplet)
        strength: 0.0 (no change) to 1.0 (hard snap to grid)
        min_duration_sec: remove notes shorter than this threshold
        snap_to_scale: snap out-of-scale pitches to nearest in-scale semitone
    """
    if bpm <= 0:
        bpm = 120.0
    step = (60.0 / bpm) * division
    strength = float(np.clip(strength, 0.0, 1.0))
    scale_pcs = SCALE_INTERVALS.get(key_mode, SCALE_INTERVALS["major"])

    out: list[NoteEvent] = []
    for n in notes:
        if n.duration < min_duration_sec:
            continue

        target_start = round(n.start / step) * step
        new_start = n.start + (target_start - n.start) * strength

        target_end = round(n.end / step) * step
        new_end = n.end + (target_end - n.end) * strength
        if new_end <= new_start + max(0.03, step * 0.25):
            new_end = new_start + max(0.05, step * 0.5)

        pitch = n.pitch
        if snap_to_scale:
            rel = (pitch - key_tonic) % 12
            if rel not in scale_pcs:
                # Try +1 or -1 semitone
                if ((pitch + 1 - key_tonic) % 12) in scale_pcs:
                    pitch += 1
                elif ((pitch - 1 - key_tonic) % 12) in scale_pcs:
                    pitch -= 1

        c = n.clone()
        c.pitch = int(np.clip(pitch, 0, 127))
        c.start = max(0.0, round(new_start, 4))
        c.end = round(new_end, 4)
        out.append(c)

    # Resolve same-pitch overlaps cleanly
    out.sort(key=lambda x: (x.pitch, x.start))
    cleaned: list[NoteEvent] = []
    for n in out:
        if cleaned and cleaned[-1].pitch == n.pitch:
            prev = cleaned[-1]
            if n.start < prev.end:
                if abs(n.start - prev.start) < 0.025:
                    # Merge duplicate trigger, keep longer / louder
                    prev.end = max(prev.end, n.end)
                    prev.velocity = max(prev.velocity, n.velocity)
                    continue
                else:
                    prev.end = max(prev.start + 0.02, n.start - 0.005)
        cleaned.append(n)

    cleaned.sort(key=lambda x: (x.start, x.pitch))
    return cleaned
