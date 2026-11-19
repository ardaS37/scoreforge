"""Core project data model for ScoreForge.

Defines NoteEvent, ChordEvent, BeatGrid, MidiTrack, and ScoreProject,
with JSON serialization/deserialization for saving/loading `.sforge` projects.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# Default color palette & General MIDI program numbers per stem/role
STEM_DEFAULTS: dict[str, dict] = {
    "mix": {"label": "Full Mix", "color": "#4CC9F0", "program": 0, "is_drum": False},
    "piano": {"label": "Piano", "color": "#4361EE", "program": 0, "is_drum": False},
    "vocals": {"label": "Vocals", "color": "#F72585", "program": 52, "is_drum": False},
    "guitar": {"label": "Guitar", "color": "#FF9F1C", "program": 25, "is_drum": False},
    "bass": {"label": "Bass", "color": "#7209B7", "program": 33, "is_drum": False},
    "drums": {"label": "Drums", "color": "#2EC4B6", "program": 0, "is_drum": True},
    "other": {"label": "Other", "color": "#AACC00", "program": 48, "is_drum": False},
}

GM_INSTRUMENTS: list[tuple[int, str]] = [
    (0, "Acoustic Grand Piano"),
    (4, "Electric Piano 1"),
    (16, "Drawbar Organ"),
    (24, "Acoustic Guitar (Nylon)"),
    (25, "Acoustic Guitar (Steel)"),
    (27, "Electric Guitar (Clean)"),
    (29, "Overdriven Guitar"),
    (32, "Acoustic Bass"),
    (33, "Electric Bass (Finger)"),
    (38, "Synth Bass 1"),
    (40, "Violin"),
    (42, "Cello"),
    (48, "String Ensemble 1"),
    (52, "Choir Aahs"),
    (56, "Trumpet"),
    (65, "Alto Sax"),
    (73, "Flute"),
    (80, "Lead 1 (Square)"),
    (88, "Pad 1 (New Age)"),
]


@dataclass
class NoteEvent:
    """Single MIDI note event with optional pitch bend contour."""

    pitch: int                  # MIDI note number (21..108 for 88-key piano, 0..127 general)
    start: float                # Start time in seconds
    end: float                  # End time in seconds
    velocity: int = 90          # 1..127
    confidence: float = 1.0     # AI confidence score (0.0..1.0)
    pitch_bends: list[int] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def clone(self) -> "NoteEvent":
        return NoteEvent(
            pitch=self.pitch,
            start=self.start,
            end=self.end,
            velocity=self.velocity,
            confidence=self.confidence,
            pitch_bends=list(self.pitch_bends),
            id=uuid.uuid4().hex[:10],
        )


@dataclass
class ChordEvent:
    """Detected chord spanning [start, end] seconds."""

    symbol: str                 # e.g., "Am7", "Cmaj7", "G", "N"
    root: str                   # e.g., "A", "C", "G"
    quality: str                # e.g., "min7", "maj7", "maj", "min"
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class BeatGrid:
    """Rhythmic & tonal analysis metadata."""

    bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    key_name: str = "C Major"
    key_tonic: int = 0          # 0=C, 1=C#, ..., 11=B
    key_mode: str = "major"     # "major" or "minor"
    key_confidence: float = 0.0
    chords: list[ChordEvent] = field(default_factory=list)


@dataclass
class MidiTrack:
    """A single instrument/stem track containing MIDI notes and optional audio stem."""

    name: str
    stem_role: str = "mix"      # "mix", "piano", "vocals", "guitar", "bass", "drums", "other"
    program: int = 0            # General MIDI program number (0..127)
    is_drum: bool = False
    color: str = "#4CC9F0"
    notes: list[NoteEvent] = field(default_factory=list)
    visible: bool = True
    muted: bool = False
    solo: bool = False
    audio_volume: float = 1.0   # 0.0..1.5
    midi_volume: float = 1.0    # 0.0..1.5
    min_confidence: float = 0.0 # Live filter threshold
    audio_path: Optional[str] = None
    # Transient runtime audio array (not serialized to JSON)
    stem_audio: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    stem_sr: int = 44100

    def active_notes(self) -> list[NoteEvent]:
        """Return notes passing the track's live confidence filter."""
        if self.min_confidence <= 0.0:
            return self.notes
        return [n for n in self.notes if n.confidence >= self.min_confidence]

    def sort_notes(self) -> None:
        self.notes.sort(key=lambda n: (n.start, n.pitch))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stem_role": self.stem_role,
            "program": self.program,
            "is_drum": self.is_drum,
            "color": self.color,
            "notes": [asdict(n) for n in self.notes],
            "visible": self.visible,
            "muted": self.muted,
            "solo": self.solo,
            "audio_volume": self.audio_volume,
            "midi_volume": self.midi_volume,
            "min_confidence": self.min_confidence,
            "audio_path": self.audio_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MidiTrack":
        notes = [NoteEvent(**n) for n in data.get("notes", [])]
        return cls(
            name=data["name"],
            stem_role=data.get("stem_role", "mix"),
            program=data.get("program", 0),
            is_drum=data.get("is_drum", False),
            color=data.get("color", "#4CC9F0"),
            notes=notes,
            visible=data.get("visible", True),
            muted=data.get("muted", False),
            solo=data.get("solo", False),
            audio_volume=data.get("audio_volume", 1.0),
            midi_volume=data.get("midi_volume", 1.0),
            min_confidence=data.get("min_confidence", 0.0),
            audio_path=data.get("audio_path"),
        )


@dataclass
class ScoreProject:
    """Top-level ScoreForge project holding source audio, spectrogram, tracks, and theory."""

    audio_path: Optional[str] = None
    title: str = "Untitled"
    duration: float = 0.0
    sample_rate: int = 44100
    tracks: list[MidiTrack] = field(default_factory=list)
    beat_grid: BeatGrid = field(default_factory=BeatGrid)

    # Runtime-only cached audio & CQT spectrogram data
    audio_mono: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    audio_stereo: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    cqt_db: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    cqt_times: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    cqt_fmin_midi: int = 21     # A0
    cqt_bins_per_semitone: int = 3

    def total_notes(self) -> int:
        return sum(len(t.notes) for t in self.tracks)

    def get_track(self, index: int) -> Optional[MidiTrack]:
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return None

    def save(self, path: str | Path) -> None:
        p = Path(path)
        payload = {
            "version": 1,
            "audio_path": self.audio_path,
            "title": self.title,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "beat_grid": {
                "bpm": self.beat_grid.bpm,
                "time_signature": list(self.beat_grid.time_signature),
                "beat_times": self.beat_grid.beat_times,
                "downbeat_times": self.beat_grid.downbeat_times,
                "key_name": self.beat_grid.key_name,
                "key_tonic": self.beat_grid.key_tonic,
                "key_mode": self.beat_grid.key_mode,
                "key_confidence": self.beat_grid.key_confidence,
                "chords": [asdict(c) for c in self.beat_grid.chords],
            },
            "tracks": [t.to_dict() for t in self.tracks],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ScoreProject":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        bg_data = data.get("beat_grid", {})
        chords = [ChordEvent(**c) for c in bg_data.get("chords", [])]
        ts = bg_data.get("time_signature", [4, 4])
        beat_grid = BeatGrid(
            bpm=bg_data.get("bpm", 120.0),
            time_signature=(int(ts[0]), int(ts[1])),
            beat_times=bg_data.get("beat_times", []),
            downbeat_times=bg_data.get("downbeat_times", []),
            key_name=bg_data.get("key_name", "C Major"),
            key_tonic=bg_data.get("key_tonic", 0),
            key_mode=bg_data.get("key_mode", "major"),
            key_confidence=bg_data.get("key_confidence", 0.0),
            chords=chords,
        )
        tracks = [MidiTrack.from_dict(t) for t in data.get("tracks", [])]
        return cls(
            audio_path=data.get("audio_path"),
            title=data.get("title", p.stem),
            duration=data.get("duration", 0.0),
            sample_rate=data.get("sample_rate", 44100),
            tracks=tracks,
            beat_grid=beat_grid,
        )
