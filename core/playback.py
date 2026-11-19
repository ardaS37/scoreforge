"""Real-time synchronized Audio + MIDI playback engine using sounddevice.

Features:
- Plays original audio or per-stem separated audio respecting Mute / Solo / Volume
- Renders polyphonic MIDI tracks in real time using a lightweight built-in harmonic
  synthesizer (piano, bass, guitar, vocals/pad, and synthesized drum kit) with zero
  external SoundFont dependencies
- Smooth Audio <-> MIDI mix balance (0.0 = Audio only, 0.5 = 50/50, 1.0 = MIDI only)
- Instant single-note audition when clicking piano keys or dragging notes
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.project import MidiTrack, ScoreProject


@dataclass
class _AuditionVoice:
    pitch: int
    velocity: float
    is_drum: bool
    phase: float = 0.0
    elapsed: float = 0.0
    duration: float = 0.45


class SyncPlaybackEngine:
    """Thread-safe audio + MIDI real-time mixer and transport."""

    def __init__(self, sample_rate: int = 44100, block_size: int = 1024) -> None:
        self.sr = sample_rate
        self.block_size = block_size
        self._project: Optional[ScoreProject] = None
        self._lock = threading.RLock()

        self._playing = False
        self._position_samples: int = 0
        self._duration_samples: int = 0
        self._audio_midi_mix: float = 0.5   # 0.0 = Audio only, 1.0 = MIDI only
        self._master_volume: float = 0.85
        self._loop_enabled: bool = False
        self._loop_start: float = 0.0
        self._loop_end: float = 0.0

        # Pre-rendered MIDI stereo buffer for glitch-free low-CPU playback
        self._midi_buffer: Optional[np.ndarray] = None
        self._midi_dirty: bool = True

        # Active one-shot audition voices (for clicking piano keys)
        self._audition_voices: list[_AuditionVoice] = []
        self._stream = None

    def set_project(self, project: Optional[ScoreProject]) -> None:
        with self._lock:
            self.stop()
            self._project = project
            if project is not None:
                self.sr = project.sample_rate
                self._duration_samples = int(project.duration * self.sr)
                self._loop_start = 0.0
                self._loop_end = project.duration
            else:
                self._duration_samples = 0
            self._position_samples = 0
            self._midi_dirty = True
            self._midi_buffer = None

    def mark_midi_dirty(self) -> None:
        """Call whenever MIDI notes or track mute/solo/volume change."""
        with self._lock:
            self._midi_dirty = True

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def position_seconds(self) -> float:
        return self._position_samples / max(1, self.sr)

    def seek(self, seconds: float) -> None:
        with self._lock:
            s = max(0.0, min(seconds, self._duration_samples / max(1, self.sr)))
            self._position_samples = int(s * self.sr)

    def set_mix(self, mix: float) -> None:
        """0.0 = 100% Audio, 0.5 = Balanced, 1.0 = 100% MIDI."""
        self._audio_midi_mix = float(np.clip(mix, 0.0, 1.0))

    def set_master_volume(self, volume: float) -> None:
        self._master_volume = float(np.clip(volume, 0.0, 1.5))

    def set_loop(self, enabled: bool, start: float = 0.0, end: float = 0.0) -> None:
        self._loop_enabled = enabled
        self._loop_start = max(0.0, start)
        self._loop_end = max(self._loop_start, end)

    def _ensure_stream(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd

            self._stream = sd.OutputStream(
                samplerate=self.sr,
                channels=2,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception:
            self._stream = None

    def play(self) -> None:
        with self._lock:
            if self._project is None:
                return
            if self._midi_dirty:
                self._render_midi_buffer()
            if self._position_samples >= max(1, self._duration_samples - 256):
                self._position_samples = 0
            self._playing = True
        self._ensure_stream()

    def pause(self) -> None:
        self._playing = False

    def stop(self) -> None:
        self._playing = False
        with self._lock:
            self._position_samples = 0

    def close(self) -> None:
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def audition_note(self, pitch: int, velocity: int = 95, is_drum: bool = False) -> None:
        """Play a short preview note immediately (e.g., on piano key click)."""
        with self._lock:
            self._audition_voices.append(
                _AuditionVoice(
                    pitch=int(np.clip(pitch, 0, 127)),
                    velocity=float(np.clip(velocity / 127.0, 0.1, 1.0)),
                    is_drum=is_drum,
                    duration=0.22 if is_drum else 0.45,
                )
            )
            # Cap concurrent audition voices
            if len(self._audition_voices) > 16:
                self._audition_voices = self._audition_voices[-16:]
        self._ensure_stream()

    def _render_midi_buffer(self) -> None:
        """Synthesize all active MIDI tracks into a stereo float32 buffer."""
        if self._project is None or self._duration_samples <= 0:
            self._midi_buffer = None
            self._midi_dirty = False
            return

        proj = self._project
        total_len = self._duration_samples + self.sr  # +1s tail for note release
        buf = np.zeros((2, total_len), dtype=np.float32)

        any_solo = any(t.solo for t in proj.tracks)

        for track in proj.tracks:
            if not track.visible or track.muted:
                continue
            if any_solo and not track.solo:
                continue
            if track.midi_volume <= 0.001:
                continue

            notes = track.active_notes()
            if not notes:
                continue

            track_gain = float(track.midi_volume) * 0.22
            self._synth_track_into(buf, track, notes, track_gain)

        # Soft limiter so dense chords never clip harshly
        buf = np.tanh(buf * 1.15).astype(np.float32)
        self._midi_buffer = buf
        self._midi_dirty = False

    def _synth_track_into(
        self,
        buf: np.ndarray,
        track: MidiTrack,
        notes: list,
        gain: float,
    ) -> None:
        sr = self.sr
        total_len = buf.shape[1]
        role = track.stem_role

        for n in notes:
            s0 = int(n.start * sr)
            if s0 >= total_len or s0 < 0:
                continue
            dur = max(0.04, min(n.duration, 8.0))
            release = 0.12 if not track.is_drum else 0.05
            n_samples = int((dur + release) * sr)
            s1 = min(total_len, s0 + n_samples)
            length = s1 - s0
            if length <= 8:
                continue

            t = np.arange(length, dtype=np.float32) / sr
            amp = (n.velocity / 127.0) * gain

            if track.is_drum:
                sig = self._synth_drum(n.pitch, t, amp)
            else:
                freq = 440.0 * (2.0 ** ((n.pitch - 69) / 12.0))
                sig = self._synth_pitched(freq, t, dur, amp, role)

            buf[0, s0:s1] += sig
            buf[1, s0:s1] += sig

    @staticmethod
    def _synth_pitched(
        freq: float,
        t: np.ndarray,
        dur: float,
        amp: float,
        role: str,
    ) -> np.ndarray:
        """Lightweight musical additive synthesis tailored per instrument role."""
        w = 2.0 * np.pi * freq * t
        if role == "bass":
            wave = 0.72 * np.sin(w) + 0.22 * np.sin(2.0 * w) + 0.06 * np.sin(3.0 * w)
            decay = np.exp(-2.2 * t)
        elif role == "guitar":
            wave = (
                0.55 * np.sin(w)
                + 0.25 * np.sin(2.0 * w)
                + 0.13 * np.sin(3.0 * w)
                + 0.07 * np.sin(4.0 * w)
            )
            decay = np.exp(-4.2 * t)
        elif role == "vocals":
            vib = 0.004 * np.sin(2.0 * np.pi * 5.2 * t)
            wv = 2.0 * np.pi * freq * (t + vib)
            wave = 0.65 * np.sin(wv) + 0.25 * np.sin(2.0 * wv) + 0.10 * np.sin(3.0 * wv)
            decay = np.ones_like(t)
        else:
            # Piano / Mix / Other — warm Rhodes/Grand harmonic blend
            wave = (
                0.62 * np.sin(w)
                + 0.24 * np.sin(2.0 * w) * np.exp(-2.5 * t)
                + 0.10 * np.sin(3.0 * w) * np.exp(-4.0 * t)
                + 0.04 * np.sin(4.0 * w) * np.exp(-6.0 * t)
            )
            decay = np.exp(-1.8 * t)

        # Attack & Release envelope
        attack = np.minimum(1.0, t / 0.008)
        rel_mask = t > dur
        release_env = np.ones_like(t)
        if np.any(rel_mask):
            release_env[rel_mask] = np.exp(-24.0 * (t[rel_mask] - dur))

        return (wave * decay * attack * release_env * amp).astype(np.float32)

    @staticmethod
    def _synth_drum(pitch: int, t: np.ndarray, amp: float) -> np.ndarray:
        """Synthesize Kick, Snare, or Hi-Hat/Cymbal for GM drum pitches."""
        if pitch in (35, 36):  # Kick
            f_env = 52.0 + 115.0 * np.exp(-32.0 * t)
            phase = 2.0 * np.pi * np.cumsum(f_env) / 44100.0
            return (np.sin(phase) * np.exp(-16.0 * t) * amp * 1.4).astype(np.float32)
        elif pitch in (38, 40):  # Snare
            rng = np.random.default_rng(42)
            noise = rng.standard_normal(len(t)).astype(np.float32)
            body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-22.0 * t)
            snap = noise * np.exp(-28.0 * t)
            return ((0.55 * body + 0.65 * snap) * amp).astype(np.float32)
        else:  # Hi-hat / Cymbal
            rng = np.random.default_rng(13)
            noise = rng.standard_normal(len(t)).astype(np.float32)
            # High-pass feel via first difference
            hp = np.diff(noise, prepend=0.0)
            rate = 12.0 if pitch in (46, 49, 51) else 55.0
            return (hp * np.exp(-rate * t) * amp * 0.65).astype(np.float32)

    def _mix_audio_slice(self, start: int, end: int) -> np.ndarray:
        """Return stereo (2, length) slice from stems or original mix."""
        length = end - start
        out = np.zeros((2, length), dtype=np.float32)
        proj = self._project
        if proj is None:
            return out

        has_stems = any(t.stem_audio is not None for t in proj.tracks)
        if has_stems and len(proj.tracks) > 1:
            any_solo = any(t.solo for t in proj.tracks)
            for t in proj.tracks:
                if t.stem_audio is None or t.muted or not t.visible:
                    continue
                if any_solo and not t.solo:
                    continue
                arr = t.stem_audio
                if start < arr.shape[1]:
                    s_end = min(end, arr.shape[1])
                    n = s_end - start
                    out[:, :n] += arr[:, start:s_end] * float(t.audio_volume)
        elif proj.audio_stereo is not None:
            arr = proj.audio_stereo
            if start < arr.shape[1]:
                s_end = min(end, arr.shape[1])
                n = s_end - start
                out[:, :n] = arr[:, start:s_end]

        return out

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        out = np.zeros((2, frames), dtype=np.float32)

        with self._lock:
            if self._playing and self._project is not None:
                if self._midi_dirty:
                    self._render_midi_buffer()

                start = self._position_samples
                end = start + frames

                # Audio component
                audio_gain = float(np.cos(self._audio_midi_mix * np.pi * 0.5))
                midi_gain = float(np.sin(self._audio_midi_mix * np.pi * 0.5))

                if audio_gain > 0.005:
                    out += self._mix_audio_slice(start, end) * audio_gain

                if midi_gain > 0.005 and self._midi_buffer is not None:
                    mb = self._midi_buffer
                    if start < mb.shape[1]:
                        s_end = min(end, mb.shape[1])
                        n = s_end - start
                        out[:, :n] += mb[:, start:s_end] * midi_gain

                self._position_samples = end

                # Loop or stop at end
                if self._loop_enabled and self._loop_end > self._loop_start + 0.1:
                    loop_end_s = int(self._loop_end * self.sr)
                    if self._position_samples >= loop_end_s:
                        self._position_samples = int(self._loop_start * self.sr)
                elif self._position_samples >= self._duration_samples:
                    self._playing = False
                    self._position_samples = 0

            # Mix one-shot audition voices (piano key clicks)
            if self._audition_voices:
                t_block = np.arange(frames, dtype=np.float32) / self.sr
                remaining: list[_AuditionVoice] = []
                for v in self._audition_voices:
                    t_v = t_block + v.elapsed
                    if v.is_drum:
                        sig = self._synth_drum(v.pitch, t_v, v.velocity * 0.35)
                    else:
                        freq = 440.0 * (2.0 ** ((v.pitch - 69) / 12.0))
                        sig = self._synth_pitched(freq, t_v, v.duration * 0.7, v.velocity * 0.30, "piano")
                    out[0] += sig
                    out[1] += sig
                    v.elapsed += frames / self.sr
                    if v.elapsed < v.duration:
                        remaining.append(v)
                self._audition_voices = remaining

        out *= self._master_volume
        np.clip(out, -1.0, 1.0, out=out)
        outdata[:] = out.T
