"""Export engine: Multi-track Standard MIDI (.mid) and MusicXML (.musicxml)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pretty_midi

from core.project import MidiTrack, ScoreProject


def export_midi(project: ScoreProject, output_path: str | Path) -> Path:
    """Export the ScoreProject to a multi-track Standard MIDI File (.mid)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bpm = max(30.0, min(300.0, float(project.beat_grid.bpm or 120.0)))
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Time signature & key signature metadata
    ts_num, ts_den = project.beat_grid.time_signature
    pm.time_signature_changes.append(pretty_midi.TimeSignature(ts_num, ts_den, 0.0))

    try:
        key_num = project.beat_grid.key_tonic + (12 if project.beat_grid.key_mode == "minor" else 0)
        pm.key_signature_changes.append(pretty_midi.KeySignature(int(key_num % 24), 0.0))
    except Exception:
        pass

    any_solo = any(t.solo for t in project.tracks)

    for track in project.tracks:
        if track.muted or not track.visible:
            continue
        if any_solo and not track.solo:
            continue

        notes = track.active_notes()
        if not notes:
            continue

        inst = pretty_midi.Instrument(
            program=int(np.clip(track.program, 0, 127)),
            is_drum=bool(track.is_drum),
            name=track.name,
        )

        for n in notes:
            start = max(0.0, float(n.start))
            end = max(start + 0.02, float(n.end))
            vel = int(np.clip(n.velocity, 1, 127))
            pitch = int(np.clip(n.pitch, 0, 127))
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end))

            # Optional pitch bend contour
            if n.pitch_bends and not track.is_drum and len(n.pitch_bends) > 1:
                times = np.linspace(start, end, len(n.pitch_bends))
                for bt, bv in zip(times, n.pitch_bends):
                    # Basic Pitch bend units are in 1/3 semitone; scale gently to MIDI 14-bit range
                    midi_bend = int(np.clip(bv * 1365, -8192, 8191))
                    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=midi_bend, time=float(bt)))

        inst.notes.sort(key=lambda x: (x.start, x.pitch))
        pm.instruments.append(inst)

    pm.write(str(out_path))
    return out_path


def build_music21_score(
    project: ScoreProject,
    selected_tracks: Optional[list[MidiTrack]] = None,
    quantize_quarter_div: int = 4,
):
    """Build a structured music21 Score with measures, key/time signature, chords, and grand staff."""
    from music21 import chord as m21_chord
    from music21 import clef, harmony, key, metadata, meter, note, stream, tempo

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = project.title or "ScoreForge Transcription"
    score.metadata.composer = "Transcribed with ScoreForge"

    bpm = max(30.0, min(300.0, float(project.beat_grid.bpm or 120.0)))
    sec_per_quarter = 60.0 / bpm
    ts_num, ts_den = project.beat_grid.time_signature

    # Determine tracks to include (skip drums for standard staff notation unless it's the only track)
    tracks = selected_tracks if selected_tracks is not None else [
        t for t in project.tracks if t.visible and not t.muted and not t.is_drum and t.active_notes()
    ]
    if not tracks:
        tracks = [t for t in project.tracks if t.active_notes()]

    def _sec_to_ql(sec: float) -> float:
        raw_ql = max(0.0, sec / sec_per_quarter)
        return round(raw_ql * quantize_quarter_div) / quantize_quarter_div

    def _populate_part(part: stream.Part, notes_list: list, clef_obj, add_chords: bool = False):
        part.insert(0, clef_obj)
        part.insert(0, tempo.MetronomeMark(number=round(bpm, 1)))
        part.insert(0, meter.TimeSignature(f"{ts_num}/{ts_den}"))
        try:
            k_root = project.beat_grid.key_name.split()[0].replace("b", "-")
            k_mode = project.beat_grid.key_mode
            part.insert(0, key.Key(k_root, k_mode))
        except Exception:
            pass

        # Group simultaneous notes at identical quantized onset into chords for clean engraving
        buckets: dict[float, list] = {}
        for n in notes_list:
            ql_start = _sec_to_ql(n.start)
            ql_dur = max(1.0 / quantize_quarter_div, _sec_to_ql(n.end) - ql_start)
            # Snap duration to clean readable musical value (max 4.0 quarter lengths)
            ql_dur = min(4.0, ql_dur)
            buckets.setdefault(ql_start, []).append((n.pitch, ql_dur, n.velocity))

        for ql_start in sorted(buckets.keys()):
            items = buckets[ql_start]
            if len(items) == 1:
                p, d, v = items[0]
                mn = note.Note(p)
                mn.quarterLength = d
                mn.volume.velocity = v
                part.insert(ql_start, mn)
            else:
                pitches = sorted({it[0] for it in items})
                dur = float(np.median([it[1] for it in items]))
                dur = max(1.0 / quantize_quarter_div, round(dur * quantize_quarter_div) / quantize_quarter_div)
                mc = m21_chord.Chord(pitches)
                mc.quarterLength = dur
                part.insert(ql_start, mc)

        if add_chords and project.beat_grid.chords:
            for ch in project.beat_grid.chords:
                ql_pos = _sec_to_ql(ch.start)
                try:
                    cs = harmony.ChordSymbol(ch.symbol)
                    part.insert(ql_pos, cs)
                except Exception:
                    pass

        try:
            part.makeMeasures(inPlace=True)
            part.makeRests(fillGaps=True, inPlace=True)
        except Exception:
            pass

    first_part = True
    for track in tracks:
        active = track.active_notes()
        if not active:
            continue

        # Split Piano / Mix into Grand Staff (Treble >= C4 / Bass < C4)
        if track.stem_role in ("piano", "mix"):
            treble_notes = [n for n in active if n.pitch >= 60]
            bass_notes = [n for n in active if n.pitch < 60]

            if treble_notes:
                p_treble = stream.Part(id=f"{track.stem_role}_rh")
                p_treble.partName = f"{track.name} (RH)" if bass_notes else track.name
                _populate_part(p_treble, treble_notes, clef.TrebleClef(), add_chords=first_part)
                score.insert(0, p_treble)
                first_part = False

            if bass_notes:
                p_bass = stream.Part(id=f"{track.stem_role}_lh")
                p_bass.partName = f"{track.name} (LH)" if treble_notes else track.name
                _populate_part(p_bass, bass_notes, clef.BassClef(), add_chords=first_part)
                score.insert(0, p_bass)
                first_part = False
        else:
            p = stream.Part(id=track.stem_role)
            p.partName = track.name
            chosen_clef = clef.BassClef() if track.stem_role == "bass" else clef.TrebleClef()
            _populate_part(p, active, chosen_clef, add_chords=first_part)
            score.insert(0, p)
            first_part = False

    return score


def export_musicxml(
    project: ScoreProject,
    output_path: str | Path,
    quantize_quarter_div: int = 4,
) -> Path:
    """Export the ScoreProject to a standard MusicXML (.musicxml) file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score = build_music21_score(project, quantize_quarter_div=quantize_quarter_div)
    score.write("musicxml", fp=str(out_path))
    return out_path


def generate_musicxml_string(project: ScoreProject, quantize_quarter_div: int = 4) -> str:
    """Generate MusicXML XML string directly in memory (for live Sheet Music preview)."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        export_musicxml(project, tmp_path, quantize_quarter_div=quantize_quarter_div)
        return tmp_path.read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
