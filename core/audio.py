"""Audio loading, FFmpeg decoding fallback, and note-aligned CQT spectrogram."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

import librosa
import numpy as np
import soundfile as sf


def _ensure_ffmpeg_in_path() -> Optional[str]:
    """Ensure imageio-ffmpeg binary is discoverable for MP3/M4A/OGG decoding."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = str(Path(exe).parent)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return exe
    except Exception:
        return None


def load_audio(
    path: str | Path,
    target_sr: int = 44100,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load an audio file as (mono_float32, stereo_float32_shape_2_N, sr).

    Uses soundfile directly for WAV/FLAC/OGG, falling back to bundled FFmpeg
    (via imageio-ffmpeg) for MP3/M4A/AAC/WMA/etc.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    data: Optional[np.ndarray] = None
    sr: int = target_sr

    # Attempt direct read with soundfile (fast, lossless)
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        # sf.read returns shape (N, C) -> transpose to (C, N)
        data = data.T
    except Exception:
        # Fallback: decode via bundled ffmpeg to temporary WAV
        ffmpeg_exe = _ensure_ffmpeg_in_path()
        if not ffmpeg_exe:
            # Final fallback to librosa
            y, sr = librosa.load(str(p), sr=target_sr, mono=False)
            data = y[np.newaxis, :] if y.ndim == 1 else y
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    str(p),
                    "-ac",
                    "2",
                    "-ar",
                    str(target_sr),
                    "-f",
                    "wav",
                    tmp_path,
                ]
                subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                wav_data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
                data = wav_data.T
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    assert data is not None

    # Resample if needed
    if sr != target_sr:
        resampled = [
            librosa.resample(data[ch], orig_sr=sr, target_sr=target_sr)
            for ch in range(data.shape[0])
        ]
        data = np.stack(resampled, axis=0).astype(np.float32)
        sr = target_sr

    # Ensure stereo (2, N) and mono (N,)
    if data.shape[0] == 1:
        stereo = np.vstack([data[0], data[0]]).astype(np.float32)
        mono = data[0].astype(np.float32)
    else:
        stereo = data[:2].astype(np.float32)
        mono = np.mean(stereo, axis=0).astype(np.float32)

    # Peak normalize softly if clipping
    peak = float(np.max(np.abs(stereo))) if stereo.size > 0 else 0.0
    if peak > 1.0:
        stereo /= peak
        mono /= peak

    return mono, stereo, sr


def compute_cqt_spectrogram(
    mono: np.ndarray,
    sr: int = 44100,
    fmin_midi: int = 21,
    n_semitones: int = 88,
    bins_per_semitone: int = 3,
    hop_length: int = 512,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a pitch-aligned Constant-Q Transform (CQT) spectrogram in dB.

    Aligns directly to the 88-key piano range (A0=21 through C8=108) with
    `bins_per_semitone` frequency bins per semitone (default 3 = 36 bins/octave).

    Returns:
        cqt_db: float32 array of shape (n_bins, n_frames) normalized to [0.0, 1.0]
        times:  float32 array of shape (n_frames,) in seconds
    """
    if progress_cb:
        progress_cb(10, "Computing CQT spectrogram...")

    # Downsample to 22050 Hz for CQT (C8 is 4186 Hz, well below 11025 Hz Nyquist,
    # and 2x faster with cleaner low-frequency resolution)
    cqt_sr = 22050
    if sr != cqt_sr:
        y = librosa.resample(mono, orig_sr=sr, target_sr=cqt_sr)
        cqt_hop = max(128, int(round(hop_length * cqt_sr / sr)))
    else:
        y = mono
        cqt_hop = hop_length

    bins_per_octave = 12 * bins_per_semitone
    n_bins = n_semitones * bins_per_semitone
    fmin = float(librosa.midi_to_hz(fmin_midi))

    cqt = np.abs(
        librosa.cqt(
            y,
            sr=cqt_sr,
            hop_length=cqt_hop,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )

    if progress_cb:
        progress_cb(70, "Normalizing spectrogram...")

    # Convert to dB and normalize to [0, 1] with dynamic range of 70 dB
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt_norm = np.clip((cqt_db + 70.0) / 70.0, 0.0, 1.0).astype(np.float32)

    times = librosa.frames_to_time(
        np.arange(cqt_norm.shape[1]), sr=cqt_sr, hop_length=cqt_hop
    ).astype(np.float32)

    if progress_cb:
        progress_cb(100, "Spectrogram ready")

    return cqt_norm, times


def render_spectrogram_rgba(
    cqt_norm: np.ndarray,
    colormap: str = "inferno",
    contrast: float = 1.15,
) -> np.ndarray:
    """Convert normalized CQT (n_bins, n_frames) to an RGBA uint8 image.

    Row 0 corresponds to the HIGHEST pitch (top of piano roll), matching Qt Y-axis.
    Returns shape (n_bins, n_frames, 4) uint8.
    """
    import matplotlib

    cmap = matplotlib.colormaps[colormap]
    # Apply slight gamma/contrast boost so harmonic peaks pop against dark background
    boosted = np.clip(np.power(cqt_norm, 1.0 / max(0.2, contrast)), 0.0, 1.0)
    # Flip vertically so highest pitch is row 0 (top of image)
    flipped = np.flipud(boosted)
    rgba = (cmap(flipped) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgba)
