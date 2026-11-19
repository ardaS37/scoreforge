"""GPU-accelerated Stem Separation using Meta Demucs (htdemucs_6s / htdemucs)."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch


def get_device_info() -> dict:
    """Return available CUDA / CPU device information."""
    cuda_ok = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_ok,
        "device": "cuda" if cuda_ok else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda_ok else "CPU",
        "vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if cuda_ok
            else 0.0
        ),
    }
    return info


def clear_gpu_cache() -> None:
    """Release cached CUDA memory between heavy pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def separate_stems(
    stereo_audio: np.ndarray,
    sr: int,
    model_name: str = "htdemucs_6s",
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict[str, tuple[np.ndarray, int, Optional[str]]]:
    """Separate stereo audio (2, N) into stems using Demucs on GPU/CPU.

    Args:
        stereo_audio: float32 array of shape (2, N)
        sr: sample rate of input audio
        model_name: "htdemucs_6s" (vocals, drums, bass, guitar, piano, other)
                    or "htdemucs" (vocals, drums, bass, other)
        device: "cuda" or "cpu" (auto-detected if None)
        output_dir: optional directory to save separated WAV files
        progress_cb: callback(percent: int, message: str)
        cancel_check: callback returning True if user cancelled

    Returns:
        dict mapping stem_name -> (stereo_float32_array, model_sr, saved_wav_path_or_None)
    """
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import torchaudio

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if progress_cb:
        progress_cb(5, f"Loading Demucs model ({model_name}) on {dev.upper()}...")

    model = get_model(model_name)
    model.to(dev)
    model.eval()
    model_sr = model.samplerate

    if cancel_check and cancel_check():
        del model
        clear_gpu_cache()
        raise InterruptedError("Stem separation cancelled")

    # Prepare tensor (1, 2, N) at model's native sample rate (44100 Hz)
    wav = torch.from_numpy(np.ascontiguousarray(stereo_audio)).float()
    if sr != model_sr:
        wav = torchaudio.functional.resample(wav, sr, model_sr)

    # Normalize as expected by Demucs
    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)
    wav_batch = wav_norm.unsqueeze(0).to(dev)

    if progress_cb:
        progress_cb(20, f"Separating {len(model.sources)} stems on {dev.upper()}...")

    # Segment length tailored for 8 GB VRAM (split=True avoids OOM on long tracks)
    with torch.no_grad():
        sources = apply_model(
            model,
            wav_batch,
            device=dev,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]

    if cancel_check and cancel_check():
        del model, sources, wav_batch
        clear_gpu_cache()
        raise InterruptedError("Stem separation cancelled")

    # De-normalize
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    sources_np = sources.cpu().numpy()

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[np.ndarray, int, Optional[str]]] = {}
    stem_names = list(model.sources)

    for idx, name in enumerate(stem_names):
        if progress_cb:
            pct = 75 + int(20 * (idx + 1) / len(stem_names))
            progress_cb(pct, f"Finalizing stem: {name}...")

        stem_arr = sources_np[idx].astype(np.float32)  # shape (2, N)
        saved_path: Optional[str] = None
        if output_dir is not None:
            wav_path = output_dir / f"{name}.wav"
            sf.write(str(wav_path), stem_arr.T, model_sr)
            saved_path = str(wav_path)

        results[name] = (stem_arr, model_sr, saved_path)

    # Free VRAM immediately for downstream transcription models
    del model, sources, wav_batch
    clear_gpu_cache()

    if progress_cb:
        progress_cb(100, f"Separated {len(results)} stems successfully")

    return results
