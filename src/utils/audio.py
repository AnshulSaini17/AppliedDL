"""Audio processing utilities."""

import torch
import torchaudio
from typing import Union
import numpy as np


def resample_audio(
    waveform: Union[torch.Tensor, np.ndarray],
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    """
    Resample audio to target sampling rate.
    
    Args:
        waveform: Audio waveform
        orig_sr: Original sampling rate
        target_sr: Target sampling rate (default 16kHz)
    
    Returns:
        Resampled audio as float32 numpy array
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    # Ensure 2D (channel, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
    
    # Return as float32 numpy array (mono, 1D)
    return waveform.squeeze().numpy().astype(np.float32)


def ensure_mono(waveform: np.ndarray) -> np.ndarray:
    """
    Convert stereo audio to mono by averaging channels.
    
    Args:
        waveform: Audio array (channels, samples) or (samples,)
    
    Returns:
        Mono audio array
    """
    if waveform.ndim > 1:
        return np.mean(waveform, axis=0).astype(np.float32)
    return waveform.astype(np.float32)


def get_audio_duration(waveform: np.ndarray, sample_rate: int) -> float:
    """
    Calculate audio duration in seconds.
    
    Args:
        waveform: Audio array
        sample_rate: Sampling rate in Hz
    
    Returns:
        Duration in seconds
    """
    return len(waveform) / sample_rate

