"""Core utility functions."""

import random
import numpy as np
import torch
import torchaudio
import logging
import sys
import re
import unicodedata
from pathlib import Path
from typing import Union, Optional


# ============================================================================
# Seed Setting
# ============================================================================

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Logging
# ============================================================================

def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Text Normalization
# ============================================================================

def normalize_text(text: str, mode: str = "basic") -> str:
    if not text:
        return ""
    
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    
    if mode == "aggressive":
        text = re.sub(r'[^\w\s]', '', text)
    else:
        text = re.sub(r'[""''′″`]', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def normalize_for_metrics(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# Audio Processing
# ============================================================================

def resample_audio(
    waveform: Union[torch.Tensor, np.ndarray],
    orig_sr: int,
    target_sr: int = 16000
) -> np.ndarray:
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr,
            new_freq=target_sr
        )
        waveform = resampler(waveform)
    
    return waveform.squeeze().numpy().astype(np.float32)


def ensure_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim > 1:
        return np.mean(waveform, axis=0).astype(np.float32)
    return waveform.astype(np.float32)


def get_audio_duration(waveform: np.ndarray, sample_rate: int) -> float:
    return len(waveform) / sample_rate

