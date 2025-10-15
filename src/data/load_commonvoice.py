"""Common Voice dataset loader with accent filtering and speaker-stratified splits."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from .accent_taxonomy import canonicalize_accent, load_accent_mappings
from ..utils.audio import get_audio_duration
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def load_commonvoice_dataset(
    dataset_candidates: List[str],
    language: str = "en",
    cache_dir: Optional[str] = None
) -> Dataset:
    """
    Load Common Voice dataset, trying candidates in order.
    
    Args:
        dataset_candidates: List of dataset names to try
        language: Language code
        cache_dir: Cache directory path
    
    Returns:
        Loaded dataset
    """
    import os
    # Get HuggingFace token from environment (needed for gated datasets)
    token = os.environ.get('HF_TOKEN', os.environ.get('HUGGING_FACE_HUB_TOKEN', None))
    
    # Use cache_dir from environment if not specified
    if cache_dir is None:
        cache_dir = os.environ.get('HF_DATASETS_CACHE', None)
    
    for dataset_name in dataset_candidates:
        # Try without trust_remote_code first (for newer versions like 17.0)
        try:
            logger.info(f"Attempting to load {dataset_name} (without trust_remote_code)...")
            ds = load_dataset(
                dataset_name,
                language,
                split="train+validation+test",
                cache_dir=cache_dir,
                token=token
            )
            logger.info(f"Successfully loaded {dataset_name}")
            return ds
        except Exception as e1:
            logger.warning(f"Failed without trust_remote_code: {e1}")
            
            # Try with trust_remote_code (for older versions with scripts)
            try:
                logger.info(f"Attempting to load {dataset_name} (with trust_remote_code)...")
                ds = load_dataset(
                    dataset_name,
                    language,
                    split="train+validation+test",
                    cache_dir=cache_dir,
                    token=token,
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded {dataset_name}")
                return ds
            except Exception as e2:
                logger.warning(f"Failed with trust_remote_code: {e2}")
                if token is None:
                    logger.error("HF_TOKEN not found. Common Voice requires authentication.")
                    logger.error("Set HF_TOKEN environment variable with your HuggingFace token.")
                continue
    
    raise RuntimeError(f"Failed to load any dataset from candidates: {dataset_candidates}")


def filter_and_canonicalize(
    dataset: Dataset,
    target_accents: List[str],
    mappings: Dict[str, List[str]],
    min_samples_per_accent: int = 100
) -> Tuple[Dataset, Dict[str, int]]:
    """
    Filter dataset by valid transcripts and accents, canonicalize accent labels.
    
    Args:
        dataset: Raw Common Voice dataset
        target_accents: List of canonical accents to keep
        mappings: Accent mapping dictionary
        min_samples_per_accent: Minimum samples required per accent
    
    Returns:
        Filtered dataset and accent counts
    """
    logger.info("Filtering and canonicalizing accents...")
    
    filtered_examples = []
    accent_counts = defaultdict(int)
    
    for example in tqdm(dataset, desc="Processing accents"):
        # Check for valid transcript
        sentence = example.get('sentence', '')
        if not sentence or not sentence.strip():
            continue
        
        # Check and canonicalize accent
        raw_accent = example.get('accent', None)
        if not raw_accent:
            continue
        
        canonical = canonicalize_accent(raw_accent, mappings)
        if canonical and canonical in target_accents:
            example['accent'] = canonical
            filtered_examples.append(example)
            accent_counts[canonical] += 1
    
    logger.info(f"Filtered {len(filtered_examples)} samples from {len(dataset)}")
    
    # Check minimum samples
    valid_accents = []
    for accent in target_accents:
        count = accent_counts[accent]
        if count >= min_samples_per_accent:
            valid_accents.append(accent)
            logger.info(f"  {accent}: {count} samples ✓")
        else:
            logger.warning(f"  {accent}: {count} samples (below threshold {min_samples_per_accent}, will be excluded)")
    
    # Filter to only valid accents
    filtered_examples = [ex for ex in filtered_examples if ex['accent'] in valid_accents]
    
    # Convert back to Dataset
    filtered_ds = Dataset.from_list(filtered_examples)
    
    return filtered_ds, dict(accent_counts)


def stratified_split_by_speaker(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> DatasetDict:
    """
    Split dataset by speaker ID to avoid data leakage.
    
    Args:
        dataset: Filtered dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    
    Returns:
        DatasetDict with train/val/test splits
    """
    logger.info("Creating speaker-stratified splits...")
    
    # Group by speaker (client_id)
    speaker_to_indices = defaultdict(list)
    for idx, example in enumerate(dataset):
        speaker_id = example.get('client_id', f'unknown_{idx}')
        speaker_to_indices[speaker_id].append(idx)
    
    unique_speakers = list(speaker_to_indices.keys())
    logger.info(f"Found {len(unique_speakers)} unique speakers")
    
    # Shuffle speakers
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_speakers)
    
    # Split speakers
    n_speakers = len(unique_speakers)
    n_train = int(n_speakers * train_ratio)
    n_val = int(n_speakers * val_ratio)
    
    train_speakers = unique_speakers[:n_train]
    val_speakers = unique_speakers[n_train:n_train + n_val]
    test_speakers = unique_speakers[n_train + n_val:]
    
    # Gather indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for speaker in train_speakers:
        train_indices.extend(speaker_to_indices[speaker])
    for speaker in val_speakers:
        val_indices.extend(speaker_to_indices[speaker])
    for speaker in test_speakers:
        test_indices.extend(speaker_to_indices[speaker])
    
    logger.info(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Create splits
    train_ds = dataset.select(train_indices)
    val_ds = dataset.select(val_indices)
    test_ds = dataset.select(test_indices)
    
    return DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })


def limit_dataset_size(
    dataset: Dataset,
    max_samples: Optional[int] = None,
    max_hours: Optional[float] = None,
    sample_rate: int = 16000,
    seed: int = 42
) -> Dataset:
    """
    Limit dataset size by number of samples or hours.
    
    Args:
        dataset: Input dataset
        max_samples: Maximum number of samples
        max_hours: Maximum hours of audio
        sample_rate: Audio sample rate
        seed: Random seed
    
    Returns:
        Limited dataset
    """
    if max_samples is None and max_hours is None:
        return dataset
    
    # Shuffle first
    dataset = dataset.shuffle(seed=seed)
    
    if max_samples is not None and len(dataset) > max_samples:
        logger.info(f"Limiting to {max_samples} samples")
        dataset = dataset.select(range(max_samples))
    
    if max_hours is not None:
        # Accumulate until max hours
        total_duration = 0.0
        selected_indices = []
        
        for idx, example in enumerate(dataset):
            audio = example['audio']
            duration = get_audio_duration(audio['array'], audio['sampling_rate'])
            
            if total_duration + duration > max_hours * 3600:
                break
            
            selected_indices.append(idx)
            total_duration += duration
        
        logger.info(f"Limiting to {len(selected_indices)} samples (~{total_duration/3600:.2f} hours)")
        dataset = dataset.select(selected_indices)
    
    return dataset


def balance_accents_in_split(
    dataset: Dataset,
    max_per_accent: Optional[int] = None,
    seed: int = 42
) -> Dataset:
    """
    Balance accents by limiting samples per accent.
    
    Args:
        dataset: Input dataset
        max_per_accent: Maximum samples per accent
        seed: Random seed
    
    Returns:
        Balanced dataset
    """
    if max_per_accent is None:
        return dataset
    
    # Group by accent
    accent_to_indices = defaultdict(list)
    for idx, example in enumerate(dataset):
        accent = example['accent']
        accent_to_indices[accent].append(idx)
    
    # Limit each accent
    rng = np.random.RandomState(seed)
    selected_indices = []
    
    for accent, indices in accent_to_indices.items():
        if len(indices) > max_per_accent:
            indices = rng.choice(indices, size=max_per_accent, replace=False).tolist()
        selected_indices.extend(indices)
    
    logger.info(f"Balanced dataset: {len(selected_indices)} samples")
    return dataset.select(sorted(selected_indices))


def compute_dataset_stats(dataset_dict: DatasetDict, sample_rate: int = 16000) -> pd.DataFrame:
    """
    Compute statistics for each split and accent.
    
    Args:
        dataset_dict: Dataset splits
        sample_rate: Audio sample rate
    
    Returns:
        DataFrame with statistics
    """
    stats = []
    
    for split_name, split_ds in dataset_dict.items():
        accent_stats = defaultdict(lambda: {'n_samples': 0, 'total_duration': 0.0})
        
        for example in tqdm(split_ds, desc=f"Computing stats for {split_name}"):
            accent = example['accent']
            audio = example['audio']
            duration = get_audio_duration(audio['array'], audio['sampling_rate'])
            
            accent_stats[accent]['n_samples'] += 1
            accent_stats[accent]['total_duration'] += duration
        
        for accent, stat in accent_stats.items():
            stats.append({
                'split': split_name,
                'accent': accent,
                'n_samples': stat['n_samples'],
                'hours': stat['total_duration'] / 3600
            })
    
    return pd.DataFrame(stats)


def prepare_commonvoice_data(config: Dict) -> Tuple[DatasetDict, pd.DataFrame]:
    """
    Main function to prepare Common Voice data end-to-end.
    
    Args:
        config: Experiment configuration dictionary
    
    Returns:
        Dataset splits and statistics DataFrame
    """
    seed = config.get('seed', 42)
    target_accents = config['accents']
    data_config = config['data']
    
    # Load accent mappings
    mappings = load_accent_mappings()
    
    # Load dataset
    dataset = load_commonvoice_dataset(
        data_config['dataset_name_candidates'],
        language=config.get('language', 'en'),
        cache_dir='data/cache'
    )
    
    # Filter and canonicalize
    dataset, accent_counts = filter_and_canonicalize(
        dataset,
        target_accents,
        mappings,
        min_samples_per_accent=data_config.get('min_per_accent_train', 100)
    )
    
    # Speaker-stratified split
    dataset_dict = stratified_split_by_speaker(dataset, seed=seed)
    
    # Limit sizes per split
    max_train = data_config.get('max_per_accent_train', None)
    max_eval = data_config.get('max_per_accent_eval', None)
    
    if max_train:
        dataset_dict['train'] = balance_accents_in_split(
            dataset_dict['train'],
            max_per_accent=max_train,
            seed=seed
        )
    
    if max_eval:
        dataset_dict['validation'] = balance_accents_in_split(
            dataset_dict['validation'],
            max_per_accent=max_eval,
            seed=seed
        )
        dataset_dict['test'] = balance_accents_in_split(
            dataset_dict['test'],
            max_per_accent=max_eval,
            seed=seed
        )
    
    # Limit total train hours if specified
    max_hours = data_config.get('limit_total_train_hours', None)
    if max_hours:
        dataset_dict['train'] = limit_dataset_size(
            dataset_dict['train'],
            max_hours=max_hours,
            seed=seed
        )
    
    # Compute stats
    stats_df = compute_dataset_stats(dataset_dict)
    
    # Save dataset
    dataset_dict.save_to_disk('data/prepared')
    logger.info("Saved prepared dataset to data/prepared")
    
    return dataset_dict, stats_df

