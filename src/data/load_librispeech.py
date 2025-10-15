"""LibriSpeech dataset loader"""

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from ..utils.audio import get_audio_duration
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def load_librispeech_data(config: Dict) -> Tuple[DatasetDict, pd.DataFrame]:
    seed = config.get('seed', 42)
    data_config = config['data']
    
    logger.info("Loading LibriSpeech dataset with LIMITED subsets to save disk space...")
    
    max_train = data_config.get('max_per_accent_train', 1500)
    max_val = data_config.get('max_per_accent_eval', 300)
    max_test = 500
    
    logger.info(f"Loading training data: train.100 (will select {max_train} samples)")
    
    train_full = load_dataset('librispeech_asr', 'clean', split='train.100')
    train_ds = train_full.select(range(min(max_train, len(train_full))))
    logger.info(f"Selected {len(train_ds)} training samples")
    
    logger.info(f"Loading validation data (will select {max_val} samples)")
    val_full = load_dataset('librispeech_asr', 'clean', split='validation')
    val_ds = val_full.select(range(min(max_val, len(val_full))))
    logger.info(f"Selected {len(val_ds)} validation samples")
    
    logger.info(f"Loading test.clean (will select {max_test} samples)")
    test_clean_full = load_dataset('librispeech_asr', 'clean', split='test')
    test_clean = test_clean_full.select(range(min(max_test, len(test_clean_full))))
    logger.info(f"Selected {len(test_clean)} test.clean samples")
    
    logger.info(f"Loading test.other (will select {max_test} samples)")
    test_other_full = load_dataset('librispeech_asr', 'other', split='test')
    test_other = test_other_full.select(range(min(max_test, len(test_other_full))))
    logger.info(f"Selected {len(test_other)} test.other samples")
    
    # Add "accent" field based on which dataset it came from
    def add_accent_clean(example):
        example['accent'] = 'clean'
        return example
    
    def add_accent_other(example):
        example['accent'] = 'other'
        return example
    
    # Add accent labels BEFORE concatenating
    train_ds = train_ds.map(add_accent_clean)
    val_ds = val_ds.map(add_accent_clean)
    test_clean = test_clean.map(add_accent_clean)
    test_other = test_other.map(add_accent_other)
    
    # Now combine test sets
    from datasets import concatenate_datasets
    test_ds = concatenate_datasets([test_clean, test_other])
    

    def rename_text_field(example):
        example['sentence'] = example['text']
        return example
    
    train_ds = train_ds.map(rename_text_field)
    val_ds = val_ds.map(rename_text_field)
    test_ds = test_ds.map(rename_text_field)
    
    max_train = data_config.get('max_per_accent_train', None)
    max_eval = data_config.get('max_per_accent_eval', None)
    
    if max_train and len(train_ds) > max_train:
        logger.info(f"Limiting training data to {max_train} samples")
        indices = np.random.RandomState(seed).choice(len(train_ds), max_train, replace=False)
        train_ds = train_ds.select(indices)
    
    if max_eval and len(val_ds) > max_eval:
        logger.info(f"Limiting validation data to {max_eval} samples")
        indices = np.random.RandomState(seed).choice(len(val_ds), max_eval, replace=False)
        val_ds = val_ds.select(indices)
    
    if max_eval and len(test_ds) > max_eval:
        logger.info(f"Limiting test data to {max_eval} samples")
        indices = np.random.RandomState(seed).choice(len(test_ds), max_eval, replace=False)
        test_ds = test_ds.select(indices)
    
    # Create dataset dict
    dataset_dict = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds
    })
    
    # Save to disk
    dataset_dict.save_to_disk('data/prepared')
    logger.info("Saved prepared dataset to data/prepared")
    
    # Compute statistics
    stats = []
    for split_name, split_ds in dataset_dict.items():
        accent_counts = defaultdict(int)
        accent_durations = defaultdict(float)
        
        for example in tqdm(split_ds, desc=f"Computing stats for {split_name}"):
            accent = example['accent']
            accent_counts[accent] += 1
            
            # Get audio duration
            audio = example['audio']
            duration = get_audio_duration(audio['array'], audio['sampling_rate'])
            accent_durations[accent] += duration
        
        for accent in accent_counts:
            stats.append({
                'split': split_name,
                'accent': accent,
                'n_samples': accent_counts[accent],
                'hours': accent_durations[accent] / 3600
            })
    
    stats_df = pd.DataFrame(stats)
    
    return dataset_dict, stats_df

