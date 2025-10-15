"""Metrics computation utilities."""

import jiwer
from typing import List, Dict, Tuple
import pandas as pd
from .utils import normalize_for_metrics


def compute_wer(references: List[str], hypotheses: List[str]) -> float:

    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    if not references:
        return 0.0
    
    # Normalize texts
    refs = [normalize_for_metrics(r) for r in references]
    hyps = [normalize_for_metrics(h) for h in hypotheses]
    
    # Remove empty pairs
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 0.0
    
    refs, hyps = zip(*pairs)
    
    try:
        wer = jiwer.wer(list(refs), list(hyps))
        return wer * 100.0
    except Exception as e:
        print(f"Warning: WER computation failed: {e}")
        return 0.0


def compute_cer(references: List[str], hypotheses: List[str]) -> float:

    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    if not references:
        return 0.0
    
    # Normalize texts
    refs = [normalize_for_metrics(r) for r in references]
    hyps = [normalize_for_metrics(h) for h in hypotheses]
    
    # Remove empty pairs
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 0.0
    
    refs, hyps = zip(*pairs)
    
    try:
        cer = jiwer.cer(list(refs), list(hyps))
        return cer * 100.0
    except Exception as e:
        print(f"Warning: CER computation failed: {e}")
        return 0.0


def compute_metrics_batch(
    pred_strs: List[str],
    ref_strs: List[str]
) -> Dict[str, float]:
    
    return {
        "wer": compute_wer(ref_strs, pred_strs),
        "cer": compute_cer(ref_strs, pred_strs)
    }


def compute_metrics_by_accent(
    predictions: pd.DataFrame
) -> pd.DataFrame:
    
    results = []
    
    for accent in predictions['accent'].unique():
        accent_df = predictions[predictions['accent'] == accent]
        
        refs = accent_df['reference'].tolist()
        hyps = accent_df['hypothesis'].tolist()
        
        metrics = compute_metrics_batch(hyps, refs)
        
        results.append({
            'accent': accent,
            'n_samples': len(accent_df),
            'wer': metrics['wer'],
            'cer': metrics['cer'],
            'avg_len_chars': accent_df['reference'].str.len().mean()
        })
    
    return pd.DataFrame(results)


def compute_overall_metrics(predictions: pd.DataFrame) -> Dict[str, float]:
    
    refs = predictions['reference'].tolist()
    hyps = predictions['hypothesis'].tolist()
    
    return compute_metrics_batch(hyps, refs)

