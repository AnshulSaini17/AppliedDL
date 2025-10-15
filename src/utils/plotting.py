"""Plotting utilities for visualization."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Union


def plot_wer_by_accent(
    baseline_csv: Union[str, Path],
    finetuned_csv: Union[str, Path],
    out_png: Union[str, Path]
):
    """
    Plot WER comparison between baseline and fine-tuned models by accent.
    
    Args:
        baseline_csv: Path to baseline metrics CSV
        finetuned_csv: Path to fine-tuned metrics CSV
        out_png: Output path for plot
    """
    baseline = pd.read_csv(baseline_csv)
    finetuned = pd.read_csv(finetuned_csv)
    
    # Merge on accent
    merged = baseline.merge(
        finetuned,
        on='accent',
        suffixes=('_baseline', '_finetuned')
    )
    
    # Sort by baseline WER
    merged = merged.sort_values('wer_baseline', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(merged))
    width = 0.35
    
    ax.bar(
        [i - width/2 for i in x],
        merged['wer_baseline'],
        width,
        label='Baseline (zero-shot)',
        alpha=0.8,
        color='#e74c3c'
    )
    ax.bar(
        [i + width/2 for i in x],
        merged['wer_finetuned'],
        width,
        label='Fine-tuned',
        alpha=0.8,
        color='#3498db'
    )
    
    ax.set_xlabel('Accent', fontsize=12)
    ax.set_ylabel('WER (%)', fontsize=12)
    ax.set_title('Word Error Rate by Accent: Baseline vs Fine-tuned', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(merged['accent'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved WER comparison plot to {out_png}")


def plot_delta_wer(
    baseline_csv: Union[str, Path],
    finetuned_csv: Union[str, Path],
    out_png: Union[str, Path]
):
    """
    Plot ΔWER (improvement) by accent.
    
    Args:
        baseline_csv: Path to baseline metrics CSV
        finetuned_csv: Path to fine-tuned metrics CSV
        out_png: Output path for plot
    """
    baseline = pd.read_csv(baseline_csv)
    finetuned = pd.read_csv(finetuned_csv)
    
    # Merge and compute delta
    merged = baseline.merge(finetuned, on='accent', suffixes=('_baseline', '_finetuned'))
    merged['delta_wer'] = merged['wer_baseline'] - merged['wer_finetuned']
    
    # Sort by delta (most improvement first)
    merged = merged.sort_values('delta_wer', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#27ae60' if d > 0 else '#e74c3c' for d in merged['delta_wer']]
    
    ax.barh(merged['accent'], merged['delta_wer'], color=colors, alpha=0.8)
    
    ax.set_xlabel('ΔWER (Baseline - Fine-tuned, %)', fontsize=12)
    ax.set_ylabel('Accent', fontsize=12)
    ax.set_title('WER Improvement by Accent', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (accent, delta) in enumerate(zip(merged['accent'], merged['delta_wer'])):
        ax.text(
            delta + 0.3 if delta > 0 else delta - 0.3,
            i,
            f'{delta:.1f}',
            va='center',
            ha='left' if delta > 0 else 'right',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ΔWER plot to {out_png}")


def plot_error_distribution(
    predictions_jsonl: Union[str, Path],
    out_png: Union[str, Path]
):
    """
    Plot distribution of errors (optional utility).
    
    Args:
        predictions_jsonl: Path to predictions JSONL
        out_png: Output path for plot
    """
    # Placeholder for future extension
    pass

