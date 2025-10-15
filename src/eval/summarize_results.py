"""Summarize and compare baseline vs fine-tuned results."""

from pathlib import Path
from typing import Dict
import yaml
import pandas as pd

from ..utils.logging import setup_logger
from ..utils.plotting import plot_wer_by_accent, plot_delta_wer

logger = setup_logger(__name__)


def compute_robustness_metric(metrics_df: pd.DataFrame) -> float:
    """
    Compute accent robustness as max_WER - min_WER.
    
    Args:
        metrics_df: DataFrame with per-accent WER
    
    Returns:
        Robustness metric (lower is better)
    """
    # Exclude overall row
    accent_metrics = metrics_df[metrics_df['accent'] != 'OVERALL']
    
    if len(accent_metrics) == 0:
        return 0.0
    
    max_wer = accent_metrics['wer'].max()
    min_wer = accent_metrics['wer'].min()
    
    return max_wer - min_wer


def summarize_results(config_path: str):
    """
    Summarize baseline vs fine-tuned results and generate plots.
    
    Args:
        config_path: Path to experiment config
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir = Path(config['paths']['out_dir'])
    plots_dir = Path(config['paths']['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_metrics_path = out_dir / 'baseline_metrics.csv'
    finetuned_metrics_path = out_dir / 'finetuned_metrics.csv'
    summary_path = out_dir / 'summary_table.csv'
    
    # Check files exist
    if not baseline_metrics_path.exists():
        logger.error(f"Baseline metrics not found: {baseline_metrics_path}")
        return
    
    if not finetuned_metrics_path.exists():
        logger.error(f"Fine-tuned metrics not found: {finetuned_metrics_path}")
        return
    
    # Load metrics
    logger.info("Loading results...")
    baseline_df = pd.read_csv(baseline_metrics_path)
    finetuned_df = pd.read_csv(finetuned_metrics_path)
    
    # Merge on accent
    merged = baseline_df.merge(
        finetuned_df,
        on='accent',
        suffixes=('_baseline', '_finetuned')
    )
    
    # Compute delta WER and CER
    merged['delta_wer'] = merged['wer_baseline'] - merged['wer_finetuned']
    merged['delta_cer'] = merged['cer_baseline'] - merged['cer_finetuned']
    
    # Compute relative improvement
    merged['rel_wer_improvement'] = (merged['delta_wer'] / merged['wer_baseline']) * 100
    merged['rel_cer_improvement'] = (merged['delta_cer'] / merged['cer_baseline']) * 100
    
    # Robustness metrics
    baseline_robustness = compute_robustness_metric(baseline_df)
    finetuned_robustness = compute_robustness_metric(finetuned_df)
    delta_robustness = baseline_robustness - finetuned_robustness
    
    # Prepare summary table
    summary = merged[[
        'accent',
        'wer_baseline',
        'wer_finetuned',
        'delta_wer',
        'rel_wer_improvement',
        'cer_baseline',
        'cer_finetuned',
        'delta_cer'
    ]].copy()
    
    # Add robustness row for overall
    robustness_summary = {
        'accent': 'ROBUSTNESS_METRIC',
        'wer_baseline': baseline_robustness,
        'wer_finetuned': finetuned_robustness,
        'delta_wer': delta_robustness,
        'rel_wer_improvement': None,
        'cer_baseline': None,
        'cer_finetuned': None,
        'delta_cer': None
    }
    
    summary = pd.concat([summary, pd.DataFrame([robustness_summary])], ignore_index=True)
    
    # Save summary
    summary.to_csv(summary_path, index=False, float_format='%.2f')
    logger.info(f"Saved summary to {summary_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: Baseline vs Fine-tuned")
    print("="*80)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*80)
    print(f"\nAccent Robustness (WER range):")
    print(f"  Baseline:   {baseline_robustness:.2f}%")
    print(f"  Fine-tuned: {finetuned_robustness:.2f}%")
    print(f"  Improvement: {delta_robustness:.2f}% (lower is better)")
    print("="*80 + "\n")
    
    # Generate plots
    logger.info("Generating plots...")
    
    # Exclude OVERALL row for plotting
    baseline_plot = baseline_df[baseline_df['accent'] != 'OVERALL']
    finetuned_plot = finetuned_df[finetuned_df['accent'] != 'OVERALL']
    
    # Save temporary CSVs for plotting
    baseline_plot.to_csv(out_dir / 'baseline_plot.csv', index=False)
    finetuned_plot.to_csv(out_dir / 'finetuned_plot.csv', index=False)
    
    # WER comparison plot
    plot_wer_by_accent(
        out_dir / 'baseline_plot.csv',
        out_dir / 'finetuned_plot.csv',
        plots_dir / 'wer_by_accent.png'
    )
    
    # Delta WER plot
    plot_delta_wer(
        out_dir / 'baseline_plot.csv',
        out_dir / 'finetuned_plot.csv',
        plots_dir / 'delta_wer_by_accent.png'
    )
    
    logger.info("Summary complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    
    summarize_results(args.config)

