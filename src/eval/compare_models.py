"""Compare baseline and fine-tuned results with comprehensive plots."""

from pathlib import Path
from typing import Dict
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def compare_models(config_path: str):
    """
    Compare baseline and fine-tuned models with comprehensive analysis.
    
    Args:
        config_path: Path to experiment config
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir = Path(config['paths']['out_dir'])
    plots_dir = Path(config['paths']['plots_dir'])
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    baseline_path = out_dir / 'baseline_metrics.csv'
    finetuned_path = out_dir / 'finetuned_metrics.csv'
    
    # Check which files exist
    results = {}
    
    if baseline_path.exists():
        results['Baseline'] = pd.read_csv(baseline_path)
        logger.info("✓ Loaded baseline results")
    else:
        logger.error(f"Baseline not found: {baseline_path}")
        return
    
    if finetuned_path.exists():
        results['Fine-tuned'] = pd.read_csv(finetuned_path)
        logger.info("✓ Loaded fine-tuned results")
    else:
        logger.error(f"Fine-tuned results not found: {finetuned_path}")
        return
    
    # Create comparison table
    comparison_rows = []
    
    accents = results['Baseline']['accent'].unique()
    
    for accent in accents:
        row = {'accent': accent}
        
        for model_name, df in results.items():
            accent_row = df[df['accent'] == accent]
            if not accent_row.empty:
                row[f'wer_{model_name.lower()}'] = accent_row['wer'].values[0]
                row[f'cer_{model_name.lower()}'] = accent_row['cer'].values[0]
        
        comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Compute delta
    if 'wer_baseline' in comparison_df.columns and 'wer_fine-tuned' in comparison_df.columns:
        comparison_df['delta_wer'] = comparison_df['wer_baseline'] - comparison_df['wer_fine-tuned']
        comparison_df['delta_cer'] = comparison_df['cer_baseline'] - comparison_df['cer_fine-tuned']
    
    # Save comparison
    comparison_path = out_dir / 'comparison_table.csv'
    comparison_df.to_csv(comparison_path, index=False, float_format='%.2f')
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Fine-tuned")
    print("="*80)
    print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*80 + "\n")
    
    # Generate comparison plot
    logger.info("Generating comparison plot...")
    
    # Exclude OVERALL row for plotting
    plot_df = comparison_df[comparison_df['accent'] != 'OVERALL'].copy()
    
    if len(plot_df) == 0:
        logger.warning("No accent data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    accents = plot_df['accent'].values
    x = np.arange(len(accents))
    width = 0.35
    
    # Plot 1: WER comparison
    ax1.bar(x - width/2, plot_df['wer_baseline'], width, label='Baseline (zero-shot)', color='#FF6B6B')
    ax1.bar(x + width/2, plot_df['wer_fine-tuned'], width, label='Fine-tuned', color='#4ECDC4')
    
    ax1.set_xlabel('Speech Quality', fontsize=12)
    ax1.set_ylabel('WER (%)', fontsize=12)
    ax1.set_title('Word Error Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(accents, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Delta WER (improvement)
    delta_values = plot_df['delta_wer'].values
    colors = ['#00C853' if d > 0 else '#FF5252' for d in delta_values]
    
    ax2.bar(x, delta_values, width, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax2.set_xlabel('Speech Quality', fontsize=12)
    ax2.set_ylabel('ΔWER (Baseline - Fine-tuned, %)', fontsize=12)
    ax2.set_title('WER Change from Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(accents, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(delta_values):
        ax2.text(i, v + (0.5 if v > 0 else -0.5), f'{v:.1f}', 
                ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plot_path = plots_dir / 'comparison_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved plot to {plot_path}")
    
    # Generate comprehensive plots
    logger.info("Generating additional analysis plots...")
    from ..utils.advanced_plotting import create_comprehensive_report
    create_comprehensive_report(out_dir, plots_dir)
    
    logger.info("Comparison complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    
    compare_models(args.config)

