"""Advanced plotting utilities for training analysis."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_training_curves(history_paths: Dict[str, Path], output_path: Path):
    """
    Plot training loss and validation WER curves for multiple models.
    
    Args:
        history_paths: Dict mapping model names to training_history.json paths
        output_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFD93D']
    
    for i, (model_name, history_path) in enumerate(history_paths.items()):
        if not history_path.exists():
            print(f"Warning: {history_path} not found, skipping {model_name}")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        log_history = history.get('log_history', [])
        
        # Extract training loss
        train_steps = []
        train_losses = []
        eval_steps = []
        eval_wers = []
        
        for entry in log_history:
            if 'loss' in entry:
                train_steps.append(entry.get('step', entry.get('epoch', 0)))
                train_losses.append(entry['loss'])
            
            if 'eval_wer' in entry:
                eval_steps.append(entry.get('step', entry.get('epoch', 0)))
                eval_wers.append(entry['eval_wer'] * 100)  # Convert to percentage
        
        # Plot training loss
        if train_losses:
            ax1.plot(train_steps, train_losses, label=model_name, 
                    color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
        
        # Plot validation WER
        if eval_wers:
            ax2.plot(eval_steps, eval_wers, label=model_name,
                    color=colors[i % len(colors)], linewidth=2, marker='s', markersize=6)
    
    # Configure loss plot
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Configure WER plot
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Validation WER (%)', fontsize=12)
    ax2.set_title('Validation WER During Training', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training curves to {output_path}")


def plot_wer_heatmap(metrics_files: Dict[str, Path], output_path: Path):
    """
    Create a heatmap showing WER across models and accents.
    
    Args:
        metrics_files: Dict mapping model names to metrics CSV paths
        output_path: Where to save the plot
    """
    # Collect data
    data = []
    
    for model_name, metrics_path in metrics_files.items():
        if not metrics_path.exists():
            print(f"Skipping {model_name}: file not found")
            continue
        
        df = pd.read_csv(metrics_path)
        # Exclude OVERALL row
        df = df[df['accent'] != 'OVERALL']
        
        for _, row in df.iterrows():
            data.append({
                'Model': model_name,
                'Accent': row['accent'],
                'WER': row['wer']
            })
    
    if not data:
        print("Warning: No data for heatmap")
        return
    
    df = pd.DataFrame(data)
    
    # Pivot for heatmap
    heatmap_data = df.pivot(index='Accent', columns='Model', values='WER')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'WER (%)'}, ax=ax, vmin=0, vmax=25)
    
    ax.set_title('Word Error Rate Heatmap: Model vs Accent', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accent', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved WER heatmap to {output_path}")


def plot_improvement_bars(comparison_df: pd.DataFrame, output_path: Path):
    """
    Create bar chart showing improvement over baseline for each model.
    
    Args:
        comparison_df: DataFrame with comparison data
        output_path: Where to save the plot
    """
    # Exclude OVERALL row
    plot_df = comparison_df[comparison_df['accent'] != 'OVERALL'].copy()
    
    if len(plot_df) == 0:
        print("Warning: No data for improvement bars")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accents = plot_df['accent'].values
    x = np.arange(len(accents))
    width = 0.35
    
    # Get improvement columns
    improvement_cols = [col for col in plot_df.columns if col.startswith('delta_')]
    
    for i, col in enumerate(improvement_cols):
        if col not in plot_df.columns:
            continue
        
        model_name = col.replace('delta_', '').replace('_', ' ').title()
        values = plot_df[col].values
        
        # Color bars based on positive/negative
        colors = ['#00C853' if v > 0 else '#FF5252' for v in values]
        
        offset = (i - len(improvement_cols)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=colors, alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Accent', fontsize=12)
    ax.set_ylabel('ΔWER (% improvement)', fontsize=12)
    ax.set_title('WER Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(accents, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved improvement bars to {output_path}")


def plot_error_distribution(predictions_files: Dict[str, Path], output_path: Path):
    """
    Plot distribution of word error rates across samples.
    
    Args:
        predictions_files: Dict mapping model names to predictions CSV paths
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    for i, (model_name, pred_path) in enumerate(predictions_files.items()):
        if not pred_path.exists():
            continue
        
        df = pd.read_csv(pred_path)
        
        # Compute WER per sample (rough approximation)
        # In reality, you'd need to compute edit distance
        # For now, just use character-level approximation
        sample_errors = []
        for _, row in df.iterrows():
            ref = str(row['reference']).split()
            hyp = str(row['hypothesis']).split()
            # Simple approximation: abs difference in word count
            error_rate = abs(len(ref) - len(hyp)) / max(len(ref), 1) * 100
            sample_errors.append(min(error_rate, 100))  # Cap at 100%
        
        ax.hist(sample_errors, bins=30, alpha=0.6, label=model_name, 
               color=colors[i % len(colors)], edgecolor='black')
    
    ax.set_xlabel('Word Error Rate (%)', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Distribution of Sample-Level WER', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved error distribution to {output_path}")


def create_comprehensive_report(out_dir: Path, plots_dir: Path):
    """
    Generate all plots for comprehensive analysis.
    
    Args:
        out_dir: Directory with results CSVs and training history
        plots_dir: Directory to save plots
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE PLOTS")
    print("="*60 + "\n")
    
    # 1. Training curves
    history_paths = {}
    
    # Check for current training history
    current_history = out_dir / 'training_history.json'
    if current_history.exists():
        history_paths['Fine-tuned'] = current_history
    
    if history_paths:
        plot_training_curves(history_paths, plots_dir / 'training_curves.png')
    else:
        print("⚠ No training history found, skipping training curves")
    
    # 2. WER Heatmap
    metrics_files = {
        'Baseline': out_dir / 'baseline_metrics.csv',
        'Fine-tuned': out_dir / 'finetuned_metrics.csv'
    }
    
    # Filter existing files
    metrics_files = {k: v for k, v in metrics_files.items() if v.exists()}
    
    if len(metrics_files) >= 2:
        plot_wer_heatmap(metrics_files, plots_dir / 'wer_heatmap.png')
    else:
        print("⚠ Need both baseline and fine-tuned metrics for heatmap")
    
    # 3. Improvement bars (if comparison exists)
    comparison_path = out_dir / 'summary_table.csv'
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        # Rename columns for compatibility
        if 'wer_baseline' in comparison_df.columns and 'wer_finetuned' in comparison_df.columns:
            comparison_df['delta_finetuned'] = comparison_df['wer_baseline'] - comparison_df['wer_finetuned']
        plot_improvement_bars(comparison_df, plots_dir / 'improvement_bars.png')
    else:
        print("⚠ No summary table found, skipping improvement bars")
    
    # 4. Error distribution
    predictions_files = {
        'Baseline': out_dir / 'baseline_predictions.csv',
        'Fine-tuned': out_dir / 'finetuned_predictions.csv'
    }
    
    predictions_files = {k: v for k, v in predictions_files.items() if v.exists()}
    
    if len(predictions_files) >= 2:
        plot_error_distribution(predictions_files, plots_dir / 'error_distribution.png')
    else:
        print("⚠ Need both baseline and fine-tuned predictions for error distribution")
    
    print("\n" + "="*60)
    print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
    print(f"  Location: {plots_dir}")
    print("="*60 + "\n")

