"""Command-line interface for the experiment."""

import argparse
import sys
from pathlib import Path
import yaml

from .utils.logging import setup_logger
from .utils.seed import set_seed

logger = setup_logger(__name__)


def prepare_data_command(args):
    """Prepare data subcommand."""
    logger.info("Starting data preparation...")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Load LibriSpeech dataset
    logger.info("Loading LibriSpeech dataset")
    from .data.load_librispeech import load_librispeech_data
    dataset_dict, stats_df = load_librispeech_data(config)
    
    # Save stats
    out_dir = Path(config['paths']['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / 'dataset_stats.csv'
    stats_df.to_csv(stats_path, index=False)
    
    logger.info(f"Saved dataset statistics to {stats_path}")
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(stats_df.to_string(index=False))
    print("="*60 + "\n")
    
    logger.info("Data preparation complete!")


def run_baseline_command(args):
    """Run baseline inference subcommand."""
    from .eval.baseline_infer import run_baseline_inference
    
    logger.info("Running baseline inference...")
    run_baseline_inference(args.config)


def finetune_command(args):
    """Fine-tune model subcommand."""
    from .train.train_whisper import train_whisper
    
    logger.info("Starting fine-tuning...")
    
    overrides = {}
    if args.epochs:
        overrides['training.num_train_epochs'] = args.epochs
    if args.use_lora:
        overrides['peft.use_lora'] = True
    
    train_whisper(args.config, **overrides)


def evaluate_command(args):
    """Evaluate checkpoint subcommand."""
    from .eval.evaluate_ckpt import evaluate_checkpoint
    
    logger.info("Evaluating checkpoint...")
    evaluate_checkpoint(args.config, args.checkpoint)


def summarize_command(args):
    """Summarize results subcommand."""
    from .eval.summarize_results import summarize_results
    
    logger.info("Summarizing results...")
    summarize_results(args.config)


def compare_command(args):
    """Three-way comparison subcommand."""
    from .eval.compare_three_models import compare_three_models
    
    logger.info("Running three-way comparison...")
    compare_three_models(args.config)


def plot_command(args):
    """Generate all plots subcommand."""
    from .utils.advanced_plotting import create_comprehensive_report
    from pathlib import Path
    import yaml
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    out_dir = Path(config['paths']['out_dir'])
    plots_dir = Path(config['paths']['plots_dir'])
    
    logger.info("Generating all plots...")
    create_comprehensive_report(out_dir, plots_dir)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Accent-aware ASR experiment with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Prepare data
    prepare_parser = subparsers.add_parser(
        'prepare-data',
        help='Prepare LibriSpeech dataset'
    )
    prepare_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    prepare_parser.set_defaults(func=prepare_data_command)
    
    # Run baseline
    baseline_parser = subparsers.add_parser(
        'run-baseline',
        help='Run zero-shot baseline inference'
    )
    baseline_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    baseline_parser.set_defaults(func=run_baseline_command)
    
    # Fine-tune
    finetune_parser = subparsers.add_parser(
        'finetune',
        help='Fine-tune Whisper model'
    )
    finetune_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    finetune_parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of training epochs'
    )
    finetune_parser.add_argument(
        '--use-lora',
        action='store_true',
        help='Enable LoRA adapters'
    )
    finetune_parser.set_defaults(func=finetune_command)
    
    # Evaluate
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate fine-tuned checkpoint'
    )
    evaluate_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    evaluate_parser.add_argument(
        '--checkpoint',
        default=None,
        help='Path to checkpoint (overrides config)'
    )
    evaluate_parser.set_defaults(func=evaluate_command)
    
    # Summarize
    summarize_parser = subparsers.add_parser(
        'summarize',
        help='Summarize and compare results'
    )
    summarize_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    summarize_parser.set_defaults(func=summarize_command)
    
    # Compare (3-way comparison)
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare baseline, fine-tuned (no LoRA), and fine-tuned (LoRA)'
    )
    compare_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    compare_parser.set_defaults(func=compare_command)
    
    # Plot
    plot_parser = subparsers.add_parser(
        'plot',
        help='Generate all analysis plots'
    )
    plot_parser.add_argument(
        '--config',
        default='configs/experiment.yaml',
        help='Path to experiment config'
    )
    plot_parser.set_defaults(func=plot_command)
    
    # Parse args
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    args.func(args)


if __name__ == '__main__':
    main()

