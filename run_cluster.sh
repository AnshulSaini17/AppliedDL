#!/bin/bash
#SBATCH --job-name=accent-asr
#SBATCH --output=logs/accent-asr-%j.out
#SBATCH --error=logs/accent-asr-%j.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Example SLURM job script for university GPU clusters (e.g., LRZ)
# Adjust SBATCH parameters above based on your cluster's requirements

# Set your HuggingFace token here (required for Common Voice)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"  # Replace with your actual token

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate accent-asr

# Create logs directory
mkdir -p logs

# Run the experiment
echo "Starting data preparation..."
python -m src.cli prepare-data --config configs/experiment.yaml

echo "Running baseline inference..."
python -m src.cli run-baseline --config configs/experiment.yaml

echo "Starting fine-tuning..."
python -m src.cli finetune --config configs/experiment.yaml

echo "Evaluating fine-tuned model..."
python -m src.cli evaluate --config configs/experiment.yaml

echo "Generating summary and plots..."
python -m src.cli summarize --config configs/experiment.yaml

echo "Experiment complete! Results in outputs/"

