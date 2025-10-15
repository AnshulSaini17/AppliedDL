#!/bin/bash
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o logs/output_%j.log
#SBATCH --job-name=accent-asr

source ~/.bashrc
mamba activate accent-asr

# === USE BIG DSS STORAGE (150GB quota!) ===
export DSS_STORAGE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/$USER"

# All caches and heavy data go to big storage
export HF_HOME="$DSS_STORAGE/hf-cache"
export HF_HUB_CACHE="$DSS_STORAGE/hf-cache/hub"
export HF_DATASETS_CACHE="$DSS_STORAGE/hf-cache/datasets"
export TRANSFORMERS_CACHE="$DSS_STORAGE/hf-cache/transformers"
export TORCH_HOME="$DSS_STORAGE/torch-cache"
export TMPDIR="$DSS_STORAGE/tmp"

# Create needed dirs
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
         "$TORCH_HOME" "$TMPDIR"

# Verify we can write
if [ ! -w "$DSS_STORAGE" ]; then
  echo "ERROR: No write permission on $DSS_STORAGE"; exit 1
fi

# Run from home directory (code is small, keep it there)
cd ~/AppliedDL || { echo "Repo not found at ~/AppliedDL"; exit 1; }

echo "=========================================="
echo "Starting Accent-ASR Pipeline"
echo "Caches:"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "  TMPDIR=$TMPDIR"
echo "=========================================="

# Helper: file exists and non-empty?
check_done() { [ -f "$1" ] && [ -s "$1" ]; }

# Step 1: Data Preparation
if check_done "data/prepared/dataset_dict.json"; then
  echo "[SKIP] Data already prepared"
else
  echo "[RUN] Step 1: Data preparation..."
  python -m src.cli prepare-data --config configs/experiment.yaml || { echo "[ERROR] Data preparation failed!"; exit 1; }
  echo "[DONE] Data preparation completed"
fi

# Step 2: Baseline Inference
if check_done "outputs/baseline_metrics.csv"; then
  echo "[SKIP] Baseline already completed"
else
  echo "[RUN] Step 2: Baseline inference..."
  python -m src.cli run-baseline --config configs/experiment.yaml || { echo "[ERROR] Baseline inference failed!"; exit 1; }
  echo "[DONE] Baseline completed"
fi

# Step 3: Fine-tuning
# Check both possible checkpoint locations (LoRA and non-LoRA)
if check_done "checkpoints/whisper-small-finetuned-lora/config.json" || check_done "checkpoints/whisper-small-finetuned/config.json"; then
  echo "[SKIP] Fine-tuning already completed"
else
  echo "[RUN] Step 3: Fine-tuning model..."
  python -m src.cli finetune --config configs/experiment.yaml || { echo "[ERROR] Fine-tuning failed!"; exit 1; }
  echo "[DONE] Fine-tuning completed"
fi

# Step 4: Evaluation
if check_done "outputs/finetuned_metrics.csv"; then
  echo "[SKIP] Evaluation already completed"
else
  # Check if fine-tuning checkpoint exists before evaluating
  if [ ! -f "checkpoints/whisper-small-finetuned-lora/config.json" ] && [ ! -f "checkpoints/whisper-small-finetuned/config.json" ]; then
    echo "[ERROR] No fine-tuned checkpoint found! Run fine-tuning first."
    exit 1
  fi
  
  echo "[RUN] Step 4: Evaluating fine-tuned model..."
  python -m src.cli evaluate --config configs/experiment.yaml || { echo "[ERROR] Evaluation failed!"; exit 1; }
  echo "[DONE] Evaluation completed"
fi

# Step 5: Summary
if check_done "outputs/summary_table.csv"; then
  echo "[SKIP] Summary already generated"
else
  echo "[RUN] Step 5: Generating summary..."
  python -m src.cli summarize --config configs/experiment.yaml || { echo "[ERROR] Summary generation failed!"; exit 1; }
  echo "[DONE] Summary generated"
fi

echo "=========================================="
echo "Pipeline Complete! ✓"
echo "Results in: ~/AppliedDL/outputs/"
echo "Checkpoints in: ~/AppliedDL/checkpoints/"
echo "Heavy data cached in: $DSS_STORAGE/hf-cache/"
echo "=========================================="
