# Running Steps Separately (For Manual Control)

If you prefer to run each step individually (so you can check results between steps), use these commands:

## Setup Once

```bash
source ~/.bashrc
mamba activate accent-asr
export HF_TOKEN="your_token_here"
cd ~/AppliedDL
```

## Individual Steps

### Step 1: Data Preparation (~15-30 min)
```bash
python -m src.cli prepare-data --config configs/experiment.yaml
# Check: data/prepared/ should exist
# Check: outputs/dataset_stats.csv should exist
```

### Step 2: Baseline Inference (~10-20 min)
```bash
python -m src.cli run-baseline --config configs/experiment.yaml
# Check: outputs/baseline_metrics.csv
# Check: outputs/baseline_predictions.jsonl
```

### Step 3: Fine-tuning (~1-3 hours)
```bash
python -m src.cli finetune --config configs/experiment.yaml
# Check: checkpoints/whisper-small-finetuned/
```

### Step 4: Evaluate Fine-tuned (~10-20 min)
```bash
python -m src.cli evaluate --config configs/experiment.yaml
# Check: outputs/finetuned_metrics.csv
# Check: outputs/finetuned_predictions.jsonl
```

### Step 5: Generate Summary (~1 min)
```bash
python -m src.cli summarize --config configs/experiment.yaml
# Check: outputs/summary_table.csv
# Check: outputs/plots/*.png
```

## Submitting Individual Steps as Jobs

You can also submit each step as a separate SLURM job:

```bash
# Step 1 only
sbatch --wrap="source ~/.bashrc; mamba activate accent-asr; export HF_TOKEN='your_token'; cd ~/AppliedDL; python -m src.cli prepare-data --config configs/experiment.yaml" -o prep_%j.log

# Step 2 only (after step 1 completes)
sbatch --wrap="source ~/.bashrc; mamba activate accent-asr; cd ~/AppliedDL; python -m src.cli run-baseline --config configs/experiment.yaml" -o baseline_%j.log

# And so on...
```

## Resume After Failure

If a step fails:
1. Fix the issue
2. Just run that specific step again
3. Continue with the next steps

The pipeline is designed so each step is independent once its prerequisites are done.

