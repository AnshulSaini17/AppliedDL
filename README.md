# Whisper ASR Fine-tuning on LibriSpeech

Baseline vs fine-tuned Whisper-small model on LibriSpeech dataset.

## Setup

```bash
conda env create -f environment.yml
conda activate accent-asr
```

## Dataset

LibriSpeech ASR dataset:
- Clean split: High-quality audio  
- Other split: More challenging audio
- ~6GB total

## Running Experiments

```bash
# 1. Prepare data
python -m src.cli prepare-data --config configs/experiment.yaml

# 2. Baseline
python -m src.cli run-baseline --config configs/experiment.yaml

# 3. Fine-tune
python -m src.cli finetune --config configs/experiment.yaml

# 4. Evaluate
python -m src.cli evaluate --config configs/experiment.yaml
python -m src.cli summarize --config configs/experiment.yaml

# 5. Generate plots
python -m src.cli compare --config configs/experiment.yaml
```

## Results

Results in `outputs/`:
- `baseline_metrics.csv` - Baseline WER/CER
- `finetuned_metrics.csv` - Fine-tuned WER/CER  
- `summary_table.csv` - Comparison
- `training_history.json` - Training metrics
- `plots/` - Visualizations

## Key Findings

Baseline performance: 5.95% WER  
Fine-tuned performance: 18.93% WER

Fine-tuning on limited data (1500 samples) caused performance degradation, demonstrating catastrophic forgetting.

## Requirements

- Python 3.10+
- CUDA GPU recommended
