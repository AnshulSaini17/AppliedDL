"""Zero-shot baseline inference with Whisper."""

import json
from pathlib import Path
from typing import Dict
import yaml
import torch
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd

from ..models.whisper_loader import load_whisper_model
from ..utils.seed import set_seed
from ..utils.logging import setup_logger
from ..utils.metrics import compute_metrics_by_accent, compute_overall_metrics

logger = setup_logger(__name__)


def run_baseline_inference(config_path: str):
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))
    
    # Output paths
    out_dir = Path(config['paths']['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_path = out_dir / 'baseline_metrics.csv'
    predictions_path = out_dir / 'baseline_predictions.jsonl'
    
    # Load test dataset
    logger.info("Loading test dataset...")
    dataset_dict = load_from_disk('data/prepared')
    test_dataset = dataset_dict['test']
    
    logger.info(f"Test samples: {len(test_dataset)}")
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model, processor = load_whisper_model(
        model_name=config['model_name_or_path'],
        language=config['language'],
        task=config['task'],
        use_lora=False,
        device=device
    )
    
    model.eval()
    
    # Generation config
    gen_config = config['training']['generation']
    generation_kwargs = {
        "max_new_tokens": 225,
        "num_beams": gen_config.get('num_beams', 1),
        "length_penalty": gen_config.get('length_penalty', 1.0),
        "no_repeat_ngram_size": gen_config.get('no_repeat_ngram_size', 0)
    }
    
    # Inference
    logger.info("Running baseline inference...")
    predictions = []
    
    batch_size = config['training']['per_device_eval_batch_size']
    
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Inference"):
        batch = test_dataset[i:i + batch_size]
        

        # Pad or truncate to exactly 30 seconds 
        import numpy as np
        max_samples = 30 * 16000
        audio_arrays = []
        for audio in batch['audio']:
            arr = audio['array']
            if len(arr) > max_samples:
                arr = arr[:max_samples]  # Truncate
            elif len(arr) < max_samples:
                # Pad with zeros
                arr = np.pad(arr, (0, max_samples - len(arr)), mode='constant')
            audio_arrays.append(arr)
        
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        input_features = inputs.input_features.to(device)
        
        # Convert to same dtype as model (fp16 on GPU)
        if device == "cuda":
            input_features = input_features.half()
        
        # Generate
        with torch.no_grad():
            predicted_ids = model.generate(input_features, **generation_kwargs)
        
        # Decode
        transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        
        # Store results
        for j, transcription in enumerate(transcriptions):
            idx = i + j
            
            predictions.append({
                'id': idx,
                'accent': batch['accent'][j],
                'reference': batch['sentence'][j],
                'hypothesis': transcription
            })
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Compute metrics by accent
    logger.info("Computing metrics...")
    metrics_by_accent = compute_metrics_by_accent(pred_df)
    
    # Add overall metrics
    overall_metrics = compute_overall_metrics(pred_df)
    overall_row = pd.DataFrame([{
        'accent': 'OVERALL',
        'n_samples': len(pred_df),
        'wer': overall_metrics['wer'],
        'cer': overall_metrics['cer'],
        'avg_len_chars': pred_df['reference'].str.len().mean()
    }])
    metrics_by_accent = pd.concat([metrics_by_accent, overall_row], ignore_index=True)
    
    # Save metrics
    metrics_by_accent.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE (Zero-shot) RESULTS")
    print("="*60)
    print(metrics_by_accent.to_string(index=False))
    print("="*60 + "\n")
    
    # Save predictions
    with open(predictions_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    logger.info(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    
    run_baseline_inference(args.config)

