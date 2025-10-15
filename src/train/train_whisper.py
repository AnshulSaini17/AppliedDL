"""Whisper fine-tuning script."""

import os
import sys
from pathlib import Path
from typing import Dict
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate

from ..models.whisper_loader import load_whisper_model
from ..models.collator import WhisperDataCollator
from ..utils.utils import set_seed, setup_logger
from ..utils.metrics import compute_wer

logger = setup_logger(__name__)


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    def training_step(self, model, inputs):
        # Filter inputs to only what Whisper needs
        keys_to_remove = [k for k in inputs.keys() if k not in ['input_features', 'decoder_input_ids', 'labels']]
        for key in keys_to_remove:
            inputs.pop(key, None)
        return super().training_step(model, inputs)
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        keys_to_remove = [k for k in inputs.keys() if k not in ['input_features', 'decoder_input_ids', 'labels']]
        for key in keys_to_remove:
            inputs.pop(key, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def prepare_compute_metrics(processor, tokenizer):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = compute_wer(label_str, pred_str)
        return {"wer": wer}
    
    return compute_metrics


def train_whisper(config_path: str, **overrides):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    for key, value in overrides.items():
        if '.' in key:
            # Nested key
            parts = key.split('.')
            d = config
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value
        else:
            config[key] = value
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Load prepared dataset
    logger.info("Loading prepared dataset...")
    dataset_dict = load_from_disk('data/prepared')
    
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['validation']
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Eval samples: {len(eval_dataset)}")
    
    # Load model
    use_lora = config['peft'].get('use_lora', False)
    model, processor = load_whisper_model(
        model_name=config['model_name_or_path'],
        language=config['language'],
        task=config['task'],
        use_lora=use_lora,
        lora_config=config['peft'] if use_lora else None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        for_training=True
    )
    
    data_collator = WhisperDataCollator(processor=processor)
    
    train_config = config['training']
    output_dir = Path(config['paths']['ckpt_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        warmup_ratio=train_config['warmup_ratio'],
        num_train_epochs=train_config['num_train_epochs'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=train_config.get('fp16', False) and torch.cuda.is_available(),
        bf16=train_config.get('bf16', False) and torch.cuda.is_available(),
        logging_steps=50,
        report_to=["none"],
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=train_config['generation'].get('num_beams', 1),
        label_smoothing_factor=train_config.get('label_smoothing_factor', 0.0),
        push_to_hub=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
        seed=seed
    )
    
    compute_metrics_fn = prepare_compute_metrics(processor, processor.tokenizer)
    
    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    # Save training history
    import json
    history_path = Path(config['paths']['out_dir']) / 'training_history.json'
    history_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract training metrics from trainer.state.log_history
    training_history = {
        'log_history': trainer.state.log_history,
        'train_loss': train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        'metrics': train_result.metrics if hasattr(train_result, 'metrics') else {}
    }
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Saved training history to {history_path}")
    
    # Save metrics
    metrics_path = Path(config['paths']['out_dir']) / 'train_logs.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(metrics_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use-lora", action="store_true")
    
    args = parser.parse_args()
    
    overrides = {}
    if args.epochs:
        overrides['training.num_train_epochs'] = args.epochs
    if args.use_lora:
        overrides['peft.use_lora'] = True
    
    train_whisper(args.config, **overrides)

