from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import numpy as np
from transformers import WhisperProcessor


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_samples = 30 * 16000
        audio_arrays = []
        for feature in features:
            arr = feature['audio']['array']
            if len(arr) > max_samples:
                arr = arr[:max_samples]
            elif len(arr) < max_samples:
                arr = np.pad(arr, (0, max_samples - len(arr)), mode='constant')
            audio_arrays.append(arr)
        
        batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        input_features = batch.input_features
        
        sentences = [feature['sentence'] for feature in features]
        
        labels = self.processor.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448
        ).input_ids
        
        batch_size = labels.shape[0]
        
        # Add Whisper prefix tokens
        prefix_tokens = [
            self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|en|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            self.processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        ]
        
        prefix = torch.tensor([prefix_tokens] * batch_size, dtype=labels.dtype, device=labels.device)
        labels_with_prefix = torch.cat([prefix, labels], dim=1)
        
        # Create decoder inputs by shifting labels
        decoder_input_ids = labels_with_prefix.new_zeros(labels_with_prefix.shape)
        decoder_input_ids[:, 1:] = labels_with_prefix[:, :-1].clone()
        decoder_input_ids[:, 0] = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        
        decoder_input_ids = decoder_input_ids.masked_fill(
            decoder_input_ids == -100,
            self.processor.tokenizer.pad_token_id
        )
        
        labels_with_prefix = labels_with_prefix.masked_fill(
            labels_with_prefix == self.processor.tokenizer.pad_token_id, 
            -100
        )
        
        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels_with_prefix
        }


@dataclass
class WhisperInferenceCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_samples = 30 * 16000
        audio_arrays = []
        for feature in features:
            arr = feature['audio']['array']
            if len(arr) > max_samples:
                arr = arr[:max_samples]
            elif len(arr) < max_samples:
                arr = np.pad(arr, (0, max_samples - len(arr)), mode='constant')
            audio_arrays.append(arr)
        
        batch = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        return {"input_features": batch.input_features}
