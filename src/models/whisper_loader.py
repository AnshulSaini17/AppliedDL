from typing import Optional, Dict
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..utils.utils import setup_logger

logger = setup_logger(__name__)


def load_whisper_model(
    model_name: str = "openai/whisper-small",
    language: str = "en",
    task: str = "transcribe",
    use_lora: bool = False,
    lora_config: Optional[Dict] = None,
    device: str = "cuda",
    for_training: bool = False
) -> tuple:
    logger.info(f"Loading Whisper model: {model_name}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    
    if for_training:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    
    model.generation_config.language = language
    model.generation_config.task = task
    
    forced_decoder_ids = [
        (0, processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")),
        (1, processor.tokenizer.convert_tokens_to_ids("<|en|>")),
        (2, processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")),
        (3, processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>"))
    ]
    
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    
    if for_training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    
    # Apply LoRA if requested
    if use_lora:
        logger.info("Applying LoRA adapters")
        
        # Freeze feature encoder
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        
        # Configure LoRA
        lora_cfg = lora_config or {}
        peft_config = LoraConfig(
            r=lora_cfg.get('r', 16),
            lora_alpha=lora_cfg.get('alpha', 32),
            lora_dropout=lora_cfg.get('dropout', 0.1),
            target_modules=lora_cfg.get('target_modules', ['q_proj', 'v_proj']),
            bias="none",
            task_type="SEQ_2_SEQ_LM"  
        )
        
        # Prepare model for quantized training (if needed)
        if for_training:
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        logger.info("Using full fine-tuning (no LoRA)")
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        model = model.to("cpu")
    
    return model, processor


def load_whisper_for_inference(
    model_path: str,
    language: str = "en",
    task: str = "transcribe",
    device: str = "cuda"
) -> tuple:
    
    logger.info(f"Loading model from checkpoint: {model_path}")
    
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    
    if device == "cuda" and torch.cuda.is_available():
        model = model.to(device)
    else:
        model = model.to("cpu")
    
    model.eval()
    
    return model, processor


def freeze_encoder(model):
    """Freeze the encoder (feature extraction) of Whisper model."""
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    logger.info("Froze encoder parameters")