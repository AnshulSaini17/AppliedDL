"""Text normalization utilities."""

import re
import unicodedata


def normalize_text(text: str, mode: str = "basic") -> str:
    """
    Normalize text for ASR evaluation.
    
    Args:
        text: Input text
        mode: Normalization mode ('basic' or 'aggressive')
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Convert to lowercase
    text = text.lower()
    
    if mode == "aggressive":
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
    else:
        # Basic: keep some punctuation, just normalize
        text = re.sub(r'[""''′″`]', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_for_metrics(text: str) -> str:
    """
    Normalize text specifically for WER/CER computation.
    Uses consistent normalization for both reference and hypothesis.
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Lowercase and unicode normalize
    text = unicodedata.normalize("NFKC", text.lower())
    
    # Remove punctuation except apostrophes in contractions
    text = re.sub(r"[^\w\s']", "", text)
    
    # Collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

