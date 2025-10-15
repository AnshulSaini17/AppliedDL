"""Accent canonicalization and taxonomy utilities."""

import yaml
from pathlib import Path
from typing import Optional, Dict, List
from collections import Counter


def load_accent_mappings(config_path: str = "configs/accents.yaml") -> Dict[str, List[str]]:
    """
    Load accent mapping configuration.
    
    Args:
        config_path: Path to accents config YAML
    
    Returns:
        Dictionary mapping canonical accent -> list of variants
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # Return default mappings if config not found
        return get_default_mappings()
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('mappings', get_default_mappings())


def get_default_mappings() -> Dict[str, List[str]]:
    """Return default accent mappings."""
    return {
        "american": ["american", "united states", "us", "usa", "en-us", "america"],
        "british": ["british", "england", "uk", "united kingdom", "en-gb", "gb", "scottish", "scotland", "wales", "welsh"],
        "indian": ["indian", "india", "en-in"],
        "african": ["african", "africa", "south african", "south africa", "nigerian", "nigeria", "kenyan", "kenya"],
        "australian": ["australian", "australia", "en-au"],
        "irish": ["irish", "ireland"],
        "canadian": ["canadian", "canada", "en-ca"]
    }


def canonicalize_accent(raw_accent: str, mappings: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    """
    Canonicalize a raw accent string to a standard label.
    
    Args:
        raw_accent: Raw accent string from dataset
        mappings: Accent mapping dictionary (canonical -> variants)
    
    Returns:
        Canonical accent name or None if no match
    """
    if not raw_accent or not isinstance(raw_accent, str):
        return None
    
    if mappings is None:
        mappings = get_default_mappings()
    
    # Normalize input
    normalized = raw_accent.lower().strip()
    
    # Check each canonical accent's variants
    for canonical, variants in mappings.items():
        for variant in variants:
            if variant.lower() in normalized or normalized in variant.lower():
                return canonical
    
    return None


def get_top_k_accents(examples: List[Dict], k: int = 10, mappings: Optional[Dict[str, List[str]]] = None) -> List[str]:
    """
    Find the top-k most common accents in a dataset after canonicalization.
    
    Args:
        examples: List of dataset examples with 'accent' field
        k: Number of top accents to return
        mappings: Accent mapping dictionary
    
    Returns:
        List of top k canonical accent names
    """
    if mappings is None:
        mappings = get_default_mappings()
    
    # Canonicalize all accents
    canonical_accents = []
    for ex in examples:
        raw_accent = ex.get('accent', '')
        canonical = canonicalize_accent(raw_accent, mappings)
        if canonical:
            canonical_accents.append(canonical)
    
    # Count and return top k
    counter = Counter(canonical_accents)
    return [accent for accent, _ in counter.most_common(k)]


def filter_by_accent(examples: List[Dict], target_accents: List[str], mappings: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    """
    Filter dataset examples by target accents.
    
    Args:
        examples: List of dataset examples
        target_accents: List of canonical accent names to keep
        mappings: Accent mapping dictionary
    
    Returns:
        Filtered list of examples with canonicalized accent field
    """
    if mappings is None:
        mappings = get_default_mappings()
    
    filtered = []
    for ex in examples:
        raw_accent = ex.get('accent', '')
        canonical = canonicalize_accent(raw_accent, mappings)
        
        if canonical and canonical in target_accents:
            ex['accent'] = canonical  # Replace with canonical
            filtered.append(ex)
    
    return filtered

