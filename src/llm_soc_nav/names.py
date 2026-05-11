"""Name loading and filtering helpers."""

from __future__ import annotations

import random
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_unique_baby_names(path: str | Path) -> list[str]:
    df = pd.read_csv(path)
    return (
        df["Child's First Name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .drop_duplicates()
        .tolist()
    )


def filter_similar_names(names: Iterable[str], threshold: float = 0.80) -> list[str]:
    kept: list[str] = []
    for name in names:
        if not any(SequenceMatcher(None, name, old).ratio() > threshold for old in kept):
            kept.append(name)
    return kept


def filter_one_token_names(names: Iterable[str], max_chars: int = 4) -> list[str]:
    """Simple tokenizer-free one-token proxy for cluster-friendly prompt generation."""
    return [name.strip() for name in names if 1 < len(name.strip()) <= max_chars]


def generate_random_names(count: int, length: int = 4, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    chars = string.ascii_lowercase
    names: set[str] = set()
    while len(names) < count:
        names.add("".join(rng.choices(chars, k=length)))
    return sorted(names)


def select_names(
    baby_names_path: str | Path,
    source: str = "baby_names",
    random_count: int = 1000,
    seed: int = 42,
) -> list[str]:
    if source == "random_strings":
        return generate_random_names(random_count, seed=seed)
    if source != "baby_names":
        raise ValueError("name_source must be 'baby_names' or 'random_strings'.")

    names = load_unique_baby_names(baby_names_path)
    return filter_similar_names(filter_one_token_names(names))
