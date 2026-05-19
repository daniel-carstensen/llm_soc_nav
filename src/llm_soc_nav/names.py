"""Name loading and filtering helpers."""

from __future__ import annotations

import random
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


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


def token_ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def is_one_token_for_all(name: str, model_tokenizers: dict[str, TokenizerLike]) -> bool:
    return all(len(token_ids(tokenizer, name)) == 1 for tokenizer in model_tokenizers.values())


def filter_one_token_names(
    names: Iterable[str],
    max_chars: int = 12,
    model_tokenizers: dict[str, TokenizerLike] | None = None,
) -> list[str]:
    """Simple tokenizer-free one-token proxy for cluster-friendly prompt generation."""
    candidates = [name.strip() for name in names if 1 < len(name.strip()) <= max_chars]
    if not model_tokenizers:
        return candidates
    return [name for name in candidates if is_one_token_for_all(name, model_tokenizers)]


def assert_one_token_names(names: Iterable[str], model_tokenizers: dict[str, TokenizerLike]) -> None:
    failures: list[str] = []
    for name in names:
        bad_models = []
        for model, tokenizer in model_tokenizers.items():
            if len(token_ids(tokenizer, name)) != 1:
                bad_models.append(model)
                break
        if bad_models:
            failures.append(f"{name}: {', '.join(bad_models)}")

    if failures:
        preview = "\n".join(failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n... {len(failures) - 20} more"
        raise ValueError(f"Names are not one token for all configured model tokenizers:\n{preview}{suffix}")


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
    model_tokenizers: dict[str, TokenizerLike] | None = None,
    random_candidate_multiplier: int = 1000,
) -> list[str]:
    if source == "random_strings":
        candidate_count = random_count if not model_tokenizers else random_count * random_candidate_multiplier
        names = filter_one_token_names(
            generate_random_names(candidate_count, seed=seed),
            model_tokenizers=model_tokenizers,
        )
        baby_name_set = set(load_unique_baby_names(baby_names_path))
        names = [n for n in names if n not in baby_name_set]
        if len(names) < random_count:
            raise ValueError(
                f"Only {len(names)} random names were one token for all configured tokenizers; "
                f"needed {random_count}. Increase random_candidate_multiplier."
            )
        return names[:random_count]
    if source != "baby_names":
        raise ValueError("name_source must be 'baby_names' or 'random_strings'.")

    names = load_unique_baby_names(baby_names_path)
    return filter_one_token_names(names, model_tokenizers=model_tokenizers)


def names_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    from llm_soc_nav.config import resolve_path

    names_dir = resolve_path(cfg["paths"]["names_dir"])
    return names_dir / "baby_names.csv", names_dir / "random_strings.csv"


def configured_model_names(cfg: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for models in cfg.get("model_groups", {}).values():
        for model in models:
            name = model if isinstance(model, str) else model["name"]
            if name not in names:
                names.append(name)
    return names


def tokenizer_refs_for_config(cfg: dict[str, Any]) -> dict[str, str]:
    check_cfg = cfg.get("name_token_check", {})
    exact = check_cfg.get("model_tokenizers", {})
    families = check_cfg.get("family_tokenizers", {})
    refs: dict[str, str] = {}
    missing: list[str] = []

    for model_name in configured_model_names(cfg):
        if model_name in exact:
            refs[model_name] = exact[model_name]
            continue

        family = model_name.split(":", 1)[0]
        if family in families:
            refs[model_name] = families[family]
            continue

        missing.append(model_name)

    if missing:
        raise ValueError(
            "Missing tokenizer refs for configured models: "
            + ", ".join(missing)
            + ". Add exact model_tokenizers or family_tokenizers under name_token_check."
        )
    return refs


def load_model_tokenizers(cfg: dict[str, Any]) -> dict[str, TokenizerLike]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers and sentencepiece to run tokenizer-based name checks.") from exc

    check_cfg = cfg.get("name_token_check", {})
    local_files_only = bool(check_cfg.get("local_files_only", False))
    tokenizers: dict[str, TokenizerLike] = {}
    for model_name, tokenizer_ref in tokenizer_refs_for_config(cfg).items():
        try:
            extra_kwargs: dict[str, Any] = {}
            if "mistral" in tokenizer_ref.lower():
                extra_kwargs["fix_mistral_regex"] = True
            tokenizers[model_name] = AutoTokenizer.from_pretrained(
                tokenizer_ref,
                local_files_only=local_files_only,
                trust_remote_code=True,
                **extra_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load tokenizer '{tokenizer_ref}' for Ollama model '{model_name}'. "
                "If the tokenizer repo is gated, authenticate with Hugging Face or point the "
                "config to a local tokenizer directory."
            ) from exc
    return tokenizers


def save_names(cfg: dict[str, Any]) -> dict[str, int]:
    from llm_soc_nav.config import resolve_path

    model_tokenizers = load_model_tokenizers(cfg)
    prompt_cfg = cfg.get("prompt_generation", {})
    check_cfg = cfg.get("name_token_check", {})
    raw_paths = cfg["paths"]["raw"]
    baby_names_path = resolve_path(raw_paths["baby_names"])
    random_count = int(prompt_cfg.get("random_name_count", 1000))
    seed = int(cfg.get("seed", 42))
    random_candidate_multiplier = int(check_cfg.get("random_candidate_multiplier", 1000))

    print(f"Selecting baby names from {baby_names_path}...")
    baby_names = select_names(
        baby_names_path,
        source="baby_names",
        seed=seed,
        model_tokenizers=model_tokenizers,
        random_candidate_multiplier=random_candidate_multiplier,
    )
    print(f"Selecting random string names...")
    random_names = select_names(
        baby_names_path,
        source="random_strings",
        random_count=random_count,
        seed=seed,
        model_tokenizers=model_tokenizers,
        random_candidate_multiplier=random_candidate_multiplier,
    )

    baby_path, random_path = names_paths(cfg)
    baby_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": baby_names}).to_csv(baby_path, index=False)
    pd.DataFrame({"name": random_names}).to_csv(random_path, index=False)
    return {
        "baby_names": len(baby_names),
        "random_strings": len(random_names),
        "models": len(model_tokenizers),
    }


def check_config_name_tokens(cfg: dict[str, Any]) -> dict[str, int]:
    baby_path, random_path = names_paths(cfg)
    missing = [p for p in (baby_path, random_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Name CSV(s) not found: {', '.join(str(p) for p in missing)}. "
            "Run 'save-names' first to generate them."
        )

    model_tokenizers = load_model_tokenizers(cfg)
    baby_names = pd.read_csv(baby_path)["name"].tolist()
    random_names = pd.read_csv(random_path)["name"].tolist()

    assert_one_token_names(baby_names, model_tokenizers)
    assert_one_token_names(random_names, model_tokenizers)
    return {
        "baby_names": len(baby_names),
        "random_strings": len(random_names),
        "models": len(model_tokenizers),
    }
