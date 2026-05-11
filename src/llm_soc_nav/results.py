"""Result filenames, saving, and summaries."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from llm_soc_nav.config import resolve_path


def sanitize_label(value: str) -> str:
    value = value.replace(":", "-")
    value = re.sub(r"[^A-Za-z0-9._+-]+", "-", value)
    return value.strip("-").lower()


def result_filename(model: str, task: str, prompt: str, search: str) -> str:
    return (
        f"model-{sanitize_label(model)}_"
        f"task-{sanitize_label(task)}_"
        f"prompt-{sanitize_label(prompt)}_"
        f"{sanitize_label(search)}_responses.csv"
    )


def result_path(cfg: dict, model: str, task: str, prompt: str, search: str) -> Path:
    out_dir = resolve_path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / result_filename(model, task, prompt, search)


def accuracy(results: pd.DataFrame) -> float:
    valid = results.loc[results["is_valid_choice"], ["llm_choice", "correct_choice"]]
    if valid.empty:
        return float("nan")
    return float((valid["llm_choice"] == valid["correct_choice"]).mean())
