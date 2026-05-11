"""Generate deterministic social-navigation prompt CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_soc_nav.config import resolve_path
from llm_soc_nav.graph import friendship_sentences
from llm_soc_nav.names import select_names
from llm_soc_nav.prompt_templates import classifier_question, path_question

CANONICAL_PROMPT = "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled"

PROMPT_COLUMNS = [
    "question",
    "path_question",
    "startpoint",
    "endpoint",
    "opt1",
    "opt2",
    "correct_choice",
    "question_id",
    "prompt_id",
]


@dataclass(frozen=True)
class PromptSettings:
    n_prompt_sets: int = 100
    name_source: str = "baby_names"
    random_name_count: int = 1000


def prompt_filename(prompt_name: str = CANONICAL_PROMPT) -> str:
    return f"questions_{prompt_name}.csv"


def generate_adjacency_sets(
    names: list[str],
    settings: PromptSettings,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    rng = np.random.default_rng(seed)
    sets: list[tuple[list[str], list[str]]] = []

    for _ in range(settings.n_prompt_sets):
        selected = list(rng.choice(names, size=13, replace=False))
        sentences = friendship_sentences(selected)
        rng.shuffle(sentences)
        sets.append((sentences, selected))

    return sets


def generate_questions(
    adjacency_sets: list[tuple[list[str], list[str]]],
    tasks: pd.DataFrame,
    settings: PromptSettings,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for prompt_id, (sentences, names) in enumerate(adjacency_sets):
        for question_id, task in tasks.iterrows():
            prompt_sentences = list(sentences)
            rng.shuffle(prompt_sentences)

            start = names[int(task["startpoint_id"]) - 1]
            end = names[int(task["endpoint_id"]) - 1]
            options = [
                names[int(task["opt1_id"]) - 1],
                names[int(task["opt2_id"]) - 1],
            ]
            rng.shuffle(options)

            base_prompt = " ".join(prompt_sentences)
            rows.append(
                {
                    "question": " ".join([base_prompt, classifier_question(start, end, options[0], options[1])]),
                    "path_question": " ".join([base_prompt, path_question(start, end, options[0], options[1])]),
                    "startpoint": start,
                    "endpoint": end,
                    "opt1": options[0],
                    "opt2": options[1],
                    "correct_choice": names[int(task["correct_choice"]) - 1],
                    "question_id": question_id,
                    "prompt_id": prompt_id,
                }
            )

    return pd.DataFrame(rows, columns=PROMPT_COLUMNS)


def generate_prompts(
    cfg: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    seed = int(cfg.get("seed", 42))
    prompt_cfg = cfg.get("prompt_generation", {})
    settings = PromptSettings(
        n_prompt_sets=int(prompt_cfg.get("n_prompt_sets", 100)),
        name_source=prompt_cfg.get("name_source", "baby_names"),
        random_name_count=int(prompt_cfg.get("random_name_count", 1000)),
    )
    raw_paths = cfg["paths"]["raw"]
    prompts_dir = resolve_path(output_dir or cfg["paths"]["prompts_dir"])

    names = select_names(
        resolve_path(raw_paths["baby_names"]),
        source=settings.name_source,
        random_count=settings.random_name_count,
        seed=seed,
    )
    tasks = pd.read_csv(resolve_path(raw_paths["tasks"]))

    adjacency_sets = generate_adjacency_sets(names, settings, seed)
    questions = generate_questions(adjacency_sets, tasks, settings, seed)

    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / prompt_filename()
    questions.to_csv(out_path, index=False)
    return out_path


def generate_all_prompts(cfg: dict[str, Any]) -> list[Path]:
    return [generate_prompts(cfg)]
