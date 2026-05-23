"""Generate deterministic social-navigation prompt CSVs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_soc_nav.config import resolve_path
from llm_soc_nav.graph import SOCIAL_GRAPH, graph_sentences
from llm_soc_nav.names import names_paths, select_names
from llm_soc_nav.prompt_templates import classifier_question, path_question, random_walk_question

NAV_PROMPT_BASE = "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled"
RANDOM_WALK_PROMPT_BASE = "random-walk-next-node"
CANONICAL_PROMPT = f"{NAV_PROMPT_BASE}_graph-social_names-baby"
RANDOM_WALK_PROMPT = f"{RANDOM_WALK_PROMPT_BASE}_graph-social_names-baby"

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
    graph_context: str = "social"


def prompt_filename(prompt_name: str = CANONICAL_PROMPT) -> str:
    return f"questions_{prompt_name}.csv"


def default_prompt_conditions() -> list[dict[str, str]]:
    return [
        {
            "id": "graph-social_names-baby",
            "graph_context": "social",
            "name_source": "baby_names",
        },
        {
            "id": "graph-social_names-random",
            "graph_context": "social",
            "name_source": "random_strings",
        },
        {
            "id": "graph-generic_names-baby",
            "graph_context": "generic",
            "name_source": "baby_names",
        },
        {
            "id": "graph-generic_names-random",
            "graph_context": "generic",
            "name_source": "random_strings",
        },
        {
            "id": "graph-generic-adj-list_names-baby",
            "graph_context": "generic_adj_list",
            "name_source": "baby_names",
        },
        {
            "id": "graph-generic-adj-list_names-random",
            "graph_context": "generic_adj_list",
            "name_source": "random_strings",
        },
    ]


def prompt_conditions(cfg: dict[str, Any]) -> list[dict[str, str]]:
    return list(cfg.get("prompt_conditions", default_prompt_conditions()))


def condition_label(condition: dict[str, str]) -> str:
    graph_context = condition["graph_context"].replace("_", "-")
    graph_label = f"graph-{graph_context}"
    name_label = "names-baby" if condition["name_source"] == "baby_names" else "names-random"
    return f"{graph_label}_{name_label}"


def nav_prompt_name(condition: dict[str, str]) -> str:
    return f"{NAV_PROMPT_BASE}_{condition_label(condition)}"


def random_walk_prompt_name(condition: dict[str, str]) -> str:
    return f"{RANDOM_WALK_PROMPT_BASE}_{condition_label(condition)}"


def generate_adjacency_sets(
    names: list[str],
    settings: PromptSettings,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    rng = np.random.default_rng(seed)
    sets: list[tuple[list[str], list[str]]] = []

    for _ in range(settings.n_prompt_sets):
        selected = list(rng.choice(names, size=13, replace=False))
        sentences = graph_sentences(selected, settings.graph_context)
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
                    "question": " ".join(
                        [base_prompt, classifier_question(start, end, options[0], options[1], settings.graph_context)]
                    ),
                    "path_question": " ".join(
                        [base_prompt, path_question(start, end, options[0], options[1], settings.graph_context)]
                    ),
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


RANDOM_WALK_COLUMNS = [
    "random_walk_question",
    "graph_context",
    "name_source",
    "path_so_far",
    "prefix_length",
    "start_node",
    "current_node",
    "sampled_next_node",
    "valid_next_nodes",
    "all_nodes",
    "walk_id",
    "prompt_id",
]


def sample_walk(names: list[str], current_id: int, prefix_length: int, rng: np.random.Generator) -> list[str]:
    node_id = current_id
    path_ids = [node_id]
    for _ in range(prefix_length):
        node_id = int(rng.choice(SOCIAL_GRAPH[node_id]))
        path_ids.append(node_id)
    path_ids.reverse()
    return [names[i] for i in path_ids]


def generate_random_walk_questions(
    adjacency_sets: list[tuple[list[str], list[str]]],
    prefix_lengths: list[int],
    seed: int,
    graph_context: str,
    name_source: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    walk_id = 0

    for prompt_id, (sentences, names) in enumerate(adjacency_sets):
        all_nodes = list(names)
        graph_text = " ".join(sentences)
        for current_id in range(len(SOCIAL_GRAPH)):
            for prefix_length in prefix_lengths:
                path = sample_walk(names, current_id, prefix_length, rng)
                valid_next_nodes = [names[i] for i in SOCIAL_GRAPH[current_id]]
                sampled_next_node = str(rng.choice(valid_next_nodes))
                rows.append(
                    {
                        "random_walk_question": random_walk_question(graph_text, path),
                        "graph_context": graph_context,
                        "name_source": name_source,
                        "path_so_far": " -> ".join(path),
                        "prefix_length": prefix_length,
                        "start_node": path[0],
                        "current_node": path[-1],
                        "sampled_next_node": sampled_next_node,
                        "valid_next_nodes": "|".join(valid_next_nodes),
                        "all_nodes": "|".join(all_nodes),
                        "walk_id": walk_id,
                        "prompt_id": prompt_id,
                    }
                )
                walk_id += 1

    df = pd.DataFrame(rows, columns=RANDOM_WALK_COLUMNS)
    for pid, group in df.groupby("prompt_id"):
        counts = group["current_node"].value_counts()
        assert counts.nunique() == 1, (
            f"current_node counts are not balanced in prompt_id={pid}: {counts.to_dict()}"
        )
    return df


def generate_prompts(
    cfg: dict[str, Any],
    condition: dict[str, str],
    output_dir: str | Path | None = None,
) -> Path:
    seed = int(cfg.get("seed", 42))
    prompt_cfg = cfg.get("prompt_generation", {})
    settings = PromptSettings(
        n_prompt_sets=int(prompt_cfg.get("n_prompt_sets", 100)),
        name_source=condition["name_source"],
        random_name_count=int(prompt_cfg.get("random_name_count", 1000)),
        graph_context=condition["graph_context"],
    )
    raw_paths = cfg["paths"]["raw"]
    prompts_dir = resolve_path(output_dir or cfg["paths"]["prompts_dir"])

    baby_path, random_path = names_paths(cfg)
    names_path = baby_path if settings.name_source == "baby_names" else random_path
    if not names_path.exists():
        raise FileNotFoundError(
            f"Name CSV not found: {names_path}. Run 'save-names' first."
        )
    names = pd.read_csv(names_path)["name"].tolist()
    tasks = pd.read_csv(resolve_path(raw_paths["tasks"]))

    adjacency_sets = generate_adjacency_sets(names, settings, seed)
    questions = generate_questions(adjacency_sets, tasks, settings, seed)

    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / prompt_filename(nav_prompt_name(condition))
    questions.to_csv(out_path, index=False)
    return out_path


def generate_random_walk_prompt(
    cfg: dict[str, Any],
    condition: dict[str, str],
    output_dir: str | Path | None = None,
) -> Path:
    seed = int(cfg.get("seed", 42))
    prompt_cfg = cfg.get("prompt_generation", {})
    walk_cfg = cfg.get("random_walk_generation", {})
    settings = PromptSettings(
        n_prompt_sets=int(prompt_cfg.get("n_prompt_sets", 100)),
        name_source=condition["name_source"],
        random_name_count=int(prompt_cfg.get("random_name_count", 1000)),
        graph_context=condition["graph_context"],
    )
    raw_paths = cfg["paths"]["raw"]
    prompts_dir = resolve_path(output_dir or cfg["paths"]["prompts_dir"])

    baby_path, random_path = names_paths(cfg)
    names_path = baby_path if settings.name_source == "baby_names" else random_path
    if not names_path.exists():
        raise FileNotFoundError(
            f"Name CSV not found: {names_path}. Run 'save-names' first."
        )
    names = pd.read_csv(names_path)["name"].tolist()
    adjacency_sets = generate_adjacency_sets(names, settings, seed)
    prefix_lengths = [int(length) for length in walk_cfg.get("prefix_lengths", [0, 1, 2, 3, 5, 8])]
    questions = generate_random_walk_questions(
        adjacency_sets,
        prefix_lengths,
        seed,
        graph_context=settings.graph_context,
        name_source=settings.name_source,
    )

    prompts_dir.mkdir(parents=True, exist_ok=True)
    out_path = prompts_dir / prompt_filename(random_walk_prompt_name(condition))
    questions.to_csv(out_path, index=False)
    return out_path


def generate_all_prompts(cfg: dict[str, Any]) -> list[Path]:
    paths = [generate_prompts(cfg, condition) for condition in prompt_conditions(cfg)]
    paths.extend(generate_random_walk_prompt(cfg, condition) for condition in prompt_conditions(cfg))
    return paths
