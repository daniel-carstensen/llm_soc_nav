"""Run named specs against Ollama."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from llm_soc_nav.config import resolve_path
from llm_soc_nav.parsing import parse_next_node_response, parse_response
from llm_soc_nav.prompt_generation import CANONICAL_PROMPT, RANDOM_WALK_PROMPT, prompt_filename
from llm_soc_nav.prompt_templates import LLM_INSTRUCTIONS, SEARCH_INSTRUCTIONS
from llm_soc_nav.results import accuracy, result_path
from llm_soc_nav.run_specs import ModelSpec, RunSpec

KEEP_COLUMNS = [
    "question_id",
    "prompt_id",
    "startpoint",
    "endpoint",
    "opt1",
    "opt2",
    "correct_choice",
]

NEXT_NODE_KEEP_COLUMNS = [
    "walk_id",
    "prompt_id",
    "prefix_length",
    "path_so_far",
    "start_node",
    "current_node",
    "sampled_next_node",
    "valid_next_nodes",
    "all_nodes",
]


class GenerateClient(Protocol):
    def generate(self, **kwargs: Any) -> Any:
        ...


def build_model_prompt(row: pd.Series, spec: RunSpec, model: ModelSpec | None = None) -> str:
    llm_instruction = LLM_INSTRUCTIONS[spec.llm_instruct]
    search_instruction = ""

    if spec.llm_instruct == "llm-path":
        question = str(row["path_question"])
        search_instruction = SEARCH_INSTRUCTIONS[spec.search_instruct]
    elif spec.llm_instruct == "llm-next-node":
        question = str(row["random_walk_question"])
        search_instruction = SEARCH_INSTRUCTIONS[spec.search_instruct]
    else:
        question = str(row["question"])
        question = f"{question}\n\noutput which option is correct: '{row['opt1']}' or '{row['opt2']}'."

    prompt = f"{llm_instruction}{question}\n\n{search_instruction}".strip("\n")
    if model and model.prompt_prefix:
        prompt = f"{model.prompt_prefix}{prompt}"
    return prompt


def build_generation_options(generation: dict[str, Any], model: ModelSpec) -> dict[str, Any]:
    options = dict(generation.get("options", {}))
    options["num_predict"] = generation["num_predict"]
    options.update(model.options or {})
    return options


def call_ollama(
    model: str,
    prompt: str,
    options: dict[str, Any],
    use_logprobs: bool,
    top_logprobs: int,
    client: GenerateClient | None = None,
):
    if client is None:
        import ollama

        client = ollama

    if use_logprobs:
        return client.generate(
            model=model,
            prompt=prompt,
            logprobs=True,
            top_logprobs=top_logprobs,
            options=options,
        )
    return client.generate(model=model, prompt=prompt, options=options)


def run_model(
    cfg: dict[str, Any],
    spec: RunSpec,
    model: ModelSpec,
    questions: pd.DataFrame,
    limit: int | None = None,
    client: GenerateClient | None = None,
) -> pd.DataFrame:
    generation = cfg["generation"][spec.llm_instruct]
    use_logprobs = bool(generation.get("logprobs", False))

    if limit is not None:
        questions = questions.iloc[:limit, :]

    records: list[dict[str, Any]] = []
    for i, row in questions.iterrows():
        prompt = build_model_prompt(row, spec, model)
        options = build_generation_options(generation, model)

        resp = call_ollama(model.name, prompt, options, use_logprobs, cfg["generation"]["top_logprobs"], client=client)
        if spec.llm_instruct == "llm-next-node":
            all_nodes = str(row["all_nodes"]).split("|")
            record = {col: row[col] for col in NEXT_NODE_KEEP_COLUMNS}
            record.update(parse_next_node_response(resp, all_nodes))
        else:
            record = {col: row[col] for col in KEEP_COLUMNS}
            record.update(parse_response(resp, row["opt1"], row["opt2"]))
        record["model"] = model.name
        record["model_label"] = model.label
        record["spec"] = spec.name
        records.append(record)
        print(f"  [{len(records)}/{len(questions)}] {record['raw_response']!r:>12}")

    results = pd.DataFrame(records)
    out_path = result_path(cfg, model.label, spec.output_label, spec.prompt, spec.search_instruct)
    results.to_csv(out_path, index=False)
    if spec.llm_instruct == "llm-next-node":
        print(f"  Saved {len(results)} rows -> {out_path}")
    else:
        print(f"  Saved {len(results)} rows -> {out_path} | accuracy={accuracy(results):.3f}")
    return results


def run_spec(cfg: dict[str, Any], spec: RunSpec, limit: int | None = None) -> None:
    prompt_name = spec.prompt or CANONICAL_PROMPT
    prompt_path = resolve_path(cfg["paths"]["prompts_dir"]) / prompt_filename(prompt_name)
    questions = pd.read_csv(prompt_path)

    for model in spec.models:
        print(f"\n=== {spec.name}: {model.label} ({len(questions)} questions) ===")
        run_model(cfg, spec, model, questions, limit=limit)
