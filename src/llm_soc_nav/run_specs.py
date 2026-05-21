"""Named run-spec loading and expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_soc_nav.prompt_generation import (
    CANONICAL_PROMPT,
    RANDOM_WALK_PROMPT,
    nav_prompt_name,
    prompt_conditions,
    random_walk_prompt_name,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    prompt_prefix: str = ""
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunSpec:
    name: str
    prompt: str
    llm_instruct: str
    search_instruct: str
    models: list[ModelSpec]
    output_label: str


def model_spec(raw: str | dict[str, Any]) -> ModelSpec:
    if isinstance(raw, str):
        return ModelSpec(name=raw, label=raw.replace(":", "-"))
    return ModelSpec(
        name=raw["name"],
        label=raw.get("label", raw["name"].replace(":", "-")),
        prompt_prefix=raw.get("prompt_prefix", ""),
        options=raw.get("options"),
    )


def _models_for_spec(cfg: dict[str, Any], raw: dict[str, Any]) -> list[ModelSpec]:
    models = raw.get("models", cfg["model_groups"][raw["model_group"]])
    return [model_spec(model) for model in models]


def model_group_alias(model_group: str) -> str:
    return model_group.removesuffix("_llm")


def search_alias(search_instruct: str) -> str:
    return (
        search_instruct.removeprefix("search-")
        .replace("sr-low-gamma", "sr_low")
        .replace("sr-medium-gamma", "sr_medium")
        .replace("sr-high-gamma", "sr_high")
        .replace("-", "_")
    )


def build_matrix_specs(cfg: dict[str, Any], task_name: str, matrix: dict[str, Any]) -> dict[str, RunSpec]:
    specs: dict[str, RunSpec] = {}
    conditions_by_id = {condition["id"]: condition for condition in prompt_conditions(cfg)}

    condition_ids = matrix.get("prompt_conditions", list(conditions_by_id))
    for model_group in matrix["model_groups"]:
        models = [model_spec(model) for model in cfg["model_groups"][model_group]]
        for condition_id in condition_ids:
            condition = conditions_by_id[condition_id]
            prompt = (
                random_walk_prompt_name(condition)
                if matrix["llm_instruct"] == "llm-next-node"
                else nav_prompt_name(condition)
            )
            for search_instruct in matrix.get("search_instructs", ["search-base"]):
                parts = [task_name, model_group_alias(model_group), condition_id]
                if search_instruct not in {"search-base", "search-random-walk"}:
                    parts.append(search_alias(search_instruct))
                name = "_".join(parts)
                specs[name] = RunSpec(
                    name=name,
                    prompt=prompt,
                    llm_instruct=matrix["llm_instruct"],
                    search_instruct=search_instruct,
                    models=models,
                    output_label=matrix.get("output_label", matrix["llm_instruct"]),
                )

    return specs


def matrix_specs(cfg: dict[str, Any]) -> dict[str, RunSpec]:
    specs: dict[str, RunSpec] = {}
    for task_name, matrix in cfg.get("run_matrices", {}).items():
        specs.update(build_matrix_specs(cfg, task_name, matrix))

    return specs


def filtered_matrix(cfg: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    matrix = dict(cfg["run_matrices"][item["matrix"]])
    if "model_group" in item:
        matrix["model_groups"] = [item["model_group"]]
    if "prompt_condition" in item:
        matrix["prompt_conditions"] = [item["prompt_condition"]]
    if "prompt_conditions" in item:
        matrix["prompt_conditions"] = list(item["prompt_conditions"])
    return matrix


def load_run_spec(cfg: dict[str, Any], name: str, model_override: list[str] | None = None) -> RunSpec:
    if name in cfg.get("run_specs", {}):
        raw = cfg["run_specs"][name]
        spec = RunSpec(
            name=name,
            prompt=raw.get("prompt", RANDOM_WALK_PROMPT if raw["llm_instruct"] == "llm-next-node" else CANONICAL_PROMPT),
            llm_instruct=raw["llm_instruct"],
            search_instruct=raw.get("search_instruct", "search-base"),
            models=_models_for_spec(cfg, raw),
            output_label=raw.get("output_label", raw["llm_instruct"]),
        )
    else:
        spec = matrix_specs(cfg)[name]

    if model_override:
        return RunSpec(
            name=spec.name,
            prompt=spec.prompt,
            llm_instruct=spec.llm_instruct,
            search_instruct=spec.search_instruct,
            models=[model_spec(model) for model in model_override],
            output_label=spec.output_label,
        )
    return spec


def group_spec_names(cfg: dict[str, Any], group: str) -> list[str]:
    names: list[str] = []
    for item in cfg["run_groups"][group]:
        if isinstance(item, dict):
            matrix = item["matrix"]
            names.extend(sorted(build_matrix_specs(cfg, matrix, filtered_matrix(cfg, item))))
            continue
        if item in cfg.get("run_matrices", {}):
            names.extend(sorted(build_matrix_specs(cfg, item, cfg["run_matrices"][item])))
        else:
            names.append(item)
    return names


def expand_specs(
    cfg: dict[str, Any],
    spec: str | None = None,
    group: str | None = None,
    model_override: list[str] | None = None,
) -> list[RunSpec]:
    if bool(spec) == bool(group):
        raise ValueError("Choose exactly one of --spec or --group.")

    names = [spec] if spec else group_spec_names(cfg, group)
    return [load_run_spec(cfg, name, model_override=model_override) for name in names]


def list_run_specs(cfg: dict[str, Any]) -> str:
    specs = matrix_specs(cfg)
    for name in cfg.get("run_specs", {}):
        specs[name] = load_run_spec(cfg, name)

    sorted_names = sorted(specs)
    name_w = max((len(n) for n in sorted_names), default=4)
    llm_w = max((len(specs[n].llm_instruct) for n in sorted_names), default=3)
    search_w = max((len(specs[n].search_instruct) for n in sorted_names), default=6)

    header = f"  {'NAME':<{name_w}}  {'LLM':<{llm_w}}  {'SEARCH':<{search_w}}  MODELS"
    sep = "  " + "-" * (name_w + llm_w + search_w + 14)
    lines = ["Run specs:", header, sep]
    for name in sorted_names:
        spec = specs[name]
        lines.append(
            f"  {name:<{name_w}}  {spec.llm_instruct:<{llm_w}}  {spec.search_instruct:<{search_w}}  {len(spec.models)}"
        )

    lines.append("\nRun groups:")
    for group_name in cfg["run_groups"]:
        group_specs = group_spec_names(cfg, group_name)
        lines.append(f"  {group_name} ({len(group_specs)}):")
        for spec_name in group_specs:
            lines.append(f"    {spec_name}")
    return "\n".join(lines)
