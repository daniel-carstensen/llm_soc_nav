"""Named run-spec loading and expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_soc_nav.prompt_generation import CANONICAL_PROMPT


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


def load_run_spec(cfg: dict[str, Any], name: str, model_override: list[str] | None = None) -> RunSpec:
    raw = cfg["run_specs"][name]
    models = [model_spec(model) for model in model_override] if model_override else _models_for_spec(cfg, raw)
    return RunSpec(
        name=name,
        prompt=CANONICAL_PROMPT,
        llm_instruct=raw["llm_instruct"],
        search_instruct=raw.get("search_instruct", "search-base"),
        models=list(models),
        output_label=raw.get("output_label", name),
    )


def expand_specs(
    cfg: dict[str, Any],
    spec: str | None = None,
    group: str | None = None,
    model_override: list[str] | None = None,
) -> list[RunSpec]:
    if bool(spec) == bool(group):
        raise ValueError("Choose exactly one of --spec or --group.")

    names = [spec] if spec else list(cfg["run_groups"][group])
    return [load_run_spec(cfg, name, model_override=model_override) for name in names]


def list_run_specs(cfg: dict[str, Any]) -> str:
    lines = ["Run specs:"]
    for name, raw in cfg["run_specs"].items():
        model_source = raw.get("model_group", "inline-models")
        lines.append(
            f"  {name}: prompt={CANONICAL_PROMPT} llm={raw['llm_instruct']} "
            f"search={raw.get('search_instruct', 'search-base')} models={model_source}"
        )

    lines.append("\nRun groups:")
    for name, specs in cfg["run_groups"].items():
        lines.append(f"  {name}: {', '.join(specs)}")
    return "\n".join(lines)
