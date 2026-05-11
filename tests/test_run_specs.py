import pytest

from llm_soc_nav.run_specs import expand_specs, load_run_spec


CFG = {
    "model_groups": {
        "instruct_llm": [
            "gemma3:4b",
            {
                "name": "qwen3:8b",
                "label": "qwen3-8b-no-think",
                "prompt_prefix": "/no_think\n\n",
                "options": {"temperature": 0},
            },
        ]
    },
    "run_specs": {
        "classifier_instruct": {
            "model_group": "instruct_llm",
            "llm_instruct": "llm-classifier",
            "search_instruct": "search-base",
        },
        "next_node_instruct": {
            "model_group": "instruct_llm",
            "llm_instruct": "llm-next-node",
            "search_instruct": "search-random-walk",
        }
    },
    "run_groups": {"classifier_runs": ["classifier_instruct"]},
}


def test_load_run_spec_uses_model_group():
    spec = load_run_spec(CFG, "classifier_instruct")
    assert [model.label for model in spec.models] == ["gemma3-4b", "qwen3-8b-no-think"]
    assert spec.models[1].prompt_prefix == "/no_think\n\n"
    assert spec.models[1].options == {"temperature": 0}
    assert spec.prompt == "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled"


def test_expand_group():
    specs = expand_specs(CFG, group="classifier_runs")
    assert [spec.name for spec in specs] == ["classifier_instruct"]


def test_model_override_keeps_spec_otherwise_same():
    spec = expand_specs(CFG, spec="classifier_instruct", model_override=["gemma3:4b"])[0]
    assert [model.name for model in spec.models] == ["gemma3:4b"]


def test_unknown_spec_is_rejected():
    with pytest.raises(KeyError):
        load_run_spec(CFG, "missing")


def test_next_node_spec_uses_random_walk_prompt():
    spec = load_run_spec(CFG, "next_node_instruct")
    assert spec.prompt == "random-walk-next-node"
    assert spec.llm_instruct == "llm-next-node"
