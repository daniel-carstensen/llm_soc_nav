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
        ],
        "reasoning_llm": ["deepseek-r1:7b"],
    },
    "prompt_conditions": [
        {
            "id": "graph-social_names-baby",
            "graph_context": "social",
            "name_source": "baby_names",
        },
        {
            "id": "graph-generic_names-baby",
            "graph_context": "generic",
            "name_source": "baby_names",
        },
    ],
    "run_matrices": {
        "classifier": {
            "model_groups": ["instruct_llm"],
            "prompt_conditions": ["graph-social_names-baby", "graph-generic_names-baby"],
            "llm_instruct": "llm-classifier",
            "search_instructs": ["search-base"],
            "output_label": "llm-classifier",
        },
        "next_node": {
            "model_groups": ["instruct_llm"],
            "prompt_conditions": ["graph-social_names-baby", "graph-generic_names-baby"],
            "llm_instruct": "llm-next-node",
            "search_instructs": ["search-random-walk"],
            "output_label": "llm-next-node",
        },
        "classifier_all": {
            "model_groups": ["instruct_llm", "reasoning_llm"],
            "prompt_conditions": ["graph-social_names-baby", "graph-generic_names-baby"],
            "llm_instruct": "llm-classifier",
            "search_instructs": ["search-base"],
            "output_label": "llm-classifier",
        },
    },
    "run_specs": {
        "classifier_instruct": {
            "model_group": "instruct_llm",
            "llm_instruct": "llm-classifier",
            "search_instruct": "search-base",
        },
    },
    "run_groups": {
        "classifier_runs": ["classifier"],
        "next_node_runs": ["next_node"],
        "classifier_all_instruct_baby_runs": [
            {
                "matrix": "classifier_all",
                "model_group": "instruct_llm",
                "prompt_condition": "graph-social_names-baby",
            }
        ],
        "classifier_all_baby_runs": [
            {"matrix": "classifier_all", "prompt_condition": "graph-social_names-baby"}
        ],
    },
}


def test_load_run_spec_uses_model_group():
    spec = load_run_spec(CFG, "classifier_instruct")
    assert [model.label for model in spec.models] == ["gemma3-4b", "qwen3-8b-no-think"]
    assert spec.models[1].prompt_prefix == "/no_think\n\n"
    assert spec.models[1].options == {"temperature": 0}
    assert spec.prompt == "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled_graph-social_names-baby"


def test_expand_group():
    specs = expand_specs(CFG, group="classifier_runs")
    assert [spec.name for spec in specs] == [
        "classifier_instruct_graph-generic_names-baby",
        "classifier_instruct_graph-social_names-baby",
    ]


def test_group_can_filter_model_group_and_prompt_condition():
    specs = expand_specs(CFG, group="classifier_all_instruct_baby_runs")
    assert [spec.name for spec in specs] == ["classifier_all_instruct_graph-social_names-baby"]
    assert [model.name for model in specs[0].models] == ["gemma3:4b", "qwen3:8b"]


def test_group_can_filter_prompt_condition_only():
    specs = expand_specs(CFG, group="classifier_all_baby_runs")
    assert [spec.name for spec in specs] == [
        "classifier_all_instruct_graph-social_names-baby",
        "classifier_all_reasoning_graph-social_names-baby",
    ]


def test_model_override_keeps_spec_otherwise_same():
    spec = expand_specs(CFG, spec="classifier_instruct", model_override=["gemma3:4b"])[0]
    assert [model.name for model in spec.models] == ["gemma3:4b"]


def test_unknown_spec_is_rejected():
    with pytest.raises(KeyError):
        load_run_spec(CFG, "missing")


def test_next_node_spec_uses_random_walk_prompt():
    spec = load_run_spec(CFG, "next_node_instruct_graph-social_names-baby")
    assert spec.prompt == "random-walk-next-node_graph-social_names-baby"
    assert spec.llm_instruct == "llm-next-node"


def test_next_node_spec_uses_generic_prompt():
    spec = load_run_spec(CFG, "next_node_instruct_graph-generic_names-baby")
    assert spec.prompt == "random-walk-next-node_graph-generic_names-baby"


def test_next_node_social_random_spec_uses_social_random_prompt():
    cfg = {
        **CFG,
        "prompt_conditions": [
            *CFG["prompt_conditions"],
            {"id": "graph-social_names-random", "graph_context": "social", "name_source": "random_strings"},
        ],
        "run_matrices": {
            **CFG["run_matrices"],
            "next_node": {
                **CFG["run_matrices"]["next_node"],
                "prompt_conditions": [
                    "graph-social_names-baby",
                    "graph-generic_names-baby",
                    "graph-social_names-random",
                ],
            },
        },
    }
    spec = load_run_spec(cfg, "next_node_instruct_graph-social_names-random")
    assert spec.prompt == "random-walk-next-node_graph-social_names-random"


def test_matrix_generated_next_node_specs_are_named_by_condition():
    specs = expand_specs(CFG, group="next_node_runs")
    assert [spec.name for spec in specs] == [
        "next_node_instruct_graph-generic_names-baby",
        "next_node_instruct_graph-social_names-baby",
    ]
    assert specs[0].prompt == "random-walk-next-node_graph-generic_names-baby"
