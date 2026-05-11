import pandas as pd

from llm_soc_nav.ollama_client import build_generation_options, build_model_prompt
from llm_soc_nav.ollama_client import run_model
from llm_soc_nav.run_specs import ModelSpec, RunSpec


def test_path_prompt_uses_path_question():
    row = pd.Series(
        {
            "question": "classifier wording",
            "path_question": "path wording",
            "opt1": "b",
            "opt2": "d",
        }
    )
    spec = RunSpec(
        name="path_reasoning",
        prompt="tiny",
        llm_instruct="llm-path",
        search_instruct="search-base",
        models=[ModelSpec(name="qwen3:8b", label="qwen3-8b")],
        output_label="llm-path",
    )

    prompt = build_model_prompt(row, spec)
    assert "path wording" in prompt
    assert "classifier wording" not in prompt


def test_search_instruction_is_only_added_to_path_prompts():
    row = pd.Series(
        {
            "question": "classifier wording",
            "path_question": "path wording",
            "opt1": "b",
            "opt2": "d",
        }
    )
    classifier_spec = RunSpec(
        name="classifier_instruct",
        prompt="tiny",
        llm_instruct="llm-classifier",
        search_instruct="search-bfs",
        models=[],
        output_label="llm-classifier",
    )
    path_spec = RunSpec(
        name="path_reasoning_bfs",
        prompt="tiny",
        llm_instruct="llm-path",
        search_instruct="search-bfs",
        models=[],
        output_label="llm-path",
    )

    assert "breadth-first search" not in build_model_prompt(row, classifier_spec).lower()
    assert "breadth-first search" in build_model_prompt(row, path_spec).lower()


def test_model_prompt_prefix_is_applied():
    row = pd.Series(
        {
            "question": "classifier wording",
            "path_question": "path wording",
            "opt1": "b",
            "opt2": "d",
        }
    )
    spec = RunSpec(
        name="classifier_instruct",
        prompt="tiny",
        llm_instruct="llm-classifier",
        search_instruct="search-base",
        models=[ModelSpec(name="qwen3:8b", label="qwen3-8b-no-think", prompt_prefix="/no_think\n\n")],
        output_label="llm-classifier",
    )

    prompt = build_model_prompt(row, spec, spec.models[0])
    assert prompt.startswith("/no_think")


def test_next_node_prompt_uses_random_walk_question_and_search_rule():
    row = pd.Series(
        {
            "question": "classifier wording",
            "path_question": "path wording",
            "random_walk_question": "Path so far:\na -> b\n\nWhat is the next node?",
            "opt1": "b",
            "opt2": "d",
        }
    )
    spec = RunSpec(
        name="next_node_instruct",
        prompt="random-walk-next-node",
        llm_instruct="llm-next-node",
        search_instruct="search-random-walk",
        models=[],
        output_label="llm-next-node",
    )

    prompt = build_model_prompt(row, spec)
    assert "Path so far" in prompt
    assert "choose uniformly at random" in prompt
    assert "classifier wording" not in prompt


def test_model_options_override_generation_defaults(monkeypatch, tmp_path):
    calls = []

    def fake_call_ollama(model, prompt, options, use_logprobs, top_logprobs, **kwargs):
        calls.append(options)
        return {"response": "b", "logprobs": None}

    monkeypatch.setattr("llm_soc_nav.ollama_client.call_ollama", fake_call_ollama)
    cfg = {
        "paths": {"results_dir": str(tmp_path)},
        "generation": {
            "top_logprobs": 12,
            "llm-classifier": {
                "num_predict": 12,
                "logprobs": False,
                "options": {"temperature": 0, "top_p": 0},
            },
        },
    }
    spec = RunSpec(
        name="classifier_instruct",
        prompt="tiny",
        llm_instruct="llm-classifier",
        search_instruct="search-base",
        models=[],
        output_label="llm-classifier",
    )
    model = ModelSpec(
        name="custom:1b",
        label="custom-1b",
        options={"temperature": 0.7, "num_predict": 99},
    )
    questions = pd.DataFrame(
        [
            {
                "question_id": 0,
                "prompt_id": 0,
                "startpoint": "a",
                "endpoint": "c",
                "opt1": "b",
                "opt2": "d",
                "correct_choice": "b",
                "question": "a asks b or d?",
                "path_question": "path",
            }
        ]
    )

    run_model(cfg, spec, model, questions)
    assert calls[0]["temperature"] == 0.7
    assert calls[0]["top_p"] == 0
    assert calls[0]["num_predict"] == 99


def test_build_generation_options_merges_task_and_model_settings():
    generation = {
        "num_predict": 12,
        "options": {"temperature": 0, "top_p": 0, "seed": 42},
    }
    model = ModelSpec(
        name="custom:1b",
        label="custom-1b",
        options={"temperature": 0.7, "num_predict": 99},
    )

    options = build_generation_options(generation, model)
    assert options == {"temperature": 0.7, "top_p": 0, "seed": 42, "num_predict": 99}
