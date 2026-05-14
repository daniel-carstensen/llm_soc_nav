import pandas as pd

from llm_soc_nav.prompt_generation import (
    PROMPT_COLUMNS,
    PromptSettings,
    generate_questions,
    generate_random_walk_questions,
    prompt_conditions,
)


def test_generate_questions_is_deterministic():
    adjacency_sets = [(["a is friends with b.", "b is friends with c."], list("abcdefghijklm"))]
    tasks = pd.DataFrame(
        [
            {
                "startpoint_id": 1,
                "endpoint_id": 3,
                "opt1_id": 2,
                "opt2_id": 8,
                "correct_choice": 2,
            }
        ]
    )
    settings = PromptSettings(
        n_prompt_sets=1,
    )

    first = generate_questions(adjacency_sets, tasks, settings, seed=42)
    second = generate_questions(adjacency_sets, tasks, settings, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == PROMPT_COLUMNS
    assert "Find the shortest path" in first.loc[0, "path_question"]


def test_generate_random_walk_questions_uses_configurable_prefix_lengths():
    adjacency_sets = [(["a is friends with b."], list("abcdefghijklm"))]
    questions = generate_random_walk_questions(
        adjacency_sets,
        prefix_lengths=[0, 2],
        seed=42,
        graph_context="social",
        name_source="baby_names",
    )

    assert set(questions["prefix_length"]) == {0, 2}
    assert len(questions) == 26
    zero_prefix = questions.loc[questions["prefix_length"] == 0].iloc[0]
    assert "Graph:" in zero_prefix["random_walk_question"]
    assert "Path so far:" in zero_prefix["random_walk_question"]
    assert zero_prefix["path_so_far"] == zero_prefix["start_node"]
    assert zero_prefix["graph_context"] == "social"
    assert zero_prefix["name_source"] == "baby_names"


def test_prompt_conditions_include_all_graph_name_controls():
    conditions = prompt_conditions({})
    nav_prompts = {condition["nav_prompt"] for condition in conditions}
    walk_prompts = {condition["random_walk_prompt"] for condition in conditions}
    assert "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled_social-names" in nav_prompts
    assert "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled_generic-random-strings" in nav_prompts
    assert "random-walk-next-node_social-random-strings" in walk_prompts
    assert "random-walk-next-node_generic-names" in walk_prompts
