from llm_soc_nav.results import result_filename, sanitize_label


def test_sanitize_label_makes_model_names_filename_safe():
    assert sanitize_label("gemma3:4b") == "gemma3-4b"


def test_result_filename_is_stable():
    assert result_filename(
        "gemma3:4b",
        "llm-path",
        "adj-list-shuffled_adj-prompt-shuffled_choices-shuffled",
        "search-base",
    ) == (
        "model-gemma3-4b_task-llm-path_"
        "prompt-adj-list-shuffled_adj-prompt-shuffled_choices-shuffled_"
        "search-base_responses.csv"
    )
