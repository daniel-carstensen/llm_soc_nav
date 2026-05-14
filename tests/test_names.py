import pytest

from llm_soc_nav.names import (
    assert_one_token_names,
    configured_model_names,
    filter_one_token_names,
    select_names,
    tokenizer_refs_for_config,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "amy" or (len(text) == 4 and text.islower()):
            return [1]
        return [1, 2]


def test_filter_one_token_names_uses_model_tokenizers():
    names = ["amy", "ann", "bcdx", "toolong"]
    filtered = filter_one_token_names(names, model_tokenizers={"model-a": FakeTokenizer()})
    assert filtered == ["amy", "bcdx"]


def test_assert_one_token_names_reports_bad_models():
    with pytest.raises(ValueError, match="ann: model-a"):
        assert_one_token_names(["amy", "ann"], {"model-a": FakeTokenizer()})


def test_select_random_names_can_filter_with_tokenizers(tmp_path):
    baby_names = tmp_path / "names.csv"
    baby_names.write_text("Child's First Name\nAmy\nAnn\n", encoding="utf-8")

    names = select_names(
        baby_names,
        source="random_strings",
        random_count=1,
        seed=1,
        model_tokenizers={"model-a": FakeTokenizer()},
        random_candidate_multiplier=100,
    )

    assert len(names) == 1
    assert len(names[0]) == 4


def test_tokenizer_refs_cover_all_configured_models():
    cfg = {
        "model_groups": {
            "instruct_llm": ["gemma3:4b", {"name": "qwen3:8b"}],
            "reasoning_llm": ["gemma3:4b"],
        },
        "name_token_check": {
            "family_tokenizers": {
                "gemma3": "google/gemma-3-4b-it",
                "qwen3": "Qwen/Qwen3-8B",
            }
        },
    }

    assert configured_model_names(cfg) == ["gemma3:4b", "qwen3:8b"]
    assert tokenizer_refs_for_config(cfg) == {
        "gemma3:4b": "google/gemma-3-4b-it",
        "qwen3:8b": "Qwen/Qwen3-8B",
    }


def test_tokenizer_refs_reject_missing_model():
    cfg = {
        "model_groups": {"instruct_llm": ["missing:1b"]},
        "name_token_check": {"family_tokenizers": {}},
    }

    with pytest.raises(ValueError, match="missing:1b"):
        tokenizer_refs_for_config(cfg)
