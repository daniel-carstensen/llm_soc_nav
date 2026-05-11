from llm_soc_nav.parsing import normalize_choice


def test_normalize_choice_accepts_plain_and_punctuation():
    assert normalize_choice("caroline", "caroline", "eden") == "caroline"
    assert normalize_choice("answer: eden.", "caroline", "eden") == "eden"


def test_normalize_choice_accepts_path_style_last_token():
    assert normalize_choice("oliver->caroline->reese caroline", "caroline", "eden") == "caroline"


def test_normalize_choice_rejects_unknown_text():
    assert normalize_choice("not sure", "caroline", "eden") is None
