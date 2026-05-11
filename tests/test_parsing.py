from llm_soc_nav.parsing import candidate_logprobs, normalize_choice, parse_next_node_response


def test_normalize_choice_accepts_plain_and_punctuation():
    assert normalize_choice("caroline", "caroline", "eden") == "caroline"
    assert normalize_choice("answer: eden.", "caroline", "eden") == "eden"


def test_normalize_choice_accepts_path_style_last_token():
    assert normalize_choice("oliver->caroline->reese caroline", "caroline", "eden") == "caroline"


def test_normalize_choice_rejects_unknown_text():
    assert normalize_choice("not sure", "caroline", "eden") is None


def test_parse_next_node_response_scores_candidate_nodes():
    resp = {
        "response": "b",
        "logprobs": [
            {
                "token": "b",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "b", "logprob": -0.1},
                    {"token": "c", "logprob": -2.0},
                ],
            }
        ],
    }

    parsed = parse_next_node_response(resp, ["a", "b", "c"])
    scores = candidate_logprobs(resp, ["a", "b", "c"])
    assert parsed["llm_next_node"] == "b"
    assert parsed["is_valid_node"] is True
    assert scores == {"a": None, "b": -0.1, "c": -2.0}
