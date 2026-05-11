"""Response parsing helpers."""

from __future__ import annotations

import json
from typing import Any

STRIP_CHARS = ",.;:!?()[]{}<>\"'`"


def normalize_choice(raw: str, opt1: str, opt2: str) -> str | None:
    if not raw:
        return None
    token = raw.strip().strip("\"'`").split()[-1].strip(STRIP_CHARS)
    return {opt1.lower(): opt1, opt2.lower(): opt2}.get(token.lower())


def response_field(resp: Any, field: str, default: Any = None) -> Any:
    if isinstance(resp, dict):
        return resp.get(field, default)
    return getattr(resp, field, default)


def serialize_logprobs(logprobs: Any) -> tuple[str, str]:
    if not isinstance(logprobs, list):
        return json.dumps(None), json.dumps(None)

    logprobs_json = [
        {
            "token": response_field(lp, "token"),
            "logprob": response_field(lp, "logprob"),
        }
        for lp in logprobs
    ]
    top_logprobs_json = [
        [
            {
                "token": response_field(tlp, "token"),
                "logprob": response_field(tlp, "logprob"),
            }
            for tlp in response_field(lp, "top_logprobs", []) or []
        ]
        for lp in logprobs
    ]
    return (
        json.dumps(logprobs_json, ensure_ascii=False),
        json.dumps(top_logprobs_json, ensure_ascii=False),
    )


def parse_response(resp: Any, opt1: str, opt2: str) -> dict[str, Any]:
    raw = str(response_field(resp, "response", "") or "").strip()
    thinking = str(response_field(resp, "thinking", "") or "").strip()
    logprobs_json, top_logprobs_json = serialize_logprobs(response_field(resp, "logprobs"))
    llm_choice = normalize_choice(raw, opt1, opt2)

    return {
        "raw_response": raw,
        "llm_choice": llm_choice or "",
        "is_valid_choice": llm_choice is not None,
        "thinking": thinking,
        "logprobs_json": logprobs_json,
        "top_logprobs_json": top_logprobs_json,
    }


def normalize_node(raw: str, candidates: list[str]) -> str | None:
    if not raw:
        return None
    token = raw.strip().strip("\"'`").split()[0].strip(STRIP_CHARS)
    return {candidate.lower(): candidate for candidate in candidates}.get(token.lower())


def first_token_top_logprobs(resp: Any) -> dict[str, float]:
    logprobs = response_field(resp, "logprobs")
    if not isinstance(logprobs, list) or not logprobs:
        return {}

    top_logprobs = response_field(logprobs[0], "top_logprobs", []) or []
    scores: dict[str, float] = {}
    for item in top_logprobs:
        token = str(response_field(item, "token", "")).strip().strip(STRIP_CHARS).lower()
        logprob = response_field(item, "logprob")
        if token and logprob is not None:
            scores[token] = float(logprob)
    return scores


def candidate_logprobs(resp: Any, candidates: list[str]) -> dict[str, float | None]:
    top_scores = first_token_top_logprobs(resp)
    return {candidate: top_scores.get(candidate.lower()) for candidate in candidates}


def parse_next_node_response(resp: Any, all_nodes: list[str]) -> dict[str, Any]:
    raw = str(response_field(resp, "response", "") or "").strip()
    thinking = str(response_field(resp, "thinking", "") or "").strip()
    logprobs_json, top_logprobs_json = serialize_logprobs(response_field(resp, "logprobs"))
    node = normalize_node(raw, all_nodes)
    scores = candidate_logprobs(resp, all_nodes)

    return {
        "raw_response": raw,
        "llm_next_node": node or "",
        "is_valid_node": node is not None,
        "thinking": thinking,
        "candidate_logprobs_json": json.dumps(scores, ensure_ascii=False),
        "logprobs_json": logprobs_json,
        "top_logprobs_json": top_logprobs_json,
    }
