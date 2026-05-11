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
