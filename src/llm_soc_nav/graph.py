"""Fixed graph used to generate the social navigation prompts."""

from __future__ import annotations

SOCIAL_GRAPH: dict[int, list[int]] = {
    0: [1, 7],
    1: [0, 3, 2],
    2: [1, 3, 4],
    3: [1, 2, 4, 5],
    4: [2, 3, 6],
    5: [3, 6],
    6: [4, 5],
    7: [0, 8, 9],
    8: [7, 10],
    9: [7, 11],
    10: [8, 11, 12],
    11: [9, 10, 12],
    12: [10, 11],
}


def friendship_sentences(names: list[str]) -> list[str]:
    """Return directed friendship sentences for a 13-name graph assignment."""
    sentences: list[str] = []
    for node_id in range(len(SOCIAL_GRAPH)):
        for neighbor_id in sorted(SOCIAL_GRAPH[node_id]):
            sentences.append(f"{names[node_id]} is friends with {names[neighbor_id]}.")
    return sentences


def generic_graph_sentences(names: list[str]) -> list[str]:
    """Return directed generic edge sentences for a 13-label graph assignment."""
    sentences: list[str] = []
    for node_id in range(len(SOCIAL_GRAPH)):
        for neighbor_id in sorted(SOCIAL_GRAPH[node_id]):
            sentences.append(f"{names[node_id]} is connected to {names[neighbor_id]}.")
    return sentences


def graph_sentences(names: list[str], graph_context: str = "social") -> list[str]:
    if graph_context == "social":
        return friendship_sentences(names)
    if graph_context == "generic":
        return generic_graph_sentences(names)
    raise ValueError("graph_context must be 'social' or 'generic'.")
