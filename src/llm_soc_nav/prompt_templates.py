"""Prompt text kept separate from generated prompt tables."""

from __future__ import annotations

LLM_INSTRUCTIONS: dict[str, str] = {
    "llm-classifier": (
        "You are a strict classifier and will be given a question with two options. "
        "Output exactly ONE word corresponding to the correct option and nothing else. "
        "No punctuation. No explanation. Ensure that the spelling of your response "
        "matches the correct option exactly. Always respond in lowercase.\n\n"
    ),
    "llm-path": (
        "You will be given a problem with two possible answer options. Interpret the "
        "problem as a graph traversal task:\nEach person is a node in a graph.\n"
        "Each friendship is an edge connecting two nodes.\n\nThe two answer options "
        "correspond to the first node in the path from the start node to the target "
        "node. Your task is to determine which option is the correct first step on "
        "the shortest path.\n\nAfter identifying the correct option, output the full "
        "shortest path from the start node to the target node.\nThe full path will "
        "be of length 3 - 5 nodes, including the start and target. The first node in "
        "the path must be the start node, and the last node must be the target node.\n\n"
        "Output format requirements:\nReturn only the shortest path as a sequence of "
        "connected node names.\nSeparate each step using '->'.\nDo not include spaces, "
        "punctuation, or explanations.\nAdditional constraints:\nUse exact node names "
        "as provided in the problem.\nOutput must be entirely lowercase.\nProvide only "
        "the path and nothing else.\n\n"
    ),
    "llm-next-node": (
        "You predict the next node in a random walk on a friendship graph. "
        "Respond with exactly one node label and nothing else. No punctuation. "
        "No explanation. Always match the node label spelling exactly.\n\n"
    ),
}

SEARCH_INSTRUCTIONS: dict[str, str] = {
    "search-base": "",
    "search-bfs": (
        "Use breadth-first search (BFS) to determine the correct choice. BFS is a graph traversal algorithm that explores all the vertices of a graph in breadth-first order, "
        "starting from a given source vertex. In BFS, vertices are visited in layers, where the vertices at distance 1 from the source vertex are visited first, followed "
        "by the vertices at distance 2, and so on. BFS uses a queue data structure to keep track of the vertices to be visited, and it ensures that no vertex is visited more than once. "
        "BFS is useful for finding the shortest path between two vertices in an unweighted graph, or for exploring all the vertices in a graph."
    ),
    "search-sr-low-gamma": (
        "The friendships listed above represent most of the relevant connections in the network. Additional friendships are uncommon. "
        "When deciding where to pass the message first, focus primarily on the friendships that are explicitly listed. Choose the option that leads to the target most efficiently using the connections shown."
    ),
    "search-sr-medium-gamma": (
        "The friendships listed above may not include every connection in the network. People who share friends are often connected as well, even if those connections are not directly observed. "
        "For example, if A is friends with B and B is friends with C, then A and C are more likely to be friends even if that connection is not listed. "
        "When choosing where to send the message first, think about how each option might lead to the target through both observed and likely unobserved connections. "
        "Choose the option that is most likely to lead to the target through the network in the fewest steps."
    ),
    "search-sr-high-gamma": (
        "The friendships listed above are only a small sample of a much larger network. Many additional connections likely exist beyond those shown, "
        "especially between people who share friends or belong to the same parts of the network. "
        "For example, if A is friends with B, B is friends with C, and C is friends with D, then it is more likely that A, B, C, and D and part of the same clique and are all connected with each other. "
        "When deciding where to send the message first, consider both the listed friendships and the broader network they imply through likely unobserved connections. Choose the option most likely to lead to the target through the fewest steps."
    ),
    "search-random-walk": (
        "A walker follows this rule: at each step, choose uniformly at random among the current node's neighbors. "
        "The walker may revisit nodes, and immediate backtracking is allowed."
    ),
}


def classifier_question(start: str, end: str, opt1: str, opt2: str) -> str:
    return (
        f"{start} wants to pass a message most efficiently to {end}, should {start} "
        f"begin by passing the message on to {opt1} or {opt2}?"
    )


def path_question(start: str, end: str, opt1: str, opt2: str) -> str:
    return (
        f"Find the shortest path from {start} to {end}. The first step after {start} "
        f"must be either {opt1} or {opt2}."
    )


def random_walk_question(path: list[str]) -> str:
    return (
        "Path so far:\n"
        f"{' -> '.join(path)}\n\n"
        "What is the next node? Respond with exactly one node label."
    )
