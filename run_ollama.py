"""
run_ollama.py — research MVP for binary-choice probing across Ollama models.
"""

from __future__ import annotations
import os
import argparse
from dataclasses import dataclass, field
import json
from cv2 import log
import ollama
import pandas as pd
import pdb


# ---------------------------------------------------------------------------
# Config — edit here
# ---------------------------------------------------------------------------
@dataclass
class Config:
    questions_paths: dict[str, str] = field(default_factory=lambda: {
        "adj-ordered_choices-ordered": "prompts/questions_adj-ordered_choices-ordered.csv",
        "adj-shuffled_choices-ordered": "prompts/questions_adj-shuffled_choices-ordered.csv",
        "adj-ordered_choices-shuffled": "prompts/questions_adj-ordered_choices-shuffled.csv",
        "adj-shuffled_choices-shuffled": "prompts/questions_adj-shuffled_choices-shuffled.csv",
        "adj-list-shuffled_one-token-names_adj-prompt-ordered_choices-shuffled": "prompts/questions_adj-list-shuffled_one-token-names_adj-prompt-ordered_choices-shuffled.csv"
    })
    output_dir: str = "/users/dlcarste/data/llm_soc_nav/results"
    models: dict[str,list[str]] = field(default_factory=lambda: {
        "llm-classifier": [
        # "gemma3:270m",
        # "gemma3:1b",
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b",
        # "llama3.1:8b",
        # "llama3.2:1b",
        # "llama3.2:3b",
        # "mistral-nemo:latest",
        "mistral-small3.2:24b",
        # "qwen2.5vl:3b",
        # "qwen2.5vl:7b",
        # "qwen2.5vl:32b", # very slow, maybe try later
        ],
        "llm-path-classifier" :[
        "qwen3:8b",
        # "qwen3:14b",
        # "qwen3:30b",
        # "qwen3:32b",
        "deepseek-r1:7b",
        # "deepseek-r1:8b",
        # "deepseek-r1:14b",
        # "deepseek-r1:32b",
        # "deepseek-r1:70b",
        ],
        "llm-path": [
        # "gemma3:270m",
        "gemma3:1b",
        "gemma3:4b",
        "gemma3:12b",
        "gemma3:27b",
        ]
    })
    temperature: float = 0
    num_predict: dict[str, int] = field(default_factory=lambda: {
        "llm-classifier": 12,
        "llm-path-classifier": 5000,
        "llm-path": 12
    })
    top_logprobs: int = 12
    stop_tokens: list[str] = field(default_factory=lambda: [
        "\n", " ", "\t", ".", ",", ";", ":", "!", "?", "\"", "'"
    ])
    llm_instructs: dict[str, str] = field(default_factory=lambda: {
        "llm-classifier": ("You are a strict classifier and will be given a question with two options. Output exactly ONE word corresponding to the correct option and nothing else. "
                     "No punctuation. No explanation. Ensure that the spelling of your response matches the correct option exactly. Always respond in lowercase.\n\n"),
        "llm-path-classifier": ("You are a strict classifier and will be given a question with two options. Output the shortest path corresponding to the correct option and ONE word corresponding to the correct option. "
                          "The shortest path should be a sequence of connected names/nodes in the friendship graph, with each step separated by '->'. After the path, output the correct option as a single word, separated by a space."
                          "No punctuation. No further explanation. Ensure that the spelling of your response matches the correct option exactly. Always respond in lowercase.\n\n"),
        "llm-path": (
                    "You will be given a problem with two possible answer options. Interpret the problem as a graph traversal task:\n"
                    "Each person is a node in a graph.\n"
                    "Each friendship is an edge connecting two nodes.\n"
                    "\n"
                    "The two answer options correspond to the first node in the path from the start node to the target node. "
                    "Your task is to determine which option is the correct first step on the shortest path.\n"
                    "\n"
                    "After identifying the correct option, output the full shortest path from the start node to the target node.\n"
                    "The full path will be of length 3 - 5 nodes, including the start and target. The first node in the path must be the start node, and the last node must be the target node.\n"
                    "\n"
                    "Output format requirements:\n"
                    "Return only the shortest path as a sequence of connected node names.\n"
                    "Separate each step using '->'.\n"
                    "Do not include spaces, punctuation, or explanations.\n"
                    "Additional constraints:\n"
                    "Use exact node names as provided in the problem.\n"
                    "Output must be entirely lowercase.\n"
                    "Provide only the path and nothing else.\n\n"
                )
    })
    search_instructs: dict[str, str] = field(default_factory=lambda: {
        "search-base": "",
        "search-bfs": ("Use breadth-first search (BFS) to determine the correct choice. BFS is a graph traversal algorithm that explores all the vertices of a graph in breadth-first order, " 
        "starting from a given source vertex. In BFS, vertices are visited in layers, where the vertices at distance 1 from the source vertex are visited first, followed "
        "by the vertices at distance 2, and so on. BFS uses a queue data structure to keep track of the vertices to be visited, and it ensures that no vertex is visited more than once. "
        "BFS is useful for finding the shortest path between two vertices in an unweighted graph, or for exploring all the vertices in a graph."),
        "search-sr-low-gamma": ("The friendships listed above represent most of the relevant connections in the network. Additional friendships are uncommon. "
        "When deciding where to pass the message first, focus primarily on the friendships that are explicitly listed. Choose the option that leads to the target most efficiently using the connections shown."),
        "search-sr-medium-gamma": ("The friendships listed above may not include every connection in the network. People who share friends are often connected as well, even if those connections are not directly observed. "
        "For example, if A is friends with B and B is friends with C, then A and C are more likely to be friends even if that connection is not listed. "
        "When choosing where to send the message first, think about how each option might lead to the target through both observed and likely unobserved connections. "
        "Choose the option that is most likely to lead to the target through the network in the fewest steps."),
        "search-sr-high-gamma": ("The friendships listed above are only a small sample of a much larger network. Many additional connections likely exist beyond those shown, "
        "especially between people who share friends or belong to the same parts of the network. "
        "For example, if A is friends with B, B is friends with C, and C is friends with D, then it is more likely that A, B, C, and D and part of the same clique and are all connected with each other. "
        "When deciding where to send the message first, consider both the listed friendships and the broader network they imply through likely unobserved connections. Choose the option most likely to lead to the target through the fewest steps.")
    })


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
_STRIP = ",.;:!?()[]{}<>\"'`"


def normalize_choice(raw: str, opt1: str, opt2: str) -> str | None:
    """Return opt1 or opt2 if the last token of raw matches, else None."""
    if not raw:
        return None
    token = raw.strip().strip("\"'`").split()[-1].strip(_STRIP)
    return {opt1.lower(): opt1, opt2.lower(): opt2}.get(token.lower())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class OllamaClassifier:
    def __init__(self, model: str, cfg: Config, llm_instruct: str, search_instruct: str):
        self.model = model
        self.cfg = cfg
        self.llm_instruct = llm_instruct  # --llm-instruct: selects model list, generation config, and system-prompt prefix
        self.search_instruct = search_instruct

    def _call(self, question: str, startpoint: str, endpoint: str, opt1: str, opt2: str):
        llm_instruct    = self.cfg.llm_instructs.get(self.llm_instruct, "")      # --llm-instruct
        search_instruct = self.cfg.search_instructs.get(self.search_instruct, "")  # --search-instruct

        if self.llm_instruct == "llm-path":
            # print(f"output in the shortest path from {startpoint} to {endpoint} by filling in '{startpoint}->...->{endpoint}' beginning with either '{opt1}' or '{opt2}'.")
            question = f"{question}\n\noutput in the shortest path from {startpoint} to {endpoint} by filling in '{startpoint}->...->{endpoint}' beginning with either '{opt1}' or '{opt2}'."
            stop_tokens = self.cfg.stop_tokens + [endpoint]  # stop generation after the endpoint is generated
        else:
            # print(f"output which option is correct: '{opt1}' or '{opt2}'.")
            question = f"{question}\n\noutput which option is correct: '{opt1}' or '{opt2}'."
            stop_tokens = self.cfg.stop_tokens

        prompt = (
            f"{llm_instruct}"
            f"{question}\n\n"
            f"{search_instruct}"
        ).strip("\n")

        # print(question)
        
        if self.llm_instruct == "llm-path-classifier":
            resp = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.num_predict[self.llm_instruct],
                },
            )      
        else:
            resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            logprobs=True,
            top_logprobs=self.cfg.top_logprobs,
            options={
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.num_predict[self.llm_instruct],
                "stop": stop_tokens,
                },
            )
        return resp

    def _parse(self, resp, opt1: str, opt2: str) -> dict:
        raw = (resp.response or "").strip()
        thinking = (resp.thinking or "").strip()
        logprobs = resp.logprobs
        logprobs_json = (
            [{"token": lp.token, "logprob": lp.logprob} for lp in logprobs]
            if isinstance(logprobs, list) else None
        )
        top_logprobs_json = (
            [
                [{"token": tlp.token, "logprob": tlp.logprob} for tlp in lp.top_logprobs]
                for lp in logprobs
            ]
            if isinstance(logprobs, list) else None
        )
        llm_choice = normalize_choice(raw, opt1, opt2)

        return {
            "raw_response": raw,
            "llm_choice": llm_choice or "",
            "is_valid_choice": llm_choice is not None,
            "thinking": thinking,
            "logprobs_json": json.dumps(logprobs_json, ensure_ascii=False),
            "top_logprobs_json": json.dumps(top_logprobs_json, ensure_ascii=False),
        }

    def run(self, questions: pd.DataFrame) -> pd.DataFrame:
        keep_cols = ["question_id", "prompt_id", "startpoint", "endpoint",
                     "opt1", "opt2", "correct_choice"]
        records = []
        for i, row in questions.iterrows():
            resp = self._call(str(row["question"]), str(row["startpoint"]), str(row["endpoint"]),
                              str(row["opt1"]), str(row["opt2"]))
            record = {col: row[col] for col in keep_cols}
            record.update(self._parse(resp, row["opt1"], row["opt2"]))
            record["model"] = self.model
            records.append(record)
            print(f"  [{i+1}/{len(questions)}] {record['raw_response']!r:>12}  "
                  f"correct={record['correct_choice']}")
        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Get Ollama responses')
    parser.add_argument('--llm-instruct', type=str, default="llm-classifier",
                        help='Selects model list, generation config, and system-prompt prefix ("llm-classifier", "llm-path-classifier", "llm-path")')
    parser.add_argument('--prompt-type', type=str, default="adj-ordered_choices-ordered",
                        help='Prompt type to analyze (e.g., "adj-ordered_choices-ordered", "adj-shuffled_choices-ordered", etc.)')
    parser.add_argument('--search-instruct', type=str, default="search-base",
                        help='Search instruct type to use (e.g. "search-base", "search-bfs", etc.)')
    args = parser.parse_args()

    print(f"Running Ollama classifier with arguments: {args}")
    
    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)
    questions = pd.read_csv(cfg.questions_paths[args.prompt_type]).iloc[:5750, :]

    model_list = cfg.models[args.llm_instruct]

    for model in model_list:
        print(f"\n=== {model} ({len(questions)} questions) ===")
        clf = OllamaClassifier(model, cfg, args.llm_instruct, args.search_instruct)
        results = clf.run(questions)
        out_path = os.path.join(cfg.output_dir, f"{model.replace(':', '-')}_{args.llm_instruct}_{args.prompt_type}_{args.search_instruct}.csv")
        # out_path = os.path.join(cfg.output_dir, f"test.csv")
        results.to_csv(out_path, index=False)
        acc = results.loc[results["is_valid_choice"],
                          ["llm_choice", "correct_choice"]].pipe(
            lambda df: (df["llm_choice"] == df["correct_choice"]).mean()
        )
        print(f"  Saved {len(results)} rows → {out_path}  |  accuracy={acc:.3f}")
if __name__ == "__main__":
    main()