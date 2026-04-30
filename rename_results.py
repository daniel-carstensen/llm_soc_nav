#!/usr/bin/env python3
"""
rename_results.py — Rename result CSVs from old naming scheme to new naming scheme.

Old scheme (3-part):  {llm}_{adj}_{choices}.csv
New scheme (3-part):  {llm}_{adj-ordered|adj-shuffled}_{choices-ordered|choices-shuffled}.csv

Old scheme (5-part):  {llm}_{instruct}_{adj}_{choices}_{search}.csv
New scheme (5-part):  {llm}_{llm-classifier|llm-path-classifier}_{adj-ordered|adj-shuffled}_{choices-ordered|choices-shuffled}_{search-base|search-bfs|search-sr-*}.csv
"""

import os

RESULTS_DIR = "/users/dlcarste/data/llm_soc_nav/results"

INSTRUCT_MAP = {
    "base":      "llm-classifier",
    "reasoning": "llm-path-classifier",
}

ADJ_MAP = {
    "ordered":  "adj-ordered",
    "shuffled": "adj-shuffled",
}

CHOICES_MAP = {
    "ordered":  "choices-ordered",
    "shuffled": "choices-shuffled",
}

SEARCH_MAP = {
    "base":           "search-base",
    "bfs":            "search-bfs",
    "sr-low-gamma":   "search-sr-low-gamma",
    "sr-medium-gamma":"search-sr-medium-gamma",
    "sr-high-gamma":  "search-sr-high-gamma",
}


def rename_stem(stem: str) -> str | None:
    """Return new stem, or None if the file should be skipped."""
    parts = stem.split("_")
    n = len(parts)

    if n == 3:
        llm, adj, choices = parts
        # map if still in old format, or pass through if already converted
        new_adj     = ADJ_MAP.get(adj, adj if adj.startswith("adj-") else None)
        new_choices = CHOICES_MAP.get(choices, choices if choices.startswith("choices-") else None)
        if new_adj is None or new_choices is None:
            return None
        return f"{llm}_llm-classifier_{new_adj}_{new_choices}_search-base"

    elif n > 3:
        llm      = parts[0]
        instruct = parts[1]
        adj      = parts[2]
        choices  = parts[3]
        search   = parts[4] if n >= 5 else None

        # already converted?
        if instruct.startswith("llm-"):
            return None

        new_instruct = INSTRUCT_MAP.get(instruct)
        new_adj      = ADJ_MAP.get(adj)
        new_choices  = CHOICES_MAP.get(choices)
        if new_instruct is None or new_adj is None or new_choices is None:
            return None

        if search is not None:
            new_search = SEARCH_MAP.get(search)
            if new_search is None:
                return None
            suffix = "_".join(parts[5:])
            tail = f"_{suffix}" if suffix else ""
            return f"{llm}_{new_instruct}_{new_adj}_{new_choices}_{new_search}{tail}"
        else:
            return f"{llm}_{new_instruct}_{new_adj}_{new_choices}"

    return None


def main():
    files = sorted(
        f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")
    )

    renames: list[tuple[str, str]] = []
    skipped: list[str] = []

    for filename in files:
        stem = filename[:-4]
        new_stem = rename_stem(stem)
        if new_stem is None:
            skipped.append(filename)
        else:
            new_filename = new_stem + ".csv"
            if new_filename != filename:
                renames.append((filename, new_filename))
            else:
                skipped.append(filename)

    if not renames:
        print("No files require renaming.")
        return

    print(f"Files to rename ({len(renames)}):")
    for old, new in renames:
        print(f"  {old}")
        print(f"    -> {new}")

    print(f"\nFiles already up-to-date / skipped: {len(skipped)}")

    confirm = input("\nProceed with renaming? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    for old, new in renames:
        old_path = os.path.join(RESULTS_DIR, old)
        new_path = os.path.join(RESULTS_DIR, new)
        if os.path.exists(new_path):
            print(f"  SKIP (target exists): {new}")
            continue
        os.rename(old_path, new_path)
        print(f"  OK  {old} -> {new}")

    print("\nDone.")


if __name__ == "__main__":
    main()
