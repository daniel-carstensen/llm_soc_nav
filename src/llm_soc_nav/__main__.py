"""Command-line interface for the research workflow."""

from __future__ import annotations

import argparse

from llm_soc_nav.config import DEFAULT_CONFIG, load_config
from llm_soc_nav.names import check_config_name_tokens, save_names
from llm_soc_nav.ollama_client import run_spec
from llm_soc_nav.prompt_generation import generate_all_prompts
from llm_soc_nav.results import missing_runs
from llm_soc_nav.run_specs import expand_specs, list_run_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm_soc_nav")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-prompts", help="Regenerate canonical prompt CSVs.")
    generate.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config.")

    list_runs = subparsers.add_parser("list-runs", help="List named run specs and groups.")
    list_runs.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config.")

    missing = subparsers.add_parser("missing-runs", help="Show runs with no result file yet.")
    missing.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config.")

    save = subparsers.add_parser("save-names", help="Generate and save name CSVs to data/names/.")    
    save.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config.")
    save.add_argument("--local-files-only", action="store_true", help="Load tokenizers only from local cache/paths.")

    check_names = subparsers.add_parser("check-name-tokens", help="Hard-check names against configured model tokenizers.")
    check_names.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config.")
    check_names.add_argument("--local-files-only", action="store_true", help="Load tokenizers only from local cache/paths.")

    run = subparsers.add_parser("run", help="Run a named spec or group against Ollama.")
    run.add_argument("--config", default=argparse.SUPPRESS, help="Path to YAML config.")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--spec", help="Single run spec name from config.")
    target.add_argument("--group", help="Run group name from config.")
    run.add_argument("--limit", type=int, help="Limit number of questions per model.")
    run.add_argument("--models", nargs="+", help="Override models for this run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)

    if args.command == "save-names":
        if args.local_files_only:
            cfg.setdefault("name_token_check", {})["local_files_only"] = True
        counts = save_names(cfg)
        print(
            f"Saved {counts['baby_names']} baby names and {counts['random_strings']} random strings "
            f"(checked against {counts['models']} model tokenizers)."
        )
    elif args.command == "generate-prompts":
        for path in generate_all_prompts(cfg):
            print(f"Wrote {path}")
    elif args.command == "list-runs":
        print(list_run_specs(cfg))
    elif args.command == "missing-runs":
        missing = missing_runs(cfg)
        if not missing:
            print("All runs complete.")
        else:
            current_spec = None
            for spec_name, model_label in sorted(missing):
                if spec_name != current_spec:
                    print(f"\n{spec_name}")
                    current_spec = spec_name
                print(f"  {model_label}")
            print(f"\n{len(missing)} run(s) missing.")
    elif args.command == "check-name-tokens":
        if args.local_files_only:
            cfg.setdefault("name_token_check", {})["local_files_only"] = True
        counts = check_config_name_tokens(cfg)
        print(
            f"Checked {counts['baby_names']} baby names and {counts['random_strings']} random strings "
            f"against {counts['models']} configured model tokenizers."
        )
    elif args.command == "run":
        specs = expand_specs(cfg, spec=args.spec, group=args.group, model_override=args.models)
        for spec in specs:
            run_spec(cfg, spec, limit=args.limit)


if __name__ == "__main__":
    main()
