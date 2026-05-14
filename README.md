# LLM Social Navigation

Minimal research code for probing Ollama LLMs on social network navigation problems. The repo generates deterministic prompt CSVs from a fixed graph, runs named Ollama experiment specs, and saves BIDS-style response CSVs.

## Setup

```bash
uv sync --all-groups
```

Run commands through the uv-managed environment:

```bash
uv run python -m llm_soc_nav list-runs
```

On the cluster, use `scripts/run_ollama.sh`; it assumes `uv` is available on `PATH`.

## Layout

```text
configs/default.yaml      # paths, prompt generation, model groups, named run specs
data/raw/                 # source CSVs
data/prompts/             # canonical generated prompt CSVs
src/llm_soc_nav/          # small research package
scripts/run_ollama.sh     # Slurm entrypoint
archive/                  # old sandbox scripts kept for reference
```

## Workflow

List available named runs:

```bash
uv run python -m llm_soc_nav list-runs
```

Regenerate canonical prompt CSVs:

```bash
uv run python -m llm_soc_nav generate-prompts
```

Run one experiment spec:

```bash
uv run python -m llm_soc_nav run --spec classifier_instruct_social_names
```

Run a group:

```bash
uv run python -m llm_soc_nav run --group path_runs
```

Run random-walk next-node scoring for instruct models:

```bash
uv run python -m llm_soc_nav run --spec next_node_instruct_social_names
```

Run the two non-social next-node controls:

```bash
uv run python -m llm_soc_nav run --group next_node_control_runs
```

Tiny smoke run with one model and two questions:

```bash
uv run python -m llm_soc_nav run --spec classifier_instruct_social_names --limit 2 --models gemma3:4b
```

Run tests:

```bash
uv run pytest
```

## Config

`configs/default.yaml` is the main editing surface. It defines raw data paths, prompt conditions, model groups, generation settings, compact run matrices, and run groups. Concrete run specs are generated from those matrices and can be inspected with `list-runs`.

`name_source` can be `baby_names` or `random_strings`. `baby_names` filters the raw baby-name CSV to short one-token-style names; `random_strings` generates random four-character lowercase strings.

Model groups are intentionally combinable with the task instructions:

- `instruct_llm`
- `reasoning_llm`

Reasoning models that support disabling reasoning can also appear in `instruct_llm` with a `prompt_prefix`, such as `/no_think`.

Each model can optionally define its own Ollama `options`. These override the defaults for the selected LLM instruction, so model-specific recommended settings can live next to the model:

```yaml
model_groups:
  reasoning_llm:
    - name: qwen3:8b
      label: qwen3-8b
      options:
        temperature: 0.6
        top_p: 0.95
        top_k: 20
```

Search instructions are only applied to `llm-path` runs. Use `path_search_runs` to run the BFS and social-reasoning prompt variants:

```bash
uv run python -m llm_soc_nav run --group path_search_runs
```

The `llm-next-node` run specs use instruct models only. Prompt generation creates random-walk prefixes with configurable lengths, such as `[0, 1, 2, 3, 5, 8]`, and records candidate logprobs for all graph node labels returned by the API. Four random-walk conditions are generated:

- `next_node_instruct_social_names`: social friendship graph with baby-name node labels.
- `next_node_instruct_social_names`: social friendship graph with baby-name node labels.
- `next_node_instruct_social_random_strings`: social friendship graph with random four-character node labels.
- `next_node_instruct_generic_names`: generic graph wording with baby-name node labels.
- `next_node_instruct_generic_random_strings`: generic graph wording with random four-character node labels.

## Results

Results save outside the repo by default:

```text
/users/dlcarste/data/llm_soc_nav/results
```

Filenames use a simple BIDS-style convention:

```text
model-gemma3-4b_task-llm-path_prompt-adj-list-shuffled_adj-prompt-shuffled_choices-shuffled_search-base_responses.csv
```

Output columns include question metadata, raw model response, normalized choice, validity flag, optional thinking text, serialized logprobs, model name, and run spec.

## Slurm

Submit the cluster job with:

```bash
sbatch scripts/run_ollama.sh
```

Edit the final `srun` line in `scripts/run_ollama.sh` to choose the spec or group.
