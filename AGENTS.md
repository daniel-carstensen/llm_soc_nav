# AGENTS.md

## Research repo rules

- Keep this a minimal research repo, not a production service.
- Prefer direct, readable modules over broad abstractions.
- Avoid broad `try/except` blocks; let research errors surface clearly.
- Preserve deterministic prompt generation unless the researcher explicitly changes the seed.
- Do not commit logs, large result files, caches, or temporary outputs.
- Keep cluster-specific setup isolated in `scripts/run_ollama.sh`.
- Make frequent use of tiny smoke tests to validate assumptions and cluster setup without risking large compute runs.
- Make frequent git commits by feature or fix, with clear messages. Avoid large commits that mix unrelated changes.

## Cluster and interactive-job rules

### Node check (MANDATORY)

Before running compute commands:

```bash
hostname
echo "$SLURM_JOB_ID"
```

If `SLURM_JOB_ID` is empty, you are on a login node and you should automatically request to run an interactive job.

### Interactive job usage

Example:

```bash
interact -n 4 -t 01:00:00 -m 16g
```

GPU:

```bash
interact -n 4 -t 01:00:00 -m 32g -g 1
```

### Codex rules on cluster

Codex must:

1. Check node status before running compute.
2. NEVER run compute-heavy commands on a login node.
3. USE interactive jobs when needed.
4. NEVER run full GPU jobs automatically.
5. Automatically request a small CPU interactive job before running tests or other compute.
6. Prefer tiny smoke tests.

### Allowed on login node

```bash
ls
pwd
cat
grep
git diff
mkdir
touch
```

### NOT allowed on login node

```bash
pytest
python scripts/smoke_test.py
```
