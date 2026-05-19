#!/bin/bash
#SBATCH --job-name=ollama_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

# --- site-specific environment setup ---
module purge
module load ollama/0.21.0-llj6

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not available on PATH" >&2
    exit 1
fi

uv sync --frozen --all-groups

find_available_port() {
    local port=11434
    local max_attempts=100
    local attempt=0

    while [[ "${attempt}" -lt "${max_attempts}" ]]; do
        if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            echo "${port}"
            return 0
        fi
        port=$((port + 1))
        attempt=$((attempt + 1))
    done

    echo "Could not find available port after ${max_attempts} attempts" >&2
    return 1
}

OLLAMA_PORT=$(find_available_port)
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_KEEP_ALIVE=-1
echo "Using Ollama on ${OLLAMA_HOST}"

cleanup() {
    if [[ -n "${OLLAMA_PID:-}" ]]; then
        kill "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

ollama serve > "logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

ready=0
for _ in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/ps" >/dev/null; then
        ready=1
        break
    fi
    sleep 2
done

if [[ "${ready}" -ne 1 ]]; then
    echo "Ollama failed to start" >&2
    exit 1
fi

# Edit --spec or --group for the desired experiment.
srun uv run python -m llm_soc_nav run --group instruct_llm_runs
