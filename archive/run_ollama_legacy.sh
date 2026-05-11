#!/bin/bash
#SBATCH --job-name=ollama_batch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- site-specific environment setup ---
module purge
module load ollama/0.17.7
module load miniforge3/25.3.0-3-a6hh
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate ulg_v1

# --- Ollama config with dynamic port assignment ---
# Find an available port starting from 11434
find_available_port() {
    local port=11434
    local max_attempts=100
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if ! timeout 1 bash -c "</dev/tcp/127.0.0.1/$port" 2>/dev/null; then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
        attempt=$((attempt + 1))
    done
    
    echo "Could not find available port after $max_attempts attempts" >&2
    return 1
}

OLLAMA_PORT=$(find_available_port)
if [[ $? -ne 0 ]]; then
    exit 1
fi

export OLLAMA_HOST=127.0.0.1:${OLLAMA_PORT}
export OLLAMA_KEEP_ALIVE=-1
echo "Using Ollama on port ${OLLAMA_PORT}"

cleanup() {
    if [[ -n "${OLLAMA_PID:-}" ]]; then
        kill "${OLLAMA_PID}" 2>/dev/null || true
        wait "${OLLAMA_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start Ollama on the allocated node
ollama serve > "logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

# Wait until the API is actually reachable
ready=0
for _ in $(seq 1 60); do
    if curl -sf http://127.0.0.1:11434/api/ps >/dev/null; then
        ready=1
        break
    fi
    sleep 2
done

if [[ "$ready" -ne 1 ]]; then
    echo "Ollama failed to start" >&2
    exit 1
fi

# Run your Python client inside the same allocation
# srun /users/dlcarste/.conda/envs/ulg_v1/bin/python3 run_ollama.py --llm-instruct llm-classifier --prompt-type adj-ordered_choices-shuffled --search-instruct search-sr-medium-gamma
srun /users/dlcarste/.conda/envs/ulg_v1/bin/python3 run_ollama.py --llm-instruct llm-path --prompt-type adj-list-shuffled_one-token-names_adj-prompt-ordered_choices-shuffled