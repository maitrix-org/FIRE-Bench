#!/bin/bash
# =====================================================
# Run experiment script
# Usage:
#   bash run_experiment.sh [--agents ...] [--models ...] [--tasks ...] [--run_times ...]
# Example:
#   bash run_experiment.sh --agents openhands --models gpt-5 --tasks to_cot_or_not
#   bash run_experiment.sh --agents "openhands codex" --models "gpt-5 claude-sonnet-4-20250514" --tasks "to_cot_or_not rational"
# =====================================================

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Exit immediately on error
set -e

# ==============================
# Default arguments
# ==============================
DEFAULT_AGENTS="openhands"      # example agents
DEFAULT_TASKS="all"   # example tasks
DEFAULT_MODELS="gpt-5"          # example models
DEFAULT_RUN_TIMES=1

# ==============================
# Parse CLI arguments (optional overrides)
# ==============================
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agents) AGENTS="$2"; shift ;;
        --tasks) TASKS="$2"; shift ;;
        --models) MODELS="$2"; shift ;;
        --run_times) RUN_TIMES="$2"; shift ;;
        *) echo "[!] Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Fallback to defaults if unset
AGENTS=${AGENTS:-$DEFAULT_AGENTS}
TASKS=${TASKS:-$DEFAULT_TASKS}
MODELS=${MODELS:-$DEFAULT_MODELS}
RUN_TIMES=${RUN_TIMES:-$DEFAULT_RUN_TIMES}

# Convert space-separated strings to arrays
IFS=' ' read -ra AGENT_IDS <<< "$AGENTS"
IFS=' ' read -ra TASK_IDS <<< "$TASKS"
IFS=' ' read -ra LLM_MODELS <<< "$MODELS"

# Display configuration summary
echo "===================================="
echo "Starting Experiment"
echo "------------------------------------"
echo "Agents:    ${AGENT_IDS[*]}"
echo "Tasks:     ${TASK_IDS[*]}"
echo "Models:    ${LLM_MODELS[*]}"
echo "Run Times: $RUN_TIMES"
echo "===================================="
echo

# ==============================
# Run all combinations
# ==============================
for agent in "${AGENT_IDS[@]}"; do
  for task in "${TASK_IDS[@]}"; do
    for model in "${LLM_MODELS[@]}"; do

      echo "===================================================="
      echo "Running Agent: $agent | Task: $task | Model: $model"
      echo "===================================================="

      # run_agent.py auto-detects the API key from env vars
      python run_agent.py \
        --AGENT_ID "$agent" \
        --TASK_ID "$task" \
        --LLM_MODEL "$model" \
        --RUN_TIMES "$RUN_TIMES"

    done
  done
done
