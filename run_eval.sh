#!/bin/bash
# =====================================================
# Run evaluation script for RAG log analysis
# Usage:
#   bash run_eval.sh [--agents ...] [--models ...] [--tasks ...] [--timestamp ...]
# Example:
#   bash run_eval.sh --agents all --models all --tasks all
#   bash run_eval.sh --agents openhands --models claude-sonnet-4-20250514 --tasks known_evaluation --timestamp 20251016232701_10997
# =====================================================

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Exit immediately on error
set -e

# Path to your Python evaluation script
EVAL_SCRIPT="eval/RAGChecker/eval.py"

# Default arguments (you can modify them here)
DEFAULT_AGENTS=("codex")
DEFAULT_MODELS=("gpt-5")
DEFAULT_TASKS="rational"
DEFAULT_TIMESTAMP=""

# Parse CLI arguments (optional overrides)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agents) AGENTS=("$2"); shift ;;
        --models) MODELS=("$2"); shift ;;
        --tasks) TASKS=("$2"); shift ;;
        --timestamp) TIMESTAMP="$2"; shift ;;
        *) echo "[!] Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Fallback to defaults if unset
AGENTS=${AGENTS:-$DEFAULT_AGENTS}
MODELS=${MODELS:-$DEFAULT_MODELS}
TASKS=${TASKS:-$DEFAULT_TASKS}

# Display configuration summary
echo "===================================="
echo "Starting Evaluation"
echo "------------------------------------"
echo "Agents:    $AGENTS"
echo "Models:    $MODELS"
echo "Tasks:     $TASKS"
echo "Timestamp: ${TIMESTAMP:-<all>}"
echo "===================================="
echo

# Build the Python command dynamically
CMD="python $EVAL_SCRIPT --agents $AGENTS --models $MODELS --tasks $TASKS"
if [ -n "$TIMESTAMP" ]; then
    CMD="$CMD --timestamp $TIMESTAMP"
fi

# Run the evaluation
echo "[*] Running command:"
echo "    $CMD"
echo
$CMD

# Done
echo
echo "Evaluation completed successfully!"
echo "------------------------------------"