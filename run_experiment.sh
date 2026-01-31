#!/bin/bash

# ==============================
# Agent / Task / Model lists
# ==============================
AGENT_IDS=("openhands")      # example agents
TASK_IDS=("to_cot_or_not")  # example tasks
LLM_MODELS=("gpt-5")  # example models

# ==============================
# API keys loaded from environment
# (set OPENAI_API_KEY and/or ANTHROPIC_API_KEY before running)
# ==============================

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
        --RUN_TIMES 1

    done
  done
done
