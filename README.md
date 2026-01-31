# FIRE-Bench

## Setup

### Environment

```bash
mamba create -n firebench python=3.11  # or conda
mamba activate firebench
pip install -r requirements.txt
```

### API Keys

Create a `.env` file:

```
OPENAI_API_KEY=
GOOGLE_API_KEY=
HF_TOKEN=
CLAUDE_API_KEY=
```

### Agent Dependencies

- **Codex**: `npx @openai/codex@0.39.0 --version` (v0.39.0 required for timestamp logging)
- **Claude Code**: [Setup guide](https://code.claude.com/docs/en/setup) or `curl -fsSL https://claude.ai/install.sh | bash`
- **OpenHands**: `export OPENHANDS_HOME=./.openhands; mkdir  -p ./.openhands` (Requires docker)

## Usage

### 1. Download Data for Specific Benchmark

Some datasets in certain benchmarks are too big or cannot be directly loaded from huggingface. Instructions of getting them are provided in `dataset.txt` in the benchmark folder.

### 2. Run Experiments

Edit `run_experiment.sh` to configure your agent/task/model combinations, then run:

```bash
bash run_experiment.sh
```

This iterates over all combinations of `AGENT_IDS`, `TASK_IDS`, and `LLM_MODELS`, calling `run_agent.py` for each. Results are saved to the `log/` folder.

**Parameters in `run_experiment.sh`:**
- `AGENT_IDS`: agents to run (e.g., `codex`, `claude_code`, `openhands`)
- `TASK_IDS`: benchmark tasks (e.g., `rational`)
- `LLM_MODELS`: models to use (e.g., `gpt-5`)

### 3. Evaluate Results

After experiments finish, evaluate the generated logs:

```bash
# Evaluate all agents/models/tasks
bash run_eval.sh --agents all --models all --tasks all

# Evaluate a specific run
bash run_eval.sh --agents codex --models gpt-5 --tasks rational --timestamp 20251016232701_10997
```

**Options:**
- `--agents`: agent name or `all`
- `--models`: model name or `all`
- `--tasks`: task name or `all`
- `--timestamp`: (optional) evaluate a specific run by timestamp

## Status

| Agent | Status |
|-------|--------|
| Codex | Ready |
| Claude Code | Not examined (no API credit) |
| OpenHands | Ready |