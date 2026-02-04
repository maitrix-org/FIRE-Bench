import argparse
import os
import sys
import subprocess
import time
import random
import toml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# MODEL PRICING dictionary
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4o":      {"input": 2.50, "cached_input": 1.25,  "output": 10.00},
    "gpt-4.1":      {"input": 2.00, "cached_input": 0.50,  "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10,  "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "o1":       {"input": 15.00, "cached_input": 7.50,  "output": 60.00},    
    "o1-mini":  {"input": 1.10,  "cached_input": 0.55,  "output": 4.40},      
    "o3":       {"input": 2.00,  "cached_input": 0.50,  "output": 8.00}, 
    "o3-mini":  {"input": 1.10,  "cached_input": 0.55,  "output": 4.40},
    "o4-mini":  {"input": 1.10,  "cached_input": 0.275, "output": 4.40},
    "gpt-4-turbo":  {"input": 10.00, "cached_input": None, "output": 30.00},
    "gpt-3.5-turbo":{"input": 0.50,  "cached_input": None, "output": 1.50}
}


def normalize_model_name(name):
    # Converts 'openai/o4-mini' to 'o4-mini'.
    if "/" in name:
        return name.split("/")[-1]
    return name


def update_config_toml(model_name, config_path):
    """Updates config.toml with the correct token costs for the specified model."""
    model_key = normalize_model_name(model_name)
    if model_key not in MODEL_PRICING:
        raise ValueError(f"Model '{model_key}' not found in MODEL_PRICING. Please update MODEL_PRICING.")

    pricing = MODEL_PRICING[model_key]
    input_cost_per_token = pricing["input"] / 1_000_000
    output_cost_per_token = pricing["output"] / 1_000_000

    config = toml.load(config_path)
    if "llm" not in config:
        config["llm"] = {}

    config["llm"]["input_cost_per_token"] = input_cost_per_token
    config["llm"]["output_cost_per_token"] = output_cost_per_token

    with open(config_path, "w") as f:
        toml.dump(config, f)

    print(f"Updated {config_path}: input_cost_per_token = {input_cost_per_token}, output_cost_per_token = {output_cost_per_token}")


def run_task(cmd, env, task_id, iteration, timeout):
    """Run one task iteration with timeout and logging."""
    start_time = time.time()
    try:
        subprocess.run(cmd, env=env, timeout=timeout)
        elapsed = time.time() - start_time
        return f"TASK_ID={task_id} Iteration {iteration} completed in {elapsed/60:.1f} minutes."
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return f"TASK_ID={task_id} Iteration {iteration} exceeded {timeout/60:.0f} minutes ({elapsed/60:.1f} elapsed). Terminated."

def main():
    parser = argparse.ArgumentParser(description="Run Agent Runner in parallel with time limits.")
    parser.add_argument("--AGENT_ID", type=str, required=True, help="Which agent to run")
    parser.add_argument("--TASK_ID", type=str, nargs="+", required=True, help="List of research task IDs to run")
    parser.add_argument("--LLM_MODEL", type=str, required=True, help="Which LLM model to use")
    parser.add_argument("--RUN_TIMES", type=int, default=1, help="Number of times to run each task")
    parser.add_argument("--MAX_PARALLEL", type=int, default=3, help="Maximum number of parallel processes")

    args = parser.parse_args()

    env_base = os.environ.copy()
    env_base["AGENT_ID"] = args.AGENT_ID
    env_base["LLM_MODEL"] = args.LLM_MODEL

    Main_Path = Path.cwd()
    cmd = [sys.executable, f"{Main_Path}/agents/{args.AGENT_ID}/run.py"]

    TIME_LIMIT = 3600  # 1 hour per run

    print(f"Starting runs: {len(args.TASK_ID)} tasks × {args.RUN_TIMES} iterations "
          f"= {len(args.TASK_ID) * args.RUN_TIMES} total runs (max {args.MAX_PARALLEL} in parallel)\n")

    futures = []
    with ThreadPoolExecutor(max_workers=args.MAX_PARALLEL) as executor:
        for task_id in args.TASK_ID:
            for i in range(args.RUN_TIMES):
                env = env_base.copy()
                env["TASK_ID"] = task_id
                futures.append(executor.submit(run_task, cmd, env, task_id, i + 1, TIME_LIMIT))

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()
