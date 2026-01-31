import subprocess
import os
import time
from pathlib import Path
import shutil

# Repo path
Main_Path = Path.cwd()

def main():
    # Prepare environment variables
    agent_id = os.environ.get("AGENT_ID", "")
    task_id = os.environ.get("TASK_ID", "")
    figure_id = os.environ.get("FIGURE_ID", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "claude-3-5-sonnet-20240620")  # default fallback
    
    # Timestamp for run naming
    timestamp = time.strftime("%Y%m%d%H%M%S")
    run_name = f"claude-code-run-{timestamp}"

    # Ensure run/ directory exists
    run_dir = Main_Path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the sandbox directory
    sandbox_volume_filename = f"{agent_id}_{LLM_MODEL.replace('/','-')}_{timestamp}"
    sandbox_volume_path = run_dir / sandbox_volume_filename

    # Copy experiment setup
    instances_src = (
        Main_Path / f"benchmark/papers/{task_id}/{figure_id}/data"
        if figure_id
        else Main_Path / f"benchmark/papers/{task_id}/data"
    )
    shutil.copytree(instances_src, sandbox_volume_path)
    utils_src = Main_Path / "utils"
    shutil.copytree(utils_src, sandbox_volume_path / "utils")

    # Instruction path
    INSTRUCTION_PATH = Main_Path / "benchmark" / "papers" / task_id / (figure_id if figure_id else "") / "instruction"
    instruction_file = INSTRUCTION_PATH / "instruction.txt"

    # Log file
    log_file = Main_Path / "log" / f"{agent_id}" / f"{LLM_MODEL}" / f"{task_id}" / f"{timestamp}" / "log.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Write metadata to log
    with open(log_file, "w") as f:
        f.write(f"agent_id: {agent_id}\n")
        f.write(f"task_id: {task_id}\n")
        f.write(f"llm_model: {LLM_MODEL}\n")
        f.write("=" * 40 + "\n")

    with open(instruction_file, "r") as f:
        instruction_text = f.read().strip()

    # Write .env into sandbox so utils/llm_inference.py can load API keys
    api_key = os.environ.get("LLM_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    with open(sandbox_volume_path / ".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
        f.write(f"ANTHROPIC_API_KEY={api_key}\n")
        f.write(f"CLAUDE_API_KEY={api_key}\n")
        f.write(f"GOOGLE_API_KEY={google_key}\n")
        f.write(f"HF_TOKEN={hf_token}\n")

    # Build Claude Code CLI command
    cmd = [
        "claude",
        "-p", instruction_text,
        "--model", LLM_MODEL,
        "--output-format", "stream-json",   # trajectory output
        "--verbose",                        # required for stream-json
        "--add-dir", str(sandbox_volume_path),
        "--add-dir", str(sandbox_volume_path / "utils"),
        "--dangerously-skip-permissions"
    ]


    # Run the Claude Code command and log output
    with open(log_file, "a") as f:
        process = subprocess.run(cmd, cwd=sandbox_volume_path, stdout=f, stderr=subprocess.STDOUT)

    print(f"Run complete. Logs saved to {log_file}")

if __name__ == "__main__":
    main()
