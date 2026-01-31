import subprocess
import os
import time
import random
from pathlib import Path
import shutil


# Repo path
Main_Path = Path.cwd()

def main():

    runtime_image = "docker.all-hands.dev/all-hands-ai/runtime:0.55-nikolaik"
    openhands_image = "docker.all-hands.dev/all-hands-ai/openhands:0.55"
    
    # Pull the latest runtime image first
    subprocess.run(["docker", "pull", runtime_image], check=True)

    # Prepare environment variables
    agent_id = os.environ.get("AGENT_ID", "")
    task_id = os.environ.get("TASK_ID", "")
    figure_id = os.environ.get("FIGURE_ID", "")
    LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
    LLM_MODEL = os.environ.get("LLM_MODEL", "")

    # Get current user ID
    user_id = subprocess.check_output(["id", "-u"]).decode().strip()
    
    # A random suffix
    rd = random.randint(10000, 99999)

    # Timestamp for container name and filename
    timestamp = time.strftime("%Y%m%d%H%M%S")
    container_name = f"openhands-app-{timestamp}-{rd}"

    # Ensure run/ directory exists
    run_dir = Main_Path / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build the sandbox volume filename and path
    if "/" in LLM_MODEL:
        sandbox_volume_filename = f"{agent_id}_{LLM_MODEL.split('/')[1]}_{timestamp}"
    else:
        sandbox_volume_filename = f"{agent_id}_{LLM_MODEL}_{timestamp}"
    print(agent_id, LLM_MODEL, timestamp)
    sandbox_volume_path = run_dir / f"{sandbox_volume_filename}_{rd}"

    # Copy experiment setup
    instances_src = Main_Path / f"benchmark/papers/{task_id}/{figure_id}/data" if figure_id != "" else Main_Path / f"benchmark/papers/{task_id}/data"
    shutil.copytree(instances_src, sandbox_volume_path)
    utils_src = Main_Path / "utils"
    shutil.copytree(utils_src, sandbox_volume_path / "utils")

    # Write .env into sandbox so utils/llm_inference.py can load API keys
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    with open(sandbox_volume_path / ".env", "w") as f:
        f.write(f"OPENAI_API_KEY={LLM_API_KEY}\n")
        f.write(f"ANTHROPIC_API_KEY={LLM_API_KEY}\n")
        f.write(f"CLAUDE_API_KEY={LLM_API_KEY}\n")
        f.write(f"GOOGLE_API_KEY={google_key}\n")
        f.write(f"HF_TOKEN={hf_token}\n")

    # Set the SANDBOX_VOLUMES environment variable to this path
    SANDBOX_VOLUMES = str(sandbox_volume_path)

    # Instruction path
    INSTRUCTION_PATH = Main_Path / "benchmark" / "papers" / task_id / figure_id / "instruction" if figure_id != "" else Main_Path / "benchmark" / "papers" / task_id / "instruction"

    # log file
    log_file = f"openhands_run_{timestamp}.log"

    # Build the docker command as a list for subprocess
    cmd = [
        "docker", "run", "-it", "--rm",
        "--pull=always",
        "-e", f"SANDBOX_RUNTIME_CONTAINER_IMAGE={runtime_image}",
        "-e", f"SANDBOX_USER_ID={user_id}",
        "-e", f"SANDBOX_VOLUMES={SANDBOX_VOLUMES}:/workspace:rw",
        "-e", f"LLM_API_KEY={LLM_API_KEY}",
        "-e", f"LLM_MODEL={LLM_MODEL}",
        "-e", "SANDBOX_ENABLE_GPU=true",
        "-e", "LOG_ALL_EVENTS=true",
        # "-e", "SANDBOX_TIMEOUT",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{Main_Path}/.openhands:/.openhands",
        "-v", f"{INSTRUCTION_PATH}:/instruction",
        "--add-host", "host.docker.internal:host-gateway",
        # "-v", f"{Main_Path}/agents/openhands/config.toml:/config.toml",
        "--name", container_name,
        openhands_image,
        "python", "-m", "openhands.core.main", "-f", "/instruction/instruction.txt"
    ]

    log_file = Main_Path / "log" / f"{agent_id}" / f"{LLM_MODEL}" / f"{task_id}" / f"{timestamp}_{rd}" / "log.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"agent_id: {agent_id}\n")
        f.write(f"task_id: {task_id}\n")
        f.write(f"llm_model: {LLM_MODEL}\n")
        f.write("=" * 40 + "\n")
        
    # Run the docker command and log execution
    with open(log_file, "w") as f:
        process = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"Run complete. Logs saved to {log_file}")


if __name__ == "__main__":
    main()
