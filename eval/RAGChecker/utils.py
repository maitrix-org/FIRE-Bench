import os
import re
import json
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union, Tuple
from dotenv import load_dotenv


query = {
    "medical_bias": "Does the GPT-3.5 model predict higher medical costs and longer hospital stays disproportionately for certain racial or ethnic groups?",
    "lost_in_the_middle": "How does model performance vary based on relevant information position in context?",
    "self_correction": "How do self-correction methods impact large language model performance across math, commonsense reasoning, and multi-hop question answering benchmarks?",
    "known_evaluation": "Can language models detect that a given interaction transcript comes from an evaluation rather than real-world deployment?",
    "cot_decoding": "Can large language models, without any chain of thought prompts, reveal reasoning paths and improve answer accuracy by altering its decoding approach?",
    "reasoning_model": "To what extent do reasoning models’ chains-of-thought faithfully reflect their internal reasoning processes when they exploit external hints?",
    "rational":"Do large language models accurately capture the irrational aspects of human decision￾making, or do they instead assume that people behave according to rational choice theories (such as expected value theory)",
    "to_cot_or_not" :"How much performance improvement does Chain-of-Thought (CoT) prompting provide over Direct Answer prompting across different reasoning categories (Commonsense, Knowledge, Symbolic, Mathematical, Soft Reasoning), and which categories show statistically signficant gains?",
    "uncertainty_instruction_following":"How effectively can large language models (LLMs) estimate their uncertainty when following user instructions, and how can this ability be systematically evaluated in a controlled setting?",
    "fractal_complexities":"When and why do LLMs deviate from the narrow fractal parameter range characteristic of natural language, as visualized by Holder and Hurst exponents?",
    "consistent_values":"Do large language models exhibit the same value structure as humans, including the ranking of values and the correlations between values, and how does this depend on the way the model is prompted?"
}


gt = {
    "medical_bias": "Assessment and plans created by the model showed significant association between demographic attributes and recommendations for more expensive procedures as well as differences in patient perception.",
    "lost_in_the_middle": "Models are better at using relevant information that occurs at the very beginning or end of its input context, and performance degrades significantly when models must access and use information located in the middle of its input context.",
    "self_correction": "After self-correction, the accuracies of all models drop or remain nearly the same across all three benchmarks.",
    "known_evaluation": "Language models can distinguish evaluation from real-world transcripts.",
    "reasoning_model": "Reasoning models rarely verbalize when they are using hints in their reasoning. CoTs do not faithfully reflect the internal reasoning that led to the model’s final answer.",
    "cot_decoding": "Large language models already contain latent CoT reasoning paths that can be surfaced without any prompting. By branching on alternative top-k first tokens and then choosing the path with the highest answer-confidence margin, these hidden reasoning trajectories can be exposed and final-answer accuracy increases.",
    "to_cot_or_not": "Chain-of-Thought achieves performance gain on math and formal logic. CoT does not achieve statistically significant performance gain, sometimes even has slight loss, in other and most fields.",
    "rational": "LLMs struggle to predict or simulate human behavior in a classic risky choice setting, assuming that people make decisions more rationally than we actually do. LLMs also assume people act rationally when reasoning backwards from observed actions to internal utilities, aligning with how humans make inferences about others' choices",
    "uncertainty_instruction_following": "Verbalized self-evaluation methods outperform logit-based approaches in Controlled-Easy tasks, while internal model states provide more reliable uncertainty signals in both Controlled-Easy and Realistic settings. All methods struggle with more complex tasks in ControlledHard",
    "fractal_complexities": "Various strategies, such as the decoding temperature and prompting method, can impact fractal parameters even when log-perplexity scores seem to be unaffected.\n For pretrained models, larger architectures are more effective at capturing such fractal properties.\n With instruction-tuned models, the similarity to human language does not improve monotonically as the amount of contextual information in the prompt increases.\n The Hurst parameter emerged as a strong predictor of quality in generated texts, among other significant findings.",
    "consistent_values": "Using the Basic prompt, LLM's answers show variance across different generated personas, and show internally consistent outputs. These results suggest that LLMs cannot be treated as individuals holding a coherent set of value priorities. Prompts that endow the LLM with a personality improved the consistency of each specific value profile. However, with targeted prompt, LLM can be guided to display corresponding persona."
}


def get_latest_log_file(base_directory):
    """
    Finds the most recently modified folder under the given base directory
    and returns the full path to the 'log.log' file inside it.

    Parameters:
        base_directory (str): The path to the directory containing log folders.

    Returns:
        str or None: Full path to 'log.log' in the latest folder, or None if not found.
    """
    # Get list of subdirectories with their modification times
    try:
        subdirs = [
            os.path.join(base_directory, d)
            for d in os.listdir(base_directory)
            if os.path.isdir(os.path.join(base_directory, d))
        ]
    except FileNotFoundError:
        print(f"Directory not found: {base_directory}")
        return None

    if not subdirs:
        print("No subdirectories found.")
        return None

    # Sort subdirectories by last modified time, descending
    latest_subdir = max(subdirs, key=os.path.getmtime)

    # Construct the path to log.log inside the latest directory
    log_file_path = os.path.join(latest_subdir, "log.log")

    if os.path.isfile(log_file_path):
        return log_file_path
    else:
        print(f"'log.log' not found in {latest_subdir}")
        return None


def read_log_metadata(log_file_path):
    """
    Reads the agent_id, task_id, and llm_model from the top of the given log file.
    Returns a dictionary with those values.
    """
    metadata = {}
    with open(log_file_path, "r") as f:
        for line in range(3):
            entry = f.readline()
            if not entry:
                break
            key, value = entry.strip().split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata
    

def extract_json_from_markdown(response: str) -> str:
    if response and response.strip().startswith('```') and '```' in response:
        code_content = response.split('```', 2)[1]
        if code_content.startswith('json'):
            code_content = code_content[4:].strip()
        response = code_content.strip()
    return response


def parse_json_response(response: str, default_value: Any = None) -> Any:
    if not response:
        return default_value
    
    response = extract_json_from_markdown(response)
    
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        print(f"Response was: {response[:100]}..." if len(response) > 100 else f"Response was: {response}")
        return default_value


def parse_gpt_response(response: str, expected_fields: List[str] = None, field_defaults: Dict[str, Any] = None) -> Dict[str, Any]:
    if field_defaults is None:
        field_defaults = {}
    
    result = parse_json_response(response, {})
    
    if not expected_fields:
        return result
    
    output = {}
    for field in expected_fields:
        output[field] = result.get(field, field_defaults.get(field))
    
    return output


def parse_nested_json_response(response: str) -> Tuple[Dict[str, Any], bool]:
    extracted = extract_json_from_markdown(response)
    
    try:
        result = json.loads(extracted)
        
        if isinstance(result, dict) and len(result) == 1 and next(iter(result.values())).startswith('{'):
            key = next(iter(result.keys()))
            try:
                nested_json = json.loads(result[key])
                return nested_json, True
            except json.JSONDecodeError:
                pass
        
        return result, True
    except json.JSONDecodeError as e:
        print(f"\nWarning: Failed to parse JSON response: {e}")
        return {}, False


def extract_core_idea(conclusion, client, eval_model):
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert summarizer. "
                    "Given a conclusion, your task is to extract only the core insight or main idea, "
                    "omitting all concrete values, specific numbers, background details, methods, "
                    "file names, or references to artifacts. "
                    "Focus on general trends or main conclusions/findings, and express them in a "
                    "generalized way without referring to any precise data or background context. "
                    "DO NOT infer any detail, context information, or background knowledge that is not mentioned "
                    "in the original conclusion."
                ),
            },
            {"role": "user", "content": f"Input: {conclusion}"}
        ]
    )
    return response.choices[0].message.content


import re
import json

def extract_single_final_thought(log_path):
    final_thought = ""
    paper = ""

    # Regex for OpenHands logs
    import re

    openhands_pattern = re.compile(
        r"final_thought\s*=\s*(?:'|\")(.+?)(?:'|\"),\s*outputs=",
        re.DOTALL
    )

    # Timestamp pattern for Codex-like logs (e.g., [2025-09-19T10:05:41])
    codex_timestamp_pattern = re.compile(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\]")

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        extracted = None

        # --- 1. OpenHands format (last occurrence) ---
        matches = openhands_pattern.findall(content)
        if matches:
            extracted = matches[-1].encode("utf-8").decode("unicode_escape").strip()
            print(extracted)

        # --- 2. Codex format (between third-last and second-last timestamp) ---
        if extracted is None:
            timestamps = list(codex_timestamp_pattern.finditer(content))
            if len(timestamps) >= 3:
                start = timestamps[-3].start()
                end = timestamps[-1].start()
                extracted = content[start:end].strip()

        # --- 3. Claude format (JSON in last line) ---
        if extracted is None:
            last_line = content.strip().splitlines()[-1]
            try:
                parsed = json.loads(last_line)
                if isinstance(parsed, dict) and "result" in parsed:
                    extracted = parsed["result"].strip()
            except json.JSONDecodeError:
                pass

    except Exception as e:
        print(f"Error reading file: {e}")

    return extracted