import os
import re
import json
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union, Tuple
from dotenv import load_dotenv


query = {
    "llm_racial_bias_in_medicine": "Does the GPT-3.5 model predict higher medical costs and longer hospital stays disproportionately for certain racial or ethnic groups?",
    "lost_in_the_middle": "How does model performance vary based on relevant information position in context?",
    "llms_lack_self_correction": "How do self-correction methods impact large language model performance across math, commonsense reasoning, and multi-hop question answering benchmarks?",
    "awareness_detection": "Can language models detect that a given interaction transcript comes from an evaluation rather than real-world deployment?",
    "cot_without_prompting": "Can large language models, without any chain of thought prompts, reveal reasoning paths and improve answer accuracy by altering its decoding approach?",
    "cot_faithfulness_gaps": "To what extent do reasoning models’ chains-of-thought faithfully reflect their internal reasoning processes when they exploit external hints?",
    "llms_assume_rationality":"Do large language models accurately capture the irrational aspects of human decision￾making, or do they instead assume that people behave according to rational choice theories (such as expected value theory)",
    "to_cot_or_not_to_cot" :"How much performance improvement does Chain-of-Thought (CoT) prompting provide over Direct Answer prompting across different reasoning categories (Commonsense, Knowledge, Symbolic, Mathematical, Soft Reasoning), and which categories show statistically signficant gains?",
    "uncertainty_in_instruction_following":"How effectively can large language models (LLMs) estimate their uncertainty when following user instructions, and how can this ability be systematically evaluated in a controlled setting?",
    "fractal_complexity_of_language":"When and why do LLMs deviate from the narrow fractal parameter range characteristic of natural language, as visualized by Holder and Hurst exponents?",
    "llm_value_consistency":"Do large language models exhibit the same value structure as humans, including the ranking of values and the correlations between values, and how does this depend on the way the model is prompted?",
    "hallucination_snowballing": "How do language model hallucinations propagate and compound over the course of a generation, and what mechanisms cause errors to snowball?",
    "counterfactual_simulatability": "Do natural language explanations provided by language models enable humans to accurately simulate the model's behavior under counterfactual inputs?",
    "premise_order_effects": "Does the order of premises affect the reasoning performance of LLMs, even when the logical content remains the same?",
    "persona_reasoning_biases": "Could persona assignment influence the fundamental reasoning capabilities of an LLM, even when the assigned persona is arguably tangential to the task at hand?",
    "mcq_selection_bias": "Are modern large language models robust in handling multiple choice questions, and if not, what causes their vulnerability, especially regarding their sensitivity to option position changes?",
    "prompt_formatting_sensitivity": "How sensitive are language models to superficial formatting choices in prompts, and do such spurious features significantly impact model performance?",
    "space_time_representations": "Do large language models learn coherent and grounded representations that reflect the real world, such as spatial and temporal representations, rather than just superficial statistics?",
    "llm_confidence_elicitation": "Can LLMs accurately express their uncertainty through black-box approaches, and how effective are various strategies for eliciting calibrated confidence?",
    "icl_from_repetition": "What is the underlying mechanism of in-context learning in LLMs, and how do surface repetitions, particularly token co-occurrence reinforcement, influence ICL?",
    "introspective_learning": "Can large language models acquire knowledge about their own internal behavioral tendencies through introspection, predicting their own behavior more accurately than another model trained on the same data?",
    "fallback_behaviors": "What behaviors do language models exhibit when they are uncertain, and how do these fallback patterns manifest across different models and tasks?",
    "cot_in_planning": "Does chain-of-thought prompting truly enable large language models to learn generalizable algorithmic reasoning abilities, or does it merely rely on highly specific, pattern-matching prompts?",
    "seca_hallucination": "Can semantically equivalent adversarial perturbations to input prompts cause language models to hallucinate or produce inconsistent outputs?",
    "distributive_fairness": "How fair are large language models when making resource allocation decisions across different demographic groups?",
    "lifebench_length_following": "How well do LLMs follow explicit length constraints in their generated outputs?",
    "hallucination_awareness": "Are large language models metacognitively aware of when they are hallucinating or producing unreliable outputs?",
    "questbench": "How well can large language models identify the single minimal clarification question needed to solve underspecified reasoning problems?",
    "persona_with_catch": "Does increasing the amount of LLM-generated persona content systematically worsen population-level simulation fidelity?",
    "activation_control": "Can we efficiently elicit long chain-of-thought reasoning in language models through activation-level interventions?",
}


gt = {
    "llm_racial_bias_in_medicine": "Assessment and plans created by the model showed significant association between demographic attributes and recommendations for more expensive procedures as well as differences in patient perception.",
    "lost_in_the_middle": "Models are better at using relevant information that occurs at the very beginning or end of its input context, and performance degrades significantly when models must access and use information located in the middle of its input context.",
    "llms_lack_self_correction": "After self-correction, the accuracies of all models drop or remain nearly the same across all three benchmarks.",
    "awareness_detection": "Language models can distinguish evaluation from real-world transcripts.",
    "cot_faithfulness_gaps": "Reasoning models rarely verbalize when they are using hints in their reasoning. CoTs do not faithfully reflect the internal reasoning that led to the model’s final answer.",
    "cot_without_prompting": "Large language models already contain latent CoT reasoning paths that can be surfaced without any prompting. By branching on alternative top-k first tokens and then choosing the path with the highest answer-confidence margin, these hidden reasoning trajectories can be exposed and final-answer accuracy increases.",
    "to_cot_or_not_to_cot": "Chain-of-Thought achieves performance gain on math and formal logic. CoT does not achieve statistically significant performance gain, sometimes even has slight loss, in other and most fields.",
    "llms_assume_rationality": "LLMs struggle to predict or simulate human behavior in a classic risky choice setting, assuming that people make decisions more rationally than we actually do. LLMs also assume people act rationally when reasoning backwards from observed actions to internal utilities, aligning with how humans make inferences about others' choices",
    "uncertainty_in_instruction_following": "Verbalized self-evaluation methods outperform logit-based approaches in Controlled-Easy tasks, while internal model states provide more reliable uncertainty signals in both Controlled-Easy and Realistic settings. All methods struggle with more complex tasks in ControlledHard",
    "fractal_complexity_of_language": "Various strategies, such as the decoding temperature and prompting method, can impact fractal parameters even when log-perplexity scores seem to be unaffected.\n For pretrained models, larger architectures are more effective at capturing such fractal properties.\n With instruction-tuned models, the similarity to human language does not improve monotonically as the amount of contextual information in the prompt increases.\n The Hurst parameter emerged as a strong predictor of quality in generated texts, among other significant findings.",
    "llm_value_consistency": "Using the Basic prompt, LLM's answers show variance across different generated personas, and show internally consistent outputs. These results suggest that LLMs cannot be treated as individuals holding a coherent set of value priorities. Prompts that endow the LLM with a personality improved the consistency of each specific value profile. However, with targeted prompt, LLM can be guided to display corresponding persona.",
    "hallucination_snowballing": "LMs often commit to an initial answer within the first token and then produce further incorrect explanatory claims that the model can separately recognize as wrong. When presented with incorrect explanations in isolation, models frequently identify the mistakes, showing they possess the knowledge but over-commit to early hallucinations. This snowballing phenomenon persists under higher temperature sampling, beam search, and zero-shot chain-of-thought prompting.",
    "counterfactual_simulatability": "LLM explanations exhibit low counterfactual simulatability: they fail to enable accurate prediction of the model's behavior on diverse counterfactual inputs. Explanation precision does not correlate with plausibility, implying that optimizing for human approval via RLHF may not ensure faithful or informative explanations.",
    "premise_order_effects": "LLM reasoning is highly sensitive to the ordering of premises. Models perform best when premises are arranged to match the ground-truth proof order, and random or permuted orderings can reduce accuracy by over 30 percentage points. This sensitivity persists across model sizes and architectures.",
    "persona_reasoning_biases": "Persona assignment surfaces implicit reasoning biases and can substantially reduce reasoning performance across diverse tasks. ChatGPT-3.5 showed pervasive persona-induced bias affecting 80% of personas, while GPT-4-Turbo exhibited less but still problematic bias at 42% of personas. Simple de-biasing prompts had minimal effect.",
    "mcq_selection_bias": "Modern LLMs exhibit a strong selection bias in MCQs, preferring certain option IDs (e.g., A) due to token-level prior probabilities rather than position-order preference. Accuracy shifts dramatically when the correct answer's option position is moved. The PriDe debiasing method, which estimates and subtracts the option-ID prior, can effectively mitigate this bias.",
    "prompt_formatting_sensitivity": "Small changes in prompt formatting can cause very large performance swings in few-shot settings, up to 76 accuracy points for LLaMA-2-13B. Sensitivity persists across larger model sizes, more few-shot examples, and instruction tuning. Format-specific performance correlates only weakly across models, so comparing models using a single arbitrary prompt format is unreliable.",
    "space_time_representations": "LLMs learn linear representations of space and time across multiple scales. These spatial and temporal embeddings are robust to prompting variations and unified across entity types. Individual space neurons and time neurons can be identified that encode geographic coordinates and temporal information.",
    "llm_confidence_elicitation": "LLMs tend to be overconfident when verbalizing their confidence, potentially imitating human patterns of expressing confidence. Both calibration and failure prediction improve with model capability but remain far from ideal. Human-inspired prompting strategies mitigate overconfidence with diminishing returns for advanced models. Sampling strategies paired with specific aggregators can enhance failure prediction.",
    "icl_from_repetition": "In-context learning is substantially driven by token co-occurrence reinforcement: repeated contextual co-occurrences in the demonstration examples strengthen token relationships and drive ICL behavior. This surface repetition mechanism explains both the beneficial functions and detrimental effects of ICL, including cases where spurious correlations in demonstrations mislead the model.",
    "introspective_learning": "LLMs can exhibit a form of introspection: a model predicts its own behavior in hypothetical scenarios more accurately than a different model trained on the same ground-truth behavioral data. This privileged self-prediction holds on simple tasks and survives intentional modifications to ground-truth behavior, but fails to generalize to more complex or out-of-distribution tasks.",
    "fallback_behaviors": "Language models exhibit a consistent ordering of fallback behaviors under uncertainty: as models become more advanced, they shift from sequence repetitions to degenerate text to hallucinations. The same ordering appears within single-generation trajectories as uncertainty increases. Common decoding strategies like random sampling reduce obvious failures such as repetitions but increase harder-to-detect hallucinations.",
    "cot_in_planning": "Chain-of-thought prompting does not reliably enable generalizable algorithmic reasoning in planning tasks. Performance depends heavily on prompt specificity and degrades when problem complexity increases beyond patterns seen in demonstrations. CoT provides superficial benefits from pattern matching rather than true algorithmic understanding.",
    "seca_hallucination": "Semantically equivalent and coherent adversarial prompt perturbations can reliably elicit hallucinations in both open-source and commercial LLMs. SECA achieves higher attack success rates while maintaining semantic equivalence and coherence constraints, highlighting the sensitivity of LLMs to plausible prompt variations even when meaning is preserved.",
    "distributive_fairness": "LLMs are poorly aligned with human distributional fairness preferences. They struggle to use transferable resources like money to reduce inequality, are sensitive to prompt and template changes, but perform better when selecting from predefined menus rather than generating allocations freely.",
    "lifebench_length_following": "Most models follow short-length instructions reasonably but deteriorate sharply beyond a certain threshold. Almost all models fail to reach vendor-claimed maximum output lengths in practice. Long-context LLMs do not reliably improve length-instruction following despite extended input-output windows. Reasoning models outperform even specialized long-text generation models at length following.",
    "hallucination_awareness": "LLMs carry internal signals about truthfulness and confidence, but often fail to express that knowledge when generating answers. Metacognitive signals have limited resolution, emerge contextually, differ across models, and detectors often fail to generalize. Models may encode correct answers internally yet still output incorrect ones.",
    "questbench": "LLMs struggle to identify the single minimal clarification question needed to solve underspecified reasoning problems. Performance varies substantially across algebra, logic, and planning tasks, and degrades as problem complexity increases. Models often fail to recognize what information is missing to disambiguate the problem.",
    "persona_with_catch": "Increasing the amount of LLM-generated persona content systematically worsens population-level simulation fidelity. While richer persona descriptions may appear more detailed, they introduce artifacts and biases that reduce the accuracy of simulating real-world population distributions in opinion surveys and election predictions.",
    "activation_control": "A small set of high-impact activations in the last few layers of LLMs largely governs long-form CoT attributes such as output length and self-reflection. Amplifying these activations and inserting wait tokens can invoke long CoT reasoning without training, significantly increasing self-reflection rates and accuracy. Activation dynamics follow predictable trajectories with a sharp rise after special tokens and exponential decay.",
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