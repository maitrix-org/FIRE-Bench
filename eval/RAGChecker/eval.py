from utils import *
import argparse
from refchecker import LLMExtractor, LLMChecker
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics
from pathlib import Path
import datetime
import re


from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
eval_model = "gpt-5.2"
base_dir = "log"

def evaluate_single_prompt_pc(claim, ground_truth):
    judge_prompt = """

        You are a neutral judge asked to determine if a claim is part of the given ground truth. 
        The claim does not need to mention every aspect of the ground truth. 
        If it overlaps with part of the ground truth, return True. 
        If it conflicts with or cannot be inferred from the ground truth, return False.

        =================  INPUT  =================
        [1]  CLAIM:
        {claim}

        [2]  GROUNG TRUTH:
        {ground_truth}
        ===========================================

        **Output format (JSON)**
        ```json
        {{
        "rationale": "<2-3 sentence explanation>",
        "result": <true/false>,
        }}
    """

    _judge_prompt = judge_prompt.format(claim=claim, ground_truth=ground_truth)

    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an impartial judge engine. "
                    "Follow the instructions exactly and output *only* the JSON."
                ),
            },
            {"role": "user", "content": _judge_prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    result = parse_gpt_response(response.choices[0].message.content)
    return result


def evaluate_single_prompt_rc(claim, conclusion):
    judge_prompt = """

        You are a neutral judge asked to determine if a claim is part of the given conclusion. 
        The claim does not need to mention every aspect of the conclusion. 
        If it overlaps with part of the conclusion, return True. 
        If it conflicts with or cannot be inferred from the conclusion, return False.

        =================  INPUT  =================
        [1]  CLAIM:
        {claim}

        [2]  CONCLUSION:
        {conclusion}
        ===========================================

        **Output format (JSON)**
        ```json
        {{
        "rationale": "<2-3 sentence explanation>",
        "result": <true/false>,
        }}
    """

    _judge_prompt = judge_prompt.format(claim=claim, conclusion=conclusion)

    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an impartial judge engine. "
                    "Follow the instructions exactly and output *only* the JSON."
                ),
            },
            {"role": "user", "content": _judge_prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    result = parse_gpt_response(response.choices[0].message.content)
    return result


def decompose_conclusion(conclusion):
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a claim, break it down into a list of meaningful and reasonable subclaims. "
                    "Each subclaim should be a standalone and specific statement. "
                    "DO NOT infer any detail, context information, or background knowledge that is not mentioned in the original conclusion."
                    "Return the result in the following JSON format: "

                    """{{
                    "statement": "<The original statement>",
                    "subclaims": [[
                        "<First subclaim>",
                        "<Second subclaim>",
                        ...
                    ]]
                    }}
                    """
                ),
            },
            {"role": "user", "content": f"Input statement: {conclusion}"}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    return parse_gpt_response(response.choices[0].message.content)


def extract_llm_conclusion_openhands(log_file):
    with open(log_file, 'r', encoding='utf-8') as file:
        log_data = file.read()
    match = re.search(r"final_thought='(.*?)', task_completed=", log_data, re.DOTALL)
    if match:
        final_thought = match.group(1)
        return final_thought
    else:
        raise ValueError("No final_thought field found in log.")
    

def eval_precision(raw_conclusion, ground_truth):
    precision = 0
    cnt = 0

    core_idea = extract_core_idea(raw_conclusion)
    print('core idea:', core_idea, '\n')
    decomposed_subclaims = decompose_conclusion(core_idea)
    print('decomposed subclaims:', decomposed_subclaims, '\n')

    for sc in decomposed_subclaims['subclaims']:
        res = evaluate_single_prompt_pc(sc, ground_truth)
        print(res)
        cnt += res['result']
    
    precision = cnt / len(decomposed_subclaims['subclaims'])
    print('\n', 'precision:', precision)
    return precision


def eval_recall(raw_conclusion, ground_truth):
    recall = 0
    cnt = 0

    core_idea = extract_core_idea(raw_conclusion)
    print('core idea:', core_idea, '\n')
    decomposed_subclaims = decompose_conclusion(ground_truth)
    print('decomposed ground truth:', decomposed_subclaims, '\n')

    for sc in decomposed_subclaims['subclaims']:
        res = evaluate_single_prompt_rc(sc, core_idea)
        print(res)
        cnt += res['result']
    
    recall = cnt / len(decomposed_subclaims['subclaims'])
    
    print('\n', 'recall:', recall)
    return recall


def eval_single_log(log_path, agent, model, task, timestamp, client, eval_model):
    base_dir = Path.cwd()
    log_path = base_dir / f"{log_path}"
    if not os.path.exists(log_path):
        print(f"Error: The path '{log_path}' does not exist.")
        return
    print(f"Processing logs from: {log_path}")

    raw_conclusion = extract_single_final_thought(log_path)
    core_idea = extract_core_idea(raw_conclusion, client, eval_model)
    
    gt_vs_response = {
        "results": [
            {
                "query_id": "000",
                "query": f"{query[task]}",
                "gt_answer": f"{gt[task]}",
                "response": f"{core_idea}",
                "retrieved_context": []
            }
        ]
    }

    evaluator = RAGChecker(
        extractor_name="openai/gpt-4.1",
        checker_name="openai/gpt-4.1",
        batch_size_extractor=8,  
        batch_size_checker=8
    )
    rag_results = RAGResults.from_json(json.dumps(gt_vs_response))
    evaluator.evaluate(rag_results, all_metrics)

    return rag_results


def eval_all_logs(base_dir, agents, models, tasks, timestamps, client, eval_model):
    print(f"[*] Evaluating logs for agents={agents}, models={models}, tasks={tasks}")
    results = []

    # Expand "all" dynamically into directory lists
    agents_list = os.listdir(base_dir) if "all" in agents else agents
    log_paths = []
    for agent in agents_list:
        agent_path = os.path.join(base_dir, agent)
        if not os.path.isdir(agent_path):
            print(f"Agent Error: The path '{agent_path}' does not exist.")
            continue

        models_list = os.listdir(agent_path) if "all" in models else models
        for model in models_list:
            model_path = os.path.join(agent_path, model)
            if not os.path.isdir(model_path):
                print(f"Model Error: The path '{model_path}' does not exist.")
                continue

            task_list = os.listdir(model_path) if "all" in tasks else tasks
            for task in task_list:
                task_path = os.path.join(model_path, task)
                if not os.path.isdir(task_path):
                    print(f"Task Error: The path '{task_path}' does not exist.")
                    continue

                timestamp_list = os.listdir(task_path) if "all" in timestamps else timestamps
                for timestamp in timestamp_list:
                    log_file = os.path.join(task_path, timestamp, "log.log")
                    print(f"[>] Processing log: {log_file}")
                    res = eval_single_log(log_file, agent, model, task, timestamp, client, eval_model)

                    if res is not None:
                        results.append(res)
                        log_paths.append(log_file)

    print(f"[*] Completed evaluation of {len(results)} log(s).")
    print(log_paths)
    return results, log_paths


def dump_results_to_json(results, output_dir="results", prefix="eval_results"):
    """Save evaluation results to a timestamped JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n[✓] Results saved to: {output_path}")
    return output_path


import tempfile
import shutil


error_analysis_prompt_fp = """
You are an error analysis expert.

You have access to two attached files:
1. The original paper of research question "{query}".
2. The logged trajectory of an AI agent doing research about the same question.

The correct conclusion found by human researchers is: "{gt}".
And the {fp_or_fn} conclusion generated by an AI research agent: "{f_statment}".

Based on the original research of human researchers and the logged trajectory of AI research agent, what error did the agent make so it get the {fp_or_fn} conclusion?
You should first take a close look at the original paper and the logged trajectory. Then, follow the taxonomy below carefully follow the instructions and provide the output in the same format as the example.

# Taxonomy
├── Contradictory Conclusion
├── Unrelated Conclusion
├── Overgeneralized Conclusion (Draw conclusion that is too broad)
└── Alternative Conclusion (The approach of the agent is different from the original one but it is plausible, and the conclusion generated by the agent is another possible answer)

- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it. 
- You must be exhaustive and find all the errors in the trace. Only include the final subcategories of the taxonomy (i.e. "Resource Not Found" and not "API Issues" or "System Execution Errors").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).

**Output Format**
```json
{{
  "query": "<the original research question>",
  "false_positive_conclusion": "<the false positive conclusion of agent>",
  "correct_conclusion": "<the correct conclusion found by human researchers>",
  "error_type": "<one of the error categories>",
  "evidence": "<detailed explanation of why this error type fits the agent’s behavior>"
}}

"""

error_analysis_prompt_fn = """
You are an error analysis expert.

You have access to two attached files:
1. The original paper of research question "{query}".
2. The logged trajectory of an AI agent doing research about the same question.

The correct conclusion found by human researchers is: "{gt}".
And the {fp_or_fn} conclusion missed by AI research agent: "{f_statment}".

Based on the original research of human researchers and the logged trajectory of AI research agent, what error did the agent make so it get the {fp_or_fn} conclusion?
Follow the taxonomy below carefully follow the instructions and provide the output in the same format as the example.

# Taxonomy
├── Research Planning
│   ├── Method Deviation (Agents use a different method from the original one used by human researcher)
│   └── Goal Deviation (Agents deviate from the given research question and plan to answer a different one)
├── System Errors
│   ├── Environment Setup Errors (Includes permission problems and inability to access resources or API keys)
│   ├── API Call Issues
│   ├── Policy Violation
│   ├── Timeout Issues
│   └── Other System Errors (Other internal errors of the agent system)
├── Execution Errors
│   ├── Laziness (Agents do not conduct full experiment but runs with only very few samples)
│   ├── Endless loop (Agents fail to end their actions; often repeatedly attempting to conclude or launching unnecessary additional experiments)
│   └── Premature termination (Agents do not run the experiment but end their action after completing the scripts or experiment plan)
├── Implementation Errors
│   └── Unsound Implementation (Agents fail to complete a reasonable implementation e.g. No normalization or no extraction of final answer which leads to 0 accuracy across all datasets)
├── Analysis & Conclusion
│   └── Analysis Failure (Agents follow the exact same step as the original paper and run the correct experiment but fail to draw the correct conclusion from the experiment data, e.g. fail to notice a trend in the data; you should first check for the research planning stage error and then the analysis failure)

- Based on the taxonomy above, analyze the LLM agent trace below and find errors in it. 
- Only include the final subcategories of the taxonomy (i.e. "Method Deviation", "Environment Setup Errors" or "Laziness").
- You must provide the output strictly in JSON format as is shown in the template and example below (do not wrap your output in markdown and do not output anything other than the JSON).

**Output Format**
```json
{{
  "query": "<the original research question>",
  "false_negative_conclusion": "<the false negative conclusion of agent>",
  "correct_conclusion": "<the correct conclusion found by human researchers>",
  "error_type": "<one of the error categories>",
  "evidence": "<detailed explanation of why this error type fits the agent’s behavior>"
}}

"""


def upload_files(paper_path, log_path):
    paper_file = client.files.create(file=open(paper_path, "rb"), purpose="assistants")
    # ensure tmp file is closed before uploading
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = tmp.name
    shutil.copyfile(log_path, tmp_path)
    log_file = client.files.create(file=open(tmp_path, "rb"), purpose="assistants")
    return paper_file.id, log_file.id


def extract_claims(text: str):
    # Pattern to capture claims under the two categories
    fp_pattern = r"False positive atomized claims:\s*((?:\s*-\s.*\n)+)"
    fn_pattern = r"False negative atomized claims:\s*((?:\s*-\s.*\n)+)"

    def clean_claims(match):
        if not match:
            return []
        claims_block = match.group(1)
        # Split lines, strip dashes and whitespace, remove '|'
        claims = [
            line.strip("- ").replace("| ", "").strip()
            for line in claims_block.strip().splitlines()
        ]
        return claims

    false_positives = clean_claims(re.search(fp_pattern, text))
    false_negatives = clean_claims(re.search(fn_pattern, text))

    return false_positives, false_negatives


def upload_files_to_vector_store(paper_path, log_path):
    # Upload files (purpose="assistants" is still fine for vector stores in current SDK examples)
    paper_file = client.files.create(file=open(paper_path, "rb"), purpose="assistants")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp_path = tmp.name
    shutil.copyfile(log_path, tmp_path)
    log_file = client.files.create(file=open(tmp_path, "rb"), purpose="assistants")

    # Create a short-lived vector store so you don’t pay storage forever
    vector_store = client.vector_stores.create(
        name=f"tmp-ragchecker-{paper_file.id}",
        expires_after={"anchor": "last_active_at", "days": 1},
    )  # create pattern shown in docs/examples :contentReference[oaicite:2]{index=2}

    # Add both files to the store and wait for indexing
    client.vector_stores.files.create(vector_store.id, file_id=paper_file.id)
    client.vector_stores.files.create(vector_store.id, file_id=log_file.id)
    client.vector_stores.files.poll(paper_file.id, vector_store_id=vector_store.id)
    client.vector_stores.files.poll(log_file.id,   vector_store_id=vector_store.id)
    # add/poll pattern shown in examples :contentReference[oaicite:3]{index=3}

    return vector_store.id


def analyze_claims(fp_claims, fn_claims, gt, query, paper_path, log_path, paper_name):
    vs_id = upload_files_to_vector_store(paper_path, log_path)

    def run_with_file_search(prompt: str):
        resp = client.responses.create(
            model=eval_model,   # "gpt-5.1" is OK here
            input=prompt,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vs_id],
                "max_num_results": 5,
            }],
        )
        return resp.output_text
        # Responses+file_search usage with vector_store_ids :contentReference[oaicite:4]{index=4}

    fp_results, fn_results = [], []

    for fp in fp_claims:
        prompt = error_analysis_prompt_fp.format(
            gt=gt, fp_or_fn="false positive", f_statment=fp, query=query
        )
        fp_results.append(parse_gpt_response(run_with_file_search(prompt)))

    for fn in fn_claims:
        prompt = error_analysis_prompt_fn.format(
            gt=gt, fp_or_fn="false negative", f_statment=fn, query=query
        )
        fn_results.append(parse_gpt_response(run_with_file_search(prompt)))

    return fp_results, fn_results



def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more log files.")
    parser.add_argument("--agents", nargs="+", default=["all"], help="List of agents or 'all'")
    parser.add_argument("--models", nargs="+", default=["all"], help="List of models or 'all'")
    parser.add_argument("--tasks", nargs="+", default=["all"], help="List of tasks or 'all'")
    parser.add_argument("--timestamp", nargs="+", default=["all"], help="Optional timestamp for a single run")

    args = parser.parse_args()

    results, log_paths = eval_all_logs(base_dir, args.agents, args.models, args.tasks, args.timestamp, client, eval_model)
    eval_stats = [json.loads(r.to_json()) for r in results]
    error_eval_stats = [repr(r) for r in results]
    dump_results_to_json(error_eval_stats)
    print("\n=== Evaluation Complete ===")

    do_error_analysis = True

    if do_error_analysis:
        error_analysis_outputs = []
        for eval_stat, error_eval_stat, log_path in zip(eval_stats, error_eval_stats, log_paths):
            fp_claims, fn_claims = extract_claims(error_eval_stat)
            print("False Positives:", fp_claims)
            print("False Negatives:", fn_claims)

            task = log_path.split("/")[3] if "openai" not in log_path else log_path.split("/")[4]
            fp_results, fn_results = analyze_claims(
                fp_claims, fn_claims, eval_stat["results"][0].get("gt_answer"), eval_stat["results"][0].get("query"),
                f"./eval/RAGChecker/papers/{task}.pdf",  # fixed folder
                log_path,
                task                     # pass in
            )


            from collections import Counter

            fp_stats = Counter()
            for res in fp_results:
                if isinstance(res, dict) and "error_type" in res:
                    fp_stats[res["error_type"]] += 1
                else:
                    fp_stats["unparsed_or_missing_error_type"] += 1

            # FN stats
            fn_stats = Counter()
            for res in fn_results:
                if isinstance(res, dict) and "error_type" in res:
                    fn_stats[res["error_type"]] += 1
                else:
                    fn_stats["unparsed_or_missing_error_type"] += 1

            # still optional to print a little
            print("fp stats", fp_stats)
            print("fn stats", fn_stats)

            # collect everything for saving
            error_analysis_outputs.append({
                "log_path": log_path,
                "task": task,
                "query": eval_stat["results"][0].get("query"),
                "gt_answer": eval_stat["results"][0].get("gt_answer"),
                "false_positives": fp_claims,
                "false_negatives": fn_claims,
                "fp_results": fp_results,
                "fn_results": fn_results,
                "fp_stats": dict(fp_stats),
                "fn_stats": dict(fn_stats),
            })

        # save error analysis results to json file
        dump_results_to_json(error_analysis_outputs, prefix="error_analysis_results")

            # stats = Counter()
            # for res in fp_results:
            #     if isinstance(res, dict) and "error_type" in res:
            #         print(res)
            #         stats[res["error_type"]] += 1
            #     else:
            #         # stats[parse_gpt_response(res)["error_type"]] += 1
            #         print("key error")
            #     print("fp stats", stats)

            # stats = Counter()
            # for res in fn_results:
            #     if isinstance(res, dict) and "error_type" in res:
            #         print(res)
            #         stats[res["error_type"]] += 1
            #     else:
            #         # stats[parse_gpt_response(res)["error_type"]] += 1
            #         print("key error")
            #     print("fn stats", stats)


if __name__ == "__main__":
    main()