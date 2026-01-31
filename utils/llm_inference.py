import openai
import requests
import json
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import anthropic
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


# --------------------------
# Model pricing
# --------------------------
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

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate cost based on token usage and model pricing."""
    if model not in MODEL_PRICING:
        print(f"Warning: No pricing info for {model}, using gpt-4o-mini pricing")
        model = "gpt-4o-mini"
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost

# --------------------------
# LLMInference class
# --------------------------
class LLMInference:
    """Unified interface for calling LLMs (OpenAI, Gemini, LLaMA, Claude)."""

    def __init__(self, provider: str, model_name: str, api_key: str = "", hf_token: str = ""):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.hf_token = hf_token

        # Track usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

        if self.provider == "openai":
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)

        elif self.provider == "gemini":
            if not self.api_key:
                raise ValueError("Gemini requires a Google API key")

        elif self.provider == "llama":
            if not self.hf_token:
                raise ValueError("LLaMA requires a Hugging Face token")
            self.pipe = self._load_llama_pipeline(model_name)

        elif self.provider == "claude":
            if not self.api_key:
                raise ValueError("Claude requires an API key")
            self.client = anthropic.Anthropic(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # ---- OpenAI ----
    def _generate_openai(self, prompt, max_tokens=None, temperature=0):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cost = calculate_cost(input_tokens, output_tokens, self.model_name)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        return {
            "content": response.choices[0].message.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": cost
        }

    # ---- Claude ----
    def _generate_claude(self, prompt, max_tokens=256, temperature=0.7):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text

        # Claude gives token usage in response
        input_tokens = getattr(response.usage, "input_tokens", len(prompt.split()))
        output_tokens = getattr(response.usage, "output_tokens", len(text.split()))
        cost = calculate_cost(input_tokens, output_tokens, self.model_name)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        return {
            "content": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost
        }

    # ---- Gemini ----
    def _generate_gemini(self, prompt, max_tokens=None, temperature=0):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=payload)
        data = response.json()
        try:
            text = data['candidates'][0]['content']['parts'][0]['text']
        except KeyError:
            text = f"Error: {data}"

        input_tokens = len(prompt.split())
        output_tokens = len(text.split())
        cost = 0.0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return {
            "content": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost
        }

    # ---- LLaMA ----
    def _load_llama_pipeline(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.hf_token)
        model = AutoModelForCausalLM.from_pretrained(model_name, token=self.hf_token, device_map="auto")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    def _generate_llama(self, prompt, max_tokens=256, temperature=0.7):
        output = self.pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        text = output[0]['generated_text']

        input_tokens = len(prompt.split())
        output_tokens = len(text.split())
        cost = 0.0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        return {
            "content": text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost
        }

    # ---- Unified generate ----
    def generate(self, prompt, **kwargs):
        if self.provider == "openai":
            return self._generate_openai(prompt, **kwargs)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, **kwargs)
        elif self.provider == "llama":
            return self._generate_llama(prompt, **kwargs)
        elif self.provider == "claude":
            return self._generate_claude(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def batch_generate(self, prompts: List[str], max_workers=20, **kwargs):
        results = [None] * len(prompts)

        def call_generate(i, prompt):
            try:
                result = self.generate(prompt, **kwargs)
                return i, result
            except Exception as e:
                return i, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(call_generate, i, prompt): i
                for i, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_index):
                i, result = future.result()
                results[i] = result

        return results

    def get_usage_summary(self, summary_file="llm_usage_summary.json"):
        summary = {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


# Test Examples

# --------------------------
# Prompts to test
# --------------------------
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a short poem about autumn.",
    "Summarize the benefits of machine learning."
]

def test_openai():
    print("\n=== Testing OpenAI (gpt-4o-mini) ===")
    client = LLMInference(provider="openai", model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
    results = client.batch_generate(prompts, max_workers=20, max_tokens=100)
    for i, res in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {res['content'][:200]}...")  # show first 200 chars
    print("Usage summary:", client.get_usage_summary())

def test_gemini():
    print("\n=== Testing Gemini (gemini-pro) ===")
    client = LLMInference(provider="gemini", model_name="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
    results = client.batch_generate(prompts, max_workers=20)
    for i, res in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {res['content'][:200]}...")
    print("Usage summary:", client.get_usage_summary())

def test_llama():
    print("\n=== Testing LLaMA (Meta-Llama-3-8B-Instruct) ===")
    client = LLMInference(provider="llama", model_name="meta-llama/Meta-Llama-3-8B-Instruct", hf_token=HF_TOKEN)
    results = client.batch_generate(prompts, max_workers=20, max_tokens=100)
    for i, res in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {res['content'][:200]}...")
    print("Usage summary:", client.get_usage_summary())

def test_claude():
    print("\n=== Testing Claude (claude-3-5-sonnet) ===")
    client = LLMInference(provider="claude", model_name="claude-3-5-sonnet-20241022", api_key=CLAUDE_API_KEY)
    results = client.batch_generate(prompts, max_workers=20, max_tokens=200)
    for i, res in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        if "content" in res:
            print(f"Response: {res['content'][:200]}...")
        else:
            print(f"Error: {res.get('error', 'Unknown error')}")
    print("Usage summary:", client.get_usage_summary())

if __name__ == "__main__":
    if OPENAI_API_KEY:
        test_openai()
    else:
        print("Skipping OpenAI test: no OPENAI_API_KEY")

    if GOOGLE_API_KEY:
        test_gemini()
    else:
        print("Skipping Gemini test: no GOOGLE_API_KEY")

    if HF_TOKEN:
        test_llama()
    else:
        print("Skipping LLaMA test: no HF_TOKEN")

    if CLAUDE_API_KEY:
        test_claude()
    else:
        print("Skipping Claude test: no CLAUDE_API_KEY")
