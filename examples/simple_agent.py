"""
Simple baseline agent for WhipStudio.

This is a minimal example showing how to interact with the WhipStudio environment
using direct code submission (no tool use). Good for understanding the basic API.

Usage:
    python examples/simple_agent.py --env-url http://localhost:7860
    python examples/simple_agent.py --env-url https://your-space.hf.space
"""

import argparse
import os
import httpx
from openai import OpenAI


SYSTEM_PROMPT = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs in it.

Rules:
- Return ONLY the complete corrected Python code, nothing else.
- No markdown, no backticks, no explanation text.
- The script must print losses in format: LOSSES:[v1, v2, ...]
- For tasks requiring validation metrics, also print: VAL_ACC:X.XX
- Keep all torch.manual_seed() calls intact.""".strip()


def get_client():
    """Initialize OpenAI-compatible client."""
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("Set HF_TOKEN or OPENAI_API_KEY environment variable")
    
    return OpenAI(base_url=api_base, api_key=api_key)


def generate_fix(client, buggy_code: str, task_description: str, error_log: str = "") -> str:
    """Use LLM to generate a fix for the buggy code."""
    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
    
    user_prompt = f"""Task: {task_description}

Buggy Code:
```python
{buggy_code}
```"""
    
    if error_log:
        user_prompt += f"\n\nPrevious Error:\n{error_log}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
        temperature=0.2,
    )
    
    return response.choices[0].message.content.strip()


def run_task(env_url: str, task_id: str, client, max_attempts: int = 3) -> float:
    """Run a single task with multiple attempts."""
    print(f"\n{'='*60}")
    print(f"Starting {task_id}")
    print(f"{'='*60}")
    
    # Reset environment
    with httpx.Client(timeout=60.0) as http_client:
        resp = http_client.post(f"{env_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json().get("observation", resp.json())
    
    buggy_code = obs.get("buggy_code", "")
    task_description = obs.get("task_description", "")
    
    print(f"Task: {task_description[:100]}...")
    
    best_reward = 0.0
    error_log = ""
    
    for attempt in range(1, max_attempts + 1):
        # Reset for each attempt (except first, already reset above)
        if attempt > 1:
            try:
                with httpx.Client(timeout=30.0) as http_client:
                    resp = http_client.post(f"{env_url}/reset", json={"task_id": task_id})
                    resp.raise_for_status()
                    obs = resp.json().get("observation", resp.json())
                print(f"[Reset for attempt {attempt}]")
            except Exception as e:
                print(f"Reset Error: {e}")
                continue
        
        print(f"\n--- Attempt {attempt}/{max_attempts} ---")
        
        # Generate fix using LLM
        try:
            fixed_code = generate_fix(client, buggy_code, task_description, error_log)
            
            # Clean up markdown if present
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python", 1)[1].split("```", 1)[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```", 1)[1].split("```", 1)[0].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            continue
        
        # Submit fix
        action = {
            "action_type": "submit_fix",
            "fixed_code": fixed_code,
            "attempt_number": attempt,
        }
        
        try:
            with httpx.Client(timeout=120.0) as http_client:
                resp = http_client.post(f"{env_url}/step", json={"action": action})
                resp.raise_for_status()
                result = resp.json()
        except Exception as e:
            print(f"API Error: {e}")
            continue
        
        obs = result.get("observation", {})
        reward = float(result.get("reward", 0.0) or 0.0)
        done = result.get("done", False)
        
        print(f"Reward: {reward:.4f}")
        
        if reward > best_reward:
            best_reward = reward
        
        error_log = obs.get("error_log", "")
        
        # Only stop if we got a great score
        if reward >= 0.95:
            print(f"Task solved! Stopping attempts.")
            break
    
    print(f"\nBest reward for {task_id}: {best_reward:.4f}")
    return best_reward


def main():
    parser = argparse.ArgumentParser(description="Simple WhipStudio Agent")
    parser.add_argument("--env-url", default="http://localhost:7860", help="Environment URL")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3", "task4", "task5", "task6"],
                        help="Task IDs to run")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max attempts per task")
    args = parser.parse_args()
    
    client = get_client()
    
    results = {}
    for task_id in args.tasks:
        try:
            score = run_task(args.env_url, task_id, client, args.max_attempts)
            results[task_id] = score
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            results[task_id] = 0.0
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    total = 0.0
    for task_id, score in results.items():
        emoji = "✅" if score >= 0.7 else ("📈" if score >= 0.3 else "❌")
        print(f"{emoji} {task_id}: {score:.4f}")
        total += score
    
    avg = total / len(results) if results else 0.0
    print(f"\nAverage: {avg:.4f}")


if __name__ == "__main__":
    main()
