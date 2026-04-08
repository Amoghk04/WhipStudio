"""
Tool-using agent for WhipStudio.

This example demonstrates how to use WhipStudio's debugging tools
to iteratively analyze and fix bugs before submitting a final solution.

The agent uses:
1. execute_snippet - To test hypotheses about the code
2. inspect_tensor - To check tensor shapes, dtypes, and gradients
3. get_variable_state - To evaluate multiple expressions
4. run_training_probe - To test potential fixes
5. inspect_diff - To review changes before submission
6. submit_fix - Final submission

Usage:
    python examples/tool_agent.py --env-url http://localhost:7860 --task task1
    python examples/tool_agent.py --env-url https://your-space.hf.space --task task6
"""

import argparse
import json
import os
import re
import httpx
from openai import OpenAI


SYSTEM_PROMPT = """You are an expert PyTorch debugging agent that fixes buggy ML code.
You have debugging tools available, but your PRIMARY GOAL is to SUBMIT A FIX.

AVAILABLE TOOLS:
1. execute_snippet - Run Python code to test hypotheses
2. inspect_tensor - Check tensor shape, dtype, gradients, NaN/Inf
3. get_variable_state - Inspect multiple variable values  
4. run_training_probe - Run training steps to see loss curve
5. inspect_diff - Preview your changes before submitting
6. submit_fix - Submit your final fix (ALWAYS DO THIS BEFORE RUNNING OUT OF TURNS)

RESPONSE FORMAT - ALWAYS respond with valid JSON only:
{
  "reasoning": "Brief analysis",
  "action_type": "tool_name",
  "action_params": { ... }
}

For submit_fix, the fixed_code must be COMPLETE working Python code:
{
  "reasoning": "Fix explanation",
  "action_type": "submit_fix",
  "action_params": {
    "fixed_code": "import torch\\nimport torch.nn as nn\\n..."
  }
}

CRITICAL RULES:
1. You MUST call submit_fix before your turns run out
2. If you have 2 or fewer turns remaining, IMMEDIATELY submit your fix
3. Don't waste turns - analyze, test once if needed, then SUBMIT
4. Fixed code must print: LOSSES:[v1, v2, ...] or similar metrics
5. Keep torch.manual_seed() calls intact for reproducibility
6. Use \\n for newlines in code strings

EFFICIENT DEBUGGING:
- Turn 1-2: Analyze bug, maybe one quick test
- Turn 3+: SUBMIT YOUR FIX - don't keep testing!""".strip()


def get_client():
    """Initialize OpenAI-compatible client."""
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("Set HF_TOKEN or OPENAI_API_KEY environment variable")
    
    return OpenAI(base_url=api_base, api_key=api_key)


def parse_agent_response(response: str) -> dict:
    """Parse JSON response from agent, handling common formatting issues."""
    response = response.strip()
    
    # Remove markdown code blocks
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    
    # Fallback
    return {
        "reasoning": "Fallback: could not parse response",
        "action_type": "submit_fix",
        "action_params": {"fixed_code": response}
    }


def format_tool_result(obs: dict, action_type: str) -> str:
    """Format tool result for display."""
    turn = obs.get("turn", 0)
    error = obs.get("error")
    
    if error:
        return f"[Turn {turn}] {action_type} ERROR: {error}"
    
    if action_type == "execute_snippet":
        stdout = obs.get("stdout", "")[:2000]  # Increased from 500
        stderr = obs.get("stderr", "")[:1000]  # Increased from 200
        exit_code = obs.get("exit_code", 0)
        result = f"[Turn {turn}] execute_snippet (exit={exit_code}):\n{stdout}"
        if stderr:
            result += f"\nSTDERR:\n{stderr}"
        return result
    
    elif action_type == "inspect_tensor":
        return f"""[Turn {turn}] inspect_tensor:
  shape: {obs.get('shape')}
  dtype: {obs.get('dtype')}
  requires_grad: {obs.get('requires_grad')}
  grad_is_none: {obs.get('grad_is_none')}
  min/max/mean: {obs.get('min_val')}/{obs.get('max_val')}/{obs.get('mean_val')}
  is_nan: {obs.get('is_nan')}, is_inf: {obs.get('is_inf')}"""
    
    elif action_type == "run_training_probe":
        losses = obs.get("losses", [])[:10]
        final_loss = obs.get("final_loss")
        grad_norms = obs.get("grad_norms", {})
        return f"[Turn {turn}] run_training_probe:\n  losses: {losses}\n  final_loss: {final_loss}\n  grad_norms: {grad_norms}"
    
    elif action_type == "get_variable_state":
        results = obs.get("results", {})
        lines = [f"[Turn {turn}] get_variable_state:"]
        for expr, res in results.items():
            if res.get("error"):
                lines.append(f"  {expr}: ERROR - {res['error']}")
            else:
                val = res.get("repr", str(res.get("value", "?")))[:200]  # Increased from 80
                lines.append(f"  {expr}: {val}")
        return "\n".join(lines)
    
    elif action_type == "inspect_diff":
        lines_changed = obs.get("lines_changed", 0)
        additions = obs.get("additions", 0)
        deletions = obs.get("deletions", 0)
        diff = obs.get("diff", "")[:500]
        return f"[Turn {turn}] inspect_diff: {lines_changed} lines changed (+{additions}/-{deletions})\n{diff}"
    
    elif action_type == "submit_fix":
        reward = obs.get("reward", 0.0)
        return f"[Turn {turn}] submit_fix: reward={reward}"
    
    return f"[Turn {turn}] {action_type}: {json.dumps(obs, default=str)[:500]}"


def run_tool_agent(env_url: str, task_id: str, client, max_turns: int = 10) -> float:
    """Run a tool-using agent on a single task."""
    print(f"\n{'='*60}")
    print(f"Tool Agent: {task_id}")
    print(f"{'='*60}")
    
    model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-3B-Instruct")
    
    # Reset environment
    with httpx.Client(timeout=60.0) as http_client:
        resp = http_client.post(f"{env_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json().get("observation", resp.json())
    
    buggy_code = obs.get("buggy_code", "")
    task_description = obs.get("task_description", "")
    episode_id = obs.get("episode_id", "")  # Track episode_id for session persistence
    
    print(f"Task: {task_description[:100]}...")
    print(f"Episode ID: {episode_id[:16]}..." if episode_id else "No episode_id")
    
    tool_history = []
    best_reward = 0.0
    
    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn}/{max_turns} ---")
        
        turns_remaining = max_turns - turn
        
        # Build context
        history_text = "\n".join(tool_history[-5:]) if tool_history else "No previous tool calls."
        
        # Force submission on last turn
        if turns_remaining == 0:
            urgency = "\n⚠️ THIS IS YOUR LAST TURN! You MUST call submit_fix NOW with your best fix."
        elif turns_remaining <= 2:
            urgency = f"\n⚠️ ONLY {turns_remaining} TURN(S) LEFT! Submit your fix soon!"
        else:
            urgency = ""
        
        user_prompt = f"""Task: {task_description}

Buggy Code:
```python
{buggy_code}
```

Turn {turn}/{max_turns}
Best reward so far: {best_reward}
Turns remaining: {turns_remaining}{urgency}

Tool History:
{history_text}

Analyze the code and submit your fix. Don't waste turns on unnecessary testing.""".strip()
        
        # Get LLM response
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            response_text = response.choices[0].message.content.strip()
            parsed = parse_agent_response(response_text)
        except Exception as e:
            print(f"LLM Error: {e}")
            tool_history.append(f"[Turn {turn}] LLM ERROR: {e}")
            continue
        
        action_type = parsed.get("action_type", "submit_fix")
        action_params = parsed.get("action_params", {})
        reasoning = parsed.get("reasoning", "")[:100]
        
        # Force submit_fix on last turn if agent didn't choose it
        if turns_remaining == 0 and action_type != "submit_fix":
            print(f"[OVERRIDE] Last turn - forcing submit_fix instead of {action_type}")
            action_type = "submit_fix"
            # Use fixed_code from action_params if available, otherwise use any code param
            fixed_code = action_params.get("fixed_code") or action_params.get("code") or action_params.get("proposed_code") or buggy_code
            action_params = {"fixed_code": fixed_code}
        
        print(f"Action: {action_type}")
        print(f"Reasoning: {reasoning}...")
        
        # Build action payload - ALWAYS include episode_id for session tracking
        action = {
            "action_type": action_type,
            "episode_id": episode_id,  # Critical for session persistence in HTTP mode
        }
        
        if action_type == "execute_snippet":
            action["code"] = action_params.get("code", "print('test')")
        elif action_type == "inspect_tensor":
            action["setup_code"] = action_params.get("setup_code", "")[:8000]
            action["target_expression"] = action_params.get("target_expression", "")
        elif action_type == "run_training_probe":
            action["code"] = action_params.get("code", buggy_code)[:8000]
            action["steps"] = min(action_params.get("steps", 5), 10)
        elif action_type == "get_variable_state":
            action["setup_code"] = action_params.get("setup_code", "")[:8000]
            action["expressions"] = action_params.get("expressions", [])[:10]
        elif action_type == "inspect_diff":
            action["proposed_code"] = action_params.get("proposed_code", "")
        elif action_type == "submit_fix":
            fixed_code = action_params.get("fixed_code", "")
            # Clean markdown
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python", 1)[1].split("```", 1)[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```", 1)[1].split("```", 1)[0].strip()
            action["fixed_code"] = fixed_code
        
        # Execute action
        try:
            with httpx.Client(timeout=120.0) as http_client:
                resp = http_client.post(f"{env_url}/step", json={"action": action})
                resp.raise_for_status()
                result = resp.json()
        except Exception as e:
            print(f"API Error: {e}")
            tool_history.append(f"[Turn {turn}] API ERROR: {e}")
            continue
        
        obs = result.get("observation", {})
        reward = float(result.get("reward", 0.0) or 0.0)
        done = result.get("done", False) or obs.get("episode_done", False)
        
        # Format and store result
        tool_result = format_tool_result(obs, action_type)
        tool_history.append(tool_result)
        print(tool_result)
        
        if reward > best_reward:
            best_reward = reward
        
        if action_type == "submit_fix":
            print(f"Reward: {reward:.4f}")
            if reward >= 0.95 or done:
                break
        
        if done:
            break
    
    print(f"\nFinal reward for {task_id}: {best_reward:.4f}")
    return best_reward


def main():
    parser = argparse.ArgumentParser(description="Tool-using WhipStudio Agent")
    parser.add_argument("--env-url", default="http://localhost:7860", help="Environment URL")
    parser.add_argument("--task", default="task1", help="Task ID to run")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per task")
    args = parser.parse_args()
    
    client = get_client()
    
    tasks = ["task1", "task2", "task3", "task4", "task5", "task6"] if args.all_tasks else [args.task]
    
    results = {}
    for task_id in tasks:
        try:
            score = run_tool_agent(args.env_url, task_id, client, args.max_turns)
            results[task_id] = score
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            results[task_id] = 0.0
    
    if len(results) > 1:
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
