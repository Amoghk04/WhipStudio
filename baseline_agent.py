import asyncio
import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

import httpx

# ── Task difficulty configuration ──────────────────────────────────────────

TASK_CONFIG = {
    "task1": {"difficulty": "easy", "max_turns": 8, "description": "Broken training loop"},
    "task2": {"difficulty": "medium", "max_turns": 10, "description": "Silent NaN loss"},
    "task3": {"difficulty": "hard", "max_turns": 12, "description": "OOM and data leakage"},
    "task4": {"difficulty": "medium", "max_turns": 10, "description": "Wrong loss function"},
    "task5": {"difficulty": "medium", "max_turns": 10, "description": "Frozen backbone"},
    "task6": {"difficulty": "hard", "max_turns": 15, "description": "Input-Output mismatch"},
}

# ── System prompts ─────────────────────────────────────────────────────────

SYSTEM_PROMPT_TOOLS = """
You are an expert PyTorch debugging agent with access to debugging tools.
You receive a broken training script and must systematically debug and fix ALL bugs.

AVAILABLE TOOLS:
1. execute_snippet - Run a quick Python snippet to test hypotheses
2. inspect_tensor - Check tensor shape, dtype, gradients, NaN/Inf
3. get_variable_state - Inspect multiple variable values
4. run_training_probe - Run a few training steps to see loss curve
5. inspect_diff - Review your proposed changes before submitting
6. submit_fix - Submit your final fix (terminal action)

DEBUGGING STRATEGY:
1. First, analyze the buggy code carefully
2. Use execute_snippet or get_variable_state to verify your hypotheses
3. Use inspect_tensor to check gradient flow and tensor properties
4. Use run_training_probe to test potential fixes
5. Use inspect_diff to review your changes
6. Only submit_fix when confident

RESPONSE FORMAT - CRITICAL:
You MUST respond with ONLY a valid JSON object. No markdown, no explanation outside JSON.

For submit_fix (THE fixed_code MUST BE ACTUAL PYTHON CODE STRING, NOT JSON):
{
  "reasoning": "Why this fix should work",
  "action_type": "submit_fix",
  "action_params": {
    "fixed_code": "import torch\\nimport torch.nn as nn\\n# Full Python script here\\nprint('LOSSES:', losses)"
  }
}

For execute_snippet:
{
  "reasoning": "Testing hypothesis",
  "action_type": "execute_snippet",
  "action_params": {"code": "print('test')"}
}

For inspect_tensor:
{
  "reasoning": "Check gradients",
  "action_type": "inspect_tensor", 
  "action_params": {
    "setup_code": "import torch\\nmodel = ...",
    "target_expression": "model.weight.grad"
  }
}

For get_variable_state:
{
  "reasoning": "Verify shapes",
  "action_type": "get_variable_state",
  "action_params": {
    "setup_code": "import torch\\ndata = ...",
    "expressions": ["data.shape", "data.dtype"]
  }
}

For run_training_probe:
{
  "reasoning": "Test my fix",
  "action_type": "run_training_probe",
  "action_params": {
    "code": "import torch\\n# full script",
    "steps": 5
  }
}

For inspect_diff:
{
  "reasoning": "Review changes",
  "action_type": "inspect_diff",
  "action_params": {"proposed_code": "import torch\\n# your fix"}
}

CRITICAL RULES:
- Respond ONLY with valid JSON
- For submit_fix: fixed_code = PYTHON STRING (use \\n for newlines), NOT nested JSON
- Fixed code must print: LOSSES:[v1, v2, ...]
- For task3: also print VAL_ACCS:[...] and FINAL_LOSS:X.XX
- Keep torch.manual_seed() intact
""".strip()

SYSTEM_PROMPT_SIMPLE = """
You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs in it.
Rules:
- Return ONLY the complete corrected Python code, nothing else.
- No markdown, no backticks, no explanation text.
- The script must print losses in format: LOSSES:[v1, v2, ...]
- For task3, also print: VAL_ACCS:[v1,...] and FINAL_LOSS:X.XX
- Keep all torch.manual_seed() calls intact.
""".strip()


SUPPORTED_MODEL_IDS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def get_model(model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct"):
    from smolagents import InferenceClientModel

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Set HF_TOKEN to run /baseline with InferenceClientModel."
        )

    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(
            f"Unsupported model_id '{model_id}'. Supported options: {SUPPORTED_MODEL_IDS}"
        )

    return InferenceClientModel(
        model_id=model_id,
        token=hf_token,
    )


def _extract_text(response) -> str:
    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text:
                        chunks.append(str(text))
            if chunks:
                return "\n".join(chunks)

    if isinstance(response, dict):
        text = response.get("content") or response.get("text")
        if isinstance(text, str):
            return text

    return str(response)


def _generate_response(model, system_prompt: str, prompt: str) -> str:
    """Generate a response from the model."""
    if hasattr(model, "generate"):
        generate = getattr(model, "generate")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return _extract_text(generate(messages=messages))
        except TypeError:
            return _extract_text(generate(messages))

    if callable(model):
        try:
            return _extract_text(model(prompt, system_prompt=system_prompt))
        except TypeError:
            return _extract_text(model(prompt))

    raise AttributeError("Model does not support callable() or generate() inference APIs")


def _parse_agent_response(response: str) -> dict:
    """Parse the agent's JSON response, handling potential markdown wrapping."""
    response = response.strip()
    
    # Remove markdown code blocks if present
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
        # Try to extract JSON from the response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    
    # Fallback: treat entire response as code for submit_fix
    return {
        "reasoning": "Fallback: treating response as code",
        "action_type": "submit_fix",
        "action_params": {"fixed_code": response}
    }


def _format_tool_result(obs: dict, action_type: str) -> str:
    """Format tool observation for the agent's context."""
    turn = obs.get("turn", 0)
    error = obs.get("error")
    
    if error:
        return f"Turn {turn} - {action_type}: ERROR - {error}"
    
    if action_type == "execute_snippet":
        stdout = obs.get("stdout", "")[:1500]
        stderr = obs.get("stderr", "")[:500]
        exit_code = obs.get("exit_code", 0)
        timed_out = obs.get("timed_out", False)
        result = f"Turn {turn} - execute_snippet (exit={exit_code}, timed_out={timed_out}):\n"
        if stdout:
            result += f"stdout:\n{stdout}\n"
        if stderr:
            result += f"stderr:\n{stderr}\n"
        return result
    
    elif action_type == "inspect_tensor":
        parts = [f"Turn {turn} - inspect_tensor:"]
        if obs.get("shape"): parts.append(f"  shape: {obs['shape']}")
        if obs.get("dtype"): parts.append(f"  dtype: {obs['dtype']}")
        if obs.get("requires_grad") is not None: parts.append(f"  requires_grad: {obs['requires_grad']}")
        if obs.get("grad_is_none") is not None: parts.append(f"  grad_is_none: {obs['grad_is_none']}")
        if obs.get("min_val") is not None: parts.append(f"  min: {obs['min_val']:.6f}")
        if obs.get("max_val") is not None: parts.append(f"  max: {obs['max_val']:.6f}")
        if obs.get("mean_val") is not None: parts.append(f"  mean: {obs['mean_val']:.6f}")
        if obs.get("is_nan"): parts.append("  ⚠️ CONTAINS NaN")
        if obs.get("is_inf"): parts.append("  ⚠️ CONTAINS Inf")
        return "\n".join(parts)
    
    elif action_type == "run_training_probe":
        losses = obs.get("losses", [])
        grad_norms = obs.get("grad_norms", {})
        final_loss = obs.get("final_loss")
        loss_is_nan = obs.get("loss_is_nan", False)
        loss_is_inf = obs.get("loss_is_inf", False)
        timed_out = obs.get("timed_out", False)
        
        result = f"Turn {turn} - run_training_probe:\n"
        result += f"  losses: {losses[:10]}\n"
        result += f"  final_loss: {final_loss}\n"
        if grad_norms:
            result += f"  grad_norms: {dict(list(grad_norms.items())[:5])}\n"
        if loss_is_nan: result += "  ⚠️ NaN LOSS DETECTED\n"
        if loss_is_inf: result += "  ⚠️ Inf LOSS DETECTED\n"
        if timed_out: result += "  ⚠️ TIMED OUT\n"
        return result
    
    elif action_type == "get_variable_state":
        results = obs.get("results", {})
        lines = [f"Turn {turn} - get_variable_state:"]
        for expr, res in results.items():
            if res.get("error"):
                lines.append(f"  {expr}: ERROR - {res['error']}")
            else:
                val = res.get("repr", str(res.get("value", "?")))[:100]
                typ = res.get("type", "?")
                shape = res.get("shape")
                shape_str = f" shape={shape}" if shape else ""
                lines.append(f"  {expr}: {val} ({typ}{shape_str})")
        return "\n".join(lines)
    
    elif action_type == "inspect_diff":
        lines_changed = obs.get("lines_changed", 0)
        additions = obs.get("additions", 0)
        deletions = obs.get("deletions", 0)
        diff = obs.get("diff", "")[:2000]
        return f"Turn {turn} - inspect_diff: {lines_changed} lines changed (+{additions}/-{deletions})\n{diff}"
    
    return f"Turn {turn} - {action_type}: {json.dumps(obs, default=str)[:500]}"


async def run_single_task(
    task_id: str,
    env_url: str = "http://localhost:7860",
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
) -> float:
    """Backwards-compatible wrapper that returns just the score."""
    result = await run_single_task_detailed(task_id, env_url, model_id)
    return result["score"]


async def run_single_task_detailed(
    task_id: str,
    env_url: str = "http://localhost:7860",
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    use_tools: bool = True,
) -> dict:
    """Run the baseline agent on a single task with optional tool use."""
    model = get_model(model_id)
    timeout = httpx.Timeout(900.0, connect=10.0)
    
    task_config = TASK_CONFIG.get(task_id, {"max_turns": 10, "difficulty": "medium"})
    max_turns = task_config["max_turns"]
    
    tool_history = []
    attempts_log = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Reset environment
        reset_resp = await client.post(f"{env_url}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        obs = reset_resp.json().get("observation", reset_resp.json())
        
        buggy_code = obs.get("buggy_code", "")
        task_description = obs.get("task_description", "")
        
        best_reward = 0.0
        best_code = ""
        best_output = ""
        turn = 0
        
        if use_tools:
            # Multi-step tool-using agent
            while turn < max_turns:
                turn += 1
                
                # Build context with tool history
                tool_context = "\n\n".join(tool_history[-5:]) if tool_history else "No previous tool calls."
                
                prompt = f"""
Task: {task_description}

Buggy code:
```python
{buggy_code}
```

Turn {turn}/{max_turns} - Tool History:
{tool_context}

Best reward so far: {best_reward}

Analyze the buggy code and decide your next action. Remember:
- Use tools to understand the bugs before fixing
- You have {max_turns - turn} turns remaining
- Submit your fix when confident

Respond with a JSON object containing your reasoning and action.
""".strip()
                
                try:
                    response = _generate_response(model, SYSTEM_PROMPT_TOOLS, prompt)
                    parsed = _parse_agent_response(response)
                    
                    action_type = parsed.get("action_type", "submit_fix")
                    action_params = parsed.get("action_params", {})
                    reasoning = parsed.get("reasoning", "")
                    
                    # Build action payload
                    action = {"action_type": action_type}
                    
                    if action_type == "execute_snippet":
                        code = action_params.get("code", "print('test')")
                        # Validate it's actual Python, not JSON
                        if code.strip().startswith("{") and '"action_type"' in code:
                            raise ValueError("Model returned nested JSON instead of Python code")
                        action["code"] = code
                    elif action_type == "inspect_tensor":
                        setup_code = action_params.get("setup_code", "")
                        # Truncate if too long (model may be exceeding context)
                        if len(setup_code) > 8000:
                            setup_code = setup_code[:8000] + "\n# ... truncated ..."
                        action["setup_code"] = setup_code
                        action["target_expression"] = action_params.get("target_expression", "")
                    elif action_type == "run_training_probe":
                        code = action_params.get("code", buggy_code)
                        if len(code) > 8000:
                            code = code[:8000]
                        if code.strip().startswith("{") and '"action_type"' in code:
                            raise ValueError("Model returned nested JSON instead of Python code")
                        action["code"] = code
                        action["steps"] = min(action_params.get("steps", 5), 10)
                    elif action_type == "get_variable_state":
                        setup_code = action_params.get("setup_code", "")
                        if len(setup_code) > 8000:
                            setup_code = setup_code[:8000]
                        action["setup_code"] = setup_code
                        action["expressions"] = action_params.get("expressions", [])[:10]
                    elif action_type == "inspect_diff":
                        action["proposed_code"] = action_params.get("proposed_code", "")
                    elif action_type == "submit_fix":
                        fixed_code = action_params.get("fixed_code", "")
                        
                        # Clean up markdown if present
                        if "```python" in fixed_code:
                            fixed_code = fixed_code.split("```python", 1)[1].split("```", 1)[0].strip()
                        elif "```json" in fixed_code:
                            fixed_code = fixed_code.split("```json", 1)[1].split("```", 1)[0].strip()
                        elif "```" in fixed_code:
                            fixed_code = fixed_code.split("```", 1)[1].split("```", 1)[0].strip()
                        
                        # CRITICAL: Detect if model returned JSON instead of Python code
                        if fixed_code.strip().startswith("{") and ('"action_type"' in fixed_code or '"reasoning"' in fixed_code):
                            raise ValueError(
                                "Model returned nested JSON instead of Python code for submit_fix. "
                                "The fixed_code field must contain actual Python code, not JSON."
                            )
                        
                        # Validate it's not empty
                        if not fixed_code or len(fixed_code) < 20:
                            raise ValueError("submit_fix received empty or too-short code")
                        
                        action["fixed_code"] = fixed_code
                    else:
                        # Unknown action, treat as submit_fix
                        action = {"action_type": "submit_fix", "fixed_code": str(action_params)}
                    
                    # Execute action
                    step_resp = await client.post(f"{env_url}/step", json={"action": action})
                    step_resp.raise_for_status()
                    result = step_resp.json()
                    
                    obs = result.get("observation", {})
                    reward = float(result.get("reward", 0.0) or 0.0)
                    done = result.get("done", False) or obs.get("episode_done", False)
                    
                    # Format and store tool result
                    tool_result = _format_tool_result(obs, action_type)
                    tool_history.append(f"[Turn {turn}] Reasoning: {reasoning[:200]}\n{tool_result}")
                    
                    # Log attempt if it was a submit
                    if action_type == "submit_fix":
                        output_log = obs.get("error_log", "") if isinstance(obs, dict) else ""
                        attempts_log.append({
                            "turn": turn,
                            "action": "submit_fix",
                            "code": action.get("fixed_code", "")[:2000],
                            "output": output_log[:2000],
                            "reward": reward,
                        })
                        
                        if reward > best_reward:
                            best_reward = reward
                            best_code = action.get("fixed_code", "")
                            best_output = output_log
                        
                        if reward >= 0.95 or done:
                            break
                    else:
                        attempts_log.append({
                            "turn": turn,
                            "action": action_type,
                            "params": {k: str(v)[:200] for k, v in action_params.items()},
                            "result": tool_result[:500],
                        })
                    
                    if done:
                        break
                        
                except ValueError as ve:
                    # Model error (nested JSON, empty code, etc.)
                    tool_history.append(f"[Turn {turn}] MODEL ERROR: {str(ve)[:300]}")
                    attempts_log.append({
                        "turn": turn,
                        "action": "error",
                        "params": {"error": str(ve), "response_preview": response[:500]},
                        "result": "Model generated invalid response - skipping turn",
                    })
                    # Continue to next turn, give model another chance
                    continue
                    
                except httpx.HTTPError as he:
                    # API error
                    tool_history.append(f"[Turn {turn}] API ERROR: {str(he)[:200]}")
                    attempts_log.append({
                        "turn": turn,
                        "action": "error",
                        "params": {"error": str(he)},
                        "result": "API call failed",
                    })
                    # Continue to next turn
                    continue
                    
                except Exception as e:
                    tool_history.append(f"[Turn {turn}] UNEXPECTED ERROR: {str(e)[:200]}")
                    attempts_log.append({
                        "turn": turn,
                        "action": "error",
                        "params": {"error": str(e), "type": type(e).__name__},
                        "result": "Unexpected error",
                    })
                    # Continue to next turn
                    continue
            
            # If we haven't submitted yet, do a final submit
            if best_reward == 0.0 and turn < max_turns:
                # Generate a simple fix without tools
                simple_prompt = f"""
Task: {task_description}
Buggy code:
{buggy_code}

Tool debugging history:
{chr(10).join(tool_history[-3:])}

Generate the complete fixed Python code. Return ONLY the code, no explanation.
""".strip()
                
                try:
                    fixed_code = _generate_response(model, SYSTEM_PROMPT_SIMPLE, simple_prompt)
                    if "```python" in fixed_code:
                        fixed_code = fixed_code.split("```python", 1)[1].split("```", 1)[0].strip()
                    elif "```" in fixed_code:
                        fixed_code = fixed_code.split("```", 1)[1].split("```", 1)[0].strip()
                    
                    step_resp = await client.post(f"{env_url}/step", json={
                        "action": {"action_type": "submit_fix", "fixed_code": fixed_code}
                    })
                    result = step_resp.json()
                    reward = float(result.get("reward", 0.0) or 0.0)
                    obs = result.get("observation", {})
                    
                    if reward > best_reward:
                        best_reward = reward
                        best_code = fixed_code
                        best_output = obs.get("error_log", "")
                except Exception:
                    pass
        
        else:
            # Simple direct submission (fallback mode)
            for attempt in range(1, 4):
                prompt = f"""
Task: {task_description}
Buggy code:
{buggy_code}
Previous execution output (if any):
{obs.get('error_log', 'None')}
Previous score: {obs.get('last_reward', 0.0)}
""".strip()

                fixed_code = _generate_response(model, SYSTEM_PROMPT_SIMPLE, prompt)
                if "```python" in fixed_code:
                    fixed_code = fixed_code.split("```python", 1)[1].split("```", 1)[0].strip()
                elif "```" in fixed_code:
                    fixed_code = fixed_code.split("```", 1)[1].split("```", 1)[0].strip()

                step_payload = {"action": {"action_type": "submit_fix", "fixed_code": fixed_code}}
                step_resp = await client.post(f"{env_url}/step", json=step_payload)
                step_resp.raise_for_status()
                result = step_resp.json()

                reward = float(result.get("reward", 0.0) or 0.0)
                obs = result.get("observation", obs)
                output_log = obs.get("error_log", "") if isinstance(obs, dict) else ""

                attempts_log.append({
                    "attempt": attempt,
                    "code": fixed_code,
                    "output": output_log[:3000],
                    "reward": reward,
                })

                if reward > best_reward:
                    best_reward = reward
                    best_code = fixed_code
                    best_output = output_log

                if result.get("done") or reward >= 0.95:
                    break

    return {
        "score": best_reward,
        "fixed_code": best_code,
        "output": best_output[:3000],
        "attempts": attempts_log,
        "tool_history": tool_history,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool use")
    parser.add_argument("--task", default=None, help="Run single task")
    args = parser.parse_args()

    async def main():
        tasks = [args.task] if args.task else ["task1", "task2", "task3", "task4", "task5", "task6"]
        scores = {}
        
        for tid in tasks:
            try:
                result = await asyncio.wait_for(
                    run_single_task_detailed(tid, args.env_url, use_tools=not args.no_tools),
                    timeout=900.0
                )
                s = result["score"]
                print(f"{tid}: {s:.4f}")
                if result.get("tool_history"):
                    print(f"  Tool calls: {len(result['tool_history'])}")
            except TimeoutError:
                s = 0.0
                print(f"{tid}: TIMEOUT")
            scores[tid] = round(s, 4)
        
        if len(scores) > 1:
            print(f"Average: {sum(scores.values()) / len(scores):.4f}")

    asyncio.run(main())
