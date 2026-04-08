# Multi-Turn Agents: Why They Make Sense

## The Confusion

**Common misconception**: "Each LLM API call is stateless, so multi-turn is pointless!"

**Reality**: The **agent** maintains state and provides context to the LLM across turns.

---

## Two Types of Agents

### 1. Simple One-Shot Agent (No Tools)

```python
# Attempt 1
obs = reset("task1")
prompt = f"Fix this code: {obs['buggy_code']}"
fix1 = llm.chat(prompt)
reward1 = submit_fix(fix1)  # 0.3

# Attempt 2 (RESET, fresh start)
obs = reset("task1")
prompt = f"Fix this code: {obs['buggy_code']}"  # SAME PROMPT
fix2 = llm.chat(prompt)  # Likely same response
reward2 = submit_fix(fix2)  # 0.3 again
```

**Giving 10 turns is useless** - each attempt is identical!

### 2. Tool-Calling Agent (Context Accumulation)

```python
obs = reset("task1")
history = []

# Turn 1: Inspect the model
action = {"action_type": "execute_snippet", "code": "print(model)"}
result = step(action)
history.append(f"Checked model: {result['stdout']}")

# Turn 2: Run training probe
action = {"action_type": "run_training_probe", "code": buggy_code, "steps": 5}
result = step(action)
history.append(f"Training probe: losses={result['losses']}")

# Turn 3: Inspect gradients
action = {"action_type": "inspect_tensor", "setup_code": setup, "target_expression": "model.weight.grad"}
result = step(action)
history.append(f"Gradients: {result['grad_is_none']}")

# Turn 4: Generate fix with ALL context
prompt = f"""
Task: {obs['buggy_code']}

Investigation results:
{chr(10).join(history)}

Now generate the complete fix.
"""
fix = llm.chat(prompt)  # LLM sees ALL tool results!
reward = submit_fix(fix)  # 0.85
```

**10-15 turns make sense** - each turn adds information!

---

## How It Works

### The Agent Loop

```python
class ToolCallingAgent:
    def __init__(self):
        self.history = []  # Agent maintains state
        
    def solve_task(self, task_id):
        obs = reset(task_id)
        
        for turn in range(1, 11):  # Up to 10 turns
            # Build prompt with accumulated context
            prompt = self.build_prompt(obs, self.history)
            
            # LLM decides next action (stateless call)
            response = llm.chat(prompt)
            action = parse_action(response)
            
            # Execute action
            if action.type == "submit_fix":
                return step(action)
            else:
                # Tool call - accumulate result
                result = step(action)
                self.history.append({
                    "turn": turn,
                    "action": action,
                    "result": result
                })
                # Loop continues with more context
```

### Example Prompt Evolution

**Turn 1 Prompt:**
```
Fix this buggy training loop:
[buggy code]
```

**Turn 5 Prompt (after 4 tool calls):**
```
Fix this buggy training loop:
[buggy code]

Investigation history:
Turn 1: execute_snippet("print(model)")
→ Output: Linear(in_features=10, out_features=2)

Turn 2: inspect_tensor(setup, "model.weight.grad")
→ grad_is_none=True (gradients are None!)

Turn 3: run_training_probe(buggy_code, steps=5)
→ losses=[0.8, NaN, NaN, NaN, NaN] (loss becomes NaN!)

Turn 4: get_variable_state(setup, ["optimizer.param_groups[0]['lr']"])
→ lr=0.1 (maybe too high?)

Based on these findings, generate the complete fix.
```

The LLM gets **much more information** on turn 5!

---

## When Multi-Turn Makes Sense

### ✅ Good Use Cases:

1. **Tool-calling agents** (like `examples/tool_agent.py`)
   - Accumulate debugging information
   - Each turn narrows down the bug
   - Final submission uses all gathered insights

2. **RL agents with memory**
   - Neural networks maintain state
   - Learn from experience across episodes
   - Explore different debugging strategies

3. **Agentic workflows** (AutoGPT, ReAct)
   - Plan → Execute → Observe → Replan
   - Each turn refines the strategy

### ❌ Bad Use Cases:

1. **One-shot inference** (`inference.py` without tools)
   - Each attempt is independent
   - No context accumulation
   - Just hoping for different random samples

2. **Non-agentic LLM calls**
   - If you're not passing history to the LLM
   - If you're not using tool results

---

## WhipStudio's Multi-Turn Design

WhipStudio supports **both** workflows:

### Simple Agents (3 attempts)
```bash
python examples/simple_agent.py --max-attempts 3
```
- Each attempt: Reset → LLM → Submit → Done
- No tools, no context accumulation
- 3 attempts = 3 chances with different random seeds

### Tool-Calling Agents (10 turns per attempt)
```bash
python examples/tool_agent.py --all-tasks
```
- Each episode: Reset → Tools (9×) → Submit (1×)
- Context accumulates across tools
- 10 turns = investigate thoroughly before submission

---

## Comparison

| Feature | Simple Agent | Tool Agent |
|---------|--------------|------------|
| **Turns per episode** | 1 | Up to 10 |
| **Context accumulation** | ❌ None | ✅ Tool results |
| **Multiple attempts** | ✅ 3 | ✅ 1 |
| **LLM calls per task** | 3 | 5-10 |
| **Success rate** | Lower | Higher |
| **Cost** | Lower | Higher |
| **Debugging depth** | Shallow | Deep |

---

## Your Original Question

> "Giving 10 steps or even 100 steps to these models won't make sense, because each call is completely new, right?"

**Answer**: It depends!

- **For `inference.py` (simple agent)**: You're right! Each attempt should reset and try again with a fresh prompt.
  
- **For `tool_agent.py` (tool-calling)**: You're wrong! The agent passes accumulated tool results to the LLM, making each turn more informed.

---

## Example: Task 1 (Broken Training Loop)

### Simple Agent (3 attempts, no tools)

```
Attempt 1: LLM guesses "maybe learning rate is wrong" → 0.3
Attempt 2: LLM guesses "maybe optimizer is wrong" → 0.4
Attempt 3: LLM guesses "maybe loss function is wrong" → 0.3
Best: 0.4
```

### Tool Agent (1 attempt, 10 turns)

```
Turn 1: inspect_tensor(model.weight.grad) → "grad is None!"
Turn 2: execute_snippet("print(optimizer)") → "SGD(lr=0.01)"
Turn 3: run_training_probe(buggy_code, 5) → "losses=[0.8,0.8,0.8,...]"
Turn 4: get_variable_state(["model.training"]) → "False" (AHA!)
Turn 5: inspect_diff(fixed_code) → "Added model.train()"
Turn 6: submit_fix(fixed_code) → 0.85
Best: 0.85
```

The tool agent **systematically debugs** the issue, while the simple agent **randomly guesses**.

---

## Recommendations

1. **For one-shot agents**: Use 3-5 attempts with reset between each
2. **For tool agents**: Use 10-15 turns to investigate before submitting
3. **For RL training**: Use 10-20 turns to explore different strategies
4. **For production**: Start with tools, fall back to multiple attempts if budget-constrained

---

## See Also

- [WORKFLOW.md](WORKFLOW.md) - Episode flow
- [TOOLS.md](TOOLS.md) - Tool usage guide
- [examples/tool_agent.py](../examples/tool_agent.py) - Reference implementation
