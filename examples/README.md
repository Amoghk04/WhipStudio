# WhipStudio Examples

Example agent implementations for the WhipStudio ML Debugging environment.

## Prerequisites

```bash
# Set environment variables
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
```

## Available Examples

### 1. Simple Agent (`simple_agent.py`)

A minimal agent that directly generates and submits fixes without using tools.
Good for understanding the basic API.

```bash
# Run on localhost
python examples/simple_agent.py --env-url http://localhost:7860

# Run on HF Space
python examples/simple_agent.py --env-url https://your-space.hf.space

# Run specific tasks
python examples/simple_agent.py --tasks task1 task2 task3 --max-attempts 3
```

**Features:**
- Direct code submission
- Multiple retry attempts
- Uses LLM to generate fixes

### 2. Tool-Using Agent (`tool_agent.py`)

An advanced agent that uses debugging tools to iteratively analyze bugs
before submitting a fix. Demonstrates the full tool-calling API.

```bash
# Run single task
python examples/tool_agent.py --env-url http://localhost:7860 --task task1

# Run all tasks with more turns
python examples/tool_agent.py --env-url http://localhost:7860 --all-tasks --max-turns 15

# Run a hard task
python examples/tool_agent.py --task task6 --max-turns 15
```

**Features:**
- Uses `execute_snippet` to test hypotheses
- Uses `inspect_tensor` to check shapes/gradients
- Uses `get_variable_state` to inspect variables
- Uses `run_training_probe` to test fixes
- Uses `inspect_diff` to review changes
- Iterative debugging before submission

## Tool Usage Guide

### execute_snippet
Run a quick Python snippet to test something:
```python
action = {
    "action_type": "execute_snippet",
    "code": "import torch; print(torch.__version__)"
}
```

### inspect_tensor
Check tensor properties:
```python
action = {
    "action_type": "inspect_tensor",
    "setup_code": "import torch; t = torch.randn(3, 4, requires_grad=True)",
    "target_expression": "t"
}
# Returns: shape, dtype, requires_grad, grad_is_none, min/max/mean, is_nan, is_inf
```

### get_variable_state
Evaluate multiple expressions:
```python
action = {
    "action_type": "get_variable_state",
    "setup_code": "import torch; model = torch.nn.Linear(10, 2)",
    "expressions": ["model.weight.shape", "model.training", "list(model.parameters())"]
}
```

### run_training_probe
Test if code trains properly:
```python
action = {
    "action_type": "run_training_probe",
    "code": "import torch\n# Full training script",
    "steps": 5
}
# Returns: losses per step, gradient norms, optimizer param count
```

### inspect_diff
Preview changes before submitting:
```python
action = {
    "action_type": "inspect_diff",
    "proposed_code": "import torch\n# Your fixed code"
}
# Returns: unified diff, lines_changed, additions, deletions
```

### submit_fix
Submit final solution:
```python
action = {
    "action_type": "submit_fix",
    "fixed_code": "import torch\n# Complete fixed script"
}
# Returns: reward (0.0-1.0), episode_done=True
```

## Debugging Tips

1. **Start with inspection**: Use `get_variable_state` to understand the code structure
2. **Check shapes**: Use `inspect_tensor` to verify tensor dimensions match
3. **Test incrementally**: Use `run_training_probe` with `steps=2` to quickly verify fixes
4. **Review before submit**: Always use `inspect_diff` to catch mistakes
5. **Handle errors gracefully**: Tool outputs include error messages if execution fails

## Expected Scores

| Task | Difficulty | Expected Score (good fix) |
|------|------------|---------------------------|
| task1 | Easy | 0.85-1.0 |
| task2 | Medium | 0.75-1.0 |
| task3 | Hard | 0.70-0.95 |
| task4 | Medium | 0.75-1.0 |
| task5 | Medium | 0.80-1.0 |
| task6 | Hard | 0.65-0.95 |

## API Reference

### Reset Endpoint
```python
POST /reset
{"task_id": "task1"}
# Returns observation with buggy_code, task_description
```

### Step Endpoint
```python
POST /step
{"action": {"action_type": "...", ...}}
# Returns observation, reward, done
```

### Tools Endpoint
```python
GET /tools
# Returns list of available tools with schemas
```

### State Endpoint
```python
GET /state
# Returns current session state (turn, task_id, submitted)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `HF_TOKEN` | HuggingFace token | Required |
| `OPENAI_API_KEY` | Alternative to HF_TOKEN | Optional |
