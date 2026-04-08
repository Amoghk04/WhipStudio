# WhipStudio API Documentation

Complete API reference for the WhipStudio ML Debugging environment.

## Base URL

- Local: `http://localhost:7860`
- HF Space: `https://your-space.hf.space`

---

## Endpoints

### POST /reset

Start a new debugging episode.

**Request:**
```json
{
  "task_id": "task1"
}
```

**Response:**
```json
{
  "observation": {
    "task_id": "task1",
    "task_description": "This 2-class linear classifier training loop has bugs...",
    "buggy_code": "import torch\nimport torch.nn as nn...",
    "error_log": "",
    "last_reward": 0.0,
    "turn": 0,
    "episode_done": false
  }
}
```

---

### POST /step

Execute an action (tool call or submit_fix).

**Request:**
```json
{
  "action": {
    "action_type": "submit_fix",
    "fixed_code": "import torch\n..."
  }
}
```

**Response:**
```json
{
  "observation": { /* action-specific fields */ },
  "reward": 0.85,
  "done": true
}
```

---

### GET /state

Get current session state.

**Response:**
```json
{
  "turn": 3,
  "task_id": "task1",
  "submitted": false,
  "best_reward": 0.0,
  "tool_call_history": [
    {"turn": 1, "action_type": "execute_snippet"},
    {"turn": 2, "action_type": "inspect_tensor"}
  ]
}
```

---

### GET /tasks

List available debugging tasks.

**Response:**
```json
{
  "tasks": [
    {"id": "task1", "name": "Broken Training Loop", "difficulty": "easy"},
    {"id": "task2", "name": "Silent NaN Loss", "difficulty": "medium"},
    {"id": "task3", "name": "OOM + Data Leakage", "difficulty": "hard"},
    {"id": "task4", "name": "Wrong Loss Function", "difficulty": "medium"},
    {"id": "task5", "name": "Frozen Backbone", "difficulty": "medium"},
    {"id": "task6", "name": "Input-Output mismatch", "difficulty": "hard"}
  ]
}
```

---

### GET /tools

List available debugging tools with schemas.

**Response:**
```json
{
  "tools": [
    {
      "name": "execute_snippet",
      "description": "Run a Python code snippet...",
      "action_schema": { /* JSON Schema */ },
      "observation_schema": { /* JSON Schema */ }
    },
    // ... more tools
  ]
}
```

---

### POST /grader

Manually grade a code submission.

**Request:**
```json
{
  "task_id": "task1",
  "code": "import torch\n..."
}
```

**Response:**
```json
{
  "score": 0.85,
  "details": {
    "val_acc": 0.92,
    "final_loss": 0.15
  }
}
```

---

### GET /baseline

Run baseline agent on all tasks.

**Query Parameters:**
- `model_id` (optional): LLM model to use (default: `Qwen/Qwen2.5-Coder-32B-Instruct`)
- `use_tools` (optional): Enable tool use (default: `true`)

**Response:**
```json
{
  "baseline_scores": {
    "task1": 0.85,
    "task2": 0.72,
    "task3": 0.65,
    "task4": 0.78,
    "task5": 0.80,
    "task6": 0.60
  },
  "average": 0.73,
  "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
  "use_tools": true
}
```

---

### GET /baseline/task/{task_id}

Run baseline agent on a single task with detailed output.

**Query Parameters:**
- `model_id` (optional): LLM model to use
- `use_tools` (optional): Enable tool use (default: `true`)

**Response:**
```json
{
  "task_id": "task1",
  "score": 0.85,
  "status": "ok",
  "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
  "use_tools": true,
  "max_turns": 8,
  "fixed_code": "import torch\n...",
  "output": "LOSSES:[0.8, 0.5, 0.3...]",
  "attempts": [ /* per-turn details */ ],
  "tool_history": [ /* tool call results */ ]
}
```

---

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "version": "1.1.0",
  "tasks_available": 6,
  "tools_available": 6
}
```

---

### GET /metrics

Runtime metrics (if enabled).

**Response:**
```json
{
  "total_resets": 150,
  "total_steps": 423,
  "avg_reward": 0.52,
  "task_distribution": {
    "task1": 45,
    "task2": 30,
    "task3": 20,
    "task4": 25,
    "task5": 20,
    "task6": 10
  },
  "tool_usage": {
    "execute_snippet": 89,
    "inspect_tensor": 45,
    "submit_fix": 150
  }
}
```

---

## Action Schemas

### submit_fix

Submit a final fix for grading.

```json
{
  "action_type": "submit_fix",
  "fixed_code": "import torch\nimport torch.nn as nn\n...",
  "explanation": "Fixed learning rate and optimizer order",
  "attempt_number": 1
}
```

**Observation:**
```json
{
  "action_type": "submit_fix",
  "turn": 5,
  "episode_done": true,
  "reward": 0.85,
  "error_log": "LOSSES:[0.8, 0.5, 0.3, 0.2, 0.15]\nVAL_ACC:0.92",
  "grader_details": {
    "val_acc": 0.92,
    "final_loss": 0.15,
    "losses": [0.8, 0.5, 0.3, 0.2, 0.15]
  }
}
```

---

### execute_snippet

Run a Python code snippet to test hypotheses.

```json
{
  "action_type": "execute_snippet",
  "code": "import torch\nprint(torch.__version__)\nprint(torch.cuda.is_available())"
}
```

**Observation:**
```json
{
  "action_type": "execute_snippet",
  "turn": 1,
  "episode_done": false,
  "stdout": "2.2.0\nFalse\n",
  "stderr": "",
  "exit_code": 0,
  "timed_out": false
}
```

---

### inspect_tensor

Inspect tensor shape, dtype, gradients, and statistics.

```json
{
  "action_type": "inspect_tensor",
  "setup_code": "import torch\nimport torch.nn as nn\nmodel = nn.Linear(10, 2)\nx = torch.randn(5, 10)\ny = model(x)\nloss = y.sum()\nloss.backward()",
  "target_expression": "model.weight.grad"
}
```

**Observation:**
```json
{
  "action_type": "inspect_tensor",
  "turn": 2,
  "episode_done": false,
  "shape": [2, 10],
  "dtype": "torch.float32",
  "requires_grad": false,
  "grad_is_none": false,
  "min_val": -1.234,
  "max_val": 2.567,
  "mean_val": 0.123,
  "is_nan": false,
  "is_inf": false,
  "error": null
}
```

---

### run_training_probe

Run N training steps and observe loss curve and gradients.

```json
{
  "action_type": "run_training_probe",
  "code": "import torch\nimport torch.nn as nn\n...",
  "steps": 5
}
```

**Observation:**
```json
{
  "action_type": "run_training_probe",
  "turn": 3,
  "episode_done": false,
  "losses": [0.8, 0.65, 0.52, 0.41, 0.33],
  "grad_norms": {
    "layer1.weight": 0.234,
    "layer1.bias": 0.089,
    "layer2.weight": 0.156
  },
  "optimizer_param_count": 122,
  "final_loss": 0.33,
  "loss_is_nan": false,
  "loss_is_inf": false,
  "stderr": "",
  "timed_out": false
}
```

---

### get_variable_state

Evaluate multiple expressions and return their state.

```json
{
  "action_type": "get_variable_state",
  "setup_code": "import torch\nmodel = torch.nn.Linear(10, 2)\noptimizer = torch.optim.Adam(model.parameters(), lr=0.01)",
  "expressions": [
    "model.training",
    "optimizer.param_groups[0]['lr']",
    "list(model.parameters())[0].shape"
  ]
}
```

**Observation:**
```json
{
  "action_type": "get_variable_state",
  "turn": 4,
  "episode_done": false,
  "results": {
    "model.training": {
      "repr": "True",
      "type": "bool",
      "value": true,
      "shape": null,
      "error": null
    },
    "optimizer.param_groups[0]['lr']": {
      "repr": "0.01",
      "type": "float",
      "value": 0.01,
      "shape": null,
      "error": null
    },
    "list(model.parameters())[0].shape": {
      "repr": "torch.Size([2, 10])",
      "type": "torch.Size",
      "value": null,
      "shape": [2, 10],
      "error": null
    }
  }
}
```

---

### inspect_diff

Compare proposed fix against original buggy code.

```json
{
  "action_type": "inspect_diff",
  "proposed_code": "import torch\nimport torch.nn as nn\n# Fixed version..."
}
```

**Observation:**
```json
{
  "action_type": "inspect_diff",
  "turn": 5,
  "episode_done": false,
  "diff": "--- original\n+++ proposed\n@@ -10,7 +10,7 @@\n-    lr = 10.0\n+    lr = 0.01\n",
  "lines_changed": 5,
  "additions": 3,
  "deletions": 2
}
```

---

## Error Responses

### Invalid Task ID
```json
{
  "error": "Unknown task_id: task99. Available: task1, task2, task3, task4, task5, task6"
}
```

### Episode Already Complete
```json
{
  "error": "Episode already complete. Call /reset to start a new episode.",
  "episode_done": true,
  "reward": 0.0
}
```

### Max Turns Exceeded
```json
{
  "error": "Maximum turns (10) exceeded",
  "episode_done": true,
  "reward": 0.0,
  "turn": 11
}
```

### Tool Execution Error
```json
{
  "action_type": "execute_snippet",
  "turn": 3,
  "episode_done": false,
  "stdout": "",
  "stderr": "NameError: name 'undefined_var' is not defined",
  "exit_code": 1,
  "timed_out": false
}
```

### Security Violation
```json
{
  "error": "Security violation: import 'requests' is not allowed. Allowed: torch, numpy, sklearn, pandas, matplotlib, scipy, math, random, os, sys, collections, itertools, functools, json, re, typing",
  "episode_done": false
}
```

---

## Rate Limits

- `/baseline`: 1 request per 3 minutes (runs all 6 tasks)
- `/baseline/task/{id}`: 1 request per 30 seconds
- `/step`: 60 requests per minute
- `/reset`: 30 requests per minute

---

## Authentication

No authentication required for local deployments.
HF Space deployments use HuggingFace token for baseline agent LLM calls.

Set environment variables:
```bash
export HF_TOKEN="your_token"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
```

---

## WebSocket Support

Not currently supported. Use polling with `/state` endpoint for real-time updates.

---

## OpenEnv Compliance

This environment follows the [OpenEnv specification](https://github.com/huggingface/openenv):

- `openenv.yaml`: Environment metadata and configuration
- Typed Pydantic models for actions and observations
- Standard endpoints: `/reset`, `/step`, `/state`, `/tasks`
- Continuous reward scoring (0.0-1.0)
- Episode-based interaction model
