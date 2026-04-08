# WhipStudio Workflow Guide

## Understanding the Episode Flow

WhipStudio follows the standard OpenEnv episodic structure with debugging tools:

```
┌─────────────────────────────────────────────────────────────┐
│                       1. RESET                              │
│  POST /reset with task_id → Get buggy code & description   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              2. DEBUG (Optional, up to 10 turns)            │
│  Use tools to investigate the bug:                          │
│  • execute_snippet   - Run test code                        │
│  • inspect_tensor    - Check tensor properties              │
│  • run_training_probe - Verify partial fixes                │
│  • get_variable_state - Inspect variables                   │
│  • inspect_diff      - Preview your changes                 │
│                                                              │
│  Each tool call:                                            │
│  - Increments turn counter                                  │
│  - Returns episode_done=False                               │
│  - Returns reward=0.0                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. SUBMIT FIX (Once)                     │
│  POST /step with action_type="submit_fix"                  │
│  • Ends the episode (episode_done=True)                    │
│  • Returns final reward (0.0-1.0)                          │
│  • No more actions allowed after this                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│            4. NEW EPISODE (requires reset)                  │
│  To continue, call POST /reset again                        │
└─────────────────────────────────────────────────────────────┘
```

## Common Workflow Patterns

### Pattern 1: Direct Submission (No Tools)
```python
1. reset("task1")
2. submit_fix(fixed_code)  # Episode ends, get reward
3. reset("task2")          # Start new task
```

### Pattern 2: Iterative Debugging (Recommended)
```python
1. reset("task1")
2. execute_snippet("print(model)")           # Turn 1
3. inspect_tensor(setup, "model.weight")     # Turn 2
4. run_training_probe(partial_fix, steps=5)  # Turn 3
5. inspect_diff(final_fix)                   # Turn 4
6. submit_fix(final_fix)                     # Turn 5, episode ends
```

### Pattern 3: Maximum Tool Usage
```python
1. reset("task1")
2-10. Use 9 different tool calls to debug
11. submit_fix(fixed_code)  # Turn 10, episode ends
```

## Gradio UI Workflow

### ✅ Correct Usage:

1. **Select Task** → Click "Reset"
2. **Switch to "Debug Tools" tab**
3. Run tools (inspect_tensor, execute_snippet, etc.)
4. Analyze results
5. **Switch back to "Code Editor" tab**
6. Edit your fix
7. Click "Submit Fix (Final Submission)" **ONCE**

### ❌ Common Mistake:

- Clicking "Submit Fix" immediately → Episode ends with no debugging

## Episode Termination

An episode terminates when **ANY** of these occur:

1. ✅ `submit_fix` action is called
2. ❌ Turn limit exceeded (default: 10 turns)
3. ❌ Reward reaches 1.0 (perfect fix)

After termination:
- `episode_done=True` in response
- No further actions accepted
- Must call `/reset` to start new episode

## Turn Counter

- Increments on **every** action (tools + submit_fix)
- Visible in observations as `"turn": N`
- Maximum configurable via `MAX_TURNS_PER_EPISODE` (default: 10)

## Reward Signal

- **Tools return `reward=0.0`** (no penalty, no reward)
- **Only `submit_fix` returns non-zero reward** (0.0-1.0)
- Reward calculated by task-specific grader
- Measures how well the fix solves the bug

## API Examples

### Using Tools Before Submission

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1"}'

# Tool call 1: Execute snippet
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "execute_snippet",
      "code": "import torch\nprint(torch.__version__)"
    }
  }'
# Response: {"reward": 0.0, "episode_done": false, "turn": 1}

# Tool call 2: Inspect tensor
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "inspect_tensor",
      "setup_code": "import torch\nt = torch.randn(3, 4)",
      "target_expression": "t"
    }
  }'
# Response: {"reward": 0.0, "episode_done": false, "turn": 2}

# Final submission
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "submit_fix",
      "fixed_code": "import torch\n# ... fixed code ..."
    }
  }'
# Response: {"reward": 0.85, "episode_done": true, "turn": 3}
```

## FAQs

**Q: Why does clicking "Submit Fix" end the episode?**  
A: This is correct! `submit_fix` is the **terminal action**. Use tools first, submit last.

**Q: Can I submit multiple fixes?**  
A: No. Only one `submit_fix` per episode. Use `run_training_probe` to test fixes before final submission.

**Q: Do tool calls cost reward?**  
A: No. Tools always return `reward=0.0`. Only `submit_fix` is graded.

**Q: Can I use tools after submit_fix?**  
A: No. Episode ends immediately. You must reset to start a new episode.

**Q: How many tools can I use?**  
A: Up to 10 total actions (tools + submit_fix) per episode.

**Q: What if I exceed 10 turns?**  
A: Episode terminates with `reward=0.0` and error message.

## Best Practices

1. **Always use tools before submitting** - Don't waste your one submission
2. **Start with inspection tools** - Understand the bug before fixing
3. **Use run_training_probe** - Test partial fixes before final submission
4. **Use inspect_diff** - Review your changes before submitting
5. **Read error logs carefully** - They contain hints about the bug
6. **Stay within turn limit** - Plan your tool usage (≤9 tools + 1 submit)

## See Also

- [API Documentation](API.md) - Full endpoint reference
- [Tools Guide](TOOLS.md) - Detailed tool usage
- [Tasks Guide](TASKS.md) - Task descriptions & grading
