# ML Debug Environment

An OpenEnv-compatible RL environment where agents debug broken PyTorch training scripts.

## Environment Description
The agent receives a broken Python training script and must return a corrected version.
Three tasks simulate real ML production bugs with increasing complexity.

## Action Space
- fixed_code (str, required): Complete corrected Python script
- explanation (str, optional): Description of bugs found
- attempt_number (int, 1-3): Which fix attempt this is

## Observation Space
- task_id: Which task (task1/task2/task3)
- task_description: Plain English instructions
- buggy_code: The broken script
- error_log: stdout+stderr from previous attempt
- last_reward: Score from previous attempt (0.0 on first step)
- metrics: {exit_code, elapsed_seconds, timed_out, step, best_reward_so_far}

## Reward Function
Continuous score 0.0–1.0. Partial credit for every improvement.
See `server/tasks/graders.py` for per-task scoring logic.

## Tasks
| Task | Difficulty | Bug Type |
|------|-----------|----------|
| task1 | Easy | Wrong optimizer order + bad learning rate |
| task2 | Medium | Silent NaN from log(0) numerical instability |
| task3 | Hard | OOM memory leak + train/val data leakage |

## Setup
```bash
pip install openenv-core
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Endpoints
POST /reset, POST /step, GET /state, GET /tasks, POST /grader, GET /baseline

## Gradio UI
Run the API server and the Gradio dashboard from the repository root:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
python gradio_app.py
```

By default the UI runs on `http://localhost:7861` and targets API base URL `http://localhost:8000`.
