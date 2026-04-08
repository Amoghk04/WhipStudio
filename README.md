---
title: WhipStudio Env
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /ui/
---

# 🔧 WhipStudio — ML Debug Arena

An OpenEnv-compatible RL environment where agents debug broken PyTorch training scripts.
Features **6 debugging tasks** with continuous reward scoring (0.0-1.0).

## 🎯 Overview

WhipStudio presents agents with broken ML training code and challenges them to fix it.
Agents must diagnose bugs, fix all issues, and meet performance thresholds.

## 📋 Tasks

| Task | Difficulty | Bug Type |
|------|------------|----------|
| task1 | Easy | Wrong optimizer order + bad LR |
| task2 | Medium | Silent NaN from log(0) |
| task3 | Hard | OOM leak + data leakage |
| task4 | Medium | Wrong loss function |
| task5 | Medium | Frozen backbone |
| task6 | Hard | IO mismatch (4 bugs) |

## 🚀 Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r server/requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Inference

```bash
# Set required environment variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export HF_TOKEN="your_token"

# Run inference
python inference.py --env-url http://localhost:7860
```

### Docker

```bash
docker build -t whipstudio .
docker run -p 7860:7860 whipstudio
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode with `{"task_id": "task1"}` |
| `/step` | POST | Submit fix with `{"action": {"action_type": "submit_fix", "fixed_code": "..."}}` |
| `/state` | GET | Get current session state |
| `/tasks` | GET | List available tasks |
| `/health` | GET | Health check (returns 200) |

## 📊 Inference Output Format

The `inference.py` script emits structured logs:

```
[START] task_id=task1
[STEP] task_id=task1 step=1 action=submit_fix(1234chars) reward=0.4500 done=true
[END] task_id=task1 final_score=0.4500
```

## 🏗️ Project Structure

```
whipstudio/
├── server/
│   ├── app.py              # FastAPI application
│   ├── environment.py      # OpenEnv environment
│   └── tasks/              # Task definitions + graders
├── inference.py            # Hackathon inference script
├── models.py               # Pydantic schemas
├── openenv.yaml            # OpenEnv specification
├── Dockerfile
└── README.md
```

## ✅ Hackathon Compliance

- ✅ HF Space deploys and responds to `/health` (200)
- ✅ OpenEnv spec compliance (`openenv.yaml`, typed models, `/reset`, `/step`, `/state`)
- ✅ Dockerfile builds
- ✅ `inference.py` uses OpenAI client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- ✅ Structured stdout logs: `[START]`, `[STEP]`, `[END]`
- ✅ 6 tasks with graders returning scores in 0.0-1.0 range
- ✅ Runtime < 20 min, runs on vcpu=2, memory=8gb

## 📝 License

Apache-2.0
