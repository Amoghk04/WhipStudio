# WhipStudio - OpenEnv Hackathon Submission Guide

Complete guide for running inference, training, and evaluation for the Scaler Meta PyTorch Hackathon.

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Set your HuggingFace token
export HF_TOKEN="your_token_here"

# For HuggingFace models (recommended)
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Or use the convenience script
./run_inference.sh https://amogh-kal1-whipstudio.hf.space
```

### 2. Run Hackathon Inference

The `inference.py` script meets all hackathon requirements:
- ✅ Uses OpenAI-compatible client
- ✅ Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
- ✅ Emits [START], [STEP], [END] logs
- ✅ Runs all 5 tasks with max 3 attempts each

```bash
python inference.py --env-url https://amogh-kal1-whipstudio.hf.space
```

## 📊 Training with GRPO

Train a model using Group Relative Policy Optimization:

### Basic Training
```bash
python improved_agent.py \
    --env_url https://amogh-kal1-whipstudio.hf.space \
    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output_dir ./trained-model \
    --num_iterations 50
```

### Memory-Efficient Training (8GB VRAM)
```bash
python improved_agent.py \
    --env_url https://amogh-kal1-whipstudio.hf.space \
    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --use_lora \
    --use_4bit \
    --gradient_checkpointing \
    --output_dir ./trained-model-lora
```

### Training Features
- **Curriculum Learning**: Starts with easier tasks, progresses to harder ones
- **LoRA Support**: Efficient fine-tuning with adapters
- **4-bit Quantization**: Train on GPUs with limited VRAM
- **Checkpoint Saving**: Best model saved automatically
- **Early Stopping**: Stops when no improvement
- **Wandb Logging**: Optional tracking with `--use_wandb`

## 🎯 Evaluation on MNIST

Compare base vs trained models on an out-of-distribution MNIST debugging task:

### Compare Two Models
```bash
python evaluate_mnist.py \
    --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --trained_model ./trained-model/best \
    --num_runs 3
```

### Use Real MNIST Dataset
```bash
python evaluate_mnist.py \
    --use_real_mnist \
    --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --trained_model ./trained-model/best
```

### Compare Multiple Models
```bash
python evaluate_mnist.py \
    --use_real_mnist \
    --models Qwen/Qwen2.5-Coder-1.5B-Instruct \
             Qwen/Qwen2.5-Coder-7B-Instruct \
             ./trained-model-v1/best \
             ./trained-model-v2/best
```

## 🔧 Configuration

### HuggingFace API (Recommended)
```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
export HF_TOKEN="hf_your_token"
```

### OpenAI API
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="sk-your-key"
```

### Local Model Inference
```bash
# Use vLLM or similar OpenAI-compatible server
export API_BASE_URL="http://localhost:8000/v1"
export MODEL_NAME="your-local-model"
export HF_TOKEN="dummy"  # Still required by script
```

## 📝 Hackathon Requirements Checklist

- ✅ **HF Space deploys**: https://amogh-kal1-whipstudio.hf.space
- ✅ **OpenEnv spec compliance**: openenv.yaml, typed models, endpoints
- ✅ **Dockerfile builds**: server/Dockerfile
- ✅ **inference.py exists**: Root directory
- ✅ **Uses OpenAI Client**: With API_BASE_URL, MODEL_NAME, HF_TOKEN
- ✅ **Structured logs**: [START], [STEP], [END] format
- ✅ **3+ tasks with graders**: 5 tasks (task1-task5)

## 🐛 Troubleshooting

### 500 Error from HF Space
```
[ERROR] Server error '500 Internal Server Error'
```

**Solution**: 
1. Visit your HF Space in a browser first: https://amogh-kal1-whipstudio.hf.space
2. Wait for it to fully start (cold start can take 1-2 minutes)
3. Check the Space logs for errors
4. Try the /health endpoint: `curl https://amogh-kal1-whipstudio.hf.space/health`

### Missing Dependencies
```bash
pip install openai httpx transformers torch trl peft bitsandbytes accelerate datasets
```

### Out of Memory During Training
Use memory-efficient options:
```bash
python improved_agent.py \
    --use_4bit \
    --use_lora \
    --gradient_checkpointing \
    --lora_r 8  # Lower rank for less memory
```

### HuggingFace API Rate Limits
If you hit rate limits with HuggingFace's free tier:
1. Use a smaller model (e.g., 1.5B instead of 32B)
2. Reduce `--num_iterations` for training
3. Reduce `--num_runs` for evaluation

## 📚 File Descriptions

| File | Purpose |
|------|---------|
| `inference.py` | **Hackathon submission script** - runs all tasks with structured logging |
| `improved_agent.py` | Train model with GRPO (curriculum learning, LoRA, 4-bit) |
| `evaluate_mnist.py` | Compare models on out-of-distribution MNIST debugging |
| `run_inference.sh` | Convenience script for quick inference runs |
| `baseline_agent.py` | Original baseline (not hackathon-compliant) |

## 🎓 Example Workflow

```bash
# 1. Run baseline inference
export HF_TOKEN="your_token"
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
python inference.py --env-url https://amogh-kal1-whipstudio.hf.space

# 2. Train model with GRPO
python improved_agent.py \
    --env_url https://amogh-kal1-whipstudio.hf.space \
    --use_lora --use_4bit \
    --num_iterations 30 \
    --output_dir ./my-trained-model

# 3. Evaluate on MNIST
python evaluate_mnist.py \
    --use_real_mnist \
    --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --trained_model ./my-trained-model/best \
    --num_runs 5

# 4. Validate submission
./vaidate-submission.sh https://amogh-kal1-whipstudio.hf.space
```

## 🏆 Tips for Best Results

1. **Start with small experiments**: Use `--num_iterations 10` first
2. **Monitor training**: Use `--use_wandb` to track progress
3. **Curriculum helps**: Keep `--curriculum_stages 3` for better learning
4. **Real MNIST is harder**: Expect lower scores but more realistic evaluation
5. **Multiple runs**: Use `--num_runs 5` for statistical significance

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your HF Space is running: visit the URL in browser
3. Check environment variables: `echo $API_BASE_URL $MODEL_NAME $HF_TOKEN`
4. Review the logs for detailed error messages
