# 🎯 SOLUTION: Updated HuggingFace API Configuration (2026)

## What Changed

HuggingFace **deprecated** the old API endpoint and unified everything under **Inference Providers**:

❌ **OLD (No longer works):**
```
https://api-inference.huggingface.co/v1
```

✅ **NEW (Works now):**
```
https://router.huggingface.co/v1
```

## 🚀 Quick Fix

Run these commands to fix your setup:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="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"

# Test the API first
./test_api.sh

# Then run inference
python inference.py --env-url https://amogh-kal1-whipstudio.hf.space
```

## 📋 Recommended Models (Free Tier)

These models work with the free HuggingFace tier:

| Model | Size | Best For |
|-------|------|----------|
| `Qwen/Qwen2.5-7B-Instruct` | 8B | Code debugging ⭐ |
| `Qwen/Qwen3-8B` | 8B | General coding |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | General purpose |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Fast/lightweight |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | Balanced |

**Recommendation:** Start with `Qwen/Qwen2.5-7B-Instruct` - it's optimized for code and free!

## 🔧 Step-by-Step Fix

### 1. Update Environment Variables

```bash
# Add to ~/.bashrc or run before each session
echo 'export API_BASE_URL="https://router.huggingface.co/v1"' >> ~/.bashrc
echo 'export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"' >> ~/.bashrc
echo 'export HF_TOKEN="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"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Test the Configuration

```bash
./test_api.sh
```

You should see: `✅ API Test Successful!`

### 3. Run Inference

```bash
python inference.py --env-url https://amogh-kal1-whipstudio.hf.space
```

## 💡 Why This Works Now

1. **Unified API**: HuggingFace consolidated all inference providers into one endpoint
2. **Router System**: Automatically selects the fastest available provider (Groq, Together, Novita, etc.)
3. **Free Tier**: Many models are available on the free tier
4. **OpenAI Compatible**: Same API format as OpenAI

## 📚 Additional Resources

- **MODEL_OPTIONS.md** - Complete list of supported models and configurations
- **test_api.sh** - Test script to verify your API configuration
- **HACKATHON_GUIDE.md** - Full guide for training and evaluation

## 🎓 Training Models

You can now train with any supported model:

```bash
# Train with Qwen 2.5 7B
python improved_agent.py \
    --env_url https://amogh-kal1-whipstudio.hf.space \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --use_lora --use_4bit
```

## 📊 Evaluate Models

Compare multiple models easily:

```bash
# Compare base vs trained
python evaluate_mnist.py \
    --use_real_mnist \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --trained_model ./trained-model/best

# Compare multiple models
python evaluate_mnist.py \
    --use_real_mnist \
    --models Qwen/Qwen2.5-7B-Instruct \
             meta-llama/Llama-3.1-8B-Instruct \
             ./trained-model/best
```

## ⚠️ Important Notes

1. **Free tier limits**: ~10-20 requests per minute per model
2. **Larger models may need Pro**: Some 70B+ models require HuggingFace Pro
3. **Provider selection**: Router automatically picks fastest, or add `:cheapest` or `:together` suffix
4. **Rate limits**: If you hit limits, wait 60 seconds or upgrade to Pro

## ✅ Verification Checklist

- [ ] API_BASE_URL = `https://router.huggingface.co/v1`
- [ ] MODEL_NAME = `Qwen/Qwen2.5-7B-Instruct` (or another supported model)
- [ ] HF_TOKEN is set and valid
- [ ] `./test_api.sh` returns success
- [ ] `inference.py` runs without 410/400 errors
