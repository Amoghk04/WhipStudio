# Model Configuration for HuggingFace Inference Providers (2026)

## 🚀 The New HuggingFace API

HuggingFace has unified all inference under **Inference Providers** with a single endpoint:

```
https://router.huggingface.co/v1
```

❌ **Deprecated (no longer works):**
- `https://api-inference.huggingface.co/v1`

## ✅ Working Configuration

### For Code Generation (Recommended)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"  # Free tier supported
export HF_TOKEN="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"
```

### Best Models for ML Debugging (Free Tier)

| Model | Size | Best For | Status |
|-------|------|----------|--------|
| `Qwen/Qwen2.5-7B-Instruct` | 8B | Code generation ⭐ | ✅ Free |
| `Qwen/Qwen3-8B` | 8B | General coding | ✅ Free |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | General purpose | ✅ Free |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Fast & lightweight | ✅ Free |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | General purpose | ✅ Free |

### Larger Models (May require provider credits)

| Model | Size | Best For |
|-------|------|----------|
| `Qwen/Qwen3-Coder-Next` | 80B | Advanced code ⭐⭐ |
| `openai/gpt-oss-120b` | 120B | General purpose |
| `deepseek-ai/DeepSeek-R1` | 685B | Reasoning tasks |

## 🎯 Quick Fix for Your Setup

```bash
# Set the correct configuration
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export HF_TOKEN="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"

# Run inference
python inference.py --env-url https://amogh-kal1-whipstudio.hf.space
```

## 📝 How It Works

HuggingFace's new Inference Providers system:

1. **Router automatically selects fastest provider**: By default uses `:fastest` policy
2. **Multiple providers support each model**: Groq, Together, Novita, etc.
3. **Free tier available**: Many models work on free tier
4. **OpenAI-compatible API**: Drop-in replacement

### Provider Selection

You can specify a provider policy by appending to the model name:

```python
# Automatic (fastest)
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"

# Cheapest provider
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct:cheapest"

# Specific provider
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct:together"
```

## 🔧 Testing Model Availability

```bash
curl https://router.huggingface.co/v1/chat/completions \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

## 💡 Tips

1. **Start with Qwen/Qwen2.5-7B-Instruct** - best free model for code
2. **Use `:fastest` suffix** for quickest responses (default)
3. **Free tier limits**: ~10-20 requests/minute per model
4. **Upgrade to Pro**: More credits and higher limits

## 🆘 Troubleshooting

### "Model not supported by any provider"
- Check the [model list](https://huggingface.co/models?inference_provider=all&sort=trending&other=conversational)
- Try a different model from the supported list above
- Some models require HF Pro subscription

### "Rate limit exceeded"
- Wait a minute between requests
- Use smaller models
- Consider HF Pro for higher limits

## 📚 Resources

- [Inference Providers Docs](https://huggingface.co/docs/api-inference)
- [Available Models](https://huggingface.co/models?inference_provider=all)
- [HF Pro Subscription](https://huggingface.co/subscribe/pro)
