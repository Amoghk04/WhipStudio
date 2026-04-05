#!/bin/bash
# Test HuggingFace Inference Providers API

echo "Testing HuggingFace Inference Providers..."
echo ""

HF_TOKEN="${HF_TOKEN:-$(grep -v '^#' .env 2>/dev/null | head -1)}"

if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not set"
    exit 1
fi

echo "Testing with model: Qwen/Qwen2.5-7B-Instruct"
echo ""

response=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
  https://router.huggingface.co/v1/chat/completions \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in 5 words"}],
    "max_tokens": 20
  }')

http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | sed '/HTTP_CODE/d')

if [ "$http_code" = "200" ]; then
    echo "✅ API Test Successful!"
    echo ""
    echo "Response:"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
else
    echo "❌ API Test Failed (HTTP $http_code)"
    echo ""
    echo "Response:"
    echo "$body"
    exit 1
fi

echo ""
echo "===================="
echo "✅ Configuration is working!"
echo "Use this in your .bashrc or .env:"
echo ""
echo "export API_BASE_URL=\"https://router.huggingface.co/v1\""
echo "export MODEL_NAME=\"Qwen/Qwen2.5-7B-Instruct\""
echo "export HF_TOKEN=\"$HF_TOKEN\""
