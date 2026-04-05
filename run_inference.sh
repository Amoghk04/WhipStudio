#!/bin/bash
# Quick setup script for WhipStudio OpenEnv Hackathon

set -e

echo "=========================================="
echo "WhipStudio Hackathon Setup"
echo "=========================================="

# Step 1: Check environment variables
echo ""
echo "Step 1: Checking environment variables..."

if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN not set"
    if [ -f .env ]; then
        echo "   Loading from .env file..."
        export HF_TOKEN=$(grep -v '^#' .env | head -1)
        echo "   ✓ HF_TOKEN loaded"
    else
        echo "   ❌ Please set HF_TOKEN environment variable or create .env file"
        exit 1
    fi
else
    echo "   ✓ HF_TOKEN is set"
fi

if [ -z "$API_BASE_URL" ]; then
    echo "⚠️  API_BASE_URL not set, using HuggingFace Inference API"
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
fi
echo "   ✓ API_BASE_URL: $API_BASE_URL"

if [ -z "$MODEL_NAME" ]; then
    echo "⚠️  MODEL_NAME not set, using default"
    export MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
fi
echo "   ✓ MODEL_NAME: $MODEL_NAME"

# Step 2: Check HF Space
ENV_URL="${1:-https://amogh-kal1-whipstudio.hf.space}"
echo ""
echo "Step 2: Checking HF Space at $ENV_URL..."

if curl -s --max-time 10 "$ENV_URL/health" > /dev/null 2>&1; then
    echo "   ✓ HF Space is reachable"
else
    echo "   ❌ HF Space not reachable or still starting up"
    echo "   Try visiting $ENV_URL in your browser first"
    exit 1
fi

# Step 3: Run inference
echo ""
echo "Step 3: Running inference..."
echo ""

python3 inference.py --env-url "$ENV_URL"

echo ""
echo "=========================================="
echo "✅ Inference complete!"
echo "=========================================="
