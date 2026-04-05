FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false

CMD ["python", "-c", "import os; import uvicorn; uvicorn.run('server.app:app', host='0.0.0.0', port=int(os.getenv('PORT','7860')), proxy_headers=True, forwarded_allow_ips='*')"]
