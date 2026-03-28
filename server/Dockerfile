FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu torchvision==0.17.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=true

COPY server/start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
