FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup appuser

COPY ./src src
COPY ./data data
COPY ./prompts prompts
COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    mkdir -p /app/.adk /app/.config /app/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
    wget -O /app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz && \
    tar -xzf /app/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz -C /app/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
    chmod -R 755 /app/.adk /app/.config /app/.cache && \
    chown -R appuser:appgroup /app
    

USER appuser

EXPOSE 8000
