# Dockerfile — HotpotQA Multi-Hop Agent
#
# Build:
#   docker build -t hotpotqa-agent .
#
# Run (GPU, pre-built FAISS index mounted from host):
#   docker run --gpus all \
#     -p 8000:8000 -p 7860:7860 \
#     -v $(pwd)/data:/app/data \
#     -e LOAD_INDEX=1 \
#     hotpotqa-agent
#
# The image exposes two ports:
#   8000 — FastAPI REST API  (api.py)
#   7860 — Gradio demo UI   (serve.py)
#
# By default only the FastAPI server starts. To also start Gradio,
# override CMD or use docker-compose.

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    pip3 install fastapi uvicorn[standard]

# Copy project source
COPY . .

# Create data dir in case it's not mounted
RUN mkdir -p /app/data

EXPOSE 8000 7860

# Default: start FastAPI server
# Set LOAD_INDEX=1 if data/faiss.index is mounted, 0 to rebuild from corpus
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
