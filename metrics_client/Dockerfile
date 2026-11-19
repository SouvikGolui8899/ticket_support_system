FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup appuser

COPY ./src src
COPY ./requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8001
