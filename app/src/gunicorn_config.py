import multiprocessing
import os
from dotenv import load_dotenv

load_dotenv()

workers = int(os.getenv("GUNICORN_WORKERS", "0")) or (
    multiprocessing.cpu_count() * 2 + 1
)

bind = "0.0.0.0:8000"

# Worker configuration
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout configuration (seconds)
timeout = 300
graceful_timeout = 60
keepalive = 5

# Worker lifecycle
max_requests = 1000
max_requests_jitter = 50
