import json
import logging

import requests
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.schemas import PipelineResult
from src.config import CONFIG


logger = logging.getLogger(__name__)


# Only these endpoints are monitored
MONITORED_ENDPOINTS = {
    "/run_sse",
}


class MetricsLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Ignore non-HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "").rstrip("/") or "/"
        # Ignore all endpoints except the monitored ones
        if path not in MONITORED_ENDPOINTS:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        endpoint = path

        status_code = None
        response_complete = False
        response = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_complete, response

            if message["type"] == "http.response.start":
                status_code = message["status"]

            elif message["type"] == "http.response.body":
                if message.get("body", None):
                    response = message["body"]
                if not message.get("more_body", False):
                    response_complete = True

            await send(message)

        try:
            app_response = await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "Request failed | method=%s endpoint=%s",
                method,
                endpoint,
            )
            raise
        try:
            response = PipelineResult.model_validate(
                json.loads(response[6:].strip())["output"]
            )
        except Exception:
            logger.warning("Logging metrics failed")
        if response_complete and response is not None:
            try:
                log_metric_resp = requests.post(
                    url=f"{CONFIG.metrics_app_url}/log_metrics",
                    json=response.model_dump(),
                )
                log_metric_resp.raise_for_status()
            except Exception:
                logger.warning("Logging metrics failed")
        return app_response
