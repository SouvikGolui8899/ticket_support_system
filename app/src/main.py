from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pathlib import Path
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import InMemorySessionService
from google.adk import Runner
from google.genai import types
import requests

from src.middleware import MetricsLogMiddleware
from src.config import CONFIG
from src.agents.agent import root_agent
from src.db_models import create_all_tables
from src.schemas import PipelineResult, RawTicket


CURRENT_PATH = Path(__file__)
AGENT_DIR = str(CURRENT_PATH.parent)
SQLITE_URI_FOR_WEB = r"sqlite:///./storage/web_agent.db"


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=CONFIG.log_level,
)
logger = logging.getLogger(__name__)

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    session_service = InMemorySessionService()
    runner_dict = dict()
    runner_dict["ticket_support_system"] = Runner(
        app_name=root_agent.name,
        agent=root_agent,
        session_service=session_service,
    )
    app.state.runner_dict = runner_dict
    app.state.session_service = session_service
    create_all_tables()
    yield
    runner_dict.clear()


# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "http://localhost:8000",
#     "http://127.0.0.1:8000",
#     "http://localhost:9090",
#     "http://127.0.0.1:9090",
# ]


web_app = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=SQLITE_URI_FOR_WEB,
    web=True,
)


app = FastAPI(
    title="CloudServe Support System",
    description=(
        "Auto-responds to support tickets and sends low confident tickets for triage"
    ),
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.post("/tickets/process")
async def process_ticket(request: Request, ticket: RawTicket):
    start_time = time.perf_counter()

    runner: Runner = request.app.state.runner_dict.get("ticket_support_system")

    user_id = f"user_{uuid4().hex}"
    session_id = f"session_{uuid4().hex}"

    session = await request.app.state.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session_id,
    )

    if not session:
        session = await request.app.state.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )

    message_content = ticket.model_dump_json()
    message = types.Content(role="user", parts=[types.Part(text=message_content)])

    final_event = None
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=message,
        ):
            if event.is_final_response():
                final_event = event

        if final_event is None:
            raise RuntimeError("Agent did not produce a final response")

        result: PipelineResult = final_event.output
        result.latency_seconds = time.perf_counter() - start_time
        log_metric_resp = requests.post(
            url=f"{CONFIG.metrics_app_url}/log_metrics", json=result.model_dump()
        )
        log_metric_resp.raise_for_status()
        return result.to_summary()
    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(status_code=500, detail="Processing Error") from ex


app.mount("/", web_app)

app.add_middleware(MetricsLogMiddleware)
