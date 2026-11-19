from prometheus_client import Counter, Histogram, start_http_server

from src.schemas import Action, PipelineResult


TICKETS = Counter(
    "tickets_processed_total", "Tickets processed", ["channel", "outcome"]
)

LATENCY = Histogram(
    "response_seconds",
    "End to end response time",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

CONFIDENCE = Histogram(
    "response_confidence",
    "Distribution of AI response confidence scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

GUARDRAIL = Counter("guardrail_blocks_total", "Responses blocked", ["guardrail"])


def update_metrics(result: PipelineResult):
    LATENCY.observe(result.latency_seconds)

    CONFIDENCE.observe(result.classification.intent_confidence)

    outcome = (
        "auto_respond" if result.final_action == Action.AUTO_RESPOND else "escalate"
    )

    TICKETS.labels(channel=result.channel, outcome=outcome).inc()

    if result.guardrails and result.guardrails.blocked:
        GUARDRAIL.labels(guardrail="blocked").inc()


def start_metrics_server():
    start_http_server(8001)
