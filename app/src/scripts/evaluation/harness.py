"""
Usage:

    python -m src.scripts.evaluation.harness --input data/validation_tickets.json --output evaluation/results/
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import statistics
import time
import datetime
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from google.adk.sessions import InMemorySessionService, Session
from google.adk import Runner
from google.genai import types
import requests

from src.config import CONFIG
from src.agents.agent import root_agent
from src.logging_store import reconcile
from src.schemas import PipelineResult, RawTicket


logger = logging.getLogger(__name__)


STATIC_USER_ID = f"user_{uuid4().hex}"
STATIC_SESSION_ID = f"session_{uuid4().hex}"


async def setup_session_and_runner():
    session_service = InMemorySessionService()
    session = await session_service.get_session(
        app_name=root_agent.name,
        user_id=STATIC_USER_ID,
        session_id=STATIC_SESSION_ID,
    )

    if not session:
        session = await session_service.create_session(
            app_name=root_agent.name,
            user_id=STATIC_USER_ID,
            session_id=STATIC_SESSION_ID,
        )
    runner = Runner(
        agent=root_agent, app_name=root_agent.name, session_service=session_service
    )
    return session, runner


async def process_ticket(
    ticket: RawTicket, session: Session, runner: Runner
) -> PipelineResult:
    start_time = time.perf_counter()

    message_content = ticket.model_dump_json()
    message = types.Content(role="user", parts=[types.Part(text=message_content)])
    final_event = None
    async for event in runner.run_async(
        user_id=STATIC_USER_ID,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response():
            final_event = event

    if not final_event:
        RuntimeError("Agent did not produce a final response")

    result: PipelineResult = final_event.output
    result.latency_seconds = time.perf_counter() - start_time
    return result


def _has_labels(tickets: List[Dict]) -> bool:
    return bool(tickets) and isinstance(tickets[0].get("labels"), dict)


def round_nested_dict(dict1: Dict) -> Dict:
    for k in dict1.keys():
        if isinstance(dict1[k], float):
            dict1[k] = round(dict1[k], 2)
        elif isinstance(dict1, dict):
            dict1[k] = round_nested_dict(dict1[k])
    return dict1


def _classification_report(y_true, y_pred) -> Dict:
    try:
        from sklearn.metrics import classification_report

        return round_nested_dict(
            classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        )
    except Exception:
        return {}


@dataclass
class RoutePrediction:
    confidence: float
    correct: bool


def calibration_table(predictions, bands=5):
    """Compare stated confidence against observed accuracy per band."""
    rows = []
    for i in range(bands):
        low, high = i / bands, (i + 1) / bands
        group = [p for p in predictions if low <= p.confidence < high]
        if not group:
            continue
        stated = sum(p.confidence for p in group) / len(group)
        observed = sum(1 for p in group if p.correct) / len(group)
        rows.append(
            {
                "confidence_band_range": f"{low:.2f} - {high:.2f}",
                "confidence_band_count": len(group),
                "stated_confidence": round(stated, 2),
                "observed": round(observed, 2),
                "calibration_error": round(stated - observed, 2),
            }
        )
    return rows


async def run(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    tickets = json.loads(input_path.read_text())

    session, runner = await setup_session_and_runner()

    # a sanity check by checking the whether the first ticket has a label
    labeled = _has_labels(tickets)

    per_ticket_rows = []
    latencies = []
    outcome_counts = {"auto_respond": 0, "escalate": 0, "blocked_escalate": 0}
    guardrail_block_counts: Dict[str, int] = {}
    pii_detections = 0
    processed_ticket_ids = set()

    y_true_intent, y_pred_intent = [], []
    y_true_route, y_pred_route = [], []
    route_predictions = []
    retrieval_hits, retrieval_total = 0, 0

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    wall_clock_start = time.perf_counter()

    for raw in tickets:
        try:
            result = await process_ticket(
                ticket=RawTicket.model_validate(raw), session=session, runner=runner
            )
        except Exception as exc:
            per_ticket_rows.append(
                {"ticket_id": raw.get("ticket_id", "UNKNOWN"), "error": str(exc)}
            )
            continue

        processed_ticket_ids.add(result.ticket_id)
        latencies.append(result.latency_seconds)
        outcome_counts[result.final_action] = (
            outcome_counts.get(result.final_action, 0) + 1
        )

        if result.guardrails:
            for check, outcome in result.guardrails.checks.items():
                if outcome == "fail":
                    guardrail_block_counts[check] = (
                        guardrail_block_counts.get(check, 0) + 1
                    )
                    if check == "private_data":
                        pii_detections += 1
        row = result.to_summary()

        if labeled:
            labels = raw.get("labels", {}) or {}
            true_intent = labels.get("intent")
            if true_intent:
                y_true_intent.append(true_intent)
                y_pred_intent.append(result.classification.intent)

            expected_route = labels.get("expected_route")
            if expected_route:
                predicted_route = (
                    "auto_respond"
                    if result.final_action.value == "auto_respond"
                    else "escalate"
                )
                y_true_route.append(expected_route)
                y_pred_route.append(predicted_route)

                route_predictions.append(
                    RoutePrediction(
                        confidence=result.routing.classification_confidence,
                        correct=expected_route == predicted_route,
                    )
                )

            expected_doc_ids = set(labels.get("expected_doc_ids", []) or [])
            if expected_doc_ids:
                retrieval_total += 1
                retrieved_ids = {p.doc_id for p in result.valid_docs}
                if retrieved_ids & expected_doc_ids:
                    retrieval_hits += 1

            row["expected_intent"] = true_intent
            row["expected_route"] = expected_route
            row["expected_doc_ids"] = sorted(expected_doc_ids)

        per_ticket_rows.append(row)

        requests.post(
            url=f"{CONFIG.metrics_app_url}/log_metrics", json=result.model_dump()
        )

    wall_clock_seconds = time.perf_counter() - wall_clock_start

    reconciliation = reconcile(processed_ticket_ids)

    total = len(tickets)
    answered = outcome_counts.get("auto_respond", 0)
    escalated = outcome_counts.get("escalate", 0) + outcome_counts.get(
        "blocked_escalate", 0
    )

    report: Dict = {
        "run_metadata": {
            "started_at": started_at,
            "input_file": str(input_path),
            "tickets_in_file": total,
            "tickets_processed": len(processed_ticket_ids),
            "wall_clock_seconds": round(wall_clock_seconds, 3),
        },
        "volume": {
            "tickets_processed": len(processed_ticket_ids),
            "answered_automatically": answered,
            "escalated": escalated,
            "blocked_by_guardrail": outcome_counts.get("blocked_escalate", 0),
        },
        "business_outcomes": {
            "first_contact_resolution_pct": round(100.0 * answered / total, 2)
            if total
            else 0.0,
            "escalation_rate_pct": round(100.0 * escalated / total, 2)
            if total
            else 0.0,
        },
        "technical_performance": {
            "latency_mean_seconds": round(statistics.mean(latencies), 4)
            if latencies
            else None,
            "latency_median_seconds": round(statistics.median(latencies), 4)
            if latencies
            else None,
            "latency_p95_seconds": round(
                sorted(latencies)[int(0.95 * len(latencies)) - 1], 4
            )
            if len(latencies) >= 2
            else (latencies[0] if latencies else None),
        },
        "governance": {
            "decision_log_reconciliation": reconciliation,
            "guardrail_block_counts": guardrail_block_counts,
            "private_data_detections": pii_detections,
        },
    }

    if labeled:
        report["technical_performance"]["intent_classification_report"] = (
            _classification_report(y_true_intent, y_pred_intent)
        )
        if y_true_route:
            correct = sum(1 for a, b in zip(y_true_route, y_pred_route) if a == b)
            report["technical_performance"]["routing_accuracy_pct"] = round(
                100.0 * correct / len(y_true_route), 2
            )
            report["technical_performance"]["routing_calibration_table"] = (
                calibration_table(route_predictions, 10)
            )
        if retrieval_total:
            report["technical_performance"]["retrieval_hit_rate_pct"] = round(
                100.0 * retrieval_hits / retrieval_total, 2
            )

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y_%m_%d_%H_%M_%S"
    )
    report_path = output_dir / f"evaluation_report_{timestamp}.json"
    per_ticket_path = output_dir / f"per_ticket_results_{timestamp}.json"

    report_path.write_text(json.dumps(report, indent=2))
    per_ticket_path.write_text(json.dumps(per_ticket_rows, indent=2))

    print(
        f"Processed {len(processed_ticket_ids)}/{total} tickets in {wall_clock_seconds:.2f}s",
        flush=True,
    )
    print(f"Report written to {report_path}", flush=True)
    print(f"Per-ticket results written to {per_ticket_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CloudServe evaluation harness."
    )
    parser.add_argument("--input", required=True, help="Path to a ticket JSON file.")
    parser.add_argument(
        "--output", required=True, help="Directory to write the metrics report into."
    )
    args = parser.parse_args()
    asyncio.run(run(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
