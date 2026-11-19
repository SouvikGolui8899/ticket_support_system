import datetime
import json
from typing import Dict, List, Optional, Set
import uuid

from sqlalchemy.orm import Session
import sqlalchemy as sa

from src.db_models import ENGINE, Decision


def log_decision(
    ticket_id: str,
    stage: str,
    input_summary: str,
    model_name: str,
    prediction: str,
    action_taken: str,
    reason: str,
    confidence: Optional[float],
    alternatives: Optional[List] = None,
    sources_used: Optional[List] = None,
    threshold_applied: Optional[float] = None,
    guardrail_results: Optional[Dict] = None,
) -> str:
    decision_id = str(uuid.uuid4())

    with Session(ENGINE) as session:
        decision_obj = Decision(
            decision_id=decision_id,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ticket_id=ticket_id,
            stage=stage,
            input_summary=input_summary[:500] if input_summary else "",
            model=model_name,
            prediction=prediction,
            confidence=confidence,
            alternatives=json.dumps(alternatives or []),
            sources_used=json.dumps(sources_used or []),
            threshold_applied=threshold_applied,
            action_taken=action_taken,
            reason=reason,
            guardrail_results=json.dumps(guardrail_results or {}),
        )
        session.add(decision_obj)
        session.commit()
    return decision_id


def reconcile(processed_ticket_ids: Set[str]) -> Dict:
    with Session(ENGINE) as session:
        stmt = sa.select(Decision.ticket_id).distinct()
        logged_ids = {row[0] for row in session.execute(stmt).all()}

    missing = processed_ticket_ids - logged_ids
    return {
        "tickets_processed": len(processed_ticket_ids),
        "tickets_logged": len(logged_ids),
        "reconciled": len(missing) == 0,
        "missing_ticket_ids": sorted(missing),
    }
