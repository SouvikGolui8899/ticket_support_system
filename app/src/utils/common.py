from typing import Any, List

from src.schemas import ClassificationResult, NormalizedTicket, RetrievedDocument


def clean_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def generate_escalation_summary(
    ticket: NormalizedTicket,
    classification: ClassificationResult,
    retrieved: List[RetrievedDocument],
    reason: str,
) -> dict:
    return {
        "ticket_id": ticket.ticket_id,
        "channel": ticket.channel,
        "predicted_intent": classification.intent,
        "intent_confidence": round(classification.intent_confidence, 3),
        "predicted_urgency": classification.urgency,
        "candidate_documents": [
            {"doc_id": p.doc_id, "title": p.title, "score": round(p.score, 3)}
            for p in retrieved
        ],
        "escalation_reason": reason,
        "ticket_text": ticket.text,
    }
