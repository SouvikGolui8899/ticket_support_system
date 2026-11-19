import pytest

from src.schemas import ClassificationResult, RawTicket, RetrievedDocument
from src.utils.common import generate_escalation_summary


def _classification():
    return ClassificationResult(
        intent="authentication_failure",
        intent_confidence=0.9,
        intent_alternatives=[],
        urgency="medium",
        urgency_confidence=0.7,
        category="authentication",
        category_confidence=0.9,
        category_alternatives=[],
    )


@pytest.mark.asyncio
async def test_generate_answer_is_grounded_and_cited_when_sources_exist(
    mock_ingester, mock_generator
):
    ticket = mock_ingester.normalize_ticket(
        raw=RawTicket(
            ticket_id="T-1", channel="email", subject="", body="Cannot log in"
        )
    )
    passage = RetrievedDocument(
        doc_id="DOC-AUTH-001",
        title="Resolving invalid credential errors",
        chunk_text="Check the security page of the console for a lock banner.",
        score=0.4,
    )
    answer = mock_generator.generate_answer(ticket, _classification(), [passage])
    assert "DOC-AUTH-001" in answer.citations


@pytest.mark.asyncio
async def test_escalation_summary_shows_its_working(mock_ingester):
    ticket = mock_ingester.normalize_ticket(
        raw=RawTicket(
            ticket_id="T-1", channel="chat", subject="", body="Billing dispute"
        )
    )
    summary = generate_escalation_summary(
        ticket,
        _classification(),
        [],
        reason="low confidence",
    )
    assert summary["ticket_id"] == "T-1"
    assert summary["predicted_intent"] == "authentication_failure"
    assert summary["escalation_reason"] == "low confidence"
    assert summary["ticket_text"] == "Billing dispute"
