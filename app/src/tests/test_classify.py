import math

import pytest

from src.schemas import RawTicket


@pytest.mark.asyncio
async def test_classifier_predicts_known_intent_pattern(mock_ingester, mock_classifier):
    ticket = mock_ingester.normalize_ticket(
        raw=RawTicket(
            ticket_id="T-1",
            channel="email",
            subject="Cannot log in to the console",
            body=(
                "I have been trying to sign in for the last day and keep getting "
                "an invalid credentials message. Nothing has changed at our end."
            ),
        )
    )
    result = mock_classifier.classify(ticket=ticket)
    assert result.intent
    assert 0.0 <= result.intent_confidence <= 1.0
    assert result.urgency in {"high", "medium", "low"}


@pytest.mark.asyncio
async def test_classifier_falls_back_on_empty_text(mock_ingester, mock_classifier):
    ticket = mock_ingester.normalize_ticket(
        raw=RawTicket(ticket_id="T-1", channel="chat")
    )
    result = mock_classifier.classify(ticket)
    assert result.intent == mock_classifier.FALLBACK_INTENT
    assert result.intent_confidence == 0.0


@pytest.mark.asyncio
async def test_classifier_is_deterministic(mock_ingester, mock_classifier):
    ticket = mock_ingester.normalize_ticket(
        raw=RawTicket(
            ticket_id="T-1",
            channel="chat",
            subject="",
            body="How do I revert to an earlier revision after a bad release?",
        )
    )
    r1 = mock_classifier.classify(ticket)
    r2 = mock_classifier.classify(ticket)
    assert r1.intent == r2.intent
    assert math.isclose(r1.intent_confidence, r2.intent_confidence)
    assert r1.urgency == r2.urgency
    assert math.isclose(r1.urgency_confidence, r2.urgency_confidence)
    assert r1.category == r2.category
    assert math.isclose(r1.category_confidence, r2.category_confidence)
