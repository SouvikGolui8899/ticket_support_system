import pytest

from src.schemas import Action, ClassificationResult, RetrievedDocument


def _classification(
    intent="authentication_failure", category="authentication", confidence=0.9
):
    return ClassificationResult(
        intent=intent,
        intent_confidence=confidence,
        intent_alternatives=[],
        urgency="medium",
        urgency_confidence=0.7,
        category=category,
        category_confidence=confidence,
        category_alternatives=[],
    )


def _passage():
    return RetrievedDocument(
        doc_id="DOC-AUTH-001", title="t", chunk_text="x", score=0.5
    )


@pytest.mark.asyncio
async def test_auto_responds_when_confident_and_grounded(mock_router):
    decision = mock_router.route_ticket(_classification(confidence=0.95), [_passage()])
    assert decision.action == Action.AUTO_RESPOND


@pytest.mark.asyncio
async def test_escalates_when_confidence_below_threshold(mock_router):
    decision = mock_router.route_ticket(_classification(confidence=0.3), [_passage()])
    assert decision.action == Action.ESCALATE


@pytest.mark.asyncio
async def test_escalates_when_no_retrieval_results(mock_router):
    decision = mock_router.route_ticket(_classification(confidence=0.95), [])
    assert decision.action == Action.ESCALATE


@pytest.mark.asyncio
async def test_never_auto_respond_intents_always_escalate(mock_router):
    decision = mock_router.route_ticket(
        _classification(
            intent="security_incident", category="security", confidence=0.99
        ),
        [_passage()],
    )
    assert decision.action == Action.ESCALATE
    assert decision.never_auto_respond_rule_hit is True


@pytest.mark.asyncio
async def test_routing_is_deterministic(mock_router):
    classification = _classification(confidence=0.9)
    retrieved = [_passage()]
    d1 = mock_router.route_ticket(
        classification=classification,
        valid_docs=retrieved,
    )
    d2 = mock_router.route_ticket(
        classification=classification,
        valid_docs=retrieved,
    )
    assert d1.action == d2.action
