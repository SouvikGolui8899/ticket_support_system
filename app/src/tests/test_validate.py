import pytest

from src.agents.validate.agent import run_guardrails
from src.schemas import ClassificationResult, GeneratedAnswer, RetrievedDocument


def _classification(confidence=0.9):
    return ClassificationResult(
        intent="authentication_failure",
        intent_confidence=confidence,
        intent_alternatives=[],
        urgency="medium",
        urgency_confidence=0.7,
        category="authentication",
        category_confidence=confidence,
        category_alternatives=[],
    )


def _passage(text="Resources can be found on the billing page."):
    return RetrievedDocument(
        doc_id="DOC-BILL-001", title="t", chunk_text=text, score=0.5
    )


@pytest.mark.asyncio
async def test_guardrail_blocks_hallucinated_private_data():
    passage = _passage()
    answer = GeneratedAnswer(
        text="Please contact jane.doe@example.com for a refund of your invoice billing.",
        citations=["DOC-BILL-001"],
        mode="extractive",
        grounded_documents=[passage],
    )
    result = run_guardrails(answer, _classification())
    assert result.blocked is True
    assert result.checks["private_data"] == "fail"


@pytest.mark.asyncio
async def test_guardrail_blocks_forbidden_claims():
    passage = _passage()
    answer = GeneratedAnswer(
        text="Good news -- a refund has been issued for the billing charge you flagged.",
        citations=["DOC-BILL-001"],
        mode="extractive",
        grounded_documents=[passage],
    )
    result = run_guardrails(answer, _classification())
    assert result.blocked is True
    assert result.checks["tone_and_scope"] == "fail"


@pytest.mark.asyncio
async def test_guardrail_blocks_ungrounded_answer():
    passage = _passage(
        text="Billing periods run from the subscription anniversary date."
    )
    answer = GeneratedAnswer(
        text="Your rocket ship deployment uses quantum flux capacitors for authentication.",
        citations=["DOC-BILL-001"],
        mode="extractive",
        grounded_documents=[passage],
    )
    result = run_guardrails(answer, _classification())
    assert result.blocked is True
    assert result.checks["grounding"] == "fail"


@pytest.mark.asyncio
async def test_guardrail_blocks_missing_confidence():
    passage = _passage()
    answer = GeneratedAnswer(
        text="Billing periods run from the subscription anniversary date, as explained on the billing page.",
        citations=["DOC-BILL-001"],
        mode="extractive",
        grounded_documents=[passage],
    )
    result = run_guardrails(answer, _classification(confidence=0.0))
    assert result.blocked is True
    assert result.checks["confidence_floor"] == "fail"


@pytest.mark.asyncio
async def test_guardrail_passes_clean_grounded_answer():
    passage = _passage(
        text="Billing periods run from the subscription anniversary date."
    )
    answer = GeneratedAnswer(
        text="Billing periods run from the subscription anniversary date, as explained on the billing page.",
        citations=["DOC-BILL-001"],
        mode="extractive",
        grounded_documents=[passage],
    )
    result = run_guardrails(answer, _classification(confidence=0.9))
    assert result.blocked is False
    assert all(v == "pass" for v in result.checks.values())
