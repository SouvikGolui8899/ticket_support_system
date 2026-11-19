import re
from typing import AsyncGenerator, Dict, List, Set, Tuple, override

from google.adk import Context
from google.adk.events import Event
from google.adk.workflow import Node

from src import config
from src.logging_store import log_decision
from src.schemas import (
    Action,
    ClassificationResult,
    GeneratedAnswer,
    GuardrailResult,
    NormalizedTicket,
    PipelineResult,
    RetrievedDocument,
)
from src.utils.common import generate_escalation_summary


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
API_KEY_RE = re.compile(r"\b(sk-[A-Za-z0-9]{10,}|AKIA[0-9A-Z]{10,}|[A-Fa-f0-9]{32,})\b")
PHONE_RE = re.compile(r"\b\+?\d[\d\s().-]{8,}\d\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

PII_PATTERNS = [EMAIL_RE, API_KEY_RE, PHONE_RE, CREDIT_CARD_RE]


def _check_private_data(answer: GeneratedAnswer) -> Tuple[bool, Dict]:
    source_text = " ".join(p.chunk_text for p in answer.grounded_documents)
    findings = []
    for pattern in PII_PATTERNS:
        for match in pattern.finditer(answer.text):
            token = match.group(0)
            if token not in source_text:
                findings.append(token)
    passed = len(findings) == 0
    return passed, {"findings": findings}


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _check_grounding(answer: GeneratedAnswer) -> Tuple[bool, Dict]:
    if not answer.grounded_documents:
        admits_uncertainty = any(
            phrase in answer.text.lower()
            for phrase in ["do not have enough", "not confident", "pass this to"]
        )
        return admits_uncertainty, {"reason": "no_sources_used"}

    answer_tokens = _tokenize(answer.text)
    source_tokens = _tokenize(" ".join(p.chunk_text for p in answer.grounded_documents))
    if not answer_tokens:
        return False, {"reason": "empty_answer"}

    overlap = answer_tokens & source_tokens
    overlap_ratio = len(overlap) / max(len(answer_tokens), 1)
    passed = overlap_ratio >= config.CONFIG.grounding_similarity_floor
    return passed, {"overlap_ratio": round(overlap_ratio, 3)}


def _check_tone_and_scope(answer: GeneratedAnswer) -> Tuple[bool, Dict]:
    lowered = answer.text.lower()
    hits = [claim for claim in config.FORBIDDEN_CLAIMS if claim in lowered]
    return (len(hits) == 0), {"forbidden_claims_found": hits}


def _check_confidence_floor(classification: ClassificationResult) -> Tuple[bool, Dict]:
    confidence = classification.intent_confidence
    passed = confidence is not None and confidence > 0.0
    return passed, {"confidence": confidence}


def run_guardrails(
    answer: GeneratedAnswer, classification: ClassificationResult
) -> GuardrailResult:
    checks: Dict[str, str] = {}
    details: Dict[str, Dict] = {}

    for name, fn, arg in [
        ("private_data", _check_private_data, answer),
        ("grounding", _check_grounding, answer),
        ("tone_and_scope", _check_tone_and_scope, answer),
        ("confidence_floor", _check_confidence_floor, classification),
    ]:
        passed, detail = fn(arg)
        checks[name] = "pass" if passed else "fail"
        details[name] = detail

    blocked = any(v == "fail" for v in checks.values())
    return GuardrailResult(blocked=blocked, checks=checks, details=details)


class Validator(Node):
    model_config = {"arbitrary_types_allowed": True}
    rerun_on_resume: bool = True

    @override
    async def run_node_impl(
        self, ctx: Context, node_input: PipelineResult
    ) -> AsyncGenerator[Event, None]:
        ticket: NormalizedTicket = ctx.state["ticket"]
        classification: ClassificationResult = ctx.state["classification"]
        valid_docs: List[RetrievedDocument] = ctx.state["valid_documents"]
        rejected_docs: List[RetrievedDocument] = ctx.state["rejected_documents"]
        result = node_input.model_copy(deep=True)

        if result.answer:
            guardrail_result = run_guardrails(
                answer=result.answer, classification=classification
            )
            result.guardrails = guardrail_result

            if guardrail_result.blocked:
                result.final_action = Action.BLOCKED_ESCALATE
                result.escalation_context = generate_escalation_summary(
                    ticket=ticket,
                    classification=classification,
                    retrieved=valid_docs + rejected_docs,
                    reason=f"Guardrail(s) failed: {guardrail_result.checks}",
                )
                log_decision(
                    ticket_id=ticket.ticket_id,
                    stage="validation",
                    input_summary=result.answer.text,
                    model_name=result.answer.mode,
                    prediction="blocked",
                    confidence=classification.intent_confidence,
                    sources_used=result.answer.citations,
                    action_taken="block",
                    reason=f"Guardrail check failed: {guardrail_result.checks}",
                    guardrail_results=guardrail_result.checks,
                )
            else:
                result.final_action = Action.AUTO_RESPOND
                log_decision(
                    ticket_id=ticket.ticket_id,
                    stage="validation",
                    input_summary=result.answer.text,
                    model_name=result.answer.mode,
                    prediction="sent",
                    confidence=classification.intent_confidence,
                    sources_used=result.answer.citations,
                    action_taken=Action.AUTO_RESPOND,
                    reason="All guardrail checks passed; answer sent to customer.",
                    guardrail_results=guardrail_result.checks,
                )
        event = Event(
            author="validator",
            invocation_id=ctx.invocation_id,
            output=result,
        )
        yield event
