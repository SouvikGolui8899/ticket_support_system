"""Deliberately duplicated code for prometheus client"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class RawTicket(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticket_id: Optional[str] = None
    channel: Optional[str] = None
    subject: Optional[str] = ""
    body: Optional[str] = ""
    received_at: Optional[str] = None
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_tier: Optional[str] = None
    customer_region: Optional[str] = None
    language_fluency: Optional[str] = None


class NormalizedTicket(BaseModel):
    ticket_id: str
    channel: Optional[str] = None
    subject: Optional[str] = ""
    body: Optional[str] = ""
    text: str  # subject + body, concatenated for downstream NLP
    received_at: Optional[str] = None
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    customer_tier: Optional[str] = None
    customer_region: Optional[str] = None
    language_fluency: Optional[str] = None
    raw: RawTicket


class ClassificationResult(BaseModel):
    intent: str
    intent_confidence: float
    intent_alternatives: List[Tuple[str, float]]
    urgency: str
    urgency_confidence: float
    category: Optional[str]
    category_confidence: Optional[float]
    category_alternatives: List[Tuple[str, float]]
    model_name: str = "tfidf-logreg-v1"


class RetrievedDocument(BaseModel):
    doc_id: str
    title: str
    chunk_text: str
    score: float


class Action(str, Enum):
    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"
    BLOCKED_ESCALATE = "blocked_escalate"


class RoutingDecision(BaseModel):
    action: Action  # AUTO_RESPOND or ESCALATE (pre-guardrail stage)
    reason: str
    confidence_threshold_applied: float
    classification_confidence: float
    retrieval_cleared_threshold: bool
    never_auto_respond_rule_hit: bool = False


class GenerativeMode(str, Enum):
    LLM = "llm"
    EXTRACTIVE = "extractive"


class GeneratedAnswer(BaseModel):
    text: str
    citations: List[str]  # doc_ids actually used
    mode: GenerativeMode
    grounded_documents: List[RetrievedDocument] = Field(default_factory=list)


class GuardrailResult(BaseModel):
    blocked: bool
    checks: dict = Field(default_factory=dict)
    details: dict = Field(default_factory=dict)


class PipelineResult(BaseModel):
    ticket_id: str
    channel: str
    classification: ClassificationResult
    valid_docs: List[RetrievedDocument]
    rejected_docs: List[RetrievedDocument]
    routing: RoutingDecision
    final_action: Action
    answer: Optional[GeneratedAnswer] = None
    guardrails: Optional[GuardrailResult] = None
    escalation_context: Optional[Dict] = None
    latency_seconds: float = 0.0

    def to_summary(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "channel": self.channel,
            "predicted_intent": self.classification.intent,
            "intent_confidence": round(self.classification.intent_confidence, 3),
            "predicted_urgency": self.classification.urgency,
            "retrieved_doc_ids": [
                p.doc_id for p in self.valid_docs + self.rejected_docs
            ],
            "routing_action": self.routing.action.value,
            "routing_confidence": self.routing.classification_confidence,
            "final_action": self.final_action.value,
            "answer_text": self.answer.text if self.answer else None,
            "citations": self.answer.citations if self.answer else [],
            "guardrail_checks": self.guardrails.checks if self.guardrails else [],
            "latency_seconds": round(self.latency_seconds, 4),
        }
