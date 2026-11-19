from typing import Any, AsyncGenerator, List, override

from google.adk import Context
from google.adk.events import Event
from google.adk.workflow import Node

from src import config
from src.logging_store import log_decision
from src.schemas import (
    Action,
    ClassificationResult,
    NormalizedTicket,
    RetrievedDocument,
    RoutingDecision,
)


class Router(Node):
    model_config = {"arbitrary_types_allowed": True}
    rerun_on_resume: bool = True

    def route_ticket(
        self,
        classification: ClassificationResult,
        valid_docs: List[RetrievedDocument],
        confidence_threshold: float = config.CONFIG.intent_confidence_threshold,
    ) -> RoutingDecision:
        if classification.intent in config.NEVER_AUTO_RESPOND_INTENTS:
            return RoutingDecision(
                action=Action.ESCALATE,
                reason=(
                    f"Intent '{classification.intent}' is configured as always escalated regardless of model confidence"
                ),
                confidence_threshold_applied=confidence_threshold,
                classification_confidence=classification.intent_confidence,
                retrieval_cleared_threshold=bool(valid_docs),
                never_auto_respond_rule_hit=True,
            )

        if classification.intent_confidence < confidence_threshold:
            return RoutingDecision(
                action=Action.ESCALATE,
                reason=(
                    f"Classification confidence {classification.intent_confidence:.2f} "
                    f"is below the routing threshold {confidence_threshold:.2f}"
                ),
                confidence_threshold_applied=confidence_threshold,
                classification_confidence=classification.intent_confidence,
                retrieval_cleared_threshold=bool(valid_docs),
            )

        if not valid_docs:
            return RoutingDecision(
                action=Action.ESCALATE,
                reason="No documentation cleared the retrieval relevance threshold.",
                confidence_threshold_applied=confidence_threshold,
                classification_confidence=classification.intent_confidence,
                retrieval_cleared_threshold=False,
            )

        return RoutingDecision(
            action=Action.AUTO_RESPOND,
            reason=(
                f"Confidence {classification.intent_confidence:.2f} clears the "
                f"threshold {confidence_threshold:.2f} and {len(valid_docs)} grounded "
                "source(s) were retrieved."
            ),
            confidence_threshold_applied=confidence_threshold,
            classification_confidence=classification.intent_confidence,
            retrieval_cleared_threshold=True,
        )

    @override
    async def run_node_impl(
        self, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Event, None]:
        ticket: NormalizedTicket = ctx.state["ticket"]
        classification: ClassificationResult = ctx.state["classification"]
        valid_docs: List[RetrievedDocument] = ctx.state["valid_documents"]

        routing = self.route_ticket(
            classification=classification, valid_docs=valid_docs
        )

        log_decision(
            ticket_id=ticket.ticket_id,
            stage="routing",
            input_summary=ticket.text,
            model_name="rule-based-router-v1",
            prediction=routing.action.value,
            confidence=classification.intent_confidence,
            threshold_applied=routing.confidence_threshold_applied,
            action_taken=routing.action.value,
            reason=routing.reason,
        )

        event = Event(author="router", invocation_id=ctx.invocation_id, output=routing)
        yield event
