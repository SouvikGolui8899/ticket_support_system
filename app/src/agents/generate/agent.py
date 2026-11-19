from typing import AsyncGenerator, List, Optional, override

from google.adk import Context
from google.adk.events import Event
from google.adk.workflow import Node

import requests

from src import config
from src.logging_store import log_decision
from src.schemas import (
    Action,
    ClassificationResult,
    GeneratedAnswer,
    GenerativeMode,
    NormalizedTicket,
    PipelineResult,
    RetrievedDocument,
    RoutingDecision,
)
from src.utils.common import generate_escalation_summary


SYSTEM_INSTRUCTIONS = config.CONFIG.get_prompts("generative_system_instructions")
DYNAMIC_INSTRUCTIONS = config.CONFIG.get_prompts("generative_dynamic_instructions")


def _call_llm(prompt: str) -> Optional[str]:
    if not config.CONFIG.openrouter_api_key:
        return None
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {config.CONFIG.openrouter_api_key}"},
            json={
                "model": config.CONFIG.llm_model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=config.CONFIG.llm_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


class Generator(Node):
    model_config = {"arbitrary_types_allowed": True}
    rerun_on_resume: bool = True

    def _extractive_answer(
        self, ticket: NormalizedTicket, retrieved: List[RetrievedDocument]
    ) -> str:
        intro = "Thanks for getting in touch."
        body_parts = [intro]
        for passage in retrieved:
            snippet = passage.chunk_text.strip().replace("\n", " ")
            body_parts.append(f"{snippet} (source: {passage.doc_id})")
        body_parts.append(
            "If this does not resolve things, reply and a member of the team "
            "will take a closer look at your account directly."
        )
        return "\n\n".join(body_parts)

    def generate_answer(
        self,
        ticket: NormalizedTicket,
        classification: ClassificationResult,
        valid_docs: List[RetrievedDocument],
    ) -> GeneratedAnswer:
        if not valid_docs:
            return GeneratedAnswer(
                text=(
                    "I do not have enough grounded information to answer this "
                    "confidently, so I am passing it to a member of the team."
                ),
                citations=[],
                mode=GenerativeMode.EXTRACTIVE,
                grounded_documents=[],
            )

        context_block = "\n\n".join(
            f"[{p.doc_id}] {p.title}\n{p.chunk_text}" for p in valid_docs
        )
        prompt = DYNAMIC_INSTRUCTIONS.format(
            channel=ticket.channel,
            ticket_description=ticket.text,
            context_block=context_block,
        )
        llm_text = _call_llm(prompt)
        if llm_text:
            return GeneratedAnswer(
                text=llm_text,
                citations=[p.doc_id for p in valid_docs],
                mode=GenerativeMode.LLM,
                grounded_documents=valid_docs,
            )

        return GeneratedAnswer(
            text=self._extractive_answer(ticket=ticket, retrieved=valid_docs),
            citations=[p.doc_id for p in valid_docs],
            mode=GenerativeMode.EXTRACTIVE,
            grounded_documents=valid_docs,
        )

    @override
    async def run_node_impl(
        self, ctx: Context, node_input: RoutingDecision
    ) -> AsyncGenerator[Event, None]:
        ticket: NormalizedTicket = ctx.state["ticket"]
        classification: ClassificationResult = ctx.state["classification"]
        valid_docs: List[RetrievedDocument] = ctx.state["valid_documents"]
        rejected_docs: List[RetrievedDocument] = ctx.state["rejected_documents"]

        result = PipelineResult(
            ticket_id=ticket.ticket_id,
            channel=ticket.channel,
            classification=classification,
            valid_docs=valid_docs,
            rejected_docs=rejected_docs,
            routing=node_input,
            final_action=node_input.action,
        )

        if node_input.action == Action.ESCALATE:
            result.escalation_context = generate_escalation_summary(
                ticket=ticket,
                classification=classification,
                retrieved=valid_docs + rejected_docs,
                reason=node_input.reason,
            )

            log_decision(
                ticket_id=ticket.ticket_id,
                stage="generation",
                input_summary=ticket.text,
                model_name="escalation-summary-v1",
                prediction="escalation_context_attacted",
                confidence=classification.intent_confidence,
                sources_used=result.escalation_context.get("candidate_documents", []),
                action_taken=Action.ESCALATE.value,
                reason="Escalated with drafted summary and candidate documents attached.",
            )

            event = Event(
                author="generator", invocation_id=ctx.invocation_id, output=result
            )
            yield event
        else:
            answer = self.generate_answer(
                ticket=ticket, classification=classification, valid_docs=valid_docs
            )
            result.answer = answer

            event = Event(
                author="generator", invocation_id=ctx.invocation_id, output=result
            )
            yield event
