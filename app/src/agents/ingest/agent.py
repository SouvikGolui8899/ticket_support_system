from typing import AsyncGenerator, override

from google.adk import Context
from google.adk.events import Event
from google.adk.workflow import Node

from src import config
from src.schemas import NormalizedTicket, RawTicket
from src.utils.common import clean_str


class Ingester(Node):
    model_config = {"arbitrary_types_allowed": True}
    rerun_on_resume: bool = True

    def normalize_ticket(self, raw: RawTicket) -> NormalizedTicket:
        channel = clean_str(raw.channel, default="unknown").lower()
        if channel not in config.VALID_CHANNELS:
            channel = channel or "unknown"

        subject = clean_str(raw.subject)
        body = clean_str(raw.body)
        text = (subject + "\n" + body).strip() if subject else body

        ticket_id = clean_str(raw.ticket_id, default="UNKNOWN-ID")

        return NormalizedTicket(
            ticket_id=ticket_id,
            channel=channel,
            subject=subject,
            body=body,
            text=text,
            received_at=raw.received_at,
            customer_id=raw.customer_id,
            customer_name=raw.customer_name,
            customer_tier=raw.customer_tier,
            customer_region=raw.customer_region,
            language_fluency=raw.language_fluency,
            raw=raw,
        )

    @override
    async def run_node_impl(
        self, ctx: Context, node_input: RawTicket
    ) -> AsyncGenerator[Event, None]:
        ticket = self.normalize_ticket(raw=node_input)
        ctx.state["ticket"] = ticket

        event = Event(
            author="ingester",
            invocation_id=ctx.invocation_id,
            output=ticket,
        )
        yield event
