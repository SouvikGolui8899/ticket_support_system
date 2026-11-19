import pytest

from src.schemas import RawTicket


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["email", "chat", "docs_comment", "forum"])
async def test_normalize_channel(channel: str, mock_ingester):
    ticket_id = f"T-{channel}"
    raw_ticket = {
        "ticket_id": ticket_id,
        "channel": channel,
        "subject": "Subject",
        "body": "Body text",
    }
    ticket = mock_ingester.normalize_ticket(raw=RawTicket.model_validate(raw_ticket))
    assert ticket.channel == channel
    assert ticket.ticket_id == ticket_id
    assert "Body text" in ticket.text


@pytest.mark.asyncio
async def test_handles_missing_fields_without_raising(mock_ingester):
    ticket = mock_ingester.normalize_ticket(raw=RawTicket())
    assert ticket.ticket_id == "UNKNOWN-ID"
    assert ticket.text == ""
    assert ticket.channel in {"unknown", ""}


@pytest.mark.asyncio
async def test_chat_ticket_with_empty_subject_uses_body_only(mock_ingester):
    raw_ticket = RawTicket(
        ticket_id="T-1",
        channel="chat",
        subject="",
        body="hello",
    )
    ticket = mock_ingester.normalize_ticket(raw_ticket)
    assert ticket.text == "hello"
