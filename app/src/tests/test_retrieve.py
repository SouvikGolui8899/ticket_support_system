import pytest


@pytest.mark.asyncio
async def test_retrieval_returns_relevant_passage_for_known_topic(mock_retriever):
    valid_docs, _ = mock_retriever.retrieve(
        "my deployment keeps rolling back with a health check timeout"
    )
    assert len(valid_docs) >= 1
    assert any(r.doc_id == "DOC-DEPLOY-001" for r in valid_docs)


@pytest.mark.asyncio
async def test_retrieval_returns_nothing_for_irrelevant_query(mock_retriever):
    valid_docs, _ = mock_retriever.retrieve(
        "zzz qqq nonsense gibberish unrelated xylophone", confidence_threshold=0.5
    )
    assert valid_docs == []


@pytest.mark.asyncio
async def test_retrieval_handles_empty_query(mock_retriever):
    valid_docs, _ = mock_retriever.retrieve("")
    assert valid_docs == []
