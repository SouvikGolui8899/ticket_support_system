import pytest

from src.db_models import create_all_tables
from src.agents.classify.agent import get_classifier
from src.agents.generate.agent import Generator
from src.agents.ingest.agent import Ingester
from src.agents.retrieve.agent import DocumentRetriever
from src.agents.route.agent import Router


create_all_tables()


@pytest.fixture
def mock_ingester():
    return Ingester(name="ingester")


@pytest.fixture
def mock_classifier():
    return get_classifier()


@pytest.fixture
def mock_retriever():
    return DocumentRetriever.get_instance()


@pytest.fixture
def mock_router():
    return Router(name="router")


@pytest.fixture
def mock_generator():
    return Generator(name="generator")
