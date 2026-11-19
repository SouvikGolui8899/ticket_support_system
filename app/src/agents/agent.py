from google.adk import Workflow, Context
from google.adk.events import Event
from google.adk.workflow import START, node
from google.genai import types

from src.agents.classify.agent import Classifier
from src.agents.generate.agent import Generator
from src.agents.ingest.agent import Ingester
from src.agents.retrieve.agent import Retriever
from src.agents.route.agent import Router
from src.agents.validate.agent import Validator
from src.schemas import RawTicket


@node(name="input_parser", rerun_on_resume=True)
async def input_parser(ctx: Context, node_input: types.Content):
    raw_ticket = RawTicket.model_validate_json(node_input.parts[0].text)
    return Event(output=raw_ticket)


ingester = Ingester(name="ingester")
classifier = Classifier(name="classifier")
retriever = Retriever(name="retriever")
router = Router(name="router")
generator = Generator(name="generator")
validator = Validator(name="validator")


root_agent = Workflow(
    name="ticket_support_system",
    edges=[
        (
            START,
            input_parser,
            ingester,
            classifier,
            retriever,
            router,
            generator,
            validator,
        ),
    ],
)
