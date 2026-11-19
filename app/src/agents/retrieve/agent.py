import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, override

from google.adk import Context
from google.adk.events import Event
from google.adk.workflow import Node

from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from chromadb.utils import embedding_functions
import numpy as np


from src import config
from src.logging_store import log_decision
from src.schemas import ClassificationResult, NormalizedTicket, RetrievedDocument
from src.utils.singleton import Singleton


logger = logging.getLogger(__file__)


class ChromaEmbeddingAdapter(Embeddings):
    def __init__(self):
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[np.float32]]:
        embeddings = self.embedding_function(texts)
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[np.float32]:
        embedding = self.embedding_function([text])[0]
        return embedding.tolist()


class DocumentRetriever(metaclass=Singleton):
    def __init__(
        self,
        documentation_file=None,
    ) -> None:
        self.documentation_file = documentation_file or config.CONFIG.documentation_file
        self.vectorstore: Optional[Chroma] = None
        self.embeddings = ChromaEmbeddingAdapter()
        self._build_store()

    @classmethod
    def get_instance(cls) -> "DocumentRetriever":
        if cls not in cls._instances:
            return DocumentRetriever()
        return cls._instances[cls]

    def load_documents(self):
        return json.loads(self.documentation_file.read_text())

    def _chunk_text(self, contents: List[str], metadatas: List[Dict[str, Any]]):
        text_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )

        docs = text_splitter.create_documents(contents, metadatas)
        chunks = text_splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            ids=list(map(str, range(len(chunks)))),
            persist_directory=config.CONFIG.chroma_path,
            collection_name="support_documents",
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _build_store(self) -> None:
        documents = self.load_documents()
        contents = []
        metadatas = []
        for doc in documents:
            if not doc.get("content", "").strip():
                continue
            contents.append(doc["content"])
            metadatas.append(
                {
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "applies_to": doc["applies_to"],
                    "related_docs": doc["related_docs"],
                }
            )
        self._chunk_text(contents=contents, metadatas=metadatas)

    def retrieve(
        self,
        query: str,
        category: str = None,
        top_k: int = config.CONFIG.retrieval_top_k,
        confidence_threshold: float = config.CONFIG.retrieval_confidence_threshold,
    ) -> Tuple[List[RetrievedDocument], List[RetrievedDocument]]:
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={"category": category} if category else None,
        )
        docs = []
        for res, score in results:
            similarity_score = 1.0 - score
            docs.append(
                RetrievedDocument(
                    doc_id=res.metadata["doc_id"],
                    title=res.metadata["title"],
                    chunk_text=res.page_content,
                    score=similarity_score,
                )
            )

        valid_docs, rejected_docs = [], []
        for doc in docs:
            (valid_docs if doc.score >= confidence_threshold else rejected_docs).append(
                doc
            )

        return valid_docs, rejected_docs


class Retriever(Node):
    model_config = {"arbitrary_types_allowed": True}
    rerun_on_resume: bool = True

    @override
    async def run_node_impl(
        self, ctx: Context, node_input: ClassificationResult
    ) -> AsyncGenerator[Event, None]:
        ticket: NormalizedTicket = ctx.state["ticket"]
        retriever = DocumentRetriever.get_instance()
        valid_docs, rejected_docs = retriever.retrieve(
            ticket.text,
            category=node_input.category,
        )
        ctx.state["valid_documents"] = valid_docs
        ctx.state["rejected_documents"] = rejected_docs
        all_docs = valid_docs + rejected_docs
        log_decision(
            ticket_id=ticket.ticket_id,
            stage="retrieval",
            input_summary=ticket.text,
            model_name="chroma-db-vector-similarity-search",
            prediction=",".join(p.doc_id for p in all_docs) or "none",
            confidence=(max((p.score for p in all_docs), default=0.0)),
            sources_used=[p.model_dump() for p in all_docs],
            action_taken="retrieved" if valid_docs else "no_match",
            reason=f"{len(valid_docs)} passage(s) cleared the relevance threshold.",
        )

        event = Event(
            author="retriever",
            invocation_id=ctx.invocation_id,
            output={"valid_documents": valid_docs, "rejected_documents": rejected_docs},
        )
        yield event
