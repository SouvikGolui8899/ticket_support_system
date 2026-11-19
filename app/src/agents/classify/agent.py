from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, override

from google.adk import Context
from google.adk.events import Event
from google.adk.workflow import Node

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from src import config
from src.logging_store import log_decision
from src.schemas import ClassificationResult, NormalizedTicket


logger = logging.getLogger(__file__)


class IntentClassifier:
    FALLBACK_INTENT = "unclear_request"
    FALLBACK_URGENCY = "medium"
    FALLBACK_CATEGORY = None

    def __init__(self) -> None:
        self.vectorizer = None
        self.intent_model: Optional[RandomForestClassifier] = None
        self.urgency_model: Optional[RandomForestClassifier] = None
        self.category_model: Optional[RandomForestClassifier] = None
        self.intent_classes: List[str] = []
        self.urgency_classes: List[str] = []
        self.category_classes: List[str] = []

    def fit(self, tickets: List[Dict], document: List[Dict]) -> "IntentClassifier":
        texts, intents, urgencies, catgeories = [], [], [], []
        doc_category_map = {doc["doc_id"]: doc["category"] for doc in document}
        for t in tickets:
            labels = t.get("labels", {}) or {}
            intent = labels.get("intent", None)
            urgency = labels.get("urgency", None)
            if not intent or not urgency:
                continue
            subject = t.get("subject", "").strip()
            body = t.get("body", "").strip()
            texts.append((subject + "\n" + body).strip() if subject else body)
            intents.append(intent)
            urgencies.append(urgency)

            doc_ids = labels.get("expected_doc_ids", [])
            catgeories.append(doc_category_map[doc_ids[0]] if doc_ids else "")

        self.vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=1, stop_words="english"
        )
        X = self.vectorizer.fit_transform(texts)

        self.intent_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        self.intent_model.fit(X, intents)
        self.intent_classes = list(self.intent_model.classes_)

        self.urgency_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        self.urgency_model.fit(X, urgencies)
        self.urgency_classes = list(self.urgency_model.classes_)

        self.category_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        self.category_model.fit(X, catgeories)
        self.category_classes = list(self.category_model.classes_)
        return self

    def save(self, path: Path = config.CONFIG.classification_model_path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_or_train(
        cls, path: Path = config.CONFIG.classification_model_path
    ) -> "IntentClassifier":
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    obj = pickle.load(fh)
                if isinstance(obj, cls) and obj.vectorizer is not None:
                    return obj
            except Exception:
                pass

        import json

        tickets = json.loads(config.CONFIG.development_tickets_file.read_text())
        document = json.loads(config.CONFIG.documentation_file.read_text())
        clf = cls().fit(tickets, document)
        try:
            clf.save(path)
        except Exception:
            pass
        return clf

    def classify(self, ticket: NormalizedTicket) -> ClassificationResult:
        text = ticket.text.strip()

        if not text:
            logger.info("No text found")
            return ClassificationResult(
                intent=self.FALLBACK_INTENT,
                intent_confidence=0.0,
                intent_alternatives=[],
                urgency=self.FALLBACK_URGENCY,
                urgency_confidence=0.0,
                category=self.FALLBACK_CATEGORY,
                category_confidence=0.0,
                category_alternatives=[],
            )
        if self.vectorizer is None or self.intent_model is None:
            raise Exception("Intent Classifier not initialized properly")
        try:
            X = self.vectorizer.transform([text])
            intent_proba = self.intent_model.predict_proba(X)[0]
            urgency_proba = self.urgency_model.predict_proba(X)[0]

            order = np.argsort(intent_proba)[::-1]
            top_idx = order[0]

            alternatives = [
                (self.intent_classes[i], float(intent_proba[i])) for i in order[1:4]
            ]

            urgency_idx = int(np.argmax(urgency_proba))

            category_proba = self.category_model.predict_proba(X)[0]

            category_order = np.argsort(category_proba)[::-1]
            category_top_idx = category_order[0]
            catgeory_alternatives = [
                (self.category_classes[i], float(category_proba[i]))
                for i in category_order[1:4]
            ]

            return ClassificationResult(
                intent=self.intent_classes[top_idx],
                intent_confidence=float(intent_proba[top_idx]),
                intent_alternatives=alternatives,
                urgency=self.urgency_classes[urgency_idx],
                urgency_confidence=float(urgency_proba[urgency_idx]),
                category=self.category_classes[category_top_idx],
                category_confidence=float(category_proba[category_top_idx]),
                category_alternatives=catgeory_alternatives,
            )

        except Exception as ex:
            logger.exception(ex)
            # Classification must never crash the pipeline; a ticket that cannot be classified
            # is routed to a safe, low-confidence fallback instead.
            return ClassificationResult(
                intent=self.FALLBACK_INTENT,
                intent_confidence=0.0,
                intent_alternatives=[],
                urgency=self.FALLBACK_URGENCY,
                urgency_confidence=0.0,
                category=self.FALLBACK_CATEGORY,
                category_confidence=0.0,
                category_alternatives=[],
            )


_classifier_singleton: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = IntentClassifier.load_or_train()
    return _classifier_singleton


class Classifier(Node):
    model_config = {"arbitrary_types_allowed": True}
    rerun_on_resume: bool = True

    @override
    async def run_node_impl(
        self, ctx: Context, node_input: NormalizedTicket
    ) -> AsyncGenerator[Event, None]:
        classifier = get_classifier()
        classification = classifier.classify(ticket=node_input)
        ctx.state["classification"] = classification
        log_decision(
            ticket_id=node_input.ticket_id,
            stage="classification",
            input_summary=node_input.text,
            model_name=classification.model_name,
            prediction=classification.intent,
            confidence=classification.intent_confidence,
            alternatives=classification.intent_alternatives,
            action_taken="classified",
            reason=f"Predicted intent={classification.intent}, urgency={classification.urgency}",
        )
        event = Event(
            author="classifier",
            invocation_id=ctx.invocation_id,
            output=classification,
        )
        yield event
