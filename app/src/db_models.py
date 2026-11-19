from typing import Optional

from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import Text, REAL, create_engine

from src import config


class Base(DeclarativeBase):
    pass


class Decision(Base):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(
        Text, primary_key=True, autoincrement=False
    )
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_id: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    input_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    prediction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    alternatives: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sources_used: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    threshold_applied: Mapped[Optional[float]] = mapped_column(REAL, nullable=True)
    action_taken: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    guardrail_results: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


ENGINE = create_engine(config.CONFIG.database_url, echo=False)


def create_all_tables():
    Base.metadata.create_all(ENGINE)
