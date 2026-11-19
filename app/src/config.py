from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    data_dir_name: str = Field(default="data", validation_alias="DATA_DIR")
    storage_dir_name: str = Field(default="storage", validation_alias="STORAGE_DIR")
    documentation_file_name: str = Field(default="documentation.json")
    development_tickets_file_name: str = Field(default="development_tickets.json")

    openrouter_api_key: Optional[str] = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    llm_model_name: str = Field(
        default="meta-llama/llama-3.1-8b-instruct", validation_alias="LLM_MODEL_NAME"
    )
    llm_timeout_seconds: float = Field(
        default=8.0, validation_alias="LLM_TIMEOUT_SECONDS"
    )

    retrieval_top_k: int = Field(default=3, validation_alias="RETRIEVAL_TOP_K")
    retrieval_confidence_threshold: float = Field(
        default=0.12, validation_alias="RETRIEVAL_CONFIDENCE_THRESHOLD"
    )
    intent_confidence_threshold: float = Field(
        default=0.80, validation_alias="INTENT_CONFIDENCE_THRESHOLD"
    )
    grounding_similarity_floor: float = Field(
        default=0.08, validation_alias="GROUNDING_SIMILARITY_FLOOR"
    )

    database_url_name: Optional[str] = Field(
        default=None, validation_alias="DATABASE_URL"
    )
    chroma_path_name: Optional[str] = Field(
        default=None, validation_alias="CHROMA_PATH"
    )
    # chroma_embedding_model_path: str = Field(
    #     default="./storage/embedding_model", validation_alias="CHROMA_CACHE_DIR"
    # )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    metrics_app_url: str = Field(validation_alias="METRICS_APP_URL")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def data_dir(self) -> Path:
        return ROOT_DIR / self.data_dir_name

    @property
    def storage_dir(self) -> Path:
        return ROOT_DIR / self.storage_dir_name

    @property
    def documentation_file(self) -> Path:
        return self.data_dir / self.documentation_file_name

    @property
    def development_tickets_file(self) -> Path:
        return self.data_dir / self.development_tickets_file_name

    @property
    def database_url(self) -> str:
        if self.database_url_name:
            return self.database_url_name
        return f"sqlite:///{self.storage_dir / 'decisions.db'}"

    @property
    def chroma_path(self) -> str:
        if self.chroma_path_name:
            return self.chroma_path_name
        return f"{self.storage_dir / 'chroma'}"

    @property
    def classification_model_path(self) -> Path:
        return self.storage_dir / "classifier.pkl"

    @property
    def prompt_library(self) -> Path:
        return ROOT_DIR / "prompts"

    def get_prompts(self, prompt_name: str) -> str:
        prompt_map = {
            "generative_system_instructions": self.prompt_library / "generative_system_instructions_01.md",
            "generative_dynamic_instructions": self.prompt_library / "generative_dynamic_instructions_01.md",
        }
        return prompt_map[prompt_name].read_text(encoding="utf-8")


NEVER_AUTO_RESPOND_INTENTS = {
    "security_incident",
    "compliance_request",
    "feature_request",
    "unclear_request",
}

FORBIDDEN_CLAIMS = [
    "refund has been issue",
    "issue has been fixed on our side",
    "specific delivery date for a fix",
    "we will fix this by",
    "guaranteed to work",
    "100% guaranteed",
]

VALID_CHANNELS = {"email", "chat", "docs_comment", "forum"}

CONFIG = Config()

os.makedirs(CONFIG.data_dir, exist_ok=True)
os.makedirs(CONFIG.storage_dir, exist_ok=True)
# os.makedirs(CONFIG.chroma_embedding_model_path, exist_ok=True)
