"""
All env vars loaded from .env file
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GEMINI_API_KEY: str
    QDRANT_URL: str
    QDRANT_API_KEY: str
    COHERE_API_KEY: str
    GROQ_API_KEY: str

    QDRANT_COLLECTION: str = "multilang_rag"

    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 768

    RERANK_MODEL: str = "rerank-v3.5"

    GROQ_MODEL: str = "openai/gpt-oss-120b"

    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50

    TOP_K_CANDIDATES: int = 15
    TOP_N_FINAL: int = 4

    RAW_DOCS_DIR: str = "data/raw"


settings = Settings()
