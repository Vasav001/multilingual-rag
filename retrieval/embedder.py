"""
Wraps Gemini embedding-001 so ingestion and retrieval call the exact
same function/model/dimensionality.
"""
import numpy as np
from google import genai
from google.genai import types

from config import settings

_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _normalize(vec: list[float]) -> list[float]:
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return vec
    return (arr / norm).tolist()


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    if not texts:
        return []
    result = _client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIM,
        ),
    )
    return [_normalize(e.values) for e in result.embeddings]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed chunks at ingestion time."""
    return _embed(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a single user query at retrieval time."""
    return _embed([text], task_type="RETRIEVAL_QUERY")[0]
