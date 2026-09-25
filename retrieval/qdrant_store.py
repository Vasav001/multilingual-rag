"""
Qdrant collection has a langs data. Payload schema in `upsert_chunks`.
"""
import hashlib
import uuid

from qdrant_client import QdrantClient, models

from config import settings

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return _client


def ensure_collection() -> None:
    client = get_client()
    if client.collection_exists(settings.QDRANT_COLLECTION):
        return
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=settings.EMBEDDING_DIM,
            distance=models.Distance.COSINE,
        ),
    )
    # Index document_id so ingestion can cheaply check already ingested
    client.create_payload_index(
        collection_name=settings.QDRANT_COLLECTION,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


def document_already_ingested(document_id: str) -> bool:
    client = get_client()
    hits, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        ),
        limit=1,
    )
    return len(hits) > 0


def _point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_index}"))


def upsert_chunks(document_id: str, filename: str, file_type: str, language: str, chunks: list[dict], vectors: list[list[float]]) -> None:
    """`chunks` is a list of {"text", "chunk_index", "page", "source"}.
    Payload stored per point: document_id, filename, file_type, language,
    chunk_index, text, source, page.
    """
    client = get_client()
    points = [
        models.PointStruct(
            id=_point_id(document_id, c["chunk_index"]),
            vector=vec,
            payload={
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "language": language,
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "source": c.get("source", filename),
                "page": c.get("page"),
            },
        )
        for c, vec in zip(chunks, vectors)
    ]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)


def search(query_vector: list[float], limit: int) -> list[dict]:
    client = get_client()
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points
    return [{"score": r.score, **r.payload} for r in results]


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
