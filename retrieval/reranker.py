"""
Cohere multilingual reranking
"""
import cohere

from config import settings

_client = cohere.ClientV2(api_key=settings.COHERE_API_KEY)


def rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    if not candidates:
        return []
    docs = [c["text"] for c in candidates]
    response = _client.rerank(
        model=settings.RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )
    reranked = []
    for result in response.results:
        item = dict(candidates[result.index])
        item["rerank_score"] = result.relevance_score
        reranked.append(item)
    return reranked
