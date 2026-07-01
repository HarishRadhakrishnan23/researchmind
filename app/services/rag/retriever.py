"""
Hybrid retrieval pipeline:
  1. Dense search (Qdrant)
  2. BM25 keyword search over stored chunk texts
  3. Reciprocal Rank Fusion to merge results
  4. Cohere reranker for final ordering
"""

from rank_bm25 import BM25Okapi
import cohere
from app.core.config import get_settings
from app.services.rag.embedder import embed_single
from app.services.rag.vector_store import dense_search

_cohere_client: cohere.AsyncClient | None = None


def _get_cohere() -> cohere.AsyncClient | None:
    settings = get_settings()
    global _cohere_client
    if not settings.cohere_api_key:
        return None
    if _cohere_client is None:
        _cohere_client = cohere.AsyncClient(api_key=settings.cohere_api_key)
    return _cohere_client


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked lists into one using RRF.
    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            uid = item["id"]
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            items[uid] = item

    merged = sorted(items.values(), key=lambda x: scores[x["id"]], reverse=True)
    for item in merged:
        item["rrf_score"] = scores[item["id"]]
    return merged


def bm25_search(
    query: str,
    corpus: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """BM25 over a pre-fetched corpus of chunk dicts."""
    if not corpus:
        return []
    tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in top_indices:
        item = dict(corpus[i])
        item["bm25_score"] = float(scores[i])
        results.append(item)
    return results


async def retrieve(
    query: str,
    collection: str,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline → reranked top-k chunks.
    """
    settings = get_settings()

    # 1. Dense search
    query_vector = await embed_single(query)
    dense_results = await dense_search(query_vector, collection, top_k=retrieval_top_k)

    if not dense_results:
        return []

    # 2. BM25 over the dense candidates (avoids scanning entire index)
    bm25_results = bm25_search(query, dense_results, top_k=retrieval_top_k)

    # 3. RRF merge
    fused = reciprocal_rank_fusion([dense_results, bm25_results])
    candidates = fused[:retrieval_top_k]

    # 4. Cohere reranker (if API key available)
    cohere_client = _get_cohere()
    if cohere_client and candidates:
        try:
            rerank_response = await cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=[c["text"] for c in candidates],
                top_n=rerank_top_k,
            )
            reranked = [
                {**candidates[r.index], "rerank_score": r.relevance_score}
                for r in rerank_response.results
            ]
            return reranked
        except Exception:
            pass  # Fall through to RRF results if reranker fails

    return candidates[:rerank_top_k]
