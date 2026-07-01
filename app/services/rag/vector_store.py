"""
Qdrant vector store wrapper.
Handles collection creation, upsert, and dense vector search.
"""

from uuid import UUID
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
from app.core.config import get_settings
from app.services.rag.chunker import Chunk

VECTOR_DIM = 1536  # text-embedding-3-small

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    global _client
    settings = get_settings()
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


async def ensure_collection(collection: str) -> None:
    """Create collection if it doesn't exist."""
    client = get_qdrant()
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if collection not in names:
        await client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


async def upsert_chunks(
    chunks: list[Chunk],
    vectors: list[list[float]],
    collection: str,
) -> None:
    """Upsert chunk vectors + payload into Qdrant."""
    client = get_qdrant()
    points = [
        PointStruct(
            id=chunk.id,
            vector=vector,
            payload={
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page": chunk.page,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "parent_id": chunk.parent_id,
                "token_count": chunk.token_count,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    await client.upsert(collection_name=collection, points=points)


async def dense_search(
    query_vector: list[float],
    collection: str,
    top_k: int = 20,
    document_id: str | None = None,
) -> list[dict]:
    """
    Dense cosine similarity search.
    Optionally filter to a specific document.
    Returns list of payload dicts with an added 'score' key.
    """
    client = get_qdrant()
    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    results = await client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {**hit.payload, "score": hit.score, "id": str(hit.id)}
        for hit in results
    ]


async def get_parent_chunks(
    parent_ids: list[str],
    collection: str,
) -> dict[str, dict]:
    """Fetch parent chunks by ID for parent-child retrieval."""
    client = get_qdrant()
    results = await client.retrieve(
        collection_name=collection,
        ids=parent_ids,
        with_payload=True,
    )
    return {str(r.id): r.payload for r in results}
