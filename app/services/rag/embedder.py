"""
Async OpenAI embeddings with automatic batching.
text-embedding-3-small: 1536 dims, cheap, fast.
"""

from openai import AsyncOpenAI
from app.core.config import get_settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


async def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of strings. Handles batching automatically.
    Returns list of float vectors in the same order as input.
    """
    settings = get_settings()
    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await client.embeddings.create(
            model=settings.openai_embed_model,
            input=batch,
        )
        # response.data is sorted by index
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


async def embed_single(text: str) -> list[float]:
    results = await embed_texts([text])
    return results[0]
