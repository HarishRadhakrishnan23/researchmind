"""
RAG pipeline orchestrator.
ingest_document: PDF bytes → chunk → embed → store
query_rag:       question → retrieve → LLM → answer + sources
"""

import uuid
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.utils.pdf_parser import extract_pages
from app.services.rag.chunker import parent_child_chunks
from app.services.rag.embedder import embed_texts
from app.services.rag.vector_store import ensure_collection, upsert_chunks
from app.services.rag.retriever import retrieve

_openai: AsyncOpenAI | None = None


def _llm() -> AsyncOpenAI:
    global _openai
    if _openai is None:
        _openai = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _openai


async def ingest_document(
    file_bytes: bytes,
    filename: str,
    collection: str | None = None,
) -> dict:
    """
    Full ingest flow:
    PDF bytes → extract pages → parent-child chunks → embed → upsert Qdrant
    """
    settings = get_settings()
    collection = collection or settings.qdrant_collection
    document_id = str(uuid.uuid4())

    await ensure_collection(collection)

    pages = extract_pages(file_bytes, filename)
    if not pages:
        raise ValueError(f"No extractable text found in '{filename}'")

    all_parents = []
    all_children = []

    for page in pages:
        parents, children = parent_child_chunks(
            text=page.text,
            document_id=document_id,
            filename=filename,
            page=page.page_num,
            parent_size=settings.chunk_size,
            child_size=128,
            overlap=settings.chunk_overlap,
        )
        all_parents.extend(parents)
        all_children.extend(children)

    # Embed child chunks (indexed for retrieval)
    child_texts = [c.text for c in all_children]
    child_vectors = await embed_texts(child_texts)
    await upsert_chunks(all_children, child_vectors, collection)

    # Embed parent chunks (stored for context retrieval)
    parent_texts = [p.text for p in all_parents]
    parent_vectors = await embed_texts(parent_texts)
    await upsert_chunks(all_parents, parent_vectors, collection)

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_created": len(all_children),
        "message": f"Ingested {len(pages)} pages → {len(all_children)} chunks",
    }


async def query_rag(
    query: str,
    collection: str | None = None,
    top_k: int = 5,
) -> dict:
    """
    RAG query: retrieve relevant chunks → generate grounded answer.
    """
    settings = get_settings()
    collection = collection or settings.qdrant_collection

    chunks = await retrieve(
        query=query,
        collection=collection,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=top_k,
    )

    if not chunks:
        return {
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "query": query,
        }

    # Build context block for LLM
    context = "\n\n---\n\n".join(
        f"[Source {i+1}: {c['filename']}, page {c.get('page', '?')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are a research assistant. Answer the user's question using ONLY "
        "the provided source excerpts. "
        "Cite sources as [Source N] inline. "
        "If the answer is not in the sources, say so explicitly."
    )
    user_prompt = f"Sources:\n{context}\n\nQuestion: {query}"

    response = await _llm().chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    sources = [
        {
            "document_id": c["document_id"],
            "filename": c["filename"],
            "page": c.get("page"),
            "text": c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
            "score": c.get("rerank_score") or c.get("rrf_score") or c.get("score") or 0.0,
        }
        for c in chunks
    ]

    return {"answer": answer, "sources": sources, "query": query}


async def stream_query_rag(query: str, collection: str | None = None, top_k: int = 5):
    """
    Async generator for streaming SSE responses.
    Yields answer tokens one by one.
    """
    settings = get_settings()
    collection = collection or settings.qdrant_collection

    chunks = await retrieve(query=query, collection=collection, rerank_top_k=top_k)

    if not chunks:
        yield "data: No relevant documents found.\n\n"
        return

    context = "\n\n---\n\n".join(
        f"[Source {i+1}: {c['filename']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    stream = await _llm().chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": "Answer using only the sources. Cite as [Source N]."},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"},
        ],
        stream=True,
        temperature=0.2,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {delta}\n\n"

    yield "data: [DONE]\n\n"
